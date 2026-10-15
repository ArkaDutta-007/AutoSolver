#!/usr/bin/env python3
"""LoRA the 32B prover on its own verified proofs. Runs on the GPU box or Colab.

WHAT "LOSSLESS" CAN MEAN HERE
-----------------------------
Speed: the adapter is merged into the bf16 weights, so the served model has the
same architecture, size and dtype as the base -- decode tokens/s cannot change.
Capability: no weight update is lossless by construction, so it is made
unlikely and then MEASURED. Every target is the model's OWN output
(scripts/oai_sample.py), verified by Lean: plain and repair samples are exactly
on-policy, hinted ones are its own reasoning written with a reference proof in
view and trained without it. Low rank, low LR, one epoch, research AND
competition problems in the mix. The tuned model replaces the base only if it
is no worse on held-out research problems and on miniF2F.

TWO FAILURES THIS TARGETS, BESIDES KNOWLEDGE
--------------------------------------------
Not concluding: reasoning that runs out of tokens, or circles, before the code
block. Every SFT target is the SHORTEST verified, non-looping answer for its
prompt, and a preference pair puts that answer above a truncated or looping one
for the same prompt.
Repeating a mistake: in the Autosolver's repair turn the model often resubmits
the error it was just shown. Repair samples that fixed it are SFT targets; one
that hit the SAME error again is the rejected side of a pair.

OBJECTIVE
---------
sft item:  mean token NLL of the target.
pair item: DPO (beta) against the frozen base -- the adapter switched off, so
           no second copy of 32B weights -- plus alpha x the chosen side's NLL,
           which keeps DPO from lowering both likelihoods together.

    ~/venvs/train/bin/python train_prover.py --data plain.jsonl hint.jsonl repair.jsonl \\
        --out ~/oaimath/run/train --merge-after
"""

import argparse
import collections
import json
import math
import os
import random
import shutil
import time

BASE = "Goedel-LM/Goedel-Prover-V2-32B"
REJECT_ORDER = ("truncated", "loop", "repeat", "compile", "statement_changed", "no_code")


def _key(r):
    return json.dumps(r.get("messages") or [{"role": "user", "content": r["prompt"]}],
                      ensure_ascii=False, sort_keys=True)


def _fault(r):
    return "loop" if r.get("loop") and r["why"] != "truncated" else r["why"]


def rows(paths, per_problem=2, easy_share=0.15, max_examples=0, max_pairs=0, seed=0):
    """SFT items and preference pairs from sample files (any mix of modes)."""
    rng = random.Random(seed)
    groups = collections.OrderedDict()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:          # a killed sampler's partial last line
                continue
            groups.setdefault(_key(r), []).append(r)
    good = lambda r: r["ok"] and not r.get("loop")
    shortest = lambda rs: sorted(rs, key=lambda r: (r.get("ntok") or len(r["text"]), r["text"]))
    sft, easy, pairs = [], [], []
    for key, g in groups.items():
        ctx = json.loads(key)
        onpol = [r for r in g if r["mode"] in ("plain", "repair")]
        wins = shortest([r for r in onpol if good(r)])
        hint_wins = shortest([r for r in g if r["mode"] == "hint" and good(r)])
        if wins and len(wins) == len(onpol) and onpol[0]["mode"] == "plain":
            easy.append({"kind": "replay", "messages": ctx, "target": wins[0]["text"]})
        else:
            kind = ("repair" if onpol and onpol[0]["mode"] == "repair" else "sharpen") \
                if wins else "rationalized"
            seen = []
            for r in wins or hint_wins:
                if r["text"] not in seen and len(seen) < per_problem:
                    seen.append(r["text"])
                    sft.append({"kind": kind, "messages": ctx, "target": r["text"]})
        best = (wins or hint_wins or [None])[0]
        bad = sorted((r for r in onpol if not r["ok"] and _fault(r) in REJECT_ORDER),
                     key=lambda r: REJECT_ORDER.index(_fault(r)))
        if best and bad:
            pairs.append({"kind": f"pair:{_fault(bad[0])}", "messages": ctx,
                          "chosen": best["text"], "rejected": bad[0]["text"]})
    rng.shuffle(sft)
    rng.shuffle(easy)
    rng.shuffle(pairs)
    sft = sft[: max_examples or None]
    sft += easy[: int(easy_share * len(sft) / max(1e-9, 1 - easy_share))]
    out = sft + pairs[: max_pairs or None]
    rng.shuffle(out)
    return out


def chunked_loss(model, ids, labels, chunk=2048):
    """Summed token NLL without a [T, 151k] fp32 logits tensor: at 12k tokens
    that alone is 7 GB, the headroom an A6000 has left beside its share of the
    weights. Each chunk's logits are recomputed in backward, never stored."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    inner = model.get_base_model()
    h = inner.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    y = labels[:, 1:].to(h.device)

    def piece(hc, yc):
        logits = inner.lm_head(hc).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), yc.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[1], chunk):
        yc = y[:, s:s + chunk]
        if (yc != -100).any():
            total = total + (checkpoint(piece, h[:, s:s + chunk], yc, use_reentrant=False)
                             if torch.is_grad_enabled() else piece(h[:, s:s + chunk], yc))
    return total


def encode(tok, items, max_len):
    """Token ids per item. A rejected answer is cut to fit (a truncated one is
    long by definition, and where it goes wrong is early); a target or chosen
    answer must fit whole, so what is reinforced always ends in its conclusion."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    head = lambda m: tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                         add_special_tokens=False).input_ids
    body = lambda t: tok(t, add_special_tokens=False).input_ids + [end]
    out = []
    for it in items:
        p = head(it["messages"])
        if len(p) > max_len - 256:
            continue
        c = body(it.get("target") or it["chosen"])
        if len(p) + len(c) > max_len:
            continue
        seqs = [(p + c, len(p))]
        if "rejected" in it:
            seqs.append((p + body(it["rejected"])[:max_len - len(p)], len(p)))
        out.append({"kind": it["kind"], "seqs": seqs})
    return out


def train(a):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    items = rows(a.data, a.per_problem, a.easy_share, a.max_examples, a.max_pairs)
    enc = encode(tok, items, a.max_len)
    kinds = collections.Counter(e["kind"] for e in enc)
    toks = sum(len(s) for e in enc for s, _ in e["seqs"])
    print(f"{len(enc)}/{len(items)} items fit {a.max_len} tokens: {dict(kinds)}, {toks} tokens",
          flush=True)

    caps = {i: a.gpu_mem for i in range(torch.cuda.device_count())} if a.gpu_mem else None
    model = AutoModelForCausalLM.from_pretrained(      # "auto" without caps = balanced
        a.base, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        max_memory=caps)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    adapter, state_path = os.path.join(a.out, "adapter"), os.path.join(a.out, "adapter", "state.json")
    recover_adapter(adapter)
    start = json.load(open(state_path))["step"] if os.path.exists(state_path) else 0
    if start:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        print(f"resuming after step {start}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    groups = [enc[i:i + a.accum] for i in range(0, len(enc), a.accum)] * a.epochs
    warm = max(1, len(groups) // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, len(groups)))))))
    for _ in range(start):
        sched.step()
    dev = model.get_base_model().model.embed_tokens.weight.device

    def nll(ids, n_prompt):
        x = torch.tensor([ids], device=dev)
        lab = torch.tensor([[-100] * n_prompt + ids[n_prompt:]])
        return chunked_loss(model, x, lab), len(ids) - n_prompt

    model.train()
    t0, seen, skipped = time.time(), 0, 0
    for step in range(start, len(groups)):
        stats = collections.defaultdict(float)
        for e in groups[step]:
            try:
                if len(e["seqs"]) == 1:
                    loss_sum, n = nll(*e["seqs"][0])
                    loss = loss_sum / n
                else:
                    (c, nc), (r, nr) = e["seqs"]
                    with torch.no_grad(), model.disable_adapter():
                        ref_c, ref_r = -nll(c, nc)[0], -nll(r, nr)[0]
                    pol_c, n_c = nll(c, nc)
                    pol_r, _ = nll(r, nr)
                    margin = a.beta * ((-pol_c - ref_c) - (-pol_r - ref_r))
                    loss = -F.logsigmoid(margin) + a.alpha * pol_c / n_c
                    stats["pairs"] += 1
                    stats["margin"] += margin.item()
                    stats["acc"] += float(margin.item() > 0)
                (loss / len(groups[step])).backward()
                stats["loss"] += loss.item() / len(groups[step])
                seen += sum(len(s) for s, _ in e["seqs"])
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                model.zero_grad(set_to_none=True)   # this group's partial update is dropped
                torch.cuda.empty_cache()
                print(f"OOM on a {max(len(s) for s, _ in e['seqs'])}-token item, skipped", flush=True)
                break
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        pp = max(stats["pairs"], 1)
        print(f"step {step + 1}/{len(groups)} loss {stats['loss']:.4f} "
              f"pair-acc {stats['acc'] / pp:.2f} margin {stats['margin'] / pp:+.3f} "
              f"lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} tok/s "
              f"skipped {skipped}", flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 or step + 1 == len(groups):
            save_adapter(model, adapter, {"step": step + 1, "of": len(groups)})
    print("saved", adapter, flush=True)
    if a.merge_after:
        save_merged(model.merge_and_unload(), a)


def save_adapter(model, adapter, state):
    """The next checkpoint is written beside the last, then swapped in, so a kill
    at any moment leaves one whole adapter (train() finishes an interrupted swap).
    state.json goes last when writing and first when deleting: it marks whole."""
    new = adapter + ".new"
    shutil.rmtree(new, ignore_errors=True)
    model.save_pretrained(new)
    with open(os.path.join(new, "state.json"), "w") as fh:
        json.dump(state, fh)
    if os.path.exists(os.path.join(adapter, "state.json")):
        os.remove(os.path.join(adapter, "state.json"))
    shutil.rmtree(adapter, ignore_errors=True)
    os.rename(new, adapter)


def recover_adapter(adapter):
    """Finish a swap save_adapter was killed in: the new checkpoint is whole."""
    if not os.path.exists(os.path.join(adapter, "state.json")) \
            and os.path.exists(os.path.join(adapter + ".new", "state.json")):
        shutil.rmtree(adapter, ignore_errors=True)
        os.rename(adapter + ".new", adapter)


def save_merged(merged, a):
    """New weights; every other file -- config, tokenizer, chat template,
    generation defaults -- copied byte for byte from the base, so vLLM serves
    the merged model exactly as it serves the base."""
    from huggingface_hub import snapshot_download
    dst = a.merged or os.path.join(a.out, "merged")
    merged.save_pretrained(dst, max_shard_size="5GB")
    src = a.base if os.path.isdir(a.base) else snapshot_download(a.base)
    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)) and not f.startswith("model.safetensors") \
                and not f.endswith(".safetensors"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print("merged ->", dst, flush=True)


def merge(a):
    """Fold the adapter into bf16 weights: on the GPUs when there are any, else
    on the CPU (65 GB of RAM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="auto" if torch.cuda.is_available() else "cpu")
    recover_adapter(os.path.join(a.out, "adapter"))
    save_merged(PeftModel.from_pretrained(base, os.path.join(a.out, "adapter")).merge_and_unload(), a)


def smoke(a):
    """The whole train path -- chat template, LoRA, both losses, save, merge --
    on the base's own config at 2 layers x width 64, on the CPU, in seconds.
    The pipeline runs it at preflight, so an environment that cannot train fails
    before the sampling hours rather than after them: Colab's own torchao 0.10
    made peft refuse to build a single LoRA layer, 13 hours into a run."""
    import tempfile
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    d = tempfile.mkdtemp(prefix="train_smoke_")
    try:
        cfg = AutoConfig.from_pretrained(a.base)
        cfg.update(dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16))
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:2]
        tiny = os.path.join(d, "tiny")
        AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(tiny)
        AutoTokenizer.from_pretrained(a.base).save_pretrained(tiny)
        ctx = [{"role": "user", "content": "Complete the following Lean 4 code:\n"
                "```lean4\ntheorem t : True := by\n  sorry\n```"}]
        data = os.path.join(d, "plain.jsonl")
        with open(data, "w", encoding="utf-8") as fh:      # one SFT item and one pair
            for tac, ok in (("trivial", True), ("simp", False)):
                fh.write(json.dumps({"mode": "plain", "messages": ctx, "ok": ok, "loop": False,
                                     "why": "ok" if ok else "compile",
                                     "text": f"```lean4\ntheorem t : True := by\n  {tac}\n```"}) + "\n")
        os.makedirs(os.path.join(d, "train", "adapter"))
        train(argparse.Namespace(**{**vars(a), "base": tiny, "data": [data], "gpu_mem": "",
                                    "out": os.path.join(d, "train"), "merged": os.path.join(d, "merged"),
                                    "merge_after": True, "max_len": 512, "accum": 2, "save_every": 1}))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("smoke ok: this environment can train and merge", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="oai_sample.py output files")
    ap.add_argument("--out", default="")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--merge", action="store_true", help="only merge an existing adapter")
    ap.add_argument("--merge-after", action="store_true")
    ap.add_argument("--merged", default="", help="where merged weights go (default OUT/merged)")
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--easy-share", type=float, default=0.15)
    ap.add_argument("--max-examples", type=int, default=1200)
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--gpu-mem", default="", help='per-GPU cap, e.g. "38GiB"')
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="run the train path on a 2-layer copy of --base, on the CPU, and exit")
    a = ap.parse_args()
    if a.smoke:
        return smoke(a)
    if not a.out:
        ap.error("--out is required")
    os.makedirs(os.path.join(a.out, "adapter"), exist_ok=True)
    merge(a) if a.merge else train(a)


if __name__ == "__main__":
    main()
