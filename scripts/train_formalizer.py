#!/usr/bin/env python3
"""LoRA a 7B specialist on informal -> formal Lean signatures. Runs on the A6000 box.

WHY A SPECIALIST, NOT THE ORCHESTRATOR
--------------------------------------
Measured on the 210-item benchmark: the 27B orchestrator formalizes at 28% (T2),
42% of items produce nothing, and 77% of those failures are SYSTEMATIC -- a
retry does not reach them (12/88 recovery where independence predicts 58%).
Every prompt-level fix measured null. The only lever left that touches the base
rate is changing what a model knows, and Mathlib hands us 9,566 clean
(docstring, signature) pairs to do it with.

A tuned 7B on a narrow translation task is the textbook case where small+tuned
beats large+zero-shot, it trains in under an hour on one A6000, and it serves
on the idle GPU 0 without touching the Mac's single Ollama slot.

DETERMINISM BY CONSTRUCTION
---------------------------
The model is trained to emit EXACTLY one signature line -- no JSON schema, no
reasoning preamble. At temperature 0 the whole think/format failure class that
bit this project six times cannot exist on this path: greedy decode, one line,
straight into the Lean typecheck.

TRAINING/INFERENCE CONTRACT
---------------------------
PROMPT below is the canonical prompt. src/formalizer.py carries a byte-identical
copy, and tests/test_formalize.py asserts the two files agree -- a drifted
prompt is a silent distribution shift, which is exactly the kind of bug this
project keeps finding by measurement instead of by eye.

    ~/venvs/train/bin/python train_formalizer.py --data ~/formalizer/formalize_sft.jsonl
"""

import argparse
import json
import os

PROMPT = (
    "Translate the mathematical claim into one Lean 4 theorem signature that "
    "elaborates against Mathlib.\n\n"
    "CLAIM: {informal}\n\n"
    "Answer with ONLY the signature, one line, no proof and no commentary:\n"
    "theorem <name> <binders> : <conclusion>"
)

BASE = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 640


def rows(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tgt = f"theorem {r['name']} {r['formal']}".strip()
            out.append({"prompt": PROMPT.format(informal=r["informal"]),
                        "target": tgt, "split": r.get("split", "train")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/formalizer/run1"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(BASE)
    data = rows(a.data)

    def encode(r):
        msgs = [{"role": "user", "content": r["prompt"]}]
        # tokenize=False then one tok() call: apply_chat_template's tokenized
        # return type changed across transformers versions (list -> BatchEncoding),
        # and this form is stable on both. The template string carries its
        # special tokens as text, so add_special_tokens=False avoids doubling.
        ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
        prompt_ids = tok(ptxt, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + tok(r["target"], add_special_tokens=False)["input_ids"] \
            + [tok.eos_token_id]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels}

    enc = [encode(r) | {"split": r["split"]} for r in data]
    dropped = sum(1 for e in enc if len(e["input_ids"]) > MAX_TOKENS)
    enc = [e for e in enc if len(e["input_ids"]) <= MAX_TOKENS]
    train = [e for e in enc if e["split"] == "train"]
    val = [e for e in enc if e["split"] == "val"]
    print(f"train {len(train)}  val {len(val)}  dropped(>{MAX_TOKENS} tok) {dropped}",
          flush=True)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"]))
                            for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"]))
                            for b in batch])
        att = (ids != pad).long()
        # A pad id that equals eos would zero the real eos in the mask; restore it.
        for i, b in enumerate(batch):
            att[i, :len(b["input_ids"])] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=a.bs, eval_strategy="epoch",
        save_strategy="epoch", logging_steps=20, bf16=True,
        lr_scheduler_type="cosine", warmup_steps=20, report_to=[],
        save_total_limit=2, remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
            data_collator=collate).train()

    model.save_pretrained(os.path.join(a.out, "adapter"))

    # Merge for vLLM serving: one plain HF model dir, no adapter plumbing, no
    # version-dependent --enable-lora flags.
    merged = model.merge_and_unload()
    mdir = os.path.join(a.out, "merged")
    merged.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    print(f"MERGED -> {mdir}", flush=True)

    # Cheap sanity before anyone pays for serving: greedy exact-match on val.
    # Exact match is a HARSH lower bound (a different-but-equivalent signature
    # scores 0 here and T2 would credit it) -- useful only as a smoke signal.
    merged.eval()
    hits = 0
    sample = val[:80]
    for e in sample:
        cut = e["labels"].index(next(x for x in e["labels"] if x != -100))
        ids = torch.tensor([e["input_ids"][:cut]]).to(merged.device)
        with torch.no_grad():
            out = merged.generate(ids, max_new_tokens=160, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        want = tok.decode([x for x in e["labels"] if x != -100],
                          skip_special_tokens=True).strip()
        hits += got == want
    print(f"VAL EXACT-MATCH {hits}/{len(sample)}", flush=True)


if __name__ == "__main__":
    main()
