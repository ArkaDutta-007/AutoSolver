#!/usr/bin/env python3
"""Train the formalizer on Colab (A100 80GB). Fetches its own data from HF.

    !pip -q install "transformers>=4.44" peft accelerate pyarrow
    !python train_colab.py --drive /content/drive/MyDrive/formalizer

WHAT THIS RUN IS TESTING, AND WHY THE BASE MODEL IS UNCHANGED
--------------------------------------------------------------
An identical recipe on 2,276 in-house Mathlib docstring pairs produced a model
scoring 20% then 4% T1 against the 27B's 58%. The suspected cause was data
volume, so this run changes ONE variable: 58,031 public pairs instead of 2,276,
same Qwen2.5-7B base, same LoRA rank, same prompt. If it still loses, the
specialist idea is dead for a reason other than data and should be abandoned
rather than tuned further.

COLAB REALITIES THIS ACCOUNTS FOR
---------------------------------
Sessions die. Everything is checkpointed to Drive every 200 steps and the run
resumes from the newest checkpoint automatically, so a disconnect costs minutes
rather than the run. The dataset is pulled from HuggingFace inside the script
rather than uploaded, so a fresh VM needs no manual file shuffling.

CONTAMINATION
-------------
The 210-item benchmark is Mathlib-derived and so is Herald. `bench.json` (the
benchmark) must be uploaded once; every row colliding with it by name,
normalised statement, or normalised docstring is dropped. Without that file the
script REFUSES to run -- silently training on the test set would make every
downstream number meaningless, and that failure is invisible afterwards.
"""

import argparse
import io
import json
import os
import re
import urllib.request

PROMPT = (
    "Translate the mathematical claim into one Lean 4 theorem signature that "
    "elaborates against Mathlib.\n\n"
    "CLAIM: {informal}\n\n"
    "Answer with ONLY the signature, one line, no proof and no commentary:\n"
    "theorem <name> <binders> : <conclusion>"
)

BASE = "Qwen/Qwen2.5-7B-Instruct"
SOURCES = [("FrenzyMath/Herald_proofs", "informal_theorem", "formal_theorem"),
           ("internlm/Lean-Workbook", "natural_language_statement",
            "formal_statement")]
_DECL = re.compile(r"(?:theorem|lemma)\s+[^\s:]+")
_norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())


def fetch(bench_path):
    bench = json.load(open(bench_path))
    names = {r["name"] for r in bench}
    stmts = {_norm(r["signature"]) for r in bench}
    docs = {_norm(r["informal"]) for r in bench}
    import pyarrow.parquet as pq
    rows, seen, blocked = [], set(), 0
    for ds, ifield, ffield in SOURCES:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60) as r:
            tree = json.load(r)
        urls = []
        def walk(x):
            if isinstance(x, str) and x.endswith(".parquet"):
                urls.append(x)
            elif isinstance(x, dict):
                [walk(v) for v in x.values()]
            elif isinstance(x, list):
                [walk(v) for v in x]
        walk(tree)
        for u in urls:
            with urllib.request.urlopen(u, timeout=300) as r:
                tbl = pq.read_table(io.BytesIO(r.read())).to_pylist()
            for x in tbl:
                inf = str(x.get(ifield) or "").strip()
                fml = re.split(r":=", " ".join(str(x.get(ffield) or "").split()))[0].strip()
                if not inf or len(inf) < 25 or not _DECL.match(fml):
                    continue
                if (str(x.get("name") or "") in names or _norm(fml) in stmts
                        or _norm(inf) in docs):
                    blocked += 1
                    continue
                if _norm(fml) in seen:
                    continue
                seen.add(_norm(fml))
                rows.append({"informal": inf[:1200], "formal": fml[:600]})
    print(f"fetched {len(rows):,} pairs, blocked {blocked} contaminated", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/formalizer")
    ap.add_argument("--bench", default="bench.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    if not os.path.exists(a.bench):
        raise SystemExit(
            f"{a.bench} not found. Upload evals/formalize_bench.json -- without it "
            "the run would train on its own test set and every later number "
            "would be a memorisation score.")

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    os.makedirs(a.drive, exist_ok=True)
    cache = os.path.join(a.drive, "pairs.jsonl")
    if os.path.exists(cache):
        rows = [json.loads(l) for l in open(cache)]
        print(f"reusing cached {len(rows):,} pairs", flush=True)
    else:
        rows = fetch(a.bench)
        with open(cache, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(BASE)

    def encode(r):
        ptxt = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(informal=r["informal"])}],
            add_generation_prompt=True, tokenize=False)
        pid = tok(ptxt, add_special_tokens=False)["input_ids"]
        full = pid + tok(r["formal"], add_special_tokens=False)["input_ids"] \
            + [tok.eos_token_id]
        return {"input_ids": full,
                "labels": [-100] * len(pid) + full[len(pid):]}

    enc = [encode(r) for r in rows]
    enc = [e for e in enc if len(e["input_ids"]) <= 768]
    val, train = enc[::40], [e for i, e in enumerate(enc) if i % 40]
    print(f"train {len(train):,}  val {len(val):,}", flush=True)

    def collate(b):
        n = max(len(x["input_ids"]) for x in b)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in b])
        lab = torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in b])
        att = torch.zeros_like(ids)
        for i, x in enumerate(b):
            att[i, :len(x["input_ids"])] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    out = os.path.join(a.drive, "run")
    # save_steps, not save_strategy="epoch": a Colab session that dies 80% of
    # the way through an epoch would otherwise lose everything.
    args = TrainingArguments(
        output_dir=out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=a.bs, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        logging_steps=25, bf16=True, lr_scheduler_type="cosine",
        warmup_steps=50, report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      eval_dataset=val, data_collator=collate)

    ckpts = [d for d in os.listdir(out)] if os.path.isdir(out) else []
    resume = any(d.startswith("checkpoint-") for d in ckpts)
    print(f"{'resuming from checkpoint' if resume else 'fresh start'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    merged = model.merge_and_unload()
    mdir = os.path.join(a.drive, "merged")
    merged.save_pretrained(mdir, safe_serialization=True)
    tok.save_pretrained(mdir)
    print(f"MERGED -> {mdir}", flush=True)


if __name__ == "__main__":
    main()
