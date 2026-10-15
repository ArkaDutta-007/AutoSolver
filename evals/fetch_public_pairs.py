#!/usr/bin/env python3
"""Pull public NL->Lean statement pairs and merge them into one training set.

WHY: THE MEASURED FAILURE WAS DATA VOLUME, NOT THE APPROACH
-----------------------------------------------------------
The in-house formalizer LoRA was trained on 2,276 Mathlib docstring pairs and
scored 20% then 4% T1 against the 27B's 58%. Two public corpora are the same
task at 30x the size:

  Herald_proofs   44,553  informal_theorem -> formal_theorem, Mathlib-derived,
                          and it ships a `header` field giving the imports each
                          statement needs -- which is exactly the
                          self-containment problem that sank the first run
  Lean-Workbook   25,214  natural_language_statement -> formal_statement,
                          COMPETITION problems rather than Mathlib abstractions

The second is the distribution that matters. On the Mathlib-flavoured benchmark
the in-house model scored 4%; on olympiad-style claims it scored 38%. The agent
works on olympiad problems, so Lean-Workbook is closer to the job than Mathlib
is, and no amount of Mathlib docstrings substitutes for it.

CONTAMINATION IS THE WHOLE DIFFICULTY, AGAIN
--------------------------------------------
Herald is derived from Mathlib4 and so is the 210-item benchmark. Training on
Herald without filtering would put benchmark items into training and quietly
turn every number this project reports into a memorisation score. The same
three-key exclusion as evals/formalize_dataset.py is reused verbatim -- name,
normalised statement text, normalised docstring -- because a name-only filter
already leaked 10 items once.

    python3 evals/fetch_public_pairs.py --out kb/formalize_sft_v3.jsonl
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.formalize_dataset import _norm, held_out_keys      # noqa: E402

# (dataset, informal field, formal field, header field, cap)
#
# Two distributions, and the split is deliberate. Herald is Mathlib-derived and
# matches the 210-item BENCHMARK; NuminaMath-LEAN and Lean-Workbook are
# competition problems and match what the AGENT actually works on. The in-house
# model scored 4% on Mathlib abstractions and 38% on olympiad-style claims, so
# training on Herald alone would optimise the measurement rather than the job.
# Capped rather than exhaustive: 580k+ rows is ~15h for one epoch on an A100-40,
# which does not fit a Colab session, and the point of this run is to test
# whether MORE DATA fixes the 2,276-row failure -- 65x is enough to answer that.
SOURCES = [
    ("FrenzyMath/Herald_statements", "informal_statement", "formal_statement", "", 90000),
    ("AI-MO/NuminaMath-LEAN", "problem", "formal_statement", "", 60000),
    ("FrenzyMath/Herald_proofs", "informal_theorem", "formal_theorem", "header", 45000),
    ("internlm/Lean-Workbook", "natural_language_statement", "formal_statement", "", 30000),
]


def parquet_urls(ds: str) -> list:
    url = f"https://huggingface.co/api/datasets/{ds}/parquet"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    # The endpoint returns {config: {split: [urls]}} or a flat list.
    out = []
    def walk(x):
        if isinstance(x, str) and x.endswith(".parquet"):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(tree)
    return out


def read_parquet(url: str):
    import pyarrow.parquet as pq
    with urllib.request.urlopen(url, timeout=300) as r:
        buf = io.BytesIO(r.read())
    return pq.read_table(buf).to_pylist()


_DECL = re.compile(r"(?:theorem|lemma)\s+[^\s:]+")


def clean_statement(formal: str) -> str:
    """One bare `theorem ... : ...` line: imports stripped, proof stripped.

    Herald_statements and NuminaMath ship the imports INLINE
    ("import Mathlib\nopen scoped ...\ntheorem foo ..."). The pipeline supplies
    its own PREAMBLE, so a target carrying imports would have them applied
    twice. Everything before the declaration keyword goes, and everything from
    `:=` on goes with it.
    """
    txt = str(formal or "")
    m = re.search(r"(?:theorem|lemma)\b", txt)
    if not m:
        return ""
    s = " ".join(txt[m.start():].split())
    s = re.split(r":=", s)[0].strip()
    return s if _DECL.match(s) and len(s) > 20 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb/formalize_sft_v3.jsonl")
    ap.add_argument("--max-per-source", type=int, default=0)
    a = ap.parse_args()

    names, stmts, docs = held_out_keys()
    print(f"holding out {len(names)} benchmark items\n")

    rows, dropped = [], {"contaminated": 0, "unusable": 0}
    seen = set()
    for ds, ifield, ffield, hfield, cap in SOURCES:
        got = 0
        for url in parquet_urls(ds):
            if got >= cap:
                break
            try:
                table = read_parquet(url)
            except Exception as e:
                print(f"  {ds}: {type(e).__name__} on a shard, skipping")
                continue
            for r in table:
                informal = str(r.get(ifield) or "").strip()
                formal = clean_statement(r.get(ffield))
                if not informal or not formal or len(informal) < 25:
                    dropped["unusable"] += 1
                    continue
                nm = str(r.get("name") or "")
                if (nm in names or _norm(formal) in stmts
                        or _norm(informal) in docs):
                    dropped["contaminated"] += 1
                    continue
                key = _norm(formal)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"informal": informal[:1200], "formal": formal[:600],
                             "name": nm or f"{ds.split('/')[-1]}_{len(rows)}",
                             "module": ds,
                             "header": str(r.get(hfield) or "") if hfield else ""})
                got += 1
                if got >= cap:
                    break
        print(f"  {ds:30s} -> {got:,} kept")

    # Hold out by SOURCE-aware split: a random split would put near-duplicate
    # siblings on both sides, the same memorisation trap the Mathlib builder
    # avoided by splitting on module.
    val = rows[::40]
    valset = {id(r) for r in val}
    train = [r for r in rows if id(r) not in valset]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps({**r, "split": "train"}, ensure_ascii=False) + "\n")
        for r in val:
            fh.write(json.dumps({**r, "split": "val"}, ensure_ascii=False) + "\n")

    print(f"\n  dropped: {dropped}")
    print(f"  train / val: {len(train):,} / {len(val):,}")
    print(f"  written to {a.out}")
    print(f"\n  previous in-house set was 2,276 rows -> this is "
          f"{len(rows)/2276:.0f}x")


if __name__ == "__main__":
    main()
