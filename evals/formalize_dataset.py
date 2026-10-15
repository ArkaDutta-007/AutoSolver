#!/usr/bin/env python3
"""Build the informal -> formal training set from Mathlib, contamination-free.

WHY THIS IS THE HIGHEST-VALUE UNEXPLOITED ASSET
-----------------------------------------------
Earlier in this project I computed that the agent loop would need roughly 880
GPU-hours to manufacture a training set, because it produces about one verified
proof per GPU-hour and a LoRA wants thousands of examples. That number is right
-- for PROOFS.

For FORMALIZATION it is irrelevant, and I under-weighted that for most of a day.
Mathlib ships ~8,300 files in which a natural-language docstring sits directly
above the formal statement it describes. That is an (informal, formal) corpus
written by mathematicians, already labelled, already correct, and free. The
extractor built for the benchmark yields over eleven thousand such pairs.

So the data-rate problem that blocks proof training does not block formalization
training at all. Formalization is also the measured bottleneck (39% accurate,
and 44% of items produce nothing), which makes this the one place where a cheap
fine-tune attacks the actual constraint.

THE CONTAMINATION HAZARD, WHICH IS THE WHOLE DIFFICULTY
--------------------------------------------------------
The benchmark is drawn from this identical corpus. Training on it would produce
a model that has memorised the test set, and every number this project reports
would silently become meaningless -- the failure would look exactly like
success, which is the worst kind.

So exclusion here is deliberately paranoid and works on THREE keys, because any
one of them alone leaks:

  * by NAME      -- the obvious one, and not sufficient on its own
  * by STATEMENT -- Mathlib restates the same theorem under several names
                    (`gcd_dvd_left` / `Nat.gcd_dvd_left` / a `protected` alias),
                    so a name-only filter lets the identical statement through
  * by DOCSTRING -- one docstring frequently sits above a family of related
                    lemmas, so an item's informal text can reappear attached to
                    a sibling declaration

A pair is dropped if it collides on ANY key. Over-excluding costs a handful of
training examples out of eleven thousand; under-excluding costs the validity of
every measurement.

    python3 evals/formalize_dataset.py --out kb/formalize_sft.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals.formalize_bench as B                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "formalize_bench.json")
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "kb", "formalize_sft.jsonl")


def _norm(s: str) -> str:
    """Whitespace- and case-insensitive key, so trivial edits do not defeat it."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def held_out_keys(bench_path: str = BENCH) -> tuple:
    try:
        bench = json.load(open(bench_path))
    except Exception:
        return set(), set(), set()
    names = {r["name"] for r in bench}
    stmts = {_norm(r["signature"]) for r in bench}
    docs = {_norm(r["informal"]) for r in bench}
    return names, stmts, docs


def build(pairs: list, bench_path: str = BENCH) -> dict:
    names, stmts, docs = held_out_keys(bench_path)
    kept, dropped = [], {"name": 0, "statement": 0, "docstring": 0}
    for r in pairs:
        if r["name"] in names:
            dropped["name"] += 1
            continue
        if _norm(r["signature"]) in stmts:
            dropped["statement"] += 1
            continue
        if _norm(r["informal"]) in docs:
            dropped["docstring"] += 1
            continue
        kept.append({"informal": r["informal"],
                     "formal": r["signature"],
                     "name": r["name"],
                     "module": r.get("module", "")})
    return {"rows": kept, "dropped": dropped,
            "held_out": len(names)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    a = ap.parse_args()

    # SELF-CONTAINED ONLY, and this was got wrong the first time at real cost.
    #
    # The first dataset dropped this filter on the reasoning that "a statement
    # needing its section context is still a correct target to learn from".
    # That confuses the TRAINING target with the INFERENCE requirement. Mathlib
    # statements inherit variables from `variable` lines, so 77% of the corpus
    # carries free variables; a model trained on it learns to emit exactly what
    # the prover and the benchmark reject.
    #
    # MEASURED on the model trained without this filter: 3/15 elaborate (20%)
    # against the 27B's 58%, and the failures are precisely unbound variables --
    # `(h : n = a^2+b^2)` with no `n`, `(hD : Convex R D)` with no `D`.
    pairs = B.extract(self_contained_only=True)
    out = build(pairs)
    rows = out["rows"]

    # Split by MODULE, not by row. Sibling lemmas in one file share phrasing and
    # machinery, so a random row split puts near-duplicates on both sides and
    # reports a validation score that is really a memorisation score.
    mods = sorted({r["module"] for r in rows})
    n_val = max(1, int(len(mods) * a.val_fraction))
    val_mods = set(mods[::max(len(mods) // n_val, 1)][:n_val])
    train = [r for r in rows if r["module"] not in val_mods]
    val = [r for r in rows if r["module"] in val_mods]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps({**r, "split": "train"}, ensure_ascii=False) + "\n")
        for r in val:
            fh.write(json.dumps({**r, "split": "val"}, ensure_ascii=False) + "\n")

    print(f"extracted pairs           : {len(pairs)}")
    print(f"held out (benchmark items): {out['held_out']}")
    print(f"dropped for contamination : {sum(out['dropped'].values())} "
          f"{out['dropped']}")
    print(f"train / val               : {len(train)} / {len(val)} "
          f"({len(val_mods)} modules held for val)")
    print(f"written to {a.out}")
    print("\nEvery benchmark item is excluded by name, by statement text, and by")
    print("docstring. Train on this and the 210-item benchmark stays a fair test.")


if __name__ == "__main__":
    main()
