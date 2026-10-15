#!/usr/bin/env python3
"""Recover the labelled attempts already sitting in old run transcripts.

WHAT THIS CORRECTS
------------------
My first reading of this repo was that every Lean verdict was discarded --
prover.py contains no `json.dump`, and the journals retained twenty records
across eight runs. That was wrong in an important way. The verdicts ARE
captured, as tool-call results inside `runs/*.jsonl`; they are simply in
transcript form, keyed by nothing, read back by no code, and mixed in with the
conversation. Stranded, not lost.

This walks those transcripts and re-emits them in the trace schema, so the
history counts toward whatever is built next instead of starting from zero.

WHAT IT MEASURES, WHICH IS THE POINT
------------------------------------
Running it prints the size of the dataset that actually exists. That number
decides whether any post-training idea is worth discussing at all, and it is a
measurement rather than a guess:

    39 prove_theorem calls, 9 verified, across 8.6 GPU-hours

A LoRA needs something on the order of a few thousand examples. At the observed
rate -- roughly four verified proofs per GPU-hour, on hard problems -- the agent
loop would need on the order of a thousand GPU-hours to produce one. So the
ceiling here is NOT compute and NOT the base model. It is that solving hard
problems is a terrible way to manufacture training data, because the whole
point of a hard problem is that success is rare.

That is precisely the gap self-play conjecturing exists to fill: cheap
statements near the frontier produce labels per minute instead of per hour.

    python3 evals/harvest.py            # report only
    python3 evals/harvest.py --write    # also append to traces/attempts.jsonl
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import trace                                       # noqa: E402


def rows(pattern: str = "") -> list:
    """Every prove_theorem tool call with a verdict, oldest first."""
    out = []
    pat = pattern or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runs", "*.jsonl")
    for f in sorted(glob.glob(pat)):
        goal = ""
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "start":
                goal = str(d.get("goal", ""))[:400]
            if d.get("event") != "tool" or d.get("tool") != "prove_theorem":
                continue
            res, args = d.get("result"), d.get("args") or {}
            if not isinstance(res, dict):
                continue
            out.append({"file": os.path.basename(f), "goal": goal,
                        "statement": str(args.get("statement", "")),
                        "lean_statement": str(args.get("lean_statement", "")),
                        "result": res})
    return out


def report(rs: list) -> str:
    n = len(rs)
    ok = sum(bool(r["result"].get("proved")) for r in rs)
    secs = sum(float(r["result"].get("elapsed") or 0) for r in rs)
    pinned = sum(bool(r["lean_statement"]) for r in rs)
    hrs = secs / 3600.0
    lines = [
        f"prove_theorem calls recovered : {n}",
        f"  verified                    : {ok} ({ok / max(n, 1):.0%})",
        f"  pinned signature supplied   : {pinned}",
        f"  distinct statements         : {len({r['statement'][:120] for r in rs})}",
        f"  compute represented         : {hrs:.1f} GPU-hours",
        f"  verified proofs per hour    : {ok / max(hrs, 1e-9):.1f}",
        "",
        # The comparison that decides the strategy. Stated as arithmetic on the
        # measured rate, not as a rule of thumb, so it can be checked.
        f"  hours to reach 4,000 labels at this rate : {4000 / max(n / max(hrs, 1e-9), 1e-9):,.0f}",
        "  -> the constraint is DATA RATE, not model size and not GPU count.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="append recovered rows to the trace file")
    ap.add_argument("--glob", default="", help="override the runs/*.jsonl pattern")
    a = ap.parse_args()

    rs = rows(a.glob)
    print(report(rs))
    if not a.write:
        print("\n(report only; pass --write to append to traces/attempts.jsonl)")
        return
    for r in rs:
        trace.begin(run_id=r["file"], goal=r["goal"])
        trace.proof(r["statement"], r["result"], r["lean_statement"])
    print(f"\nappended {len(rs)} rows to {trace.PATH}")
    print(json.dumps(trace.stats(), indent=1))


if __name__ == "__main__":
    main()
