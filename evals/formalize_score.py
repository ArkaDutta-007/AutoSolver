#!/usr/bin/env python3
"""Score informal -> formal against a gold, with the COMPILER as the judge.

WHY NOT AN LLM JUDGE
--------------------
The obvious way to score faithfulness is to ask a model whether the produced
statement means the same as the informal claim. That is exactly the mechanism
this repo caught being a coin flip: on one hand-checked faithful signature,
three draws of identical input returned MISMATCH, FAITHFUL, FAITHFUL, and the
dissenting draw was truncated reasoning rather than a judgement.

That gate has been fixed, but it remains a model judging a model. For a
benchmark meant to drive iteration it cannot be the primary signal, because
every improvement would be confounded with the judge's own noise.

So the primary judge is Lean, in three tiers, reported separately and never
merged into one number:

  T1  ELABORATES     the produced statement compiles at all. Necessary, and far
                     from sufficient -- a tautology compiles, and the agent has
                     already been observed proving one.

  T2  EQUIVALENT     the produced statement and the gold imply each other, shown
                     by the compiler. This is the real score. It is a
                     bidirectional check because one direction alone is
                     worthless: `True` is implied by everything, and a
                     statement strong enough to imply the gold may say far more
                     than the claim did.

  T3  JUDGED         the model reviewer's opinion, kept only for items where T2
                     is inconclusive, and always reported apart from T2. An
                     equivalence a tactic ladder cannot find is not the same
                     thing as an equivalence that does not hold, and conflating
                     them would silently inflate the score.

WHAT T2 CANNOT DO
-----------------
Two genuinely equivalent statements can be beyond a fixed tactic ladder --
different-but-equivalent formulations of the same fact are common and some need
real proof. So T2 is a LOWER BOUND on faithfulness, and the gap between T2 and
T3 is the interesting quantity, not a nuisance. Reporting T2 alone would
understate; reporting T3 alone would be unfalsifiable.

    python3 evals/formalize_score.py --n 40
"""

import argparse
import concurrent.futures as futures
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prover, review                              # noqa: E402
from src.lean import check_lean, split_signature, statement_ok  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "formalize_bench.json")
OUT = os.path.join(HERE, "formalize_score.json")

# Tried in order, cheapest first; the first that closes the goal settles it.
#
# COST IS PART OF THE DESIGN, not an afterthought. The first version inherited
# LEAN_TIMEOUT=120 and ran eight rungs in each direction, so one equivalence
# check could consume sixteen minutes -- and one observed item spent 575s before
# giving up. A benchmark that slow cannot be iterated against, which was the
# entire reason for building it.
#
# So: every rung gets a SHORT timeout, and the whole ladder gets a wall-clock
# budget. An equivalence a fast tactic cannot see is exactly the case T3 exists
# to catch, so spending minutes here buys almost nothing.
LADDER = ["intros; exact?",          # THE workhorse: put it FIRST -- see below
          "exact h",                 # identical statements, no re-binding needed
          "intros; simp_all",        # reindexing, arithmetic rearrangement
          "intros; omega",           # linear arithmetic over Nat/Int
          "intros; aesop"]
RUNG_TIMEOUT = 30          # seconds per tactic
LADDER_BUDGET = 110        # seconds per direction, all rungs together

# `intros;` IS THE LOAD-BEARING PART, and leaving it off made this benchmark
# report a score of zero.
#
# The two sides of an equivalence rarely share binder structure. A Mathlib gold
# often has NO explicit binders at all -- its variables come from a section
# `variable` line and get auto-bound at the top of the `example` -- while the
# derived statement quantifies explicitly. So the hypothesis `h` is a ∀-statement
# and the goal is an instance of it, and bare `exact h` cannot bridge that. It
# reports a type mismatch that reads exactly like a genuine disagreement.
#
# MEASURED on three items hand-checked as equivalent, old ladder vs new:
#
#   extend_partialOrder   exact h False(17s)  ->  intros; exact?   True(14s)
#   prod_subset_prod_iff  exact h False(14s)  ->  intros; exact?   True( 9s)
#   Prime.pow_inj         exact h False(10s)  ->  intros; simp_all True( 4s)
#
# `exact?` was also wrongly blamed for an earlier 575s item and demoted. It
# resolves these in 9-22 seconds; the 575s was the cumulative cost of eight
# rungs each inheriting LEAN_TIMEOUT=120, not this tactic.
#
# ORDER MATTERS AS MUCH AS MEMBERSHIP, and getting it wrong scored a zero even
# after `exact?` was restored. Measured on prod_subset_prod_iff, fwd direction:
#
#   exact h           False  25s
#   intros; simp_all  False  20s   } 71 seconds spent on rungs that do not
#   intros; omega     False  26s   } fire, against a 110s budget
#   intros; exact?    TRUE   22s   <- never reached inside the budget
#
# The cheap-first intuition is wrong here because the "cheap" rungs are not
# cheap: each pays a full timeout when it fails. The rung most likely to
# SUCCEED goes first, since a success costs ~22s and ends the search.


def as_prop(sig: str) -> str:
    """The statement as a single closed proposition, so it can be quantified over."""
    binders, concl = split_signature(sig)
    return f"∀ {binders}, {concl}" if binders else concl


def elaborates(sig: str) -> tuple:
    """T1. Does this statement compile standalone?"""
    code = f"{prover.PREAMBLE}theorem _t1 {sig} := by sorry"
    try:
        v = check_lean(code)
        return statement_ok(v), str(v.get("errors", ""))[:200]
    except Exception as e:
        return False, f"{type(e).__name__}"


def _implies(a: str, b: str) -> bool:
    """Can the compiler derive proposition `b` from proposition `a`, cheaply?"""
    deadline = time.time() + LADDER_BUDGET
    for tac in LADDER:
        if time.time() >= deadline:
            break
        code = (f"{prover.PREAMBLE}example : ({a}) → ({b}) := by\n"
                f"  intro h\n  {tac}")
        try:
            if check_lean(code, timeout=RUNG_TIMEOUT).get("ok"):
                return True
        except Exception:
            continue
    return False


def equivalent(derived: str, gold: str) -> dict:
    """T2. Mutual implication, both directions, judged by the compiler.

    One direction is not enough in either direction, and for different reasons:
    anything implies `True`, and a statement that implies the gold may be
    strictly stronger than the claim warranted.
    """
    a, b = as_prop(derived), as_prop(gold)
    fwd = _implies(a, b)
    bwd = _implies(b, a) if fwd else False      # short-circuit: fwd is cheaper
    return {"forward": fwd, "backward": bwd, "equivalent": bool(fwd and bwd)}


def score_one(item: dict) -> dict:
    """Derive a statement from the informal text, then grade it."""
    t0 = time.time()
    out = dict(name=item["name"], informal=item["informal"][:300],
               gold=item["signature"][:300])
    try:
        derived = prover.derive_signature(item["informal"], tries=1)
    except Exception as e:
        derived = ""
        out["error"] = type(e).__name__
    out["derived"] = (derived or "")[:400]
    if not derived:
        out.update(t1=False, t2=False, t3=None, seconds=round(time.time() - t0, 1))
        return out

    # The derivation returns a full `theorem name ... :=` line; the grader wants
    # only the signature part.
    m = re.search(r"(?:theorem|lemma)\s+[\w'₀-₉.]+\s*(.*?)(?::=|$)", derived, re.S)
    sig = (m.group(1) if m else derived).strip()

    t1, why = elaborates(sig)
    out["t1"] = t1
    out["t1_why"] = why if not t1 else ""
    if not t1:
        out.update(t2=False, t3=None, seconds=round(time.time() - t0, 1))
        return out

    eq = equivalent(sig, item["signature"])
    out.update(t2=eq["equivalent"], forward=eq["forward"], backward=eq["backward"])
    if not eq["equivalent"]:
        try:
            v = review.review_statement(item["informal"], f"theorem t {sig} := by sorry")
            out["t3"] = v.get("faithful")
            out["t3_why"] = str(v.get("verdict", ""))[:160]
        except Exception:
            out["t3"] = None
    else:
        out["t3"] = None
    out["seconds"] = round(time.time() - t0, 1)
    return out


def report(rows: list) -> str:
    n = len(rows)
    if not n:
        return "no rows"
    t1 = sum(1 for r in rows if r.get("t1"))
    t2 = sum(1 for r in rows if r.get("t2"))
    t3 = sum(1 for r in rows if r.get("t3") is True)
    fwd = sum(1 for r in rows if r.get("forward"))
    secs = sum(r.get("seconds", 0) for r in rows) / n
    L = [f"n = {n}   mean {secs:.0f}s per item", ""]
    L.append(f"  T1 elaborates            {t1:3d}/{n}  {t1/n:5.0%}")
    L.append(f"  T2 equivalent to gold    {t2:3d}/{n}  {t2/n:5.0%}   <- the score")
    L.append(f"     (one direction only   {fwd - t2:3d}       "
             f"i.e. compiles and is strictly stronger or weaker)")
    L.append(f"  T3 judge says faithful   {t3:3d}      among the {n - t2} T2 failures")
    L.append("")
    L.append("  T2 is a LOWER bound: a real equivalence the ladder cannot find")
    L.append("  scores as a failure. The T2/T3 gap is the measurement error bar,")
    L.append("  not a nuisance -- do not add them together.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--names", nargs="+", default=[],
                    help="score only these benchmark items, by name")
    ap.add_argument("--names-file", default="",
                    help="JSON list of item names to score. Prefer this over "
                         "--names: zsh does not word-split an unquoted "
                         "expansion, so a shell variable holding many names "
                         "arrives as ONE argument and matches nothing.")
    ap.add_argument("--out", default="",
                    help="write results here instead of the default path")
    ap.add_argument("--repeat", type=int, default=1,
                    help="score each item N times; the per-attempt derivation "
                         "rate is 40-67%%, so a single pass cannot separate a "
                         "prompt change from a coin flip")
    a = ap.parse_args()

    if not os.path.exists(BENCH):
        print(f"no benchmark at {BENCH}; run formalize_bench.py --verify first")
        return
    all_items = json.load(open(BENCH))
    names = list(a.names)
    if a.names_file:
        names += json.load(open(a.names_file))
    if names:
        want = set(names)
        items = [it for it in all_items if it["name"] in want]
        missing = want - {it["name"] for it in items}
        if missing:
            print(f"not in benchmark: {sorted(missing)}")
    else:
        items = all_items[a.offset:a.offset + a.n]
    if a.repeat > 1:
        items = [dict(it, _rep=k) for it in items for k in range(a.repeat)]
    global OUT
    if a.out:
        OUT = a.out
    print(f"scoring {len(items)} item-runs\n")

    rows = []
    if a.workers > 1:
        with futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(score_one, items):
                rows.append(r)
                print(f"  {'T1' if r.get('t1') else '--'} "
                      f"{'T2' if r.get('t2') else '--'}  {r['name'][:52]}", flush=True)
                json.dump(rows, open(OUT, "w"), indent=1)
    else:
        for it in items:
            r = score_one(it)
            rows.append(r)
            print(f"  {'T1' if r.get('t1') else '--'} "
                  f"{'T2' if r.get('t2') else '--'}  {r['name'][:52]}", flush=True)
            json.dump(rows, open(OUT, "w"), indent=1)

    print("\n" + "=" * 66)
    print(report(rows))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
