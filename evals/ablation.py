#!/usr/bin/env python3
"""A/B the game-theoretic layer. Does any of it earn its cost?

WHAT THIS IS FOR
----------------
Four mechanisms were added -- recall, scout, forecast, retirement -- each with a
plausible story. Plausible stories are exactly what this repo keeps disproving:
`think=high` looked like a large improvement until the noise floor was measured
and three of four flips turned out to be coin tosses. So the mechanisms ship
behind flags and get measured, arm against arm, on hard problems.

THE STATISTICAL PROBLEM, STATED UP FRONT
----------------------------------------
The measured per-attempt success rate on this machine is 40%, and a full
end-to-end problem run costs 30-150 minutes. That combination is brutal for an
end-to-end A/B: at n=1 per arm, an arm that "wins" 1-0 is reporting a coin flip.
Distinguishing 40% from 60% at any useful confidence needs tens of runs per arm,
i.e. weeks of continuous compute.

So this harness offers two modes, and the cheap one is the one that can actually
conclude something:

  --mode component   Measures each mechanism against ground truth DIRECTLY, on
                     many claims, without running the agent loop. Does the scout
                     predict what Lean later says? Does the forecast RANK
                     correctly? These have n in the dozens within an hour or two
                     and give real answers.

  --mode e2e         Full agent runs, arm against arm, on whole problems. This
                     is what everyone wants and it is nearly uninterpretable at
                     affordable n. Reported with an explicit warning rather than
                     a p-value it cannot support.

Component mode is not a lesser substitute. A mechanism that cannot beat chance
at predicting Lean has no route to helping end-to-end, so it is the cheaper
falsification test and it comes first.

    python3 evals/ablation.py --mode component --questions questions/PUTNAM25
    python3 evals/ablation.py --mode e2e --arms none all --minutes 45
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, forecast, plan, recall, scout, trace   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation.json")

# Each arm is the environment it runs under. `none` is the pre-existing system
# and is the only baseline that means anything.
ARMS = {
    "none":     {"GT_RECALL": "0", "GT_SCOUT": "0", "GT_FORECAST": "0", "GT_RETIRE": "0"},
    "recall":   {"GT_RECALL": "1", "GT_SCOUT": "0", "GT_FORECAST": "0", "GT_RETIRE": "0"},
    "scout":    {"GT_RECALL": "0", "GT_SCOUT": "1", "GT_FORECAST": "0", "GT_RETIRE": "0"},
    "forecast": {"GT_RECALL": "0", "GT_SCOUT": "0", "GT_FORECAST": "1", "GT_RETIRE": "0"},
    "few":      {"GT_RECALL": "1", "GT_SCOUT": "1", "GT_FORECAST": "0", "GT_RETIRE": "0"},
    "all":      {"GT_RECALL": "1", "GT_SCOUT": "1", "GT_FORECAST": "1", "GT_RETIRE": "1"},
}


def problems(paths: list) -> list:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.txt")))
        elif os.path.isfile(p):
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# COMPONENT MODE
# ---------------------------------------------------------------------------

def claims_for(path: str, n: int = 3) -> list:
    """Obligations a problem decomposes into -- the unit the agent works on.

    Uses the same ledger builder the agent uses, so the claims measured are the
    claims that would actually have been attempted, not a hand-picked set that
    flatters whichever mechanism is being tested.
    """
    goal = open(path, encoding="utf-8", errors="ignore").read()
    try:
        plan.build(goal, force=True)
        return [o["claim"] for o in (plan.remaining(goal) or [])
                if o.get("claim") and o.get("kind") != "closure"][:n]
    except Exception:
        return []


def component(paths: list, per_problem: int = 3) -> dict:
    """Score scout and forecast against what Lean actually says.

    The expensive step -- the real proof attempt -- is the ground truth, and it
    is run ONCE per claim and shared by both mechanisms. Neither gets its own
    sample, so neither can win on luck the other did not get.
    """
    from src import prover
    rows, t0 = [], time.time()
    for path in paths:
        name = os.path.basename(path)
        for claim in claims_for(path, per_problem):
            print(f"\n--- {name}: {claim[:64]}", flush=True)
            sc = scout.scout_one(claim)
            print(f"    scout    {sc['status']:6s} ({sc['seconds']}s)", flush=True)
            p, why = forecast.ask(claim, extra=sc.get("note", ""))
            print(f"    forecast p={p}", flush=True)
            if sc["status"] == scout.MISMATCH:
                # No ground truth to collect: a proof of a signature that says
                # the wrong thing tells us nothing about the claim.
                print("    LEAN     skipped -- signature does not match the claim",
                      flush=True)
                rows.append({"problem": name, "claim": claim[:200],
                             "scout": sc["status"], "scout_secs": sc["seconds"],
                             "p": p, "truth": None, "prove_secs": 0.0,
                             "note": sc.get("note", "")[:200]})
                json.dump(rows, open(OUT, "w"), indent=1)
                continue
            g0 = time.time()
            try:
                res = prover.prove_theorem(claim, sc.get("lean", ""),
                                           time_budget=600)
                truth = bool(res.get("proved"))
            except Exception as e:
                print(f"    prover failed: {type(e).__name__}")
                continue
            print(f"    LEAN     {'PROVED' if truth else 'failed'} "
                  f"({time.time() - g0:.0f}s)", flush=True)
            rows.append({"problem": name, "claim": claim[:200],
                         "scout": sc["status"], "scout_secs": sc["seconds"],
                         "p": p, "truth": truth,
                         "prove_secs": round(time.time() - g0, 1)})
            json.dump(rows, open(OUT, "w"), indent=1)
    return {"rows": rows, "elapsed": round(time.time() - t0, 1)}


def analyse(rows: list) -> str:
    """What the component run supports, and explicitly what it does not."""
    if not rows:
        return "no rows"
    # Claims whose signature did not match are reported, never scored: their
    # `truth` is None because no proof of them would have meant anything.
    mismatched = [r for r in rows if r.get("truth") is None]
    rows = [r for r in rows if r.get("truth") is not None]
    if not rows:
        return f"no scorable rows ({len(mismatched)} mismatched signatures)"
    n = len(rows)
    base = sum(r["truth"] for r in rows) / n
    out = [f"n = {n} scorable claims, base rate {base:.0%} proved"]
    if mismatched:
        out.append(f"  ({len(mismatched)} excluded: the derived signature did "
                   "not say what the claim said -- proving those would have "
                   "banked a verified proof of the wrong statement)")
    out.append("")

    # SCOUT. The claim under test: `solved` should be near-certainly provable
    # and `broken` should be near-certainly not. Anything else and the scout is
    # spending time to produce a label nobody should act on.
    out.append("SCOUT -- does a cheap probe predict the expensive verdict?")
    for st in (scout.SOLVED, scout.OPEN, scout.BROKEN):
        sub = [r for r in rows if r["scout"] == st]
        if sub:
            hit = sum(r["truth"] for r in sub) / len(sub)
            secs = sum(r["scout_secs"] for r in sub) / len(sub)
            out.append(f"  {st:7s} n={len(sub):3d}  proved {hit:5.0%}   "
                       f"scout cost {secs:5.0f}s")
    saved = [r for r in rows if r["scout"] == scout.BROKEN and not r["truth"]]
    if saved:
        wasted = sum(r["prove_secs"] for r in saved) / 60
        out.append(f"  -> {len(saved)} claim(s) flagged BROKEN did indeed fail, "
                   f"after {wasted:.0f} min of prover time that a scout could "
                   "have redirected.")

    # FORECAST. Ranking and calibration, separately, because they fail
    # independently and allocation only needs the first.
    pairs = [(r["p"], r["truth"]) for r in rows if r.get("p") is not None]
    out.append("")
    out.append("FORECAST -- does the model know what it can prove?")
    if len(pairs) < 4:
        out.append(f"  only {len(pairs)} scored forecasts; not enough to say")
    else:
        b = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        br = sum(y for _, y in pairs) / len(pairs)
        bb = sum((br - y) ** 2 for _, y in pairs) / len(pairs)
        out.append(f"  n={len(pairs)}  Brier {b:.4f}  vs base-rate {bb:.4f}  "
                   f"skill {(bb - b) / max(bb, 1e-9):+.1%}")
        out.append(f"  AUC {forecast._auc(pairs):.3f}  "
                   "(0.5 = coin; ranking is what allocation needs)")
        if len(pairs) < 20:
            out.append("  CAUTION: under ~20 adjudicated forecasts an AUC this "
                       "far from 0.5 can arise by chance. Treat as directional.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# E2E MODE
# ---------------------------------------------------------------------------

def e2e(arms: list, paths: list, minutes: int) -> list:
    """Full agent runs, one per (arm, problem). Expensive and low-powered."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = []
    for arm in arms:
        env = dict(os.environ, **ARMS[arm])
        for path in paths:
            print(f"\n=== arm={arm}  {os.path.basename(path)}  "
                  f"({minutes} min) ===", flush=True)
            before = len(trace.load())
            t0 = time.time()
            try:
                subprocess.run(
                    [sys.executable, "run.py", "--questions", path,
                     "--problem-minutes", str(minutes), "--max-steps", "0"],
                    cwd=root, env=env, timeout=minutes * 60 + 900)
            except subprocess.TimeoutExpired:
                print("  (timed out)", flush=True)
            rows = trace.load()[before:]
            proofs = [r for r in rows if r.get("kind") == "proof"]
            row = {"arm": arm, "problem": os.path.basename(path),
                   "minutes": minutes, "elapsed": round(time.time() - t0, 1),
                   "attempts": len(proofs),
                   "verified": sum(bool(r.get("ok")) for r in proofs)}
            res.append(row)
            print(f"  -> {row['verified']}/{row['attempts']} verified", flush=True)
            json.dump(res, open(OUT.replace(".json", "_e2e.json"), "w"), indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("component", "e2e"), default="component")
    ap.add_argument("--questions", nargs="+",
                    default=["questions/IMO26_test", "questions/PUTNAM25"])
    ap.add_argument("--arms", nargs="+", default=["none", "all"])
    ap.add_argument("--minutes", type=int, default=45)
    ap.add_argument("--per-problem", type=int, default=2)
    a = ap.parse_args()

    paths = problems(a.questions)
    if not paths:
        print(f"no problems found under {a.questions}")
        return
    print(f"{len(paths)} problem(s); mode={a.mode}")

    if a.mode == "component":
        r = component(paths, a.per_problem)
        print("\n" + "=" * 76)
        print(analyse(r["rows"]))
        print(f"\n{r['elapsed'] / 60:.0f} min; written to {OUT}")
    else:
        bad = [x for x in a.arms if x not in ARMS]
        if bad:
            print(f"unknown arms {bad}; available: {sorted(ARMS)}")
            return
        rows = e2e(a.arms, paths, a.minutes)
        print("\n" + "=" * 76)
        for arm in a.arms:
            sub = [r for r in rows if r["arm"] == arm]
            v = sum(r["verified"] for r in sub)
            t = sum(r["attempts"] for r in sub)
            print(f"  {arm:9s} {v:3d} verified / {t:3d} attempts "
                  f"over {len(sub)} problem(s)")
        print("\n  With one run per (arm, problem) and a measured 40% per-attempt")
        print("  noise floor, a small difference here is NOT evidence. Use this")
        print("  to catch a mechanism that BREAKS the loop, not to rank the ones")
        print("  that do not.")


if __name__ == "__main__":
    main()
