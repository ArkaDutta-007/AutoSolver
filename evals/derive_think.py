#!/usr/bin/env python3
"""Does thinking help or destroy signature derivation? Measured, not argued.

THE CONFLICT THIS SETTLES
-------------------------
Two rules in this repo point opposite ways at the same call site.

  * `_derive_once` sends `format: SIG_SCHEMA` together with `think`. Four times
    now, a grammar-constrained schema and a reasoning budget have been observed
    competing for ONE token allowance, with the JSON losing -- it truncates
    mid-object, json.loads throws, and the call yields nothing at all.
  * But test_units.py deliberately asserts that think=high DOES reach this call,
    on the view that turning informal prose into a Lean signature is a genuine
    translation problem that reasoning should help with.

Both are plausible. The observation that forced the question: one derivation
under think=high ran 2,457 seconds -- roughly 22,000 tokens at the measured 9.2
tok/s -- and returned an empty signature. That is not a budget that ran out
early; it is a budget consumed by reasoning that never got to the answer.

WHAT IS MEASURED
----------------
The same claims, derived under each think setting, N times each. Two outcomes
matter and they are not the same question:

  RATE     did a signature come back that Lean will elaborate?
  SECONDS  what did it cost? A mode that is 5% better and 40x slower is worse,
           because the scout mechanism this feeds is only worth having if
           observation is cheaper than commitment.

The per-attempt derivation rate on this machine was previously measured at 40%,
so N below ~5 per cell cannot separate 40% from 60%. This prints the counts and
refuses to dress them up.

    python3 evals/derive_think.py -n 3
    python3 evals/derive_think.py -n 5 --modes high off
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, prover                              # noqa: E402
import evals.formalization_v2 as v2                         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derive_think.json")


def one(claim: str) -> tuple:
    """One derivation attempt. (ok, seconds, signature)."""
    t0 = time.time()
    try:
        sig = prover._derive_once_untraced(claim)
    except Exception:
        sig = ""
    secs = round(time.time() - t0, 1)
    if not sig:
        return False, secs, ""
    try:
        ok, _ = v2.elaborates(sig)
    except Exception:
        ok = False
    return bool(ok), secs, sig[:160]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3, help="attempts per (claim, mode)")
    ap.add_argument("--modes", nargs="+", default=["high", "off"])
    ap.add_argument("--cases", nargs="+", default=[])
    a = ap.parse_args()

    wanted = {c[0]: c for c in v2.CASES}
    ids = a.cases or ["co_sum_first_n", "ge_triangle_ineq", "an_cont_compact"]
    ids = [i for i in ids if i in wanted]
    if not ids:
        print(f"no known cases; available: {sorted(wanted)[:12]}")
        return

    print(f"model={config.ORCHESTRATOR_MODEL[:44]}  n={a.n}  modes={a.modes}")
    print(f"{'mode':6s} {'case':20s} {'rate':>7s} {'mean s':>8s}  outcomes")
    print("-" * 68)

    results, prior = [], config.ORCH_THINK
    try:
        for mode in a.modes:
            config.ORCH_THINK = mode
            for cid in ids:
                _, _, claim, _, _, _ = wanted[cid]
                runs = []
                for i in range(a.n):
                    ok, secs, sig = one(claim)
                    runs.append({"ok": ok, "secs": secs, "sig": sig})
                    print(f"    ...{mode}/{cid} {i + 1}/{a.n}: "
                          f"{'PASS' if ok else 'fail'} ({secs}s)", flush=True)
                hits = sum(r["ok"] for r in runs)
                mean = sum(r["secs"] for r in runs) / max(len(runs), 1)
                print(f"{mode:6s} {cid:20s} {hits}/{a.n:<5d} {mean:8.0f}  "
                      f"{''.join('P' if r['ok'] else '.' for r in runs)}", flush=True)
                results.append({"mode": mode, "case": cid, "n": a.n,
                                "passes": hits, "mean_secs": round(mean, 1),
                                "runs": runs})
                json.dump(results, open(OUT, "w"), indent=1)
    finally:
        config.ORCH_THINK = prior

    print("\n" + "=" * 68)
    for mode in a.modes:
        sub = [r for r in results if r["mode"] == mode]
        if not sub:
            continue
        p = sum(r["passes"] for r in sub)
        n = sum(r["n"] for r in sub)
        secs = sum(r["mean_secs"] * r["n"] for r in sub) / max(n, 1)
        print(f"  think={mode:5s}  {p}/{n} derivations elaborate  "
              f"({p / max(n, 1):.0%})   mean {secs:.0f}s per attempt")
    print("\n  Cost matters as much as rate here: this call feeds the scout, and")
    print("  a scout is only worth having when observing is cheaper than")
    print("  committing. A 41-minute observation is not an observation.")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
