#!/usr/bin/env python3
"""Is a PASS/FAIL flip a real effect, or a coin toss? Measure the noise floor.

WHY THIS EXISTS
---------------
Raising the orchestrator from think=low to think=high flipped four cases:

    an_cont_compact    FAIL -> PASS    1345s
    co_sum_first_n     FAIL -> PASS     445s   (never passed in ANY earlier run)
    ge_triangle_ineq   FAIL -> PASS    4322s   (geometry was 0/N everywhere)
    ph_shm_energy      PASS -> FAIL    2196s

Three improved, one regressed -- and every one of those is a SINGLE sample.
SIG@3 derives at temperature 0.2, which is low but not zero, so each case is a
coin of unknown bias. `la_det_mul` demonstrated this directly: it failed both
hand-tests and passed twice in the harness, same model, same prompt.

So a flip means nothing until the per-case variance is known. This runs the same
case N times at a FIXED setting and reports the pass rate. Only after that is a
temperature sweep interpretable -- otherwise a "temperature effect" is
indistinguishable from the noise it is sitting on.

    python3 evals/stability.py --cases ge_triangle_ineq co_sum_first_n -n 5
    python3 evals/stability.py --cases ph_shm_energy -n 5 --temps 0.2 0.5 0.8
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, prover                              # noqa: E402
import evals.formalization_v2 as v2                          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stability.json")


def one(claim: str, temp: float) -> tuple:
    """One signature derivation at a given temperature. (ok, seconds, sig)."""
    prior = prover._DERIVE_TEMP[0] if hasattr(prover, "_DERIVE_TEMP") else None
    t0 = time.time()
    try:
        if prior is not None:
            prover._DERIVE_TEMP[0] = temp
        sig = prover._derive_once(claim)
    except Exception:
        sig = ""
    finally:
        if prior is not None:
            prover._DERIVE_TEMP[0] = prior
    ok, _ = v2.elaborates(sig)
    return bool(ok), round(time.time() - t0, 1), sig[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="case ids from v2.CASES")
    ap.add_argument("--models", nargs="+", default=[config.ORCHESTRATOR_MODEL])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.2])
    ap.add_argument("-n", type=int, default=5, help="repeats per (case, temp)")
    a = ap.parse_args()

    wanted = {c[0]: c for c in v2.CASES}
    missing = [c for c in a.cases if c not in wanted]
    if missing:
        print(f"unknown case ids: {missing}\navailable: {sorted(wanted)}")
        return

    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    print(f"think={config.ORCH_THINK}  repeats={a.n}  temps={a.temps}")
    print(f"{'model':22s} {'case':20s} {'temp':>5s} {'pass rate':>10s}  {'mean s':>7s}  outcomes")
    print("-" * 88)

    for model in a.models:
        prior_model, config.ORCHESTRATOR_MODEL = config.ORCHESTRATOR_MODEL, model
        try:
            for cid in a.cases:
                _, _, claim, _, _, _ = wanted[cid]
                for temp in a.temps:
                    runs = []
                    for i in range(a.n):
                        ok, secs, sig = one(claim, temp)
                        runs.append({"ok": ok, "secs": secs, "sig": sig})
                        print(f"  ...{cid} t={temp} run {i+1}/{a.n}: "
                              f"{'PASS' if ok else 'FAIL'} ({secs}s)", flush=True)
                    hits = sum(r["ok"] for r in runs)
                    mean = sum(r["secs"] for r in runs) / max(len(runs), 1)
                    marks = "".join("P" if r["ok"] else "." for r in runs)
                    print(f"{model[:22]:22s} {cid:20s} {temp:5.1f} "
                          f"{hits}/{a.n:<8d} {mean:7.0f}  {marks}", flush=True)
                    results.append({"model": model, "case": cid, "temp": temp,
                                    "n": a.n, "passes": hits,
                                    "think": config.ORCH_THINK, "runs": runs})
                    json.dump(results, open(OUT, "w"), indent=1)
        finally:
            config.ORCHESTRATOR_MODEL = prior_model

    print("\n" + "=" * 88)
    print("A case that is 5/5 or 0/5 is STABLE -- a flip there means something real.")
    print("Anything in between is a coin, and one sample of it proves nothing.")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
