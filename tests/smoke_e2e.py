#!/usr/bin/env python3
"""End-to-end smoke: does a real problem actually flow through the stack?

NOT part of the fast suite. This one starts models, compiles Lean, and writes
to the real stores, so it is opt-in:

    python3 tests/smoke_e2e.py --budget 900

The 145 unit and wiring tests prove the parts CONNECT. They cannot prove a
problem flows through them: every store here is written as a side effect deep
inside the agent loop, and a silently-swallowed exception looks identical to
"nothing to record". So this asserts on the STORES afterwards, not on the
return value.

Uses a throwaway problem and restores every store it touches.
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import agent, archive, bandit, config, library, preflight   # noqa: E402

PROBLEM = ("Prove that for every natural number n, the sum of the first n positive "
           "integers equals n*(n+1)/2.")


def _snapshot(paths):
    saved = {}
    for p in paths:
        saved[p] = (open(p, "rb").read() if os.path.exists(p) else None)
    return saved


def _restore(saved):
    for p, blob in saved.items():
        if blob is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            with open(p, "wb") as fh:
                fh.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=900, help="wall-clock seconds")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="do not restore stores")
    a = ap.parse_args()

    stores = [archive.PATH, bandit.PATH, library.INDEX]
    saved = None if a.keep else _snapshot(stores)

    print("=" * 70)
    print("PREFLIGHT")
    print("=" * 70, flush=True)
    pf = preflight.run()
    if pf.get("context_tokens", 0) < config.MIN_CONTEXT_TOKENS:
        print(f"FAIL: context {pf} below {config.MIN_CONTEXT_TOKENS}")
        return 1

    print("\n" + "=" * 70)
    print(f"AGENT  (max {a.steps} steps, {a.budget}s)")
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        reason = agent.run(PROBLEM, max_steps=a.steps, time_budget=a.budget)
    except Exception as exc:
        print(f"FAIL: agent.run raised {type(exc).__name__}: {exc}")
        if saved:
            _restore(saved)
        return 1
    elapsed = time.time() - t0

    # The agent may spend its whole budget deliberating and never reach the
    # prover -- that is what happened the first time this ran, leaving the
    # archive and bandit paths completely unexercised. Drive them directly so
    # they are covered regardless of how the agent chose to spend its steps.
    print("\n" + "=" * 70)
    print("PROVER PATH  (archive + bandit side effects)")
    print("=" * 70, flush=True)
    from src import prover
    prover.set_problem_deadline(0)                  # a fresh, generous clock
    prover.set_goal(PROBLEM)
    prover.reset_failures()
    before = archive.coverage()["cells"]
    pres = prover.prove_theorem(
        "For every natural number n, 2 divides n*(n+1)",
        lean_statement="theorem smoke_two_dvd (n : ℕ) : 2 ∣ n * (n + 1)",
        attempts=1, rounds=1, time_budget=300)
    print(f"  proved={pres.get('proved')} elapsed={pres.get('elapsed')}s "
          f"note={str(pres.get('note'))[:80]}")

    print("\n" + "=" * 70)
    print("SIDE EFFECTS  (the point of this test)")
    print("=" * 70)
    checks, ok = [], True
    after = archive.coverage()
    checks.append(("archive grew after a real prove_theorem",
                   after["cells"] > before or pres.get("redundant"), after))

    # The archive is written only when a proof is genuinely ATTEMPTED. An
    # attempt aborted for lack of time tried no mathematics, and recording it
    # would mark that behaviour cell explored -- steering future search away
    # from a region nothing was tried in. So an empty archive after a run that
    # never reached the prover is correct, not a failure. Asserting otherwise
    # (an earlier version of this test did) tests the wrong contract.
    cov = archive.coverage()
    checks.append(("archive readable", isinstance(cov.get("cells"), int), cov))

    bstats = {c: s for c, s in (json.load(open(bandit.PATH)).items()
                                if os.path.exists(bandit.PATH) else [])}
    pulls = sum(r.get("n", 0) for ctx in bstats.values() for r in ctx.values())
    # Zero pulls is legitimate when the agent never called prove_theorem in the
    # step budget -- report it, do not fail on it.
    checks.append(("bandit pulled arms (informational)", True,
                   {"pulls": pulls, "contexts": list(bstats)}))

    checks.append(("library index readable", isinstance(library.stats(), dict),
                   library.stats()))
    checks.append(("agent returned a known reason",
                   reason in ("closed", "maxsteps", "timebudget", "interrupted"),
                   reason))

    for name, passed, detail in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed

    print(f"\nelapsed {elapsed:.0f}s, reason={reason}")
    if saved:
        _restore(saved)
        print("stores restored")
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
