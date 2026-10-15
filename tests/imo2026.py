"""IMO 2026 tractable cores, run against the escalating multi-attempt prover.

Full problems are out of reach: AxiomProver's formal solutions run 457-4229
lines and took 24-869 minutes each. This probes extractable cores plus one full
problem to locate the ceiling honestly.

Search is DISABLED (default) -- solutions are on the open web.
"""

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, prover, search  # noqa: E402

TESTS = [
    ("P5 core: QM-AM-GM chain",
     "For all positive real numbers x and y, "
     "Real.sqrt ((x^2 + y^2)/2) >= (x + y)/2 and (x + y)/2 >= Real.sqrt (x*y)."),

    ("P5: f(x)=x+c satisfies AM-GM half",
     "For all positive real numbers x, y and all c >= 0, "
     "((x + c) + y)/2 >= Real.sqrt (x * (y + c))."),

    ("P6 core: gcd characterization",
     "For all positive integers a and b, Nat.gcd a b divides a and Nat.gcd a b divides b, "
     "and if d divides a and d divides b then d divides Nat.gcd a b."),

    ("P1 core: product becomes lcm",
     "For all natural numbers m and n, Nat.gcd m n * (Nat.lcm m n / Nat.gcd m n) "
     "= Nat.lcm m n."),

    ("P5 FULL (ceiling probe)",
     "Determine all functions f from positive reals to positive reals such that "
     "Real.sqrt ((x^2 + (f y)^2)/2) >= (f x + y)/2 and (f x + y)/2 >= Real.sqrt (x * f y) "
     "for all positive x and y. Prove that f(x) = x + c for some c >= 0."),
]


def main() -> None:
    print(f"search: {'ENABLED' if search.enabled() else 'disabled'} | "
          f"attempts={config.PROVE_ATTEMPTS} rounds={config.PROVE_ROUNDS} "
          f"budget={config.PROVE_TIME_BUDGET}s\n", flush=True)

    summary, reasons, winning = [], Counter(), Counter()

    for name, statement in TESTS:
        print(f"\n{'=' * 68}\n{name}\n{'=' * 68}", flush=True)
        t0 = time.time()
        r = prover.prove_theorem(statement)
        elapsed = round(time.time() - t0, 1)
        tries = r.get("attempts", [])

        status = ("PROVED" if r.get("proved")
                  else "ALL-MISFORMALIZED" if r.get("formalization_suspect")
                  else "FAILED")
        print(f"{status} | {len(tries)} attempt(s) | {elapsed}s", flush=True)

        # Per-attempt detail is where the limitations show up.
        for t in tries:
            mark = "OK" if t.get("ok") else (t.get("formalization") or t.get("reason") or "?")
            print(f"   attempt {t.get('attempt')} (T={t.get('temp')}): "
                  f"{mark} after {t.get('rounds')} round(s)", flush=True)
            if not t.get("ok"):
                reasons[str(t.get("reason"))] += 1

        if r.get("proved"):
            won = tries[-1]
            winning[f"attempt {won.get('attempt')} (T={won.get('temp')})"] += 1
            print("\n" + r["lean"], flush=True)

        summary.append({"test": name, "status": status,
                        "attempts": len(tries), "s": elapsed})

    print("\n" + "=" * 68 + "\nSUMMARY")
    for s in summary:
        print(f"  {s['status']:18} {s['test']}  ({s['attempts']} attempts, {s['s']}s)")
    print("\nfailure reasons:", dict(reasons))
    print("winning attempts:", dict(winning))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
