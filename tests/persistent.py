"""Persistent run: keep trying new approaches until each claim is proved.

Unbounded attempts (`attempts=0`) -- the loop cycles strategies, escalates
temperature each cycle, and feeds every failed approach forward so later attempts
cannot rediscover them. Stops a claim only when proved or its time cap expires.

Includes sqrt-2, which previously failed six identical rounds on a hallucinated
lemma; it is the direct test of whether the anti-stall work bites.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, prover, search  # noqa: E402

# (name, informal claim, optional pinned Lean signature)
TESTS = [
    ("sqrt2 irrational (known-hard)",
     "The square root of 2 is irrational.", ""),

    ("P6 core: gcd characterization",
     "For all natural numbers a b d, if d divides a and d divides b then "
     "d divides Nat.gcd a b.",
     "theorem p6_core (a b d : ℕ) (ha : d ∣ a) (hb : d ∣ b) : d ∣ Nat.gcd a b"),

    ("P5 core: QM-AM-GM chain",
     "For all positive real numbers x and y, "
     "Real.sqrt ((x^2 + y^2)/2) >= (x + y)/2 and (x + y)/2 >= Real.sqrt (x*y).", ""),

    ("P1 core: product becomes lcm",
     "For all natural numbers m and n, "
     "Nat.gcd m n * (Nat.lcm m n / Nat.gcd m n) = Nat.lcm m n.", ""),

    ("P5 FULL (ceiling probe)",
     "Determine all functions f from positive reals to positive reals such that "
     "Real.sqrt ((x^2 + (f y)^2)/2) >= (f x + y)/2 and (f x + y)/2 >= Real.sqrt (x * f y) "
     "for all positive x and y. Prove that f(x) = x + c for some c >= 0.", ""),
]

PER_CLAIM_SECONDS = int(os.environ.get("PER_CLAIM_SECONDS", "1500"))


def main() -> None:
    print(f"search: {'ENABLED' if search.enabled() else 'disabled'} | "
          f"UNBOUNDED attempts | {PER_CLAIM_SECONDS}s per claim | "
          f"rounds={config.PROVE_ROUNDS}\n", flush=True)
    summary = []

    for name, claim, pinned in TESTS:
        print(f"\n{'=' * 70}\n{name}{'   [PINNED STATEMENT]' if pinned else ''}\n{'=' * 70}",
              flush=True)
        t0 = time.time()
        r = prover.prove_theorem(claim, lean_statement=pinned, attempts=0,
                                 time_budget=PER_CLAIM_SECONDS)
        elapsed = round(time.time() - t0, 1)
        tries = r.get("attempts", [])
        status = ("PROVED" if r.get("proved")
                  else "ALL-MISFORMALIZED" if r.get("formalization_suspect")
                  else "EXHAUSTED")
        print(f"\n{status} | {len(tries)} attempt(s) | {elapsed}s", flush=True)
        for t in tries:
            mark = "OK" if t.get("ok") else (t.get("formalization") or t.get("reason") or "?")
            print(f"   attempt {t.get('attempt')} T={t.get('temp')} "
                  f"mode={t.get('mode') or '-'}: {mark} ({t.get('rounds')}r)", flush=True)
        if r.get("proved"):
            print("\n" + r["lean"], flush=True)

        summary.append({"test": name, "status": status, "attempts": len(tries),
                        "s": elapsed, "pinned": bool(pinned)})

    print("\n" + "=" * 70 + "\nSUMMARY")
    for s in summary:
        print(f"  {s['status']:18} {s['test']}  ({s['attempts']} attempts, {s['s']}s)")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
