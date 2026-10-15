"""Smoke test: real problems through the full prove_theorem pipeline."""

import json
import sys
import time

import tools

TESTS = [
    ("IMO 2026 P1 invariant",
     "For all positive integers m and n, gcd(m, n) * lcm(m, n) = m * n. "
     "(This is the invariant behind IMO 2026 Problem 1: replacing m, n on a "
     "blackboard by gcd(m,n), lcm(m,n) preserves the product of all entries.)"),

    ("Infinitude of primes",
     "There are infinitely many prime numbers."),

    ("Irrationality of sqrt 2",
     "The square root of 2 is irrational."),

    ("Physics: SHM energy",
     "For real numbers A, w, t, the quantity (A*w*cos(w*t))^2 + (w*A*sin(w*t))^2 "
     "equals (A*w)^2. (Conservation of energy for a simple harmonic oscillator: "
     "kinetic plus potential energy is constant.)"),
]


def main() -> None:
    which = sys.argv[1:] or None
    for name, statement in TESTS:
        if which and not any(w.lower() in name.lower() for w in which):
            continue
        print(f"\n{'=' * 64}\n{name}\n{'=' * 64}", flush=True)
        started = time.time()
        r = tools.prove_theorem(statement)
        print(f"proved: {r.get('proved')} | rounds: {r.get('rounds')} | "
              f"{round(time.time() - started, 1)}s", flush=True)
        if r.get("formalization_suspect"):
            print("  !! FORMALIZATION SUSPECT — compiled, but statement != claim")
            print("  reviewer:", (r.get("formalization", {}).get("verdict") or "")[-300:])
            print("  statement:", (r.get("formalization", {}).get("statement") or "")[:300])
        elif r.get("formalization", {}).get("faithful"):
            print("  formalization reviewed: FAITHFUL")
        if r.get("proved"):
            print("--- verified Lean ---")
            print(r["lean"])
        else:
            for a in r.get("attempts", []):
                print(f"  round {a['round']}: {a['errors'][:220]}")
        print(json.dumps({"test": name, "proved": r.get("proved"),
                          "rounds": r.get("rounds")}), flush=True)


if __name__ == "__main__":
    main()
