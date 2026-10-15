#!/usr/bin/env python3
"""Does the model know what it can prove? Scored against Lean, not against itself.

WHY THIS IS THE GATING MEASUREMENT
----------------------------------
A large family of appealing ideas -- budget allocation across obligations,
Kelly-style betting on sub-lemmas, "attempt the barely-provable statement to
maximise information", Stackelberg scouting before committing an hour -- all
reduce to the same requirement: a usable estimate of P(this succeeds) BEFORE
paying for the attempt. Every one of them is worthless if that estimate is
noise, and every one of them is nearly free if it is not.

So this measures it directly, and it can do so honestly because the ground
truth already exists: `runs/*.jsonl` holds 39 prove_theorem calls whose Lean
verdicts are known. The model is asked for a probability on each statement with
no access to the outcome, and is scored by a STRICTLY PROPER rule, which is the
only kind that cannot be gamed by hedging.

  Brier   = mean (p - outcome)^2         lower is better
  Base    = the score of always guessing the base rate. BEATING THIS IS THE
            WHOLE TEST -- a model that says 0.5 to everything scores 0.25 and
            has told you nothing.
  AUC     = does it RANK correctly, even if the absolute numbers are off?
            Ranking is what budget allocation actually needs, so a model can
            fail calibration and still be useful here.

Both numbers matter and they fail independently. Reporting only one would hide
the case this is being run to detect.

    python3 evals/calibration.py            # all recovered statements
    python3 evals/calibration.py -n 20      # cheaper sample
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests                                             # noqa: E402

from src import config                                      # noqa: E402
import evals.harvest as harvest                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

_SCHEMA = {"type": "object",
           "properties": {"p": {"type": "number"}, "why": {"type": "string"}},
           "required": ["p"]}

PROMPT = """You are about to attempt a formal proof in Lean 4 with Mathlib.

CLAIM: {claim}

Before spending any compute, estimate the probability that a strong Lean prover
(Goedel-Prover-V2-32B, best-of-32 sampling, roughly 15 minutes of GPU) will
produce a COMPLETE proof that the compiler accepts -- no `sorry`, no axioms.

Be honest rather than encouraging. Most nontrivial olympiad sub-lemmas fail.
A statement that is one `simp` away should score above 0.9; a statement needing
a genuine construction should score below 0.2.

Answer as JSON: "p" = your probability between 0 and 1, "why" = one short line.
"""


def ask(claim: str, temp: float = 0.3, timeout: int = 180):
    """One probability. Thinking OFF -- this is a judgement, not a derivation.

    Measured three times in this repo now: reasoning and a grammar-constrained
    schema compete for one token budget and the JSON is what loses.
    """
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.ORCHESTRATOR_MODEL,
                  "messages": [{"role": "user", "content": PROMPT.format(claim=claim[:1200])}],
                  "stream": False, "format": _SCHEMA, "think": False,
                  "options": {"num_predict": 300, "temperature": temp}},
            timeout=timeout)
        r.raise_for_status()
        d = json.loads(r.json()["message"]["content"])
        p = float(d.get("p", -1))
        return (p if 0.0 <= p <= 1.0 else None), str(d.get("why", ""))[:160]
    except Exception as e:
        return None, f"({type(e).__name__})"


def auc(pairs) -> float:
    """Probability a random success is ranked above a random failure.

    Computed by counting concordant pairs rather than by sorting, so ties score
    0.5 instead of silently depending on sort order.
    """
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0, help="sample this many (0 = all)")
    ap.add_argument("--temp", type=float, default=0.3)
    a = ap.parse_args()

    rs = harvest.rows()
    seen, cases = set(), []
    for r in rs:                              # one row per distinct statement
        k = r["statement"][:120]
        if k and k not in seen:
            seen.add(k)
            cases.append((r["statement"], bool(r["result"].get("proved"))))
    if a.n:
        random.seed(0)
        cases = random.sample(cases, min(a.n, len(cases)))

    base = sum(y for _, y in cases) / max(len(cases), 1)
    print(f"{len(cases)} statements with known Lean verdicts; base rate {base:.0%}")
    print(f"{'p':>6} {'actual':>7}  claim")
    print("-" * 78)

    pairs, out = [], []
    for claim, y in cases:
        p, why = ask(claim, a.temp)
        if p is None:
            print(f"{'--':>6} {'PROVED' if y else 'failed':>7}  (no answer) {why}")
            continue
        pairs.append((p, y))
        out.append({"claim": claim[:300], "p": p, "proved": y, "why": why})
        print(f"{p:6.2f} {'PROVED' if y else 'failed':>7}  {claim[:56]}", flush=True)

    if len(pairs) < 4:
        print("\nnot enough answers to score")
        return
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    baserate = sum(y for _, y in pairs) / len(pairs)
    bbase = sum((baserate - y) ** 2 for _, y in pairs) / len(pairs)
    mp = sum(p for p, _ in pairs) / len(pairs)

    print("\n" + "=" * 78)
    print(f"  n                     {len(pairs)}")
    print(f"  Brier                 {brier:.4f}")
    print(f"  Brier, base-rate only {bbase:.4f}   <- must beat this to have said anything")
    print(f"  skill vs base rate    {(bbase - brier) / max(bbase, 1e-9):+.1%}")
    print(f"  AUC (ranking)         {auc(pairs):.3f}   <- 0.5 is a coin; ranking is what")
    print(f"                                     budget allocation actually needs")
    print(f"  mean predicted        {mp:.3f}  vs actual {baserate:.3f}"
          f"   ({'OVER' if mp > baserate else 'under'}confident by {abs(mp - baserate):.3f})")
    json.dump({"pairs": out, "brier": brier, "brier_base": bbase,
               "auc": auc(pairs), "n": len(pairs), "temp": a.temp,
               "model": config.ORCHESTRATOR_MODEL, "when": time.time()},
              open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
