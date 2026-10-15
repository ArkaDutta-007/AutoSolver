"""Which strategy gets the next expensive attempt? A bandit, not a fixed ladder.

THE PROBLEM
-----------
`prove_theorem` escalates through a fixed ladder of attempt modes -- plain,
sketch, search, and temperature variations -- in the same order every time,
regardless of what has actually been working. With PROVE_ATTEMPTS=5 and a
per-attempt cost measured in minutes, that ladder is spending the most
expensive resource in the system on a hard-coded guess.

It is a textbook multi-armed bandit: several arms (strategies), a stochastic
binary reward (Lean accepts or does not), and a budget far smaller than the
number of pulls needed to be sure. UCB1 is the right default -- it needs no
tuning, has a regret bound, and its exploration term automatically re-tries an
arm that has been starved.

WHY NOT JUST PICK THE BEST ARM
------------------------------
Because "best" is domain-dependent and this system works across number theory,
analysis, and physics. An arm that wins on divisibility can be useless on
compactness. Stats are therefore kept PER CONTEXT (the archive's object tag),
so the bandit learns "sketch mode works on analysis" rather than a single
global ranking that is wrong half the time.

Persisted across sessions: a bandit that forgets on restart never leaves the
exploration phase, and this system's whole problem is that each run started
from nothing.
"""

import json
import math
import os
import threading
import time

from . import config

PATH = os.path.join(config.ROOT, "kb", "bandit.json")
_lock = threading.RLock()

# Default arms mirror prover.STRATEGY_NAMES. Not imported from prover -- prover
# imports this module, and a cycle here would break both. Callers pass their own
# arms explicitly; this default exists so the module is usable standalone.
ARMS = ("direct", "leansearch", "decompose", "reroute", "automation", "generalize")


def _load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save(st: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, PATH)


def _ctx(context: str) -> str:
    return (context or "general").strip().lower()[:40] or "general"


def stats(context: str = "general") -> dict:
    return _load().get(_ctx(context), {})


def select(context: str = "general", arms: tuple = ARMS, exclude: tuple = ()) -> str:
    """UCB1: the arm maximising mean + sqrt(2 ln N / n).

    An UNTRIED arm is always chosen first. That is not a special case bolted on
    -- it is UCB1's own limit behaviour (infinite uncertainty) and it is what
    guarantees the bandit cannot lock onto arm 1 because it happened to win
    once before anything else was measured.
    """
    pool = [a for a in arms if a not in exclude]
    if not pool:
        return arms[0]
    st = stats(context)
    total = sum(st.get(a, {}).get("n", 0) for a in pool)
    untried = [a for a in pool if st.get(a, {}).get("n", 0) == 0]
    if untried:
        return untried[0]
    best, best_score = pool[0], -1.0
    for a in pool:
        rec = st.get(a, {"n": 1, "wins": 0})
        n = max(rec.get("n", 1), 1)
        mean = rec.get("wins", 0) / n
        score = mean + math.sqrt(2.0 * math.log(max(total, 2)) / n)
        if score > best_score:
            best, best_score = a, score
    return best


def update(arm: str, won: bool, context: str = "general", seconds: float = 0.0) -> dict:
    """Record the outcome of one pull."""
    with _lock:
        st = _load()
        ctx = st.setdefault(_ctx(context), {})
        rec = ctx.setdefault(arm, {"n": 0, "wins": 0, "seconds": 0.0})
        rec["n"] += 1
        rec["wins"] += bool(won)
        rec["seconds"] = round(rec.get("seconds", 0.0) + max(seconds, 0.0), 1)
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save(st)
        return dict(rec)


def ranking(context: str = "general") -> list:
    """Arms by measured win rate, with counts. For reporting, not selection."""
    st = stats(context)
    out = []
    for a in ARMS:
        rec = st.get(a)
        if not rec or not rec.get("n"):
            out.append({"arm": a, "n": 0, "rate": None, "avg_seconds": None})
            continue
        out.append({"arm": a, "n": rec["n"],
                    "rate": round(rec["wins"] / rec["n"], 3),
                    "avg_seconds": round(rec.get("seconds", 0.0) / rec["n"], 1)})
    return sorted(out, key=lambda r: (-(r["rate"] or -1), -r["n"]))


def reset() -> None:
    with _lock:
        _save({})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="strategy bandit")
    ap.add_argument("--context", default="general")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        reset(); print("bandit cleared")
    else:
        print(f"context: {a.context}")
        for r in ranking(a.context):
            print(f"  {r['arm']:8s} n={r['n']:3d} rate={r['rate']} avg={r['avg_seconds']}s")
        print(f"next pick: {select(a.context)}")
