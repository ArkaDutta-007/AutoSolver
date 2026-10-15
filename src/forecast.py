"""Ask what it expects to happen, score it against Lean, allocate accordingly.

THE MECHANISM
-------------
A whole family of allocation ideas -- spend the remaining hour on the obligation
most likely to close; attempt the barely-provable one because its label carries
the most information; abandon a line after two failures rather than six -- all
need the same input: P(success) BEFORE paying. None of them can be built on a
number nobody has checked.

So the elicitation here is scored by a STRICTLY PROPER rule. The Brier score is
minimised in expectation only by reporting one's actual belief, which is what
makes it worth asking at all; under an improper rule the right move is to hedge
toward the base rate and the answer carries no information. Every forecast is
written to the trace with the outcome that followed, so the score accumulates as
a by-product of ordinary running rather than needing a separate study.

TWO WAYS THIS CAN BE USEFUL AND THEY FAIL INDEPENDENTLY
-------------------------------------------------------
  CALIBRATION  does 0.7 mean 70%? Needed if a probability is used as a number,
               e.g. for an expected-value or Kelly-style computation.
  RANKING      does it order claims correctly? Needed if the probability is only
               used to choose what to do next.

Allocation needs ranking. A model can be badly miscalibrated -- systematically
overconfident, say -- and still rank perfectly, in which case the allocator
works fine and the raw numbers should never be used for anything else. Both are
reported. Reporting only one would hide the case this exists to detect.

A NOTE ON WHAT THIS IS NOT
--------------------------
This is not a truth signal. A confident forecast on a false claim is still a
false claim, and Lean remains the only thing that settles anything. The forecast
decides SPENDING, never belief.
"""

import json

import requests

from . import config, trace

_SCHEMA = {"type": "object",
           "properties": {"p": {"type": "number"}, "why": {"type": "string"}},
           "required": ["p"]}

PROMPT = """You are about to attempt a formal proof in Lean 4 with Mathlib.

CLAIM: {claim}
{extra}
Estimate the probability that a strong Lean prover (Goedel-Prover-V2-32B,
best-of-32 sampling, ~15 minutes of GPU) produces a COMPLETE proof the compiler
accepts -- no `sorry`, no extra axioms.

Be honest rather than encouraging. Most nontrivial olympiad sub-lemmas fail.
Something one `simp` away scores above 0.9; something needing a genuine
construction scores below 0.2.

JSON: "p" = probability in [0,1], "why" = one short line.
"""


def ask(claim: str, extra: str = "", temp: float = 0.3, timeout: int = 150):
    """One forecast. Returns (p, why) with p None when the call fails.

    Thinking is OFF. Measured four times in this repo now: reasoning and a
    grammar-constrained schema compete for one token budget and the JSON is what
    loses -- it truncates mid-object and json.loads throws, yielding nothing.
    A probability is a judgement, not a derivation; the budget belongs in the
    answer.
    """
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.ORCHESTRATOR_MODEL,
                  "messages": [{"role": "user", "content": PROMPT.format(
                      claim=str(claim)[:1200],
                      extra=f"\nWHAT A CHEAP PROBE FOUND: {extra}\n" if extra else "")}],
                  "stream": False, "format": _SCHEMA, "think": False,
                  "options": {**config.ctx(config.ORCHESTRATOR_MODEL), "num_predict": 300, "temperature": temp}},
            timeout=timeout)
        r.raise_for_status()
        d = json.loads(r.json()["message"]["content"])
        p = float(d.get("p", -1))
        return (p if 0.0 <= p <= 1.0 else None), str(d.get("why", ""))[:160]
    except Exception as e:
        return None, f"({type(e).__name__})"


def forecast(claims: list, scouted: list = None) -> list:
    """A forecast per claim, enriched with the scout's finding when there is one.

    Feeding the probe result in is the whole point of running the scout first:
    it converts a blind prior into a posterior, and it is free because the scout
    has already run.
    """
    if not config.GT_FORECAST:
        return []
    by = {s.get("claim", ""): s for s in (scouted or [])}
    out = []
    for c in claims:
        claim = c if isinstance(c, str) else c.get("claim", "")
        s = by.get(claim) or {}
        p, why = ask(claim, extra=s.get("note", ""))
        if p is None:
            continue
        out.append({"claim": claim, "p": p, "why": why,
                    "scout": s.get("status", "")})
        try:
            trace.record("forecast", statement=claim, p_success=p, why=why,
                         scout=s.get("status", ""))
        except Exception:
            pass
    return sorted(out, key=lambda r: -r["p"])


def allocate(fc: list, total_seconds: float) -> dict:
    """Split a time budget across claims by forecast probability.

    Proportional to p, NOT winner-take-all, and every claim keeps a floor. Two
    measured reasons for that shape rather than a greedy argmax:

      * The per-attempt noise floor on this machine is 40% -- a p ordering built
        on one sample each is not reliable enough to justify starving anything.
      * The single highest-value proof of the last run (`vp_lcm_div_gcd_nat`)
        came from a line that had already failed twice. A greedy allocator kills
        exactly that.
    """
    if not fc or total_seconds <= 0:
        return {}
    floor = 0.10 / max(len(fc), 1)
    ws = [max(float(r.get("p") or 0.0), 1e-3) for r in fc]
    tot = sum(ws)
    return {r["claim"]: round(total_seconds * max(w / tot, floor), 1)
            for r, w in zip(fc, ws)}


def report(fc: list, alloc: dict = None) -> str:
    """Prompt block. Ranking is presented; the raw numbers are deliberately soft."""
    if not fc:
        return ""
    lines = ["", "YOUR OWN FORECASTS on the open obligations (you produced these "
                 "before any expensive attempt; they are scored against Lean "
                 "afterwards):"]
    for r in fc:
        line = f"  p={r['p']:.2f}  {r['claim'][:104]}"
        if alloc and r["claim"] in alloc:
            line += f"   [budget {alloc[r['claim']] / 60:.0f} min]"
        lines.append(line)
        if r.get("why"):
            lines.append(f"      {r['why'][:120]}")
    lines.append("  Use the ORDER, not the numbers. Work the top item first "
                 "unless you have a concrete reason not to.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring. Run over the trace whenever you want to know whether any of the above
# was worth doing.
# ---------------------------------------------------------------------------

def _auc(pairs) -> float:
    """P(a random success outranks a random failure). Ties count half."""
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def score(rows: list = None) -> dict:
    """Brier and AUC for every forecast that was later adjudicated by Lean.

    Matching is by statement text: a forecast is only scored when a `proof`
    record for the same statement exists, so unresolved forecasts are dropped
    rather than counted as failures.
    """
    rows = trace.load() if rows is None else rows
    outcome = {}
    for r in rows:
        if r.get("kind") == "proof":
            st = str(r.get("statement", ""))[:120]
            # A statement proved at any point counts as provable.
            outcome[st] = outcome.get(st, False) or bool(r.get("ok"))
    pairs = []
    for r in rows:
        if r.get("kind") != "forecast" or r.get("p_success") is None:
            continue
        st = str(r.get("statement", ""))[:120]
        if st in outcome:
            pairs.append((float(r["p_success"]), bool(outcome[st])))
    if len(pairs) < 4:
        return {"n": len(pairs), "note": "not enough adjudicated forecasts to score"}
    base = sum(y for _, y in pairs) / len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    bbase = sum((base - y) ** 2 for _, y in pairs) / len(pairs)
    return {"n": len(pairs), "brier": round(brier, 4),
            "brier_base_rate": round(bbase, 4),
            "skill": round((bbase - brier) / max(bbase, 1e-9), 4),
            "auc": round(_auc(pairs), 3),
            "mean_p": round(sum(p for p, _ in pairs) / len(pairs), 3),
            "base_rate": round(base, 3)}
