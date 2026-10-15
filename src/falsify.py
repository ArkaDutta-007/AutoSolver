"""Try to numerically refute a claim before anything expensive touches it.

THE MEASURED HOLE THIS FILLS
----------------------------
The decomposer emits confident false lemmas. On IMO26 P1 it produced "the
product of all integers on the blackboard is invariant under the move" -- false
at the very first case (m=n=2: product 4 becomes gcd*(lcm/gcd) = 2*1 = 2) --
and "M equals the product of the initial integers divided by the square of the
gcd", also false. The first consumed 41 minutes of derivation and, before the
faithfulness gate existed, nearly banked a verified proof of a wrong statement.

A falsifiable arithmetic claim can be executed. Observe -> conjecture -> prove
is how the mathematics is actually done, and this is the missing OBSERVE step
run backwards: before formalizing, spend seconds looking for a counterexample.
It would have killed both P1 falsehoods for the cost of one sandbox run each.

WHAT IS DETERMINISTIC HERE AND WHAT IS NOT
------------------------------------------
The SEARCH SCRIPT is model-written (a claim in prose cannot be executed without
translation, and that translation can be wrong). Everything after that line is
deterministic: the sandbox is seeded, time-boxed and offline, the verdict is
parsed from a fixed contract, and the counterexample VALUES travel with the
verdict so anyone -- the agent, a human, a later audit -- can recheck them
against the claim by hand.

The asymmetry is the safety property:

  * counterexample FOUND  -> the claim (as the script encoded it) fails at
                             concrete values. Strong, checkable, actionable.
  * nothing found         -> proves NOTHING. Status "unfalsified" changes no
                             behaviour; small-case search is not induction.
  * script broken/timeout -> "unavailable". Also changes nothing.

So a wrong translation can at worst KILL a true claim -- and the printed
counterexample makes that visible immediately -- while a claim is never
BLESSED by this module at all. Verification stays Lean's job alone.
"""

import json
import re

import requests

from . import config, sandbox, trace

FALSIFIED, UNFALSIFIED, UNAVAILABLE = "falsified", "unfalsified", "unavailable"

_SCHEMA = {"type": "object",
           "properties": {"code": {"type": "string"}},
           "required": ["code"]}

_PROMPT = (
    "Write a SHORT Python script that searches small cases for a counterexample "
    "to this mathematical claim.\n\n"
    "CLAIM: {claim}\n\n"
    "Rules:\n"
    "- Search small ranges only (a few thousand cases at most; values up to ~50).\n"
    "- If a counterexample is found, print exactly one line:\n"
    "    COUNTEREXAMPLE: <the variable values and the two unequal sides>\n"
    "  then stop.\n"
    "- If the search finishes with no counterexample, print exactly: NONE\n"
    "- Pure computation. No imports beyond math/itertools/fractions (already "
    "imported: gcd, lcm, Fraction, itertools). No files, no network, no input().\n"
    "- If the claim is not about computable objects (it quantifies over all "
    "reals, involves limits, infinite sets, or is not mathematical), print "
    "exactly: NOT_TESTABLE\n"
    "Return JSON: \"code\" = the script."
)

_HIT = re.compile(r"^COUNTEREXAMPLE:\s*(.+)$", re.M)


def _ask(claim: str, timeout: int = 180) -> str:
    """One script from the model. think=False: schema call, measured six times."""
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.ORCHESTRATOR_MODEL,
                  "messages": [{"role": "user",
                                "content": _PROMPT.format(claim=str(claim)[:1200])}],
                  "stream": False, "format": _SCHEMA, "think": False,
                  "options": {**config.ctx(config.ORCHESTRATOR_MODEL), "num_predict": 1200, "temperature": 0.2}},
            timeout=timeout)
        r.raise_for_status()
        return str(json.loads(r.json()["message"]["content"]).get("code", ""))
    except Exception:
        return ""


_SEEN: dict = {}


def probe(claim: str, timeout: int = 45) -> dict:
    """One falsification attempt. Never raises; only FALSIFIED changes behaviour.

    Memoised per claim: a claim is typically scouted and then proved, and the
    verdict is deterministic given the script, so paying a second model call
    plus a second sandbox run for the identical answer is pure waste on a
    machine with one model slot.
    """
    key = " ".join(str(claim).split())[:300].lower()
    if key in _SEEN:
        return dict(_SEEN[key], cached=True)
    out = {"claim": claim, "status": UNAVAILABLE, "counterexample": "", "code": ""}
    code = _ask(claim)
    if not code.strip():
        return _record(out)
    out["code"] = code
    try:
        run = sandbox.run_python(code, timeout=timeout)
    except Exception:
        return _record(out)
    text = str(run.get("output") or run.get("stdout") or "")
    if not run.get("ok") and not text:
        return _record(out)
    m = _HIT.search(text)
    if m:
        out["status"] = FALSIFIED
        out["counterexample"] = m.group(1).strip()[:300]
    elif "NOT_TESTABLE" in text:
        out["status"] = UNAVAILABLE
    elif "NONE" in text:
        # Survived a small-case search. Weak evidence, worth REPORTING as
        # context, never worth acting on -- small cases are not induction.
        out["status"] = UNFALSIFIED
    return _record(out)


def _record(out: dict) -> dict:
    try:
        _SEEN[" ".join(str(out.get("claim", "")).split())[:300].lower()] = dict(out)
    except Exception:
        pass
    try:
        trace.record("falsify", statement=out["claim"],
                     ok=out["status"] != FALSIFIED,
                     status=out["status"],
                     counterexample=out["counterexample"] or None,
                     code=out["code"][:1500] or None)
    except Exception:
        pass
    return out


def report(results: list) -> str:
    """Prompt block. Counterexamples carry their values so they can be rechecked."""
    hits = [r for r in results if r.get("status") == FALSIFIED]
    if not hits:
        return ""
    lines = ["", "NUMERICALLY FALSIFIED (a concrete counterexample was computed; "
                 "recheck the values against the claim -- if they hold, the claim "
                 "is FALSE and must be dropped or restated, never proved):"]
    for r in hits:
        lines.append(f"  [FALSE] {str(r['claim'])[:120]}")
        lines.append(f"      counterexample: {r['counterexample'][:200]}")
    return "\n".join(lines)
