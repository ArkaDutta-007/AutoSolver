"""Back-translation gate.

The compiler proves the PROOF matches the STATEMENT. Nothing else checks that the
STATEMENT matches the CLAIM. That gap is how a trivial restatement scores as a
verified result -- observed on the very first smoke test, where "infinitely many
primes" was formalized as a statement satisfied by the constant function 2.

This is a mitigation, not a guarantee: a model is judging natural-language
faithfulness. Spot-check statements on anything that matters.
"""

import json
import re

import requests

from . import config

_STATEMENT = re.compile(r"^\s*(?:theorem|lemma|example)\b(.*?)(?::=|\bby\b)",
                        re.DOTALL | re.MULTILINE)

# Only a line-leading label counts as the verdict.
_VERDICT = re.compile(r"^\s*\**(FAITHFUL|MISMATCH)\**\s*:", re.MULTILINE | re.IGNORECASE)

PROMPT = """A prover was asked to formally state and prove this claim:

CLAIM: {informal}

It produced this Lean 4 theorem statement:

{statement}

Ignore the proof; judge ONLY whether the statement faithfully expresses the claim.

MISMATCH means the statement fails to capture the claim -- typically because it is
technically true but trivial, or quantifies over the wrong thing.

A STRICTLY STRONGER statement is FAITHFUL, not a mismatch. If the statement proves
MORE than the claim asks and the claim follows from it, that is a correct and often
preferable formalization. Examples that are FAITHFUL:
  - claim says "positive integers", statement quantifies over all naturals
    (including 0) and still holds -- stronger, therefore fine
  - claim asks for one case, statement proves the general case
Do not flag a statement merely because its domain is wider than the claim's, or
because you suspect an edge case; the compiler has already verified the statement
is true as written.

Reply with exactly one line:
FAITHFUL: <short reason>
or
MISMATCH: <what the statement actually says instead>"""


SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["FAITHFUL", "MISMATCH"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def review_statement(informal: str, code: str) -> dict:
    """Judge a Lean statement against the informal claim.

    Uses CONSTRAINED DECODING: the verdict is an enum field the model cannot
    phrase ambiguously. Parsing free text here was an active bug -- the model
    writes "no domain mismatch" inside FAITHFUL verdicts, and a substring test
    read that as a rejection, silently false-rejecting valid formalizations.
    The grammar makes that class of error impossible rather than merely unlikely.
    """
    m = _STATEMENT.search(code)
    stmt = (m.group(0) if m else code)[:1200]
    prompt = PROMPT.format(informal=informal, statement=stmt)

    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.REVIEWER_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "format": SCHEMA,
                  "options": {**config.ctx(config.REVIEWER_MODEL), "num_predict": 4000, "temperature": 0.2},
                  # An ABSENT think key is not a safe default -- it hands the
                  # decision to the model, and Qwen3.8 thinks by default. The
                  # reasoning then eats the 4000-token allowance, the schema's
                  # JSON truncates, json.loads throws, and this falls through to
                  # _review_freetext, which scrapes prose for a label.
                  #
                  # MEASURED on one hand-checked faithful signature, three draws
                  # of identical input: MISMATCH, FAITHFUL, FAITHFUL. The
                  # dissenting draw was not a judgement at all -- it read
                  # "...It's fine. Could be mismatch because it uses
                  # factorization p, which return" and was cut off mid-sentence.
                  # A verdict was extracted from unfinished reasoning.
                  #
                  # This gate decides whether a proof counts as real, so a
                  # coin-flip here is worse than anywhere else in the system.
                  "think": False},
            timeout=900)
        r.raise_for_status()
        content = (r.json().get("message", {}).get("content") or "").strip()
        data = json.loads(content)
        verdict = str(data.get("verdict", "")).upper()
        if verdict in ("FAITHFUL", "MISMATCH"):
            return {"faithful": verdict == "FAITHFUL",
                    "verdict": f"{verdict}: {str(data.get('reason',''))[:400]}",
                    "statement": stmt}
    except Exception:
        pass  # fall through to the text path

    return _review_freetext(prompt, stmt)


def _review_freetext(prompt: str, stmt: str) -> dict:
    """Fallback when constrained decoding is unavailable (e.g. a remote server)."""
    try:
        # native for the same reason as prover._sketch: /v1 cannot ask for num_ctx
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.REVIEWER_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {**config.ctx(config.REVIEWER_MODEL),
                              "num_predict": 4000, "temperature": 0.2}},
            timeout=900)
        r.raise_for_status()
        msg = r.json().get("message") or {}
        verdict = ((msg.get("content") or "").strip() or (msg.get("thinking") or "").strip())
    except Exception as exc:
        return {"faithful": None, "note": f"review unavailable: {exc}"}

    labels = _VERDICT.findall(verdict)          # line-leading label only
    if labels:
        faithful = labels[-1].upper() == "FAITHFUL"
    else:
        up = verdict.upper()
        faithful = up.rfind("FAITHFUL") >= up.rfind("MISMATCH")
    return {"faithful": faithful, "verdict": verdict[-400:], "statement": stmt}
