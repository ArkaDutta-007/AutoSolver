"""Probing for hunches: the ideas a model holds weakly, not the ones it recites.

THE PROBLEM WITH SAMPLING WHAT IS "MOST PROBABLE"
------------------------------------------------
Ordinary decoding returns the model's consensus. For a research system that is
the least useful part of its output: the consensus about gcd and lcm is already
in Mathlib, already retrievable, and already tried. What a hard problem needs is
the weakly-held association -- the "this smells like a valuation argument"
that never surfaces because it is never the argmax.

The literature calls this an ELICITATION OVERHANG: models contain capabilities
we do not know how to elicit, and naively prompting does not maximise the
probability of the useful response. Chain-of-thought paths demonstrably exist
inside top-k ALTERNATIVE branches and can be recovered by changing how you
decode rather than what you ask.

DISAGREEMENT AS THE SIGNAL
--------------------------
This module does not try to read hidden states -- ollama exposes none, and the
one server that could (vLLM) is busy holding the prover. It uses the signal
that survives a black-box API: SEMANTIC SCATTER across independent samples.

  * samples CONVERGE  -> the model is reciting. Reliable, and probably already
                         known to Mathlib. Low information.
  * samples SCATTER   -> weak latent structure. Unreliable, and exactly where a
                         hunch lives. High information, needs verification.

This is the black-box cousin of semantic-entropy probing, which finds that
uncertainty is encoded in hidden states before a single token is emitted. We
cannot see the states, but scatter across samples is a usable proxy -- and
crucially it costs only sampling, which this machine can already do.

The output is NOT a claim about truth. A scattered idea is a QUESTION worth
asking, and every one of them still has to go through Lean.
"""

import json
import os
import re

import numpy as np
import requests

from . import config

# Deliberately far apart. Analogical prompting is measured to raise solution
# diversity by 90-173% and to produce novel solutions over half the time, but
# it is fragile in FAR-analogy settings -- so the domains are named explicitly
# rather than left to the model, which otherwise stays inside the problem's
# apparent home field.
LENSES = [
    "number theory / valuations / divisibility",
    "group or ring structure / symmetry / invariants under an action",
    "combinatorics / counting / extremal arguments / pigeonhole",
    "order theory / monovariants / well-founded descent / termination",
    "geometry or topology / continuity / compactness",
    "probability / averaging / expectation arguments",
    "linear algebra / dimension / rank / eigenstructure",
    "information or entropy / potential functions",
]

_SCHEMA = {"type": "object",
           "properties": {"idea": {"type": "string"}, "why": {"type": "string"}},
           "required": ["idea"]}


def _ask(model: str, prompt: str, temp: float, timeout: int = 240) -> dict:
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "format": _SCHEMA,
                  "options": {**config.ctx(model), "num_predict": 900, "temperature": temp},
                  # MEASURED, three times now: reasoning and a grammar-constrained
                  # schema compete for ONE budget, and the JSON is what loses --
                  # it truncates mid-object and json.loads throws, so the whole
                  # call yields nothing. plan.py died this way twice; here it
                  # burned ~8 of every 9 minutes of a live run while producing
                  # zero candidates, silently, because an empty result prints
                  # nothing.
                  #
                  # These calls are STRUCTURAL -- pick a tool, state a claim --
                  # not deliberative. Thinking off puts the whole budget into
                  # the answer.
                  "think": False},
            timeout=timeout)
        r.raise_for_status()
        out = json.loads(r.json()["message"]["content"])
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _embed(texts: list) -> np.ndarray:
    from . import mathlib_kb
    try:
        vs = mathlib_kb._embed_batch(texts, timeout=120)
        a = np.asarray(vs, dtype=np.float32)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    except Exception:
        return np.zeros((len(texts), 1), dtype=np.float32)


def probe(goal: str, model: str = "", lenses: list = None,
          temp: float = 0.9) -> list:
    """One idea per LENS, then rank by how far each sits from the consensus.

    Forcing a named foreign domain is what makes this more than resampling: a
    model asked for "another idea" paraphrases, while a model asked for "the
    order-theoretic reading" has to reach somewhere else in its weights.
    """
    model = model or config.ORCH_PROPOSER_MODEL or config.ORCHESTRATOR_MODEL
    picked = lenses or LENSES
    out = []
    for lens in picked:
        got = _ask(model, (
            f"PROBLEM:\n{goal[:1500]}\n\n"
            f"Look at this problem STRICTLY through the lens of {lens}.\n"
            "Even if that lens seems like the wrong home for this problem, "
            "commit to it and say what it suggests. A vague but specific hunch "
            "is worth more here than a safe restatement.\n\n"
            'Answer as JSON: "idea" = the concrete mathematical suggestion '
            '(one or two sentences, name actual objects), '
            '"why" = what makes the analogy hold.'), temp)
        idea = str(got.get("idea", "")).strip()
        if len(idea) > 20:
            out.append({"lens": lens, "idea": idea,
                        "why": str(got.get("why", ""))[:240]})
    if len(out) < 2:
        return out

    # Scatter = distance from the centroid of all ideas. The centroid IS the
    # consensus; the outlier is the one the model would not have said if asked
    # plainly, which is the whole point of asking this way.
    vecs = _embed([o["idea"] for o in out])
    if vecs.shape[1] > 1:
        centroid = vecs.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        for o, v in zip(out, vecs):
            o["scatter"] = round(float(1.0 - v @ centroid), 4)
    else:
        for o in out:
            o["scatter"] = 0.0
    return sorted(out, key=lambda o: -o["scatter"])


def report(ideas: list, limit: int = 4) -> str:
    """Prompt block. Outliers FIRST, and labelled as questions, not answers."""
    if not ideas:
        return ""
    lines = ["", "CROSS-DOMAIN HUNCHES (each forced through a different lens, "
                 "ranked by distance from the consensus reading). These are "
                 "UNVERIFIED questions worth asking, not facts:"]
    for o in ideas[:limit]:
        lines.append(f"  [scatter {o.get('scatter', 0):.3f}] via {o['lens']}")
        lines.append(f"      {o['idea'][:220]}")
    lines.append("  The top entry is the one LEAST like the obvious reading. "
                 "If the obvious reading has already failed, start there.")
    return "\n".join(lines)


def consensus_split(ideas: list) -> dict:
    """Which ideas are recital, which are hunches. Reported, not decided.

    The threshold is a convenience for logging. Whether a low-consensus idea is
    insight or noise is not knowable from scatter alone -- only Lean settles
    that, which is exactly why this returns candidates rather than beliefs.
    """
    if not ideas:
        return {"recital": [], "hunches": []}
    scat = [o.get("scatter", 0.0) for o in ideas]
    mid = float(np.median(scat)) if scat else 0.0
    return {"recital": [o["lens"] for o in ideas if o.get("scatter", 0) <= mid],
            "hunches": [o["lens"] for o in ideas if o.get("scatter", 0) > mid],
            "median_scatter": round(mid, 4)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="cross-domain hunch probe")
    ap.add_argument("file", help="problem statement file")
    ap.add_argument("--lenses", type=int, default=0, help="cap number of lenses")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read()
    got = probe(text, lenses=LENSES[:a.lenses] if a.lenses else None)
    print(report(got, limit=len(got)))
    print("\n" + json.dumps(consensus_split(got), indent=1))
