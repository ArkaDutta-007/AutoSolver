"""Quality-diversity archive: MAP-Elites over attempted claims.

THE FAILURE THIS EXISTS TO MAKE IMPOSSIBLE
------------------------------------------
Measured over one real run: 26 prove calls, 16,968 seconds, 4 "successes" --
all four the SAME trivial divisibility fact. The mechanism was not stupidity,
it was a greedy policy meeting a deceptive landscape:

    hard claim fails expensively
      -> agent retreats to the one claim that formalises cleanly
      -> it verifies
      -> written to `established`, which reads as progress
      -> repeat

After five sessions `established` held five copies of one fact while the actual
invariant sat in `exhausted`, falsely ruled out. The lemma library independently
reproduced the same pathology: eight entries, two distinct facts.

Prompt-level discouragement does not fix this, because the incentive is real --
the easy claim genuinely does verify. Quality-diversity fixes it structurally.
QD algorithms keep an ARCHIVE of solutions that are high-performing *and*
behaviourally diverse, and the diversity is not decoration: entries in one
region are stepping stones to regions that direct search cannot reach. Novelty
search was invented precisely to stop convergence to local optima on deceptive
fitness landscapes, which is an exact description of proof search where a
trivial restatement is always available.

Here the "behaviour" of a claim is WHAT KIND of result it is -- mechanism,
objects, and role relative to the goal -- not whether it is true. A fifth copy
of the divisibility fact lands in an already-occupied cell and is refused by
construction, not by exhortation.
"""

import json
import os
import re
import threading
import time

from . import config, library

PATH = os.path.join(config.ROOT, "kb", "archive.json")

# Role: how far the claim sits from the goal. A restatement of the goal and a
# genuinely new auxiliary construction are different KINDS of work even when
# they share mechanism and objects -- and the failure mode is precisely an
# endless supply of restatements.
ROLES = ("restatement", "sublemma", "generalization", "construction", "unknown")

_lock = threading.RLock()
_STOP = {"the", "a", "an", "of", "for", "all", "and", "is", "are", "be", "to",
         "in", "on", "that", "this", "with", "any", "every", "some", "if",
         "then", "let", "show", "prove", "we", "it", "its", "there", "exists",
         "such", "given", "where", "which", "has", "have", "each", "at", "by"}


# Claims arrive as ENGLISH, proofs as Lean. library.describe reads Lean tokens
# (`ℕ`, `∣`), so "For every natural number n, 2 divides n*(n+1)" scored
# `unclassified` while "For all n : ℕ, 2 ∣ n*(n+1)" scored nat/divisibility --
# and the two landed in DIFFERENT cells despite being the same fact. That is
# the exact failure this archive exists to catch, so prose has to map onto the
# same vocabulary as code.
_PROSE_OBJECTS = {
    "nat": r"\bnatural|\bnonnegative integer|\bpositive integer|\bwhole number",
    "int": r"\binteger(?!s? greater)|\bnegative\b",
    "real": r"\breal number|\breals?\b|\bcontinuous|\bsqrt|\bsquare root",
    "rat": r"\brational",
    "complex": r"\bcomplex number",
    "matrix": r"\bmatri(x|ces)|\bdeterminant|\btranspose",
    "finset": r"\bfinite set|\bcardinalit|\bcombinat|\bpigeonhole",
    "set": r"\bsubset|\bset of\b|\bunion|\bintersection",
    "poly": r"\bpolynomial",
    "prime": r"\bprime",
    "divisibility": (r"\bdivid|\bdivisib|\bmultiple of|\bgcd|\blcm|\bcoprime|\bfactor"
                     r"|\beven\b|\bodd\b|\bparity\b|\bmodul[oa]\b"),
    "function": r"\bfunction|\bmapping\b|\bderivative|\bintegral",
}


# Mechanism read from the CLAIM, since the key cannot use the proof. Prose
# rarely names its own mechanism, so this axis is often "unclassified" and the
# real discriminators become object x role -- which is honest. Better a coarse
# key that fires than a precise one that never matches.
# EXPLICIT mechanism words only. An earlier version inferred induction from
# `for all n ... n+1`, which is surface SYNTAX, not mechanism -- it fired on
# `For all n : ℕ, 2 ∣ n * (n + 1)` and split the five divisibility
# restatements back into two cells, defeating the one guarantee this module
# owes. When prose does not name its mechanism, "unclassified" is the correct
# answer; guessing is worse than abstaining, because a wrong guess splits a
# cell that should collide.
_PROSE_MECHANISM = {
    "induction": r"\binduct|\brecursi",
    "invariant": r"\binvariant|\bpreserv|\bunchanged|\bconserv",
    "extremal": r"\bextremal|\bsmallest\b|\blargest\b|\bminimi[sz]|\bmaximi[sz]",
    "counting": r"\bcounting\b|\bcardinal|\bpigeonhole|\bhow many",
    "contradiction": r"\bcontradict|\bsuppose not\b",
    "termination": r"\btermin|\bfinitely many\b|\bhalts\b",
}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in _STOP and len(w) > 2}


def _content(text: str) -> set:
    """Content words PLUS the concept tags the text implies.

    A fully symbolic claim -- `For all n : ℕ, 2 ∣ n * (n + 1)` -- has no prose
    content words at all: every token is a stopword or a single letter. Role
    inference then returned "unknown" and the claim landed in a different cell
    from its own English paraphrase, which is the split this archive must not
    have. Concept tags give symbolic and prose statements a shared vocabulary.
    """
    low = (text or "").lower()
    tags = {k for k, pat in _PROSE_OBJECTS.items() if re.search(pat, low)}
    tags |= set(library.describe(text or "")["objects"])
    return _words(text) | tags


def infer_role(claim: str, goal: str = "") -> str:
    """Where this claim sits relative to the problem being solved."""
    if not goal:
        return "unknown"
    c, g = _content(claim), _content(goal)
    if not c or not g:
        return "unknown"
    shared = c & g
    # Containment over the SMALLER set, not Jaccard. A claim restating the goal
    # in symbols ("gcd(m,n) * lcm(m,n) = m * n") shares every content word it
    # has with a goal that merely uses more words to say it -- Jaccard scored
    # that 0.5 and called it a sublemma.
    contain = len(shared) / max(min(len(c), len(g)), 1)
    if contain >= 0.8 and len(shared) >= 2:
        return "restatement"
    if c > g and len(shared) >= 2:
        return "generalization"
    if len(shared) >= 2:
        return "sublemma"
    return "construction"


def cell(claim: str, code: str = "", goal: str = "", role: str = "") -> tuple:
    """The archive cell a claim belongs to: (mechanism, object, role).

    Descriptors are shared with the lemma library on purpose. One vocabulary
    for "what kind of result is this" keeps the archive and the library talking
    about the same thing, so a lemma already stored is recognisably the same
    KIND as the claim about to be attempted.
    """
    # THE KEY DERIVES FROM THE CLAIM ONLY -- never from the proof.
    #
    # occupied() is consulted BEFORE a proof exists; _record() runs AFTER, with
    # the Lean source in hand. When the proof contributed to the key, the two
    # computed different cells for the same claim:
    #
    #   before proving:  ('unclassified', 'divisibility', 'construction')
    #   after proving:   ('arithmetic',    'divisibility', 'construction')
    #
    # so a recorded cell could never block a later check and the guard was
    # inert in production -- while its unit test passed, because that test
    # called record() with no code and made both sides accidentally agree.
    # Proof-derived descriptors are still stored, as METADATA, never in the key.
    low = (claim or "").lower()
    mech = next((k for k, pat in _PROSE_MECHANISM.items() if re.search(pat, low)),
                "unclassified")
    objs = {k for k, pat in _PROSE_OBJECTS.items() if re.search(pat, low)}
    objs |= set(library.describe(claim or "")["objects"])
    obj = sorted(objs)[0] if objs else "unclassified"
    return (mech, obj, role or infer_role(claim, goal))


def _load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save(arc: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(arc, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, PATH)


def _key(c: tuple) -> str:
    return "|".join(c)


def fitness(ok: bool, seconds: float, code: str = "") -> float:
    """Quality WITHIN a cell. Diversity is handled by the cells themselves.

    A verified claim beats an unverified one; among verified ones, prefer the
    result that was HARD to get -- a claim that took real search is worth more
    as an elite than one `decide` closed instantly. This is deliberately not a
    pure speed score: optimising for cheap wins is how the greedy policy got
    here in the first place.
    """
    if not ok:
        return 0.0
    return 1.0 + min(seconds / 600.0, 1.0) + min(len(code or "") / 4000.0, 0.5)


def record(claim: str, ok: bool, seconds: float = 0.0, code: str = "",
           goal: str = "", role: str = "") -> dict:
    """File an attempt. Returns {"cell", "accepted", "elite"}.

    `accepted` is True when this became the cell's elite -- i.e. it is the best
    result of its KIND so far. A duplicate fact in a filled cell is recorded
    (we want the attempt count) but does not displace the incumbent.
    """
    c = cell(claim, code, goal, role)
    k, f = _key(c), fitness(ok, seconds, code)
    with _lock:
        arc = _load()
        slot = arc.get(k)
        accepted = slot is None or f > slot.get("fitness", 0.0)
        if accepted:
            arc[k] = {"cell": list(c), "claim": claim[:300], "fitness": round(f, 3),
                      "ok": bool(ok), "seconds": round(seconds, 1),
                      "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "attempts": (slot or {}).get("attempts", 0) + 1}
        else:
            slot["attempts"] = slot.get("attempts", 0) + 1
            arc[k] = slot
        _save(arc)
    return {"cell": c, "accepted": accepted, "elite": arc[k].get("claim", "")[:120]}


def occupied(claim: str, code: str = "", goal: str = "", role: str = "") -> dict:
    """Is this KIND of result already in the archive, verified?

    The guard the greedy policy needed. Returns the incumbent when the cell is
    filled by a VERIFIED entry, so the caller can refuse to spend a full proof
    budget re-deriving something of the same kind.
    """
    k = _key(cell(claim, code, goal, role))
    slot = _load().get(k)
    if slot and slot.get("ok"):
        return {"filled": True, "by": slot.get("claim", ""), "cell": slot.get("cell")}
    return {"filled": False}


def coverage() -> dict:
    """How much of the behaviour space has been touched.

    The direct anti-local-minimum metric: a run that raises its verified-cell
    count is exploring, a run that raises only its attempt count is not.
    """
    arc = _load()
    filled = [v for v in arc.values() if v.get("ok")]
    mechs = {v["cell"][0] for v in arc.values()}
    objs = {v["cell"][1] for v in arc.values()}
    roles = {v["cell"][2] for v in arc.values()}
    return {"cells": len(arc), "verified_cells": len(filled),
            "attempts": sum(v.get("attempts", 0) for v in arc.values()),
            "mechanisms": sorted(mechs), "objects": sorted(objs),
            "roles": sorted(roles)}


def gaps(limit: int = 6) -> list:
    """Combinations we have VERIFIED results near, but not in.

    Neighbours of filled cells rather than the full cross product: the point of
    a stepping stone is that it is reachable. Proposing a wholly unrelated cell
    is noise, not exploration.
    """
    arc = _load()
    if not arc:
        return []
    mechs = sorted({v["cell"][0] for v in arc.values()} | {"induction", "arithmetic",
                                                           "algebraic", "case_analysis"})
    objs = sorted({v["cell"][1] for v in arc.values()})
    have = set(arc)
    out = []
    for m in mechs:
        for o in objs:
            for r in ("sublemma", "generalization", "construction"):
                k = _key((m, o, r))
                if k not in have:
                    out.append({"mechanism": m, "object": o, "role": r})
                if len(out) >= limit:
                    return out
    return out


def advice(goal: str = "", limit: int = 4) -> str:
    """Prompt block: what has been covered, and what kind of thing to try next."""
    cov = coverage()
    if not cov["cells"]:
        return ""
    lines = ["", f"EXPLORATION MAP -- {cov['verified_cells']} distinct KINDS of result "
                 f"verified across {cov['attempts']} attempts."]
    arc = _load()
    done = [v for v in arc.values() if v.get("ok")][:limit]
    if done:
        lines.append("Already established (do NOT re-derive these kinds):")
        for v in done:
            lines.append(f"  [{'/'.join(v['cell'])}] {v['claim'][:90]}")
    g = gaps(limit)
    if g:
        lines.append("UNEXPLORED kinds -- prefer a claim that lands in one of these:")
        for x in g:
            lines.append(f"  {x['mechanism']} argument about {x['object']} "
                         f"as a {x['role']}")
    return "\n".join(lines)


def reset() -> None:
    with _lock:
        _save({})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="quality-diversity archive")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--advice", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        reset(); print("archive cleared")
    elif a.gaps:
        print(json.dumps(gaps(12), indent=1))
    elif a.advice:
        print(advice() or "(archive empty)")
    else:
        print(json.dumps(coverage(), indent=1))
