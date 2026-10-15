"""Prior evidence for the claim in hand, pushed in unasked.

WHY RETRIEVAL AS A TOOL DOES NOT WORK HERE
------------------------------------------
The obvious design is a `search_past_attempts` tool. It fails for a reason this
repo has already measured twice:

  * The agent has `search_mathlib` and did not call it -- it wrote a signature
    around invented `have_gcd`/`val` for a claim Mathlib states outright as
    `Nat.factorization_gcd`. The tool existed, was described, and went unused.
  * Nothing ever called `write_solution` either, across five problems and ten
    hours, until the plan ledger stopped asking and started withholding tools.

A pull-based store requires the model to know three things at once: that the
store exists, that this particular claim is the kind the store can help with,
and that it is worth a turn to ask. That conjunction is exactly what a model
deep in a proof does not have. So this module never waits to be asked. It is
called by the anchor builder on every turn, keyed to whatever the agent is
actually working on, and its output arrives as part of the durable state.

WHAT GETS RETRIEVED, AND WHY NEGATIVES MATTER MOST
--------------------------------------------------
Successes are the obvious payload, and the least useful half. The measured
failure mode of this system is REPETITION: eight `exhausted` entries turned out
to be two claims reworded four times each, and the agent retreated to already
solved ground because each restatement looked like progress. What prevents that
is not knowing what worked -- it is knowing what was already paid for and did
not work, with the price attached.

So a failure is retrieved with its cost ("900s, 4 attempts, no tactic worked"),
because "this costs fifteen minutes and does not work" is actionable in a way
that "this is unproved" is not.

BUDGETED, BECAUSE CONTEXT IS THE SCARCE RESOURCE
------------------------------------------------
Everything here competes with the problem statement for a 32,768-token window
that has already been measured producing empty steps 35% of the time when
over-subscribed. Hence a similarity FLOOR rather than a fixed top-k: when
nothing is relevant, this returns the empty string and costs nothing. A
retriever that always fills its quota is a retriever that injects noise on the
turns when it has nothing to say.
"""

import re

from . import config, trace

# Set from the three measured similarity profiles in _relevant, not by feel.
# A miss set spanned 0.075; the narrowest genuine hit set spanned 0.125.
_MIN_SPREAD = 0.10
# Where in the observed range the bar sits. 0.6 keeps the genuine geometry match
# at 0.541 while rejecting the next item down at 0.484.
_UPPER_FRACTION = 0.6

_STOP = {"the", "and", "for", "all", "any", "that", "with", "from", "this",
         "then", "there", "have", "has", "let", "such", "which", "are", "not",
         "prove", "show", "theorem", "lemma", "every", "some", "when", "where"}


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower())
            if len(w) >= 3 and w not in _STOP}


def _jaccard(a: str, b: str) -> float:
    """Lexical fallback. Cheap, no network, and never returns nothing."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _similarity(claim: str, texts: list) -> list:
    """Cosine over embeddings when the embedder answers, Jaccard when it does not.

    The fallback is not a nicety: the embedder shares one Ollama slot with the
    orchestrator, so under load it is exactly as likely to time out as anything
    else. Degrading to lexical overlap keeps recall working at reduced quality
    instead of silently returning nothing on the busiest turns.
    """
    try:
        import numpy as np
        from . import mathlib_kb
        vs = mathlib_kb._embed_batch([claim] + texts, timeout=45)
        a = np.asarray(vs, dtype=np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        return [float(x) for x in (a[1:] @ a[0])]
    except Exception:
        return [_jaccard(claim, t) for t in texts]


def _relevant(sims: list, floor: float) -> list:
    """Which similarities represent a real match. Returns a boolean mask.

    MEASURED, because an absolute cosine threshold turned out to be meaningless
    on its own. Over the seven-lemma library:

        query "gcd(m,n) divides m"      -> 0.868  0.843  0.749 ... 0.558
        query "process terminates ..."  -> 0.538  0.532  0.521 ... 0.463
        query "angles sum to 180"       -> 0.541  0.484  0.471 ... 0.416

    A real match sits ~0.15 ABOVE the pack. When nothing matches, everything
    piles into a 0.07-wide band and the ordering inside it is noise -- but the
    absolute values there (0.46-0.54) overlap the genuine geometry match at
    0.541, so no single cutoff separates the three cases. A floor of 0.34
    admitted all of them, which is how a lemma about triangle angles ended up
    being recommended for a claim about a process terminating.

    What separates them is the SPREAD, not the level. A hit set is stretched
    (0.868 down to 0.558, a range of 0.31); a miss set is squeezed (0.075). So:

      1. If the whole set is squeezed, nothing is differentiated -- reject all.
      2. Otherwise keep what sits in the upper part of the observed range.

    Both steps are scale-free, so this survives a change of embedding model in
    a way a hand-tuned cutoff would not. The absolute floor remains as a floor.

    Using the median as the baseline was tried first and is WRONG: when two of
    four items are genuine matches the median sits on top of one of them, and
    the test rejects the best hit in the set. Caught by
    test_a_clear_match_survives, not by inspection.
    """
    if not sims:
        return []
    lo, hi = min(sims), max(sims)
    if hi - lo < _MIN_SPREAD:
        return [False] * len(sims)
    bar = max(floor, lo + _UPPER_FRACTION * (hi - lo))
    return [s >= bar for s in sims]


def _cost(row: dict) -> str:
    secs = float(row.get("elapsed") or 0.0)
    n = int(row.get("n_attempts") or 0)
    bits = []
    if secs >= 1:
        bits.append(f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f}s")
    if n:
        bits.append(f"{n} attempt{'s' if n != 1 else ''}")
    return ", ".join(bits)


def _reason(row: dict) -> str:
    """The most informative line from the attempt history, if there is one."""
    for h in (row.get("history") or []):
        r = str(h.get("reason") or "").strip()
        if r and not h.get("ok"):
            return r[:110]
    return str(row.get("note") or "")[:110]


def evidence(claim: str, k: int = 0, floor: float = -1.0,
             rows: list = None) -> list:
    """Past attempts on claims like this one, most similar first.

    Deduplicated by statement: a claim attempted four times should occupy one
    line carrying the worst news, not four lines crowding out everything else.
    """
    k = k or config.GT_RECALL_K
    floor = config.GT_RECALL_FLOOR if floor < 0 else floor
    rows = trace.load() if rows is None else rows
    seen, cand = {}, []
    for r in rows:
        st = str(r.get("statement") or "").strip()
        if not st or _jaccard(claim, st) >= 0.98:
            continue                       # the claim itself is not evidence
        prev = seen.get(st[:120])
        # Keep the FAILURE when a statement has both: the expensive negative is
        # the one that changes what the agent does next.
        if prev is None or (prev.get("ok") and not r.get("ok")):
            seen[st[:120]] = r
    cand = list(seen.values())
    if not cand:
        return []
    sims = _similarity(claim, [str(r.get("statement", "")) for r in cand])
    ok = _relevant(sims, floor)
    out = sorted((dict(r, _sim=s) for r, s, good in zip(cand, sims, ok) if good),
                 key=lambda r: -r["_sim"])

    # Collapse restatements of ONE fact. The exact-prefix dedup above cannot see
    # these -- "gcd(m,n) divides both m and n" and "for positive integers m, n,
    # gcd(m,n) divides both m and n, and lcm is a multiple of both" are distinct
    # strings and the same evidence, and both were being injected. That is the
    # very failure this module exists to prevent, reproduced inside it: eight
    # `exhausted` entries turned out to be two claims reworded four times.
    kept = []
    for r in out:
        st = str(r.get("statement", ""))
        if any(_jaccard(st, str(o.get("statement", ""))) >= 0.62 for o in kept):
            continue
        kept.append(r)
        if len(kept) >= k:
            break
    return kept


def lemmas(claim: str, k: int = 3, floor: float = -1.0) -> list:
    """Library lemmas close to this claim, with the exact name to cite.

    A lemma the agent cannot name is a lemma it cannot use, so the Lean name is
    the payload here -- the prose description is only there to make the match
    checkable by eye.

    FLOORED, and the floor is the whole point. `library.search` is a ranker: it
    returns its top k whether or not any of them is relevant, so for a claim
    about a process TERMINATING it happily returned `shanyu_defense_lemma_v2`,
    a statement about angles of a triangle. Ranking without a threshold is how
    a retriever fills its quota with noise on exactly the turns it has nothing
    to offer, and noise costs the context this system is already short of.
    """
    floor = config.GT_RECALL_FLOOR if floor < 0 else floor
    try:
        from . import library
        recs = [r for r in (library.search(claim, k=max(k * 2, 6)) or [])
                if isinstance(r, dict) and r.get("name")]
    except Exception:
        return []
    if not recs:
        return []
    sims = _similarity(claim, [f"{r.get('name', '')} : {r.get('statement', '')}"
                               for r in recs])
    ok = _relevant(sims, floor)
    keep = [dict(r, _sim=s) for r, s, good in zip(recs, sims, ok) if good]
    return sorted(keep, key=lambda r: -r["_sim"])[:k]


def report(claim: str, k: int = 0, rows: list = None) -> str:
    """The prompt block. Empty when there is nothing relevant -- deliberately.

    Labels every item with its EVIDENTIAL STATUS and its price. The agent does
    not have to decide whether the store is worth consulting, and it does not
    have to guess whether a retrieved line is a fact or a dead end: verified,
    failed, and available-to-cite are marked differently and read differently.
    """
    if not config.GT_RECALL or not (claim or "").strip():
        return ""
    ev = evidence(claim, k=k, rows=rows)
    lem = lemmas(claim)
    if not ev and not lem:
        return ""

    lines = ["", "PRIOR EVIDENCE ON THIS CLAIM (retrieved automatically -- you did "
                 "not ask for it, and it may not be relevant; judge it):"]
    for r in ev:
        st = str(r.get("statement", ""))[:130]
        if r.get("ok"):
            lines.append(f"  [PROVED before] {st}")
            lines.append("      A verified result. Cite or reuse it rather than "
                         "reproving it.")
        else:
            cost, why = _cost(r), _reason(r)
            lines.append(f"  [FAILED before{', ' + cost if cost else ''}] {st}")
            if why:
                lines.append(f"      why it failed: {why}")
    for r in lem:
        lines.append(f"  [LIBRARY, ready to cite] {r.get('name')}")
        sig = str(r.get("statement") or r.get("lean") or "")[:150]
        if sig:
            lines.append(f"      {sig}")
    if any(not r.get("ok") for r in ev):
        # The single most expensive observed failure mode, named explicitly.
        lines.append("  A claim marked FAILED has already been paid for. Rewording "
                     "it does not make it cheaper -- change the approach or move on.")
    return "\n".join(lines)
