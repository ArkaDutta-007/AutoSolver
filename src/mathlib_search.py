"""Mathlib declaration search.

Hallucinated lemma names are the dominant proof failure. The prover burned six
rounds on `Real.sqrt_eq_iff_sq_eq` (nonexistent) while `irrational_sqrt_two` --
which would have closed that goal -- sat in Mathlib the whole time, unreachable
because nothing could look it up.

This grep over Mathlib's sources is the lookup. It is used two ways:
  * exposed as a tool the agent can call, and
  * fired AUTOMATICALLY when the compiler reports an unknown constant, so real
    candidates are injected into the retry prompt without anyone asking.
"""

import functools
import os
import re
import subprocess

from . import config

MATHLIB_SRC = os.path.join(config.LEAN_PROJECT, ".lake", "packages", "mathlib", "Mathlib")

_DECL_RE = r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|nonrec\s+)?(?:theorem|lemma) ([A-Za-z_][A-Za-z0-9_'.]*)"


def available() -> bool:
    return os.path.isdir(MATHLIB_SRC)


@functools.lru_cache(maxsize=1)
def _all_names() -> tuple:
    """Every theorem/lemma name in Mathlib. Grepped once, then cached.

    BSD grep is POSIX ERE with no lookahead, so multi-fragment matching is done
    in Python rather than in the pattern.
    """
    if not available():
        return ()
    try:
        proc = subprocess.run(
            ["grep", "-rhoE", r"^(theorem|lemma) [A-Za-z_][A-Za-z0-9_'.]*",
             MATHLIB_SRC, "--include=*.lean"],
            capture_output=True, text=True, timeout=180)
    except Exception:
        return ()
    names = {line.split(None, 1)[1] for line in proc.stdout.splitlines()
             if " " in line}
    return tuple(sorted(names))


@functools.lru_cache(maxsize=1)
def _all_decls() -> tuple:
    """(name, declaration line) for every theorem/lemma.

    Names alone are not enough. `Nat.factorization_gcd` is findable by name only
    if you already know Mathlib calls it "factorization"; the agent called it a
    "valuation" and got nothing. The declaration line carries the TYPE, where
    `gcd`, `min` and the rest actually appear.
    """
    if not available():
        return ()
    try:
        proc = subprocess.run(
            ["grep", "-rhE", r"^(theorem|lemma) [A-Za-z_][A-Za-z0-9_'.]*",
             MATHLIB_SRC, "--include=*.lean"],
            capture_output=True, text=True, timeout=300)
    except Exception:
        return ()
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        out.setdefault(parts[1], line[:400])
    return tuple(sorted(out.items()))


def search_mathlib(query: str, limit: int = 25) -> dict:
    """Find real Mathlib declarations matching space-separated fragments.

    Rewritten after watching it fail on the run that needed it. The agent asked
    four sensible questions and got almost nothing back:

        'val pval prime'              -> 0
        'gcd prime exponent valuation'-> 0
        'prime gcd lcm'               -> 0
        'gcd lcm'                     -> 4        (out of 139k declarations)

    Three causes, all fixed here. Fragments were ANDed, so one unlucky word
    zeroed the result with no fallback. Only NAMES were searched, never types.
    And the synonym table did not know that a p-adic valuation is spelled
    `factorization` or `padicValNat` in Mathlib -- the exact word the agent
    needed and could not guess.

    Now: expand synonyms, try AND over names, then AND over declaration lines,
    then fall back to OR ranked by how many fragments matched. Something useful
    comes back for any reasonable question.
    """
    raw = [f.lower() for f in query.replace(".", " ").replace("_", " ").split() if f]
    if not raw:
        return {"error": "empty query"}
    frags = []
    for f in raw:
        for alt in _SYNONYMS.get(f, f).split():
            if alt not in frags:
                frags.append(alt)

    decls = _all_decls()
    names = _all_names()
    if not decls and names:                       # fall back to the name-only index
        decls = tuple((n, n) for n in names)

    def rank(hits):
        return sorted(hits, key=lambda n: (len(n), n))

    # 1. every fragment in the NAME -- the most precise answer
    hit = [n for n, _ in decls if all(f in n.lower() for f in frags)]
    if hit:
        return {"query": query, "matched_on": "name", "count": len(hit),
                "names": rank(hit)[:limit], "truncated": len(hit) > limit}

    # 2. every fragment somewhere in the DECLARATION (name or type)
    hit = [n for n, line in decls if all(f in line.lower() for f in frags)]
    if hit:
        return {"query": query, "matched_on": "statement", "count": len(hit),
                "names": rank(hit)[:limit], "truncated": len(hit) > limit,
                "note": "matched in the theorem's type, not its name"}

    # 3. OR, ranked by how much of the query each declaration covers
    scored = []
    for n, line in decls:
        low = line.lower()
        k = sum(1 for f in frags if f in low)
        if k:
            scored.append((-k, len(n), n))
    if not scored:
        return {"query": query, "count": 0, "names": [],
                "note": f"nothing matched {frags}. Try fewer or different words -- "
                        f"Mathlib says `factorization`/`padicValNat` for valuations, "
                        f"`dvd` for divides, `Nat.Coprime` for coprime."}
    scored.sort()
    best = -scored[0][0]
    return {"query": query, "matched_on": f"partial ({best}/{len(frags)} terms)",
            "count": len(scored), "names": [n for _, _, n in scored[:limit]],
            "truncated": len(scored) > limit,
            "note": "no declaration matched every term; these match the most"}


@functools.lru_cache(maxsize=2048)
def module_for(name: str) -> str:
    """Mathlib module that declares `name`, e.g. 'Mathlib.Algebra.Ring.Parity'.

    Lets us emit NARROW imports instead of `import Mathlib`: 1.5s and 1.3 GB
    versus 4s and 5.7 GB per check, measured.
    """
    bare = name.split(".")[-1]
    if not available():
        return ""
    try:
        proc = subprocess.run(
            ["grep", "-rlE", rf"^(theorem|lemma) {re.escape(bare)}\b",
             MATHLIB_SRC, "--include=*.lean"],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return ""
    paths = [p for p in proc.stdout.splitlines() if p]
    if not paths:
        return ""
    rel = os.path.relpath(sorted(paths, key=len)[0], os.path.dirname(MATHLIB_SRC))
    return rel[:-5].replace(os.sep, ".") if rel.endswith(".lean") else ""


_CITED = re.compile(r"\b([A-Z][A-Za-z0-9_']*\.)*[a-z][A-Za-z0-9_']*(?:_[A-Za-z0-9_']+)+\b")


def narrow_imports(code: str, cap: int = 12) -> str:
    """Derive minimal import lines for the identifiers a proof actually cites."""
    cited = {m.group(0) for m in _CITED.finditer(code)}
    mods = set()
    for name in list(cited)[:60]:
        if not exists(name):
            continue
        mod = module_for(name)
        if mod:
            mods.add(mod)
        if len(mods) >= cap:
            break
    return "\n".join(f"import {m}" for m in sorted(mods))


def exists(name: str) -> bool:
    """Whether a declaration name exists in Mathlib (bare or namespaced)."""
    bare = name.split(".")[-1]
    return any(n == bare or n.endswith("." + bare) for n in _all_names())


_STOP = {"nat", "real", "int", "of", "iff", "eq", "the", "for", "all", "and", "is",
         "that", "such", "with", "prove", "show", "let", "then", "any", "there",
         "exists", "every", "if", "a", "an", "to", "in", "be", "it", "this",
         # Prose scaffolding. These carry no mathematical content but ARE rare
         # enough as name fragments to win the specificity ranking, which is how
         # "gcd(m,n) * lcm(m,n) = m*n" came back suggesting
         # `valuation_irreducible_pos` while `Nat.gcd_mul_lcm` -- the exact
         # lemma, and the first hit for "gcd lcm" -- was never surfaced.
         "positive", "negative", "non", "integers", "integer", "natural",
         "number", "numbers", "value", "values", "given", "arbitrary", "some",
         "where", "when", "which", "both", "each", "same", "have", "has",
         "define", "defined", "denote", "consider", "suppose", "assume",
         "replace", "replacing", "replaced", "obtain", "result", "results",
         "following", "above", "below", "first", "second", "also", "must",
         "can", "will", "would", "does", "not", "but", "are", "was", "were",
         # Second round of the same bug: "squared" (10 matches) outranked the
         # mathematics and suggested `J_squared`, `upperMoebius_lambdaSquared`
         # for a claim about coprimality. Prose describing an operation is not
         # the operation -- Mathlib spells these `pow`, `sq`, `mul`.
         "squared", "cubed", "times", "plus", "minus", "divided", "equals",
         "product", "quotient", "difference", "power", "raised", "root",
         "greater", "less", "least", "most", "than", "then", "over", "into"}

# Informal words -> the vocabulary Mathlib actually uses in declaration names.
_SYNONYMS = {
    "irrational": "irrational", "rational": "rat", "square": "sq", "root": "sqrt",
    "prime": "prime", "primes": "prime", "divides": "dvd", "divisible": "dvd",
    "infinitely": "infinite", "infinite": "infinite", "greatest": "gcd",
    "least": "lcm", "positive": "pos", "negative": "neg", "continuous": "continuous",
    "derivative": "deriv", "integral": "integral", "sum": "sum", "product": "prod",
    # The vocabulary gap that cost the P1 run four empty searches. The agent
    # reached for "valuation"/"pval"/"exponent"; Mathlib spells the same idea
    # `factorization` (or `padicValNat`), and nothing bridged the two.
    "valuation": "factorization", "val": "factorization", "pval": "factorization",
    "padic": "padicValNat", "exponent": "factorization", "multiplicity": "multiplicity",
    "factorisation": "factorization", "factorization": "factorization",
    "gcd": "gcd", "lcm": "lcm", "coprime": "coprime", "relatively": "coprime",
    "divisor": "divisors", "divisors": "divisors", "factor": "factor",
    "multiset": "multiset", "invariant": "invariant", "minimum": "min",
    "maximum": "max", "board": "", "blackboard": "", "move": "", "game": "",

    # Everything above is number theory, because that is the only domain this
    # system had ever been pointed at. Aimed at real maths and physics research
    # the same vocabulary gap reopens in every other area: "determinant"
    # returned ONE match because Mathlib spells it `det`, and "continuous
    # function" surfaced `valuation_surjective`.
    # --- linear algebra ---
    "determinant": "det", "determinants": "det", "matrix": "Matrix",
    "matrices": "Matrix", "transpose": "transpose", "eigenvalue": "eigenvalue",
    "eigenvector": "eigenvector", "invertible": "Invertible", "rank": "rank",
    "trace": "trace", "basis": "Basis", "span": "span", "kernel": "ker",
    "linear": "linear", "orthogonal": "orthogonal", "inner": "inner",
    # --- analysis / topology ---
    "differentiable": "Differentiable", "derivatives": "deriv",
    "limit": "Tendsto", "limits": "Tendsto", "converges": "Tendsto",
    "convergent": "Tendsto", "bounded": "Bounded", "compact": "Compact",
    "closed": "IsClosed", "open": "IsOpen", "interval": "Icc",
    "supremum": "sSup", "infimum": "sInf", "norm": "norm", "metric": "Metric",
    "uniformly": "UniformContinuous", "series": "tsum", "sequence": "seq",
    "monotone": "Monotone", "antitone": "Antitone", "smooth": "ContDiff",
    # --- measure / probability ---
    "measure": "Measure", "measurable": "Measurable", "probability": "Probability",
    "expectation": "integral", "independent": "Indep", "variance": "variance",
    "almost": "ae", "everywhere": "ae",
    # --- geometry ---
    "distance": "dist", "angle": "angle", "triangle": "triangle",
    "perpendicular": "perp", "parallel": "parallel", "circle": "sphere",
    "convex": "Convex", "affine": "Affine",
    # --- physics, which in Mathlib IS analysis ---
    "velocity": "deriv", "acceleration": "deriv", "momentum": "deriv",
    "force": "deriv", "work": "integral", "energy": "integral",
    "harmonic": "cos", "oscillation": "cos", "displacement": "sub",
}


def _keywords(text: str) -> list:
    words = re.split(r"[^A-Za-z]+", text.lower())
    out = []
    for w in words:
        if len(w) < 3 or w in _STOP:
            continue
        out.append(_SYNONYMS.get(w, w))
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _math_terms(text: str) -> set:
    """Terms in the claim that map onto vocabulary Mathlib actually uses.

    A word surviving _SYNONYMS is one we KNOW names a real Mathlib concept
    (`determinant` -> `det`, `momentum` -> `deriv`). Everything else is prose
    until proven otherwise, and prose is what keeps winning the rarity ranking.
    """
    words = re.split(r"[^A-Za-z]+", (text or "").lower())
    return {_SYNONYMS[w] for w in words if _SYNONYMS.get(w)}


def suggest_for_unknown(bad_names: list, statement: str = "", limit: int = 8) -> str:
    """Propose real Mathlib declarations after the compiler rejects invented ones.

    Searches on BOTH the rejected name and the claim text. The claim matters more:
    on the √2 failure the invented name `Real.sqrt_eq_iff_sq_eq` yields only
    generic sqrt lemmas, while the claim "square root of 2 is irrational" surfaces
    `irrational_sqrt_two` -- the lemma that actually closes the goal.
    """
    # Dense retrieval first when the index has been built. It is not a better
    # keyword search -- it is a different mechanism, and it is the one that can
    # cross vocabulary gaps this function cannot: the agent asked for a "p-adic
    # valuation" lemma and no amount of lexical overlap reaches
    # `Nat.factorization_gcd`, because the two share no token. Falls through to
    # the keyword path below when the index is absent or the embed call fails,
    # so this is strictly additive.
    try:
        from . import mathlib_kb
        if mathlib_kb.available():
            dense = mathlib_kb.suggest(statement, bad_names, limit=limit)
            if dense:
                return dense
    except Exception:
        pass

    lines = []

    for bad in bad_names[:3]:
        parts = [p for p in re.split(r"[._]", bad)
                 if p and p.lower() not in _STOP]
        hits = search_mathlib(" ".join(parts[:2]), limit).get("names", []) if parts else []
        if not hits and parts:
            hits = search_mathlib(parts[0], limit).get("names", [])
        if hits:
            lines.append(f"  instead of `{bad}` (does not exist): {', '.join(hits[:limit])}")

    # Rank keyword queries by SPECIFICITY: a term matching few declarations is
    # far more informative than a common one. Searching "sq"/"sqrt" drowns out
    # "irrational", which is the word that actually finds the lemma.
    kws = _keywords(statement)
    # Pair the RECOGNISED terms with each other first. Combos were adjacent
    # slices of the keyword list, so in "determinant of a product of two square
    # matrices" the pair `det + Matrix` was never even tried -- `det + two` was,
    # because `two` happened to sit next to it. Adjacency in an English sentence
    # says nothing about which Mathlib concepts belong together.
    known = _math_terms(statement)
    known_kws = [k for k in kws if k in known]
    combos = [[a, b] for i, a in enumerate(known_kws[:3]) for b in known_kws[i + 1:4]]
    combos += [[k] for k in known_kws[:3]]
    combos += [kws[i:i + 2] for i in range(len(kws[:4]) - 1)] + [[k] for k in kws[:4]]
    scored = []
    seen_combo = set()
    for combo in combos:
        sig = " ".join(combo)
        if sig in seen_combo:
            continue
        seen_combo.add(sig)
        res = search_mathlib(sig, limit)
        n = res.get("count", 0)
        if n:
            scored.append((n, combo, res["names"]))
    # Rank by HOW MUCH mathematical content a combo carries, and only then by
    # rarity. Pure specificity is an arms race against prose -- "positive" beat
    # "gcd lcm", then "squared" beat "coprime", then "two" beat "Matrix" -- and
    # ranking on mere PRESENCE of a known term still lost, because `det + two`
    # is rarer than `det + Matrix`. Count the known terms instead: a combo of
    # two real Mathlib concepts outranks one real concept plus a stray article.
    scored.sort(key=lambda t: (-len(set(t[1]) & known), t[0]))
    for n, combo, names in scored[:3]:
        lines.append(f"  matching the goal ({' + '.join(combo)}, {n} matches): "
                     f"{', '.join(names[:limit])}")

    if not lines:
        return ""
    return ("\n\nREAL Mathlib declaration names (from a search of 139k declarations) "
            "-- use these, do not invent names:\n" + "\n".join(lines))
