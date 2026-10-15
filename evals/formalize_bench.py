#!/usr/bin/env python3
"""Build a compiler-curated benchmark for informal -> formal.

WHY THIS IS THE TARGET
----------------------
Ten hours of running the full agent produced zero solved problems, three
non-trivial lemmas, one verified proof of a FALSE claim, and one verified proof
of a TAUTOLOGY. Every independent measurement pointed at the same stage:

  22% of prover time went to claims that never became Lean statements
  50% of failed compiles break in the first quarter of the proof, 0% in the last
  6 of 8 IMO obligations could not be stated faithfully
  write_solution has been called zero times, ever, because closure requires a
  discharged obligation and none exist

The prover is not the problem and neither is the orchestrator. Turning an
informal claim into a Lean statement that says the same thing is the problem,
and it was being attempted as an incidental step inside a 40-minute loop where
it could not be measured or improved. This makes it the object of study.

WHERE THE GOLD COMES FROM
-------------------------
Mathlib ships ~8,300 files in which a natural-language docstring sits directly
above the formal statement it describes. That is exactly an (informal, formal)
pair, written by mathematicians, for free, at a scale no hand-curated set will
reach.

TWO THINGS THAT MAKE NAIVE EXTRACTION WRONG, BOTH OBSERVED
----------------------------------------------------------
1. MISALIGNMENT. A regex that takes the nearest preceding docstring will pair a
   docstring about `ULift` algebras with a theorem called `map_algebraMap_mul`
   further down the file. Adjacency has to be strict: nothing but whitespace
   between the closing `-/` and the `theorem` keyword.

2. NOT SELF-CONTAINED. Mathlib theorems inherit `variable` declarations from
   their section, so a statement like `(hb : Injective b) : Injective
   (algebraMap F E)` mentions `b`, `F` and `E` that are bound nowhere in the
   text extracted. Such a statement cannot elaborate on its own, so it cannot
   be a target and it cannot be a gold.

The fix for (2) is not a cleverer regex, because no amount of parsing tells you
whether Lean will accept something. Each candidate gold is COMPILED. What
survives is a benchmark whose every target is known to elaborate standalone --
curated by the compiler, not by a heuristic.

    python3 evals/formalize_bench.py --extract          # candidates only, fast
    python3 evals/formalize_bench.py --verify --limit 300
"""

import argparse
import concurrent.futures as futures
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import mathlib_kb, prover                          # noqa: E402
from src.lean import check_lean, statement_ok               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "formalize_candidates.json")
BENCH = os.path.join(HERE, "formalize_bench.json")

# Strict adjacency: `-/` then only whitespace then the keyword. This is the fix
# for the misalignment above -- anything else and the docstring belongs to a
# different declaration.
PAIR = re.compile(
    # The docstring may not contain a closing marker. Writing this as `.+?`
    # lets the match skip a `-/` whose following line is not a theorem and
    # attach the prose to a LATER declaration -- observed pairing a docstring
    # about `ULift` algebras with `map_algebraMap_mul`, and swallowing a
    # `protected class IsEpi` into the docstring text.
    r"/--(?P<doc>(?:(?!-/).)*?)-/[ \t]*\r?\n"
    r"(?P<kw>theorem|lemma)\s+(?P<name>[A-Za-z_][\w'₀-₉.]*)"
    # Stop at `:=` whether the proof is tactic mode or term mode, and never
    # cross into the next declaration. Requiring `:= by` made a term-mode proof
    # scan forward to the next `by`, dragging an entire second theorem into the
    # captured signature.
    r"(?P<sig>(?:(?!\r?\n\s*(?:theorem|lemma|def|instance|abbrev|end|namespace|section)\b).)*?)"
    r"\s*:=",
    re.S)

# A docstring that is mostly Lean is not an informal statement. These reject the
# entries that would otherwise make the benchmark trivially easy or unanswerable.
_BAD_DOC = re.compile(r"^\s*(TODO|FIXME|See |Alias|Deprecated|`)", re.I)

# Docstrings that describe LEAN rather than MATHEMATICS. These are not hard
# instances of the task -- they are not instances of it at all, because they
# cannot be formalized without already knowing the declaration they refer to:
#
#   "This is the `Ideal` version of `Submodule.FG.map`."
#   "to construct a `StarOrderedRing` instance it suffices to show ..."
#   "Variant of the interaction of `sum` and `•` ..."
#
# Both of the first two produced an empty derivation in the pilot run, and
# counting them as failures would blame the model for an unanswerable question.
# The filter is deliberately NARROW: an earlier, broader one also rejected
# "for every prime p dividing the order of a finite additive group G there
# exists an element of order p", which is Cauchy's theorem and a perfectly good
# item.
_LEAN_META = re.compile(
    r"(\bversion of\b|\bvariant of\b|\bto construct an?\b[^.]{0,40}\binstance\b"
    r"|\bsimp[- ]lemma\b|\bthis is the\b[^.]{0,40}\bversion\b"
    r"|\bAPI\b|\bnamespace\b|\bnotation\b|\bcoercion\b)", re.I)


def _plausible_doc(doc: str) -> bool:
    if len(doc) < 45 or len(doc) > 600:
        return False
    if _BAD_DOC.match(doc) or _LEAN_META.search(doc):
        return False
    if doc.count("`") > 10:                    # mostly code, not prose
        return False
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"`[^`]*`", " ", doc))
    return len(words) >= 8                     # enough English to be a claim


def extract(limit_files: int = 0, self_contained_only: bool = True) -> list:
    """Every strictly-adjacent docstring/theorem pair with prose worth using."""
    root = mathlib_kb.MATHLIB_SRC
    files = []
    for dirpath, _, names in os.walk(root):
        files += [os.path.join(dirpath, n) for n in names if n.endswith(".lean")]
    files.sort()
    if limit_files:
        files = files[:limit_files]

    out = []
    for f in files:
        try:
            src = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in PAIR.finditer(src):
            doc = " ".join(m.group("doc").split())
            sig = " ".join(m.group("sig").split())
            if not _plausible_doc(doc) or not sig or len(sig) > 400:
                continue
            if ":" not in sig:                 # no conclusion to state
                continue
            # A statement whose variables come from a section `variable` line
            # has no explicit binders, and it is NOT well-posed for equivalence
            # testing even though it compiles: spliced into `example : A -> B`
            # its variables auto-bind ONCE at the example level, so it becomes a
            # specific INSTANCE while the derived statement is a forall. The
            # forward direction then proves by instantiation and the backward
            # direction is correctly unprovable -- generalising from a single
            # instance -- so such an item can never score T2 however good the
            # derivation is. Observed on prod_subset_prod_iff: fwd=True,
            # bwd=False, with a derivation identical to the gold up to renaming.
            head = sig.split(":", 1)[0].strip()
            if not head:
                continue
            # Only the BENCHMARK needs this. A statement that inherits its
            # context cannot be a standalone test item, but it is still a
            # perfectly correct formalization to learn from, and requiring it
            # discards roughly three quarters of the available training pairs.
            if self_contained_only and not likely_self_contained(sig):
                continue
            out.append({"name": m.group("name"),
                        "informal": doc,
                        "signature": sig,
                        "module": os.path.relpath(f, root).replace("/", ".")[:-5]})
    return out


_TYPEVAR = re.compile(r"(?<![\w.])([A-Zα-ω])(?![\w'])")


def likely_self_contained(sig: str) -> bool:
    """Cheap pre-filter: is every type variable used also BOUND here?

    The compile step costs 6.3 seconds per candidate and rejects ~95% of them,
    nearly always for the same reason -- the statement inherits `R`, `M`, `α`
    from a section `variable` line that the extraction cannot see. Deciding that
    textually costs microseconds and is conservative: it only rejects when a
    bare single-letter type name is USED and never BOUND, which no
    self-contained statement does.
    """
    binders, concl = sig.split(":", 1) if ":" in sig else ("", sig)
    # Everything declared anywhere in the binder section counts as bound.
    bound = set(_TYPEVAR.findall(binders))
    used = set(_TYPEVAR.findall(concl))
    return not (used - bound)


def _standalone(rec: dict) -> str:
    """The gold as a self-contained declaration, renamed to avoid clashing."""
    return f"theorem bench_{rec['name'].replace('.', '_')} {rec['signature']} := by sorry"


def verify_one(rec: dict) -> dict:
    """Does this gold elaborate on its own? The only filter that is trustworthy.

    `sorry` is expected and fine -- we are checking the STATEMENT, so the only
    complaint Lean may raise is the injected `sorry` itself.
    """
    code = _standalone(rec)
    try:
        # The prover's own preamble, so a gold that elaborates here is a gold
        # that elaborates in the exact environment the system proves in.
        v = check_lean(f"{prover.PREAMBLE}{code}")
        errs = str(v.get("errors", ""))
        ok = statement_ok(v)
    except Exception as e:
        return dict(rec, elaborates=False, why=f"{type(e).__name__}")
    return dict(rec, elaborates=bool(ok), why="" if ok else errs[:200])


def verify(cands: list, limit: int = 0, workers: int = 4, seed: int = 0) -> list:
    """Compile the golds. Everything that survives is a usable benchmark item.

    SHUFFLED first, and that is not cosmetic. Candidates come out in file order,
    so taking the first N samples a handful of adjacent Mathlib modules: the
    first 250 produced a "benchmark" that was 16 items of `CharP` and 7 of
    `FiveLemma`. Measuring formalization on that measures two files.
    """
    import random
    cands = list(cands)
    random.Random(seed).shuffle(cands)
    if limit:
        cands = cands[:limit]
    # Merge into whatever is already on disk rather than replacing it: this step
    # costs hours, and the first version wrote its results only at the very end,
    # so any interruption discarded all of it.
    try:
        existing = {r["name"]: r for r in json.load(open(BENCH))}
    except Exception:
        existing = {}
    seen = set(existing)
    done, kept = 0, list(existing.values())
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(verify_one, cands):
            done += 1
            if r["elaborates"] and r["name"] not in seen:
                kept.append(r)
                seen.add(r["name"])
            if done % 25 == 0:
                json.dump(kept, open(BENCH, "w"), indent=1)      # crash-safe
                print(f"  verified {done}/{len(cands)} -> {len(kept)} total usable",
                      flush=True)
    json.dump(kept, open(BENCH, "w"), indent=1)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.extract or not os.path.exists(CANDIDATES):
        c = extract(a.files)
        json.dump(c, open(CANDIDATES, "w"), indent=1)
        print(f"extracted {len(c)} candidate pairs -> {CANDIDATES}")
        for r in c[:3]:
            print(f"\n  {r['name']}\n    doc : {r['informal'][:130]}"
                  f"\n    sig : {r['signature'][:130]}")

    if a.verify:
        c = json.load(open(CANDIDATES))
        print(f"\ncompiling {a.limit or len(c)} golds "
              f"({a.workers} workers) -- this is the curation step")
        try:
            before = len(json.load(open(BENCH)))
        except Exception:
            before = 0
        kept = verify(c, a.limit, a.workers, a.seed)
        json.dump(kept, open(BENCH, "w"), indent=1)
        tried = max(min(a.limit or len(c), len(c)), 1)
        new = len(kept) - before
        # Report the NEW yield, not new+existing over tried -- the latter reads
        # as 50% when the true per-candidate yield is 16%, and that number is
        # what decides how long a further expansion takes.
        print(f"\n{len(kept)} usable items total ({before} before, {new} new)")
        print(f"per-candidate yield this pass: {new}/{tried} = {new / tried:.0%}")
        print("Every item's gold statement is known to elaborate standalone.")


if __name__ == "__main__":
    main()
