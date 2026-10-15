"""Accumulating lemma library, with curation.

Every proof previously started from nothing -- a lemma proved for one problem was
unavailable to the next, so the system could not build upward. Verified theorems
are appended here and made importable, turning a sequence of one-shot proofs into
a growing formal library.

APPENDING IS NOT CURATION
-------------------------
The first version of this module did exactly one thing: append, re-verify, roll
back on failure. That is necessary and nowhere near sufficient, and the live
library shows why -- it holds `lean_workbook_plus_10017`, a name that says
nothing, for a lemma nothing has ever used, with no record of where it came
from or whether it was ever worth keeping.

Four things were missing, and each one is a way the library fails to compound:

  1. FINDABILITY. Lemmas were importable but not searchable. The agent could
     not discover what it had already proved, so it reproved things.
  2. SEMANTIC DEDUP. Collision was checked on the NAME only, so the same
     statement under a different name was stored twice -- the library-level
     version of the `established` list holding five copies of one fact.
  3. USAGE. Nothing recorded which lemmas were ever used by a later proof, so
     there was no signal distinguishing an asset from clutter.
  4. GENERALISATION. DreamProver's wake/sleep split: `wake` proves, `sleep`
     mines the successful proofs for recurring structure and abstracts it.
     Without sleep, lemmas stay problem-specific and never transfer.

The .lean file remains the source of truth for CODE -- it is what Lean compiles.
The JSON sidecar is an index over it, and is rebuildable from scratch, so a
corrupt index can never cost a verified proof.
"""

import json
import os
import re
import subprocess
import threading
import time

from . import config
from .lean import LAKE, check_lean

MODULE = "Proofs.Verified"
PATH = os.path.join(config.LEAN_PROJECT, "Proofs", "Verified.lean")
INDEX = os.path.join(config.ROOT, "kb", "library.json")
HEADER = ("-- Verified lemmas accumulated by autoresearch.\n"
          "-- Every entry compiled cleanly when added. Import with:\n"
          "--   import Proofs.Verified\n"
          "import Mathlib\nset_option maxHeartbeats 400000\n\n")

_lock = threading.RLock()
_DECL_NAME = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)", re.MULTILINE)
_DECL_FULL = re.compile(r"^\s*(?:theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_']*(.*?)(?::=|\Z)",
                        re.MULTILINE | re.DOTALL)

# Behavioural descriptors. These are NOT decoration: they are the axes the
# quality-diversity archive uses to decide whether a new lemma is a genuinely
# different KIND of result or the fifth restatement of one we already have.
_MECHANISM = {
    "induction": r"\binduction\b|\bNat\.rec\b|\binduction'\b",
    "contradiction": r"\bby_contra\b|\bcontradiction\b|\babsurd\b",
    "computation": r"\bdecide\b|\bnorm_num\b|\bnative_decide\b",
    "arithmetic": r"\bomega\b|\blinarith\b|\bnlinarith\b|\bpositivity\b",
    "algebraic": r"\bring\b|\bfield_simp\b|\bgroup\b|\babel\b",
    "simplification": r"\bsimp\b|\bsimpa\b|\bnorm_cast\b",
    "case_analysis": r"\brcases\b|\bobtain\b|\bmatch\b|\binterval_cases\b",
    "existence": r"\buse\b|\bexists\b|\brefine\s*⟨",
    # No \b around non-word symbols: `\b∣\b` can never match, because ∣ is not
    # a word character, so the boundary assertion fails on both sides. This
    # silently disabled the divisibility/function/integral descriptors.
    "measure": r"\bMeasure\b|\bIntegrable\b|∫",
    "topology": r"\bContinuous\b|\bIsCompact\b|\bIsOpen\b|\bIsClosed\b",
    # Without these, every short proof got an EMPTY mechanism list -- 7 of the 8
    # live entries, because `exact`/`apply` were not patterns. An empty
    # descriptor makes the quality-diversity archive blind exactly where it
    # matters: a one-line application of an existing lemma is a real and
    # distinct KIND of result.
    "direct": r":=\s*by\s+exact\b|:=\s*[A-Z]|\bexact\b|\bapply\b",
    "structural": r"\bconstructor\b|\brefine\b|\bexact\?\b",
}
_OBJECTS = {
    "nat": r"\bℕ\b|\bNat\b", "int": r"\bℤ\b|\bInt\b", "real": r"\bℝ\b|\bReal\b",
    "complex": r"\bℂ\b", "rat": r"\bℚ\b", "matrix": r"\bMatrix\b",
    "finset": r"\bFinset\b|\bFin\b", "set": r"\bSet\b", "poly": r"\bPolynomial\b",
    "function": r"→", "prime": r"\bPrime\b", "divisibility": r"∣",
}


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def _load_index() -> dict:
    try:
        with open(INDEX, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_index(idx: dict) -> None:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    tmp = INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, INDEX)              # atomic: a crash mid-write cannot corrupt


def _ensure() -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    if not os.path.exists(PATH):
        with open(PATH, "w", encoding="utf-8") as fh:
            fh.write(HEADER)


def names() -> list:
    if not os.path.exists(PATH):
        return []
    with open(PATH, encoding="utf-8") as fh:
        return _DECL_NAME.findall(fh.read())


# --------------------------------------------------------------------------
# descriptors + identity
# --------------------------------------------------------------------------

def _tags(pattern_map: dict, text: str) -> list:
    return sorted(k for k, pat in pattern_map.items() if re.search(pat, text))


def describe(code: str) -> dict:
    """Behavioural descriptors for one proof: what it is ABOUT and HOW it works."""
    stmt = statement_of(code)
    return {"mechanism": _tags(_MECHANISM, code),
            "objects": _tags(_OBJECTS, stmt or code),
            "tactics": sorted(set(re.findall(
                r"\b(omega|simp|linarith|nlinarith|norm_num|ring|decide|aesop|"
                r"positivity|field_simp|exact\?|induction|rcases|constructor)\b", code)))}


def statement_of(code: str) -> str:
    """The signature, up to `:=`. This is the lemma's IDENTITY; the proof is not."""
    m = _DECL_FULL.search(code or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


# Lean spells the same type several ways and the library collected all of them.
# Measured on the live file: EIGHT entries, TWO distinct facts. `tt` and
# `lean_workbook_plus_10017` differ only by `Nat` vs `ℕ`; `gcd_divides_3`
# differs from `gcd_divides` only by `()` vs `{}` binders. Without normalising
# these, dedup silently passes and the library grows without learning anything
# -- the `established` local-minimum bug, one layer down.
_TYPE_ALIASES = ((r"ℕ", "Nat"), (r"ℤ", "Int"), (r"ℝ", "Real"),
                 (r"ℚ", "Rat"), (r"ℂ", "Complex"))


def _fingerprint(code: str) -> str:
    """Alpha-insensitive identity for a statement.

    Two lemmas differing only in binder names, binder EXPLICITNESS, or unicode
    type spelling are the same lemma for storage purposes. Being slightly
    aggressive is the right bias: a false duplicate costs one lemma we already
    have, a false distinct costs a permanent copy that dilutes every search.
    """
    s = statement_of(code)
    for uni, ascii_ in _TYPE_ALIASES:
        s = re.sub(uni, ascii_, s)
    s = re.sub(r"[\{\[]", "(", s)                             # implicit -> explicit
    s = re.sub(r"[\}\]]", ")", s)
    # True alpha-equivalence: rename bound variables to POSITIONAL placeholders
    # everywhere they occur, not just inside the binder. Stripping the binder
    # name alone left the body saying `x = x` vs `y = y`, so two spellings of
    # one lemma still fingerprinted differently.
    for i, nm in enumerate(n for grp in re.findall(r"\(\s*([A-Za-z_][\w'\s]*?)\s*:", s)
                           for n in grp.split()):
        s = re.sub(r"(?<![\w'])" + re.escape(nm) + r"(?![\w'])", f"v{i}", s)
    s = re.sub(r"[₀-₉']", "", s)
    return re.sub(r"\s+", "", s).lower()


def suggest_name(code: str, statement: str = "") -> str:
    """A name that says what the lemma is.

    `lean_workbook_plus_10017` is a real entry in the live library. A name that
    carries no meaning makes the library unsearchable by a human AND unusable
    as a retrieval target.
    """
    src = statement_of(code) or statement or ""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_']*", src)
    stop = {"theorem", "lemma", "by", "the", "of", "for", "all", "and", "is", "in",
            "to", "a", "an", "if", "then", "have", "this", "that", "with"}
    keep = [t for t in toks if t.lower() not in stop and len(t) > 1][:4]
    base = "_".join(t.lower() for t in keep) or "lemma"
    return re.sub(r"_+", "_", base)[:48].strip("_")


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------

def add(code: str, origin: str = "", rename: bool = True) -> dict:
    """Append a verified theorem, re-verify the module, roll back on failure.

    Returns {"ok": bool, ...}. `ok=False, reason="duplicate"` when the statement
    is already present under any name -- that is a SUCCESS for the library, not
    an error for the caller.
    """
    body = re.sub(r"^\s*(import .*|set_option .*)$\n?", "", code, flags=re.MULTILINE).strip()
    m = _DECL_NAME.search(body)
    if not m:
        return {"ok": False, "reason": "no declaration found"}

    with _lock:
        _ensure()
        idx = _load_index()
        fp = _fingerprint(body)
        for nm, rec in idx.items():
            if rec.get("fingerprint") == fp:
                return {"ok": False, "reason": "duplicate", "name": nm,
                        "note": f"already present as `{nm}`"}

        existing = set(names())
        name = m.group(1)
        # Rename meaningless auto-generated names to something searchable.
        if rename and re.match(r"^(lean_workbook|thm|lemma|test|foo|r)\b|_\d{3,}$", name):
            better = suggest_name(body, origin)
            if better and better not in existing:
                body = re.sub(r"(^\s*(?:theorem|lemma)\s+)" + re.escape(name),
                              r"\1" + better, body, count=1, flags=re.MULTILINE)
                name = better
        if name in existing:
            name = next(f"{name}_{i}" for i in range(2, 9999)
                        if f"{name}_{i}" not in existing)
            body = re.sub(r"(^\s*(?:theorem|lemma)\s+)[A-Za-z_][A-Za-z0-9_']*",
                          r"\1" + name, body, count=1, flags=re.MULTILINE)

        with open(PATH, encoding="utf-8") as fh:
            before = fh.read()
        entry = (f"\n-- {origin[:150]}\n{body}\n" if origin else f"\n{body}\n")

        verdict = check_lean(before + entry)
        if not verdict["ok"]:
            return {"ok": False,
                    "reason": f"module failed to recompile: {verdict['errors'][:200]}"}
        # Only touch the file once Lean has approved the result. The previous
        # version wrote first and rolled back on failure, which left a window
        # where a concurrent reader saw a broken library.
        with open(PATH, "w", encoding="utf-8") as fh:
            fh.write(before + entry)

        d = describe(body)
        idx[name] = {"name": name, "statement": statement_of(body),
                     "origin": origin[:200], "fingerprint": fp,
                     "added": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "uses": 0, "used_by": [], "lines": body.count("\n") + 1,
                     "generalizes": [], **d}
        _save_index(idx)
        return {"ok": True, "name": name, "module": MODULE, "total": len(idx),
                **d}


# --------------------------------------------------------------------------
# find + use
# --------------------------------------------------------------------------

def search(query: str, k: int = 5) -> list:
    """Find lemmas we already proved. Dense when the KB is built, else lexical.

    Without this the library is write-only: the agent could import it but had
    no way to learn what was in it, so it reproved what it already had.
    """
    idx = _load_index()
    if not idx:
        return []
    recs = [r for r in idx.values() if not r.get("retired")]
    if not recs:
        return []
    try:
        from . import mathlib_kb
        if mathlib_kb.available():
            qv = mathlib_kb._embed_batch([f"{mathlib_kb.Q_PREFIX}{query}"], timeout=60)
            docs = [f"{mathlib_kb.D_PREFIX}{r['name']} : {r['statement']}" for r in recs]
            dv = mathlib_kb._embed_batch(docs, timeout=300)
            if qv and dv and len(dv) == len(recs):
                import numpy as np
                q = np.asarray(qv[0], dtype=np.float32)
                d = np.asarray(dv, dtype=np.float32)
                q /= max(float(np.linalg.norm(q)), 1e-9)
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                order = np.argsort(-(d @ q))[:k]
                return [{**recs[int(i)], "score": round(float((d @ q)[int(i)]), 4)}
                        for i in order]
    except Exception:
        pass
    terms = {t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2}
    scored = [(len(terms & set(re.findall(r"[a-z]+", (r["statement"] + " " + r["origin"]).lower()))), r)
              for r in recs]
    scored.sort(key=lambda t: -t[0])
    return [r for n, r in scored[:k] if n]


def record_use(name: str, by: str = "") -> None:
    """Note that a later proof used this lemma. The only signal of real value."""
    with _lock:
        idx = _load_index()
        if name in idx:
            idx[name]["uses"] = idx[name].get("uses", 0) + 1
            if by and by[:120] not in idx[name].setdefault("used_by", []):
                idx[name]["used_by"].append(by[:120])
            _save_index(idx)


def note_uses(code: str, by: str = "") -> list:
    """Record every library lemma cited by a freshly verified proof."""
    idx = _load_index()
    hit = [n for n in idx if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w'])", code or "")]
    for n in hit:
        record_use(n, by)
    return hit


def suggest(statement: str, limit: int = 4) -> str:
    """Prompt block: lemmas we ALREADY proved that may close this goal."""
    hits = search(statement, k=limit)
    if not hits:
        return ""
    lines = ["", f"Lemmas THIS SYSTEM already proved (import {MODULE} and use them):"]
    for h in hits:
        lines.append(f"  {h['name']} : {h['statement'][:130]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sleep: generalisation
# --------------------------------------------------------------------------

def sleep_candidates(min_group: int = 2) -> list:
    """Groups of lemmas sharing mechanism+objects -- candidates to generalise.

    Wake proves; sleep abstracts. Lemmas that solve the same KIND of problem by
    the same KIND of argument are where a single general statement can replace
    several specific ones.
    """
    idx = _load_index()
    groups: dict = {}
    for rec in idx.values():
        key = (tuple(rec.get("mechanism", [])), tuple(rec.get("objects", [])))
        if not any(key):
            continue
        groups.setdefault(key, []).append(rec)
    return [{"mechanism": list(k[0]), "objects": list(k[1]), "members": v}
            for k, v in groups.items() if len(v) >= min_group]


_GEN_SCHEMA = {"type": "object",
               "properties": {"lean": {"type": "string"}},
               "required": ["lean"]}


def llm_generalizer(model: str = ""):
    """A `propose` callable for sleep(), backed by a model. "" on any failure.

    Asks for a single Lean theorem that SUBSUMES the group. The result is not
    trusted -- sleep() puts it through add(), which recompiles the whole module
    before keeping anything. An unverified generalisation would be a false
    lemma every later proof could lean on, so "the model said so" is not a
    standard this library accepts.
    """
    import requests
    from . import config

    def propose(members: list) -> str:
        listing = "\n\n".join(f"-- {m.get('origin', '')[:80]}\n{m['name']} : "
                              f"{m.get('statement', '')}" for m in members[:5])
        prompt = (
            "These lemmas were each proved separately and share a mechanism:\n\n"
            f"{listing}\n\n"
            "Write ONE Lean 4 (Mathlib) theorem that GENERALISES them -- a single "
            "statement from which each of the above follows. Requirements:\n"
            "- complete Lean 4 source: `theorem name ... := by ...`, proof included\n"
            "- it must be strictly more general, not a restatement of one member\n"
            "- give it a descriptive name\n"
            "- if no honest generalisation exists, return an empty string. A wrong "
            "generalisation is far worse than none.")
        try:
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": model or config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "format": _GEN_SCHEMA,
                      "options": {**config.ctx(model or config.ORCHESTRATOR_MODEL), "num_predict": 4000, "temperature": 0.3},
                      # Same class as prover._derive_once: a `format` schema and a
                      # reasoning budget share one allowance and the JSON loses.
                      "think": False},
                timeout=900)
            r.raise_for_status()
            return str(json.loads(r.json()["message"]["content"]).get("lean", "")).strip()
        except Exception:
            return ""

    return propose


def sleep_phase(model: str = "", max_groups: int = 2) -> dict:
    """Run the generalisation pass with a model. Safe to call unconditionally."""
    try:
        if len(_load_index()) < 2:
            return {"groups_tried": 0, "kept": [], "note": "library too small"}
        return sleep(llm_generalizer(model), max_groups=max_groups)
    except Exception as exc:
        return {"groups_tried": 0, "kept": [], "error": str(exc)}


def sleep(propose, max_groups: int = 3) -> dict:
    """Ask `propose(members)` for a generalisation, VERIFY it, keep survivors.

    `propose` is injected rather than imported so this stays testable without a
    model. It receives the member records and returns Lean source (or "").

    Verification is not optional and not a formality. An unverified abstraction
    is a false lemma that every later proof may lean on -- the library would
    stop being a set of things known to be true, which is the only property
    that makes it worth having.
    """
    kept, tried = [], 0
    for group in sleep_candidates()[:max_groups]:
        tried += 1
        try:
            code = propose(group["members"])
        except Exception:
            continue
        if not code or not code.strip():
            continue
        res = add(code, origin=f"generalisation of {', '.join(m['name'] for m in group['members'][:4])}")
        if res.get("ok"):
            with _lock:
                idx = _load_index()
                idx[res["name"]]["generalizes"] = [m["name"] for m in group["members"]]
                _save_index(idx)
            kept.append(res["name"])
    return {"groups_tried": tried, "kept": kept}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _blocks() -> list:
    """(name, source block) for each entry, preserving the header separately."""
    with open(PATH, encoding="utf-8") as fh:
        text = fh.read()
    head, *rest = re.split(r"\n(?=\s*(?:theorem|lemma)\s)", text)
    out = []
    for b in rest:
        m = _DECL_NAME.search(b)
        if m:
            out.append((m.group(1), b.rstrip() + "\n"))
    return [head] + out


def dedup(dry_run: bool = True) -> dict:
    """Collapse entries that are the same statement under different names.

    Keeps the BEST representative per fingerprint, in this order:
      1. most used by later proofs   -- demonstrated value beats everything
      2. a meaningful name           -- `tt` is not a name
      3. no collision suffix         -- `gcd_divides` reads better than `gcd_divides_2`
      4. shortest proof              -- less to break when Mathlib moves

    Rewrites the .lean file and re-verifies. Never deletes without Lean's
    approval: a library that fails to compile is worse than a redundant one.
    """
    if not os.path.exists(PATH):
        return {"removed": 0}
    with _lock:
        idx = _load_index()
        parts = _blocks()
        head, entries = parts[0], parts[1:]
        groups: dict = {}
        for name, block in entries:
            groups.setdefault(_fingerprint(block), []).append((name, block))

        def rank(item):
            name, block = item
            rec = idx.get(name, {})
            junk = bool(re.match(r"^(tt|r|foo|thm|test)$|^lean_workbook|_\d{3,}$", name))
            suffixed = bool(re.search(r"_\d+$", name))
            return (-rec.get("uses", 0), junk, suffixed, block.count("\n"))

        keep, drop = [], []
        for fp, members in groups.items():
            members.sort(key=rank)
            keep.append(members[0])
            drop.extend(m[0] for m in members[1:])

        if not drop:
            return {"removed": 0, "note": "no duplicates"}
        if dry_run:
            return {"would_remove": drop, "kept": [k[0] for k in keep]}

        order = {name: i for i, (name, _) in enumerate(entries)}
        keep.sort(key=lambda kv: order.get(kv[0], 0))
        rebuilt = head.rstrip() + "\n" + "".join("\n" + b for _, b in keep)
        verdict = check_lean(rebuilt)
        if not verdict["ok"]:
            return {"removed": 0,
                    "error": f"deduped module would not compile: {verdict['errors'][:200]}"}
        with open(PATH, "w", encoding="utf-8") as fh:
            fh.write(rebuilt)
    reindex()
    return {"removed": len(drop), "dropped": drop, "remaining": len(keep)}


def stats() -> dict:
    idx = _load_index()
    if not idx:
        return {"total": 0}
    used = [r for r in idx.values() if r.get("uses", 0) > 0]
    mech: dict = {}
    for r in idx.values():
        for m in r.get("mechanism", []):
            mech[m] = mech.get(m, 0) + 1
    return {"total": len(idx), "used": len(used),
            "unused": len(idx) - len(used),
            "generalisations": sum(1 for r in idx.values() if r.get("generalizes")),
            "mechanisms": dict(sorted(mech.items(), key=lambda t: -t[1])[:6])}


def reindex() -> dict:
    """Rebuild the JSON index from the .lean file. The file is the truth."""
    if not os.path.exists(PATH):
        return {"total": 0}
    with open(PATH, encoding="utf-8") as fh:
        text = fh.read()
    blocks = re.split(r"\n(?=\s*(?:theorem|lemma)\s)", text)
    with _lock:
        old, idx = _load_index(), {}
        for b in blocks:
            m = _DECL_NAME.search(b)
            if not m:
                continue
            name = m.group(1)
            prev = old.get(name, {})
            idx[name] = {"name": name, "statement": statement_of(b),
                         "origin": prev.get("origin", ""),
                         "fingerprint": _fingerprint(b),
                         "added": prev.get("added", time.strftime("%Y-%m-%d %H:%M:%S")),
                         "uses": prev.get("uses", 0),
                         "used_by": prev.get("used_by", []),
                         "lines": b.count("\n") + 1,
                         "generalizes": prev.get("generalizes", []),
                         **describe(b)}
        _save_index(idx)
    return {"total": len(idx)}


def build() -> dict:
    """Compile the module so later proofs can `import Proofs.Verified` cheaply."""
    if not os.path.exists(PATH):
        return {"ok": True, "note": "library empty"}
    try:
        proc = subprocess.run([LAKE, "build", MODULE], cwd=config.LEAN_PROJECT,
                              capture_output=True, text=True, timeout=900)
        return {"ok": proc.returncode == 0,
                "errors": (proc.stdout + proc.stderr)[-400:] if proc.returncode else ""}
    except Exception as exc:
        return {"ok": False, "errors": str(exc)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="verified lemma library")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--search")
    a = ap.parse_args()
    if a.reindex:
        print(json.dumps(reindex(), indent=1))
    elif a.search:
        for h in search(a.search, k=8):
            print(f"  {h.get('score', '')} {h['name']:40s} {h['statement'][:80]}")
    else:
        print(json.dumps(stats(), indent=1))


# ---------------------------------------------------------------------------
# CARRYING CAPACITY.
#
# The library grows monotonically and the context window does not. That is an
# ecology with a birth rate and no death rate, and it has the ending such
# systems always have: measured here as a prompt reaching 28,700 of 32,768
# tokens and 35% of steps producing nothing at all.
#
# Retirement is therefore not tidiness, it is what keeps the run alive. Note
# that retiring is a RETRIEVAL decision only -- the proof stays in Verified.lean
# and the index record stays on disk. Nothing verified is ever destroyed; it
# simply stops competing for context it was not earning.
# ---------------------------------------------------------------------------

def fitness(rec: dict) -> float:
    """How much context this lemma has earned.

    Downstream citations dominate deliberately: a lemma another proof actually
    used is the only direct evidence of value this system produces. Everything
    else here is a weak prior for lemmas too young to have been cited yet.
    """
    uses = float(rec.get("uses", 0) or 0)
    # Generality proxy, in the minimum-description-length spirit: a short
    # statement covering the same ground is likelier to apply again. Weak, and
    # weighted as such.
    length = len(str(rec.get("statement", "")))
    brevity = 1.0 / (1.0 + length / 90.0)
    # A lemma that generalises others stands in for them, so it should outlive
    # them rather than be crowded out by them.
    breadth = 0.4 * len(rec.get("generalizes") or [])
    return 3.0 * uses + breadth + brevity


def retire(cap: int = 0, dry_run: bool = False) -> dict:
    """Keep the fittest `cap` lemmas retrievable; retire the rest.

    A lemma with a downstream use is NEVER retired regardless of cap -- the one
    hard-won signal of value in the whole system outranks a budget heuristic.
    """
    cap = cap or config.GT_LIBRARY_CAP
    with _lock:
        idx = _load_index()
        recs = sorted(idx.values(), key=lambda r: -fitness(r))
        keep, drop = [], []
        for i, r in enumerate(recs):
            if i < cap or (r.get("uses", 0) or 0) > 0:
                keep.append(r["name"])
            else:
                drop.append(r["name"])
        if not dry_run:
            for r in idx.values():
                r["retired"] = r["name"] in drop
            _save_index(idx)
    return {"kept": keep, "retired": drop, "cap": cap, "total": len(recs)}
