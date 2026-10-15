"""Proof cache. Re-proving a claim is pure waste in a perpetual run."""

import hashlib
import json
import os
import re
import threading

from . import config

PATH = os.path.join(config.ROOT, "runs", "proof_cache.json")
_lock = threading.Lock()
_mem: dict = {}
_loaded = False


_WS = re.compile(r"\s+")
_LATEX_NOISE = re.compile(r"\\(?:displaystyle|text|mathrm|left|right|,|;|!)\b|[{}$]")


def _normalize(text: str) -> str:
    """Collapse the cosmetic drift that made this cache miss on identical claims.

    The same P1 statement was submitted as `2026`, `{2026}`, and
    `\\displaystyle \\gcd(m,n)` across sessions. Keyed on the raw string those are
    three different claims and each one is re-proved from scratch.
    """
    # Trailing punctuation counts as drift too: the smoke test produced two
    # separate cache entries for the same claim, differing only by a full stop.
    return _WS.sub(" ", _LATEX_NOISE.sub("", (text or "").lower())).strip().rstrip(".;, ")


def _key(statement: str, lean_statement: str = "") -> str:
    # The Lean signature is normalized for whitespace only -- it is code, and
    # `{}` is meaningful there (implicit binders).
    sig = _WS.sub(" ", (lean_statement or "").strip())
    return hashlib.sha256(f"{_normalize(statement)}||{sig}".encode()).hexdigest()[:32]


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if os.path.exists(PATH):
        try:
            with open(PATH, encoding="utf-8") as fh:
                _mem.update(json.load(fh))
        except Exception:
            pass


def get(statement: str, lean_statement: str = "") -> dict:
    _load()
    return _mem.get(_key(statement, lean_statement)) or {}


def put(statement: str, result: dict, lean_statement: str = "") -> None:
    """Only successes are cached -- a failure may succeed on a later run."""
    if not result.get("proved"):
        return
    _load()
    with _lock:
        _mem[_key(statement, lean_statement)] = {
            "lean": result.get("lean"), "solved_by": result.get("solved_by"),
            "elapsed": result.get("elapsed"), "statement": statement[:300]}
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_mem, fh, indent=1)
        os.replace(tmp, PATH)


def stats() -> dict:
    _load()
    return {"cached_proofs": len(_mem), "path": PATH}
