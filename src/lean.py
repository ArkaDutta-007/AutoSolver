"""Lean 4 verification. The only source of ground truth in this system."""

import os
import re
import shutil
import subprocess
import tempfile
import time

from . import config
from .sandbox import SANDBOX, clean_env, lean_argv

LAKE = shutil.which("lake") or "/opt/homebrew/bin/lake"

# `import Mathlib` costs ~25s; a narrow module ~1.2s.
_FULL_IMPORT = re.compile(r"^\s*import\s+Mathlib\s*$", re.MULTILINE)

# `sorry` compiles with exit 0 and only a WARNING. Without this check, stubbed
# proofs are recorded as verified.
_SORRY = re.compile(r"\bsorry\b|\bsorryAx\b|\badmit\b|declaration uses 'sorry'")

# An empty or declaration-free file is valid Lean and compiles cleanly, so a
# model returning nothing would otherwise score as a proof.
_DECL = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)

# A real Lean diagnostic is `file.lean:LINE:COL: error: ...`. Testing for the
# bare substring "error" instead marks a proof failed whenever the word appears
# anywhere in the output -- including inside a `#print axioms` result, a
# docstring, or a lemma named `*_error_*`.
_ERROR_LINE = re.compile(r"^\S*\.lean:\d+:\d+: error:", re.MULTILINE)

_DECL_NAME = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)", re.MULTILINE)
_AXIOM_LINE = re.compile(r"'[^']+' depends on axioms: \[([^\]]*)\]")


_TOOL: dict = {}


def _lean(path: str):
    """argv and env to compile `path`, sandboxed where macOS allows it.

    Inside the sandbox lean runs directly, with the search path lake would
    give it resolved once outside: lake's own git checks fail in there and
    it sets about deleting Mathlib to re-clone it (seen; the sandbox refused).
    """
    name = os.path.basename(path)
    if not os.path.exists(SANDBOX):
        return [LAKE, "env", "lean", name], clean_env()
    if not _TOOL:
        def ask(*a):
            return subprocess.run([LAKE, "env", *a], cwd=config.LEAN_PROJECT, capture_output=True,
                                  text=True, timeout=300, env=clean_env()).stdout.strip()
        _TOOL.update(bin=os.path.join(ask("lean", "--print-prefix"), "bin", "lean"),
                     path=ask("printenv", "LEAN_PATH"))
    env = {"PATH": "/usr/bin:/bin", "HOME": tempfile.gettempdir(), "LEAN_PATH": _TOOL["path"]}
    return lean_argv([_TOOL["bin"], name], config.LEAN_PROJECT), env


def check_lean(code: str, timeout: int = None) -> dict:
    """Compile a Lean 4 snippet against Mathlib. Returns ok/errors/elapsed."""
    timeout = timeout or config.LEAN_TIMEOUT
    started = time.time()

    if not _DECL.search(code or ""):
        return {"ok": False, "elapsed": 0.0,
                "errors": "NO DECLARATION: expected a `theorem`/`lemma`/`example`; "
                          "got nothing to verify."}

    # Try the persistent REPL first -- same compiler, same kernel, but Mathlib is
    # already imported. Returns None when it is not built or the worker died, in
    # which case we fall through to `lake env lean` exactly as before.
    if config.LEAN_REPL != "off":
        try:
            from . import repl as _repl
            hit = _repl.POOL.check(code, timeout)
        except Exception:
            hit = None
        if hit is not None:
            hit["elapsed"] = round(time.time() - started, 2)
            if hit["ok"] and _SORRY.search(code or ""):
                hit["ok"] = False
                hit["errors"] = ("PROOF INCOMPLETE: uses `sorry`/`admit`. "
                                 "Produce a complete proof.")
            return hit

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", dir=config.LEAN_PROJECT,
                                     delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name

    try:
        argv, env = _lean(path)
        proc = subprocess.run(argv, env=env,
                              cwd=config.LEAN_PROJECT, capture_output=True,
                              text=True, timeout=timeout)
        output = (proc.stdout + proc.stderr).strip()
        ok = proc.returncode == 0 and not _ERROR_LINE.search(output)
        if ok and _SORRY.search(output + "\n" + code):
            ok = False
            output = "PROOF INCOMPLETE: uses `sorry`/`admit`. Produce a complete proof."
        result = {"ok": ok, "errors": "" if ok else output,
                  "elapsed": round(time.time() - started, 2)}
    except subprocess.TimeoutExpired:
        result = {"ok": False, "errors": f"timed out after {timeout}s", "elapsed": timeout}
    finally:
        os.unlink(path)

    if result["ok"] and _FULL_IMPORT.search(code):
        result["hint"] = "used `import Mathlib` (~25s); a narrow import runs ~20x faster"
    return result


def audit_axioms(code: str, timeout: int = None) -> dict:
    """Reject proofs that lean on axioms beyond Lean's standard three.

    `sorry` and empty files are already blocked, which closes the two obvious
    holes. This closes the next one: a proof can also "succeed" by introducing
    its own `axiom`, or by `native_decide` (which bypasses the kernel and trusts
    the compiler). Both compile. Neither is a proof.

    Fails OPEN -- if the audit itself cannot run, the proof is not rejected,
    because a broken check must not manufacture false negatives.
    """
    code = code or ""
    bad, found = set(), set()

    # Static checks first: these need no compiler, so they still fire when Lean
    # is slow, unavailable, or the project path is wrong.
    if re.search(r"\bnative_decide\b", code):
        bad.add("native_decide (bypasses the kernel)")
    if re.search(r"^\s*axiom\s+", code, re.MULTILINE):
        bad.add("locally declared `axiom`")

    names = _DECL_NAME.findall(code)
    if names and not bad:
        probe = code.rstrip() + "\n\n" + "\n".join(f"#print axioms {n}" for n in names) + "\n"
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lean",
                                             dir=config.LEAN_PROJECT, delete=False,
                                             encoding="utf-8") as fh:
                fh.write(probe)
                path = fh.name
            argv, env = _lean(path)
            proc = subprocess.run(argv, env=env,
                                  cwd=config.LEAN_PROJECT, capture_output=True,
                                  text=True, timeout=timeout or config.LEAN_TIMEOUT)
            output = proc.stdout + proc.stderr
        except Exception as exc:
            # Fail OPEN. A check that cannot run must not manufacture a false
            # negative -- rejecting real proofs is worse than missing a rare cheat.
            return {"ok": True, "checked": [], "note": f"axiom audit unavailable: {exc}"}
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        for group in _AXIOM_LINE.findall(output):
            for ax in (a.strip() for a in group.split(",") if a.strip()):
                found.add(ax)
                if ax not in config.ALLOWED_AXIOMS:
                    bad.add(ax)

    if bad:
        return {"ok": False, "axioms": sorted(found), "disallowed": sorted(bad),
                "errors": ("PROOF UNSOUND: depends on "
                           f"{', '.join(sorted(bad))}. Allowed axioms are "
                           f"{', '.join(sorted(config.ALLOWED_AXIOMS))}. "
                           "Prove it without them.")}
    return {"ok": True, "axioms": sorted(found), "checked": names}


def split_signature(sig: str) -> tuple:
    """Split `(a : T) (h : P) : Concl` into (binders, conclusion).

    The separating colon is the first at BRACKET DEPTH ZERO. Splitting on the
    LAST colon breaks on any conclusion containing one (`∀ x : T, ...`), and
    splitting on the FIRST breaks on every binder -- a bug that silently
    refused every instance repair, because the binder text it checked against
    was truncated to `theorem t (α`.
    """
    depth = 0
    for i, ch in enumerate(sig):
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif ch == ":" and depth == 0:
            return sig[:i].strip(), sig[i + 1:].strip()
    return "", sig.strip()


def statement_ok(verdict: dict) -> bool:
    """True when Lean's only complaint is the `sorry` we injected.

    A signature is checked by compiling it with `:= by sorry`, so the expected
    result is an error mentioning sorry and nothing else. This test was written
    out by hand in four places (prover twice, formalize_score, formalize_bench)
    and they had already drifted -- one of them omitted the "besides sorry"
    clause entirely and accepted statements with real errors alongside it.
    """
    errs = str(verdict.get("errors", "") or "")
    if verdict.get("ok"):
        return True
    return ("sorry" in errs.lower()
            and not any("error:" in ln and "sorry" not in ln.lower()
                        for ln in errs.splitlines()))
