"""Lean 4 verification tool for the research agent.

Exposes check_lean(code) -> {ok, errors, elapsed}. Designed to be wired
directly into an LLM tool-calling loop: the `errors` string is what you
feed back to the prover model for self-correction.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))  # this Lake project
LAKE = shutil.which("lake") or "/opt/homebrew/bin/lake"
TIMEOUT = 120

# Full `import Mathlib` costs ~25s; narrow imports ~1.2s. Encourage narrow.
_FULL_IMPORT = re.compile(r"^\s*import\s+Mathlib\s*$", re.MULTILINE)

# `sorry` compiles with exit 0 and only a warning — must be caught explicitly,
# or stubbed-out proofs get recorded as verified.
_SORRY = re.compile(r"\bsorry\b|\bsorryAx\b|declaration uses 'sorry'")

# An empty file is valid Lean and compiles with exit 0. Require a real declaration.
_DECL = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)


def check_lean(code: str, timeout: int = TIMEOUT) -> dict:
    """Compile a Lean 4 snippet against Mathlib. Returns ok/errors/elapsed."""
    started = time.time()

    # An empty or declaration-free file compiles cleanly, so without this guard
    # a model that returns nothing scores as a proof.
    if not _DECL.search(code or ""):
        return {"ok": False, "elapsed": 0.0,
                "errors": "NO DECLARATION: expected a `theorem`/`lemma`/`example`; got nothing to verify."}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=PROJECT, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        path = fh.name

    try:
        proc = subprocess.run(
            [LAKE, "env", "lean", os.path.basename(path)],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()
        ok = proc.returncode == 0 and "error" not in output

        # Lean reports `sorry` as a WARNING with exit 0, so a proof stubbed out
        # with `sorry` would otherwise be recorded as verified. Reject it.
        if ok and _SORRY.search(output + "\n" + code):
            ok = False
            output = "PROOF INCOMPLETE: uses `sorry`. Produce a complete proof with no `sorry`."

        result = {"ok": ok, "errors": "" if ok else output, "elapsed": round(time.time() - started, 2)}
    except subprocess.TimeoutExpired:
        result = {"ok": False, "errors": f"timed out after {timeout}s", "elapsed": timeout}
    finally:
        os.unlink(path)

    if result["ok"] and _FULL_IMPORT.search(code):
        result["hint"] = "used `import Mathlib` (~25s); a narrow import runs ~20x faster"
    return result


# Tool schema for the agent's tool-calling loop.
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "lean_check",
        "description": (
            "Compile Lean 4 code against Mathlib and return compiler errors. "
            "Use narrow imports (e.g. Mathlib.Algebra.Ring.Parity) rather than "
            "`import Mathlib`, which is ~20x slower."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete Lean 4 source"}},
            "required": ["code"],
        },
    },
}


if __name__ == "__main__":
    good = "import Mathlib.Algebra.Ring.Parity\n\ntheorem t (n : ℕ) : 2 ∣ n * (n + 1) :=\n  (Nat.even_mul_succ_self n).two_dvd\n"
    bad = "import Mathlib.Algebra.Ring.Parity\n\ntheorem t (n : ℕ) : 2 ∣ n * (n + 1) := by\n  exact?\n"
    for name, src in (("valid", good), ("invalid", bad)):
        print(f"--- {name} ---")
        print(json.dumps(check_lean(src), indent=2)[:600])
