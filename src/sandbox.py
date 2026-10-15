"""A calculator for the agent.

The agent had Lean and a journal and NOTHING ELSE -- no way to compute anything.
That is why it went straight from reading IMO 2026 P1 to trying to reason the
invariant out in its head.

For P1 the empirical route is ~15 lines: enumerate small boards, apply random
moves to a fixpoint, and watch what is conserved. The invariant falls out in
seconds, and only THEN is it worth formalizing. Observe -> conjecture -> prove is
how the problem is actually done; the system could only ever do the third step.

Deliberately boring: subprocess, hard timeout, output capped. The code is a
model's, so on macOS it runs under Seatbelt (sandbox-exec): no network, no
fork or spawn, no Mach lookups or Apple Events (LaunchServices would open a
URL for it), writes only in its own throwaway directory, and nothing under
the home directory readable except the interpreter's prefix. Elsewhere it is
a plain subprocess with a scrubbed environment -- run it in a container there.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

SANDBOX = "/usr/bin/sandbox-exec"
POLICY = """(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny mach-lookup)
(deny appleevent-send)
(deny file-write*)
(allow file-write* (subpath "{tmp}"))
(allow file-write-data (literal "/dev/null") (literal "/dev/zero")
       (regex #"^/dev/tty") (regex #"^/dev/fd/") (literal "/dev/dtracehelper"))
(deny file-read-data (subpath "{home}"))
(allow file-read-data (subpath "{prefix}") (subpath "{base}") (subpath "{tmp}"))
"""


# Lean runs model-written code too: #eval, run_cmd and any elab can do IO at
# compile time. Same walls, except it may spawn (lake starts lean; children
# inherit the sandbox) and reads the toolchain and the Lean project.
LEAN_POLICY = """(version 1)
(allow default)
(deny network*)
(deny mach-lookup)
(deny appleevent-send)
(deny file-write*)
(allow file-write* (subpath "{tmp}"))
(allow file-write-data (literal "/dev/null") (literal "/dev/zero")
       (regex #"^/dev/tty") (regex #"^/dev/fd/") (literal "/dev/dtracehelper"))
(deny file-read-data (subpath "{home}"))
(allow file-read-data (subpath "{elan}") (subpath "{project}") (subpath "{tmp}"))
"""


def clean_env(**extra) -> dict:
    """This process's environment minus anything named like a credential:
    a Lean #eval can read its environment and print it back to the model."""
    import re
    env = {k: v for k, v in os.environ.items()
           if not re.search(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH", k, re.I)}
    env.update(extra)
    return env


def lean_argv(argv: list, project: str) -> list:
    """`argv` wrapped for Lean; unchanged where there is no sandbox-exec."""
    if not os.path.exists(SANDBOX):
        return list(argv)
    real = os.path.realpath
    return [SANDBOX, "-p", LEAN_POLICY.format(
        tmp=real(tempfile.gettempdir()), home=real(os.path.expanduser("~")),
        elan=real(os.path.expanduser("~/.elan")), project=real(project))] + list(argv)


def _argv(path: str, tmp: str) -> list:
    argv = [sys.executable, "-I", path]
    if not os.path.exists(SANDBOX):
        return argv
    real = os.path.realpath           # a rule on a symlink never matches
    return [SANDBOX, "-p", POLICY.format(
        tmp=real(tmp), home=real(os.path.expanduser("~")),
        prefix=real(sys.prefix), base=real(sys.base_prefix))] + argv

TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "60"))
MAX_OUTPUT = 12_000

PRELUDE = textwrap.dedent("""
    import math, itertools, functools, fractions, random, collections, re
    from fractions import Fraction
    from math import gcd, lcm, factorial, comb, perm, isqrt
    random.seed(0)          # deterministic: a conjecture that only holds for one
                            # seed is not a conjecture
    try:
        import sympy
        from sympy import *
    except ImportError:
        pass
    try:
        import numpy as np
    except ImportError:
        pass
""")


def run_python(code: str, timeout: int = None) -> dict:
    """Execute Python for numerical/symbolic exploration. Returns stdout+stderr.

    Use it to TEST a conjecture on small cases before spending prover time on it.
    A conjecture that fails at n=4 is not worth formalizing.
    """
    timeout = min(int(timeout or TIMEOUT), 300)
    if not (code or "").strip():
        return {"ok": False, "error": "empty code"}

    tmp = tempfile.mkdtemp(prefix="autosolver-py-")
    path = os.path.join(tmp, "run.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PRELUDE + "\n" + code)

    try:
        proc = subprocess.run(
            _argv(path, tmp),
            capture_output=True, text=True, timeout=timeout,
            # No network credentials, no repo access, no surprises.
            cwd=tmp,
            env={"PATH": os.environ.get("PATH", ""), "HOME": tmp, "TMPDIR": tmp,
                 "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                 # headless plots: the macOS backend puts a Python icon in the
                 # Dock, and plt.show() would block until the timeout
                 "MPLBACKEND": "Agg"},
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        truncated = len(out) > MAX_OUTPUT
        return {"ok": proc.returncode == 0,
                "stdout": out[:MAX_OUTPUT],
                "stderr": err[-2000:],
                "truncated": truncated,
                "exit_code": proc.returncode}
    except subprocess.TimeoutExpired as exp:
        # Hand back whatever it printed BEFORE hanging. Returning only "timed
        # out" tells the model nothing about where, so it rewrites the same
        # shape of loop and times out again -- observed twice in a row on
        # IMO26_P1, 120s for no information. The partial output usually shows
        # the last state reached, which is exactly the diagnosis.
        partial = (exp.stdout or "")
        errs = (exp.stderr or "")
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        if isinstance(errs, bytes):
            errs = errs.decode("utf-8", "replace")
        partial, errs = partial.strip(), errs.strip()
        made_progress = bool(partial)
        return {"ok": False, "error": f"timed out after {timeout}s",
                "stdout": partial[:MAX_OUTPUT], "stderr": errs[-2000:],
                "printed_before_timeout": made_progress,
                "hint": ("Output stopped mid-run -- look at the last line printed "
                         "to see where it hung. If your loop never reached a new "
                         "state, it is not slow, it is not terminating: check that "
                         "you mutate the data the loop condition reads."
                         if made_progress else
                         "Nothing was printed before the timeout. Either the loop "
                         "never terminates, or the search range is far too large -- "
                         "print inside the loop so the next run shows where it hangs. "
                         "This tool is for SMALL cases.")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
