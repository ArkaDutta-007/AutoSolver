"""Persistent Lean REPL: pay `import Mathlib` once per process, not per check.

Measured in this repo: ~25-29s with `import Mathlib`, ~1.2-1.9s narrow. That
cost is paid PER CHECK by `lake env lean`, and it now dominates in three
separate places:

  * signature validation            1 import
  * derive_signature recovery       up to 3 imports
  * the 12-rung tactic ladder       12 imports, /WORKERS

so agreeing on a single signature costs ~2 minutes before a prover token is
spent, and one prove_theorem call can spend 12 minutes in `lake` startup alone.

leanprover-community/repl keeps a Lean process alive and hands back an
environment id. Import Mathlib once into env 0, then run every later check
against `{"cmd": ..., "env": 0}` and the import is free. Same compiler, same
kernel, same verdicts -- only the startup is amortised.

Falls back silently to the subprocess path when the REPL is not built, so this
is safe to ship before `scripts/setup_repl.sh` has ever been run.
"""

import json
import os
import queue
import subprocess
import threading
import time

from . import config

# `repl` speaks JSON objects separated by BLANK lines, not newlines.
_TERM = "\n\n"

# Measured in this repo: `import Mathlib` costs ~5.7 GB resident (vs ~1.3 GB for
# a narrow import). This is the number every sizing decision here turns on.
GB_PER_WORKER = float(os.environ.get("LEAN_GB_PER_WORKER", "5.7"))
# What the resident models hold. Default matches the 40 GB wired limit; override
# when inference lives on the lab server and the Mac is only running Lean.
MODEL_RESERVED_GB = float(os.environ.get("MODEL_RESERVED_GB", "40"))
# macOS itself, plus whatever else is open.
OS_RESERVED_GB = float(os.environ.get("OS_RESERVED_GB", "8"))


def total_gb() -> float:
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def _free_gb() -> float:
    t = total_gb()
    return max(0.0, t - MODEL_RESERVED_GB - OS_RESERVED_GB) if t else 0.0


def _affordable_workers() -> int:
    t = total_gb()
    if not t:                                  # unknown machine: be conservative
        return 1
    return int(_free_gb() // GB_PER_WORKER)


class Repl:
    """One persistent Lean process with Mathlib pre-imported."""

    def __init__(self, cmd: list, cwd: str, preamble: str, boot_timeout: int = 300):
        self.cmd, self.cwd, self.preamble = cmd, cwd, preamble
        self.boot_timeout = boot_timeout
        self.proc = None
        self.env = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            from .sandbox import clean_env, lean_argv
            self.proc = subprocess.Popen(
                lean_argv(self.cmd, self.cwd), cwd=self.cwd, env=clean_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        except Exception:
            self.proc = None
            return False
        # Import once. Everything after this is free.
        out = self._exchange({"cmd": self.preamble}, timeout=self.boot_timeout)
        if out is None or "env" not in out:
            self.stop()
            return False
        self.env = out["env"]
        return True

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc, self.env = None, None

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def _exchange(self, payload: dict, timeout: int) -> dict:
        """Write one command, read one JSON object. None on any failure."""
        if not self.alive():
            return None
        try:
            self.proc.stdin.write(json.dumps(payload) + _TERM)
            self.proc.stdin.flush()
        except Exception:
            return None

        # A hung Lean must not hang the agent: read on a worker and time out.
        box: queue.Queue = queue.Queue(maxsize=1)

        def reader():
            buf = []
            try:
                for line in self.proc.stdout:
                    if line.strip() == "":
                        if buf:
                            break
                        continue
                    buf.append(line)
            except Exception:
                pass
            try:
                box.put_nowait("".join(buf))
            except queue.Full:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            raw = box.get(timeout=timeout)
        except queue.Empty:
            self.stop()                      # poisoned; the pool will respawn
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def check(self, code: str, timeout: int) -> dict:
        """Elaborate `code` against the pre-imported environment."""
        with self._lock:
            if not self.alive() and not self.start():
                return None
            body = _strip_header(code)
            out = self._exchange({"cmd": body, "env": self.env}, timeout=timeout)
        if out is None:
            return None
        return _verdict(out)


def _strip_header(code: str) -> str:
    """Drop import/set_option lines -- env 0 already has them."""
    keep = [ln for ln in (code or "").splitlines()
            if not ln.lstrip().startswith(("import ", "set_option maxHeartbeats"))]
    return "\n".join(keep).strip()


def _verdict(out: dict) -> dict:
    """Translate a repl response into the shape check_lean already returns."""
    msgs = out.get("messages") or []
    errors = [m for m in msgs if m.get("severity") == "error"]
    # `sorries` is structured, which is strictly better than grepping stdout --
    # it is also the goal-state channel that per-lemma refinement will need.
    sorries = out.get("sorries") or []

    def fmt(m):
        pos = m.get("pos") or {}
        return (f"repl:{pos.get('line', '?')}:{pos.get('column', '?')}: "
                f"{m.get('severity')}: {m.get('data', '')}")

    text = "\n".join(fmt(m) for m in msgs)
    if sorries:
        text += ("\n" if text else "") + "\n".join(
            f"repl:{(s.get('pos') or {}).get('line','?')}:0: warning: "
            f"declaration uses 'sorry'; goal: {s.get('goal', '')}" for s in sorries)
    return {"ok": not errors and not sorries, "errors": "" if (not errors and not sorries) else text,
            "sorries": sorries, "messages": msgs}


class Pool:
    """A few REPLs, handed out round-robin. Lean checking is CPU-bound."""

    def __init__(self):
        self._repls: list = []
        self._free: queue.Queue = queue.Queue()
        self._started = False
        self._lock = threading.Lock()
        self.disabled = False

    def _spec(self):
        """Where the repl binary is, and how to launch it. None if unavailable."""
        exe = config.LEAN_REPL_PATH
        if exe and os.path.exists(exe):
            return [exe], config.LEAN_PROJECT
        guess = os.path.join(config.LEAN_PROJECT, ".lake", "build", "bin", "repl")
        if os.path.exists(guess):
            return [guess], config.LEAN_PROJECT
        return None

    def ensure(self) -> bool:
        with self._lock:
            if self.disabled:
                return False
            if self._started:
                return bool(self._repls)
            self._started = True
            spec = self._spec()
            if not spec or config.LEAN_REPL == "off":
                self.disabled = True
                return False
            cmd, cwd = spec
            # Refuse to start if the machine cannot afford it. A worker holds
            # ~5.7 GB of imported Mathlib; on a 48 GB Mac already wiring 36 GB of
            # models that leaves ~6 GB for macOS, and the resulting swap costs
            # more than the imports ever did -- including on inference, because
            # the memory is unified. Better to stay on the subprocess path and
            # say why than to quietly make everything slower.
            want = max(1, config.LEAN_REPL_WORKERS)
            afford = _affordable_workers()
            if afford < 1:
                self.disabled = True
                print(f"[lean repl] NOT starting: needs ~{GB_PER_WORKER} GB and only "
                      f"~{_free_gb():.0f} GB is free after models. Falling back to "
                      f"`lake env lean`. Move inference to the lab server, or set "
                      f"MODEL_RESERVED_GB if this estimate is wrong.", flush=True)
                return False
            if afford < want:
                print(f"[lean repl] reducing {want} -> {afford} worker(s) to fit "
                      f"~{_free_gb():.0f} GB free", flush=True)
                want = afford
            preamble = "import Mathlib\nset_option maxHeartbeats 400000"
            t0 = time.time()
            for _ in range(want):
                r = Repl(cmd, cwd, preamble)
                if r.start():
                    self._repls.append(r)
                    self._free.put(r)
            if not self._repls:
                self.disabled = True
                return False
            print(f"[lean repl] {len(self._repls)} worker(s) up with Mathlib "
                  f"preloaded in {time.time() - t0:.0f}s -- imports are now free",
                  flush=True)
            return True

    def check(self, code: str, timeout: int) -> dict:
        if not self.ensure():
            return None
        try:
            r = self._free.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            self._last_use = time.time()
            out = r.check(code, timeout)
        finally:
            self._last_use = time.time()
            if not r.alive():
                r.start()                    # respawn a poisoned worker
            self._free.put(r)
        return out

    # --- idle reaping -------------------------------------------------------
    # A worker holds ~5.7 GB of imported Mathlib. Holding that through a long
    # stretch of pure model inference -- which is most of a research run -- is
    # rent for nothing. Release it when idle; the next check pays one import to
    # bring it back, which is the cost we used to pay on EVERY check anyway.
    def _reaper(self) -> None:
        idle = config.LEAN_REPL_IDLE_SECONDS
        while True:
            time.sleep(min(60, max(5, idle // 4)))
            with self._lock:
                if not self._repls or time.time() - self._last_use < idle:
                    continue
                drained = []
                try:
                    while True:
                        drained.append(self._free.get_nowait())
                except queue.Empty:
                    pass
                if len(drained) != len(self._repls):     # someone is mid-check
                    for r in drained:
                        self._free.put(r)
                    continue
                for r in drained:
                    r.stop()
                self._repls = []
                self._started = False                    # ensure() will rebuild
                print(f"[lean repl] idle {idle}s -- released "
                      f"~{len(drained) * 5.7:.1f} GB; will reload on next check",
                      flush=True)

    def shutdown(self) -> None:
        with self._lock:
            for r in self._repls:
                r.stop()
            self._repls = []
            self._started = False


POOL = Pool()
