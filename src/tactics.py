"""Tactic pre-probe: let Lean try before the model does.

Measured on goals the model had already solved:

    sqrt2 irrational   349.8s via prover  ->  6.7s via `exact?`
    P6 gcd (pinned)     12.0s via prover  -> 30.5s via `exact?`
    2 | n(n+1)                            ->  7.0s via `exact?`

`exact?` searches Mathlib natively, so it finds the lemma the prover has to guess
at. It costs NO GPU, which matters when two models are already holding 36 GB.
Run it before spending prover attempts; fall through when it finds nothing.
"""

import concurrent.futures as futures
import os
import time

from . import config
from .lean import check_lean

# Ordered cheapest-plausible first. `exact?` is the workhorse -- it is Mathlib
# search, which is exactly the capability the prover lacks.
LADDER = [
    "exact?",
    "simp",
    "omega",
    "decide",
    "norm_num",
    "positivity",
    "aesop",
    "ring",
    "linarith",
    "nlinarith [sq_nonneg 1]",
    "field_simp; ring",
    "constructor <;> exact?",
]


# Every rung pays `import Mathlib` (~25s) on its own. Run SERIALLY, a 12-rung
# ladder needs ~300s and the 240s budget expires around rung 9 -- so the "cheap
# fast path" was spending its entire budget on imports and frequently never
# reached `nlinarith` at all.
#
# Lean checking is CPU-bound and these rungs are independent, so the ladder is
# embarrassingly parallel: wall-clock becomes the SLOWEST rung instead of the
# SUM. The real fix is a persistent REPL with a pickled post-import environment;
# this is the version that needs no new dependency.
#
# The width is bounded by MEMORY, not cores. Each rung runs `import Mathlib`,
# measured in this repo at ~5.7 GB resident per process. Twelve at once is ~68
# GB on top of ~36 GB of wired models -- the "speedup" becomes swap. Cores are
# the wrong limit to pick; RAM left over after the models is the right one.
TRANSIENT_OS_RESERVE_GB = float(os.environ.get("TRANSIENT_OS_RESERVE_GB", "4"))


def _default_workers() -> int:
    """Concurrent `lake env lean` processes the machine can actually afford.

    A `lake env lean` process is TRANSIENT -- it holds Mathlib for a few seconds
    and the memory comes straight back, so a brief overshoot is absorbed rather
    than sustained. A REPL worker holds the same memory indefinitely and needs
    real headroom. Same 5.7 GB, different risk, so different reserves.

    Empirically 2 workers ran the full 12-rung ladder in 46.5s on a 48 GB Mac
    with both models resident. Applying the REPL's conservative reserve here
    would have cut that to 1 and doubled the probe, so the floor below encodes
    the measurement rather than my estimate of it.

    When the persistent REPL is running this is mostly moot: every check queues
    against the REPL pool instead of spawning a process, so Lean's footprint is
    capped there rather than scaling with this number.
    """
    from . import repl
    total = repl.total_gb()
    if not total:
        return min(2, os.cpu_count() or 2)
    # Transient spikes need less OS headroom than a permanent resident does.
    spare = max(0.0, total - repl.MODEL_RESERVED_GB - TRANSIENT_OS_RESERVE_GB)
    afford = int(spare // repl.GB_PER_WORKER)
    floor = 2 if total >= 32 else 1
    return max(floor, min(8, os.cpu_count() or 4, max(afford, floor)))


WORKERS = int(os.environ.get("TACTIC_WORKERS", "0")) or _default_workers()


def probe(lean_statement: str, ladder: list = None, budget: float = 240.0) -> dict:
    """Try tactics against a pinned Lean signature. Returns a verified proof or none.

    `lean_statement` must be a signature WITHOUT `:=` (e.g. "theorem f (n : ℕ) : P n").
    Rungs run concurrently; the first success wins and the rest are abandoned.
    """
    sig = lean_statement.strip().rstrip(":=").rstrip()
    rungs = list(ladder or LADDER)
    started = time.time()

    def attempt(tac: str) -> tuple:
        code = f"import Mathlib\nset_option maxHeartbeats 400000\n\n{sig} := by {tac}\n"
        return tac, code, check_lean(code)

    tried, winner = [], None
    with futures.ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        jobs = {pool.submit(attempt, t): t for t in rungs}
        try:
            for fut in futures.as_completed(jobs, timeout=budget):
                try:
                    tac, code, verdict = fut.result()
                except Exception:
                    continue
                tried.append(tac)
                if config.PROVER_TRACE:
                    print(f"         [{len(tried):>2}/{len(rungs)}] {tac:<22} "
                          f"{'SOLVED' if verdict['ok'] else 'no'}", flush=True)
                if verdict["ok"]:
                    winner = {"ok": True, "lean": code, "tactic": tac}
                    break
        except futures.TimeoutError:
            pass
        finally:
            for fut in jobs:
                fut.cancel()

    elapsed = round(time.time() - started, 1)
    if winner:
        # Report the ladder honestly: `tried` is what COMPLETED, and under
        # concurrency that is not a prefix of LADDER. The prover's "do not retry
        # these" hint is built from it, so it must not claim rungs that never ran.
        return {**winner, "tried": tried, "elapsed": elapsed}
    return {"ok": False, "tried": tried,
            "not_run": [t for t in rungs if t not in tried], "elapsed": elapsed}
