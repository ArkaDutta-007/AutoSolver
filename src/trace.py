"""Append-only record of every Lean-adjudicated attempt. The training set.

WHY THIS EXISTS
---------------
Measured, not assumed: eight runs and roughly ten hours of compute across five
IMO problems left behind exactly TWENTY retained records -- 17 `exhausted` and
3 `established` lines in the journals. `grep` finds no `json.dump` anywhere in
prover.py. Every signature derivation, every tactic probe, every proof attempt
and its compiler verdict was built in memory, returned to the agent, and then
dropped on the floor.

That is the expensive half of the system throwing away the cheap half's whole
reason for existing. A Lean verdict is a FREE, CORRECT, non-noisy label -- the
exact supervision that expert iteration, STP-style self-play conjecturing, and
every calibration objective in the literature are built on. None of them can be
attempted here today, not because the machine is too small, but because the
system keeps no memory of what it tried.

So this module does one thing and does it without opinions: it writes what
happened. It does NOT decide what is interesting, does not filter, does not
score. Those are later choices, and later choices are cheap to change; data you
failed to write down at the time is gone.

DESIGN CONSTRAINTS
------------------
  * NEVER raise. A recorder that can break a fifteen-hour run is worse than no
    recorder. Every public function swallows everything.
  * NEVER block. One O_APPEND line write, no read-modify-write, so a crash
    mid-run costs the last line and nothing else, and two processes can write
    the same file without a lock.
  * NEVER grow the prompt. Nothing here is read back into the model's context.

WHAT A CONSUMER GETS
--------------------
Each line is one attempt with its verdict, so the file supports, with no
further instrumentation:

  * expert iteration  -- filter ok=True, you have (statement, proof) pairs
  * calibration       -- pair `p_success` against `ok` for a Brier score
  * curriculum        -- per-statement pass rate finds the "barely provable"
                         frontier that STP trains its conjecturer toward
  * library R0        -- how many later proofs a lemma actually enabled
  * cost accounting   -- seconds per verdict, by mode and by model
"""

import json
import os
import time

from . import config

PATH = os.path.join(config.ROOT, "traces", "attempts.jsonl")

# Proofs and error dumps can be long, and a trace file that becomes unreadable
# is a trace file nobody consults. These caps keep a line inspectable by eye
# while still holding enough to train on.
_MAX_CODE = 8000
_MAX_TEXT = 2000

_RUN = {"id": "", "goal": ""}


def begin(run_id: str = "", goal: str = "") -> None:
    """Tag subsequent records with the run and problem they belong to.

    Without this every line is an orphan: pass rates could still be computed,
    but nothing could be attributed to a problem, and the per-problem frontier
    is the whole point of a curriculum.
    """
    try:
        _RUN["id"] = str(run_id or _RUN["id"] or int(time.time()))
        _RUN["goal"] = str(goal or "")[:400]
    except Exception:
        pass


def _needs_newline() -> bool:
    """True when the file ends mid-line, i.e. its last writer was killed."""
    try:
        with open(PATH, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except Exception:
        return False


def _clip(v, n):
    if isinstance(v, str):
        return v[:n]
    return v


def record(kind: str, **fields) -> None:
    """Append one attempt. Swallows everything -- see module docstring."""
    try:
        row = {"t": round(time.time(), 3), "kind": str(kind),
               "run": _RUN["id"], "goal": _RUN["goal"]}
        for k, v in fields.items():
            if v is None:
                continue
            row[k] = _clip(v, _MAX_CODE if k in ("lean", "code") else _MAX_TEXT)
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, default=str)
        # O_APPEND on a single write() under the pipe-buffer size is atomic on
        # macOS and Linux, so concurrent writers interleave lines but never
        # corrupt one. That is why this is a bare append and not a rewrite.
        with open(PATH, "a", encoding="utf-8") as fh:
            # Heal a truncated tail before appending. A writer killed mid-line
            # leaves no trailing newline, and appending straight onto it fuses
            # the corpse to the next record -- so one crash would cost TWO rows,
            # the second of them a perfectly good one written afterwards. Caught
            # by test_a_corrupt_line_does_not_lose_the_file, not by inspection.
            if fh.tell() and _needs_newline():
                fh.write("\n")
            fh.write(line + "\n")
    except Exception:
        pass


def proof(statement: str, out: dict, lean_statement: str = "") -> None:
    """One `prove_theorem` call, flattened with its per-attempt history.

    The history entries are where the signal density is: a call that failed
    after four attempts says far more about the frontier than a bare False.
    """
    try:
        hist = out.get("attempts") or []
        record("proof",
               statement=str(statement)[:_MAX_TEXT],
               pinned=bool(lean_statement),
               lean_statement=str(lean_statement)[:_MAX_TEXT] or None,
               ok=bool(out.get("proved")),
               elapsed=out.get("elapsed"),
               rounds=out.get("rounds"),
               redundant=bool(out.get("redundant")) or None,
               note=str(out.get("note", ""))[:_MAX_TEXT] or None,
               lean=out.get("lean"),
               n_attempts=len(hist),
               history=[{k: _clip(v, 600) for k, v in h.items()
                         if k in ("attempt", "model", "mode", "ok", "reason",
                                  "rounds", "strategy")}
                        for h in hist][:12])
    except Exception:
        pass


def signature(statement: str, sig: str, ok: bool, seconds: float = 0.0,
              err: str = "", rejected: str = "", repaired: bool = False) -> None:
    """One signature derivation. The highest-volume labelled event in the system.

    Measured at 40% per attempt on four of five stability cases, which is
    exactly the regime where a learned prior would pay for itself -- and
    exactly the data that was being discarded.
    """
    record("signature", statement=str(statement)[:_MAX_TEXT],
           sig=str(sig or "")[:_MAX_TEXT], ok=bool(ok),
           elapsed=round(float(seconds or 0.0), 1),
           err=str(err or "")[:600] or None,
           rejected=str(rejected or "")[:_MAX_TEXT] or None,
           repaired=bool(repaired) or None)


def load(path: str = "") -> list:
    """Every well-formed line. Skips corrupt ones rather than failing the read.

    A half-written final line is the expected state of an append-only file whose
    writer was killed, and refusing to read the other 4,000 rows because of it
    would make the file useless precisely after the crash worth studying.
    """
    rows = []
    try:
        with open(path or PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def stats(path: str = "") -> dict:
    """Counts a consumer needs before deciding anything is trainable."""
    rows = load(path)
    by = {}
    for r in rows:
        k = r.get("kind", "?")
        d = by.setdefault(k, {"n": 0, "ok": 0, "secs": 0.0})
        d["n"] += 1
        d["ok"] += 1 if r.get("ok") else 0
        d["secs"] += float(r.get("elapsed") or 0.0)
    for d in by.values():
        d["rate"] = round(d["ok"] / d["n"], 3) if d["n"] else 0.0
        d["secs"] = round(d["secs"], 1)
    return {"rows": len(rows), "runs": len({r.get("run") for r in rows}),
            "by_kind": by}
