"""The implicit reward: what one step was worth, read off what already happened.

No grader and no reference answer. Every signal below is something the loop
sees anyway -- Lean's verdict, the checklist moving, a computation running
clean, a dead end recorded, a wasted turn -- so it costs nothing to compute and
the model cannot talk its way into it. Two jobs:

  * READ: printed and logged per step, so a run is a return curve rather than
    a wall of text, and the JSONL is a (state, action, reward) trace a later
    training pass can use without re-labelling anything;
  * STEER: when the recent return goes flat the loop says so, names what was
    missing, and brings the pass@K proposals forward -- the same "don't ask,
    change the options" lever the loop already pulls on a step count, now
    pulled on evidence of no progress instead.
"""

import hashlib
import re

WEIGHTS = {
    "obligation": 1.0,      # the checklist moved: an obligation discharged in Lean
    "solution": 0.8,        # write_solution accepted
    "verified": 0.4,        # Lean accepted a lemma that is not itself an obligation
    "computed": 0.15,       # run_python ran clean and printed something
    "ruled_out": 0.1,       # a dead end, recorded, is knowledge
    "banked": 0.05,         # a substantive journal_entry
    "failed_proof": -0.05,  # proved=false: a result, but a cheap one
    "error": -0.15,         # a tool error or a malformed call
    "truncated": -0.2,      # cut off at the generation cap
    "no_call": -0.25,       # a turn that ended without acting
    "repeat": -0.3,         # the same statement, code or approach again
    "misformalized": -0.3,  # compiled, but proved a different statement
    "step": -0.05,          # every step costs time
}
WINDOW = 6                  # steps that make up the "recent" return
STALL = 0.0                 # a recent return at or below this is a stall


def _key(name: str, args: dict) -> str:
    what = (args.get("statement") or args.get("lean_statement") or args.get("code")
            or args.get("approach") or args.get("claim") or "")
    what = re.sub(r"\s+", " ", str(what)).strip().lower()
    return hashlib.sha1(f"{name}:{what}".encode()).hexdigest() if what else ""


class Tracker:
    """One problem's return. begin() a step, feed it what happened, end() it."""

    def __init__(self) -> None:
        self.rewards: list = []
        self.total = 0.0
        self.events: list = []
        self._seen: set = set()
        self._last_stall = -WINDOW

    def begin(self) -> None:
        self.events = ["step"]

    def turn(self, finish: str, called: bool) -> None:
        if not called:
            self.events.append("truncated" if finish == "length" else "no_call")

    def tool(self, name: str, args: dict, result: dict, discharged: bool = False) -> None:
        k = _key(name, args)
        if k and k in self._seen and name in ("prove_theorem", "run_python", "rule_out"):
            self.events.append("repeat")
            return
        if k:
            self._seen.add(k)
        if not isinstance(result, dict) or result.get("error"):
            self.events.append("error")
        elif name == "prove_theorem":
            self.events.append("obligation" if discharged
                               else "verified" if result.get("proved")
                               else "misformalized" if result.get("formalization_suspect")
                               else "failed_proof")
        elif name == "run_python":
            ok = result.get("ok") and str(result.get("stdout") or "").strip()
            self.events.append("computed" if ok else "error")
        elif name == "rule_out":
            self.events.append("ruled_out")
        elif name == "journal_entry" and len(str(args.get("text") or args.get("content") or "")) >= 200:
            self.events.append("banked")
        elif name == "write_solution":
            self.events.append("solution")

    def end(self) -> dict:
        r = round(sum(WEIGHTS[e] for e in self.events), 3)
        self.rewards.append(r)
        self.total = round(self.total + r, 3)
        recent = round(sum(self.rewards[-WINDOW:]), 3)
        n = len(self.rewards)
        # one warning per window: nagging every step is noise the model learns to skip
        stalled = n >= WINDOW and recent <= STALL and n - self._last_stall >= WINDOW
        if stalled:
            self._last_stall = n
        return {"reward": r, "total": self.total, "recent": recent,
                "events": [e for e in self.events if e != "step"], "stalled": stalled}

    def line(self, got: dict) -> str:
        why = ", ".join(got["events"]) or "nothing happened"
        return (f"  [reward {got['reward']:+.2f} | total {got['total']:+.2f} | "
                f"recent {got['recent']:+.2f} | {why}]")


def stall_message(got: dict, kind: str = "proof") -> str:
    act = ("prove_theorem a WEAKER lemma or a special case of the obligation you are on, "
           "or switch to a different remaining obligation" if kind == "proof" else
           "attack the answer another way: a different method, a limiting case that "
           "must agree, or a numeric instance in run_python that checks it")
    return (f"IMPLICIT REWARD: your last {WINDOW} steps returned {got['recent']:+.2f} -- "
            "nothing was verified, computed cleanly or ruled out. The current line is not "
            f"paying. Change it now: {act}; or call conjecture / hunches for a new angle; "
            "or rule_out the current approach with the reason. Do not continue it unchanged.")


def demo() -> None:
    t = Tracker()
    t.begin(); t.tool("run_python", {"code": "print(1)"}, {"ok": True, "stdout": "1"})
    assert t.end()["reward"] == 0.1
    t.begin(); t.tool("prove_theorem", {"statement": "a"}, {"proved": True}, discharged=True)
    got = t.end()
    assert got["reward"] == 0.95 and got["total"] == 1.05 and got["events"] == ["obligation"], got
    t.begin(); t.tool("prove_theorem", {"statement": "A "}, {"proved": False})
    assert t.end()["events"] == ["repeat"], "the same statement again is a repeat"
    flags = []
    for _ in range(5):
        t.begin(); t.turn("stop", called=False); got = t.end(); flags.append(got["stalled"])
    # step 6 is the first full window, and it is under water: flagged there, once
    assert flags == [False, False, True, False, False], flags
    got = {"recent": -0.2}
    assert "WEAKER" in stall_message(got) and "limiting case" in stall_message(got, "physics")
    print("reward ok: verdicts scored, repeats caught, stalls flagged once per window")


if __name__ == "__main__":
    demo()
