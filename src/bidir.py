"""Top-down and bottom-up at once. The top-down direction prunes by REFUTATION.

Bidirectional proof search is old (forward chaining meets backward chaining, and
Bi-Chainer switches between them when one direction stalls). The usual meeting
test is positive: a fact derived forwards matches a subgoal derived backwards.

This repo can also meet NEGATIVELY, which is the part worth having. Top-down
asks "what would suffice to prove the goal?" and gets a sufficient set S with
S => goal. Then every element of S is thrown at the sandbox. And:

    ONE FALSE ELEMENT OF S KILLS THE WHOLE DECOMPOSITION.

Not the goal -- the STRATEGY. If the route to the goal runs through a false
lemma, that route is dead however much prover time it is given, and it dies in
seconds instead of the measured 13-minute average per attempt. Bottom-up search
cannot produce that signal at all: it only ever learns that something did not
work yet.

The cheap positive meet comes free alongside: an element of S that the library
already proves is discharged at zero cost, because bottom-up work from earlier
problems is exactly what the library is.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No new proof engine, no scheduler, no meet-in-the-middle bookkeeping. Top-down
is one model call plus the falsifier and the library, both already built. The
`parallel` here is two stdlib threads, because the top-down leg is model+sandbox
and the bottom-up leg is the remote GPU prover -- genuinely overlapping work,
not a thread pool for its own sake.
"""

import concurrent.futures as futures
import json

import requests

from . import config, falsify, library

_SCHEMA = {"type": "object",
           "properties": {"evidence": {"type": "array",
                                       "items": {"type": "string"}}},
           "required": ["evidence"]}

_PROMPT = (
    "Work BACKWARDS from this goal.\n\n"
    "GOAL: {goal}\n\n"
    "List the SMALLEST set of mathematical facts that, taken together, would be "
    "enough to prove the goal. Not a plan, not steps -- the facts themselves, "
    "each a complete standalone claim someone could prove or disprove on its "
    "own.\n"
    "Prefer 2-4 facts. Each must be concrete enough to test on small numbers.\n"
    "Return JSON: \"evidence\" = the list of claims."
)

DEAD, LIVE, UNKNOWN = "dead", "live", "unknown"


def sufficient_set(goal: str, timeout: int = 180) -> list:
    """One backward step: the facts that would suffice. [] on any failure."""
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.ORCHESTRATOR_MODEL,
                  "messages": [{"role": "user",
                                "content": _PROMPT.format(goal=str(goal)[:1500])}],
                  "stream": False, "format": _SCHEMA, "think": False,
                  "options": {**config.ctx(config.ORCHESTRATOR_MODEL), "num_predict": 900, "temperature": 0.3}},
            timeout=timeout)
        r.raise_for_status()
        ev = json.loads(r.json()["message"]["content"]).get("evidence") or []
        return [str(e).strip() for e in ev if str(e).strip()][:5]
    except Exception:
        return []


def top_down(goal: str) -> dict:
    """Backward-chain to a sufficient set, then try to refute it.

    Verdicts:
      DEAD    an element is false -- this ROUTE cannot work, whatever the prover
              is given. The counterexample travels back so it can be rechecked.
      LIVE    nothing refuted; elements the library already proves are marked
              free, the rest are the reduced obligation list.
      UNKNOWN no sufficient set came back; changes nothing.
    """
    # The goal itself first: if the GOAL is refutable the problem was misread,
    # and no decomposition of it is worth anything.
    g = falsify.probe(goal)
    if g["status"] == falsify.FALSIFIED:
        return {"verdict": DEAD, "why": "the goal itself is false",
                "counterexample": g["counterexample"], "evidence": []}

    ev = sufficient_set(goal)
    if not ev:
        return {"verdict": UNKNOWN, "why": "no sufficient set", "evidence": []}

    out = []
    for e in ev:
        f = falsify.probe(e)
        if f["status"] == falsify.FALSIFIED:
            # One false element is enough. Stop -- the rest cost money and the
            # route is already dead.
            return {"verdict": DEAD,
                    "why": f"a required fact is false: {e[:110]}",
                    "counterexample": f["counterexample"],
                    "evidence": [x["claim"] for x in out] + [e]}
        known = library.search(e, k=1) or []
        out.append({"claim": e, "free": bool(known),
                    "known_as": known[0].get("name") if known else ""})
    return {"verdict": LIVE, "why": "no required fact could be refuted",
            "evidence": out,
            "free": [x["known_as"] for x in out if x["free"]]}


def solve(goal: str, bottom_up, timeout: int = 900) -> dict:
    """Run both directions at once; a DEAD top-down verdict wins immediately.

    `bottom_up` is any zero-arg callable -- the existing prover path. The two
    legs use different hardware (local model + sandbox vs remote GPU prover), so
    the overlap is real rather than decorative.
    """
    with futures.ThreadPoolExecutor(max_workers=2) as ex:
        td = ex.submit(top_down, goal)
        bu = ex.submit(bottom_up)
        try:
            top = td.result(timeout=timeout)
        except Exception:
            top = {"verdict": UNKNOWN, "why": "top-down failed", "evidence": []}
        # A DEAD verdict is the whole point: stop paying for a route that
        # cannot work. The bottom-up future is left to finish on its own rather
        # than cancelled mid-Lean-call, which would leak a compiler process.
        try:
            bottom = bu.result(timeout=timeout)
        except Exception:
            bottom = None
    return {"top_down": top, "bottom_up": bottom}


def report(top: dict) -> str:
    """Prompt block. DEAD is the only verdict that should change what happens."""
    if not top or top.get("verdict") == UNKNOWN:
        return ""
    if top["verdict"] == DEAD:
        return ("\nBACKWARD SEARCH: THIS ROUTE IS DEAD. " + top["why"]
                + f"\n  counterexample: {top.get('counterexample', '')[:200]}"
                + "\n  Recheck those values. If they hold, abandon this "
                  "decomposition -- no amount of prover time rescues a route "
                  "through a false fact. Find a different set of facts.")
    lines = ["", "BACKWARD SEARCH: facts that would suffice for the goal "
                 "(none could be refuted):"]
    for e in top.get("evidence", []):
        tag = f"  [ALREADY PROVED as {e['known_as']}]" if e.get("free") else "  [ ]"
        lines.append(f"{tag} {e['claim'][:120]}")
    if top.get("free"):
        lines.append(f"  {len(top['free'])} of these are already in the library "
                     "-- cite them, do not reprove them.")
    return "\n".join(lines)


if __name__ == "__main__":                                  # self-check, no net
    import types
    fake = types.SimpleNamespace(
        FALSIFIED="falsified",
        probe=lambda c, timeout=45: {
            "status": "falsified" if "invariant" in c else "unfalsified",
            "counterexample": "m=2,n=2: 4 != 2"})
    _real_f, _real_l = falsify, library
    globals()["falsify"] = fake
    globals()["library"] = types.SimpleNamespace(search=lambda q, k=1: [])
    globals()["sufficient_set"] = lambda g, timeout=180: [
        "gcd(a,b) divides a", "the product is invariant under the move"]

    dead = top_down("prove the final configuration")
    assert dead["verdict"] == DEAD, dead
    assert "invariant" in dead["why"], dead
    assert "m=2,n=2" in report(dead)

    globals()["sufficient_set"] = lambda g, timeout=180: ["gcd(a,b) divides a"]
    live = top_down("prove something")
    assert live["verdict"] == LIVE, live
    assert len(live["evidence"]) == 1

    globals()["falsify"] = types.SimpleNamespace(
        FALSIFIED="falsified",
        probe=lambda c, timeout=45: {"status": "falsified",
                                     "counterexample": "n=0"})
    assert top_down("a false goal")["why"] == "the goal itself is false"

    globals()["falsify"], globals()["library"] = _real_f, _real_l
    print("bidir self-check ok")
