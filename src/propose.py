"""pass@K for the ORCHESTRATOR: propose several next actions, then choose.

THE ASYMMETRY THIS FIXES
------------------------
`prove_theorem` samples a goal 32 ways and lets Lean pick the winner. The
orchestrator -- which decides WHICH goal to attempt -- was sampled once. So the
system explores 32 proofs of a claim it chose in a single shot, and when that
choice is wrong all 32 samples are wasted. Choosing what to prove is where P1
died five times; proving it was never the hard part.

WHY TEMPERATURE IS NOT DIVERSITY
--------------------------------
Raising temperature on one model with one prompt resamples the same basin --
measured on the prover, where hint-only variation produced attempt 2 that
restated attempt 1 in different words, and where MAX_PROVER_TEMP is capped at
1.0 because above it a model mostly emits broken syntax rather than new ideas.
Diversity here therefore comes from three independent axes:

  * ANGLE   -- each proposal is asked for a structurally different kind of move
  * MODEL   -- a fast proposer generates volume, the strong model judges
  * HEAT    -- a modest temperature spread on top, not instead

The fast/strong split is AlphaEvolve's, and this machine measured the gap that
justifies it: cascade-2 produces 1.07 signatures/min against Qwen's 0.24. Volume
is cheap from the fast model; judgement is worth the slow one.

SELECTION IS NOT A SECOND OPINION
---------------------------------
Candidates are scored against things already known -- the quality-diversity
archive, the recent action history, the cost of the tool being called -- not by
asking a model which it likes. A model asked to rank its own proposals mostly
restates its first one, which is the anchoring failure this module exists to
break.
"""

import json
import os
import re
import threading

import requests

from . import archive, config, tools

# Each angle asks for a structurally different move. The point is not politeness
# -- "propose something else" gets a paraphrase -- but naming a DIFFERENT
# relationship to the problem, which changes what the model reaches for.
ANGLES = [
    ("direct", "the most direct next step toward settling the goal"),
    ("orthogonal",
     "a step that uses a DIFFERENT mathematical mechanism from anything tried "
     "so far -- if recent work was computational, try structural, and vice versa"),
    ("falsify",
     "a step that would DISPROVE or find a counterexample to the current "
     "approach. If the approach survives it, that is information; if it dies, "
     "you saved the remaining budget"),
    ("reduce",
     "a drastic simplification: a special case, a smaller instance, or a "
     "weaker statement that would still teach you the shape of the argument"),
    ("analogy",
     "a step borrowed from a DIFFERENT area of mathematics than the one this "
     "problem appears to live in -- name the analogy explicitly"),
]

_lock = threading.Lock()


def _angle_prompt(name: str, brief: str, goal: str, recent: list) -> str:
    hist = "\n".join(f"- {r}" for r in recent[-6:]) or "- (nothing yet)"
    return (
        f"GOAL: {goal[:1200]}\n\n"
        f"Recent actions in this session:\n{hist}\n\n"
        f"Propose ONE next action: {brief}.\n\n"
        "Answer as JSON with keys:\n"
        '  "tool"   - one of: ' + ", ".join(sorted(tools.REGISTRY)) + "\n"
        '  "why"    - one sentence: what this buys, and what it would rule out\n'
        '  "claim"  - if the tool is prove_theorem or conjecture, the exact '
        "mathematical statement; otherwise \"\"\n"
        "Do not restate an action from the history above.")


_SCHEMA = {"type": "object",
           "properties": {"tool": {"type": "string"},
                          "why": {"type": "string"},
                          "claim": {"type": "string"}},
           "required": ["tool", "why"]}


def _ask(model: str, prompt: str, temperature: float, timeout: int = 300) -> dict:
    try:
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "format": _SCHEMA,
                  "options": {**config.ctx(model), "num_predict": 1200, "temperature": temperature},
                  # MEASURED, three times now: reasoning and a grammar-constrained
                  # schema compete for ONE budget, and the JSON is what loses --
                  # it truncates mid-object and json.loads throws, so the whole
                  # call yields nothing. plan.py died this way twice; here it
                  # burned ~8 of every 9 minutes of a live run while producing
                  # zero candidates, silently, because an empty result prints
                  # nothing.
                  #
                  # These calls are STRUCTURAL -- pick a tool, state a claim --
                  # not deliberative. Thinking off puts the whole budget into
                  # the answer.
                  "think": False},
            timeout=timeout)
        r.raise_for_status()
        out = json.loads(r.json()["message"]["content"])
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def propose(goal: str, recent: list, k: int = 0, model: str = "") -> list:
    """K candidate next actions, one per angle. [] when disabled or unreachable."""
    k = k or config.ORCH_PASS_K
    if k <= 1:
        return []
    model = model or config.ORCH_PROPOSER_MODEL or config.ORCHESTRATOR_MODEL
    angles = ANGLES[:max(2, min(k, len(ANGLES)))]
    out = []
    for i, (name, brief) in enumerate(angles):
        # Heat rises across angles so the later, stranger ones are also sampled
        # more loosely -- but stays under 1.0, above which output degrades into
        # syntax errors rather than ideas.
        temp = min(0.95, 0.55 + 0.1 * i)
        got = _ask(model, _angle_prompt(name, brief, goal, recent), temp)
        tool = str(got.get("tool", "")).strip()
        if not tool or tool not in tools.REGISTRY:
            continue
        out.append({"angle": name, "tool": tool,
                    "why": str(got.get("why", ""))[:300],
                    "claim": str(got.get("claim", ""))[:400],
                    "temp": round(temp, 2)})
    return out


# Rough cost in "how much of the budget does this burn", used to break ties
# toward cheap information. prove_theorem at pass@32 was measured at 874s on a
# real lemma; a journal entry is free.
_COST = {"prove_theorem": 1.0, "conjecture": 0.35, "run_python": 0.1,
         "search_mathlib": 0.05, "read_journal": 0.02, "journal_entry": 0.0,
         "rule_out": 0.0, "open_problem": 0.0, "write_solution": 0.0,
         "close_session": 0.0, "web_search": 0.1, "fetch_url": 0.1}


def score(cand: dict, goal: str, recent: list) -> dict:
    """Rank against what is already KNOWN, not against a model's opinion.

    A model asked to rank its own proposals mostly re-endorses the first one --
    the anchoring failure this module exists to break. Every term below is
    computed from state the system already holds.
    """
    why, s = [], 0.0

    claim = cand.get("claim") or ""
    if claim:
        occ = archive.occupied(claim, goal=goal)
        if occ.get("filled"):
            s -= 1.0
            why.append("same KIND as an established result")
        else:
            s += 0.6
            why.append("unexplored cell")

    # Repeating the last action is the loop signature. Penalise the tool AND
    # the claim separately: five restatements were five different strings
    # calling the same tool.
    tool = cand.get("tool", "")
    tail = recent[-4:]
    if any(tool in r for r in tail):
        s -= 0.35 * sum(tool in r for r in tail)
        why.append("repeats a recent tool")
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    if claim and any(norm(claim)[:60] and norm(claim)[:60] in norm(r) for r in recent):
        s -= 0.8
        why.append("restates a recent claim")

    # An expensive action must justify itself; a cheap one that buys
    # information is nearly free to take.
    s -= 0.5 * _COST.get(tool, 0.2)

    # `falsify` and `reduce` are systematically under-generated by a model
    # optimising for looking productive, and they are the two that end grinding
    # early. A small thumb on the scale, not a mandate.
    if cand.get("angle") in ("falsify", "reduce"):
        s += 0.15
        why.append(f"{cand['angle']} move")

    # `score_why` and not `why`: the candidate already carries the MODEL's
    # rationale under `why`, and merging the two dicts would silently replace
    # the mathematics with the bookkeeping.
    return {"score": round(s, 3), "score_why": "; ".join(why) or "neutral"}


def choose(goal: str, recent: list, k: int = 0, model: str = "") -> dict:
    """Propose K, score them, return the winner plus the full ranking.

    Returns {} when proposing is disabled or nothing came back, so the caller
    falls through to its ordinary single-shot path.
    """
    cands = propose(goal, recent, k=k, model=model)
    if not cands:
        return {}
    ranked = sorted(({**c, **score(c, goal, recent)} for c in cands),
                    key=lambda c: -c["score"])
    return {"best": ranked[0], "ranked": ranked}


def brief(result: dict, limit: int = 5) -> str:
    """The ranking, as a prompt block. The orchestrator still DECIDES.

    Deliberately advisory. Executing the winner directly would replace the
    orchestrator's judgement with a heuristic score, and the score knows about
    novelty and cost but nothing about the mathematics. What it does reliably
    catch is the loop -- so it is presented as evidence, not as an instruction.
    """
    if not result.get("ranked"):
        return ""
    lines = ["", "CANDIDATE NEXT ACTIONS (proposed from different angles, scored "
                 "against the exploration map; you decide, but justify ignoring "
                 "the top one):"]
    for c in result["ranked"][:limit]:
        claim = f" -- {c['claim'][:110]}" if c.get("claim") else ""
        lines.append(f"  [{c['score']:+.2f}] ({c['angle']}) {c['tool']}{claim}")
        if c.get("why"):
            lines.append(f"          why: {c['why'][:130]}")
        lines.append(f"          scored: {c.get('score_why', '')[:110]}")
    return "\n".join(lines)
