"""The obligation ledger: what would it take to FINISH, and what is left.

WHY NOTHING EVER CLOSED
-----------------------
Across ten hours and five IMO problems the system verified two real lemmas and
closed nothing. The reason was structural, not a matter of budget:

  * every single `prove_theorem` call was a SUB-LEMMA -- not one attempted the
    problem's own statement
  * `write_solution` was called ZERO times, in any run, ever
  * no code path assembled lemmas into a final theorem; the only "decompose" in
    the codebase splits ONE goal into `have` steps, which is the downward
    direction

So the loop was: decompose -> prove a piece -> journal -> decompose again.
There is no upward move. Repeated attempts did not converge on an answer; they
grew a pile. P1 has a correct, expensively-proved invariant sitting in the
library and nothing whose job was to use it to finish.

WHAT THIS ADDS
--------------
A problem is decomposed ONCE into the specific obligations that would, together,
settle it. After that every action is measured against that ledger:

    which obligation does this discharge?  what is still open?  can I finish?

`plan-reason-act-reflect` becomes scoped: each entry names the obligation it
serves, so "progress" means an obligation moved, not that a turn happened.

CLOSURE IS FORCED, NOT HOPED FOR
--------------------------------
Two triggers, because waiting for the agent to volunteer never worked:

  * ALL obligations discharged      -> attempt the top-level statement now
  * budget mostly spent             -> stop opening new lines, write up what
                                       IS established, and say what remains

The second matters more than the first. A partial solution with three proved
lemmas and an honest statement of the gap is a result; a pile of lemmas nobody
assembled is not.
"""

import json
import os
import re
import threading
import time

from . import config

PATH = os.path.join(config.ROOT, "kb", "plans.json")
# Budget for the THINKING phase only. Generous: this is where the analysis
# happens, and running long here costs detail rather than the whole result --
# unlike the schema phase, where overrun means truncated JSON and nothing at
# all. At ~9 tok/s this is a few minutes, paid once per problem.
PLAN_THINK_TOKENS = int(os.environ.get("PLAN_THINK_TOKENS", "6000"))
_lock = threading.RLock()

OPEN, PROVED, FAILED, BLOCKED = "open", "proved", "failed", "blocked"

# REFUTED is not FAILED. FAILED means "we could not prove it" -- still open in
# principle, worth another approach. REFUTED means a counterexample was COMPUTED:
# the claim is false, no approach will ever close it, and it must stop counting
# against completion.
#
# This is the one lever that touches the composition arithmetic. Every
# obligation must succeed for a problem to close, so the end-to-end rate is a
# PRODUCT -- measured at 6 obligations, formalize-and-prove-all is 0.000%. A
# false obligation makes that product exactly zero and blocks closure forever.
# Removing a term is worth more than improving one.
REFUTED = "refuted"

# What a step of an IMO-style solution actually IS. Naming the kinds keeps the
# decomposition honest: "determine all f" needs an ANSWER obligation, and a
# ledger of pure lemmas silently drops the half of the problem that asks what
# the answer even is.
KINDS = ("answer",        # what IS the value / set / configuration
         "lemma",         # a supporting mathematical fact
         "construction",  # an explicit example or witness
         "bound",         # an inequality or extremal claim
         "closure")       # the top-level statement itself

_SCHEMA = {
    "type": "object",
    "properties": {"obligations": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "kind": {"type": "string"},
                       "claim": {"type": "string"}, "why": {"type": "string"}},
        "required": ["id", "kind", "claim"]}}},
    "required": ["obligations"]}


def _load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, PATH)


def _key(goal: str) -> str:
    return re.sub(r"\W+", "", (goal or "")[:120].lower())[:60] or "unknown"


# --------------------------------------------------------------------------
# the planner: one cloud call, before any local work
# --------------------------------------------------------------------------

_RULES = (f"- kind is one of: {', '.join(KINDS)}\n"
          "- the LAST obligation must be kind='closure': the problem's own statement\n"
          "- 4 to 8 obligations. Each must be a precise mathematical claim, not a "
          "task like 'investigate' or 'explore'\n"
          "- ids are short slugs like 'inv' or 'termination'\n")


def blueprint(goal: str, specs: list) -> dict:
    """The planner's one call: formulate, choose the approach, order the tools.

    It PLANS and never solves -- the local model does every derivation,
    computation and proof, so the answer is still the local model's. A strong
    cloud model may know a published solution, which is why it is told to name
    what must be found, never the value. {} when the planner is off or fails;
    the caller then decomposes locally, exactly as before.
    """
    import requests
    if not config.PLANNER_MODEL:
        return {}
    if not config.PLANNER_KEY:
        print("[planner] no DASHSCOPE_API_KEY -- planning locally", flush=True)
        return {}
    names = [t["function"]["name"] for t in specs]
    menu = "\n".join(f"- {t['function']['name']}: "
                     f"{t['function'].get('description', '').split('. ')[0][:160]}"
                     for t in specs)
    prompt = (
        "You are the PLANNER for an autonomous solver. A smaller local model will do "
        "ALL of the work -- every derivation, computation and proof -- using only the "
        "tools below. Your job: formulate the problem precisely and lay out the plan "
        "it follows. Do NOT solve it: no final values, no proofs, no worked "
        "computations. Where something must be found, say what, not what it is.\n\n"
        f"PROBLEM:\n{goal[:6000]}\n\nTOOLS THE LOCAL MODEL HAS:\n{menu}\n\n"
        "Answer with ONE JSON object and nothing else:\n"
        '{"formulation": {"restated": "the problem, precisely, in your own words", '
        '"given": ["..."], "unknown": ["..."], "assumptions": ["anything implicit"], '
        '"kind": "proof | compute | physics"},\n'
        ' "strategy": "the approach in 2-4 sentences, and why it should work",\n'
        ' "steps": [{"do": "one concrete action", "tool": "one tool name from the list", '
        '"why": "what it establishes", "done_when": "how the local model knows"}],\n'
        ' "pitfalls": ["traps a solver typically falls into here"],\n'
        ' "obligations": [{"id": "", "kind": "", "claim": "", "why": ""}]}\n\n'
        "steps: 4 to 10, in the order to do them; start with checks that are cheap "
        "(small cases, units, a numeric instance) and end with write_solution.\n"
        "obligations: the checklist a complete solution must discharge.\n" + _RULES +
        "- if the problem says 'determine' or 'find', the FIRST obligation is "
        "kind='answer', naming what must be determined -- not its value")
    err = ""
    try:
        r = requests.post(f"{config.PLANNER_BASE.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {config.PLANNER_KEY}"},
                          json={"model": config.PLANNER_MODEL, "temperature": 0.3,
                                "max_tokens": config.PLANNER_MAX_TOKENS,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=600)
        if r.status_code != 200:
            err = f"HTTP {r.status_code}: {r.text[:160]}"
        else:
            text = r.json()["choices"][0]["message"].get("content") or ""
            bp = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:160]}"
    if err or not isinstance(bp, dict):
        print(f"[planner] {config.PLANNER_MODEL} failed ({err or 'not an object'}) "
              "-- planning locally", flush=True)
        return {}
    f = bp.get("formulation") if isinstance(bp.get("formulation"), dict) else {}
    lst = lambda v: [str(x)[:300] for x in v][:12] if isinstance(v, list) else []
    steps = [{"do": str(s.get("do", ""))[:300],
              # a tool the local model does not have is a plan it cannot follow
              "tool": str(s.get("tool", "")) if s.get("tool") in names else "",
              "why": str(s.get("why", ""))[:200], "done_when": str(s.get("done_when", ""))[:200]}
             for s in (bp.get("steps") or []) if isinstance(s, dict) and s.get("do")][:12]
    return {"model": config.PLANNER_MODEL,
            "formulation": {"restated": str(f.get("restated", ""))[:2000],
                            "given": lst(f.get("given")), "unknown": lst(f.get("unknown")),
                            "assumptions": lst(f.get("assumptions")),
                            "kind": str(f.get("kind", "")).strip().lower()},
            "strategy": str(bp.get("strategy", ""))[:1500], "steps": steps,
            "pitfalls": lst(bp.get("pitfalls")),
            "obligations": [o for o in (bp.get("obligations") or []) if isinstance(o, dict)]}


def blueprint_text(bp: dict) -> str:
    """The plan as a system-prompt block. The system prompt survives compaction."""
    f = bp.get("formulation") or {}
    out = [f"== PLAN FROM THE PLANNER ({bp.get('model', '?')}): it planned, you do the work =="]
    if f.get("restated"):
        out.append(f"Problem: {f['restated']}")
    for label, key in (("Given", "given"), ("Unknown", "unknown"), ("Assumptions", "assumptions")):
        if f.get(key):
            out.append(f"{label}: " + "; ".join(f[key]))
    if bp.get("strategy"):
        out.append(f"Strategy: {bp['strategy']}")
    if bp.get("steps"):
        out.append("Steps, in order:")
        for i, s in enumerate(bp["steps"], 1):
            out.append(f"  {i}. {s['do']}" + (f"  -> {s['tool']}" if s["tool"] else "")
                       + (f"  (done when: {s['done_when']})" if s["done_when"] else ""))
    if bp.get("pitfalls"):
        out.append("Pitfalls: " + "; ".join(bp["pitfalls"]))
    out.append("Follow these steps in order. If a result shows a step is wrong, say why "
               "in journal_entry and adapt. The plan is a guide, not a proof: verify "
               "everything yourself.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# building the ledger
# --------------------------------------------------------------------------

def build(goal: str, model: str = "", force: bool = False, specs: list = None) -> dict:
    """Decompose the goal into obligations, ONCE. Returns the ledger.

    Built once and then held: re-decomposing every turn is how the agent
    wandered, because each new decomposition quietly replaced the last and
    nothing was ever finished.
    """
    import requests
    k = _key(goal)
    with _lock:
        store = _load()
        cached = store.get(k)
        # A DEGRADED ledger is not a cache hit. The first version persisted
        # whatever it got, so one HTTP 500 wrote a zero-obligation ledger to
        # disk and every later run returned it instantly -- the retry, the
        # error report and the minimal-ledger fallback all became unreachable,
        # and the run printed "[plan] 0 obligation(s)" with no explanation.
        # Failure must never be sticky.
        if cached and not force and not cached.get("degraded"):
            if cached.get("blueprint") or specs is None or not config.PLANNER_MODEL:
                return cached
    # `specs` given = the planner may run (tests and tools pass none: no network).
    bp = blueprint(goal, specs) if specs is not None else {}
    if cached and not force and not cached.get("degraded"):
        # a ledger from before the planner: add its plan, keep the progress
        if bp:
            with _lock:
                store = _load()
                store.setdefault(k, cached)["blueprint"] = {
                    k2: v for k2, v in bp.items() if k2 != "obligations"}
                _save(store)
                return store[k]
        return cached
    got = bp.get("obligations") or []

    # PHASE 1 -- THINK, in free text, with the configured effort.
    #
    # Reasoning and JSON compete for ONE token budget, and a grammar-constrained
    # nested schema cannot half-finish gracefully: at think=low/n=3000 the call
    # burned 337s and returned a truncated object. Rather than lower the effort
    # (which loses the analysis that makes a decomposition worth anything), the
    # thinking happens here with no schema at all, where running long costs
    # detail instead of the whole result.
    analysis = ""
    if not got and config.ORCH_THINK and config.ORCH_THINK.lower() not in ("off", "false", "none", "0"):
        try:
            r0 = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": model or config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content":
                                    f"PROBLEM:\n{goal[:2000]}\n\n"
                                    "What must a COMPLETE solution establish? Work out "
                                    "the mathematical structure: the key invariant or "
                                    "mechanism, what has to be proved, and in what "
                                    "order. Be concrete and name actual objects. Do not "
                                    "solve it -- map what solving it requires."}],
                      "stream": False,
                      "options": {**config.ctx(model or config.ORCHESTRATOR_MODEL), "num_predict": PLAN_THINK_TOKENS, "temperature": 0.4},
                      "think": config.ORCH_THINK},
                timeout=1200)
            if r0.status_code == 200:
                m0 = r0.json().get("message", {}) or {}
                analysis = ((m0.get("content") or "").strip()
                            or (m0.get("thinking") or "").strip())[:4000]
        except Exception:
            analysis = ""

    # PHASE 2 -- FORMAT, with thinking OFF, so the whole budget goes to the JSON.
    # Measured: think=off produced 8 obligations in 94s at 826 tokens, where
    # think=low failed after 337s. Structure is a transcription task.
    prompt = (
        f"PROBLEM:\n{goal[:2000]}\n\n"
        + (f"ANALYSIS OF WHAT A SOLUTION REQUIRES:\n{analysis}\n\n" if analysis else "")
        + "List the obligations that TOGETHER would completely settle this problem "
        "-- the checklist a full solution must discharge. Rules:\n"
        f"- kind is one of: {', '.join(KINDS)}\n"
        "- if the problem says 'determine' or 'find', the FIRST obligation must be "
        "kind='answer': stating what the answer actually is\n"
        "- the LAST obligation must be kind='closure': the problem's own statement\n"
        "- 4 to 8 obligations. Each must be a precise mathematical claim, not a "
        "task like 'investigate' or 'explore'\n"
        "- ids are short slugs like 'inv' or 'termination'\n"
        'Answer as JSON: {"obligations": [{"id","kind","claim","why"}]}')
    # Retried, and the reason is REPORTED. The first version swallowed every
    # failure into an empty list, and an empty ledger silently disables the
    # whole closure mechanism -- the run printed "[plan] 0 obligation(s)" and
    # carried on with exactly the open-ended behaviour the ledger exists to
    # stop. A single HTTP 500 (ollama serialises against one loaded model, so a
    # concurrent request is enough) was all it took.
    err = ""
    for attempt in range(0 if got else 2):   # the planner's list stands
        try:
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": model or config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "format": _SCHEMA,
                      "options": {**config.ctx(model or config.ORCHESTRATOR_MODEL), "num_predict": 2000 * (attempt + 1),
                                  "temperature": 0.3},
                      # MEASURED: think=off is 3.6x faster AND succeeds where
                      # think=low fails. At n=3000 with think=low the call ran
                      # 337s and returned TRUNCATED JSON -- reasoning ate the
                      # budget, then the nested array-of-objects could not
                      # finish inside what was left, so json.loads got half an
                      # object. With thinking off, all 826 tokens went to the
                      # JSON: 200 OK in 94s with 8 obligations.
                      #
                      # Decomposition is structural -- listing what a solution
                      # must CONTAIN -- not a task that needs deliberation. The
                      # budget also doubles on retry, so a genuinely long
                      # decomposition gets a second, roomier attempt rather
                      # than failing identically twice.
                      "think": False},
                timeout=900)
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {r.text[:160]}"
                time.sleep(5)
                continue
            got = json.loads(r.json()["message"]["content"]).get("obligations", [])
            if got:
                break
            err = "model returned no obligations"
        except Exception as exc:
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
            time.sleep(5)
    if err and not got:
        print(f"[plan] decomposition failed ({err}) -- using the minimal ledger",
              flush=True)

    obs = []
    for i, o in enumerate(got[:10]):
        claim = str(o.get("claim", "")).strip()
        if len(claim) < 10:
            continue
        obs.append({"id": str(o.get("id") or f"o{i}")[:24],
                    "kind": (str(o.get("kind", "lemma")).lower()
                             if str(o.get("kind", "")).lower() in KINDS else "lemma"),
                    "claim": claim[:400], "why": str(o.get("why", ""))[:200],
                    "status": OPEN, "note": "", "attempts": 0})
    # A closure obligation ALWAYS exists -- note the missing `if obs` guard that
    # used to be here. When decomposition failed, the empty list skipped this
    # too, so the ledger had nothing to finish against and `should_close` could
    # never fire. The minimal ledger (just the problem itself) is a far weaker
    # checklist than a real decomposition, but it still triggers the budget
    # close and still forces a write-up, which is the behaviour that matters.
    if not any(o["kind"] == "closure" for o in obs):
        obs.append({"id": "closure", "kind": "closure", "claim": goal[:400],
                    "why": "the problem's own statement", "status": OPEN,
                    "note": "", "attempts": 0})
    # `degraded` = the model contributed nothing and this is only the minimal
    # fallback. It is still RETURNED (so closure can fire) but it is not
    # treated as a cache hit next time, so the next run retries the real
    # decomposition instead of inheriting the failure forever.
    ledger = {"goal": goal[:400], "built": time.strftime("%Y-%m-%d %H:%M:%S"),
              "degraded": not got, "obligations": obs}
    if bp:
        ledger["blueprint"] = {k2: v for k2, v in bp.items() if k2 != "obligations"}
    with _lock:
        store = _load()
        store[k] = ledger
        _save(store)
    return ledger


def ledger(goal: str) -> dict:
    return _load().get(_key(goal), {})


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------

_MATCH_STOP = {"the", "and", "for", "all", "any", "that", "this", "with",
               "are", "was", "has", "have", "such", "then", "there", "each",
               "from", "into", "some", "every", "which", "where", "must"}


def _overlap(a: str, b: str) -> float:
    """Jaccard over content words, keeping THREE-letter mathematical tokens.

    A `len(w) > 3` filter looked harmless and silently dropped `gcd`, `lcm`,
    `min`, `max`, `sum`, `set` -- the most discriminating words in this domain.
    Two gcd claims then compared as empty sets and matched nothing, so the
    ledger never moved while the prover was verifying exactly those claims.
    """
    pick = lambda t: {w for w in re.findall(r"[a-z]+", (t or "").lower())
                      if len(w) >= 3 and w not in _MATCH_STOP}
    wa, wb = pick(a), pick(b)
    return len(wa & wb) / max(len(wa | wb), 1) if wa and wb else 0.0


def discharge(goal: str, claim: str, ok: bool, note: str = "",
              obligation: str = "") -> dict:
    """Match a proof result to the obligation it serves, and mark it.

    Fuzzy-matched on content: the agent rarely restates an obligation
    verbatim, and demanding that it does would leave the ledger permanently
    untouched while real work happened beside it.
    """
    k = _key(goal)
    with _lock:
        store = _load()
        led = store.get(k)
        if not led:
            return {}
        # An EXPLICIT id wins over inference. Text overlap cannot connect a
        # helper lemma to the obligation it serves -- "the exponent of p in the
        # prime factorization of gcd(m,n)" shares almost nothing with "the
        # lexicographic potential strictly decreases" -- so a run spent 35
        # minutes on genuine supporting work and the ledger recorded nothing,
        # the stuck counter never reset, and closure could never fire.
        best, score = None, 0.0
        if obligation:
            best = next((o for o in led["obligations"]
                         if o["id"] == obligation.strip()), None)
            if best:
                score = 1.0
        if best is None:
            for o in led["obligations"]:
                s = _overlap(o["claim"], claim)
                if s > score:
                    best, score = o, s
            if not best or score < 0.25:
                return {"matched": None, "score": round(score, 3)}
        if best["status"] == REFUTED:
            # A counterexample outranks a proof: if both exist, the SIGNATURE
            # does not say what the claim said. Refusing here is what stops a
            # verified proof of a false claim being banked, which has happened.
            return {"matched": best["id"], "status": REFUTED,
                    "score": round(score, 3), "note": "refuted; proof ignored"}
        best["attempts"] = best.get("attempts", 0) + 1
        if ok:
            best["status"] = PROVED
        elif best["status"] != PROVED:
            best["status"] = FAILED if best["attempts"] >= 3 else OPEN
        best["note"] = note[:200]
        _save(store)
        return {"matched": best["id"], "status": best["status"],
                "score": round(score, 3)}


def refute(goal: str, claim: str, why: str = "") -> dict:
    """Mark an obligation false on the strength of a computed counterexample.

    Same fuzzy matching as `discharge`, and deliberately IRREVERSIBLE by the
    normal path: `discharge(ok=True)` will not resurrect a refuted obligation,
    because proving something the sandbox refuted means the signature does not
    say what the claim said -- exactly the mis-formalization this project has
    already banked once.
    """
    k = _key(goal)
    with _lock:
        store = _load()
        led = store.get(k)
        if not led:
            return {}
        best, score = None, 0.0
        for o in led["obligations"]:
            s_ = _overlap(o["claim"], claim)
            if s_ > score:
                best, score = o, s_
        if not best or score < 0.25:
            return {"matched": None, "score": round(score, 3)}
        best["status"] = REFUTED
        best["note"] = ("REFUTED: " + why)[:240]
        _save(store)
        return {"matched": best["id"], "status": REFUTED, "score": round(score, 3)}


def progress(goal: str) -> dict:
    led = ledger(goal)
    obs = led.get("obligations") or []
    if not obs:
        return {"total": 0, "proved": 0, "open": 0, "fraction": 0.0}
    proved = [o for o in obs if o["status"] == PROVED]
    refuted = [o for o in obs if o["status"] == REFUTED]
    # A refuted obligation is RESOLVED, not outstanding: it is settled false, so
    # it neither counts as open nor bars "every obligation discharged". The
    # denominator drops it too -- scoring a solution against a claim known to be
    # false would understate what was actually established.
    live = [o for o in obs if o["status"] != REFUTED] or obs
    return {"total": len(obs), "live": len(live), "proved": len(proved),
            "open": sum(1 for o in obs if o["status"] == OPEN),
            "refuted": len(refuted),
            "failed": sum(1 for o in obs if o["status"] == FAILED),
            "fraction": round(len(proved) / len(live), 3),
            "next": next((o["id"] for o in obs if o["status"] == OPEN), None)}


def remaining(goal: str) -> list:
    return [o for o in (ledger(goal).get("obligations") or [])
            if o["status"] in (OPEN, FAILED)]


# --------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------

def should_close(goal: str, elapsed_fraction: float,
                 write_at: float = 0.75) -> dict:
    """Is it time to stop opening lines and write up what is established?

    Two triggers. The BUDGET one matters more: waiting for everything to be
    proved means writing nothing, because on a hard problem not everything will
    be. A partial solution with three verified lemmas and an honest statement
    of the gap is a result; an unassembled pile is not.
    """
    p = progress(goal)
    if not p["total"]:
        return {"close": False}
    if p["open"] == 0 and p["proved"]:
        return {"close": True, "why": "every obligation discharged",
                "mode": "complete", **p}
    if elapsed_fraction >= write_at and p["proved"]:
        return {"close": True,
                "why": f"{int(elapsed_fraction * 100)}% of the budget spent with "
                       f"{p['proved']}/{p['total']} discharged",
                "mode": "partial", **p}
    return {"close": False, **p}


def brief(goal: str, elapsed_fraction: float = 0.0) -> str:
    """The ledger as a prompt block. This is what makes a turn SCOPED."""
    led = ledger(goal)
    obs = led.get("obligations") or []
    if not obs:
        return ""
    mark = {PROVED: "[x]", OPEN: "[ ]", FAILED: "[!]", BLOCKED: "[-]",
            REFUTED: "[FALSE]"}
    lines = ["", "== SOLUTION CHECKLIST (this is what finishing MEANS) =="]
    for o in obs:
        lines.append(f"  {mark.get(o['status'], '[ ]')} {o['id']:<14} "
                     f"({o['kind']}) {o['claim'][:110]}")
        if o.get("note") and o["status"] in (FAILED, BLOCKED):
            lines.append(f"        blocked: {o['note'][:100]}")
        if o["status"] == REFUTED:
            # The counterexample travels with the verdict so the agent can
            # recheck it -- and so it does not quietly retry a false claim.
            lines.append(f"        {o.get('note', 'refuted')[:150]}")
            lines.append("        This claim is FALSE. Do not prove it. If the "
                         "solution needs it, restate it correctly first.")
    p = progress(goal)
    tail = f"  -- {p['proved']}/{p['live']} discharged."
    if p.get("refuted"):
        tail += (f" {p['refuted']} refuted by counterexample and dropped from "
                 "the requirement.")
    lines.append(tail + f" Next open: {p['next'] or '(none)'}")
    c = should_close(goal, elapsed_fraction)
    if c.get("close"):
        lines.append("")
        lines.append(f"  ** CLOSE NOW: {c['why']}. Attempt the `closure` obligation "
                     "with the lemmas you already hold, then call write_solution "
                     "with the argument and an explicit statement of any gap. "
                     "Do NOT open a new line of attack. **")
    else:
        lines.append("  Every action this turn must serve ONE named obligation. "
                     "Say which, in your journal entry.")
    return "\n".join(lines)


def reset(goal: str = "") -> None:
    with _lock:
        if not goal:
            _save({})
            return
        store = _load()
        store.pop(_key(goal), None)
        _save(store)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="solution obligation ledger")
    ap.add_argument("file", help="problem statement file")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read()
    led = build(text, force=a.rebuild)
    print(brief(text))
    print("\n" + json.dumps(progress(text), indent=1))
