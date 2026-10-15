"""Tool specs and dispatch for the orchestrator's tool-calling loop."""

from . import config, journal
from .mathlib_search import search_mathlib
from .prover import prove_theorem
from .sandbox import run_python
from .search import fetch_url, web_search

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information. Budget-limited; "
                       "disabled during leak-free proof evaluation.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "num": {"type": "integer", "description": "results, default 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a URL and return its readable text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "prove_theorem",
        "description": (
            "Formally prove a mathematical statement in Lean 4 with Mathlib. Makes "
            "several independent attempts with escalating strategies. Returns "
            "proved=true ONLY if the real Lean compiler accepted a complete proof "
            "AND the statement passed faithfulness review. Use this to verify any "
            "mathematical claim rather than asserting it."),
        "parameters": {"type": "object", "properties": {
            "statement": {"type": "string",
                          "description": "The claim to prove, precisely worded"},
            "lean_statement": {"type": "string",
                               "description": "Optional exact Lean 4 theorem signature. "
                                              "Supply it whenever you can state the theorem "
                                              "yourself: the prover then writes only the proof, "
                                              "which rules out mis-formalization entirely."},
            "obligation": {"type": "string",
                           "description": "The id of the SOLUTION CHECKLIST obligation this "
                                          "serves, e.g. 'potential_decrease'. Supply it "
                                          "whenever the claim supports one -- including "
                                          "helper lemmas that are not the obligation itself. "
                                          "Without it the checklist cannot tell that your "
                                          "work advanced anything."}},
            "required": ["statement"]}}},
    {"type": "function", "function": {
        "name": "search_mathlib",
        "description": ("Search 139k real Mathlib theorem/lemma names by keyword "
                        "fragments, e.g. 'gcd dvd' or 'irrational sqrt'. Use this "
                        "BEFORE citing a lemma name -- inventing names is the single "
                        "most common cause of failed proofs."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "space-separated name fragments"},
            "limit": {"type": "integer", "description": "max results, default 25"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": (
            "Run Python to EXPLORE before you formalize. sympy, numpy, itertools, "
            "math and fractions are preloaded; print() what you want to see. Use it "
            "to simulate small cases, search for a pattern, find a counterexample, "
            "or test a conjecture numerically. A conjecture that fails at n=4 is not "
            "worth a single prover second, and one that holds for all small cases is "
            "worth pinning a Lean signature for. It is also the VERIFIER for a "
            "computed answer: substitute it back, check limiting cases and units, "
            "and confirm one numeric instance. No network; 60s limit."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python source. print() your results."},
            "timeout": {"type": "integer", "description": "seconds, default 60, max 300"}},
            "required": ["code"]}}},
]

NOTEBOOK_SPECS = [
    {"type": "function", "function": {
        "name": "open_problem",
        "description": ("Open (or RESUME) the journal for a problem. Call this FIRST for "
                        "each problem. If a journal already exists it returns "
                        "`exhausted`: approaches ruled out in earlier sessions -- read "
                        "them and do NOT repeat them."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Short problem title"},
            "statement": {"type": "string", "description": "Full problem statement"},
            "source": {"type": "string", "description": "Source file, if any"}},
            "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "hunches",
        "description": ("Force this problem through several FOREIGN mathematical "
                        "lenses (order theory, probability, linear algebra, ...) and "
                        "return the readings ranked by how far they sit from the "
                        "obvious one. Use when the natural approach has stalled. "
                        "Returns UNVERIFIED suggestions -- questions worth asking, "
                        "not facts."),
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "Full problem statement."},
            "lenses": {"type": "integer",
                       "description": "How many lenses to force (2-8). Default 5."}},
            "required": ["goal"]}}},
    {"type": "function", "function": {
        "name": "conjecture",
        "description": ("Breed new candidate lemmas and get back the most NOVEL "
                        "unproved ones. Use this when you are stuck, when your last "
                        "few attempts failed, or when you notice you are circling the "
                        "same fact -- it proposes claims deliberately UNLIKE what you "
                        "already hold. Returns candidates only; you still have to "
                        "prove_theorem them."),
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string",
                     "description": "The problem you are trying to settle. Pass the "
                                    "full statement, not a summary."},
            "count": {"type": "integer",
                      "description": "Candidates per island (1-4). Default 2."}},
            "required": ["goal"]}}},
    {"type": "function", "function": {
        "name": "read_journal",
        "description": ("Read the current problem's journal back -- your own earlier "
                        "plans, attempts, and dead ends, in full. This is the ONLY way "
                        "to see the journal; never use a file or web tool for it."),
        "parameters": {"type": "object", "properties": {
            "max_chars": {"type": "integer",
                          "description": "Cap on returned characters. Omit for the "
                                         "default, which is sized to the context window. "
                                         "The most recent part is returned if the journal "
                                         "exceeds the cap."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "journal_entry",
        "description": ("Record your thinking as you go: PLAN before acting, REASON for "
                        "a choice, REFLECT after a result, SUMMARY when consolidating. "
                        "This is what lets a later session continue your work."),
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string",
                     "description": "PLAN | REASON | ACT | REFLECT | SUMMARY"},
            "text": {"type": "string"}},
            "required": ["kind", "text"]}}},
    {"type": "function", "function": {
        "name": "rule_out",
        "description": ("Mark an approach as exhausted, with the reason. Carried into "
                        "ALL future sessions so the same dead end is never re-explored."),
        "parameters": {"type": "object", "properties": {
            "approach": {"type": "string"},
            "why": {"type": "string"}},
            "required": ["approach", "why"]}}},
    {"type": "function", "function": {
        "name": "write_solution",
        "description": ("Call ONLY on success: for a proof, once its checklist is "
                        "Lean-verified; for a value or a physics quantity, once the "
                        "answer agrees two independent ways (derivation plus a "
                        "run_python check). Writes a clean step-by-step solution "
                        "document, separate from the working journal. End it with "
                        "the final answer(s) on their own line(s)."),
        "parameters": {"type": "object", "properties": {
            "solution_md": {"type": "string",
                            "description": "Clean numbered step-by-step solution in markdown"},
            "lean": {"type": "string", "description": "The verified Lean proof, if any"}},
            "required": ["solution_md"]}}},
    {"type": "function", "function": {
        "name": "close_session",
        "description": ("End work on this problem for now, and hand off to the next "
                        "session. Summaries ACCUMULATE across sessions and are shown "
                        "back to whoever picks this up, so write for that reader."),
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "description": "solved | partial | stuck"},
            "summary": {"type": "string",
                        "description": "REFLECT: what you learned this session. What the "
                                       "Lean results actually mean, which idea looks "
                                       "promising and why, what surprised you."},
            "next_step": {"type": "string",
                          "description": "PLAN: the single most concrete thing the next "
                                         "session should try FIRST -- ideally a specific "
                                         "claim to put through prove_theorem or a specific "
                                         "computation to run. Not 'continue working on it'. "
                                         "Omitting this makes the next session start blind."}},
            "required": ["status", "summary", "next_step"]}}},
]

TOOL_SPECS = TOOL_SPECS + NOTEBOOK_SPECS

# Tools that can reach the internet. For a problem with a published solution,
# offering these at all is a mistake: the model spends steps reaching for them,
# and any result is retrieval rather than reasoning.
WEB_TOOLS = {"web_search", "fetch_url"}


def specs(open_problem_mode: bool = None, suppress: set = None) -> list:
    """Tool list for this run.

    In CLOSED mode (the default) the web tools are REMOVED, not merely blocked.
    Blocking at call time still leaks the idea into the model's plan -- observed:
    an agent working IMO 2026 P1 searched for "IMO 2026 Problem 1 gcd lcm
    blackboard integers". The block held, but the step was wasted. Withholding
    the tool prevents the attempt.
    """
    from . import search as _search
    allowed = _search.enabled() if open_problem_mode is None else open_problem_mode
    out = list(TOOL_SPECS) if allowed else [t for t in TOOL_SPECS
                                            if t["function"]["name"] not in WEB_TOOLS]
    if not config.SANDBOX_ENABLED:
        out = [t for t in out if t["function"]["name"] != "run_python"]
    # Same lever as WEB_TOOLS above, for the same reason: a tool that is offered
    # gets reached for. When the agent has explored many times without
    # formalizing anything, WITHHOLDING run_python for a turn ends the loop;
    # merely asking it to stop does not.
    if suppress:
        out = [t for t in out if t["function"]["name"] not in suppress]
    return out

def hunches(goal: str = "", lenses: int = 5) -> dict:
    """Force the problem through several FOREIGN domain lenses and rank the
    results by distance from the consensus reading.

    Ordinary sampling returns what the model recites -- which for gcd/lcm is
    already in Mathlib and already tried. This deliberately asks for readings
    from domains the problem does not appear to live in, then ranks by scatter,
    because the idea furthest from the consensus is the one the model would
    never have volunteered. Everything returned is UNVERIFIED.
    """
    from . import intuition
    try:
        ideas = intuition.probe(goal, lenses=intuition.LENSES[:max(2, min(int(lenses), 8))])
        if not ideas:
            return {"error": "no hunches (proposer unreachable)"}
        return {"hunches": [{"lens": i["lens"], "idea": i["idea"],
                             "scatter": i.get("scatter", 0)} for i in ideas],
                "split": intuition.consensus_split(ideas),
                "note": ("Ranked by distance from the consensus reading. High scatter "
                         "= the model holds it weakly = worth asking, NOT worth "
                         "believing. Formalize before relying on any of it.")}
    except Exception as exc:
        return {"error": f"hunches failed: {exc}"}


def conjecture(goal: str = "", count: int = 2) -> dict:
    """Breed candidate lemmas and hand back the most novel unproved ones.

    The agent's hardest decision is WHAT to attempt next, and left to itself it
    retreats to whatever formalises cleanly -- the documented local minimum.
    This runs one evolutionary generation across islands and returns the
    frontier: fittest UNPROVED candidates, ranked by how unexplored their kind
    is according to the quality-diversity archive.
    """
    from . import evolve
    try:
        if not evolve.summary()["population"] and goal:
            evolve.seed([goal], goal=goal)
        out = evolve.generation(goal=goal, per_island=max(1, min(int(count), 4)))
        frontier = evolve.frontier(6)
        return {"generation": out["generation"], "born": out["born"],
                "reset": out.get("reset", ""),
                "try_next": [{"claim": c["claim"], "novelty": round(c["novelty"], 2)}
                             for c in frontier],
                "note": ("These are UNPROVED candidates ranked by novelty. Pick one and "
                         "prove_theorem it, or refine it first. Ignore any that restate "
                         "what you already hold.")}
    except Exception as exc:
        return {"error": f"conjecture failed: {exc}"}


REGISTRY = {"web_search": web_search, "fetch_url": fetch_url, "prove_theorem": prove_theorem,
            "conjecture": conjecture, "hunches": hunches,
            "search_mathlib": search_mathlib, "run_python": run_python,
            "open_problem": journal.open_problem,
            "read_journal": journal.read,
            "journal_entry": journal.entry,
            "rule_out": journal.rule_out,
            "write_solution": journal.solve,
            "close_session": journal.close}


def dispatch(name: str, args: dict, suppress: set = None) -> dict:
    # A model can emit a call to a tool it was never offered. REGISTRY membership
    # is therefore not authorization: check against the list actually in force.
    if name not in {t["function"]["name"] for t in specs(suppress=suppress)}:
        return {"error": f"tool {name} is not available in this run"}
    fn = REGISTRY.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": f"{name} failed: {exc}"}
