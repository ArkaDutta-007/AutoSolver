"""Evolutionary conjecture search: an island-model GA over candidate claims.

WHY EVOLUTION, AND WHY IT IS NOT JUST "SAMPLE MORE"
---------------------------------------------------
FunSearch discovered genuinely new mathematics by pairing an LLM with a
programmatic evaluator, and AlphaEvolve generalised it -- across 67 problems in
analysis, combinatorics, geometry and number theory it rediscovered best-known
constructions in most and improved several. The structural ingredients are not
"a bigger model":

  * a PROGRAMMATIC evaluator that cannot be fooled  -- here, Lean
  * a POPULATION rather than a single best candidate
  * ISLANDS that evolve separately, so a good-but-local idea cannot take over
    the whole population before better ideas have had time to develop
  * an LLM ENSEMBLE: a fast model for candidate volume, a strong one for depth

This system already has the evaluator and, since the model benchmark, the
ensemble: cascade-2 at ~29s per signature for breadth, Qwen at ~140s for depth.
It measured 1.07 signatures/min against 0.24 -- which is an argument for giving
cascade the GENERATE role, not for discarding it.

THE ISLAND MODEL IS THE POINT
-----------------------------
A single population converges: the fittest conjecture is copied, mutated
slightly, and within a few generations everything is a variation of it. That is
the same local minimum this system already fell into by hand -- five restatements
of one divisibility fact. Islands evolve independently and only occasionally
exchange migrants, so several lines of attack develop in parallel. Periodically
the WORST island is reset from the best survivors elsewhere, which is FunSearch's
mechanism for spending compute on live leads rather than dead ones.

Generation is injected (`propose`), never imported, so the whole search is
testable without a model -- and so the ensemble split can be chosen by the
caller rather than hard-coded here.
"""

import json
import os
import re
import threading
import time

from . import archive, config

PATH = os.path.join(config.ROOT, "kb", "evolve.json")
_lock = threading.RLock()

N_ISLANDS = int(os.environ.get("EVOLVE_ISLANDS", "4"))
ISLAND_CAP = int(os.environ.get("EVOLVE_ISLAND_CAP", "12"))
# Reset the weakest island every N generations. Too eager and nothing has time
# to develop; too lazy and dead islands hold compute hostage.
RESET_EVERY = int(os.environ.get("EVOLVE_RESET_EVERY", "5"))


def _blank() -> dict:
    return {"generation": 0, "islands": [[] for _ in range(N_ISLANDS)],
            "history": []}


def _load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            st = json.load(fh)
        if len(st.get("islands", [])) != N_ISLANDS:
            st["islands"] = (st.get("islands", []) + [[] for _ in range(N_ISLANDS)])[:N_ISLANDS]
        return st
    except Exception:
        return _blank()


def _save(st: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, PATH)


# --------------------------------------------------------------------------
# fitness
# --------------------------------------------------------------------------

def fitness(cand: dict, goal: str = "") -> float:
    """Verified + novel + useful. All three, because any one alone degenerates.

    Verified-only rewards triviality -- `2 ∣ n*(n+1)` verifies every time and
    teaches nothing. Novel-only rewards nonsense, since the strangest statement
    is always the most novel. Useful-only cannot be measured until later proofs
    exist. Together they describe a conjecture worth having: it is TRUE, it is
    a DIFFERENT KIND of thing from what we hold, and something CITED it.
    """
    if not cand.get("verified"):
        # An unverified candidate is not worthless -- it may be a promising
        # statement we simply have not proved yet -- but it must never outrank
        # a verified one.
        return 0.1 * cand.get("novelty", 0.0)
    score = 1.0
    score += 1.5 * cand.get("novelty", 0.0)
    score += 0.5 * min(cand.get("uses", 0), 4) / 4.0
    score += 0.5 * min(cand.get("seconds", 0.0) / 600.0, 1.0)
    return round(score, 4)


def novelty(claim: str, goal: str = "") -> float:
    """1.0 when this KIND of result is untouched, decaying as its cell fills.

    Reuses the quality-diversity archive rather than a second notion of
    "different", so evolution and the exploration guard cannot disagree about
    what counts as new.
    """
    try:
        occ = archive.occupied(claim, goal=goal)
        if occ.get("filled"):
            return 0.0
        cov = archive.coverage()
        c = archive.cell(claim, goal=goal)
        seen = 1.0 if "|".join(c) in {"|".join(v["cell"]) for v in
                                      (json.load(open(archive.PATH)).values()
                                       if os.path.exists(archive.PATH) else [])} else 0.0
        return 1.0 - 0.5 * seen if cov["cells"] else 1.0
    except Exception:
        return 0.5


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def seed(claims: list, goal: str = "") -> dict:
    """Spread initial claims across islands round-robin."""
    with _lock:
        st = _load()
        for i, c in enumerate(claims):
            cand = {"claim": c, "verified": False, "uses": 0, "seconds": 0.0,
                    "novelty": novelty(c, goal), "born": 0, "parents": []}
            cand["fitness"] = fitness(cand, goal)
            st["islands"][i % N_ISLANDS].append(cand)
        for isl in st["islands"]:
            isl.sort(key=lambda c: -c.get("fitness", 0))
            del isl[ISLAND_CAP:]
        _save(st)
        return {"islands": [len(i) for i in st["islands"]]}


def observe(claim: str, verified: bool, seconds: float = 0.0, uses: int = 0,
            goal: str = "") -> dict:
    """Fold a proof outcome back into the population."""
    with _lock:
        st = _load()
        found = False
        for isl in st["islands"]:
            for cand in isl:
                if cand["claim"].strip() == claim.strip():
                    cand.update(verified=bool(verified), seconds=seconds,
                                uses=uses, novelty=novelty(claim, goal))
                    cand["fitness"] = fitness(cand, goal)
                    found = True
        _save(st)
        return {"updated": found}


def parents(island: int, k: int = 2) -> list:
    """Fittest members of one island, as breeding stock."""
    st = _load()
    if not 0 <= island < len(st["islands"]):
        return []
    return sorted(st["islands"][island], key=lambda c: -c.get("fitness", 0))[:k]


def step(propose, goal: str = "", per_island: int = 2) -> dict:
    """One generation: breed each island, evaluate novelty, cull, maybe reset.

    `propose(parents, island, goal) -> list[str]` supplies offspring claims.
    Injected so a generation can be driven by a cheap model, an expensive one,
    or a stub in tests.
    """
    with _lock:
        st = _load()
        st["generation"] += 1
        gen = st["generation"]
        born = 0
        for i, isl in enumerate(st["islands"]):
            ps = sorted(isl, key=lambda c: -c.get("fitness", 0))[:2]
            try:
                kids = propose(ps, i, goal) or []
            except Exception:
                kids = []
            for text in list(kids)[:per_island]:
                text = (text or "").strip()
                if not text:
                    continue
                if any(c["claim"].strip() == text for c in isl):
                    continue          # island already holds it
                cand = {"claim": text[:400], "verified": False, "uses": 0,
                        "seconds": 0.0, "novelty": novelty(text, goal),
                        "born": gen, "parents": [p["claim"][:80] for p in ps]}
                cand["fitness"] = fitness(cand, goal)
                isl.append(cand)
                born += 1
            isl.sort(key=lambda c: -c.get("fitness", 0))
            del isl[ISLAND_CAP:]

        reset_note = ""
        if gen % RESET_EVERY == 0:
            reset_note = _reset_worst(st)
        st["history"].append({"generation": gen, "born": born,
                              "best": round(max((c.get("fitness", 0)
                                                 for isl in st["islands"] for c in isl),
                                                default=0.0), 3),
                              "when": time.strftime("%H:%M:%S")})
        del st["history"][:-50]
        _save(st)
        return {"generation": gen, "born": born, "reset": reset_note,
                "sizes": [len(i) for i in st["islands"]]}


def _reset_worst(st: dict) -> str:
    """Reseed the weakest island from the best survivors elsewhere.

    FunSearch's anti-stagnation move. Without it a dead island keeps consuming
    a share of every generation forever -- which is the population-level
    version of the agent grinding on an exhausted approach.
    """
    def best_of(isl):
        return max((c.get("fitness", 0) for c in isl), default=0.0)
    scores = [best_of(i) for i in st["islands"]]
    if len(scores) < 2 or max(scores) <= 0:
        return ""
    worst = scores.index(min(scores))
    donors = sorted((c for i, isl in enumerate(st["islands"]) if i != worst
                     for c in isl), key=lambda c: -c.get("fitness", 0))[:3]
    if not donors:
        return ""
    st["islands"][worst] = [dict(d, born=st["generation"], parents=["migrated"])
                            for d in donors]
    return f"island {worst} reseeded from {len(donors)} migrants"


# --------------------------------------------------------------------------
# proposer: the LLM half of the loop
# --------------------------------------------------------------------------

_SCHEMA = {"type": "object",
           "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
           "required": ["claims"]}


def llm_proposer(model: str = "", n: int = 2, temperature: float = 1.0):
    """A `propose` callable backed by a model. Returns [] on any failure.

    Defaults to the FAST model, not the orchestrator. AlphaEvolve's ensemble
    splits the roles -- a lightweight model for candidate volume, a strong one
    for depth -- and this system measured exactly that split: cascade-2 at 1.07
    signatures/min against Qwen's 0.24. Generation wants volume; the compiler,
    not the generator, is what decides correctness here, so paying orchestrator
    latency per candidate buys very little.

    Temperature is deliberately high for the same reason it is high in
    `sample_proofs`: a bad conjecture costs one cheap evaluation, and diversity
    is the entire point of a population.
    """
    import requests
    from . import config

    def propose(parents: list, island: int, goal: str = "") -> list:
        seen = "\n".join(f"- {p.get('claim', '')[:160]}" for p in (parents or [])[:4])
        # The island index is in the prompt so separate islands genuinely
        # diverge instead of all regressing to the same "obvious next lemma".
        angle = ("a DIFFERENT mathematical mechanism from the ones above"
                 if island % 2 else "a STRONGER or more general version")
        prompt = (
            "You are proposing candidate lemmas for a formal proof search.\n\n"
            f"GOAL PROBLEM: {goal or '(open exploration)'}\n\n"
            f"Existing candidates in this line of attack:\n{seen or '- (none yet)'}\n\n"
            f"Propose {n} NEW candidate claims that would help settle the goal, each using "
            f"{angle}. Rules:\n"
            "- each must be a precise mathematical statement, not a plan or a question\n"
            "- each must be plausibly TRUE and plausibly provable in Lean 4 with Mathlib\n"
            "- do NOT restate the goal, and do NOT restate any claim listed above\n"
            "- prefer claims that a proof of the goal would actually USE")
        try:
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": model or config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "format": _SCHEMA,
                      "options": {**config.ctx(model or config.ORCHESTRATOR_MODEL), "num_predict": 2000, "temperature": temperature},
                      # Same class as prover._derive_once: a `format` schema and a
                      # reasoning budget share one allowance and the JSON loses.
                      "think": False},
                timeout=600)
            r.raise_for_status()
            claims = json.loads(r.json()["message"]["content"]).get("claims", [])
            return [str(c).strip() for c in claims if str(c).strip()][:n]
        except Exception:
            return []

    return propose


def generation(goal: str = "", model: str = "", per_island: int = 2) -> dict:
    """One full evolutionary generation, driven by a model. Convenience wrapper."""
    return step(llm_proposer(model=model, n=per_island), goal=goal,
                per_island=per_island)


def best(k: int = 5) -> list:
    st = _load()
    everyone = [c for isl in st["islands"] for c in isl]
    return sorted(everyone, key=lambda c: -c.get("fitness", 0))[:k]


def frontier(k: int = 5) -> list:
    """Fittest UNVERIFIED candidates -- i.e. what to try proving next."""
    st = _load()
    todo = [c for isl in st["islands"] for c in isl if not c.get("verified")]
    return sorted(todo, key=lambda c: -c.get("novelty", 0))[:k]


def summary() -> dict:
    st = _load()
    everyone = [c for isl in st["islands"] for c in isl]
    return {"generation": st["generation"], "population": len(everyone),
            "verified": sum(1 for c in everyone if c.get("verified")),
            "islands": [len(i) for i in st["islands"]],
            "best_fitness": round(max((c.get("fitness", 0) for c in everyone),
                                      default=0.0), 3)}


def reset() -> None:
    with _lock:
        _save(_blank())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="evolutionary conjecture search")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        reset(); print("population cleared")
    elif a.best:
        for c in best(10):
            print(f"  {c['fitness']:5.3f} {'V' if c['verified'] else ' '} {c['claim'][:90]}")
    elif a.frontier:
        for c in frontier(10):
            print(f"  novelty={c['novelty']:.2f} {c['claim'][:90]}")
    else:
        print(json.dumps(summary(), indent=1))
