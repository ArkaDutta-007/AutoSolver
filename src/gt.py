"""One block of pushed context, assembled from the four mechanisms.

WHY A COORDINATOR AND NOT FOUR CALL SITES
-----------------------------------------
Two reasons, both practical.

COST. Scouting and forecasting make network calls, and the anchor is rebuilt
EVERY TURN. Wiring them in naively would run a Lean probe and a forecast per
turn per obligation, which on a 150-minute problem is more expensive than the
proving. They are therefore computed once per obligation set and cached; the
cache invalidates when the set of open obligations changes, which is exactly
when the answers could have changed and no more often.

ORDER. The mechanisms are not independent. The scout must run before the
forecast, because the probe result is what turns a blind prior into a posterior
-- and it is free by then, since the scout has already paid for it. Encoding
that ordering in one place is what keeps it true.

WHAT THE AGENT SEES
-------------------
Never a tool. Never a suggestion to go and look something up. Just durable
state, in the same message as the problem statement and the checklist, arriving
whether or not the model would have thought to ask. This is the entire design
premise, and it is the one the measurements support: this agent does not call
tools it is merely offered.
"""

import time

from . import bidir, config, forecast, plan, recall, scout

# goal -> {"key": <open-obligation fingerprint>, "scouted": [...],
#          "fc": [...], "alloc": {...}, "built": ts}
_CACHE: dict = {}
_MAX_AGE = 3600.0


def _open_claims(goal: str) -> list:
    try:
        return [o.get("claim", "") for o in (plan.remaining(goal) or [])
                if o.get("claim")]
    except Exception:
        return []


def _key(claims: list) -> str:
    return "|".join(sorted(c[:60] for c in claims))


def prepare(goal: str, seconds_left: float = 0.0, force: bool = False) -> dict:
    """Run the expensive mechanisms if the obligation set has changed.

    Returns the cache entry. Safe to call every turn -- that is the point.
    """
    claims = _open_claims(goal)
    if not claims:
        return {}
    k = _key(claims)
    hit = _CACHE.get(goal)
    if hit and hit.get("key") == k and not force \
            and (time.time() - hit.get("built", 0)) < _MAX_AGE:
        return hit

    scouted = []
    if config.GT_SCOUT:
        try:
            scouted = scout.scout(claims)
        except Exception:
            scouted = []
        # A computed counterexample settles an obligation FALSE. Recording it
        # removes that term from the conjunction the problem must satisfy --
        # otherwise the obligation stays open forever and closure, which needs
        # open==0, can never fire however much real work is done.
        for sc in scouted:
            if sc.get("status") == scout.FALSE:
                try:
                    plan.refute(goal, sc.get("claim", ""), sc.get("note", ""))
                except Exception:
                    pass
    fc, alloc = [], {}
    if config.GT_FORECAST:
        try:
            fc = forecast.forecast(claims, scouted)
            if seconds_left > 0:
                alloc = forecast.allocate(fc, seconds_left)
        except Exception:
            fc, alloc = [], {}
    # Backward leg: one sufficient set, refuted if possible. A DEAD verdict
    # kills the ROUTE, which is the one signal forward search cannot produce.
    top = {}
    if config.GT_BIDIR:
        try:
            top = bidir.top_down(goal)
        except Exception:
            top = {}
    entry = {"key": k, "scouted": scouted, "fc": fc, "alloc": alloc,
             "top_down": top, "built": time.time()}
    _CACHE[goal] = entry
    return entry


def brief(goal: str, seconds_left: float = 0.0, claim: str = "") -> str:
    """The combined block for the anchor. Empty when nothing has anything to say.

    Ordered by how directly each item changes the next action: a free win from
    the scout outranks a forecast, which outranks historical evidence.
    """
    parts = []
    try:
        e = prepare(goal, seconds_left)
    except Exception:
        e = {}
    if e:
        try:
            parts.append(scout.report(e.get("scouted") or []))
        except Exception:
            pass
        try:
            parts.append(forecast.report(e.get("fc") or [], e.get("alloc") or {}))
        except Exception:
            pass
        try:
            parts.append(bidir.report(e.get("top_down") or {}))
        except Exception:
            pass
    # Recall is keyed to the claim actually in hand, falling back to the first
    # open obligation -- the thing the agent is most likely about to attempt.
    try:
        target = claim or next(iter(_open_claims(goal)), "") or goal
        parts.append(recall.report(target))
    except Exception:
        pass
    return "\n".join(p for p in parts if p)


def sweep() -> dict:
    """Between problems: enforce the library's carrying capacity.

    Placed here rather than in the sleep phase because it must happen even when
    the sleep phase does not run -- which, until the import bug was fixed, was
    every single run.
    """
    if not config.GT_RETIRE:
        return {}
    try:
        from . import library
        return library.retire()
    except Exception as e:
        return {"error": f"{type(e).__name__}"}


def reset(goal: str = "") -> None:
    if goal:
        _CACHE.pop(goal, None)
    else:
        _CACHE.clear()
