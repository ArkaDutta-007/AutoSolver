"""Look before you commit. The leader's move in a Stackelberg game.

THE SITUATION THIS FIXES
------------------------
Measured, from the run logs: a single `prove_theorem` call takes 13 minutes on
average and succeeds 23% of the time, and the orchestrator chooses which claim
to spend that on with NO prior information whatsoever. In game terms it is a
leader committing to a move without ever observing the follower's best response,
which throws away the one structural advantage a leader has.

The follower here is the Lean compiler, and it is cheap to consult. `exact?`,
`simp`, `omega`, `decide` and the rest of the ladder run in SECONDS against the
hundreds a full best-of-32 proof attempt costs. Some obligations fall to a
one-line tactic; others do not move at all. Both facts are worth knowing before
allocating the expensive budget, and neither is currently known.

WHY THE SCOUT IS NOT WASTED WORK
--------------------------------
The obvious objection to scouting is that it spends budget to decide how to
spend budget. It does not, here, because of an accident of the pipeline that is
worth stating: the expensive part of a scout is deriving the pinned signature,
and a pinned signature is EXACTLY what the full attempt needs anyway. It unlocks
proof-only mode, which removed the entire mis-formalization failure class and
took P6 from 397.8s to 12.0s.

So a scouted obligation that is later attempted costs nothing extra -- the
signature is handed forward. The scout pays only for obligations it talks the
agent OUT of attempting, which is the case where the saving is largest.

WHAT IT RETURNS, AND WHAT IT REFUSES TO CLAIM
---------------------------------------------
Three outcomes, and the middle one is the common one:

  SOLVED    a tactic closed it outright. Free proof; take it.
  OPEN      it typechecks, nothing trivial worked. NO information about
            difficulty beyond "not trivial" -- and this module does not pretend
            otherwise, because a probe that fails is weak evidence.
  BROKEN    the signature does not even elaborate. Strong evidence: the claim as
            stated is not yet a Lean statement, and no amount of prover budget
            fixes that.
  MISMATCH  it elaborates, and it says something else. The most dangerous
            outcome, because everything downstream looks like success.

BROKEN and MISMATCH are the findings that matter and the ones nothing previously
detected. An unelaborable statement consumed the full attempt budget every time
it was tried. A MISMATCHED one is worse: on IMO26 P1 the claim "the product of
all integers on the blackboard is invariant under the move" yielded a signature
about SWAPPING two list entries -- true, trivial, and not the claim, since the
move sends m,n to gcd and lcm/gcd and the product is not invariant at all. The
prover proved it in 1,440 seconds and the system banked a verified proof of a
false claim.
"""

import time

from . import config, falsify, prover, review, tactics, trace

SOLVED, OPEN, BROKEN, MISMATCH, FALSE = ("solved", "open", "broken",
                                         "mismatch", "false")


def scout_one(claim: str, lean_statement: str = "",
              budget: float = 0.0) -> dict:
    """Cheapest useful verdict on one claim. Never raises.

    `lean_statement`, when supplied, skips the derivation entirely -- so an
    obligation the agent already wrote a signature for is nearly free to scout.
    """
    budget = budget or config.GT_SCOUT_BUDGET
    t0 = time.time()
    out = {"claim": claim, "status": OPEN, "lean": lean_statement or "",
           "tactic": "", "seconds": 0.0, "note": ""}
    try:
        # tries=2, not 1. BROKEN is the strongest verdict this module emits --
        # it tells the agent no prover budget can help and to restate the claim
        # -- and a single derivation attempt is a ~67% coin (measured: 6/9 with
        # thinking off). One try therefore mislabels a perfectly derivable claim
        # as BROKEN about a third of the time, which sends the agent away from
        # valid work on the strength of one bad sample. A second attempt costs
        # ~35s and cuts that to ~11%.
        #
        # Still not 3: this must stay cheap relative to the proof budget it
        # informs, and the marginal try buys 7 points for another 35 seconds.
        # CHEAPEST test first: a numeric counterexample. Both measured P1
        # falsehoods ("product invariant", "M = product/gcd^2") die here in
        # seconds, before the 100-300s derivation is ever paid for -- and a
        # false claim with a beautiful signature is still false.
        if config.GT_FALSIFY:
            f = falsify.probe(claim)
            if f["status"] == falsify.FALSIFIED:
                out.update(status=FALSE, lean="",
                           note=("numerically falsified -- counterexample: "
                                 + f["counterexample"][:200]))
                return out
        sig = lean_statement or prover.derive_signature(claim, tries=2)
        if not sig:
            out.update(status=BROKEN, note="no signature could be derived")
            return out

        # FAITHFULNESS, before anything else is done with this signature.
        #
        # Elaborating and saying-what-the-claim-says are different properties,
        # and only the first is cheap to check. Measured, on IMO26 P1: the claim
        # "the product of all integers on the blackboard is invariant under the
        # move" produced a signature about SWAPPING two entries of a list, which
        # typechecks, is true, is trivial, and is not the claim -- the move sends
        # m,n to gcd and lcm/gcd, under which the product is NOT invariant. The
        # prover then spent 1,440 seconds proving it, and the system recorded a
        # verified proof of a false claim.
        #
        # This check cannot be left to the caller. `prove_theorem` reviews a
        # signature it DERIVES, but skips the review when one is PINNED -- and
        # handing a signature forward is exactly what this function is for. So
        # scouting without this turns the review gate off for every claim it
        # touches, which is strictly worse than not scouting at all.
        try:
            verdict = review.review_statement(claim, sig + " := by sorry")
        except Exception:
            verdict = {}
        if verdict.get("faithful") is False:
            # The reason lives under "verdict" ("MISMATCH: <why>"), NOT under
            # "reason" -- reading the wrong key produced a rejection with an
            # empty explanation, which tells the agent its signature is wrong
            # and nothing about how to fix it. The explanation IS the payload
            # here: without it "restate the claim" is not an actionable
            # instruction.
            why = (str(verdict.get("verdict", "")).split(":", 1)[-1].strip()
                   or str(verdict.get("note", "")).strip()
                   or "no reason given by the reviewer")
            out.update(status=MISMATCH, lean="", rejected=sig,
                       note=("signature typechecks but does not say what the "
                             "claim says -- " + why[:200]))
            return out
        out["lean"] = sig
        hit = tactics.probe(sig, budget=budget) or {}
        if hit.get("ok") or hit.get("tactic"):
            out.update(status=SOLVED, tactic=str(hit.get("tactic", ""))[:80],
                       note="closed by a single tactic")
        else:
            tried = hit.get("tried") or []
            out.update(status=OPEN,
                       note=f"typechecks; {len(tried)} tactics tried, none closed it")
    except Exception as e:
        # A scout that can break a run is worse than no scout. An unusable
        # verdict is OPEN -- the neutral one -- never BROKEN, which would send
        # the agent away from a claim on the strength of an infrastructure
        # failure rather than a mathematical one.
        out.update(status=OPEN, note=f"scout unavailable ({type(e).__name__})")
    finally:
        out["seconds"] = round(time.time() - t0, 1)
        try:
            # `rejected` carries the signature that FAILED review. out["lean"] is
            # deliberately blanked on a mismatch so nothing downstream can pin an
            # unfaithful statement -- but that also erased the only evidence
            # needed to tell a bad decomposition from an over-eager reviewer.
            # Six mismatches were recorded with the claim and nothing else, and
            # none of them could be audited afterwards.
            trace.record("scout", statement=claim, ok=out["status"] == SOLVED,
                         status=out["status"], elapsed=out["seconds"],
                         sig=out["lean"][:400], tactic=out["tactic"],
                         rejected=out.get("rejected", "")[:400] or None,
                         note=out.get("note", "")[:300] or None)
        except Exception:
            pass
    return out


def scout(claims: list, budget: float = 0.0, limit: int = 0) -> list:
    """Scout several claims, cheapest signal first, capped so this stays cheap."""
    if not config.GT_SCOUT:
        return []
    limit = limit or config.GT_SCOUT_MAX
    return [scout_one(c if isinstance(c, str) else c.get("claim", ""),
                      "" if isinstance(c, str) else c.get("lean", ""), budget)
            for c in list(claims)[:limit]]


def report(scouted: list) -> str:
    """Prompt block. Ordered so the free wins are impossible to miss."""
    if not scouted:
        return ""
    solved = [s for s in scouted if s["status"] == SOLVED]
    broken = [s for s in scouted if s["status"] == BROKEN]
    openish = [s for s in scouted if s["status"] == OPEN]
    lines = ["", "SCOUT REPORT (cheap Lean probes run before you spend a full "
                 "proof budget):"]
    for s in solved:
        lines.append(f"  [FREE WIN] {s['claim'][:110]}")
        lines.append(f"      `{s['tactic']}` closes this outright. Call "
                     "prove_theorem on it now -- it costs seconds, not minutes.")
    for s in broken:
        lines.append(f"  [NOT A LEAN STATEMENT] {s['claim'][:110]}")
        lines.append("      No signature elaborates. Prover budget cannot fix "
                     "this; restate the claim more precisely first.")
    for s in [x for x in scouted if x["status"] == FALSE]:
        lines.append(f"  [FALSE -- COUNTEREXAMPLE FOUND] {s['claim'][:105]}")
        lines.append(f"      {s.get('note', '')[:190]}")
        lines.append("      Do not attempt a proof. Drop the claim or restate it.")
    for s in [x for x in scouted if x["status"] == MISMATCH]:
        lines.append(f"  [SAYS THE WRONG THING] {s['claim'][:110]}")
        lines.append(f"      {s.get('note', '')[:170]}")
        lines.append("      A proof of this would be worthless. Restate the "
                     "claim so a signature can capture it.")
    for s in openish:
        lines.append(f"  [NON-TRIVIAL] {s['claim'][:110]}")
        if s.get("lean"):
            lines.append(f"      signature ready, pass it as lean_statement: "
                         f"{s['lean'][:120]}")
    if openish:
        lines.append("  NON-TRIVIAL means only that no one-liner worked. It is "
                     "not evidence the claim is hard, and not evidence it is false.")
    return "\n".join(lines)
