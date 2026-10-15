#!/usr/bin/env python3
"""The obligation ledger and enforced closure. No models, no network.

Motivated by the only measurement that matters here: across ten hours and five
IMO problems, `write_solution` was called ZERO times and every `prove_theorem`
call was a sub-lemma. Nothing ever tried to finish.

    python3 -m tests.test_plan
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import plan                                        # noqa: E402

GOAL = ("There are 2026 integers greater than 1 on a blackboard. A move replaces "
        "m,n with gcd(m,n) and lcm(m,n)/gcd(m,n). Determine the final configuration.")

LEDGER = [
    {"id": "answer", "kind": "answer", "claim": "The final board is pairwise coprime",
     "why": "", "status": plan.OPEN, "note": "", "attempts": 0},
    {"id": "inv", "kind": "lemma",
     "claim": "gcd(min(a,b), max(a,b) - min(a,b)) = gcd(a,b) for naturals a b",
     "why": "", "status": plan.OPEN, "note": "", "attempts": 0},
    {"id": "term", "kind": "lemma", "claim": "The process terminates after finitely many moves",
     "why": "", "status": plan.OPEN, "note": "", "attempts": 0},
    {"id": "closure", "kind": "closure", "claim": GOAL,
     "why": "the problem itself", "status": plan.OPEN, "note": "", "attempts": 0},
]


class PlanTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, plan.PATH = plan.PATH, self._tmp.name
        plan.reset()
        plan._save({plan._key(GOAL): {"goal": GOAL, "built": "now",
                                      "obligations": [dict(o) for o in LEDGER]}})

    def tearDown(self):
        plan.PATH = self._orig
        os.unlink(self._tmp.name)


class TestLedgerShape(PlanTestCase):
    def test_progress_counts_obligations(self):
        p = plan.progress(GOAL)
        self.assertEqual(p["total"], 4)
        self.assertEqual(p["proved"], 0)
        self.assertEqual(p["next"], "answer")

    def test_a_closure_obligation_always_exists(self):
        kinds = {o["kind"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertIn("closure", kinds)

    def test_determine_problems_get_an_answer_obligation(self):
        # "Determine the final configuration" needs the ANSWER stated, not just
        # lemmas proved. A ledger of pure lemmas drops half the problem.
        kinds = {o["kind"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertIn("answer", kinds)


class TestDischarge(PlanTestCase):
    def test_a_proof_moves_the_matching_obligation(self):
        d = plan.discharge(GOAL,
                           "For all naturals a b, gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)",
                           ok=True)
        self.assertEqual(d["matched"], "inv")
        self.assertEqual(plan.progress(GOAL)["proved"], 1)

    def test_matching_is_fuzzy_not_verbatim(self):
        # The agent rarely restates an obligation word for word; demanding it
        # would leave the ledger untouched while real work happened beside it.
        d = plan.discharge(GOAL, "the process halts after finitely many moves", ok=True)
        self.assertEqual(d["matched"], "term")

    def test_an_unrelated_claim_matches_nothing(self):
        d = plan.discharge(GOAL, "Every compact metric space is complete", ok=True)
        self.assertIsNone(d["matched"])
        self.assertEqual(plan.progress(GOAL)["proved"], 0)

    def test_repeated_failure_marks_the_obligation_failed(self):
        for _ in range(3):
            plan.discharge(GOAL, "The process terminates after finitely many moves",
                           ok=False, note="prover timed out")
        st = {o["id"]: o["status"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertEqual(st["term"], plan.FAILED)

    def test_a_later_success_overrides_earlier_failure(self):
        plan.discharge(GOAL, "The process terminates after finitely many moves", ok=False)
        plan.discharge(GOAL, "The process terminates after finitely many moves", ok=True)
        st = {o["id"]: o["status"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertEqual(st["term"], plan.PROVED)


class TestClosureTriggers(PlanTestCase):
    def test_does_not_close_early_with_nothing_proved(self):
        self.assertFalse(plan.should_close(GOAL, 0.1)["close"])

    def test_closes_when_everything_is_discharged(self):
        for o in LEDGER:
            plan.discharge(GOAL, o["claim"], ok=True)
        c = plan.should_close(GOAL, 0.1)
        self.assertTrue(c["close"])
        self.assertEqual(c["mode"], "complete")

    def test_closes_on_budget_with_partial_progress(self):
        # THE important trigger: on a hard problem not everything will be
        # proved, and waiting for that means writing nothing at all.
        plan.discharge(GOAL, "gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)", ok=True)
        c = plan.should_close(GOAL, 0.8)
        self.assertTrue(c["close"])
        self.assertEqual(c["mode"], "partial")

    def test_budget_trigger_needs_something_to_write_up(self):
        # Forcing a write-up with zero verified results would just produce prose.
        self.assertFalse(plan.should_close(GOAL, 0.95)["close"])

    def test_no_ledger_never_forces_closure(self):
        self.assertFalse(plan.should_close("a goal with no ledger", 0.99)["close"])


class TestBrief(PlanTestCase):
    def test_brief_shows_state_and_scopes_the_turn(self):
        out = plan.brief(GOAL, 0.1)
        self.assertIn("SOLUTION CHECKLIST", out)
        self.assertIn("[ ]", out)
        self.assertIn("ONE named obligation", out)

    def test_brief_marks_discharged_items(self):
        plan.discharge(GOAL, "gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)", ok=True)
        self.assertIn("[x]", plan.brief(GOAL, 0.1))

    def test_brief_orders_the_close_when_triggered(self):
        plan.discharge(GOAL, "gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)", ok=True)
        out = plan.brief(GOAL, 0.9)
        self.assertIn("CLOSE NOW", out)
        self.assertIn("write_solution", out)

    def test_brief_is_empty_without_a_ledger(self):
        self.assertEqual(plan.brief("unknown goal", 0.5), "")


class TestEnforcement(unittest.TestCase):
    """Advisory did not work: write_solution was called zero times, ever."""

    def test_agent_withholds_exploration_tools_on_close(self):
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertIn("should_close", s)
        self.assertIn('"conjecture", "hunches"', s)

    def test_agent_builds_the_ledger_once_at_start(self):
        import inspect
        from src import agent
        self.assertIn("plan.build(goal, specs=", inspect.getsource(agent.run))

    def test_agent_discharges_on_every_proof(self):
        import inspect
        from src import agent
        self.assertIn("plan.discharge", inspect.getsource(agent.run))

    def test_checklist_reaches_the_prompt(self):
        import inspect
        from src import agent
        self.assertIn("plan.brief", inspect.getsource(agent._anchor_message))

    def test_write_solution_is_never_withheld(self):
        # Whatever else is suppressed, the ability to FINISH must survive.
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertNotIn('"write_solution"', s.split("suppress")[1][:400])




class TestLedgerNeverEmpty(unittest.TestCase):
    """An empty ledger silently disables closure entirely. The live run printed
    `[plan] 0 obligation(s)` after one HTTP 500 and reverted to exactly the
    open-ended behaviour the ledger exists to stop."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, plan.PATH = plan.PATH, self._tmp.name
        plan.reset()

    def tearDown(self):
        plan.PATH = self._orig
        os.unlink(self._tmp.name)

    def test_failed_decomposition_still_yields_a_closure_obligation(self):
        led = plan.build("Prove that the process terminates.",
                         model="definitely-not-a-real-model")
        obs = led["obligations"]
        self.assertTrue(obs, "ledger must never be empty")
        self.assertTrue(any(o["kind"] == "closure" for o in obs))

    def test_minimal_ledger_can_still_force_a_close(self):
        goal = "Prove that the process terminates."
        plan.build(goal, model="definitely-not-a-real-model")
        plan.discharge(goal, goal, ok=True)          # the closure obligation
        self.assertTrue(plan.should_close(goal, 0.9)["close"])

    def test_minimal_ledger_renders_a_checklist(self):
        goal = "Prove that the process terminates."
        plan.build(goal, model="definitely-not-a-real-model")
        self.assertIn("SOLUTION CHECKLIST", plan.brief(goal, 0.1))

    def test_build_reports_rather_than_swallows(self):
        import inspect
        s = inspect.getsource(plan.build)
        self.assertIn("decomposition failed", s)
        self.assertIn("range(0 if got else 2)", s)                  # retried




class TestFailureIsNotSticky(unittest.TestCase):
    """One HTTP 500 wrote a zero-obligation ledger to disk; every later run
    returned it from cache instantly, making the retry, the error report and
    the minimal-ledger fallback all unreachable."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, plan.PATH = plan.PATH, self._tmp.name
        plan.reset()

    def tearDown(self):
        plan.PATH = self._orig
        os.unlink(self._tmp.name)

    def test_a_degraded_ledger_is_marked(self):
        led = plan.build("Prove X terminates.", model="definitely-not-a-real-model")
        self.assertTrue(led.get("degraded"))

    def test_a_degraded_ledger_is_not_a_cache_hit(self):
        goal = "Prove X terminates."
        plan.build(goal, model="definitely-not-a-real-model")
        # A second call must RETRY rather than return the cached failure. With
        # a dead model it degrades again -- but it tried, which is the point.
        import inspect
        self.assertIn('cached.get("degraded")', inspect.getsource(plan.build))
        self.assertTrue(plan.build(goal, model="definitely-not-a-real-model")["obligations"])

    def test_a_good_ledger_is_cached(self):
        goal = "some goal"
        plan._save({plan._key(goal): {"goal": goal, "built": "now", "degraded": False,
                                      "obligations": [dict(o) for o in LEDGER]}})
        # Must NOT rebuild (a dead model would wipe it if it did).
        self.assertEqual(len(plan.build(goal, model="dead-model")["obligations"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRefutedCutsTheConjunction(PlanTestCase):
    """Closure needs open==0. A false obligation can never be discharged, so it
    blocks completion forever -- and end-to-end is a PRODUCT over obligations,
    measured at 0.000% for six. Removing a term beats improving one."""

    def test_a_refuted_obligation_is_not_open(self):
        plan.refute(GOAL, "The process terminates after finitely many moves",
                    "counterexample: m=2,n=2")
        self.assertEqual(plan.progress(GOAL)["refuted"], 1)
        st = {o["id"]: o["status"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertEqual(st["term"], plan.REFUTED)

    def test_refuting_unblocks_closure_that_could_never_fire(self):
        # THE case. One false obligation, the rest discharged: without REFUTED
        # `open` never reaches zero and the run can never complete.
        for o in LEDGER:
            if o["id"] != "term":
                plan.discharge(GOAL, o["claim"], ok=True)
        self.assertFalse(plan.should_close(GOAL, 0.1)["close"])
        plan.refute(GOAL, "The process terminates after finitely many moves",
                    "counterexample: n=3")
        c = plan.should_close(GOAL, 0.1)
        self.assertTrue(c["close"])
        self.assertEqual(c["mode"], "complete")

    def test_a_refuted_obligation_leaves_the_denominator(self):
        # Scoring against a claim known false would understate the real result.
        plan.discharge(GOAL, LEDGER[1]["claim"], ok=True)
        before = plan.progress(GOAL)["fraction"]
        plan.refute(GOAL, "The process terminates after finitely many moves", "x")
        self.assertGreater(plan.progress(GOAL)["fraction"], before)

    def test_a_proof_cannot_resurrect_a_refuted_claim(self):
        # If a counterexample AND a proof both exist, the SIGNATURE is wrong --
        # exactly the mis-formalization this repo has already banked once.
        plan.refute(GOAL, "The process terminates after finitely many moves", "x")
        plan.discharge(GOAL, "The process terminates after finitely many moves",
                       ok=True)
        st = {o["id"]: o["status"] for o in plan.ledger(GOAL)["obligations"]}
        self.assertEqual(st["term"], plan.REFUTED)

    def test_the_brief_shows_the_counterexample_and_forbids_proving_it(self):
        plan.refute(GOAL, "The process terminates after finitely many moves",
                    "numerically falsified -- counterexample: n=3 halts at 2")
        out = plan.brief(GOAL, 0.1)
        self.assertIn("[FALSE]", out)
        self.assertIn("n=3", out)
        self.assertIn("Do not prove it", out)

    def test_an_unrelated_claim_refutes_nothing(self):
        self.assertIsNone(plan.refute(GOAL, "Every compact space is complete",
                                      "x")["matched"])

    def test_scout_false_reaches_the_ledger(self):
        import inspect
        from src import gt
        s = inspect.getsource(gt.prepare)
        self.assertIn("scout.FALSE", s)
        self.assertIn("plan.refute", s)
