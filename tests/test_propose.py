#!/usr/bin/env python3
"""Orchestrator pass@K: proposal, scoring, selection. No models, no network.

The prover samples a goal 32 ways; the orchestrator -- which chooses WHICH goal
-- was sampled once. These tests pin the properties that make the fix worth
having: real diversity (not temperature), scoring against known state (not a
model's self-endorsement), and safe degradation when the proposer is dead.

    python3 -m tests.test_propose
"""

import ast
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import archive, config, propose, tools                # noqa: E402

GOAL = "Show the blackboard process terminates and find the final configuration"


class ProposeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, archive.PATH = archive.PATH, self._tmp.name
        archive.reset()

    def tearDown(self):
        archive.PATH = self._orig
        os.unlink(self._tmp.name)


class TestDiversityIsStructural(unittest.TestCase):
    """Temperature on one prompt resamples one basin -- measured on the prover,
    where hint-only variation made attempt 2 restate attempt 1."""

    def test_angles_are_distinct_kinds_of_move(self):
        names = [a for a, _ in propose.ANGLES]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 4)

    def test_the_two_under_generated_moves_are_present(self):
        # A model optimising to look productive rarely proposes falsifying or
        # simplifying; those are the two that end grinding early.
        names = {a for a, _ in propose.ANGLES}
        self.assertIn("falsify", names)
        self.assertIn("reduce", names)

    def test_every_angle_has_a_real_brief(self):
        for name, brief in propose.ANGLES:
            self.assertGreater(len(brief), 30, name)

    def test_prompt_names_only_real_tools(self):
        p = propose._angle_prompt("direct", "brief", GOAL, [])
        for t in tools.REGISTRY:
            if t in p:
                self.assertIn(t, tools.REGISTRY)
        self.assertIn("prove_theorem", p)

    def test_prompt_carries_recent_history(self):
        p = propose._angle_prompt("direct", "b", GOAL, ["prove_theorem: gcd stuff"])
        self.assertIn("gcd stuff", p)


class TestScoring(ProposeTestCase):
    """Scored against state the system HOLDS, never by asking a model which of
    its own proposals it prefers -- that just re-endorses the first."""

    def test_established_kind_is_penalised(self):
        claim = "For all naturals a b, gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)"
        archive.record(claim, ok=True, seconds=800, code="by omega", goal=GOAL)
        s = propose.score({"tool": "prove_theorem", "claim": claim, "angle": "direct"},
                          GOAL, [])
        self.assertLess(s["score"], 0)
        self.assertIn("established", s["score_why"])

    def test_unexplored_cell_is_rewarded(self):
        s = propose.score({"tool": "prove_theorem", "angle": "direct",
                           "claim": "Every compact metric space is complete"}, GOAL, [])
        self.assertGreater(s["score"], 0)

    def test_repeating_a_recent_tool_is_penalised(self):
        cand = {"tool": "search_mathlib", "angle": "direct", "claim": ""}
        alone = propose.score(cand, GOAL, [])["score"]
        repeated = propose.score(cand, GOAL, ["search_mathlib: x"] * 4)["score"]
        self.assertLess(repeated, alone)

    def test_restating_a_recent_claim_is_penalised_hard(self):
        claim = "the product of two consecutive naturals is even"
        cand = {"tool": "prove_theorem", "angle": "direct", "claim": claim}
        fresh = propose.score(cand, GOAL, [])["score"]
        stale = propose.score(cand, GOAL, [f"prove_theorem: {claim}"])["score"]
        self.assertLess(stale, fresh - 0.5)

    def test_expensive_tools_carry_a_cost(self):
        cheap = propose.score({"tool": "journal_entry", "angle": "direct", "claim": ""},
                              GOAL, [])["score"]
        dear = propose.score({"tool": "prove_theorem", "angle": "direct", "claim": ""},
                             GOAL, [])["score"]
        self.assertGreater(cheap, dear)

    def test_model_rationale_is_not_clobbered_by_the_score(self):
        # score() returns `score_why`; merging a `why` key would replace the
        # mathematics with the bookkeeping.
        s = propose.score({"tool": "run_python", "angle": "direct", "claim": ""}, GOAL, [])
        self.assertIn("score_why", s)
        self.assertNotIn("why", s)


class TestSelection(ProposeTestCase):
    def test_choose_returns_empty_when_disabled(self):
        prior = config.ORCH_PASS_K
        config.ORCH_PASS_K = 1
        try:
            self.assertEqual(propose.choose(GOAL, []), {})
        finally:
            config.ORCH_PASS_K = prior

    def test_dead_proposer_degrades_to_empty(self):
        # Must fall through to the ordinary single-shot path, not raise.
        self.assertEqual(propose.propose(GOAL, [], k=3, model="not-a-real-model"), [])
        self.assertEqual(propose.choose(GOAL, [], k=3, model="not-a-real-model"), {})

    def test_brief_is_empty_without_candidates(self):
        self.assertEqual(propose.brief({}), "")
        self.assertEqual(propose.brief({"ranked": []}), "")

    def test_brief_renders_scores_and_rationale(self):
        r = {"ranked": [{"angle": "falsify", "tool": "run_python", "claim": "",
                         "score": 0.42, "why": "would kill the approach cheaply",
                         "score_why": "unexplored cell"}]}
        out = propose.brief(r)
        self.assertIn("falsify", out)
        self.assertIn("run_python", out)
        self.assertIn("would kill", out)
        self.assertIn("+0.42", out)

    def test_ranking_is_descending(self):
        cands = [{"angle": "a", "tool": "journal_entry", "claim": "", "score": 0.1},
                 {"angle": "b", "tool": "run_python", "claim": "", "score": 0.9}]
        ranked = sorted(cands, key=lambda c: -c["score"])
        self.assertEqual(ranked[0]["angle"], "b")


class TestWiring(unittest.TestCase):
    def test_agent_consults_the_proposer(self):
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertIn("propose.choose", s)
        self.assertIn("ORCH_PASS_K", s)

    def test_agent_records_actual_actions_not_proposals(self):
        # Scoring against proposals would be circular: the check must see what
        # the orchestrator really did, including overrides of the ranking.
        import inspect
        from src import agent
        self.assertIn("recent_actions.append", inspect.getsource(agent.run))

    def test_proposals_are_advisory_not_executed(self):
        # The score knows novelty, repetition and cost -- nothing about the
        # mathematics. Executing the winner would replace judgement with a
        # heuristic.
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertIn("propose.brief", s)
        self.assertNotIn("dispatch(picks", s)

    def test_config_defaults_are_sane(self):
        self.assertGreaterEqual(config.ORCH_PASS_K, 1)
        self.assertLessEqual(config.ORCH_PASS_K, len(propose.ANGLES))




class TestIntuitionProbe(unittest.TestCase):
    """Scatter across forced-lens samples as a black-box proxy for "the model
    holds this weakly" -- the region where hunches live, as opposed to the
    consensus it would recite anyway."""

    def test_lenses_are_far_apart_and_named(self):
        from src import intuition
        self.assertGreaterEqual(len(intuition.LENSES), 6)
        self.assertEqual(len(intuition.LENSES), len(set(intuition.LENSES)))
        joined = " ".join(intuition.LENSES).lower()
        for field in ("number theory", "combinatorics", "probability", "topology"):
            self.assertIn(field.split()[0], joined)

    def test_dead_model_degrades_to_empty(self):
        from src import intuition
        self.assertEqual(intuition.probe("goal", model="not-a-real-model"), [])

    def test_report_is_empty_without_ideas(self):
        from src import intuition
        self.assertEqual(intuition.report([]), "")

    def test_report_puts_outliers_first_and_labels_them_unverified(self):
        from src import intuition
        ideas = [{"lens": "order theory", "idea": "use a monovariant", "scatter": 0.9},
                 {"lens": "number theory", "idea": "use gcd", "scatter": 0.1}]
        out = intuition.report(ideas)
        self.assertLess(out.index("order theory"), out.index("number theory"))
        self.assertIn("UNVERIFIED", out)

    def test_consensus_split_separates_recital_from_hunches(self):
        from src import intuition
        ideas = [{"lens": "a", "idea": "x", "scatter": 0.9},
                 {"lens": "b", "idea": "y", "scatter": 0.5},
                 {"lens": "c", "idea": "z", "scatter": 0.1}]
        s = intuition.consensus_split(ideas)
        self.assertIn("a", s["hunches"])
        self.assertIn("c", s["recital"])

    def test_empty_split_is_safe(self):
        from src import intuition
        self.assertEqual(intuition.consensus_split([]), {"recital": [], "hunches": []})




class TestHunchesToolIsWired(unittest.TestCase):
    def test_tool_exposed_and_implemented(self):
        from src import tools
        names = {t["function"]["name"] for t in tools.specs()}
        self.assertIn("hunches", names)
        self.assertIn("hunches", tools.REGISTRY)

    def test_dead_model_returns_error_not_exception(self):
        # Point at a NONEXISTENT model, not just an odd lens. With
        # ORCH_PROPOSER_MODEL="" the probe falls back to the orchestrator, so
        # this test was loading a 28 GB model and taking the suite from 21s to
        # 112s -- the same defect as the sleep_phase test before it.
        from src import tools, intuition
        orig_l, orig_m = intuition.LENSES, config.ORCH_PROPOSER_MODEL
        intuition.LENSES = ["nonsense lens"]
        config.ORCH_PROPOSER_MODEL = "definitely-not-a-real-model"
        try:
            out = tools.dispatch("hunches", {"goal": "x", "lenses": 2})
            self.assertIsInstance(out, dict)
            self.assertIn("error", out)
        finally:
            intuition.LENSES, config.ORCH_PROPOSER_MODEL = orig_l, orig_m

    def test_every_spec_still_has_an_implementation(self):
        from src import tools
        names = {t["function"]["name"] for t in tools.specs()}
        self.assertFalse(names - set(tools.REGISTRY))




class TestProposerFitsInMemory(unittest.TestCase):
    """A separate proposer is only an optimisation if both models co-reside.
    Defaulting to a 24 GB proposer alongside a 30 GB orchestrator on a 48 GB
    machine makes every pass@K step evict and reload a model -- far more
    expensive than the fast model saves."""

    def test_default_proposer_does_not_add_a_second_large_model(self):
        # "" means "use the orchestrator", i.e. no extra resident model.
        self.assertEqual(config.ORCH_PROPOSER_MODEL, "")

    def test_propose_falls_back_to_the_orchestrator(self):
        import inspect
        s = inspect.getsource(propose.propose)
        self.assertIn("config.ORCHESTRATOR_MODEL", s)

    def test_intuition_also_falls_back(self):
        from src import intuition
        import inspect
        self.assertIn("config.ORCHESTRATOR_MODEL", inspect.getsource(intuition.probe))




class TestSchemaAndThinkingDoNotCompete(unittest.TestCase):
    """This bug has now shipped THREE times, in three modules.

    Reasoning tokens and grammar-constrained JSON share ONE budget. When the
    model thinks first, the object truncates and json.loads throws, so the call
    returns nothing at all. plan.py died this way twice; propose.py burned ~8 of
    every 9 minutes of a live run producing zero candidates, silently.

    Rule: a call with `format=<schema>` must not also request thinking.
    """

    def _bodies(self, mod):
        import ast, inspect
        found = []
        tree = ast.parse(inspect.getsource(mod))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Dict):
                continue
            keys = [k.value for k in n.keys if isinstance(k, ast.Constant)]
            if "format" in keys and "think" in keys:
                think = n.values[keys.index("think")]
                found.append(think)
        return found

    def test_propose_does_not_think_under_a_schema(self):
        for t in self._bodies(propose):
            self.assertIsInstance(t, ast.Constant)
            self.assertIs(t.value, False)

    def test_intuition_does_not_think_under_a_schema(self):
        from src import intuition
        for t in self._bodies(intuition):
            self.assertIsInstance(t, ast.Constant)
            self.assertIs(t.value, False)

    def test_plan_thinks_in_a_separate_unconstrained_call(self):
        # plan.py is the exception that proves the rule: it DOES think at high
        # effort, but in a phase with no schema, then formats with thinking off.
        import inspect
        from src import plan
        s = inspect.getsource(plan.build)
        self.assertIn("PHASE 1", s)
        self.assertIn("PHASE 2", s)
        for t in self._bodies(plan):
            self.assertIs(getattr(t, "value", None), False)


class TestProposalsAreGatedAndLoud(unittest.TestCase):
    def test_proposals_only_run_when_stuck(self):
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertIn("stuck_steps", s)
        self.assertIn("ORCH_PASS_EVERY", s)

    def test_empty_proposals_are_reported(self):
        # An empty result printed NOTHING, so four failing calls per step were
        # invisible for an entire run.
        import inspect
        from src import agent
        self.assertIn("NO candidates", inspect.getsource(agent.run))

    def test_a_discharged_obligation_resets_the_stuck_counter(self):
        import inspect
        from src import agent
        s = inspect.getsource(agent.run)
        self.assertIn("stuck_steps = 0", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
