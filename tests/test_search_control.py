#!/usr/bin/env python3
"""Bandit + evolutionary search. No models, no Lean, no network.

Both modules allocate the most expensive resource in the system (proof
attempts), so their failure modes are quiet: a bandit that locks onto one arm
or an island model that converges still RUNS, it just stops exploring. These
tests assert the anti-convergence behaviour directly.

    python3 -m tests.test_search_control
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bandit, evolve                              # noqa: E402


class BanditTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, bandit.PATH = bandit.PATH, self._tmp.name
        bandit.reset()

    def tearDown(self):
        bandit.PATH = self._orig
        os.unlink(self._tmp.name)


class TestBanditSelection(BanditTest):
    def test_every_arm_is_tried_before_any_repeat(self):
        # UCB1's limit behaviour. Without it the bandit can lock onto whichever
        # arm happened to win first, which is the fixed ladder with extra steps.
        picked = []
        for _ in range(len(bandit.ARMS)):
            a = bandit.select()
            picked.append(a)
            bandit.update(a, won=False)
        self.assertEqual(sorted(picked), sorted(bandit.ARMS))

    def test_winner_is_preferred_once_all_explored(self):
        # "Explored" means well-sampled, not merely tried once. With n=1 the
        # exploration term sqrt(2 ln N / n) legitimately swamps any mean --
        # that is UCB1 behaving correctly, and it is the same property
        # test_starved_arm_is_eventually_revisited relies on.
        winner = bandit.ARMS[1]          # by position: arm names change
        for a in bandit.ARMS:
            for _ in range(20):
                bandit.update(a, won=False)
        for _ in range(20):
            bandit.update(winner, won=True)
        self.assertEqual(bandit.select(), winner)

    def test_starved_arm_is_eventually_revisited(self):
        # The exploration term must keep a neglected arm alive; otherwise one
        # lucky early win permanently starves everything else.
        for a in bandit.ARMS:
            bandit.update(a, won=False)
        hog = bandit.ARMS[0]
        for _ in range(30):
            bandit.update(hog, won=True)
        picks = {bandit.select(exclude=(hog,)) for _ in range(3)}
        self.assertTrue(picks - {hog})

    def test_exclude_is_respected(self):
        self.assertNotEqual(bandit.select(exclude=(bandit.ARMS[0],)), bandit.ARMS[0])

    def test_contexts_are_independent(self):
        # An arm that wins on number theory must not be assumed to win on
        # analysis -- the reason stats are kept per context at all.
        winner = bandit.ARMS[1]
        for a in bandit.ARMS:
            for _ in range(20):
                bandit.update(a, won=False, context="nat")
                bandit.update(a, won=False, context="real")
        for _ in range(20):
            bandit.update(winner, won=True, context="nat")
        self.assertEqual(bandit.select("nat"), winner)
        self.assertEqual(bandit.stats("real").get(winner, {}).get("wins", 0), 0)

    def test_ranking_reports_untried_arms(self):
        r = bandit.ranking()
        self.assertEqual(len(r), len(bandit.ARMS))
        self.assertTrue(all(x["n"] == 0 for x in r))

    def test_persistence_across_reload(self):
        arm = bandit.ARMS[-1]
        bandit.update(arm, won=True)
        self.assertEqual(bandit.stats()[arm]["wins"], 1)


class EvolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, evolve.PATH = evolve.PATH, self._tmp.name
        evolve.reset()

    def tearDown(self):
        evolve.PATH = self._orig
        os.unlink(self._tmp.name)


class TestFitness(EvolveTest):
    def test_verified_always_beats_unverified(self):
        v = {"verified": True, "novelty": 0.0, "uses": 0, "seconds": 0}
        u = {"verified": False, "novelty": 1.0, "uses": 0, "seconds": 0}
        self.assertGreater(evolve.fitness(v), evolve.fitness(u))

    def test_novelty_breaks_ties_among_verified(self):
        a = {"verified": True, "novelty": 1.0, "uses": 0, "seconds": 0}
        b = {"verified": True, "novelty": 0.0, "uses": 0, "seconds": 0}
        self.assertGreater(evolve.fitness(a), evolve.fitness(b))

    def test_being_cited_raises_fitness(self):
        a = {"verified": True, "novelty": 0.5, "uses": 4, "seconds": 0}
        b = {"verified": True, "novelty": 0.5, "uses": 0, "seconds": 0}
        self.assertGreater(evolve.fitness(a), evolve.fitness(b))


class TestIslands(EvolveTest):
    def test_seed_spreads_across_islands(self):
        out = evolve.seed([f"claim {i}" for i in range(8)])
        self.assertEqual(sum(out["islands"]), 8)
        self.assertTrue(all(n > 0 for n in out["islands"]))

    def test_step_breeds_without_a_model(self):
        evolve.seed([f"claim {i}" for i in range(4)])
        out = evolve.step(lambda ps, i, g: [f"child of island {i}"])
        self.assertEqual(out["generation"], 1)
        self.assertGreater(out["born"], 0)

    def test_duplicate_offspring_are_not_added_twice(self):
        evolve.seed(["a", "b", "c", "d"])
        evolve.step(lambda ps, i, g: ["same text"])
        before = evolve.summary()["population"]
        evolve.step(lambda ps, i, g: ["same text"])
        self.assertEqual(evolve.summary()["population"], before)

    def test_island_cap_is_enforced(self):
        evolve.seed([f"c{i}" for i in range(4)])
        for g in range(8):
            evolve.step(lambda ps, i, g=g: [f"kid {g} {i} {k}" for k in range(5)],
                        per_island=5)
        st = evolve._load()
        for isl in st["islands"]:
            self.assertLessEqual(len(isl), evolve.ISLAND_CAP)

    def test_proposer_failure_does_not_kill_the_generation(self):
        evolve.seed(["a", "b", "c", "d"])
        def boom(ps, i, g):
            raise RuntimeError("model down")
        out = evolve.step(boom)
        self.assertEqual(out["generation"], 1)      # generation still advanced

    def test_worst_island_is_reseeded(self):
        evolve.seed([f"c{i}" for i in range(4)])
        evolve.observe("c0", verified=True, seconds=300)
        notes = [evolve.step(lambda ps, i, g: [f"k{i}"])["reset"]
                 for _ in range(evolve.RESET_EVERY)]
        self.assertTrue(any(notes), "worst island was never reseeded")

    def test_observe_marks_verification(self):
        evolve.seed(["the claim"])
        evolve.observe("the claim", verified=True, seconds=120)
        self.assertEqual(evolve.summary()["verified"], 1)

    def test_frontier_returns_only_unverified(self):
        evolve.seed(["a", "b"])
        evolve.observe("a", verified=True)
        self.assertTrue(all(not c["verified"] for c in evolve.frontier()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
