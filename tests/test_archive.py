#!/usr/bin/env python3
"""Quality-diversity archive tests.

The central test is `test_the_actual_local_minimum`: it replays the real run --
five restatements of one divisibility fact -- and asserts the archive refuses
to count them as five results. If that test ever passes trivially, the archive
has stopped doing the one job it exists for.

    python3 -m tests.test_archive
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import archive                                     # noqa: E402

GOAL = "Show that the process terminates for all starting pairs (m, n)"

# The five claims the agent actually produced across five sessions.
RESTATEMENTS = [
    "For every natural number n, 2 divides n*(n+1)",
    "2 divides n*(n+1) for all naturals n",
    "The product of two consecutive naturals is even",
    "n*(n+1) is divisible by 2 for every n in the naturals",
    "For all n : ℕ, 2 ∣ n * (n + 1)",
]


class ArchiveTestCase(unittest.TestCase):
    """Every test runs against a private archive file."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig, archive.PATH = archive.PATH, self._tmp.name
        archive.reset()

    def tearDown(self):
        archive.PATH = self._orig
        os.unlink(self._tmp.name)


class TestCells(ArchiveTestCase):
    def test_same_fact_reworded_lands_in_one_cell(self):
        cells = {archive.cell(c, goal=GOAL) for c in RESTATEMENTS}
        self.assertEqual(len(cells), 1, f"expected one cell, got {cells}")

    def test_different_mathematics_lands_elsewhere(self):
        a = archive.cell("2 divides n*(n+1)", goal=GOAL)
        b = archive.cell("Every continuous function on a compact set attains a max",
                         goal=GOAL)
        self.assertNotEqual(a, b)

    def test_cell_is_hashable_and_three_dimensional(self):
        c = archive.cell("2 divides n*(n+1)", goal=GOAL)
        self.assertIsInstance(c, tuple)
        self.assertEqual(len(c), 3)


class TestRoles(unittest.TestCase):
    def test_restatement_of_the_goal_detected(self):
        goal = "Prove that gcd(m,n) times lcm(m,n) equals m times n"
        self.assertEqual(archive.infer_role("gcd(m,n) * lcm(m,n) = m * n", goal),
                         "restatement")

    def test_unrelated_claim_is_not_a_restatement(self):
        goal = "Prove that gcd(m,n) times lcm(m,n) equals m times n"
        self.assertNotEqual(
            archive.infer_role("Every compact metric space is complete", goal),
            "restatement")

    def test_no_goal_yields_unknown(self):
        self.assertEqual(archive.infer_role("anything at all", ""), "unknown")


class TestLocalMinimum(ArchiveTestCase):
    def test_the_actual_local_minimum(self):
        """Five verified restatements must count as ONE result, not five."""
        accepted = 0
        for claim in RESTATEMENTS:
            r = archive.record(claim, ok=True, seconds=30, goal=GOAL)
            accepted += r["accepted"]
        cov = archive.coverage()
        self.assertEqual(cov["verified_cells"], 1)
        self.assertEqual(cov["attempts"], 5)      # all attempts still counted
        self.assertEqual(accepted, 1)             # only the first became elite

    def test_second_restatement_is_reported_occupied(self):
        archive.record(RESTATEMENTS[0], ok=True, seconds=30, goal=GOAL)
        occ = archive.occupied(RESTATEMENTS[1], goal=GOAL)
        self.assertTrue(occ["filled"])
        self.assertIn("2 divides", occ["by"])

    def test_unproved_claim_does_not_fill_a_cell(self):
        archive.record(RESTATEMENTS[0], ok=False, seconds=900, goal=GOAL)
        self.assertFalse(archive.occupied(RESTATEMENTS[1], goal=GOAL)["filled"])
        self.assertEqual(archive.coverage()["verified_cells"], 0)

    def test_genuinely_new_result_is_accepted(self):
        archive.record(RESTATEMENTS[0], ok=True, seconds=30, goal=GOAL)
        r = archive.record("Every compact metric space is complete",
                           ok=True, seconds=100, goal=GOAL)
        self.assertTrue(r["accepted"])
        self.assertEqual(archive.coverage()["verified_cells"], 2)


class TestFitness(ArchiveTestCase):
    def test_verified_beats_unverified(self):
        self.assertGreater(archive.fitness(True, 10), archive.fitness(False, 900))

    def test_hard_win_outranks_trivial_win(self):
        # Optimising for cheap wins is how the greedy policy arose; a result
        # that took real search is worth more as an elite.
        self.assertGreater(archive.fitness(True, 400), archive.fitness(True, 2))

    def test_harder_result_displaces_easier_in_same_cell(self):
        archive.record(RESTATEMENTS[0], ok=True, seconds=5, goal=GOAL)
        r = archive.record(RESTATEMENTS[1], ok=True, seconds=500, goal=GOAL)
        self.assertTrue(r["accepted"])
        self.assertEqual(archive.coverage()["verified_cells"], 1)


class TestGuidance(ArchiveTestCase):
    def test_gaps_offered_once_something_is_recorded(self):
        archive.record(RESTATEMENTS[0], ok=True, seconds=30, goal=GOAL)
        self.assertTrue(archive.gaps(5))

    def test_advice_is_empty_before_any_attempt(self):
        self.assertEqual(archive.advice(), "")

    def test_advice_names_covered_and_uncovered(self):
        archive.record(RESTATEMENTS[0], ok=True, seconds=30, goal=GOAL)
        text = archive.advice(GOAL)
        self.assertIn("UNEXPLORED", text)
        self.assertIn("Already established", text)




class TestKeyIsProofIndependent(ArchiveTestCase):
    """THE bug this suite originally missed.

    occupied() runs BEFORE a proof exists; _record() runs AFTER, with Lean
    source. While the proof contributed to the key, the two computed different
    cells for the same claim, so a recorded cell could never block a later
    check -- the guard was inert in production. The original local-minimum test
    passed only because it called record() with no code, making both sides
    accidentally agree.
    """

    CLAIM = "For all naturals a b, gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)"
    CODE = "theorem gcd_min_diff (a b : ℕ) : True := by omega <;> simp <;> linarith"

    def test_cell_ignores_proof_source(self):
        self.assertEqual(archive.cell(self.CLAIM, "", GOAL),
                         archive.cell(self.CLAIM, self.CODE, GOAL))

    def test_recording_with_code_blocks_a_later_check_without_code(self):
        archive.record(self.CLAIM, ok=True, seconds=800, code=self.CODE, goal=GOAL)
        self.assertTrue(archive.occupied(self.CLAIM, goal=GOAL)["filled"])

    def test_production_shape_end_to_end(self):
        # Exactly how prover._record and prover.prove_theorem call it.
        archive.record(self.CLAIM, ok=True, seconds=874.3, code=self.CODE, goal=GOAL)
        restated = "gcd(min(a,b), max(a,b) - min(a,b)) equals gcd(a,b) for naturals"
        self.assertTrue(archive.occupied(restated, goal=GOAL)["filled"])
        self.assertEqual(archive.coverage()["verified_cells"], 1)

    def test_prose_mechanism_is_detected(self):
        self.assertEqual(archive.cell("The quantity is invariant under a move",
                                      goal=GOAL)[0], "invariant")
        self.assertEqual(archive.cell("The process terminates after finitely many moves",
                                      goal=GOAL)[0], "termination")


if __name__ == "__main__":
    unittest.main(verbosity=2)
