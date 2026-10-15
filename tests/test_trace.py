#!/usr/bin/env python3
"""The recorder must capture everything and be incapable of breaking a run.

Eight runs left twenty retained records. These tests exist so that number can
never silently be twenty again.

    python3 -m tests.test_trace
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import trace                                       # noqa: E402


class TraceTestCase(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self._orig, trace.PATH = trace.PATH, os.path.join(d, "attempts.jsonl")
        trace.begin(run_id="test-run", goal="a test goal")

    def tearDown(self):
        trace.PATH = self._orig

    def lines(self):
        return trace.load()


class TestItWritesWhatHappened(TraceTestCase):
    def test_a_proof_call_is_recorded_with_its_verdict(self):
        trace.proof("gcd(a,b) divides a",
                    {"proved": True, "elapsed": 12.5, "rounds": 2,
                     "lean": "theorem t : True := trivial",
                     "attempts": [{"attempt": 0, "model": "goedel", "mode": "proof",
                                   "ok": True, "reason": "compiled"}]})
        r, = self.lines()
        self.assertEqual(r["kind"], "proof")
        self.assertTrue(r["ok"])
        self.assertEqual(r["elapsed"], 12.5)
        self.assertEqual(r["n_attempts"], 1)

    def test_failures_are_recorded_too(self):
        # A failure marks the frontier. Keeping only successes is exactly the
        # bias that made `exhausted` and `established` contradict each other.
        trace.proof("something false", {"proved": False, "elapsed": 900.0,
                                        "attempts": [{"ok": False, "mode": "proof"}]})
        self.assertFalse(self.lines()[0]["ok"])

    def test_the_per_attempt_history_survives(self):
        # The density is in the history: four failed attempts says far more
        # about difficulty than a bare False does.
        trace.proof("x", {"proved": False, "attempts": [
            {"attempt": i, "mode": "proof", "ok": False, "reason": f"r{i}"}
            for i in range(4)]})
        self.assertEqual(len(self.lines()[0]["history"]), 4)

    def test_signature_derivations_are_recorded(self):
        trace.signature("sqrt 2 is irrational", "theorem s : Irrational (√2)",
                        ok=True, seconds=41.2)
        r, = self.lines()
        self.assertEqual(r["kind"], "signature")
        self.assertTrue(r["ok"])

    def test_records_carry_the_problem_they_belong_to(self):
        trace.signature("x", "y", ok=True)
        self.assertEqual(self.lines()[0]["goal"], "a test goal")

    def test_append_only_never_loses_earlier_rows(self):
        for i in range(25):
            trace.signature(f"claim {i}", "sig", ok=bool(i % 2))
        self.assertEqual(len(self.lines()), 25)


class TestItCannotBreakARun(TraceTestCase):
    """A recorder that can kill a fifteen-hour run is worse than none."""

    def test_an_unwritable_path_is_survived_silently(self):
        trace.PATH = "/proc/nonexistent/nope/attempts.jsonl"
        trace.proof("x", {"proved": True})          # must not raise
        trace.signature("x", "y", ok=True)

    def test_unserialisable_values_are_survived(self):
        trace.record("odd", thing=object(), fn=len)
        self.assertEqual(len(self.lines()), 1)

    def test_a_corrupt_line_does_not_lose_the_file(self):
        trace.signature("good one", "sig", ok=True)
        with open(trace.PATH, "a") as fh:
            fh.write('{"half written', )        # a killed writer's last line
        trace.signature("after", "sig", ok=True)
        rows = self.lines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["statement"], "good one")

    def test_long_proofs_are_clipped_not_dropped(self):
        trace.proof("x", {"proved": True, "lean": "A" * 50_000})
        self.assertLessEqual(len(self.lines()[0]["lean"]), trace._MAX_CODE)


class TestWhatAConsumerCanCompute(TraceTestCase):
    def test_stats_report_pass_rate_per_kind(self):
        for ok in (True, True, False, False, False):
            trace.signature("c", "s", ok=ok, seconds=10)
        s = trace.stats()
        self.assertEqual(s["rows"], 5)
        self.assertEqual(s["by_kind"]["signature"]["rate"], 0.4)

    def test_calibration_fields_round_trip(self):
        # Nothing emits p_success yet. This asserts the CHANNEL exists, so a
        # calibration study is a one-line change at the call site and not a
        # schema migration over a file that is by then large.
        trace.record("proof", statement="x", ok=False, p_success=0.8)
        self.assertEqual(self.lines()[0]["p_success"], 0.8)


class TestItIsActuallyWiredIn(unittest.TestCase):
    """The measured failure mode was not a bad recorder -- it was no recorder."""

    def test_prove_theorem_is_wrapped(self):
        from src import prover
        self.assertTrue(hasattr(prover, "_prove_theorem_untraced"))

    def test_derive_once_is_wrapped(self):
        from src import prover
        self.assertTrue(hasattr(prover, "_derive_once_untraced"))

    def test_the_wrapper_records_without_changing_the_return(self):
        import tempfile as tf
        from src import prover
        d = tf.mkdtemp()
        orig_path, trace.PATH = trace.PATH, os.path.join(d, "a.jsonl")
        sentinel = {"proved": True, "elapsed": 1.0, "attempts": [], "lean": "x"}
        orig = prover._prove_theorem_untraced
        try:
            prover._prove_theorem_untraced = lambda *a, **k: sentinel
            out = prover.prove_theorem("a claim", "")
            self.assertIs(out, sentinel)                 # caller sees it unchanged
            self.assertEqual(len(trace.load()), 1)       # and it was recorded
        finally:
            prover._prove_theorem_untraced = orig
            trace.PATH = orig_path

    def test_a_raising_prover_still_raises(self):
        # Tracing must not swallow real failures into a silent success.
        from src import prover
        orig = prover._prove_theorem_untraced
        try:
            def boom(*a, **k):
                raise RuntimeError("prover died")
            prover._prove_theorem_untraced = boom
            with self.assertRaises(RuntimeError):
                prover.prove_theorem("x", "")
        finally:
            prover._prove_theorem_untraced = orig

    def test_agent_tags_the_run_with_the_problem(self):
        import inspect
        from src import agent
        self.assertIn("trace.begin", inspect.getsource(agent.run))


if __name__ == "__main__":
    unittest.main(verbosity=2)
