#!/usr/bin/env python3
"""The game-theoretic layer: recall, scout, forecast, retirement.

No network. Everything that would call a model or a compiler is injected.

    python3 -m tests.test_gt
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, forecast, gt, recall, scout, trace   # noqa: E402

ROWS = [
    {"kind": "proof", "statement": "gcd(m,n) divides both m and n", "ok": True,
     "elapsed": 40.0, "n_attempts": 1},
    {"kind": "proof", "statement": "For positive m and n, gcd(m,n) divides both m and n",
     "ok": True, "elapsed": 50.0, "n_attempts": 1},
    {"kind": "proof", "statement": "the process terminates after finitely many moves",
     "ok": False, "elapsed": 2280.0, "n_attempts": 2,
     "history": [{"ok": False, "reason": "no tactic worked (12 tried)"}]},
    {"kind": "proof", "statement": "every compact metric space is complete",
     "ok": True, "elapsed": 10.0, "n_attempts": 1},
]


class RecallTestCase(unittest.TestCase):
    def setUp(self):
        # Lexical similarity throughout: deterministic, and it exercises the
        # documented fallback rather than whatever the embedder happens to say.
        self._sim, recall._similarity = recall._similarity, \
            lambda c, ts: [recall._jaccard(c, t) for t in ts]
        self._lem, recall.lemmas = recall.lemmas, lambda c, k=3, floor=-1.0: []

    def tearDown(self):
        recall._similarity, recall.lemmas = self._sim, self._lem


class TestRecallIsPushNotPull(RecallTestCase):
    """The premise: the agent does not call tools it is merely offered."""

    def test_the_expensive_failure_is_surfaced_with_its_price(self):
        out = recall.report("does the process terminate after finitely many moves?",
                            rows=ROWS)
        self.assertIn("FAILED before", out)
        self.assertIn("38 min", out)          # 2280s, rendered as minutes

    def test_why_it_failed_is_carried_not_just_that_it_failed(self):
        # Paraphrased, not verbatim: an exact restatement is excluded as being
        # the claim itself rather than evidence about it.
        out = recall.report("show the process terminates in finitely many moves",
                            rows=ROWS)
        self.assertIn("no tactic worked", out)

    def test_a_verified_result_is_marked_differently_from_a_failure(self):
        out = recall.report("gcd(m,n) divides m", rows=ROWS)
        self.assertIn("PROVED before", out)

    def test_restatements_of_one_fact_collapse_to_one_line(self):
        # The measured failure mode, reproduced inside the retriever: two
        # wordings of "gcd divides both" were being injected as two findings.
        out = recall.report("gcd(m,n) divides m", rows=ROWS)
        self.assertEqual(out.count("[PROVED before]"), 1)

    def test_it_is_silent_when_nothing_is_relevant(self):
        # A retriever that always fills its quota injects noise on exactly the
        # turns it has nothing to say, and context is the scarce resource.
        self.assertEqual(recall.report("the Riemann zeta function on the "
                                       "critical strip", rows=ROWS), "")

    def test_the_claim_itself_is_not_returned_as_evidence(self):
        ev = recall.evidence("every compact metric space is complete", rows=ROWS)
        self.assertTrue(all("compact metric" not in r["statement"] for r in ev))

    def test_disabling_the_flag_silences_it(self):
        config.GT_RECALL = False
        try:
            self.assertEqual(recall.report("gcd(m,n) divides m", rows=ROWS), "")
        finally:
            config.GT_RECALL = True


class TestSeparationTest(unittest.TestCase):
    """A fixed cosine cutoff recommended a triangle-angle lemma for a claim
    about a process terminating. These pin the fix."""

    def test_a_clear_match_survives(self):
        self.assertEqual(recall._relevant([0.868, 0.843, 0.749, 0.558], 0.34)[0], True)

    def test_an_undifferentiated_pack_is_all_rejected(self):
        # 0.538/0.532/0.521/0.463 -- nothing matched; ordering inside is noise.
        self.assertFalse(any(recall._relevant([0.538, 0.532, 0.521, 0.463], 0.34)))

    def test_a_real_match_at_a_low_absolute_score_still_survives(self):
        # The geometry case: 0.541 is a genuine match but below any cutoff that
        # would reject the pack above. Separation gets both right.
        self.assertTrue(recall._relevant([0.541, 0.484, 0.471, 0.416], 0.34)[0])

    def test_the_absolute_floor_still_applies(self):
        self.assertFalse(any(recall._relevant([0.2, 0.05, 0.04, 0.01], 0.34)))


class TestScout(unittest.TestCase):
    """Stackelberg: observe the follower before committing the leader's budget."""

    def setUp(self):
        self._probe, self._derive = scout.tactics.probe, scout.prover.derive_signature
        self._review = scout.review.review_statement
        scout.prover.derive_signature = lambda c, tries=1: "theorem t : True := by"
        scout.review.review_statement = lambda c, code: {"faithful": True, "reason": ""}

    def tearDown(self):
        scout.tactics.probe, scout.prover.derive_signature = self._probe, self._derive
        scout.review.review_statement = self._review

    def test_a_tactic_that_closes_it_is_a_free_win(self):
        scout.tactics.probe = lambda sig, budget=0: {"ok": True, "tactic": "simp"}
        self.assertEqual(scout.scout_one("x")["status"], scout.SOLVED)

    def test_broken_is_not_declared_on_a_single_derivation_attempt(self):
        # BROKEN tells the agent to abandon the claim. At the measured ~67%
        # per-attempt derivation rate, deciding that on one sample is wrong
        # about a third of the time.
        seen = []
        scout.prover.derive_signature = lambda c, tries=1: (seen.append(tries) or "")
        scout.scout_one("x")
        self.assertGreaterEqual(seen[0], 2)

    def test_an_underivable_signature_is_reported_as_broken(self):
        # The finding nothing currently detects: an unelaborable statement
        # consumed a full attempt budget every time and no prover could fix it.
        scout.prover.derive_signature = lambda c, tries=1: ""
        self.assertEqual(scout.scout_one("nonsense")["status"], scout.BROKEN)

    def test_nothing_trivial_working_is_only_reported_as_non_trivial(self):
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": ["simp"] * 12}
        self.assertEqual(scout.scout_one("x")["status"], scout.OPEN)

    def test_the_derived_signature_is_handed_forward(self):
        # This is what makes scouting free for anything later attempted.
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        self.assertTrue(scout.scout_one("x")["lean"])

    def test_infrastructure_failure_degrades_to_open_not_broken(self):
        # BROKEN sends the agent away from a claim. An HTTP error must never
        # be allowed to do that.
        def boom(*a, **k):
            raise RuntimeError("repl down")
        scout.tactics.probe = boom
        self.assertEqual(scout.scout_one("x")["status"], scout.OPEN)

    def test_a_signature_that_says_the_wrong_thing_is_caught(self):
        # THE expensive failure mode. On IMO26 P1 a claim about a gcd/lcm move
        # produced a signature about swapping two list entries: true, trivial,
        # not the claim. It was proved in 1,440s and banked as a verified result.
        scout.tactics.probe = lambda sig, budget=0: {"ok": True, "tactic": "simp"}
        scout.review.review_statement = lambda c, code: {
            "faithful": False, "reason": "proves a swap, not the move"}
        out = scout.scout_one("product is invariant under the move")
        self.assertEqual(out["status"], scout.MISMATCH)

    def test_the_reviewers_reason_survives_into_the_report(self):
        # review_statement returns the explanation under "verdict", not
        # "reason". Reading the wrong key rejected signatures with a blank
        # explanation, which is an unactionable instruction.
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        scout.review.review_statement = lambda c, code: {
            "faithful": False, "verdict": "MISMATCH: proves a swap, not the move"}
        self.assertIn("proves a swap", scout.scout_one("x")["note"])

    def test_a_reviewer_giving_no_reason_still_says_something(self):
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        scout.review.review_statement = lambda c, code: {"faithful": False}
        self.assertIn("no reason given", scout.scout_one("x")["note"])

    def test_a_rejected_signature_is_kept_for_audit(self):
        # Blanking `lean` protects downstream; it must not destroy the evidence
        # that distinguishes a bad decomposition from an over-eager reviewer.
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        scout.review.review_statement = lambda c, code: {
            "faithful": False, "verdict": "MISMATCH: wrong"}
        out = scout.scout_one("x")
        self.assertEqual(out["lean"], "")
        self.assertTrue(out.get("rejected"))

    def test_an_unfaithful_signature_is_never_handed_forward(self):
        # Handing it on PINS it, and prove_theorem skips its review gate for a
        # pinned signature -- so passing it forward disables the very check that
        # would have caught it.
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        scout.review.review_statement = lambda c, code: {
            "faithful": False, "verdict": "MISMATCH: x"}
        self.assertEqual(scout.scout_one("x")["lean"], "")

    def test_a_faithful_signature_still_passes(self):
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        scout.review.review_statement = lambda c, code: {"faithful": True, "reason": "ok"}
        self.assertEqual(scout.scout_one("x")["status"], scout.OPEN)

    def test_an_unavailable_reviewer_does_not_block_the_claim(self):
        # Review failing is an infrastructure fact, not a mathematical one.
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        def boom(*a, **k):
            raise RuntimeError("reviewer down")
        scout.review.review_statement = boom
        self.assertEqual(scout.scout_one("x")["status"], scout.OPEN)

    def test_the_report_leads_with_free_wins(self):
        out = scout.report([{"claim": "hard", "status": scout.OPEN, "lean": "t", "note": ""},
                            {"claim": "easy", "status": scout.SOLVED, "tactic": "simp"}])
        self.assertLess(out.index("FREE WIN"), out.index("NON-TRIVIAL"))

    def test_disabling_the_flag_skips_the_work(self):
        config.GT_SCOUT = False
        try:
            self.assertEqual(scout.scout(["a", "b"]), [])
        finally:
            config.GT_SCOUT = True


class TestForecastScoring(unittest.TestCase):
    def test_allocation_is_proportional_and_keeps_a_floor(self):
        # Greedy argmax would starve the line that produced the single most
        # valuable proof of the last run -- it had already failed twice.
        a = forecast.allocate([{"claim": "a", "p": 0.9}, {"claim": "b", "p": 0.01}], 1000)
        self.assertGreater(a["a"], a["b"])
        self.assertGreater(a["b"], 0)

    def test_scoring_needs_adjudicated_forecasts(self):
        self.assertIn("note", forecast.score(rows=[]))

    def test_brier_and_auc_are_computed_against_lean_outcomes(self):
        rows = []
        for i, (p, ok) in enumerate([(0.9, True), (0.8, True), (0.2, False), (0.1, False)]):
            rows.append({"kind": "forecast", "statement": f"c{i}", "p_success": p})
            rows.append({"kind": "proof", "statement": f"c{i}", "ok": ok})
        s = forecast.score(rows=rows)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["auc"], 1.0)              # perfect ranking
        self.assertLess(s["brier"], s["brier_base_rate"])

    def test_a_useless_forecaster_scores_no_better_than_the_base_rate(self):
        rows = []
        for i, ok in enumerate([True, False, True, False]):
            rows.append({"kind": "forecast", "statement": f"c{i}", "p_success": 0.5})
            rows.append({"kind": "proof", "statement": f"c{i}", "ok": ok})
        self.assertAlmostEqual(forecast.score(rows=rows)["skill"], 0.0, places=6)

    def test_auc_is_reported_separately_from_calibration(self):
        # Ranking and calibration fail independently, and allocation only needs
        # ranking -- so a badly calibrated but well-ordering model is usable.
        rows = []
        for i, (p, ok) in enumerate([(0.99, True), (0.98, True), (0.97, False), (0.96, False)]):
            rows.append({"kind": "forecast", "statement": f"c{i}", "p_success": p})
            rows.append({"kind": "proof", "statement": f"c{i}", "ok": ok})
        s = forecast.score(rows=rows)
        self.assertEqual(s["auc"], 1.0)              # ordering is perfect
        self.assertGreater(s["brier"], s["brier_base_rate"])   # calibration is not


class TestLibraryCarryingCapacity(unittest.TestCase):
    def test_a_cited_lemma_is_never_retired(self):
        from src import library
        recs = {"used": {"name": "used", "statement": "x" * 400, "uses": 1},
                "fresh": {"name": "fresh", "statement": "y", "uses": 0}}
        order = sorted(recs.values(), key=lambda r: -library.fitness(r))
        self.assertEqual(order[0]["name"], "used")

    def test_brevity_breaks_ties_among_uncited_lemmas(self):
        from src import library
        short = {"name": "s", "statement": "a b c", "uses": 0}
        long = {"name": "l", "statement": "a b c " * 40, "uses": 0}
        self.assertGreater(library.fitness(short), library.fitness(long))

    def test_retire_is_a_retrieval_decision_not_a_deletion(self):
        import inspect
        from src import library
        s = inspect.getsource(library.retire)
        self.assertIn("retired", s)
        self.assertNotIn("del ", s)


class TestCoordinator(unittest.TestCase):
    def test_the_layer_reaches_the_anchor(self):
        import inspect
        from src import agent
        self.assertIn("gt.brief", inspect.getsource(agent._anchor_message))

    def test_expensive_mechanisms_are_cached_per_obligation_set(self):
        # The anchor is rebuilt EVERY turn; scouting per turn would cost more
        # than the proving does.
        calls = []
        gt.reset()
        self._s, gt.scout.scout = gt.scout.scout, lambda cs, **k: (calls.append(1) or [])
        self._f, gt.forecast.forecast = gt.forecast.forecast, lambda cs, sc=None: []
        self._o, gt._open_claims = gt._open_claims, lambda g: ["a", "b"]
        try:
            gt.prepare("goal"); gt.prepare("goal"); gt.prepare("goal")
            self.assertEqual(len(calls), 1)
        finally:
            gt.scout.scout, gt.forecast.forecast, gt._open_claims = self._s, self._f, self._o
            gt.reset()

    def test_the_cache_invalidates_when_obligations_change(self):
        calls = []
        gt.reset()
        claims = [["a"]]
        self._s, gt.scout.scout = gt.scout.scout, lambda cs, **k: (calls.append(1) or [])
        self._f, gt.forecast.forecast = gt.forecast.forecast, lambda cs, sc=None: []
        self._o, gt._open_claims = gt._open_claims, lambda g: claims[0]
        try:
            gt.prepare("goal")
            claims[0] = ["a", "b"]
            gt.prepare("goal")
            self.assertEqual(len(calls), 2)
        finally:
            gt.scout.scout, gt.forecast.forecast, gt._open_claims = self._s, self._f, self._o
            gt.reset()

    def test_no_ledger_costs_nothing(self):
        gt.reset()
        self._o, gt._open_claims = gt._open_claims, lambda g: []
        try:
            self.assertEqual(gt.prepare("goal"), {})
        finally:
            gt._open_claims = self._o


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFalsify(unittest.TestCase):
    """Deterministic critical thinking: a computed counterexample kills a claim
    before derivation is paid for. Only FALSIFIED ever changes behaviour."""

    def setUp(self):
        from src import falsify
        self._ask = falsify._ask
        self._run = falsify.sandbox.run_python

    def tearDown(self):
        from src import falsify
        falsify._ask = self._ask
        falsify.sandbox.run_python = self._run

    def test_a_found_counterexample_falsifies(self):
        from src import falsify
        falsify._ask = lambda c, timeout=180: "print('COUNTEREXAMPLE: m=2,n=2: 4 != 2')"
        out = falsify.probe("the product is invariant under the move")
        self.assertEqual(out["status"], falsify.FALSIFIED)
        self.assertIn("m=2,n=2", out["counterexample"])

    def test_no_counterexample_is_only_unfalsified_never_true(self):
        # Small-case search is not induction; surviving it blesses nothing.
        from src import falsify
        falsify._ask = lambda c, timeout=180: "print('NONE')"
        self.assertEqual(falsify.probe("gcd(a,b) divides a")["status"],
                         falsify.UNFALSIFIED)

    def test_a_broken_script_never_falsifies(self):
        from src import falsify
        falsify._ask = lambda c, timeout=180: "raise RuntimeError('bad script')"
        self.assertNotEqual(falsify.probe("x")["status"], falsify.FALSIFIED)

    def test_a_dead_model_is_unavailable(self):
        from src import falsify
        falsify._ask = lambda c, timeout=180: ""
        self.assertEqual(falsify.probe("x")["status"], falsify.UNAVAILABLE)

    def test_the_real_sandbox_end_to_end_on_the_measured_p1_falsehood(self):
        # No model: a handwritten search for the actual P1 claim, through the
        # REAL sandbox. m=n=2: product 4 -> gcd*(lcm/gcd) = 2*1 = 2.
        from src import falsify
        falsify._ask = lambda c, timeout=180: (
            "for m in range(2, 12):\n"
            "  for n in range(2, 12):\n"
            "    g = gcd(m, n)\n"
            "    if m * n != g * (lcm(m, n) // g):\n"
            "      print(f'COUNTEREXAMPLE: m={m},n={n}: {m*n} != {g*(lcm(m,n)//g)}')\n"
            "      raise SystemExit\n"
            "print('NONE')\n")
        out = falsify.probe("the product of the two numbers is invariant "
                            "under replacing them with gcd and lcm/gcd")
        self.assertEqual(out["status"], falsify.FALSIFIED)
        self.assertIn("m=2,n=2", out["counterexample"])

    def test_a_true_claim_survives_the_real_sandbox(self):
        from src import falsify
        falsify._ask = lambda c, timeout=180: (
            "for a in range(1, 30):\n"
            "  for b in range(1, 30):\n"
            "    if a % gcd(a, b) != 0:\n"
            "      print(f'COUNTEREXAMPLE: a={a},b={b}')\n"
            "      raise SystemExit\n"
            "print('NONE')\n")
        self.assertEqual(falsify.probe("gcd(a,b) divides a")["status"],
                         falsify.UNFALSIFIED)

    def test_scout_kills_a_falsified_claim_before_derivation(self):
        # The saving IS the point: no derivation, no review, no probe.
        from src import falsify
        calls = []
        scout.prover.derive_signature = lambda c, tries=2: (calls.append(1) or "sig")
        falsify.probe_orig = falsify.probe
        try:
            scout.falsify.probe = lambda c, timeout=45: {
                "status": falsify.FALSIFIED, "counterexample": "m=2: 4 != 2",
                "claim": c, "code": ""}
            out = scout.scout_one("a false claim")
            self.assertEqual(out["status"], scout.FALSE)
            self.assertEqual(out["lean"], "")
            self.assertEqual(calls, [])
        finally:
            scout.falsify.probe = falsify.probe_orig

    def test_scout_proceeds_normally_when_unfalsified(self):
        from src import falsify
        scout.prover.derive_signature = lambda c, tries=2: "theorem t : True := by"
        scout.review.review_statement = lambda c, code: {"faithful": True}
        scout.tactics.probe = lambda sig, budget=0: {"ok": False, "tried": []}
        orig = scout.falsify.probe
        try:
            scout.falsify.probe = lambda c, timeout=45: {
                "status": falsify.UNFALSIFIED, "counterexample": "", "claim": c,
                "code": ""}
            self.assertEqual(scout.scout_one("x")["status"], scout.OPEN)
        finally:
            scout.falsify.probe = orig

    def test_the_report_carries_the_counterexample_values(self):
        # The values are the audit trail: a wrong translation is visible.
        out = scout.report([{"claim": "product invariant", "status": scout.FALSE,
                             "note": "numerically falsified -- counterexample: "
                                     "m=2,n=2: 4 != 2", "lean": ""}])
        self.assertIn("m=2,n=2", out)
        self.assertIn("Do not attempt a proof", out)


class TestProveTheoremRefusesFalseClaims(unittest.TestCase):
    """scout falsifies at most GT_SCOUT_MAX plan obligations, but prove_theorem
    is a TOOL the agent calls on anything -- so the gate has to sit here too."""

    def setUp(self):
        from src import falsify, prover
        self._probe = falsify.probe
        self._raw = prover._prove_theorem_untraced
        falsify._SEEN.clear()

    def tearDown(self):
        from src import falsify, prover
        falsify.probe = self._probe
        prover._prove_theorem_untraced = self._raw
        falsify._SEEN.clear()

    def test_a_falsified_claim_never_reaches_the_prover(self):
        # prover imports falsify lazily inside the function, so patching the
        # falsify MODULE is what the call site actually resolves.
        from src import falsify, prover
        falsify.probe = lambda c, timeout=45: {
            "status": "falsified", "counterexample": "m=2,n=2: 4 != 2"}
        out = prover._prove_theorem_untraced("the product is invariant")
        self.assertTrue(out.get("refuted"))
        self.assertFalse(out.get("proved"))
        self.assertIn("m=2,n=2", out["note"])

    def test_the_counterexample_is_handed_back_for_rechecking(self):
        from src import falsify, prover
        falsify.probe = lambda c, timeout=45: {
            "status": "falsified", "counterexample": "n=3 gives 7 != 8"}
        out = prover._prove_theorem_untraced("a false claim")
        self.assertIn("Check those values", out["note"])

    def test_a_pinned_signature_skips_the_gate(self):
        from src import falsify
        # A caller who wrote the Lean themselves has already committed to a
        # formalization; re-litigating the prose would spend a model call to
        # second-guess a statement the compiler will judge anyway.
        from src import prover
        hits = []
        falsify.probe = lambda c, timeout=45: (hits.append(1) or
                                                      {"status": "falsified",
                                                       "counterexample": "x"})
        try:
            prover._prove_theorem_untraced("x", "theorem t : True := by")
        except Exception:
            pass
        self.assertEqual(hits, [])

    def test_an_unfalsified_claim_proceeds(self):
        from src import falsify, prover
        falsify.probe = lambda c, timeout=45: {"status": "unfalsified",
                                                      "counterexample": ""}
        out = prover._prove_theorem_untraced("gcd(a,b) divides a", time_budget=1)
        self.assertFalse(out.get("refuted"))

    def test_the_verdict_is_memoised_across_scout_and_prove(self):
        # One slot, one model: paying twice for a deterministic verdict is waste.
        from src import falsify
        calls = []
        falsify._ask = lambda c, timeout=180: (calls.append(1) or "print('NONE')")
        falsify.probe("a claim")
        falsify.probe("a claim")
        falsify.probe("  A   CLAIM  ")          # normalised key
        self.assertEqual(len(calls), 1)


class TestBidirectional(unittest.TestCase):
    """Top-down prunes by REFUTATION: one false element of a sufficient set
    kills the route, which forward search cannot signal at all."""

    def test_a_false_required_fact_kills_the_route(self):
        from src import bidir
        f, l, ss = bidir.falsify, bidir.library, bidir.sufficient_set
        try:
            bidir.falsify = type(f)("f", (), {})  # placeholder, replaced below
        except Exception:
            pass
        import types
        bidir.falsify = types.SimpleNamespace(
            FALSIFIED="falsified",
            probe=lambda c, timeout=45: {
                "status": "falsified" if "invariant" in c else "unfalsified",
                "counterexample": "m=2,n=2: 4 != 2"})
        bidir.library = types.SimpleNamespace(search=lambda q, k=1: [])
        bidir.sufficient_set = lambda g, timeout=180: ["gcd divides",
                                                       "product is invariant"]
        try:
            out = bidir.top_down("a goal")
            self.assertEqual(out["verdict"], bidir.DEAD)
            self.assertIn("m=2,n=2", bidir.report(out))
        finally:
            bidir.falsify, bidir.library, bidir.sufficient_set = f, l, ss

    def test_a_refutable_goal_short_circuits_before_decomposing(self):
        import types
        from src import bidir
        f, ss = bidir.falsify, bidir.sufficient_set
        called = []
        bidir.falsify = types.SimpleNamespace(
            FALSIFIED="falsified",
            probe=lambda c, timeout=45: {"status": "falsified",
                                         "counterexample": "n=0"})
        bidir.sufficient_set = lambda g, timeout=180: (called.append(1) or [])
        try:
            self.assertEqual(bidir.top_down("bad goal")["why"],
                             "the goal itself is false")
            self.assertEqual(called, [])      # no decomposition paid for
        finally:
            bidir.falsify, bidir.sufficient_set = f, ss

    def test_library_hits_are_marked_free(self):
        import types
        from src import bidir
        f, l, ss = bidir.falsify, bidir.library, bidir.sufficient_set
        bidir.falsify = types.SimpleNamespace(
            FALSIFIED="falsified",
            probe=lambda c, timeout=45: {"status": "unfalsified",
                                         "counterexample": ""})
        bidir.library = types.SimpleNamespace(
            search=lambda q, k=1: [{"name": "gcd_dvd_left"}])
        bidir.sufficient_set = lambda g, timeout=180: ["gcd divides a"]
        try:
            out = bidir.top_down("a goal")
            self.assertEqual(out["verdict"], bidir.LIVE)
            self.assertIn("ALREADY PROVED as gcd_dvd_left", bidir.report(out))
        finally:
            bidir.falsify, bidir.library, bidir.sufficient_set = f, l, ss

    def test_the_backward_leg_reaches_the_anchor(self):
        import inspect
        from src import gt
        self.assertIn("bidir.top_down", inspect.getsource(gt.prepare))
        self.assertIn("bidir.report", inspect.getsource(gt.brief))
