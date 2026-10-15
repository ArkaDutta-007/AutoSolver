#!/usr/bin/env python3
"""Fast unit tests. No model calls, no Lean, no network.

Every test here is a REGRESSION test for a bug that actually shipped. The
comment on each says which one, because a test whose motivation is forgotten is
a test that gets deleted the first time it is inconvenient.

    python3 -m tests.test_units          # or: python3 tests/test_units.py
"""

import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import agent, config, inputs, journal, mathlib_kb, prover, tools  # noqa: E402


class TestGenerationBudget(unittest.TestCase):
    """The 900s fixed timeout silently capped every slow model (LIMITATIONS #0 sibling)."""

    def test_timeout_scales_with_budget(self):
        self.assertGreater(prover._gen_timeout(24000), prover._gen_timeout(8000))

    def test_uncapped_budget_uses_context_window(self):
        # budget <= 0 must NOT collapse to the 300s floor; it means "the whole
        # window", which at the floor throughput is thousands of seconds.
        self.assertGreater(prover._gen_timeout(-1), 1000)

    def test_timeout_respects_cap(self):
        self.assertLessEqual(prover._gen_timeout(10 ** 9), config.ORCH_TIMEOUT_CAP)

    def test_timeout_clamped_by_problem_deadline(self):
        prover.set_problem_deadline(time.time() + 300)
        try:
            self.assertLessEqual(prover._gen_timeout(-1), 300)
        finally:
            prover.set_problem_deadline(0)

    def test_timeout_never_zero_or_negative(self):
        prover.set_problem_deadline(time.time() - 100)      # already expired
        try:
            self.assertGreaterEqual(prover._gen_timeout(12000), 60)
        finally:
            prover.set_problem_deadline(0)


class TestThinkOption(unittest.TestCase):
    """think=high must reach the wire; "off" must become False, not the string."""

    def test_levels_pass_through(self):
        for level in ("low", "medium", "high"):
            config.ORCH_THINK = level
            self.assertEqual(prover._think_opt(), {"think": level})

    def test_off_becomes_false_not_string(self):
        for word in ("off", "false", "none", "0"):
            config.ORCH_THINK = word
            self.assertIs(prover._think_opt()["think"], False)

    def test_empty_omits_key_entirely(self):
        config.ORCH_THINK = ""
        self.assertEqual(prover._think_opt(), {})

    def tearDown(self):
        config.ORCH_THINK = "high"


class TestSignatureLint(unittest.TestCase):
    """A false PASS shipped because `Nat.gcd m * Nat.lcm m n = m * n` TYPECHECKS:
    Lean elaborates it pointwise via Pi.instMul. The compiler cannot catch this
    -- it is well-typed -- so the lint is the only defence."""

    def test_partial_application_rejected(self):
        bad = "theorem r (m n : ℕ) : Nat.gcd m * Nat.lcm m n = m * n"
        self.assertTrue(prover.arity_lint(bad))

    def test_correct_application_accepted(self):
        good = "theorem r (m n : ℕ) : Nat.gcd m n * Nat.lcm m n = m * n"
        self.assertEqual(prover.arity_lint(good), "")

    def test_min_max_binary_forms_ok(self):
        good = "theorem r (a b : ℕ) : min a b + (max a b - min a b) = max a b"
        self.assertEqual(prover.arity_lint(good), "")


class TestSignatureNormalisation(unittest.TestCase):
    """rstrip(':=') is a CHARACTER-CLASS strip: it ate the trailing colon that
    derive_signature's own prompt asks for."""

    def test_trailing_colon_variant_offered(self):
        cands = prover.normalize_signature("theorem r (n : ℕ) : 2 ∣ n * (n + 1) :")
        self.assertTrue(any(c.rstrip().endswith("1)") for c in cands))

    def test_proof_body_stripped(self):
        cands = prover.normalize_signature(
            "theorem r (n : ℕ) : 2 ∣ n * (n + 1) := by sorry")
        self.assertTrue(any(":=" not in c for c in cands))

    def test_code_fence_stripped(self):
        cands = prover.normalize_signature("```lean\ntheorem r : True\n```")
        self.assertTrue(all("```" not in c for c in cands))


class TestHeadline(unittest.TestCase):
    """35 rounds died on `failed to synthesize GCDMonoid` while the trace showed
    only `unknown tactic` -- the cascade line, not the cause."""

    def test_prefers_real_error_over_cascade(self):
        errs = ("tmp.lean:4:1: error: unknown tactic\n"
                "tmp.lean:4:9: error: failed to synthesize GCDMonoid ℕ")
        self.assertIn("synthesize", prover._headline(errs))

    def test_strips_temp_filename(self):
        self.assertNotIn("tmpabc.lean", prover._headline("tmpabc.lean:1:1: error: boom"))


class TestNativeToolCalls(unittest.TestCase):
    """Ollama's native API returns `arguments` as an OBJECT; the dispatcher
    json.loads() it. Getting this backwards breaks every tool call."""

    def test_arguments_serialised_to_string(self):
        native = [{"function": {"name": "run_python", "arguments": {"code": "print(1)"}}}]
        out = agent._from_native(native)
        self.assertIsInstance(out[0]["function"]["arguments"], str)
        self.assertEqual(json.loads(out[0]["function"]["arguments"]), {"code": "print(1)"})

    def test_id_synthesised_when_absent(self):
        out = agent._from_native([{"function": {"name": "x", "arguments": {}}}])
        self.assertTrue(out[0]["id"])

    def test_string_arguments_passed_through(self):
        native = [{"function": {"name": "x", "arguments": '{"a":1}'}}]
        self.assertEqual(agent._from_native(native)[0]["function"]["arguments"], '{"a":1}')

    def test_payload_converts_history_back_to_objects(self):
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "function": {"name": "x",
                                                          "arguments": '{"a": 1}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
        body = agent._native_payload(msgs, 100, None, False)
        sent = body["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(sent, dict)             # native wants an object
        self.assertNotIn("tool_call_id", body["messages"][1])

    def test_payload_does_not_mutate_caller_messages(self):
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c", "function": {"name": "x", "arguments": '{}'}}]}]
        agent._native_payload(msgs, 100, None, False)
        self.assertIsInstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_think_reaches_the_wire(self):
        config.ORCH_THINK = "high"
        self.assertEqual(agent._native_payload([], 100, None, False).get("think"), "high")

    def test_uncapped_num_predict_survives(self):
        body = agent._native_payload([], -1, None, False)
        self.assertEqual(body["options"]["num_predict"], -1)


class TestKnowledgeBaseExtraction(unittest.TestCase):
    """The first build recorded BARE source names: `gcd_mul_lcm` instead of
    `Nat.gcd_mul_lcm`. Retrieval looked broken (1/8) while actually returning
    the right lemma at rank 1 -- and would have fed the prover an identifier
    that does not resolve."""

    @classmethod
    def setUpClass(cls):
        cls.decls = mathlib_kb.extract(limit=0) if mathlib_kb.sources() else []
        cls.names = {d["name"] for d in cls.decls}

    def setUp(self):
        if not self.decls:
            self.skipTest("no Lean sources on this machine")

    def test_namespace_qualified(self):
        self.assertIn("Nat.gcd_mul_lcm", self.names)
        self.assertIn("Matrix.det_mul", self.names)

    def test_core_and_batteries_indexed_not_just_mathlib(self):
        # Nat.gcd_mul_lcm is in Lean CORE, not Mathlib. Indexing Mathlib alone
        # omits the most fundamental lemmas.
        prefixes = {d["module"].split(".")[0] for d in self.decls}
        self.assertIn("Init", prefixes)
        self.assertIn("Mathlib", prefixes)

    def test_no_double_prefixing(self):
        self.assertFalse([n for n in self.names if ".." in n or n.startswith(".")])

    def test_names_are_unique(self):
        self.assertEqual(len(self.names), len(self.decls))


class TestJournalDedup(unittest.TestCase):
    """`established` accumulated FIVE copies of one divisibility fact across
    sessions, and the repetition read as progress."""

    @staticmethod
    def _overlap(x: str, y: str) -> float:
        # The SAME Jaccard `known()` thresholds at 0.8. Asserting exact key
        # equality (an earlier version of this test) tested a contract the code
        # does not have and never intended to.
        a, b = journal._content_key(x), journal._content_key(y)
        return len(a & b) / max(len(a | b), 1)

    def test_restatement_is_recognised(self):
        self.assertGreaterEqual(
            self._overlap("For all naturals n, 2 divides n*(n+1)",
                          "2 divides n*(n+1) for all naturals n"), 0.8)

    def test_different_claims_are_not_confused(self):
        self.assertLess(
            self._overlap("gcd times lcm equals the product",
                          "every integer above one has a prime factor"), 0.8)

    def test_known_matches_a_restatement_in_a_live_journal(self):
        # `_cur["state"]` can exist and be None between runs, so setdefault is
        # not enough to get a dict back.
        prior_state = journal._cur.get("state")
        journal._cur["state"] = {"established": ["For all naturals n, 2 divides n*(n+1)"]}
        try:
            self.assertTrue(journal.known("2 divides n*(n+1) for all naturals n"))
            self.assertFalse(journal.known("the square root of two is irrational"))
        finally:
            journal._cur["state"] = prior_state


class TestProblemSplitting(unittest.TestCase):
    def test_explicit_markers_win(self):
        text = "Problem 1.\nFirst thing.\n\nProblem 2.\nSecond thing."
        self.assertEqual(len(inputs.split_problems(text, "t.txt")), 2)

    def test_bare_numbers_only_without_explicit_markers(self):
        # "3.14" and enumerated prose must not each become a problem.
        text = "Show that pi is about 3. 14 is not relevant here."
        self.assertLessEqual(len(inputs.split_problems(text, "t.txt")), 2)

    def test_single_problem_stays_single(self):
        self.assertEqual(len(inputs.split_problems("Prove that 2+2=4.", "t.txt")), 1)


class TestToolAuthorization(unittest.TestCase):
    """fetch_url executed a real HTTP GET during a search=False run. REGISTRY
    membership is not authorization."""

    def test_suppressed_tool_refused(self):
        names = [t["function"]["name"] for t in tools.specs()]
        if "fetch_url" not in names:
            self.skipTest("fetch_url not registered")
        out = tools.dispatch("fetch_url", {"url": "http://example.com"},
                             suppress={"fetch_url"})
        self.assertIn("error", out)

    def test_unknown_tool_refused(self):
        self.assertIn("error", tools.dispatch("no_such_tool", {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProbeBudget(unittest.TestCase):
    """The probe used a fixed 240s regardless of time remaining. Measured on an
    AM-GM claim with a 420s budget: probe 330s, ladder 90s -- one attempt on a
    goal needing several."""

    @staticmethod
    def _share(left: float) -> float:
        return max(30.0, min(240.0, left * 0.4))

    def test_probe_never_takes_most_of_the_budget(self):
        for left in (100, 300, 420, 600):
            self.assertLess(self._share(left), left * 0.5, f"left={left}")

    def test_probe_is_capped_on_long_budgets(self):
        self.assertEqual(self._share(10_000), 240.0)

    def test_probe_gets_a_floor_on_short_budgets(self):
        # Below the floor the probe is useless, and a useless probe that still
        # pays `import Mathlib` is worse than none.
        self.assertEqual(self._share(10), 30.0)

    def test_call_site_passes_a_budget(self):
        import inspect
        s = inspect.getsource(prover.prove_theorem)
        self.assertIn("tactics.probe(lean_statement, budget=", s)


class TestContextBudgetSumsToLessThanOne(unittest.TestCase):
    """35% of steps in the IMO run produced NOTHING: `finish_reason=length`
    with `reasoning_only=True`. The prompt was measured at ~28,700 of 32,768
    tokens because each budget fraction was chosen in isolation and never
    checked against the others."""

    def test_prompt_and_generation_fit_the_window(self):
        # Compaction is measured over the WHOLE prompt (anchor and tool results
        # included) right before each call, so the worst case sent is the
        # threshold plus one nudge, and generation gets the rest.
        nudge = 2_000
        total = config.COMPACT_AT * config.CONTEXT_TOKENS + nudge + config.ORCH_MAX_TOKENS
        self.assertLess(total, config.CONTEXT_TOKENS,
                        f"prompt+generation = {total:,.0f} > window "
                        f"{config.CONTEXT_TOKENS:,}")

    def test_generation_gets_a_real_allowance(self):
        # Reserving room is pointless if the reserve cannot hold a thought.
        self.assertGreaterEqual(config.ORCH_MAX_TOKENS, 8_000)

    def test_compaction_counts_real_tokens_with_a_floor(self):
        from src import agent
        msgs = [{"role": "user", "content": "x" * 30_000}]
        agent._FILL.update(chars=0, tokens=0)
        self.assertAlmostEqual(agent.prompt_tokens(msgs), 30_000 / 3.0, delta=50)
        # the server counted more than the floor: trust it, plus what grew since
        agent._note_fill(msgs, {"prompt_eval_count": 20_000})
        msgs.append({"role": "tool", "content": "y" * 3_000})
        self.assertGreater(agent.prompt_tokens(msgs), 20_900)
        # a cache-shrunk count never drags the estimate under the floor
        agent._note_fill(msgs, {"prompt_eval_count": 50})
        self.assertGreaterEqual(agent.prompt_tokens(msgs), 33_000 / 3.0 - 50)
        agent._FILL.update(chars=0, tokens=0)

    def test_generation_is_capped_not_unbounded(self):
        # -1 meant "until the window closes", which is what caused the loss.
        self.assertGreater(config.ORCH_MAX_TOKENS, 0)


class TestCloudPlanner(unittest.TestCase):
    """The cloud planner plans once, first; the local model does the work."""

    def _run(self, reply, status=200):
        import tempfile
        from unittest import mock
        from src import plan

        class R:
            status_code, text = status, "boom"
            def json(self):
                return {"choices": [{"message": {"content": reply}}]}
        with mock.patch.object(config, "PLANNER_MODEL", "qwen3.8-max"), \
                mock.patch.object(config, "PLANNER_KEY", "k"), \
                mock.patch.object(plan, "PATH", os.path.join(tempfile.mkdtemp(), "p.json")), \
                mock.patch("requests.post", return_value=R()) as post:
            return plan, plan.build("Prove n^3 - n is divisible by 6.", specs=tools.specs()), post

    def test_plan_seeds_the_ledger_and_the_prompt(self):
        reply = "```json\n" + json.dumps({
            "formulation": {"restated": "6 | n^3-n", "kind": "proof", "given": ["n>0"]},
            "strategy": "factor", "pitfalls": ["Nat subtraction"],
            "steps": [{"do": "check small n", "tool": "run_python", "done_when": "no counterexample"},
                      {"do": "fly", "tool": "teleport"}],
            "obligations": [{"id": "f", "kind": "lemma", "claim": "n^3-n = (n-1)n(n+1) for all n"}]}) + "\n```"
        plan, led, post = self._run(reply)
        self.assertEqual(post.call_count, 1, "one cloud call, and no local decomposition")
        self.assertIn("/chat/completions", post.call_args[0][0])
        self.assertEqual([o["id"] for o in led["obligations"]], ["f", "closure"])
        bp = led["blueprint"]
        self.assertEqual(bp["steps"][1]["tool"], "", "a tool the local model lacks is dropped")
        text = plan.blueprint_text(bp)
        self.assertIn("-> run_python", text)
        self.assertNotIn("obligations", bp)
        self.assertIn("Do NOT solve it", post.call_args[1]["json"]["messages"][0]["content"])

    def test_failure_falls_back_to_local(self):
        from unittest import mock
        from src import plan
        with mock.patch.object(config, "ORCH_THINK", "off"):
            _, led, post = self._run("", status=500)
        self.assertNotIn("blueprint", led)
        self.assertGreater(post.call_count, 1, "the local decomposition ran after the planner failed")

    def test_off_or_keyless_makes_no_cloud_call(self):
        from unittest import mock
        from src import plan
        for model, key in (("", "k"), ("qwen3.8-max", "")):
            with mock.patch.object(config, "PLANNER_MODEL", model), \
                    mock.patch.object(config, "PLANNER_KEY", key), \
                    mock.patch("requests.post") as post:
                self.assertEqual(plan.blueprint("x", tools.specs()), {})
                post.assert_not_called()


class TestTruncationRetryLowersEffort(unittest.TestCase):
    """`Think briefly this time` is a PROMPT; effort is an API parameter. The
    retry truncated identically until the override was plumbed through."""

    def test_payload_accepts_an_explicit_override(self):
        body = agent._native_payload([], 100, None, False, think="low")
        self.assertEqual(body["think"], "low")

    def test_override_beats_configured_effort(self):
        config.ORCH_THINK = "high"
        self.assertEqual(agent._native_payload([], 100, None, False, "low")["think"], "low")

    def test_no_override_uses_configured_effort(self):
        config.ORCH_THINK = "high"
        self.assertEqual(agent._native_payload([], 100, None, False)["think"], "high")

    def test_loop_degrades_effort_after_a_lost_turn(self):
        import inspect
        s = inspect.getsource(agent.run)
        self.assertIn('forced_think = "low"', s)
        self.assertIn("think=forced_think", s)

    def test_degradation_is_one_shot(self):
        # A permanently degraded orchestrator would trade 35% waste for a
        # permanently shallower agent.
        import inspect
        s = inspect.getsource(agent.run)
        self.assertIn('forced_think = ""', s)

    def tearDown(self):
        config.ORCH_THINK = "high"


class TestProvedClaimsRetractRuleOuts(unittest.TestCase):
    """A live run ruled the p-adic valuation lemma out FOUR times, then proved
    it. The next session would inherit `established` and `exhausted` both
    containing it -- told to build on it and never to attempt it. That is the
    documented local-minimum mechanism: the real invariant sitting in
    `exhausted`, falsely eliminated."""

    def setUp(self):
        self._prior = journal._cur.get("state")
        journal._cur["state"] = {"established": [], "exhausted": [], "log": []}

    def tearDown(self):
        journal._cur["state"] = self._prior

    def test_rule_out_dedupes_by_content_not_string(self):
        journal.rule_out("gcd(min(a,b), |a-b|) = gcd(a,b)", "prover timed out")
        journal.rule_out("gcd(min(a,b), |a-b|) = gcd(a,b)", "no tactic worked")
        journal.rule_out("gcd(min(a,b), |a-b|) = gcd(a,b)", "32 sampled, none compiled")
        self.assertEqual(len(journal._cur["state"]["exhausted"]), 1)

    def test_proving_retracts_the_matching_rule_out(self):
        claim = "For all naturals a b, gcd(min(a,b), max(a,b) - min(a,b)) = gcd(a,b)"
        journal.rule_out(claim, "prover failed")
        self.assertEqual(len(journal._cur["state"]["exhausted"]), 1)
        journal.establish(claim, "lean")
        self.assertEqual(journal._cur["state"]["exhausted"], [])
        self.assertEqual(len(journal._cur["state"]["established"]), 1)

    def test_unrelated_rule_outs_survive(self):
        journal.rule_out("coordinate bashing the geometry", "too messy")
        journal.establish("gcd(min(a,b), max(a,b)-min(a,b)) = gcd(a,b)", "lean")
        self.assertEqual(len(journal._cur["state"]["exhausted"]), 1)

    def test_a_claim_is_never_both_proved_and_exhausted(self):
        claim = "the process terminates after finitely many moves"
        journal.rule_out(claim, "could not formalise")
        journal.establish(claim, "lean")
        est = {journal._content_key(c) for c in journal._cur["state"]["established"]}
        exh = {journal._content_key(x.split("—")[0])
               for x in journal._cur["state"]["exhausted"]}
        self.assertFalse(est & exh)


class TestInstanceRepair(unittest.TestCase):
    """56% of derivation failures are typeclass errors, and the commonest shape
    is a class USED but not DECLARED. Lean names the missing instance exactly,
    so the repair is deterministic and needs no model call."""

    def test_a_missing_instance_is_added(self):
        from src import prover
        out = prover._repair_missing_instances(
            "theorem t (α : Type*) (x y : α) : x ≤ y → y ≤ x → x = y",
            "failed to synthesize instance of type class\n  LE α\n")
        self.assertIn("[LE α]", out)

    def test_greek_type_variables_are_recognised(self):
        # `[A-Za-z_]` matches none of α, β, ι -- exactly the letters Lean uses
        # for type variables. With the ASCII-only class, `LE α` parsed to zero
        # arguments and every repair was silently refused.
        from src import prover
        self.assertTrue(prover._repair_missing_instances(
            "theorem z (β : Type*) (s t : Finset β) : s = t",
            "failed to synthesize instance of type class\n  DecidableEq β\n"))

    def test_the_instance_goes_after_the_variables_it_mentions(self):
        # `theorem t [LE α] (α : Type*)` references α before binding it and
        # cannot compile. The first version placed it right after the theorem
        # name, so every repair would have failed while a test that only asked
        # "was a repair attempted?" still passed.
        from src import prover
        out = prover._repair_missing_instances(
            "theorem t (α : Type*) (x y : α) : x ≤ y → x = y",
            "failed to synthesize instance of type class\n  LE α\n")
        self.assertLess(out.index("(α : Type*)"), out.index("[LE α]"))

    def test_a_conceptual_error_is_not_papered_over(self):
        # `SDiff (Type u_1)` means set difference was applied to a Type; no
        # binder fixes that, and adding one would hide a real mistake.
        from src import prover
        self.assertEqual("", prover._repair_missing_instances(
            "theorem v (X : ι → Type*) : True",
            "failed to synthesize instance of type class\n  SDiff (Type u_1)\n"))

    def test_an_unresolved_metavariable_is_refused(self):
        from src import prover
        self.assertEqual("", prover._repair_missing_instances(
            "theorem w (a : ℝ) : True",
            "failed to synthesize instance of type class\n  HPow ℝ≥0 ℝ≥0 ?m.7\n"))

    def test_an_argument_that_is_not_already_bound_is_refused(self):
        # Adding `[Foo γ]` when γ appears nowhere would invent a variable.
        from src import prover
        self.assertEqual("", prover._repair_missing_instances(
            "theorem q (n : ℕ) : n = n",
            "failed to synthesize instance of type class\n  Bar γ\n"))

    def test_repair_is_wired_into_the_derivation_path(self):
        import inspect
        from src import prover
        self.assertIn("_repair_missing_instances",
                      inspect.getsource(prover._derive_once_untraced))


class TestInstanceRepairIsSingleShot(unittest.TestCase):
    def test_the_repair_does_not_iterate(self):
        """Following the missing-instance chain was measured on eight real
        rejected candidates and fixed ZERO of them, at 3x the compile cost."""
        import inspect
        from src import prover
        src = inspect.getsource(prover._derive_once_untraced)
        body = src[src.index("_repair_missing_instances"):]
        self.assertNotIn("for _ in range(3)", body)


class TestHardeningScriptStillPasses(unittest.TestCase):
    """`tests/hardening.py` is a standalone script whose checks run at import
    and end in sys.exit, so pytest cannot collect it directly -- renaming it to
    test_*.py aborts collection with SystemExit. It holds real regression tests
    for bugs from runs/run-20260813-171447.jsonl, so rather than restructure
    152 lines, the suite just runs it."""

    def test_hardening_suite_exits_clean(self):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run([sys.executable, os.path.join(here, "hardening.py")],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-400:])


class TestSharedLeanHelpers(unittest.TestCase):
    """One implementation of two things that were written out four times."""

    def test_split_signature_uses_the_first_top_level_colon(self):
        from src.lean import split_signature
        self.assertEqual(split_signature("(a : ℕ) (h : a > 0) : a ≥ 1"),
                         ("(a : ℕ) (h : a > 0)", "a ≥ 1"))

    def test_split_signature_ignores_a_colon_in_the_conclusion(self):
        # Splitting on the LAST colon breaks exactly here.
        from src.lean import split_signature
        self.assertEqual(split_signature(": ∀ x : ℝ, x ^ 2 ≥ 0"),
                         ("", "∀ x : ℝ, x ^ 2 ≥ 0"))

    def test_split_signature_ignores_a_colon_inside_a_binder(self):
        # Splitting on the FIRST colon outright cuts inside `(α : Type*)` and
        # leaves `theorem t (α`, which silently refused every instance repair.
        from src.lean import split_signature
        b, _ = split_signature("(f : A →ₗ[R] B) (a : A) : f a = f a")
        self.assertEqual(b, "(f : A →ₗ[R] B) (a : A)")

    def test_statement_ok_accepts_a_sorry_only_complaint(self):
        from src.lean import statement_ok
        self.assertTrue(statement_ok(
            {"ok": False, "errors": "x.lean:5:1: error: declaration uses sorry"}))

    def test_statement_ok_rejects_a_real_error_alongside_sorry(self):
        # One of the four hand-written copies omitted this clause entirely and
        # accepted statements carrying genuine errors.
        from src.lean import statement_ok
        self.assertFalse(statement_ok(
            {"ok": False,
             "errors": "error: unknown identifier foo\nerror: uses sorry"}))

    def test_the_four_call_sites_share_one_implementation(self):
        import inspect
        from src import prover
        import evals.formalize_score as S
        import evals.formalize_bench as B
        for mod in (prover, S, B):
            self.assertIn("statement_ok", inspect.getsource(mod))
        # And the duplicated bracket walk is gone from the score harness.
        self.assertNotIn("def split_signature", inspect.getsource(S))
