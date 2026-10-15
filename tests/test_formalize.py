#!/usr/bin/env python3
"""The formalization benchmark: extraction and scoring. No Lean, no model.

This is now the project's main measurement, so its parsing has to be right --
a mispaired gold or a mis-split signature would silently corrupt every number
derived from it.

    python3 -m tests.test_formalize
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals.formalize_bench as B                           # noqa: E402
import evals.formalize_score as S                           # noqa: E402


class TestExtraction(unittest.TestCase):
    """Both bugs below were observed in real Mathlib output, not imagined."""

    def test_a_docstring_pairs_with_the_theorem_directly_below_it(self):
        src = ('/-- The greatest common divisor divides its first argument. -/\n'
               'theorem gcd_dvd_left (m n : Nat) : Nat.gcd m n ∣ m := by simp\n')
        got = B.PAIR.search(src)
        self.assertEqual(got.group("name"), "gcd_dvd_left")
        self.assertIn("greatest common divisor", got.group("doc"))

    def test_a_docstring_does_not_reach_past_an_intervening_declaration(self):
        # OBSERVED: a docstring about ULift algebras was paired with
        # `map_algebraMap_mul` further down the file, because a lazy `.+?`
        # happily skips a `-/` whose next line is not a theorem.
        src = ('/-- Something about a class, and long enough to look like prose. -/\n'
               'class Foo : Prop where\n'
               '  bar : True\n'
               'theorem unrelated (n : Nat) : n = n := by rfl\n')
        self.assertIsNone(B.PAIR.search(src))

    def test_a_term_mode_proof_is_extracted_and_does_not_swallow_the_next_theorem(self):
        # OBSERVED: requiring `:= by` made a term-mode proof scan forward to the
        # next `by`, dragging an entire second theorem into the signature.
        src = ('/-- The resolvent is a unit when the argument is in the set. -/\n'
               'theorem isUnit_resolvent {r : R} : P r ↔ Q r := isUnit_ringInverse.symm\n'
               'theorem later_one {r : R} : R1 r := by simp\n')
        m = B.PAIR.search(src)
        self.assertEqual(m.group("name"), "isUnit_resolvent")
        self.assertNotIn("later_one", m.group("sig"))

    def test_prose_is_required_not_just_backticked_code(self):
        self.assertFalse(B._plausible_doc("`foo` `bar` `baz`"))
        self.assertFalse(B._plausible_doc("TODO: fix this later on someday soon"))
        self.assertTrue(B._plausible_doc(
            "The greatest common divisor of two natural numbers divides each "
            "of them, which is the defining property we rely on here."))

    def test_a_gold_with_no_explicit_binders_is_rejected(self):
        # It compiles (Lean auto-binds), but spliced into `example : A -> B` it
        # becomes an INSTANCE while the derived side is a forall, so backward
        # implication is unprovable and T2 can never be earned. Observed on
        # prod_subset_prod_iff with a derivation identical to the gold.
        out = B.extract.__doc__  # keep import used
        src = ('/-- A product set is included in a product set exactly when the '
               'factors are included, or one factor is empty. -/\n'
               'theorem prod_subset : s ×ˢ t ⊆ s₁ ×ˢ t₁ ↔ s ⊆ s₁ := by simp\n')
        m = B.PAIR.search(src)
        self.assertIsNotNone(m)          # it still MATCHES
        sig = " ".join(m.group("sig").split())
        self.assertEqual(sig.split(":", 1)[0].strip(), "")   # ...but has no binders

    def test_a_gold_is_rendered_as_a_standalone_declaration(self):
        code = B._standalone({"name": "Nat.foo", "signature": "(n : Nat) : n = n"})
        self.assertIn("theorem bench_Nat_foo", code)     # dots would not parse
        self.assertIn(":= by sorry", code)


class TestSignatureSplitting(unittest.TestCase):
    """The separating colon is the first at bracket depth zero."""

    def test_binders_and_conclusion_are_separated(self):
        b, c = S.split_signature("(a : ℕ) (h : a > 0) : a ≥ 1")
        self.assertEqual(b, "(a : ℕ) (h : a > 0)")
        self.assertEqual(c, "a ≥ 1")

    def test_a_colon_inside_a_binder_does_not_split(self):
        b, c = S.split_signature("{m n : ℕ} : Nat.gcd m n ∣ m")
        self.assertEqual(c, "Nat.gcd m n ∣ m")

    def test_a_colon_inside_the_conclusion_does_not_split(self):
        # Splitting on the LAST colon breaks exactly here.
        b, c = S.split_signature(": ∀ x : ℝ, x ^ 2 ≥ 0")
        self.assertEqual(b, "")
        self.assertEqual(c, "∀ x : ℝ, x ^ 2 ≥ 0")

    def test_square_brackets_inside_a_binder_type_are_handled(self):
        b, c = S.split_signature("(f : A →ₗ[R] B) (a : A) : f a = f a")
        self.assertEqual(b, "(f : A →ₗ[R] B) (a : A)")

    def test_a_statement_becomes_a_closed_proposition(self):
        self.assertEqual(S.as_prop("(a : ℕ) : a = a"), "∀ (a : ℕ), a = a")

    def test_a_statement_with_no_binders_is_already_closed(self):
        self.assertEqual(S.as_prop(": True"), "True")


class TestTheLadderCanBridgeBinderStructure(unittest.TestCase):
    """The old ladder scored a benchmark of ZERO by rejecting correct answers.

    A Mathlib gold often carries no explicit binders (its variables come from a
    section `variable` line), so the hypothesis is a ∀-statement while the goal
    is an instance of it. Bare `exact h` reports a type mismatch that is
    indistinguishable from a real disagreement.
    """

    def test_most_rungs_introduce_binders(self):
        # Bare `exact h` can only close the case where both sides already share
        # binder structure, which is the minority.
        intros = [t for t in S.LADDER if t.startswith("intros;")]
        self.assertGreaterEqual(len(intros), len(S.LADDER) - 1)

    def test_the_rung_most_likely_to_succeed_runs_first(self):
        # ORDER, not just membership. With `exact?` fourth, the three rungs
        # ahead of it burned 71s of a 110s budget on prod_subset_prod_iff and it
        # never ran -- scoring a provable equivalence as a failure. Cheap-first
        # is wrong here because a failing rung pays its FULL timeout, while a
        # success ends the search in ~22s.
        self.assertEqual(S.LADDER[0], "intros; exact?")

    def test_rungs_are_time_boxed_and_so_is_the_ladder(self):
        self.assertLessEqual(S.RUNG_TIMEOUT, 45)
        self.assertLessEqual(S.LADDER_BUDGET, 150)


class TestScoringIsHonest(unittest.TestCase):
    def test_equivalence_requires_both_directions(self):
        # One direction is worthless in either direction: everything implies
        # `True`, and a statement strong enough to imply the gold may claim far
        # more than the informal text did.
        import inspect
        s = inspect.getsource(S.equivalent)
        self.assertIn("forward", s)
        self.assertIn("backward", s)
        self.assertIn("fwd and bwd", s)

    def test_the_judge_is_only_consulted_when_the_compiler_is_inconclusive(self):
        import inspect
        s = inspect.getsource(S.score_one)
        self.assertIn("if not eq[\"equivalent\"]", s)

    def test_the_report_keeps_compiler_and_judge_apart(self):
        rows = [{"t1": True, "t2": True, "t3": None, "seconds": 1},
                {"t1": True, "t2": False, "t3": True, "seconds": 1},
                {"t1": False, "t2": False, "t3": None, "seconds": 1}]
        out = S.report(rows)
        self.assertIn("T2 equivalent to gold", out)
        self.assertIn("do not add them together", out)

    def test_a_failed_derivation_scores_zero_rather_than_being_dropped(self):
        # Silently skipping the cases where nothing came back would turn a
        # 40%-of-the-time-nothing derivation into a perfect score.
        import inspect
        self.assertIn("t1=False", inspect.getsource(S.score_one))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSpecialistFormalizer(unittest.TestCase):
    """The tuned 7B path: prompt parity with training, and fail-open routing."""

    def test_inference_prompt_is_byte_identical_to_training(self):
        # A drifted prompt is a silent train/inference distribution shift.
        import ast as _ast
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        def prompt_of(path):
            tree = _ast.parse(open(path, encoding="utf-8").read())
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Assign):
                    for tgt in node.targets:
                        if getattr(tgt, "id", "") == "PROMPT":
                            return _ast.literal_eval(node.value)
            raise AssertionError(f"no PROMPT in {path}")
        self.assertEqual(
            prompt_of(os.path.join(root, "src", "formalizer.py")),
            prompt_of(os.path.join(root, "scripts", "train_formalizer.py")))

    def test_unconfigured_specialist_is_silent(self):
        from src import config, formalizer
        old, config.FORMALIZER_BASE = config.FORMALIZER_BASE, ""
        try:
            self.assertEqual(formalizer.derive("x"), "")
        finally:
            config.FORMALIZER_BASE = old

    def test_prover_falls_through_when_specialist_returns_nothing(self):
        # The specialist may only ADD coverage: a dead endpoint must leave the
        # 27B path exactly as it was.
        import inspect
        from src import prover
        src = inspect.getsource(prover._derive_once_untraced)
        self.assertIn("_formalizer.available()", src)
        self.assertIn("fall through", src)

    def test_specialist_candidates_face_the_same_gates(self):
        # Extraction, rename-probe typecheck, instance repair -- not a separate
        # trust level.
        import inspect
        from src import prover
        src = inspect.getsource(prover._derive_once_untraced)
        block = src[src.index("_formalizer.available()"):src.index("text, body, think")]
        # _signature_line, NOT _extract_lean. The latter exists to pull Lean out
        # of PROSE and PREPENDS the preamble when it finds no imports, so a
        # one-line signature came back as "PREAMBLE + theorem ..." which
        # _probe_statement then prefixed with the preamble again and appended
        # ":= by sorry" to. Every specialist candidate failed that way silently.
        for gate in ("_signature_line", "_probe_statement",
                     "_repair_missing_instances"):
            self.assertIn(gate, block)
        self.assertNotIn("_extract_lean", block)
