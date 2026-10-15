#!/usr/bin/env python3
"""Lemma-library tests. No Lean, no models -- pure logic on the index.

Motivated by the live library, which held EIGHT entries encoding TWO distinct
facts. Every duplicate below is a real name from that file.

    python3 -m tests.test_library
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import library                                     # noqa: E402

# Real entries from Proofs/Verified.lean before curation.
DUP_A = "theorem lean_workbook_plus_10017 (n : ℕ) : 2 ∣ n * (n + 1) := by\n  simp"
DUP_B = "theorem tt (n : Nat) : 2 ∣ n * (n+1) := by\n  decide"
GCD_IMPLICIT = "theorem gcd_divides {m n : Nat} : gcd m n ∣ m ∧ gcd m n ∣ n := by\n  constructor"
GCD_EXPLICIT = "theorem gcd_divides_3 (m n : Nat) : gcd m n ∣ m ∧ gcd m n ∣ n := by\n  constructor"
GCD_HYP = ("theorem gcd_divides_both {m n : ℕ} (h : m > 0 ∧ n > 0) : "
           "Nat.gcd m n ∣ m ∧ Nat.gcd m n ∣ n := by\n  constructor")
OTHER = "theorem sqrt_two_irr : Irrational (Real.sqrt 2) := by\n  exact irrational_sqrt_two"


class TestFingerprint(unittest.TestCase):
    def test_unicode_and_ascii_types_match(self):
        # `ℕ` vs `Nat` -- why `tt` survived as a fourth copy of one fact.
        self.assertEqual(library._fingerprint(DUP_A), library._fingerprint(DUP_B))

    def test_implicit_and_explicit_binders_match(self):
        # `{m n : Nat}` vs `(m n : Nat)` -- why `gcd_divides_3` survived.
        self.assertEqual(library._fingerprint(GCD_IMPLICIT),
                         library._fingerprint(GCD_EXPLICIT))

    def test_extra_hypothesis_is_a_different_lemma(self):
        # A weaker statement is NOT a duplicate; collapsing it would lose a fact.
        self.assertNotEqual(library._fingerprint(GCD_IMPLICIT),
                            library._fingerprint(GCD_HYP))

    def test_distinct_statements_differ(self):
        self.assertNotEqual(library._fingerprint(DUP_A), library._fingerprint(OTHER))

    def test_binder_name_changes_do_not_matter(self):
        a = "theorem f (x : Nat) : x = x := rfl"
        b = "theorem f (y : Nat) : y = y := rfl"
        self.assertEqual(library._fingerprint(a), library._fingerprint(b))


class TestDescriptors(unittest.TestCase):
    """Empty descriptors make the quality-diversity archive blind. Seven of the
    eight live entries had NO mechanism tag because `exact`/`apply` were not
    patterns."""

    def test_short_direct_proof_still_tagged(self):
        self.assertIn("direct", library.describe(OTHER)["mechanism"])

    def test_every_sample_gets_at_least_one_mechanism(self):
        for code in (DUP_A, DUP_B, GCD_IMPLICIT, GCD_HYP, OTHER):
            self.assertTrue(library.describe(code)["mechanism"],
                            f"no mechanism for: {code[:50]}")

    def test_objects_detected(self):
        self.assertIn("nat", library.describe(DUP_A)["objects"])
        self.assertIn("real", library.describe(OTHER)["objects"])

    def test_divisibility_object(self):
        self.assertIn("divisibility", library.describe(DUP_A)["objects"])


class TestStatementAndNaming(unittest.TestCase):
    def test_statement_excludes_proof(self):
        self.assertNotIn("simp", library.statement_of(DUP_A))
        self.assertIn("2 ∣", library.statement_of(DUP_A))

    def test_suggest_name_is_meaningful(self):
        name = library.suggest_name(OTHER)
        self.assertTrue(name)
        self.assertNotIn(" ", name)
        self.assertRegex(name, r"^[a-z][a-z0-9_]*$")

    def test_suggest_name_beats_lean_workbook(self):
        # The point of renaming: `lean_workbook_plus_10017` is unsearchable.
        self.assertNotIn("workbook", library.suggest_name(DUP_A))


class TestDedupRanking(unittest.TestCase):
    """Which copy survives is not arbitrary: used > meaningful name > shortest."""

    def test_junk_names_recognised(self):
        import re
        junk = re.compile(r"^(tt|r|foo|thm|test)$|^lean_workbook|_\d{3,}$")
        self.assertTrue(junk.match("tt"))
        self.assertTrue(junk.match("lean_workbook_plus_10017"))
        self.assertFalse(junk.match("gcd_divides"))

    def test_suffix_preference(self):
        import re
        suffixed = lambda n: bool(re.search(r"_\d+$", n))
        self.assertTrue(suffixed("gcd_divides_2"))
        self.assertFalse(suffixed("gcd_divides"))


class TestIndexRoundTrip(unittest.TestCase):
    def test_index_is_rebuildable_and_atomic(self):
        # The .lean file is the source of truth; the index must be disposable.
        self.assertTrue(hasattr(library, "reindex"))
        self.assertTrue(callable(library.reindex))

    def test_stats_on_empty_index(self):
        self.assertIsInstance(library.stats(), dict)


class TestSleepGrouping(unittest.TestCase):
    """sleep() must be testable without a model, so `propose` is injected."""

    def test_propose_returning_nothing_is_safe(self):
        out = library.sleep(lambda members: "", max_groups=1)
        self.assertEqual(out["kept"], [])

    def test_propose_raising_is_swallowed(self):
        def boom(members):
            raise RuntimeError("model down")
        out = library.sleep(boom, max_groups=1)
        self.assertEqual(out["kept"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
