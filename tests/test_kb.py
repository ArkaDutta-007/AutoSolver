#!/usr/bin/env python3
"""Knowledge-base tests: extraction, tokenisation, fusion, index safety.

The retrieval layer shipped four real bugs before it had any tests -- bare
names, namespaces parsed out of prose, `\\b` around non-word characters, and a
fusion that made retrieval WORSE. Everything here is a regression guard against
one of those.

Network-dependent paths (embedding) are exercised only when an index exists;
everything else is pure logic and always runs.

    python3 -m tests.test_kb
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from src import mathlib_kb as kb                            # noqa: E402


class TestTokenisation(unittest.TestCase):
    """BM25 must read `Nat.factorization_gcd` the way a mathematician does."""

    def test_dotted_and_underscored_names_split(self):
        toks = set(kb._tokens("Nat.factorization_gcd"))
        self.assertEqual(toks, {"nat", "factorization", "gcd"})

    def test_camel_case_splits(self):
        self.assertIn("compact", kb._tokens("IsCompact.exists_isMaxOn"))

    def test_acronym_runs_survive(self):
        self.assertIn("gcd", kb._tokens("GCD"))

    def test_empty_input_is_safe(self):
        self.assertEqual(kb._tokens(""), [])
        self.assertEqual(kb._tokens(None), [])


class TestSources(unittest.TestCase):
    def test_lean_core_included_not_just_mathlib(self):
        prefixes = {p for _, p in kb.sources()}
        if not prefixes:
            self.skipTest("no Lean sources on this machine")
        # Nat.gcd_mul_lcm lives in core, and it is the lemma that closes a goal
        # this system once spent 1,190s failing.
        self.assertIn("Init", prefixes)

    def test_every_source_root_exists(self):
        for root, _ in kb.sources():
            self.assertTrue(os.path.isdir(root), root)


class TestExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decls = kb.extract(limit=4000) if kb.sources() else []

    def setUp(self):
        if not self.decls:
            self.skipTest("no Lean sources on this machine")

    def test_no_prose_derived_namespaces(self):
        # "use a namespace instead." once produced `instead..interior_sInter`.
        bad = [d["name"] for d in self.decls if ".." in d["name"]
               or d["name"].startswith(".")]
        self.assertFalse(bad, bad[:5])

    def test_names_unique_within_extraction(self):
        names = [d["name"] for d in self.decls]
        self.assertEqual(len(names), len(set(names)))

    def test_every_record_has_the_expected_shape(self):
        for d in self.decls[:200]:
            self.assertEqual(set(d), {"name", "kind", "text", "module"})
            self.assertTrue(d["name"] and d["module"])

    def test_module_paths_are_dotted_not_slashed(self):
        for d in self.decls[:200]:
            self.assertNotIn("/", d["module"])


class TestFusion(unittest.TestCase):
    """RRF is rank-based on purpose: cosine and BM25 have no shared scale."""

    @staticmethod
    def _run(*names):
        return [{"name": n, "text": "", "kind": "theorem", "module": "M"} for n in names]

    def test_unanimous_top_hit_stays_first(self):
        out = kb._rrf([self._run("a", "b", "c"), self._run("a", "c", "b")], k=3)
        self.assertEqual(out[0]["name"], "a")

    def test_union_is_preserved(self):
        out = kb._rrf([self._run("a", "b"), self._run("c", "d")], k=10)
        self.assertEqual({h["name"] for h in out}, {"a", "b", "c", "d"})

    def test_weighting_reduces_a_weak_runs_influence(self):
        # Full-weight BM25 dragged `irrational_sqrt_two` from rank 1 to 13.
        strong, weak = self._run("target", "x", "y"), self._run("junk", "target")
        full = kb._rrf([strong, weak], k=3)
        down = kb._rrf([strong, weak], k=3, weights=[1.0, 0.1])
        self.assertEqual(down[0]["name"], "target")
        self.assertIn(full[0]["name"], {"target", "junk"})

    def test_single_run_is_order_preserving(self):
        out = kb._rrf([self._run("a", "b", "c")], k=3)
        self.assertEqual([h["name"] for h in out], ["a", "b", "c"])

    def test_empty_runs_are_safe(self):
        self.assertEqual(kb._rrf([], k=5), [])


class TestModelPlumbing(unittest.TestCase):
    def test_only_nomic_gets_task_prefixes(self):
        # Feeding nomic's prefixes to a model never trained on them is a silent
        # quality loss, not an error.
        self.assertEqual(kb._prefixes("nomic-embed-text"), (kb.Q_PREFIX, kb.D_PREFIX))
        self.assertEqual(kb._prefixes("bge-m3"), ("", ""))
        self.assertEqual(kb._prefixes("mxbai-embed-large"), ("", ""))

    def test_vector_paths_are_per_model_and_filesystem_safe(self):
        a, b = kb._vec_path("bge-m3"), kb._vec_path("nomic-embed-text")
        self.assertNotEqual(a, b)
        for p in (a, b, kb._vec_path("hf.co/some/model:Q8_0")):
            self.assertNotIn("/", os.path.basename(p))

    def test_built_models_subset_of_ensemble(self):
        self.assertTrue(set(kb.built_models()) <= set(kb.ENSEMBLE) | {kb.EMBED_MODEL})


class TestIndexSafety(unittest.TestCase):
    """A vector file built against a different corpus would mis-label every
    hit: row i would name a declaration that is no longer row i."""

    def setUp(self):
        if not kb.available():
            self.skipTest("no index built")
        kb._cache.pop("decls", None)

    def test_row_count_matches_corpus(self):
        for m in kb.built_models():
            vecs, decls = kb._load(m)
            if vecs is not None:
                self.assertEqual(vecs.shape[0], len(decls), m)

    def test_mismatched_index_is_refused_not_returned(self):
        model = (kb.built_models() or [kb.EMBED_MODEL])[0]
        key = f"vecs::{model}"
        vecs, decls = kb._load(model)
        if vecs is None:
            self.skipTest("index unavailable")
        kb._cache[key] = np.zeros((len(decls) - 1, vecs.shape[1]), dtype=np.float32)
        try:
            got, _ = kb._load(model)
            self.assertIsNone(got, "mismatched index must be refused")
        finally:
            kb._cache[key] = vecs

    def test_vectors_are_prenormalised(self):
        model = (kb.built_models() or [kb.EMBED_MODEL])[0]
        vecs, _ = kb._load(model)
        if vecs is None:
            self.skipTest("index unavailable")
        norms = np.linalg.norm(vecs[:200], axis=1)
        nz = norms[norms > 0]
        if nz.size:
            self.assertTrue(np.allclose(nz, 1.0, atol=2e-2), f"norms {nz[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
