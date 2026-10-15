#!/usr/bin/env python3
"""Wiring tests: are the new subsystems actually PLUGGED IN?

Every module here passed its own unit tests while being completely inert. A
bandit nobody calls, an archive nobody records to, and a library nobody
searches all look healthy in isolation. These tests assert the connections --
the failure mode is silence, not an exception.

No models, no Lean, no network: imports, signatures, and call-graph checks.

    python3 -m tests.test_integration
"""

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import agent, archive, bandit, config, evolve, library, prover  # noqa: E402
from src import mathlib_kb, mathlib_search                               # noqa: E402


def _src(obj) -> str:
    return inspect.getsource(obj)


class TestBanditIsWired(unittest.TestCase):
    def test_strategies_are_named_and_unpack_consistently(self):
        for s in prover.STRATEGIES:
            self.assertEqual(len(s), 4, s)
        self.assertEqual(len(prover.STRATEGY_NAMES), len(prover.STRATEGIES))

    def test_every_name_resolves(self):
        for n in prover.STRATEGY_NAMES:
            self.assertIn(n, prover._BY_NAME)

    def test_bandit_default_arms_match_prover_strategies(self):
        # bandit cannot import prover (prover imports bandit), so the default
        # arm list is duplicated -- and duplication drifts unless asserted.
        self.assertEqual(set(bandit.ARMS), set(prover.STRATEGY_NAMES))

    def test_prover_selects_via_bandit(self):
        s = _src(prover.prove_theorem)
        self.assertIn("bandit", s)
        self.assertIn("select", s)

    def test_prover_reports_outcomes_to_bandit(self):
        # Selection without update is a random permutation with extra steps.
        self.assertIn("update", _src(prover.prove_theorem))


class TestArchiveIsWired(unittest.TestCase):
    def test_prover_refuses_filled_cells(self):
        self.assertIn("occupied", _src(prover.prove_theorem))

    def test_successes_are_recorded(self):
        self.assertIn("archive.record", _src(prover._record))

    def test_failures_are_recorded(self):
        # A failed cell is explored-and-hard, not untouched; counting only
        # successes would invite the same dead end again.
        self.assertIn("archive.record", _src(prover._remember))

    def test_goal_setter_exists_and_agent_calls_it(self):
        self.assertTrue(callable(prover.set_goal))
        self.assertIn("set_goal", _src(agent.run))

    def test_exploration_map_reaches_the_prompt(self):
        self.assertIn("archive", _src(agent._anchor_message))


class TestLibraryIsWired(unittest.TestCase):
    def test_proofs_are_offered_the_existing_library(self):
        self.assertIn("library.suggest", _src(prover._attempt))

    def test_usage_is_recorded_on_success(self):
        self.assertIn("note_uses", _src(prover._record))

    def test_duplicate_is_not_reported_as_failure(self):
        self.assertIn("duplicate", _src(prover._record))


class TestRetrievalIsWired(unittest.TestCase):
    def test_dense_retrieval_precedes_keyword(self):
        s = _src(mathlib_search.suggest_for_unknown)
        self.assertIn("mathlib_kb", s)
        self.assertIn("available", s)

    def test_keyword_fallback_survives(self):
        # Dense must be additive: an unbuilt index or a dead embed call has to
        # fall through, not raise.
        self.assertIn("except Exception", _src(mathlib_search.suggest_for_unknown))

    def test_suggest_uses_the_measured_best_retriever(self):
        # Fusion was tried and MEASURED WORSE: dense+BM25 and two-model both
        # held recall at 3/8 while pushing the right lemma down the ranks
        # (irrational_sqrt_two #1 -> #13, Coprime.pow #4 -> #19). This block
        # shows only the top few, so rank is what matters.
        s = _src(mathlib_kb.suggest)
        self.assertIn("search(statement", s)
        self.assertNotIn("search_hybrid(statement", s)


class TestConfigCoherence(unittest.TestCase):
    def test_orchestrator_is_a_base_tag(self):
        # The `-32768` derived tags were deleted; pointing at one would break
        # every run with a missing-model error.
        self.assertNotIn("-32768", config.ORCHESTRATOR_MODEL)

    def test_uncapped_generation_is_bounded_by_a_timeout(self):
        if config.ORCH_MAX_TOKENS < 0:
            self.assertGreater(prover._gen_timeout(-1), 0)
            self.assertLessEqual(prover._gen_timeout(-1), config.ORCH_TIMEOUT_CAP)

    def test_batch_prover_max_tokens_fit_the_served_window(self):
        # vLLM serves --max-model-len 16384; 16000 max_tokens leaves no room
        # for the prompt and every request is rejected.
        if config.PROVER_BATCH_BASE:
            self.assertLess(config.PROVER_MAX_TOKENS, 16384)

    def test_think_option_is_valid(self):
        opt = prover._think_opt()
        if opt:
            self.assertIn(opt["think"], ("low", "medium", "high", False))


class TestNoCircularImports(unittest.TestCase):
    def test_modules_import_cleanly_in_isolation(self):
        import importlib
        for name in ("archive", "bandit", "evolve", "library",
                     "mathlib_kb", "mathlib_search", "prover", "agent"):
            importlib.import_module(f"src.{name}")

    def test_bandit_does_not_import_prover(self):
        self.assertNotIn("import prover", inspect.getsource(bandit))

    def test_archive_does_not_import_prover(self):
        self.assertNotIn("import prover", inspect.getsource(archive))


class TestEvolveContract(unittest.TestCase):
    def test_generation_is_injected_not_imported(self):
        # So the search is testable without a model, and so the caller chooses
        # the fast/strong model split rather than this module hard-coding it.
        params = inspect.signature(evolve.step).parameters
        self.assertIn("propose", params)

    def test_sleep_generation_is_injected_too(self):
        self.assertIn("propose", inspect.signature(library.sleep).parameters)

    def test_novelty_defers_to_the_archive(self):
        self.assertIn("archive", _src(evolve.novelty))




class TestConjectureToolIsWired(unittest.TestCase):
    """evolve.py passed 11 unit tests while being completely unreachable: no
    tool exposed it, so the agent could never run a generation."""

    def test_tool_is_exposed_and_implemented(self):
        from src import tools
        names = {t["function"]["name"] for t in tools.specs()}
        self.assertIn("conjecture", names)
        self.assertIn("conjecture", tools.REGISTRY)

    def test_every_spec_has_an_implementation(self):
        from src import tools
        names = {t["function"]["name"] for t in tools.specs()}
        self.assertFalse(names - set(tools.REGISTRY))

    def test_unavailable_model_degrades_to_empty_not_exception(self):
        # A dead model must not take the generation down with it.
        p = evolve.llm_proposer(model="definitely-not-a-model")
        self.assertEqual(p([], 0, "goal"), [])

    def test_proposer_signature_matches_step_contract(self):
        params = list(inspect.signature(evolve.llm_proposer()).parameters)
        self.assertEqual(params[:3], ["parents", "island", "goal"])

    def test_generation_wrapper_uses_the_injected_proposer(self):
        self.assertIn("llm_proposer", _src(evolve.generation))




class TestSleepPhaseIsWired(unittest.TestCase):
    """library.sleep() existed, was tested, and nothing ever called it."""

    def test_config_flag_exists(self):
        self.assertIsInstance(config.LIBRARY_SLEEP, bool)

    def test_agent_runs_the_sleep_phase(self):
        s = _src(agent.run)
        self.assertIn("sleep_phase", s)
        self.assertIn("LIBRARY_SLEEP", s)

    def test_sleep_runs_after_the_problem_not_during(self):
        # It costs a model call and a full recompile; nothing it produces helps
        # the problem that just ended.
        s = _src(agent.run)
        self.assertLess(s.index("while not _stop"), s.index("sleep_phase"))

    def test_sleep_phase_returns_early_on_a_small_library(self):
        # Exercises the guard, NOT the model path. Calling sleep_phase() for
        # real took this suite from 3s to 274s -- a suite that slow stops being
        # run, which costs more than the coverage it buys.
        orig = library._load_index
        library._load_index = lambda: {}
        try:
            out = library.sleep_phase()
            self.assertEqual(out["kept"], [])
            self.assertIn("too small", out.get("note", ""))
        finally:
            library._load_index = orig

    def test_generalisation_is_verified_before_keeping(self):
        # "The model said so" is not a standard the library accepts.
        self.assertIn("add(", _src(library.sleep))

    def test_generalizer_degrades_on_dead_model(self):
        self.assertEqual(library.llm_generalizer(model="not-a-real-model")([]), "")




class TestEverythingLivesInTheRepo(unittest.TestCase):
    """Only genuinely external INSTALLS may sit outside: ollama's model blobs
    and the elan toolchain. Project data -- the Lean project, the lemma
    library, indexes, journals -- belongs to the repo."""

    def test_lean_project_is_in_repo(self):
        self.assertTrue(config.LEAN_PROJECT.startswith(config.ROOT),
                        config.LEAN_PROJECT)

    def test_lemma_library_is_in_repo(self):
        self.assertTrue(library.PATH.startswith(config.ROOT), library.PATH)
        self.assertTrue(library.INDEX.startswith(config.ROOT), library.INDEX)

    def test_learned_state_is_in_repo(self):
        for p in (archive.PATH, bandit.PATH, evolve.PATH, mathlib_kb.KB_DIR):
            self.assertTrue(p.startswith(config.ROOT), p)

    def test_only_the_toolchain_escapes(self):
        # Lean core source ships with the compiler install, like a model.
        outside = [r for r, _ in mathlib_kb.sources()
                   if not r.startswith(config.ROOT)]
        self.assertTrue(all(".elan" in r for r in outside), outside)

    def test_remote_serve_script_is_version_controlled(self):
        # It existed only on the server for a while, so the setup could not be
        # reproduced from a clean checkout.
        self.assertTrue(os.path.exists(os.path.join(config.ROOT, "scripts",
                                                    "serve_goedel.sh")))




class TestNoOrphanModules(unittest.TestCase):
    """Dead code accumulates silently. `src/notebook.py` sat unreferenced for
    the whole project, flagged in the README and never removed.

    Written with AST rather than string matching: `from . import prover` has
    `module=None`, so a naive scan reports half the codebase as orphaned and
    the real orphan hides among the false positives.
    """

    @staticmethod
    def _graph():
        import ast
        root = config.ROOT
        src = os.path.join(root, "src")
        files = {}
        for fn in sorted(os.listdir(src)):
            if fn.endswith(".py") and fn != "__init__.py":
                files[fn[:-3]] = ast.parse(open(os.path.join(src, fn)).read())
        entry = os.path.join(root, "run.py")
        if os.path.exists(entry):
            files["run.py"] = ast.parse(open(entry).read())
        imports = {m: set() for m in files}
        for m, tree in files.items():
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    # Three shapes reach a sibling module, and missing any one
                    # of them reports live code as dead:
                    #   from . import prover        module=None,  names=modules
                    #   from src import agent       module="src", names=modules
                    #   from .prover import x       module="prover"
                    if n.module is None or n.module.split(".")[-1] not in files:
                        for a in n.names:
                            if a.name in files:
                                imports[m].add(a.name)
                    if n.module and n.module.split(".")[-1] in files:
                        imports[m].add(n.module.split(".")[-1])
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        if a.name.split(".")[-1] in files:
                            imports[m].add(a.name.split(".")[-1])
        return files, imports

    def test_every_module_is_reachable(self):
        files, imports = self._graph()
        orphans = [m for m in files if m != "run.py"
                   and not any(m in imports[o] for o in files if o != m)]
        self.assertFalse(orphans, f"unreachable modules: {orphans}")

    def test_the_control_loops_are_on_the_live_path(self):
        # Each of these was inert at some point while its own tests passed.
        agent_src = inspect.getsource(agent)
        for mod, marker in (("plan", "plan.build"), ("propose", "propose.choose"),
                            ("archive", "archive.advice"), ("library", "sleep_phase")):
            self.assertIn(marker, agent_src, f"{mod} not reached from agent")

    def test_every_tool_module_is_reached_through_tools(self):
        from src import tools
        s = inspect.getsource(tools)
        for marker in ("intuition", "evolve", "prove_theorem", "search_mathlib",
                       "run_python", "journal"):
            self.assertIn(marker, s, f"{marker} not exposed via tools")

    def test_prover_reaches_bandit_library_and_archive(self):
        s = inspect.getsource(prover)
        for marker in ("bandit", "library.suggest", "archive"):
            self.assertIn(marker, s, f"prover does not use {marker}")

    def test_retrieval_chain_is_connected(self):
        self.assertIn("mathlib_kb", inspect.getsource(mathlib_search))


if __name__ == "__main__":
    unittest.main(verbosity=2)
