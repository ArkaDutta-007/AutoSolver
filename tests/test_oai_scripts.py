#!/usr/bin/env python3
"""The openai/math mining scripts: prompt contract and Lean parsing. No Lean, no model.

A drifted prompt is a silent distribution shift between what the prover was
trained on and what the Autosolver asks it, so the copies in scripts/ are
checked against what src/prover.py actually sends.

    python3 -m unittest tests.test_oai_scripts
"""

import argparse
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import prover                                      # noqa: E402
import oai_extract as E                                     # noqa: E402
import oai_sample as S                                      # noqa: E402
import train_prover as T                                    # noqa: E402
import oai_pipeline as P                                    # noqa: E402

SRC = '''import Mathlib

/-! # A module doc mentioning theorem fake_in_doc : True -/

namespace OAI

namespace Cocycle
open Set

noncomputable def weight (k : ℤ) : ℝ := 1

/-- Monotone from a one-step recurrence. -/
theorem antitone_of_step {F w : ℤ → ℝ} (hF : ∀ k, 0 ≤ F k) (hw : ∀ k, w k ≤ 1)
    (hstep : ∀ k, F (k+1) = w k * F k) : Antitone F := by
  apply antitone_int_of_succ_le
  intro k
  rw [hstep]
  simpa using mul_le_mul_of_nonneg_right (hw k) (hF k)

-- theorem fake_in_comment : True := trivial
theorem uses_local (k : ℤ) : weight k = 1 := by
  have h1 : weight k = 1 := by simp only [weight]
  have h2 : (1 : ℝ) = 1 := rfl
  simpa using h1

open MeasureTheory in
@[simp] lemma open_in_front (a b : ℝ) (h : a ≤ b) : Icc a b ⊆ Icc a (b + 1) := by
  have ha : a ≤ a := le_rfl
  have hb : b ≤ b + 1 := by linarith
  exact Icc_subset_Icc ha hb

theorem too_short : (1 : ℕ) = 1 := rfl

section
variable {n : ℕ}
theorem under_variable (h : 0 < n) : 0 < n * n := by
  have h2 : 0 < n * n := Nat.mul_pos h h
  have h3 : n * n = n * n := rfl
  exact h2
end

end Cocycle
end OAI
'''


class TestContract(unittest.TestCase):
    def test_prompt_is_byte_identical_to_the_autosolvers(self):
        sent = {}

        def post(url, json, timeout):
            sent.update(json)
            raise RuntimeError("no server in tests")

        stmt = "theorem t (n : ℕ) : n = n"
        with mock.patch.object(prover.requests, "post", post):
            self.assertEqual(prover.sample_proofs(stmt, 4, "http://x", "m"), [])
        self.assertEqual(sent["messages"][0]["content"], S.prompt(stmt))
        self.assertEqual(S.BASE_SYSTEM, prover.BASE_SYSTEM)
        self.assertEqual(S.PREAMBLE, prover.PREAMBLE)
        self.assertEqual(E.PREAMBLE, prover.PREAMBLE)

    def test_attempt_and_repair_turns_are_the_autosolvers(self):
        calls = []

        def call(messages, *a, **k):
            calls.append([dict(m) for m in messages])
            return "```lean4\ntheorem t : 1 = 2 := by\n  rfl\n```", False

        err = "repl:3:2: error: unknown constant `Nat.foo_bar`\nrepl:4:0: error: AddGroup ℕ"
        informal, stmt = "One is two.", "theorem t : (1 : ℕ) = 2"
        pinned = (f"{informal}\n\nProve EXACTLY this Lean 4 theorem, keeping the "
                  f"signature verbatim and supplying only the proof:\n\n{stmt}")
        with mock.patch.object(prover, "_call", call), \
                mock.patch.object(prover, "check_and_fix", lambda c: (c, {"ok": False, "errors": err})), \
                mock.patch.object(prover.library, "suggest", lambda s: ""), \
                mock.patch.object(prover.mathlib_search, "suggest_for_unknown", lambda *a: ""):
            out = prover._attempt(pinned, 1.0, "", 3, time.time() + 60, "http://x", "m")
        self.assertEqual(out["reason"], "stalled")             # same error twice
        self.assertEqual(calls[0], S.attempt_messages(stmt, informal))
        self.assertEqual(calls[1][-1], S.repair_turn(err))
        self.assertIn("IMPORTANT", calls[1][-1]["content"])
        self.assertEqual([(r.pattern, m) for r, m in S.TRAPS],
                         [(r.pattern, m) for r, m in prover._TRAPS])


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.body


class TestProverRequests(unittest.TestCase):
    def test_max_tokens_fits_the_servers_window(self):
        seen = []

        def post(url, json, timeout):
            if url.endswith("/tokenize"):
                self.assertEqual(url, "http://h:8001/tokenize")
                return FakeResponse({"count": 900, "max_model_len": 16384})
            seen.append(json["max_tokens"])
            return FakeResponse({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]})

        with mock.patch.object(prover.requests, "post", post):
            prover._call([{"role": "user", "content": "q"}], 1.0, "http://h:8001/v1", "m", 28000)
            prover._call([{"role": "user", "content": "q"}], 1.0, "http://h:8001/v1", "m", 4000)
        self.assertEqual(seen, [16384 - 900, 4000])

    def test_without_tokenize_the_budget_passes_through(self):
        def post(url, json, timeout):
            if url.endswith("/tokenize"):
                return FakeResponse({}, 404)
            return FakeResponse({"choices": [{"message": {"content": str(json["max_tokens"])}}]})

        with mock.patch.object(prover.requests, "post", post):
            self.assertEqual(prover._call([], 1.0, "http://o/v1", "m", 7000)[0], "7000")

    def test_batch_keeps_the_chunks_that_finished(self):
        def post(url, json, timeout):
            if url.endswith("/tokenize"):
                raise RuntimeError("not vLLM")
            post.k = getattr(post, "k", 0) + 1
            if post.k == 2:
                raise prover.requests.exceptions.ReadTimeout("slow chunk")
            return FakeResponse({"choices": [{"message": {"content": f"p{i}"}} for i in range(json["n"])]})

        with mock.patch.object(prover.requests, "post", post):
            got = prover.sample_proofs("theorem t : True", 10, "http://x/v1", "m")
        self.assertEqual(post.k, 3)                            # 4 + 4 + 2
        self.assertIn(len(got), (6, 8))                        # one chunk lost, the rest kept


class TestScan(unittest.TestCase):
    def setUp(self):
        local = {n.split(".")[-1] for n in E.DECLARED.findall(SRC)}
        with mock.patch("builtins.open", mock.mock_open(read_data=SRC)):
            self.got = {c["name"]: c for c in E.scan_file("x.lean", "OAI.A.B.C", local)}

    def test_keeps_standalone_lemmas_with_their_context(self):
        self.assertEqual(set(self.got), {"antitone_of_step", "open_in_front"})
        a = self.got["antitone_of_step"]
        self.assertEqual(a["opens"], ["open Set"])        # OAI namespaces not reopened
        self.assertIn("Monotone from a one-step", a["doc"])
        self.assertTrue(a["sig"].startswith("{F w : ℤ → ℝ}") and a["sig"].endswith("Antitone F"))
        self.assertTrue(a["proof"].startswith("by\n  apply antitone_int_of_succ_le"))
        self.assertEqual(self.got["open_in_front"]["opens"], ["open Set", "open MeasureTheory"])

    def test_split_assign_ignores_bracketed_defaults(self):
        self.assertEqual(E.split_assign("(x : ℕ := 3) : x = x := rfl"),
                         ("(x : ℕ := 3) : x = x", "rfl"))

    def test_open_units_split_simple_opens_only(self):
        self.assertEqual(E.open_units("open scoped A B"), ["open scoped A", "open scoped B"])
        self.assertEqual(E.open_units("open Finset hiding sum"), ["open Finset hiding sum"])


class TestSample(unittest.TestCase):
    def test_extract_takes_the_last_declaration_block_under_preamble(self):
        text = ("plan\n```lean4\ntheorem a : True := trivial\n```\nfinal:\n"
                "```lean4\nimport Mathlib\ntheorem b : True := by trivial\n```")
        self.assertEqual(S.extract(text), S.PREAMBLE + "theorem b : True := by trivial")
        self.assertEqual(S.extract("no code at all"), "")

    def test_statement_is_stable_and_carries_its_opens(self):
        p = {"module": "OAI.A.B.C", "name": "foo", "opens": ["open Set"], "sig": ": True",
             "proof": "trivial"}
        self.assertEqual(S.statement(p), S.statement(dict(p)))
        self.assertTrue(S.statement(p).startswith("open Set in\ntheorem foo : True"))
        self.assertEqual(S.split(p), S.split(dict(p, name="other")))   # by family

    def test_hint_leak_filter(self):
        self.assertTrue(S.LEAKED.search("Following the reference proof, we ..."))
        self.assertFalse(S.LEAKED.search("We prove the bound by induction on n."))


class TestBehaviour(unittest.TestCase):
    def test_looped_flags_circles_not_proofs(self):
        proof = "\n".join(f"  have h{i} : x{i} ≤ y{i} + {i} := by linarith [hx{i}, hy{i}]"
                           for i in range(40))
        self.assertFalse(S.looped(proof))
        circle = "Let me reconsider whether the bound holds for every n here.\n" * 30
        self.assertTrue(S.looped("thinking...\n" + circle))
        self.assertTrue(S.looped("x" * 2000 + "abcdefghij" * 300))
        repetitive = "  nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), mul_pos ha hb]\n" * 20
        self.assertFalse(S.looped(f"<think>\nSquares.\n</think>\n```lean4\n{repetitive}```"))
        self.assertTrue(S.looped(f"<think>\n```lean4\n{repetitive}"))     # cut off inside the block

    def test_summary_reports_pass_rates_and_not_concluding(self):
        rows = [dict(pid="a", ok=True, why="ok", ntok=100), dict(pid="a", ok=False, why="truncated", ntok=900),
                dict(pid="b", ok=False, why="repeat", ntok=300, loop=True),
                dict(pid="b", ok=False, why="compile", ntok=300)]
        path = os.path.join(tempfile.mkdtemp(), "s.jsonl")
        with open(path, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))
        s = S.summarize(path)
        self.assertEqual((s["problems"], s["pass1"], s["passn"]), (2, 0.25, 0.5))
        self.assertEqual((s["truncated"], s["repeat"], s["looped"]), (0.25, 0.25, 0.25))
        self.assertEqual((s["win_tokens"], s["solved"]), (100, ["a"]))


class TestMix(unittest.TestCase):
    def test_shortest_wins_trained_failures_pushed_away(self):
        d = tempfile.mkdtemp()
        ctx = lambda pid: [{"role": "user", "content": "P" + pid}]
        rep = [{"role": "user", "content": "Pr"}, {"role": "assistant", "content": "bad"},
               {"role": "user", "content": "fix"}]

        def rec(pid, mode, text, ok, why="ok", ntok=10, loop=False, messages=None):
            return json.dumps({"pid": pid, "mode": mode, "messages": messages or ctx(pid),
                               "text": text, "ok": ok, "why": why, "ntok": ntok, "loop": loop})

        files = {
            "plain": [rec("easy", "plain", "e1", True, ntok=10), rec("easy", "plain", "e2", True, ntok=20),
                      rec("mid", "plain", "m1", True, ntok=50), rec("mid", "plain", "m1", True, ntok=50),
                      rec("mid", "plain", "m2", False, "truncated", 14000),
                      rec("mid", "plain", "m3", True, ntok=9, loop=True),
                      rec("hard", "plain", "x", False, "compile")],
            "hint": [rec("hard", "hint", f"h{i}", True, ntok=10 + i) for i in range(3)],
            "repair": [rec("r", "repair", "r3", False, "compile", messages=rep),
                       rec("r", "repair", "r1", True, ntok=30, messages=rep),
                       rec("r", "repair", "r2", False, "repeat", messages=rep)],
        }
        paths = []
        for name, lines in files.items():
            paths.append(os.path.join(d, name + ".jsonl"))
            with open(paths[-1], "w") as fh:
                fh.write("\n".join(lines))
        items = T.rows(paths, per_problem=2, easy_share=0.34)
        got = sorted((it["kind"], it.get("target") or f"{it['chosen']}>{it['rejected']}")
                     for it in items)
        self.assertEqual(got, [("pair:compile", "h0>x"), ("pair:repeat", "r1>r2"),
                               ("pair:truncated", "m1>m2"), ("rationalized", "h0"),
                               ("rationalized", "h1"), ("repair", "r1"), ("replay", "e1"),
                               ("sharpen", "m1")])
        hard = [it for it in items if it.get("target", "").startswith("h")]
        self.assertTrue(all(it["messages"] == ctx("hard") for it in hard))   # trained without the hint


class TestResume(unittest.TestCase):
    def test_a_new_vm_does_not_trust_the_last_vms_weights_path(self):
        with tempfile.TemporaryDirectory() as d:
            c = P.Cfg(out=d)
            self.assertEqual(P.recorded_model(c), "")                      # first run
            gone = os.path.join(d, "old-vm-cache", "snapshot")
            P.jdump({"dir": gone}, c.p("model.json"))
            self.assertEqual(P.recorded_model(c), "")                      # new VM: fetch again
            os.makedirs(gone)
            open(os.path.join(gone, "config.json"), "w").close()
            self.assertEqual(P.recorded_model(c), gone)                    # same machine

    def test_sampler_cuts_a_partial_line_tops_up_and_skips_earlier_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            pairs, out, before = (os.path.join(d, f) for f in ("pairs.jsonl", "out.jsonl", "r1.jsonl"))
            with open(pairs, "w") as fh:
                fh.writelines(json.dumps({"module": "M.a.b", "name": f"t{i}", "opens": [], "sig": ": True",
                                          "proof": "by trivial"}) + "\n" for i in range(3))
            a0, a1, a2 = (p["pid"] for p in S.load(pairs))
            with open(out, "w") as fh:
                fh.writelines(json.dumps({"pid": a0, "ok": False, "why": "compile", "k": k}) + "\n" for k in range(2))
                fh.writelines(json.dumps({"pid": a1, "ok": True, "why": "ok", "k": k}) + "\n" for k in range(4))
                fh.write(json.dumps({"pid": a1, "ok": True})[:12])                 # killed mid-write
            with open(before, "w") as fh:
                fh.write(json.dumps({"pid": a2}) + "\n")
            asked = []
            proof = ("```lean4\ntheorem t0 : True := by trivial\n```", "stop", 5)
            a = argparse.Namespace(pairs=pairs, split="all", per_family=0, only=None, out=out, skip=[before],
                                   mode="plain", n=4, limit=0, env=d, lean_workers=1, hours=0, base="",
                                   model="m", max_tokens=100, model_len=1000, inflight=2, repair_from=[])
            with mock.patch.object(S, "Lean") as lean, mock.patch.object(S, "fit", return_value=100), \
                    mock.patch.object(S, "sample", side_effect=lambda *x: asked.append(x[3]) or [proof] * x[3]):
                lean.return_value.check.return_value = (True, "ok", "")
                S.run(a)
            self.assertEqual(asked, [2])                     # t0 only, and only the 2 it lacked
            with open(out, encoding="utf-8") as fh:
                rows = [json.loads(x) for x in fh]                           # every line whole again
            self.assertEqual([r["k"] for r in rows if r["pid"] == a0], [0, 1, 2, 3])
            self.assertEqual(sum(r["pid"] == a1 for r in rows), 4)
            self.assertFalse(any(r["pid"] == a2 for r in rows))               # an earlier round's

    def test_adapter_checkpoints_survive_a_kill_at_any_point(self):
        class Model:
            def save_pretrained(self, path):
                os.makedirs(path, exist_ok=True)
                open(os.path.join(path, "adapter_model.safetensors"), "w").close()

        with tempfile.TemporaryDirectory() as d:
            ad = os.path.join(d, "adapter")
            state = lambda: P.jload(os.path.join(ad, "state.json"))["step"]
            T.save_adapter(Model(), ad, {"step": 10})
            T.save_adapter(Model(), ad, {"step": 20})
            self.assertEqual((state(), os.path.exists(ad + ".new")), (20, False))
            Model().save_pretrained(ad + ".new")                     # killed after the old one's
            P.jdump({"step": 30}, os.path.join(ad + ".new", "state.json"))       # label went
            os.remove(os.path.join(ad, "state.json"))
            T.recover_adapter(ad)
            self.assertEqual(state(), 30)
            Model().save_pretrained(ad + ".new")                     # killed while writing the next
            T.recover_adapter(ad)
            self.assertEqual(state(), 30)


class TestRounds(unittest.TestCase):
    def test_pass_at_k_and_the_paired_comparison(self):
        self.assertEqual((P.pass_at(4, 1, 1), P.pass_at(4, 1, 2)), (0.25, 0.5))
        self.assertEqual((P.pass_at(16, 0, 10), P.pass_at(16, 7, 10)), (0.0, 1.0))
        self.assertAlmostEqual(P.pass_at(16, 1, 10), 10 / 16)
        with tempfile.TemporaryDirectory() as d:
            def write(name, counts):
                with open(os.path.join(d, name), "w") as fh:
                    for pid, (n, ok) in counts.items():
                        fh.writelines(json.dumps({"pid": pid, "ok": i < ok, "why": "ok" if i < ok else "compile"})
                                      + "\n" for i in range(n))
                return os.path.join(d, name)
            b, t = write("b.jsonl", {"a": (16, 2), "b": (16, 0)}), write("t.jsonl", {"a": (16, 8), "b": (4, 1)})
            full = P.compare(b, t, need=16)                   # b has 4 tuned samples: left out
            self.assertEqual((full["problems"], full["k"]), (1, 10))
            self.assertAlmostEqual(full["delta"], 8 / 16 - 2 / 16)
            self.assertEqual(sorted(full["tuned"]["passk"]), [1, 2, 4, 8, 10])
            both = P.compare(b, t)
            self.assertEqual((both["problems"], both["k"]), (2, 4))   # pass@k only as far as 4 samples go

    def test_one_labelled_merged_copy(self):
        with tempfile.TemporaryDirectory() as d:
            c = P.Cfg(out=d, merged=os.path.join(d, "merged"), python="py", model_dir="base", gpus="", mock=True)
            trains = []
            for r in ("r1", "r2"):
                trains.append(os.path.join(d, r, "train"))
                os.makedirs(os.path.join(trains[-1], "adapter"))
                P.jdump({"step": 5}, os.path.join(trains[-1], "adapter", "state.json"))
            made = []

            def merge(_c, _name, cmd, env=None):
                made.append(cmd[cmd.index("--out") + 1])
                os.makedirs(c.merged)                        # fails if the last copy was not removed
                open(os.path.join(c.merged, "config.json"), "w").close()

            with mock.patch.object(P, "run_logged", side_effect=merge), \
                    mock.patch.object(P, "check_merged"), mock.patch.object(P, "stop_server"):
                P.ensure_merged(c, trains[0])
                P.ensure_merged(c, trains[0])                # already there
                P.ensure_merged(c, trains[1])                # another round's: replaced
                P.jdump({"step": 6}, os.path.join(trains[1], "adapter", "state.json"))
                P.ensure_merged(c, trains[1])                # retrained since: merged again
            self.assertEqual(made, [trains[0], trains[1], trains[1]])

    def test_round_clocks_run_on_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            c = P.Cfg(out=os.path.join(d, "r2"), home=d)
            os.makedirs(c.p("clock"))
            P.jdump({"hours": 10}, c.h("hours.json"))
            self.assertEqual(P.remaining(c, "sample", 20), 20)
            P.jdump({"hours": 13.5}, c.h("hours.json"))      # 3.5 h of running, however long the VM was gone
            self.assertEqual(P.remaining(c, "sample", 20), 16.5)

    def test_iterate_stops_without_gain_or_hours_and_pins_the_final_round(self):
        def go(d, scores, rounds, budget, spent=0.0):
            c = P.Cfg(**P.PROFILES["selftest"])
            c.out, c.rounds, c.budget, c.merged, c.model_dir = d, rounds, budget, os.path.join(d, "m"), ""
            c.scale = 1.0
            P.jdump({"hours": spent}, c.h("hours.json"))
            ran = []

            def run(x):
                ran.append(os.path.relpath(x.out, d))
                os.makedirs(x.out, exist_ok=True)
                for n, _ in x.stages or P.STAGES:
                    P.jdump({}, x.p(f"{n}.done"))

            with mock.patch.object(P, "run", side_effect=run), mock.patch.object(P, "claim"), \
                    mock.patch.object(P, "ledger"), mock.patch.object(P, "log"), \
                    mock.patch.object(P, "score", side_effect=lambda _c, x: scores[os.path.relpath(x, d)]):
                P.iterate(c)
            return ran, os.path.relpath(P.jload(c.p("final", "best.json"))["dir"], d)

        with tempfile.TemporaryDirectory() as d:          # r3 ties r2: no r4; r2 is the best
            self.assertEqual(go(d, {".": .5, "r2": .6, "r3": .6}, 4, 100), ([".", "r2", "r3", "final"], "r2"))
            self.assertEqual(go(d, {".": .9, "r2": .1, "r3": .1, "r4": 1}, 5, 100), (["r2", "r3", "final"], "r2"))
        with tempfile.TemporaryDirectory() as d:          # no hours left for a round and the final
            P.jdump({}, os.path.join(d, "gate.done"))
            self.assertEqual(go(d, {".": .5}, 3, 90.5, spent=90), (["final"], "."))


class TestWatch(unittest.TestCase):
    """A Colab cell shows only what the pipeline prints, so while workers run it
    must say where each one is, and still report which of them failed."""

    def test_heartbeat_and_failures(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "logs"))
            c = P.Cfg(out=d)
            prog = ("import sys,time\nprint('3/9 problems, 5/12 samples verified', flush=True)\n"
                    "time.sleep(0.7)\nsys.exit(%d)")
            procs = []
            for name, code in (("ok", 0), ("bad", 2)):
                with open(os.path.join(d, "logs", name + ".log"), "w") as fh:
                    procs.append((name, P.subprocess.Popen([sys.executable, "-c", prog % code],
                                                           stdout=fh, stderr=fh)))
            with mock.patch.object(P, "log") as log:
                failed = P.watch(c, procs, beat=0.2)
            said = [a[0][0] for a in log.call_args_list]
            self.assertEqual(failed, ["bad"])
            self.assertTrue(any(m.startswith("── still working") for m in said), said)
            self.assertIn("  ok: [███████░░░░░░░░░░░░░]  33% 3/9 problems · 5/12 proofs "
                          "verified (42%)", said)
            self.assertIn("bad failed (exit 2); the end of its log:", said)
            self.assertIn("  | 3/9 problems, 5/12 samples verified", said)
            with open(os.path.join(d, "progress.txt"), encoding="utf-8") as fh:
                self.assertIn("3/9 problems", fh.read())    # readable from Drive too
            self.assertNotIn("ok failed (exit 0); the end of its log:", said)
            self.assertEqual(P.last_line(os.path.join(d, "logs", "none.log")), "(no log yet)")

    def test_progress_reads_the_trainer_and_its_pace(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "train.log")
            with open(path, "w") as fh:
                fh.write("753/847 items fit 8192 tokens\nresuming after step 2\n")
            self.assertEqual(P.progress(path, {}), "resuming after step 2")
            with open(path, "a") as fh:
                fh.write("step 4/10 loss 0.4120 pair-acc 0.71 margin +0.210 lr 1.0e-05 1840 tok/s skipped 0\n")
            # first seen at step 2 an hour ago: 2 steps an hour, 6 to go
            got = P.progress(path, {path: (2, P.time.time() - 3600)})
            self.assertEqual(got, "[████████░░░░░░░░░░░░]  40% step 4/10 · loss 0.4120 · 1840 tok/s"
                                  " · ~3.0 h left at this pace")
            self.assertNotIn("left", P.progress(path, {}))            # one sighting: no pace yet
            with open(path, "a") as fh:
                fh.write("saved /x/adapter\n")
            self.assertIn("merging the adapter", P.progress(path, {}))


if __name__ == "__main__":
    unittest.main()
