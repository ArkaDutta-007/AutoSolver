#!/usr/bin/env python3
"""Sample the prover on mined problems and verify every proof in Lean. Runs on the GPU box.

Three modes share one verifier:

  plain   the Autosolver's exact best-of-n prompt (src/prover.sample_proofs).
          Successes are on-policy training targets, and the pass rate per
          problem is the evaluation -- base and tuned are scored by this file.
  hint    the same prompt plus a verified reference proof. The model writes its
          own reasoning and proof with the answer in view, and is trained on it
          WITHOUT the hint (STaR rationalization): the route by which a proof it
          cannot find yet gets into the weights.
  repair  the Autosolver's sequential repair turn: a failed attempt, Lean's
          error in the REPL's own format, "Fix it". A reply that fails with the
          SAME error is labelled `repeat` -- the looping-on-one-mistake failure,
          which becomes the rejected side of a preference pair.

A proof counts only if it compiles under PREAMBLE, uses no axiom beyond the
standard three, and proves the statement it was given: the declared theorem's
type must be definitionally the requested one (`type_of%` + `rfl`), so a
weakened restatement is caught by Lean rather than by a reviewer. Every sample
also records whether it ran out of tokens (`truncated`) or went round in circles
(`loop`) -- the two ways of not concluding -- so training can push away from
both and the evaluation can measure them.

TRAINING/INFERENCE CONTRACT: BASE_SYSTEM, both prompts and the repair turn are
byte-identical to src/prover.py, asserted by tests/test_oai_scripts.py.

    python3 oai_sample.py --pairs pairs.jsonl --split train --mode plain --n 4 --out plain.jsonl
    python3 oai_sample.py --pairs pairs.jsonl --mode repair --repair-from plain.jsonl --out repair.jsonl
    python3 oai_sample.py --report plain.jsonl
"""

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time

PREAMBLE = ("import Mathlib\n"
            "open scoped Matrix BigOperators Topology NNReal ENNReal Pointwise\n"
            "set_option maxHeartbeats 400000\n\n")
BASE_SYSTEM = (
    "You are a Lean 4 theorem prover using Mathlib. Output ONE fenced ```lean4 "
    "code block containing the theorem and its complete proof. `import Mathlib` "
    "is supplied automatically -- do not write import lines. NEVER use `sorry` or "
    "`admit`; they are rejected as incomplete proofs. Emit exactly one code block: "
    "your final, complete answer.")
HINT = ("\n\nA verified proof of this theorem is shown below. Work out why it is "
        "correct, then give your own complete proof in the required format -- you "
        "may follow it closely. Reason as if you found the proof yourself: never "
        "mention that a proof was shown to you.\n\n```lean4\n{reference}\n```")
LEAKED = re.compile(r"\b(reference|provided|given|shown|verified|supplied)\s+(lean\s+)?proof\b"
                    r"|\bthe hint\b|\bproof (that )?(was )?(shown|provided|given) (to me|above)",
                    re.I)
UNKNOWN = re.compile(r"[Uu]nknown constant `([^`]+)`")          # prover._UNKNOWN
AXIOMS_OK = {"propext", "Classical.choice", "Quot.sound"}
TEST_SHARE = 8          # percent of families held out, never trained on
REPL_SHIFT = 2          # the REPL drops PREAMBLE's import and set_option lines


def prompt(statement):
    """prover.sample_proofs: one user message."""
    return (f"{BASE_SYSTEM}\n\nProve this Lean 4 theorem. Keep the signature "
            f"VERBATIM and supply only the proof:\n\n{statement}")


def attempt_messages(statement, informal=""):
    """prover._attempt as prove_theorem calls it with a pinned signature:
    system + user, the informal claim first when there is one."""
    lead = f"{informal}\n\n" if informal else ""
    return [{"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": f"Prove in Lean 4 with Mathlib:\n\n{lead}"
             "Prove EXACTLY this Lean 4 theorem, keeping the signature verbatim and "
             f"supplying only the proof:\n\n{statement}"}]


TRAPS = (                                                       # prover._TRAPS
    (re.compile(r"AddGroup ℕ|AddGroup Nat|SubtractionMonoid ℕ|Neg ℕ|Neg Nat"),
     "ℕ is NOT an additive group: subtraction is TRUNCATED (`3 - 5 = 0`) and "
     "there is no negation, so lemmas needing `AddGroup`/`Neg` will never apply. "
     "Either cast to ℤ (`(a : ℤ) - b`, then `omega` or `push_cast`), or stay in ℕ "
     "and use `Nat.sub_add_cancel`, `Nat.add_sub_cancel'`, `Nat.sub_le`, or "
     "`omega`, which understands truncated subtraction natively."),
    (re.compile(r"nat\.abs|Int\.natAbs.*ℕ|unknown identifier `abs`|\babs\b.*ℕ"),
     "There is no absolute value on ℕ. For |a - b| over naturals use "
     "`Nat.dist a b` (which is `(a - b) + (b - a)` with truncated subtraction), "
     "or work in ℤ with `Int.natAbs`. `max a b - min a b` is also a correct ℕ "
     "spelling of |a - b| and `omega` handles it."),
    (re.compile(r"GCDMonoid ℕ|GCDMonoid Nat"),
     "Bare `gcd`/`lcm` resolve through `GCDMonoid`. On ℕ prefer the concrete "
     "`Nat.gcd` / `Nat.lcm` and their `Nat.*` lemmas."),
    (re.compile(r"unexpected token ','|expected ';' or line break|`from`|invalid 'have'"),
     "That is Lean 3 syntax. Lean 4 uses `fun x => e` (not `λ x, e`), `have h : T := e` "
     "(not `have h : T, from e`), newline- or `<;>`-separated tactics (not commas), "
     "and `·` for anonymous arguments."),
)


def repair_turn(errors):
    """prover._attempt's feedback after a rejected round (no retrieval suggestions:
    those need the Mac's Mathlib index)."""
    banned = list(dict.fromkeys(UNKNOWN.findall(errors)))
    ban = (f"\n\nThese constants DO NOT EXIST in Mathlib -- never use them again: "
           f"{', '.join(banned)}." if banned else "")
    hits = [msg for rx, msg in TRAPS if rx.search(errors)]
    trap = ("\n\nIMPORTANT:\n" + "\n".join(f"- {h}" for h in hits)) if hits else ""
    return {"role": "user", "content": f"The Lean compiler rejected that:\n\n"
            f"{errors[:1500]}{ban}{trap}\n\n"
            "Fix it and output the corrected complete Lean 4 code block."}


def _h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def family(p):
    return ".".join(p["module"].split(".")[:3])


def split(p):
    """By paper family (by problem, for competition sets), so near-duplicate
    lemmas of one paper never straddle train and test."""
    return "test" if _h(family(p)) % 100 < TEST_SHARE else "train"


def opens(p):
    return "".join(f"{o} in\n" for o in p["opens"])


def statement(p):
    """Half bare signature (the formalizer's contract), half ending `:= by`
    (what the other paths hand the prover) -- fixed per problem."""
    sig = f"{opens(p)}theorem {p['name']} {p['sig']}"
    return sig + (" := by" if _h(p["name"] + p["module"]) % 2 else "")


def reference(p):
    head = "".join(f"{o}\n" for o in p["opens"]) + (p.get("prefix") or "")
    return f"{head}\n\ntheorem {p['name']} {p['sig']} := {p['proof']}".strip()


def has_proof(p):
    return bool(p.get("proof")) and "sorry" not in p["proof"]


BLOCK = re.compile(r"```(?:lean4?)?\s*\n(.*?)```", re.S)
HAS_DECL = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.M)


def extract(text):
    """prover._extract_lean, with the full-Mathlib fallback of check_and_fix."""
    blocks = BLOCK.findall(text)
    cands = [b for b in blocks if HAS_DECL.search(b)] or blocks
    code = (cands[-1] if cands else "").strip()
    code = re.sub(r"^\s*import .*$\n?", "", code, flags=re.M).strip()
    return PREAMBLE + code if code else ""


def looped(text):
    """Went round in circles: the same long line over and over, or a tail that
    is one passage repeated. Judged on the prose: a verified proof may repeat a
    long `nlinarith [...]` line legitimately (3% of reference proofs would
    trip), so closed code blocks are left out -- one cut off mid-way is not."""
    text = BLOCK.sub("", text)
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= 25]
    if len(lines) >= 12:
        dup = sum(v - 1 for v in collections.Counter(lines).values() if v > 1)
        if dup / len(lines) > 0.3:
            return True
    tail = text[-4000:]
    return len(tail) > 1000 and any(tail.count(tail[i:i + 80]) >= 5
                                    for i in range(0, len(tail) - 80, 400))


def error_signature(err):
    """prover._attempt's stall test: the same first 300 chars of error twice."""
    return re.sub(r"tmp\w+\.lean", "", err)[:300]


class Lean:
    def __init__(self, env_dir, workers):
        run = lambda *c: subprocess.run(c, cwd=os.path.expanduser(env_dir), capture_output=True,
                                        text=True, check=True).stdout.strip()
        self.env = dict(os.environ, LEAN_PATH=run("lake", "env", "printenv", "LEAN_PATH"))
        self.lean = run("lake", "env", "which", "lean")
        self.slots = threading.Semaphore(workers)

    def check(self, code, p, timeout=400):
        """(ok, why, errors) for one candidate proof of problem p. `errors` is in
        the Autosolver REPL's format, so a repair turn shows the model exactly
        what it would see there."""
        name = p["name"]
        tail = (f"\n\n{opens(p)}axiom oai_target_stmt {p['sig']}\n"
                f"example : type_of% @{name} = type_of% @oai_target_stmt := rfl\n"
                f"#print axioms {name}\n")
        mark = code.count("\n") + 3          # first line of the appended check
        with self.slots, tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False,
                                                     encoding="utf-8") as fh:
            fh.write(code + tail)
            fh.flush()
            try:
                r = subprocess.run([self.lean, "--json", fh.name], capture_output=True,
                                   text=True, timeout=timeout, env=self.env)
            except subprocess.TimeoutExpired:
                return False, "timeout", f"timed out after {timeout}s"
            finally:
                os.unlink(fh.name)
        msgs = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
        mine = [m for m in msgs if m["pos"]["line"] < mark and m["severity"] != "information"]
        errors = "\n".join(f"repl:{m['pos']['line'] - REPL_SHIFT}:{m['pos']['column']}: "
                           f"{m['severity']}: {m['data']}" for m in mine)
        errs = [m for m in msgs if m["severity"] == "error"]
        if any(m["pos"]["line"] < mark for m in errs):
            return False, "compile", errors
        if errs:
            return False, "statement_changed", "the proof changed the theorem statement"
        ax = [m["data"] for m in msgs if "depend" in m["data"] and m["pos"]["line"] >= mark]
        if len(ax) != 1:
            return False, "no_decl", errors
        used = set(re.findall(r"[\w.]+", ax[0].partition("axioms:")[2]))
        return (True, "ok", "") if used <= AXIOMS_OK else (False, "axioms", ax[0])


def fit(base, model, messages, max_tokens, model_len):
    """max_tokens that fits the server's window after this prompt (prover._fit);
    without vLLM's /tokenize, a cautious ~2 chars/token estimate."""
    import requests
    try:
        r = requests.post(re.sub(r"/v1/?$", "", base) + "/tokenize",
                          json={"model": model, "messages": messages}, timeout=60)
        r.raise_for_status()
        room = r.json()["max_model_len"] - r.json()["count"]
    except Exception:
        room = model_len - sum(len(m["content"]) for m in messages) // 2 - 64
    return max(256, min(max_tokens, room))


def sample(base, model, messages, n, max_tokens, timeout=3600):
    """[(text, finish_reason, tokens)] for n samples of one conversation."""
    import requests
    r = requests.post(f"{base}/chat/completions", timeout=timeout, json={
        "model": model, "messages": messages,
        "n": n, "temperature": 1.0, "top_p": 0.95, "max_tokens": max_tokens})
    r.raise_for_status()
    body = r.json()
    each = (body.get("usage") or {}).get("completion_tokens", 0) / max(1, len(body["choices"]))
    out = []
    for ch in body["choices"]:
        m = ch.get("message", {})
        out.append(((m.get("content") or "").strip() or (m.get("reasoning") or "").strip(),
                    ch.get("finish_reason"), round(each)))
    return out


def records(path):
    """A sample file's rows. A kill mid-write leaves a partial last line: it is
    skipped here, and run() cuts it off before appending."""
    out = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for x in fh:
                try:
                    out.append(json.loads(x))
                except ValueError:
                    pass
    return out


def load(path):
    pairs = [json.loads(x) for x in open(path, encoding="utf-8")]
    for i, p in enumerate(pairs):
        p.setdefault("pid", f"{p['module']}:{p['name']}:{i}")
    return pairs


def jobs(a, pairs):
    """(problem, context-to-train-on, context-actually-sent, previous error) per job."""
    if a.mode == "repair":
        by = {p["pid"]: p for p in pairs}
        seen, out = set(), []
        for path in a.repair_from:
            for r in records(path):
                if r["why"] == "compile" and r.get("err") and r["pid"] in by and r["pid"] not in seen:
                    seen.add(r["pid"])
                    p = by[r["pid"]]
                    ctx = attempt_messages(statement(p), p.get("doc", "")) + [
                        {"role": "assistant", "content": r["text"]}, repair_turn(r["err"])]
                    out.append((p, ctx, ctx, r["err"]))
        return out
    out = []
    for p in pairs:
        if a.mode == "hint" and not has_proof(p):
            continue
        ctx = [{"role": "user", "content": prompt(statement(p))}]
        sent = ctx if a.mode != "hint" else [
            {"role": "user", "content": ctx[0]["content"] + HINT.format(reference=reference(p))}]
        out.append((p, ctx, sent, ""))
    return out


def run(a):
    pairs = load(a.pairs)
    todo = [p for p in pairs if a.split in ("all", split(p))]
    if a.per_family:            # spread a fixed GPU budget over every paper
        todo.sort(key=lambda p: _h(p["pid"]))
        fams, kept = {}, []
        for p in todo:
            fams[family(p)] = fams.get(family(p), 0) + 1
            if fams[family(p)] <= a.per_family:
                kept.append(p)
        todo = kept
    if a.only:
        keep = {r["pid"] for r in records(a.only)}
        todo = [p for p in todo if p["pid"] in keep]
    if os.path.exists(a.out):
        with open(a.out, "rb+") as fh:          # a killed run's partial last line
            data = fh.read()
            if data and not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)
    # a problem short of n samples gets the rest: a resumed run, or an n=4
    # evaluation copied in and topped up to n=16
    have = collections.Counter(r["pid"] for r in records(a.out))
    skip = {r["pid"] for f in a.skip for r in records(f)}     # earlier rounds' problems
    work = [j for j in jobs(a, todo) if have[j[0]["pid"]] < a.n
            and j[0]["pid"] not in skip][: a.limit or None]
    lean = Lean(a.env, a.lean_workers)
    lock = threading.Lock()
    deadline = time.time() + a.hours * 3600 if a.hours else float("inf")
    tally = {"problems": 0, "samples": 0, "ok": 0}

    def verdict(o, p, prev):
        text, finish, _ = o
        if finish == "length":
            return False, "truncated", ""
        code = extract(text)
        if not code:
            return False, "no_code", ""
        if a.mode == "hint" and LEAKED.search(text):
            return False, "leaked_hint", ""
        ok, why, err = lean.check(code, p)
        if prev and not ok and error_signature(err) == error_signature(prev):
            why = "repeat"
        return ok, why, err

    def one(job):
        p, ctx, sent, prev = job
        if time.time() > deadline:          # time-boxed: in-flight problems finish, no new ones
            return
        try:
            outs = sample(a.base, a.model, sent, a.n - have[p["pid"]],
                          fit(a.base, a.model, sent, a.max_tokens, a.model_len))
        except Exception as exc:
            print(f"sample failed for {p['pid']}: {exc}", flush=True)
            return
        with cf.ThreadPoolExecutor(max(1, len(outs))) as pool:
            verdicts = list(pool.map(lambda o: verdict(o, p, prev), outs))
        rows = [{"pid": p["pid"], "mode": a.mode, "model": a.model, "k": have[p["pid"]] + k,
                 "ntok": nt, "messages": ctx, "text": t, "finish": fin, "ok": ok, "why": why,
                 "err": err[:1500], "loop": looped(t)}
                for k, ((t, fin, nt), (ok, why, err)) in enumerate(zip(outs, verdicts))]
        with lock, open(a.out, "a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            tally["problems"] += 1
            tally["samples"] += len(rows)
            tally["ok"] += sum(r["ok"] for r in rows)
            print(f"{tally['problems']}/{len(work)} problems, {tally['ok']}/{tally['samples']} "
                  f"samples verified", flush=True)

    with cf.ThreadPoolExecutor(a.inflight) as pool:
        list(pool.map(one, work))


def summarize(path):
    """Pass rates, the not-concluding rates, and which problems were solved."""
    by = {}
    for r in records(path):
        by.setdefault(r["pid"], []).append(r)
    rows = [r for rs in by.values() for r in rs]
    n, m = max(len(by), 1), max(len(rows), 1)
    wins = [r.get("ntok", 0) for r in rows if r["ok"]]
    return {"problems": len(by), "samples": len(rows),
            "pass1": sum(sum(r["ok"] for r in rs) / len(rs) for rs in by.values()) / n,
            "passn": sum(any(r["ok"] for r in rs) for rs in by.values()) / n,
            "truncated": sum(r["why"] == "truncated" for r in rows) / m,
            "looped": sum(bool(r.get("loop")) for r in rows) / m,
            "repeat": sum(r["why"] == "repeat" for r in rows) / m,
            "mean_tokens": sum(r.get("ntok", 0) for r in rows) / m,
            "win_tokens": sum(wins) / max(len(wins), 1),
            "outcomes": dict(collections.Counter(r["why"] for r in rows)),
            "solved": sorted(pid for pid, rs in by.items() if any(r["ok"] for r in rs))}


def report(path):
    s = summarize(path)
    print(f"{path}: {s['problems']} problems, pass@1 {s['pass1']:.3f}, pass@n {s['passn']:.3f}, "
          f"truncated {s['truncated']:.1%}, looped {s['looped']:.1%}, repeat {s['repeat']:.1%}, "
          f"~{s['mean_tokens']:.0f} tok/sample ({s['win_tokens']:.0f} when solved), "
          f"outcomes {s['outcomes']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--pairs")
    ap.add_argument("--only", help="jsonl of pids to restrict to")
    ap.add_argument("--skip", nargs="*", default=[], help="jsonl files whose pids are not sampled")
    ap.add_argument("--split", default="train", choices=("train", "test", "all"))
    ap.add_argument("--mode", default="plain", choices=("plain", "hint", "repair"))
    ap.add_argument("--repair-from", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-family", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0, help="stop starting new problems after this")
    ap.add_argument("--out")
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-32B")
    ap.add_argument("--max-tokens", type=int, default=16000, help="config.PROVER_MAX_TOKENS")
    ap.add_argument("--model-len", type=int, default=16384)
    ap.add_argument("--inflight", type=int, default=12)
    ap.add_argument("--lean-workers", type=int, default=12)
    ap.add_argument("--env", default="~/oaimath/env")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
