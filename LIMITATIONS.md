# Known limitations

Ranked by expected value of fixing. Every entry is backed by something observed
in this system, not speculation.

## Status

| # | Limitation | Status |
|---|---|---|
| 0 | **Context silently truncated to ~4k** — dominated everything below | **FIXED** — `src/preflight.py` measures it and refuses to start |
| 1 | No Mathlib lemma search | **SUPERSEDED** — `src/mathlib_kb.py`, 265,774 decls, DENSE multi-embedder retrieval; keyword search is now the fallback |
| 2 | `import Mathlib` dominates wall-clock | **NO LONGER BINDING** — measured 29s to compile 32 candidates vs 718s to generate them; generation is the constraint |
| 3 | Truncation misread as math failure | **FIXED** — `finish_reason` checked everywhere, and generation is now uncapped with a derived timeout |
| 4 | Reviewer errs both ways | **PARTIAL** — false positives addressed; reviewer is still the orchestrator judging itself |
| 5 | 8B prover ceiling | **FIXED** — Goedel-V2-**32B** bf16 on the A6000 box, **pass@32** verified end to end |
| 6 | No accumulating lemma library | **FIXED, then CURATED** — dedup/usage/wake-sleep; the raw version had 8 entries encoding 2 facts |
| 13 | **Fixed 900s timeout capped every slow model** | **FIXED 2026-08-14** — timeouts derived from budget ÷ floor throughput; see §13 |
| 14 | **Greedy retreat to the easy claim** (the local minimum) | **FIXED 2026-08-14** — `src/archive.py` refuses a filled behaviour cell before spending a proof budget |
| 15 | **Fixed strategy ladder** spent minutes-per-attempt in hard-coded order | **FIXED 2026-08-14** — `src/bandit.py`, UCB1 per domain |
| 16 | **Nothing generated candidate claims** | **FIXED 2026-08-14** — `src/evolve.py` island-model GA, exposed as the `conjecture` tool |
| 7 | No proof-only mode | **FIXED** — but "no reviewer needed" was wrong; see §9 |
| 8 | Smaller items | PARTIAL |
| 9 | **Pinning a signature silently bypassed the faithfulness review** | **FIXED 2026-08-13** |
| 10 | `exact?` successes are Mathlib lookups, not new mathematics | OPEN — inherent; affects how results are counted |
| 11 | Code changes land mid-run and don't take effect | **FIXED** — run header and JSONL stamp the git rev |
| 12 | One problem forked into two journals | **FIXED** — identity is the statement hash, not the model's title |

---

## 0. The context window was ~4k, not 1M  **[FIXED 2026-08-14]**

**This invalidated most of the measurements below.** Every entry ranked by
"expected value of fixing" was ranked against a system that could not hold a
thought, so the prover was never the binding constraint — it was never called.

**Evidence.** Across all 11 runs in `runs/`, `prove_theorem` fires **zero times**.
Six sessions on IMO 2026 P1 produced six `open_problem` + `read_journal` pairs and
nothing else; `exhausted` stayed `[]`. In `run-20260813-171447.jsonl`:

- **step 3** — 113,191 characters of *correct* P1 reasoning. It found the
  invariant: a move sends the exponent pair `(a,b) -> (min(a,b), |a-b|)`, and
  `gcd(min(a,b), |a-b|) = gcd(a,b)`, so the gcd of the exponent multiset is
  conserved. Then `finish_reason: length`, no tool call.
- **step 4** — verbatim: *"I don't have any context from a previous
  conversation."* The reasoning was gone.
- **step 7** — to satisfy the "continue from where you stopped" nudge it
  confabulated, opened a problem titled *"Continuing interrupted work"*, then
  invented *"Fundamental Theorem of Arithmetic"*. That is where
  `problems/002-*` and `problems/003-*` came from. Neither was ever requested.

**Cause.** Ollama's `/v1/chat/completions` has no `num_ctx` parameter; the server
applies `OLLAMA_CONTEXT_LENGTH` (default 4096) regardless of the model's declared
1M. Nothing in the response indicates truncation. Every budget was calibrated
against a window that did not exist: `ORCH_MAX_TOKENS=24000` (6× the real window),
`TOOL_RESULT_MAX_CHARS=32_000` (one tool result overflows it), `COMPACT_AT_CHARS
=400_000` (could never fire), `read_journal` defaulting to 200,000 chars — which
the model duly requested.

**Fix.** `src/preflight.py` needle-probes the real window at startup and exits
with the exact remediation rather than running degraded. All budgets are now
fractions of the measured window (`config.budget_chars`). Durable state
(statement / established / ruled-out / journal tail) is re-injected every turn,
so a truncated context now costs recent chatter instead of the problem.

---

## 1. No Mathlib lemma search — the single biggest win  **[FIXED]**

**Evidence.** Hallucinated lemma names are the most common failure mode observed.
The √2 test burned 6 rounds on `Real.sqrt_eq_iff_sq_eq`, which does not exist. I
independently hit the same wall with `Nat.gcd_dvd_lcm`, also nonexistent.

Meanwhile a plain grep over Mathlib's 8,311 source files resolves this instantly:

    gcd_dvd_lcm          -> DOES NOT EXIST
    dvd_lcm_left         -> EXISTS
    gcd_dvd_left         -> EXISTS
    irrational_sqrt_two  -> EXISTS

That last one is the punchline: **the lemma that would have closed the √2 test
was in Mathlib all along.** The prover simply had no way to look it up.

**Fix.** A `search_mathlib(pattern)` tool:

    grep -rhoE "^(theorem|lemma) [a-zA-Z_'.]*<pat>[a-zA-Z_'.]*" \
      lean/.lake/packages/mathlib/Mathlib --include="*.lean" | sort -u

Expose it to the prover, and add a strategy that requires searching before
inventing a name. Cheap to build, and it targets the dominant failure.

Better still: run Lean's own `exact?` / `apply?` tactics, which search Mathlib
natively. Noted during smoke testing that `exact?` *solved* the `2 ∣ n(n+1)` goal
outright.

---

## 2. `import Mathlib` dominates wall-clock — **now the binding constraint**

**Evidence, original.** 25s with `import Mathlib` vs 1.2s with a narrow import —
~20x. The prover itself answers in seconds, so compilation, not inference, is the
bottleneck.

**Evidence, 2026-08-13 — it is worse than 25s under load.** In
`run-20260813-204809.jsonl`, three `prove_theorem` calls returned
`lean_statement_rejected: True` with `attempts: 0` — nothing was proved, nothing
was generated — and still cost **291.7s, 293.4s and 320.7s**. That is 905 of the
run's ~960 seconds spent typechecking three signatures that were rejected.
`normalize_signature` produced **one** candidate for those, so ~291s is a *single*
`check_lean` call.

**Cause.** `check_lean` tries `repl.POOL` first and falls back to a cold
`lake env lean`. The REPL was not running, because it declines when there is no
memory for it. Measured during that run: **1.5 GB free, 7.7 GB compressed, 27.2 GB
wired**, with the orchestrator alone holding 25.6 GB. So every typecheck paid a
full Mathlib import while the machine was swapping.

**Fix.** The REPL (`src/repl.py`, `scripts/setup_repl.sh`) is written and correct;
it cannot run alongside a resident 25 GB orchestrator on a 48 GB machine. The real
options are (a) evict the orchestrator while Lean works, (b) a smaller
orchestrator, or (c) move inference to the lab server and keep Lean local. This is
a memory-budget problem, not a code problem.

**Partial mitigation already in place.** `_syntax_reject` (see §9) refuses
hopeless signatures without invoking Lean at all — free, and it catches the
`???`-placeholder and comment-only cases that cost 600 of those 905 seconds. It
does nothing for a signature that looks plausible and fails to elaborate.

---

## 3. Truncated generations look like math failures

**Evidence.** `prover._call` never inspects `finish_reason`. A proof cut off at
`PROVER_MAX_TOKENS` (4000) yields a partial code block that fails compilation with
a syntax error — indistinguishable in the logs from a genuine mathematical failure.
Long proofs are exactly where this bites.

**Fix.** Check `finish_reason == "length"`, and either retry that attempt with a
larger budget or mark it `truncated` rather than counting it as a failed proof.

---

## 4. The faithfulness reviewer errs in both directions

**Evidence.** It correctly caught a vacuous formalization (the "infinitely many
primes" statement satisfied by the constant function 2). It then **incorrectly**
rejected the P1 core statement, worrying about division by zero. Verified by hand
that the statement holds over all of `ℕ`:

    theorem p1_core (m n : ℕ) : Nat.gcd m n * (Nat.lcm m n / Nat.gcd m n) = Nat.lcm m n
    -- ok = True

Quantifying over all naturals is *stronger* than the "positive integers" claim and
implies it — a perfectly good formalization that the reviewer talked itself out of.

**Fix applied.** The reviewer is told explicitly that a strictly stronger
statement is acceptable. A `MISMATCH` continues to the next attempt rather than
ending the search, so a false rejection costs time, not correctness.

**Still open, and now more exposed.** `REVIEWER_MODEL` defaults to
`ORCHESTRATOR_MODEL` (`config.py:36`), so the model that derived the Lean
signature is the model that judges whether that signature expresses the claim.
That is self-review on the one gate standing between a compiling proof and an
overstated result. Since §9 made the review run on *every* success path rather
than being skipped for pinned signatures, this judgement is now load-bearing on
every result the system produces.

`nemotron-cascade-2` is the obvious second opinion — it cannot write Lean
(measured: 45,283 characters of reasoning, no proof) but back-translation is a
natural-language judgement, not a Lean task. The cost is memory: a 24 GB reviewer
against a 40 GB cap already holding 36 GB means eviction and reload per review.
Reviews only fire after a successful compile, so this may be affordable — it has
not been measured.

---

## 5. The 8B prover is the capability ceiling

**Evidence.** Failed √2 (a Mathlib one-liner). Stalled on the P6 gcd core.
Never closed a full IMO problem. For scale: AxiomProver's formal IMO 2026
solutions run 457–4,229 lines and took 24–869 minutes each; this prover emits
proofs of a few dozen lines.

**Superseded by §0 — this ranking was wrong.** The prover was never the binding
constraint because it was never called: `prove_theorem` fired zero times across
all 11 runs made before the context fix. The first run that reached it
(`run-20260813-204809`) spent 905 of ~960 seconds inside Lean typechecking and
gave the 8B prover four attempts, three of which never started. **The ceiling
remains unmeasured.** Do not re-rank this above §2 until a run exists where the
prover actually got the wall-clock.

**Fix when it is measurable.** `PROVER_FALLBACK_BASE` / `PROVER_FALLBACK_MODEL`
are already wired for Goedel-Prover-V2-32B on the lab server (88.1% MiniF2F
pass@32) — they just need configuring. Two targets turn `prover.py` into a race
via `ThreadPoolExecutor`, which is the point: different models occupy different
basins, and cross-model diversity is what rescued √2.

---

## 6. No accumulating lemma library

**Evidence.** Every proof is written to a temp file and deleted immediately. A
lemma proved for one problem is unavailable to the next, so the system cannot
build upward — which is precisely what long-horizon research needs.

**Fixed.** `_record` calls `library.add()` on every verified proof when
`GROW_LIBRARY=1` (the default), appending it to `Proofs/Verified.lean` so later
proofs can build on it; the theorem name comes back on the result as `library`.

**Caveat worth watching.** Combined with §10, what accumulates is currently
dominated by theorems `exact?` found already in Mathlib — importable, but not new.
The library becomes interesting only once the prover is the thing filling it.

---

## 7. No proof-only mode

**Evidence.** `prove_theorem` takes an *informal* claim, so the prover chooses
both the formalization and the proof. That freedom is the entire source of the
mis-formalization failure class.

**Fix.** Accept an optional explicit Lean statement. When you can state the
theorem yourself, the prover fills in only the proof and mis-formalization becomes
impossible — no reviewer needed. Keep the informal path for the agent's own
conjectures.

---

## 9. Pinning a signature silently bypassed the faithfulness review  **[FIXED 2026-08-13]**

§7 claimed proof-only mode makes mis-formalization "impossible — no reviewer
needed". That is false. Pinning does not remove the risk; it **moves** it from the
prover to whoever wrote the signature, where nothing was checking it.

**Evidence.** In `run-20260813-204809.jsonl` step 10, the claim was

> *"gcd(m,n) divides both m and n, **and lcm(m,n) is a multiple of both**"*

pinned to a signature covering only the gcd half. It recorded `proved: true` with
`formalization: {faithful: true, verdict: "statement pinned"}` — the reviewer was
never called. The result overstates what was proved.

**The path that fired was the tactic probe, not the prover.** Both of that run's
successes came from `exact?`, and `tactics.probe` returned the same
`"statement pinned"` stub. `exact?` succeeding says the *proof* is sound and
nothing whatsoever about whether the signature expresses the claim.

**Fix.** Back-translation now runs on every success path — prover *and* tactic
probe. When the signature is pinned and review returns MISMATCH the search stops
instead of retrying: every further attempt would prove the identical mismatched
statement, so the caller has to narrow the claim or repin.

**Also fixed here.** `_syntax_reject` rejects a signature by reading it, before
any Lean invocation: no `theorem`/`lemma`/`example`, comment-only body, no `:`,
a `???`/`TODO`/`...` placeholder, or an `import` line that belongs to the harness
preamble. Two of the three 5-minute rejections in that run were of this kind —
`m.val.gcd n p ??? -- need correct`, and a "signature" that was only a comment.

---

## 10. `exact?` successes are Mathlib lookups, not new mathematics

**Evidence.** Both proofs in `run-20260813-204809.jsonl` were closed by `exact?`
(`solved_by: tactic:exact?`), which by construction means the theorem was already
in Mathlib. That is a genuine speed win — §1's whole point — but it is retrieval,
and a run summary that reports "lean-verified: 2" does not distinguish it from
proving something new.

**Mitigation.** `solved_by` is recorded on every result and written to the journal,
so the distinction is *recoverable*. It is not surfaced in the run footer. When
counting results, treat `tactic:exact?` and `cache` separately from a real prover
attempt.

---

## 11. Code changes landed mid-run and did not take effect  **[FIXED 2026-08-13]**

Python binds modules at import, so a fix applied while a run is in flight reaches
nothing. This happened three times in one day: a `fetch_url` gate, a journal
format change, and the `search_mathlib` rewrite all landed against a running
process that kept the old code. Transcripts were then read as evidence about a
system that did not produce them — `search_mathlib` returning 0 four times in
`run-20260813-204809.jsonl` is the clearest case; the rewrite that fixed it was
already committed.

**Fix.** `agent._code_version()` stamps `git rev-parse --short HEAD` (plus
`+dirty`) into the run header and the JSONL `start` record. A transcript now says
which code produced it.

---

## 12. One problem forked into two journals  **[FIXED 2026-08-13]**

`find_existing` matched on the model's *title*. Mid-run the agent renamed
`IMO26_P1` to `IMO26_P1: Game with gcd and lcm/gcd`, the slug stopped matching,
and it opened a second journal for a problem it was already working — splitting
the plan from the exploration that produced it.

**Fix.** Identity is `problem_key(statement)`, a hash of the normalized statement,
with the title kept only as a fallback. A resumed problem also keeps its
*original* statement: the model retyped P1 three different ways across six
sessions (`2026` vs `{2026}`, LaTeX vs plain), and letting each session overwrite
the anchor drifts it away from the problem being solved.

---

## 8. Smaller items

- **Attempts run serially.** GPU-bound inference limits parallel gain, but Lean
  compilation could overlap with the next generation.
- **Fixed 120s Lean timeout.** Heavy `nlinarith`/`polyrith` goals may need more;
  a timeout currently reads as a failure.
- **`tests/` shadows anaconda's `tests` package.** Direct execution works; `import
  tests.x` does not. Rename to `evals/` if it becomes annoying.
- **Notebook and JSONL are parallel records** with no cross-links. Adding the run
  ID to the notebook header would let you jump between them.

---

## 13. A fixed 900s timeout capped every slow model  **[FIXED 2026-08-14]**

Every generation call used a hardcoded `timeout=900`. Nothing announced it, and
it only ever bound on models slower than `budget/900` tok/s — so the handicap
was invisible in any comparison against a fast one.

Measured: Qwen3.8-27B decodes at **9.34 tok/s**, so 900s buys ~8,400 tokens
against a configured budget of 12,000. **It could never reach its own budget.**
Worse, the truncation-retry was dead code on exactly the models that needed it:
a timeout *raises*, so `body` is `{}`, so `done_reason` is never `"length"`,
so the budget-doubling branch never ran. The mechanism only fired for models
fast enough not to need it.

Two of Qwen's benchmark failures (940.9s and 1,455.6s) exceeded 900s and were
therefore timeouts recorded as capability failures. Its 3/7 was measuring the
harness. After the fix the same cases scored **12/15**.

Fixes: `prover._gen_timeout()` derives the wait from budget ÷ a floor
throughput, clamped to `ORCH_TIMEOUT_CAP` and to the problem deadline; a
timeout now records its own `_LAST_SIG_ERROR["stage"]` instead of masquerading
as `generation_empty`; and on timeout the retry caps *thinking* rather than
raising the budget, because a bigger budget makes a wall-clock overrun worse.

**The general lesson:** a constant that only binds on slow inputs is invisible
in benchmarks made of fast ones. `evals/formalization_v2.py` compared three
models for hours and the result was decided by a number none of them was
measured against.

## 14. The local minimum, and why prompting could not fix it  **[FIXED 2026-08-14]**

26 prove calls, 16,968 seconds, 4 successes — all four the same divisibility
fact. The loop:

    hard claim fails expensively -> retreat to the claim that formalises
    cleanly -> it verifies -> written to `established`, which reads as
    progress -> repeat

After five sessions `established` held five copies of one fact while the actual
invariant sat in `exhausted`, falsely ruled out. The lemma library reproduced
the same pathology independently: **8 entries encoding 2 distinct facts.**

Prompt-level discouragement cannot fix this, because the incentive is real —
the easy claim genuinely does verify. `src/archive.py` makes it structurally
impossible instead: claims are binned by `(mechanism, objects, role)`, and a
claim landing in a cell already filled by a verified result is refused *before*
a proof budget is spent.

Two bugs found while testing it, both of which would have silently defeated it:

* `library.describe` reads Lean tokens, so the English claim "2 divides
  n*(n+1)" and the symbolic `2 ∣ n*(n+1)` landed in **different cells** — the
  exact split the archive exists to prevent. Prose now maps onto the same
  vocabulary.
* Fully symbolic claims have no prose content words at all (every token a
  stopword or single letter), so role inference returned `unknown` and split
  the cell again.

## 15. Strategy selection was a hard-coded ladder  **[FIXED 2026-08-14]**

`STRATEGIES[i % 6]` spent the most expensive resource in the system — attempts
costing minutes each — in the same order every time, regardless of what had
been working. It is a textbook bandit: several arms, a binary verifier reward,
and a budget far smaller than the pulls needed for certainty.

`src/bandit.py` is UCB1 over named strategies, with stats kept **per domain**:
an arm that wins on divisibility can be useless on compactness, and a single
global ranking would be wrong half the time. Persisted across sessions,
because a bandit that forgets on restart never leaves exploration.

## 16. Nothing proposed candidate claims  **[FIXED 2026-08-14]**

The agent chose what to attempt by whatever it could formalise, which is how
§14 happened. `src/evolve.py` is an island-model GA over conjectures: fitness is
verified **and** novel (per the §14 archive) **and** cited by later proofs —
all three, because verified-only rewards triviality, novel-only rewards
nonsense, and cited-only cannot be measured yet. Islands evolve separately and
the weakest is periodically reseeded from the best survivors elsewhere, which
is the anti-stagnation move a single population cannot make.

Exposed to the agent as the `conjecture` tool. It was fully implemented and
**unreachable** for an hour — no tool referenced it — which is why
`tests/test_integration.py` exists.

## 17. Subsystems that pass their own tests while being unplugged  **[ONGOING RISK]**

Not a bug so much as the failure *shape* this codebase keeps producing. Every
control loop above passed its unit tests while doing nothing at all at least
once: a bandit nobody called, an archive nobody recorded to, a library nobody
searched, a GA no tool exposed.

The failure mode is **silence**, not an exception, so ordinary testing does not
catch it. `tests/test_integration.py` asserts the call graph itself. It has
already caught: two hardcoded strategy names that stopped existing after a
rename; a test class defined after the `__main__` guard (discovery ran it,
direct execution silently did not); and a test that took the suite from 3s to
274s by calling a model, which is how a suite stops being run at all.
