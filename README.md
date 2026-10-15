# AutoSolver

AutoSolver is a math research agent that counts a result as proved only when
the Lean 4 compiler accepts the proof. A local language model reads the
problem, tests small cases in Python, and writes each claim as a Lean
statement. A theorem prover ([Goedel-Prover-V2](https://huggingface.co/Goedel-LM))
then searches for a proof, and Lean checks every candidate. A reviewer
translates each accepted statement back into English and compares it with the
original claim, because a proof can compile while proving something easier
than what was asked.

The agent runs unattended over a batch of problems. It keeps a resumable
notebook for each problem and adds every verified lemma to a Lean library that
later proofs can import. Everything runs locally by default. A GPU server, a
cloud planner and web search are optional add-ons.

AutoSolver includes a fine-tuned Goedel-Prover-V2-32B. We trained it on its
own verified proofs of a larger set of competition problems and research
lemmas from past open problems, with targets chosen to make it finish its
proofs and stop repeating the error Lean just reported. After the first
training round its pass@1 on held-out problems is 8.2 points above the base
model's, at the same decoding speed. [TRAINING.md](TRAINING.md) gives the
data, method and results, and the pipeline to reproduce it. The weights will
be released on Hugging Face after the final training round.

For the local model we recommend Qwen3.8-27B at Q8. It was the most accurate
formalizer we measured, and at about 30 GB it fits next to the local prover on
a 64 GB machine.

## How it works

For each problem:

1. **Plan.** An optional cloud model writes a plan: restated problem, approach,
   ordered steps and the obligations that must be discharged. Without it, the
   local orchestrator decomposes the problem itself.
2. **Explore.** The orchestrator (a 27B local model) works through the plan
   with tools: a Python sandbox for small cases, Mathlib search, its notebook,
   and, only in `--open` mode, the web.
3. **Formalize.** Each claim becomes a Lean signature. The signature is
   type-checked, and retrieval over 265k Mathlib declarations supplies real
   lemma names before the model can invent them.
4. **Try to refute.** A numeric counterexample search runs before any proof
   search. A backward pass derives what would suffice for the goal and tries to
   refute those pieces too.
5. **Prove.** Cheap tactics (`exact?`, `aesop`, `omega`, ...) go first. Then the
   prover (Goedel-Prover-V2, base or fine-tuned) runs: best-of-32 in one
   request when a vLLM server is available,
   followed by independent attempts with different strategies, each with
   several rounds of repair from Lean's error messages.
6. **Check.** Lean compiles the proof. Proofs using `sorry`, empty files,
   `native_decide`, or axioms beyond `propext`, `Classical.choice` and
   `Quot.sound` are rejected. The reviewer then checks that the statement still
   means the original claim.
7. **Keep.** Verified lemmas go into `lean/Proofs/Verified.lean`. Between
   problems the library is deduplicated and generalized, and its size is capped
   so it does not crowd the context window.

Four feedback loops decide where effort goes. Each was added after a measured
failure:

| Loop | Module | The failure it answers |
|---|---|---|
| What kind of claim to try | `archive.py` (MAP-Elites) | 26 prove calls, 4 successes, all the same fact |
| Which strategy to spend on | `bandit.py` (UCB1) | a fixed 6-step ladder spending minutes per attempt regardless of what worked |
| What to propose | `evolve.py` (island GA) | nothing generated candidates, so the agent picked whatever formalized easily |
| What to keep | `library.py` | 8 stored lemmas encoding 2 distinct facts |

All four describe results with the same `(mechanism, objects)` descriptor, and
`tests/test_integration.py` checks that they are actually wired together. Each
of them once passed its own unit tests while doing nothing in a real run.

## Requirements

- macOS on Apple silicon, or Linux. Python 3.10 or newer.
- [Ollama](https://ollama.com) for the local models.
- Lean 4 through [elan](https://github.com/leanprover/elan). The Lean project in
  `lean/` pins Lean and Mathlib v4.33.0.
- About 64 GB of RAM (unified memory on a Mac) for the recommended local
  stack: Qwen3.8-27B at Q8 as the orchestrator (~30 GB), the 8B prover
  (~9 GB) and one Lean REPL (~6 GB). A smaller orchestrator works on less
  memory, at a cost in formalization quality.
- About 60 GB of disk: models (~40 GB), Mathlib build files (~7 GB) and the
  retrieval index (~1 GB).

Optional:

- A GPU server with two 48 GB GPUs for Goedel-Prover-V2-32B (bf16, vLLM).
- A [DashScope](https://www.alibabacloud.com/en/product/modelstudio) key for
  the cloud planner, or any OpenAI-compatible endpoint.
- A [Serper](https://serper.dev) key for web search in `--open` mode.

## Setup

```bash
git clone https://github.com/ArkaDutta-007/AutoSolver.git
cd AutoSolver
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && chmod 600 .env     # then edit; every key is optional

# Lean and Mathlib. `cache get` downloads prebuilt Mathlib instead of compiling it.
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
(cd lean && lake exe cache get && lake build)
./scripts/setup_repl.sh      # optional; cuts a Lean check from ~25 s to ~1 s

# Models. Start Ollama with a 32k window: its OpenAI-compatible endpoint
# otherwise uses 4096 tokens and truncates prompts without saying so.
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 ollama serve &
ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0
ollama pull hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q8_0
ollama pull nomic-embed-text; ollama pull bge-m3; ollama pull mxbai-embed-large

# Mathlib retrieval index, about 15 minutes per embedder
python3 -m src.mathlib_kb --build

# Check the install
python3 -m unittest discover -s tests
python3 run.py --questions questions/IMO26_test --list
```

If Ollama already runs as a service, set `OLLAMA_CONTEXT_LENGTH` in the
service's environment and restart it. `ollama ps` shows the window each loaded
model actually got. On a Mac, macOS caps how much memory the GPU may wire;
`sudo sysctl iogpu.wired_limit_mb=40960` raises it until the next reboot.

To check that the server honours the window:

```bash
python3 -c 'from src import preflight; preflight.diagnose()'
```

## Running

```bash
python3 run.py "research goal"                  # runs until stopped; web tools withheld
python3 run.py "goal" --max-steps 20            # bounded by steps
python3 run.py "goal" --problem-minutes 90      # bounded by wall-clock time
python3 run.py --open "goal"                    # open problems: web tools allowed
python3 run.py --resume                         # continue unsolved notebooks
```

Ctrl-C finishes the current step and exits. State is written as it goes, so
nothing is lost. In a batch, Ctrl-C stops the batch.

### Problem files

Put problems in `questions/` as `.md`, `.txt`, `.html`, `.pdf` or images
(`.png`, `.jpg`, `.webp`, ...), and pass any mix of files, directories and
globs:

```bash
python3 run.py --questions                      # everything under questions/
python3 run.py --questions imo26.pdf            # one file
python3 run.py --questions a.txt b.pdf shots/   # mixed batch
python3 run.py --questions 'imo/*.png'          # glob (quote it)
python3 run.py --manifest set.list              # ordered list, one path per line
python3 run.py --questions ... --list           # parse and list only; runs nothing
```

Run `--list` first. It shows which problems were found and where, without
loading a model.

- One file can hold several problems. Markers such as `Problem 3.`, `Q2)` or
  `## Question 4` split it. A bare `3.` counts as a marker only when the file
  has no explicit ones, so numbered steps inside a problem stay together.
- Problems are deduplicated by normalized text, so a photo and its transcript
  do not both run.
- Images and scanned PDFs are transcribed by the vision model before the run
  starts and cached in `questions/.transcribed/`. A failed transcription is
  skipped, never passed on as problem text.
- Directory walks skip `.transcribed/`, `archive/` and dot-directories. Naming
  a file explicitly still loads it. `questions/archive/exam.png` is a small
  image example.

### Output

| Path | Contents |
|---|---|
| `problems/NNN-slug.md` | the notebook: statement, plan, every attempt, verified Lean, summaries |
| `solutions/` | clean write-ups, only for solved problems |
| `lean/Proofs/Verified.lean` | the growing library of verified lemmas |
| `runs/` | JSONL transcripts and the proof cache |
| `traces/attempts.jsonl` | every Lean-judged attempt, usable as training data |

When the context fills, the agent writes a digest into the notebook before it
trims the conversation, so compaction loses nothing on disk.

## Models

| Role | Default | Where |
|---|---|---|
| Planner (optional) | `qwen3.8-max` | DashScope, one call per problem |
| Orchestrator, reviewer, vision | `Qwen3.8-27B` Q8 | local, Ollama |
| Prover | `Goedel-Prover-V2-32B` bf16 (base or our fine-tune), best-of-32 | GPU server, vLLM |
| Prover fallback | `Goedel-Prover-V2-8B` Q8 | local, Ollama |
| Embeddings | `nomic-embed-text`, `bge-m3`, `mxbai-embed-large` | local, Ollama |

Every model is set in `.env`; see [.env.example](.env.example). Without a GPU
server, the 8B prover does all proving. To serve the fine-tuned prover, point
vLLM at its weights under the base model's name (see TRAINING.md, "Deploy").

We recommend Qwen3.8-27B at Q8 for the orchestrator, and we chose it by
measurement. On `evals/formalization_v2.py` (15
cases across 7 domains, including 4 false "mutation twins" of true theorems):

| Model | SIG@1 | SIG@3 | Wall time (SIG@3) | Leakage |
|---|---|---|---|---|
| Qwen3.8-27B | 9/15 | **12/15** | 50.9 min | none |
| nemotron-cascade-2 | 8/15 | 9/15 | 49.5 min | none |
| nemotron-3.5-lightning | 2/15 | | 45.5 min | copied a memorized theorem |

Given the false twin `gcd·lcm = m + n`, nemotron-3.5-lightning wrote out the
memorized true theorem instead. A formalizer that does that cannot be trusted
with the claim it was actually given.

### Remote prover

The 32B prover needs about 64 GB of GPU memory. On a server with two 48 GB
GPUs:

```bash
git clone https://github.com/ArkaDutta-007/AutoSolver.git && cd AutoSolver
./scripts/server_setup.sh models    # installs vLLM, downloads weights, installs Lean
./scripts/server_setup.sh serve     # Goedel-Prover-V2-32B on :8001 (and :8002 with 4 GPUs)
```

On your machine, open a tunnel and point `.env` at it:

```bash
ssh -N -L 8001:127.0.0.1:8001 gpu-box &
```

```
PROVER_FALLBACK_BASE=http://127.0.0.1:8001/v1
PROVER_FALLBACK_MODEL=Goedel-LM/Goedel-Prover-V2-32B
PROVER_BATCH_BASE=http://127.0.0.1:8001/v1
PROVER_BATCH_MODEL=Goedel-LM/Goedel-Prover-V2-32B
```

If you also set `GOEDEL_HOST=gpu-box`, the agent manages the server itself.
When the server is reachable but not serving, it runs
`scripts/goedel_tunnel.sh up`, which copies `scripts/serve_goedel.sh` to the
server, starts vLLM and opens the tunnel. Edit the GPU list in
`serve_goedel.sh` for your machine first. If the server does not answer, the
agent falls back to the local 8B prover and does not retry for ten minutes.

The server is worth having for best-of-N alone: Ollama ignores the `n`
parameter, and Goedel-Prover-V2's published 88.1% on miniF2F is pass@32. In
one measured run, sampling 32 candidates took 718 s and compiling all of them
took 29 s, so generation is the bottleneck, not Lean.

## What "proved" means

**The statement can be wrong even when the proof is right.** Lean checks that
the proof matches the statement. It does not check that the statement matches
your claim. Asked to prove that there are infinitely many primes, an early
version produced

```lean
theorem infinitude_of_primes : ∃ f : ℕ → ℕ, ∀ n, f n ≥ 2 ∧ ∀ m < f n, ¬ Nat.Prime m
  := by use fun n => 2
```

which compiles, is true, and says nothing about infinitude. Every accepted
proof therefore goes through the back-translation review, on every success
path, including the tactic probe and statements pinned by the caller. A
mismatch does not count, and the search continues. The reviewer is a model
judging natural language, and by default it is the same model that wrote the
signature, so spot-check statements that matter. The formalization benchmark
(`evals/formalize_score.py`) avoids model judges entirely: it asks Lean to
prove the produced statement and the gold statement equivalent.

**`exact?` succeeding means Mathlib already had the theorem.** That is a real
speedup but not a new result. The `solved_by` field records which path closed
each goal, so count `tactic:exact?` and `cache` separately.

**Search is closed by default.** Published solutions to competition problems
are on the web, and with search enabled a "proof" could be retrieval. Web
tools are left out of the tool list unless you pass `--open`, and each web
tool also checks the flag itself, because models do call tools they were never
offered. For open problems there is nothing to leak, so `--open` costs nothing.

## Lessons from building it

These cost real debugging time and apply to most LLM-plus-Lean setups.

- **Ollama's `/v1/chat/completions` has no `num_ctx` parameter.** It uses the
  server's `OLLAMA_CONTEXT_LENGTH`, 4096 by default, whatever the model
  supports, and nothing in the response reports truncation. Our agent did
  correct reasoning on an IMO problem, lost it to truncation, and started
  working on a problem it made up. Calls that matter now go to the native
  `/api/chat` endpoint and ask for the window explicitly.
- **Reasoning models put text in `reasoning`, not `content`.** Reading
  `content` alone yields an empty string. They also bill their thinking against
  `max_tokens`, so a small budget returns nothing with `finish_reason: length`.
- **Continue a truncated turn; do not regenerate it.** Retrying at double the
  budget throws away the partial reasoning and pays for it again: about 7
  minutes for one step across an 8k/16k/32k ladder.
- **Independent attempts beat more repair rounds.** Once a context holds a
  wrong approach, the model repeats the same error indefinitely. A fresh
  context with a different strategy breaks the loop.
- **`sorry` and empty files compile.** Lean reports `sorry` as a warning with
  exit code 0, and a file with no declarations is valid. Both are rejected
  explicitly.
- **Provers emit bare theorems.** The harness supplies the imports and `open`
  lines. Without them you get confusing errors such as `ℕ` versus `Nat`.
- **`import Mathlib` costs about 25 s per process.** A persistent REPL pays it
  once (`scripts/setup_repl.sh`). On a machine that is swapping under a large
  resident model, the same import took 291 s.
- **Reject bad signatures before Lean sees them.** A missing `theorem`, a
  comment-only body, a `???` placeholder or an `import` inside the statement
  can be caught by a syntax check at no cost. Before that check, three such
  rejections in one run cost 905 s.
- **Mathlib module names drift.** `Mathlib.Data.Nat.Parity` became
  `Mathlib.Algebra.Ring.Parity`. Search `lean/.lake/packages/mathlib` rather
  than trusting a model's memory of module paths.

[LIMITATIONS.md](LIMITATIONS.md) lists known failure modes in more detail.

## Repository layout

```
run.py                  entry point
src/
  agent.py              orchestrator loop
  plan.py               planner call and the obligation ledger
  tools.py              tool specs and dispatch
  prover.py             proof search: best-of-N, attempts, repair rounds
  lean.py, repl.py      Lean checking; persistent REPL with fallback to `lake env lean`
  tactics.py            tactic pre-probe (`exact?`, `aesop`, ...)
  review.py             back-translation gate
  falsify.py, bidir.py  counterexample search; backward refutation
  mathlib_kb.py         dense retrieval over Mathlib (multi-embedder)
  mathlib_search.py     keyword search fallback
  recall.py, scout.py   evidence and cheap probes pushed into context
  archive.py, bandit.py, evolve.py, library.py   the four feedback loops
  formalizer.py         client for an optional fine-tuned formalizer
  inputs.py, vision.py  problem files; image and scanned-PDF transcription
  journal.py, trace.py  notebooks; the attempt log
  config.py             every setting, read from .env
lean/                   Lean project (Mathlib pinned); Proofs/Verified.lean is the library
evals/                  formalization benchmarks, ablations, retrieval recall
scripts/                setup, GPU server, prover and formalizer training
questions/              problem sets (IMO 2026, Putnam 2025)
tests/                  unit and wiring tests
kb/                     built indexes and learned state (gitignored)
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The unit tests need neither models nor a GPU. `tests/smoke.py`,
`tests/smoke_e2e.py` and `tests/hardening.py` exercise a live setup and
expect Ollama and Lean to be running.

## License

MIT. See [LICENSE](LICENSE).
