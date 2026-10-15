"""Configuration. Secrets come from .env (mode 0600), never from source."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
# In-repo, not ~/proofs. Everything this project owns lives under the repo:
# the Lean project holds Proofs/Verified.lean, the accumulated lemma library,
# which is state the repo must carry with it.
LEAN_PROJECT = os.path.join(ROOT, "lean")
RUNS_DIR = os.path.join(ROOT, "runs")


def load_env(path: str = ENV_PATH) -> None:
    """Minimal .env reader. Existing environment variables win."""
    if not os.path.exists(path):
        return
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        print(f"WARNING: {path} is mode {mode:o}; secrets should be 0600. "
              f"Fix with: chmod 600 {path}")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


load_env()

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434/v1")
# Ollama's native API (constrained decoding via `format`) is NOT under /v1.
OLLAMA_NATIVE = OLLAMA_BASE[:-3].rstrip("/") if OLLAMA_BASE.endswith("/v1") else OLLAMA_BASE
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "nemotron-3.5-lightning:latest")
PROVER_MODEL = os.environ.get(
    "PROVER_MODEL", "hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q8_0")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", ORCHESTRATOR_MODEL)
# Multimodal, for transcribing image/scanned-PDF problems into markdown+LaTeX.
# NOTE: 31 GB. Loading it evicts a resident model, so transcription is a
# separate cached pass rather than an inline step.
VISION_MODEL = os.environ.get("VISION_MODEL", "muse-glimmer:30b-q8_0")
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "8000"))

# The PLANNER: one call to a large cloud model per problem, before any local
# work. It restates the problem, picks the approach and orders the tools; the
# local orchestrator then does every step. Empty PLANNER_MODEL, or no key, =
# plan locally as before. Set PLANNER_MODEL= to stay fully local.
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "qwen3.8-max")
PLANNER_BASE = os.environ.get("PLANNER_BASE",
                              "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
PLANNER_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
PLANNER_MAX_TOKENS = int(os.environ.get("PLANNER_MAX_TOKENS", "6000"))

# Remote prover (a GPU server via SSH tunnel). Empty = disabled.
PROVER_FALLBACK_BASE = os.environ.get("PROVER_FALLBACK_BASE", "")
PROVER_FALLBACK_MODEL = os.environ.get("PROVER_FALLBACK_MODEL", "")

SERPER_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"
SEARCH_BUDGET = int(os.environ.get("SEARCH_BUDGET", "200"))
# Off unless explicitly enabled: published solutions are on the open web, and a
# proof benchmark must not silently become a retrieval benchmark.
SEARCH_ENABLED = os.environ.get("SEARCH_ENABLED", "0") == "1"

# Proof effort. Auto-research is meant to be stubborn: many independent attempts,
# each with several self-correction rounds, before a claim is called unproved.
PROVE_ATTEMPTS = int(os.environ.get("PROVE_ATTEMPTS", "5"))
PROVE_ROUNDS = int(os.environ.get("PROVE_ROUNDS", "6"))
PROVE_TIME_BUDGET = int(os.environ.get("PROVE_TIME_BUDGET", "1800"))  # seconds per claim
# 4000 was sized for a 4096-token window. Goedel-Prover-V2 is a REASONING model
# -- it bills its thinking against this budget -- so on the first real run every
# one of 5 rounds came back "TRUNCATED at max_tokens" and Lean never saw a
# single generated proof in 2251 seconds. With a 32k window the generation
# budget can be a real one.
PROVER_MAX_TOKENS = int(os.environ.get("PROVER_MAX_TOKENS", "16000"))
PROVER_MAX_TOKENS_CAP = int(os.environ.get("PROVER_MAX_TOKENS_CAP", "28000"))
# Truncation retries do not consume self-correction rounds (nothing compiled, so
# nothing was learned), but they must not loop forever either.
PROVER_MAX_TRUNCATIONS = int(os.environ.get("PROVER_MAX_TRUNCATIONS", "3"))

# Derive a Lean signature from an informal claim, then pin it. Unlocks the tactic
# pre-probe and proof-only mode -- the two largest measured speedups.
SIGNATURE_FIRST = os.environ.get("SIGNATURE_FIRST", "1") == "1"
# Append verified theorems to Proofs/Verified.lean so later proofs can build on them.
GROW_LIBRARY = os.environ.get("GROW_LIBRARY", "1") == "1"

# Orchestrator generation budget. Lightning is a thinking model and reasoning is
# billed against this, so a small budget truncates mid-thought. Generous by
# default: one long pass is far cheaper than regenerating a short one twice.
# 24000 was set against a context window this system never had. In a 32k
# window, a 24k generation budget leaves
# 8k for the system prompt, the problem, the journal and every tool result --
# i.e. it guarantees the truncation it was meant to avoid. 12k is roughly a
# third of the window, which leaves room to actually think AND remember.
ORCH_MAX_TOKENS = int(os.environ.get("ORCH_MAX_TOKENS", "12000"))
# Stream the orchestrator's reasoning to the terminal as it is produced. A step
# can think for minutes; without this the terminal shows nothing until it ends,
# and grinding is indistinguishable from looping.
STREAM_TRACE = os.environ.get("STREAM_TRACE", "1") == "1"

LEAN_TIMEOUT = int(os.environ.get("LEAN_TIMEOUT", "120"))

# Persistent Lean REPL. `import Mathlib` costs ~25-29s per `lake env lean`, and
# that is paid per check: 1 for signature validation, up to 3 for recovery, 12
# for the tactic ladder. "auto" uses the REPL when it has been built (see
# scripts/setup_repl.sh) and silently falls back to the subprocess path when it
# has not, so nothing breaks if it is missing.
LEAN_REPL = os.environ.get("LEAN_REPL", "auto")          # auto | off
LEAN_REPL_PATH = os.environ.get("LEAN_REPL_PATH", "")
# MEMORY: each worker holds a fully-imported Mathlib, ~5.7 GB resident (measured
# in this repo). The subprocess path pays the same per concurrent `lake env
# lean`, so the PEAK does not rise -- but the REPL holds it continuously instead
# of in 29s spikes. It also CAPS Lean's footprint: every caller queues against
# these slots, so memory stops scaling with TACTIC_WORKERS.
#   1 worker  ~5.7 GB   -- safe alongside 36 GB of models on a 64 GB machine
#   2 workers ~11.4 GB  -- comfortable on 96 GB+, or with models on the lab server
LEAN_REPL_WORKERS = int(os.environ.get("LEAN_REPL_WORKERS", "1"))
# Hand the RAM back when nothing is being proved. Workers restart on the next
# check and pay one import; a run that is thinking rather than compiling should
# not be sitting on gigabytes. 0 disables the reaper.
LEAN_REPL_IDLE_SECONDS = int(os.environ.get("LEAN_REPL_IDLE_SECONDS", "900"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "6000"))

# --- context: 128K, asked for rather than measured -----------------------------
# Ollama's /v1 endpoint ignores num_ctx and applies OLLAMA_CONTEXT_LENGTH
# (default 4096), which is why the window used to be MEASURED at startup with
# needle probes at 4k, 8k, 16k and 32k. That cost ~7 minutes of prefill before
# the problem was even read. Every call
# that matters now goes to the NATIVE endpoint and asks for its window in
# `options.num_ctx`, so there is nothing to measure: the window is what we ask
# for. `python -m src.preflight` still diagnoses a server that will not honour it.
NUM_CTX = int(os.environ.get("NUM_CTX", "") or 131072)
# Everything downstream budgets from this.
CONTEXT_TOKENS = int(os.environ.get("CONTEXT_TOKENS", "") or NUM_CTX)
# Only the manual diagnosis in preflight uses this now.
MIN_CONTEXT_TOKENS = int(os.environ.get("MIN_CONTEXT_TOKENS", "32000"))
# Compact the conversation once the prompt fills this much of the window.
COMPACT_AT = float(os.environ.get("COMPACT_AT", "0.75"))
# Rough bytes-per-token for mixed English + LaTeX + Lean, until the server has
# counted a real prompt (agent.py calibrates from prompt_eval_count).
CHARS_PER_TOKEN = 3.5


def ctx(model: str) -> dict:
    """`options.num_ctx` for a native call, or nothing. Only the reasoning
    models get the long window: the 8B prover and the vision model would each
    pay ~10 GB of KV cache for context they never fill."""
    return {"num_ctx": NUM_CTX} if NUM_CTX and model in (ORCHESTRATOR_MODEL, REVIEWER_MODEL) else {}


def set_context_tokens(n: int) -> None:
    global CONTEXT_TOKENS
    CONTEXT_TOKENS = int(n)


def budget_chars(fraction: float) -> int:
    """A character budget expressed as a fraction of the REAL context window."""
    return int(CONTEXT_TOKENS * CHARS_PER_TOKEN * fraction)


# --- sandbox ------------------------------------------------------------------
# Numerical/symbolic exploration. Conjecture cheaply, formalize only survivors.
# Narrate proof search as it happens. The orchestrator already streams; without
# this the prover is a silent 5-15 minute black box, and a hang looks exactly
# like hard work.
PROVER_TRACE = os.environ.get("PROVER_TRACE", "1") == "1"

# Exploration budget before the agent must formalize something. Observed in
# run-20260813-191856: 15 consecutive run_python calls, 15 distinct programs,
# zero prove_theorem. A cheap tool with no forcing function replaced "never
# explores" with "only explores". Nudge at the first, withhold run_python at
# the second.
PROVE_NUDGE_AFTER = int(os.environ.get("PROVE_NUDGE_AFTER", "6"))
PROVE_FORCE_AFTER = int(os.environ.get("PROVE_FORCE_AFTER", "12"))

SANDBOX_ENABLED = os.environ.get("SANDBOX_ENABLED", "1") == "1"
SANDBOX_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "60"))

# --- proof hygiene ------------------------------------------------------------
# `sorry` and empty files were already rejected. These are the next rungs: a
# proof may also "succeed" by leaning on an axiom that trivialises it. Under any
# search pressure (and certainly under RL) the model will find these.
AXIOM_AUDIT = os.environ.get("AXIOM_AUDIT", "1") == "1"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
# Sampling above ~1.0 mostly produces syntactically broken Lean; diversify with
# premises and strategies instead of heat.
MAX_PROVER_TEMP = float(os.environ.get("MAX_PROVER_TEMP", "1.0"))

# Per-problem wall-clock cap in a batch run. 0 = unlimited. Without this one
# problem consumes an entire overnight batch and the rest never start.
PROBLEM_TIME_BUDGET = int(os.environ.get("PROBLEM_TIME_BUDGET", "0"))


# Reasoning effort for the orchestrator's own turns. NOT used on any call that
# also constrains output with a `format` schema -- see tests/test_schema_calls.py
# for the six times that combination silently returned nothing.
ORCH_THINK = os.environ.get("ORCH_THINK", "high")
ORCH_PASS_K = int(os.environ.get("ORCH_PASS_K", "4"))
ORCH_PASS_EVERY = int(os.environ.get("ORCH_PASS_EVERY", "3"))
ORCH_PROPOSER_MODEL = os.environ.get("ORCH_PROPOSER_MODEL", "")


# ---------------------------------------------------------------------------
# GAME-THEORETIC LAYER. One flag per mechanism, independently switchable so
# `evals/ablation.py` can measure whether each earns its cost.
#
# All of them write into the ANCHOR rather than exposing a tool: the agent was
# measured never to call search_mathlib even while inventing names Mathlib
# already had, so a mechanism the model must remember to invoke does not run.
# ---------------------------------------------------------------------------
GT_RECALL = os.environ.get("GT_RECALL", "1") == "1"
GT_RECALL_K = int(os.environ.get("GT_RECALL_K", "4"))
GT_RECALL_FLOOR = float(os.environ.get("GT_RECALL_FLOOR", "0.34"))

GT_SCOUT = os.environ.get("GT_SCOUT", "1") == "1"
GT_SCOUT_BUDGET = float(os.environ.get("GT_SCOUT_BUDGET", "45"))
GT_SCOUT_MAX = int(os.environ.get("GT_SCOUT_MAX", "4"))

# MEASURED DEAD, default off: across every adjudicated item the forecaster
# returned a near-constant (0.01-0.15, zero spread on scorable rows), so there
# is no variance to rank with -- and ranking was the only thing its allocator
# needed. It also costs a model call per obligation. Flip on only to re-measure.
GT_FORECAST = os.environ.get("GT_FORECAST", "0") == "1"

# Numeric counterexample search before any derivation is paid for. One sandbox
# run; only a FOUND counterexample changes behaviour.
GT_FALSIFY = os.environ.get("GT_FALSIFY", "1") == "1"

# Carrying capacity on the library: growth without a death rate overshoots the
# context window, measured at 28,700 of 32,768 tokens with 35% empty steps.
GT_RETIRE = os.environ.get("GT_RETIRE", "1") == "1"
GT_LIBRARY_CAP = int(os.environ.get("GT_LIBRARY_CAP", "12"))


# Wall-clock derivation for a generation of N tokens. The old fixed 900s meant
# the TIMEOUT, not the token budget, decided when a model stopped -- and only
# for models slower than budget/900 tok/s, so the handicap was invisible in any
# comparison against a fast one.
ORCH_MIN_TOK_PER_SEC = float(os.environ.get("ORCH_MIN_TOK_PER_SEC", "5"))
ORCH_TIMEOUT_CAP = int(os.environ.get("ORCH_TIMEOUT_CAP", "7200"))

# The batched best-of-N prover request must fit the window vLLM actually serves.
PROVER_BATCH_MAX_TOKENS = int(os.environ.get("PROVER_BATCH_MAX_TOKENS", "4096"))

# Batched best-of-N against the vLLM endpoint. Goedel-V2's headline 88.1% on
# MiniF2F is pass@32, and Ollama ignores `n` -- only vLLM honours it, so this is
# the path that reaches the published number.
PROVER_BATCH_BASE = os.environ.get("PROVER_BATCH_BASE", "")
PROVER_BATCH_MODEL = os.environ.get("PROVER_BATCH_MODEL", "")
PROVER_BATCH_N = int(os.environ.get("PROVER_BATCH_N", "32"))

# Between problems, generalise and dedupe the verified library. This never once
# ran before the missing `library` import in agent.py was fixed.
LIBRARY_SLEEP = os.environ.get("LIBRARY_SLEEP", "1") == "1"
LIBRARY_SLEEP_EVERY = int(os.environ.get("LIBRARY_SLEEP_EVERY", "1"))


# The tuned specialist formalizer (7B LoRA on Mathlib pairs), served by vLLM
# on a GPU server (scripts/serve_formalizer.sh), reached through an ssh tunnel. Empty =
# use the 27B orchestrator for signature derivation, exactly as before. The
# specialist keeps this job only if it beats the 27B's measured 28% on the
# 210-item benchmark by >= 7 points.
FORMALIZER_BASE = os.environ.get("FORMALIZER_BASE", "")
FORMALIZER_MODEL = os.environ.get("FORMALIZER_MODEL", "formalizer-v1")


# Backward (top-down) leg alongside the forward one: derive a sufficient set for
# the goal and try to REFUTE its elements. One false element kills the route,
# which forward search never learns. Measured on IMO26 P1: dead in 184s, with
# the same false "product is invariant" lemma the decomposer spent 41 minutes on.
GT_BIDIR = os.environ.get("GT_BIDIR", "1") == "1"
