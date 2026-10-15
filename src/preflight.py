"""Startup checks that fail LOUDLY rather than degrading silently.

Why this file exists
--------------------
Run `runs/run-20260813-171447.jsonl` step 3 produced 113,191 characters of
correct IMO 2026 P1 reasoning (it found the right invariant: a move sends the
exponent pair (a,b) -> (min(a,b), |a-b|), and gcd(min(a,b),|a-b|) = gcd(a,b), so
the gcd of the exponent multiset is conserved). Step 4 then said, verbatim:

    "I don't have any context from a previous conversation."

The reasoning was gone. By step 7 the agent had invented a different problem
("Fundamental Theorem of Arithmetic") to work on instead, which is where
`problems/002-*` and `problems/003-*` came from.

Cause: Ollama's OpenAI-compatible `/v1/chat/completions` endpoint has no
`num_ctx` parameter. The server applies `OLLAMA_CONTEXT_LENGTH` (default 4096)
regardless of what the model can do. A 1M-context model served with a 4k window
looks completely normal -- until it quietly forgets the problem statement.

Three independent things can each cause a small window, so all three are
measured separately rather than collapsed into one pass/fail:

  declared  -- what the model's Modelfile asks for      (/api/show)
  allocated -- what the server actually gave it         (/api/ps, = `ollama ps`)
  usable    -- what survives an end-to-end request      (needle probe)

The needle probe alone is not enough. A THINKING model spends its generation
budget on reasoning before it answers, so a probe with a small `max_tokens`
returns empty content and looks exactly like context loss. That trap is already
documented in this repo's README ("Thinking models need large max_tokens") and
the first version of this file walked straight into it.
"""

import json
import os

import requests

from . import config

NEEDLE = "PASSPHRASE-CRIMSON-VECTOR-8812"

# Ollama drops the OLDEST tokens when a prompt exceeds the window, so the needle
# goes FIRST: if it survives, everything after it did too.
_PROBE = ("Remember this exactly: {needle}\n\n"
          "{filler}\n\n"
          "What is the passphrase at the very top of this message? "
          "Reply with the passphrase and nothing else.")

_FILLER_UNIT = "The quick brown fox jumps over the lazy dog. "   # ~11 tokens

# Generous on purpose. A thinking model bills reasoning against this budget, and
# a truncated reply is indistinguishable from a lost needle.
PROBE_MAX_TOKENS = int(os.environ.get("PROBE_MAX_TOKENS", "1200"))


def _filler(tokens: int) -> str:
    return _FILLER_UNIT * max(1, int(tokens / 11))


def declared_context(model: str) -> int:
    """What the model's Modelfile asks for. 0 if unset or unavailable."""
    try:
        r = requests.post(f"{config.OLLAMA_NATIVE}/api/show",
                          json={"model": model}, timeout=60)
        r.raise_for_status()
        params = r.json().get("parameters") or ""
        for line in params.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                return int(parts[1])
    except Exception:
        pass
    return 0


def allocated_context(model: str) -> int:
    """What the server actually allocated -- the `ollama ps` CONTEXT column.

    This is the authoritative number: it is what the running process has, after
    every layer of precedence has been resolved. Only populated once the model
    is loaded, so call it AFTER a request.
    """
    try:
        r = requests.get(f"{config.OLLAMA_NATIVE}/api/ps", timeout=30)
        r.raise_for_status()
        for m in r.json().get("models") or []:
            if m.get("name") == model or m.get("model") == model:
                return int(m.get("context_length") or m.get("context") or 0)
    except Exception:
        pass
    return 0


def probe(model: str, tokens: int, native: bool = False, timeout: int = 600) -> str:
    """Needle-in-a-haystack. Returns 'ok' | 'lost' | 'no-output' | 'error: ...'.

    'no-output' is kept distinct from 'lost' precisely because a thinking model
    that ran out of generation budget must not be reported as context loss.
    """
    content = _PROBE.format(needle=NEEDLE, filler=_filler(tokens))
    try:
        if native:
            # Native /api/chat DOES accept options.num_ctx. Probing both paths
            # separates "the server cannot do this window" from "the /v1 path
            # is not asking for it".
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": content}],
                      "stream": False,
                      "options": {"num_ctx": max(tokens + 2048, 8192),
                                  "num_predict": PROBE_MAX_TOKENS, "temperature": 0}},
                timeout=timeout)
            r.raise_for_status()
            msg = r.json().get("message") or {}
        else:
            r = requests.post(
                f"{config.OLLAMA_BASE}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": content}],
                      "max_tokens": PROBE_MAX_TOKENS, "temperature": 0},
                timeout=timeout)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
    except Exception as exc:
        return f"error: {str(exc)[:120]}"

    text = ((msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")).strip()
    if NEEDLE in text:
        return "ok"
    if not text:
        return "no-output"
    return "lost"


def diagnose(model: str = None, target: int = None) -> dict:
    """Print every measurement, then say which layer is at fault.

    Run this by hand when preflight fails:
        python3 -c 'from src import preflight; preflight.diagnose()'
    """
    model = model or config.ORCHESTRATOR_MODEL
    target = target or config.MIN_CONTEXT_TOKENS
    probe_at = min(target, 16_000)         # cheap enough to run twice

    print(f"\ncontext diagnosis for {model}\n" + "-" * 60)
    dec = declared_context(model)
    print(f"  declared  (Modelfile num_ctx) : {dec or 'not set'}")

    v1 = probe(model, probe_at, native=False)
    print(f"  probe via /v1 at {probe_at:,} tokens : {v1}")
    nat = probe(model, probe_at, native=True)
    print(f"  probe via /api/chat + options : {nat}")

    alloc = allocated_context(model)
    print(f"  allocated (ollama ps CONTEXT) : {alloc or 'unknown'}")
    print("-" * 60)

    if v1 == "no-output" and nat == "no-output":
        verdict = ("The model returned NOTHING both ways. That is a generation-budget "
                   "or model problem, not a context problem. Raise PROBE_MAX_TOKENS, "
                   "or check the model actually responds: "
                   f"ollama run {model} 'say hi'")
    elif v1 == "ok":
        verdict = "Healthy. /v1 sees the whole prompt."
    elif nat == "ok" and v1 != "ok":
        verdict = ("The server CAN do this window, but the /v1 path is not getting it. "
                   "The Modelfile num_ctx is being overridden by the server default. "
                   "Set OLLAMA_CONTEXT_LENGTH in the LaunchAgent -- see REMEDY below.")
    elif dec and alloc and alloc < dec:
        verdict = (f"Modelfile asks for {dec:,} but the server allocated {alloc:,}. "
                   "OLLAMA_CONTEXT_LENGTH is winning. Set it in the LaunchAgent.")
    else:
        verdict = ("Neither path can hold the prompt. Set OLLAMA_CONTEXT_LENGTH in the "
                   "LaunchAgent and reload it (bootout + bootstrap, NOT kickstart).")
    print(f"  => {verdict}\n")
    return {"declared": dec, "allocated": alloc, "v1": v1, "native": nat,
            "verdict": verdict}


REMEDY = """
    The window is smaller than this run needs. Diagnose which layer is at fault:

        python3 -c 'from src import preflight; preflight.diagnose()'

    Ollama resolves context as:  request option > OLLAMA_CONTEXT_LENGTH >
    Modelfile PARAMETER > built-in default (4096). Calls on the native API ask
    for NUM_CTX themselves; calls on /v1 get the server's setting. The fix that
    always works is to set it on the server and restart it:

        OLLAMA_CONTEXT_LENGTH=32768 ollama serve
        # macOS launchd service: add the variable to its plist, then
        # `launchctl bootout` + `launchctl bootstrap` (kickstart does NOT
        # re-read the plist)

    Confirm what the server actually allocated (this is the ground truth):

        ollama ps      # CONTEXT column

    Watch memory: a ~25 GB orchestrator plus a 32k KV cache plus the 11 GB
    prover has to fit in GPU memory. On Apple silicon that is capped by
    `iogpu.wired_limit_mb`. If Ollama starts evicting models, drop to 16384,
    or move the prover to a GPU server (README, "Remote prover").
"""


def check_context(model: str = None, require: int = None, verbose: bool = True) -> int:
    """Measure usable context. Returns the largest probed size that passed."""
    model = model or config.ORCHESTRATOR_MODEL
    require = require or config.MIN_CONTEXT_TOKENS

    ladder = sorted({n for n in (4_000, 8_000, 16_000, 32_000, 64_000)
                     if n <= require} | {require})

    best, last = 0, ""
    # with NUM_CTX set, the calls that matter go native and ask for it, so the
    # probe must travel the same road or it measures a window nobody uses
    native = bool(config.ctx(model))
    for n in ladder:
        last = probe(model, n, native=native)
        if verbose:
            print(f"  context probe {n:>7,} tokens: {last}", flush=True)
        if last != "ok":
            break
        best = n
        if best >= require:
            break

    if best < require:
        detail = ""
        if last == "no-output":
            detail = ("\n    NOTE: the model returned NOTHING, which is a generation-budget\n"
                      "    symptom, not necessarily context loss. Thinking models bill\n"
                      "    reasoning against the same budget. Try PROBE_MAX_TOKENS=4000.\n")
        raise SystemExit(
            f"\nFATAL: usable context is ~{best:,} tokens; this run needs >= {require:,}."
            f"\n    (last probe result: {last})\n{detail}" + REMEDY
            + "\n    (Override with AUTORESEARCH_SKIP_PREFLIGHT=1 if you know what you are doing.)\n")
    return best


def check_endpoint(base: str, label: str) -> bool:
    try:
        requests.get(f"{config.OLLAMA_NATIVE}/api/tags", timeout=10).raise_for_status()
        return True
    except Exception as exc:
        print(f"  WARNING: {label} unreachable at {base} ({exc})", flush=True)
        return False


def run(verbose: bool = True) -> dict:
    """Startup check. Called from run.py before any problem is opened.

    No context probes: the window is asked for on every native call
    (config.NUM_CTX), so measuring it cost minutes and told us nothing new.
    What is still worth failing loudly on is a server that is not there.
    """
    if os.environ.get("AUTORESEARCH_SKIP_PREFLIGHT") == "1":
        print("preflight: SKIPPED (AUTORESEARCH_SKIP_PREFLIGHT=1)", flush=True)
        return {"skipped": True}

    print("preflight:", flush=True)
    check_endpoint(config.OLLAMA_BASE, "ollama")
    ctx = config.CONTEXT_TOKENS
    if config.ORCH_MAX_TOKENS > ctx * 0.5:
        print(f"  WARNING: ORCH_MAX_TOKENS={config.ORCH_MAX_TOKENS:,} is more than half "
              f"the {ctx:,}-token window; long turns will crowd out the problem "
              f"statement. Consider <= {int(ctx * 0.4):,}.", flush=True)
    print(f"  context: {ctx:,} tokens, compacting at {config.COMPACT_AT:.0%}\n", flush=True)
    return {"context_tokens": ctx}
