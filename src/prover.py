"""Lean proving with escalating retries.

The point of auto-research is stubbornness. A single prompt to a single model is
what you do when the problem is easy; this instead runs several INDEPENDENT
attempts, each with its own self-correction rounds, escalating strategy as it
goes, until the claim is proved or the budget is spent.

Why independent attempts rather than more rounds in one conversation: once a
context is poisoned by a wrong approach, the model reproduces the same error
indefinitely (observed: the identical type error six rounds running). Restarting
with fresh context and a different strategy is what actually breaks the loop.
"""

import concurrent.futures as futures
import functools
import json
import os
import re
import subprocess
import time

import requests

from . import cache, config, library, mathlib_search, tactics, trace
from .lean import audit_axioms, check_lean, split_signature, statement_ok
from .review import review_statement


def _trace(msg: str, indent: int = 5) -> None:
    """Say what the prover is doing, while it does it.

    The orchestrator got streaming for exactly this reason -- "a step is a blank
    `thinking...` for minutes followed by a wall of text", and grinding is
    indistinguishable from looping. The prover then inherited the same problem
    and none of the fix: prove_theorem could run for fifteen minutes (signature
    derivation, a 12-rung tactic ladder at ~25s an import, then attempts) and
    print absolutely nothing. Silence is not a progress indicator.
    """
    if config.PROVER_TRACE:
        print(f"{' ' * indent}{msg}", flush=True)


def _record(statement: str, lean_statement: str, out: dict) -> None:
    """Persist a verified proof: cache it, and grow the importable library."""
    try:
        cache.put(statement, out, lean_statement)
    except Exception:
        pass
    try:
        from . import archive
        archive.record(statement, ok=True, seconds=out.get("elapsed", 0.0),
                       code=out.get("lean", ""), goal=_GOAL[0])
    except Exception:
        pass
    if config.GROW_LIBRARY:
        try:
            # Which EXISTING lemmas did this proof lean on? Usage is the only
            # evidence that a library entry is an asset rather than clutter --
            # without it every lemma looks equally valuable and the library
            # cannot be pruned, only grown.
            library.note_uses(out.get("lean", ""), by=statement[:120])
        except Exception:
            pass
        try:
            added = library.add(out.get("lean", ""), origin=statement[:150])
            if added.get("ok"):
                out["library"] = added["name"]
            elif added.get("reason") == "duplicate":
                out["library"] = f"(already known as {added.get('name')})"
        except Exception:
            pass

# `import Mathlib` alone does NOT bring scoped notation into scope, so idiomatic
# Mathlib spellings failed to parse and looked like model errors. Measured: Qwen
# wrote `Aᵀᵀ = A` for "transposing twice gives back the original" -- correct, and
# rejected with `unexpected token 'ᵀ'`, while a model writing the more verbose
# `A.transpose.transpose` passed. The preamble was silently scoring VERBOSITY.
# Same class of bug as a fixed generation timeout: the harness, not the model.
# Every scope here is verified against all 15 v2 reference signatures (15/15
# still elaborate) -- `Classical` is deliberately NOT opened, since its
# decidability instances change elaboration rather than just adding notation.
PREAMBLE = ("import Mathlib\n"
            "open scoped Matrix BigOperators Topology NNReal ENNReal Pointwise\n"
            "set_option maxHeartbeats 400000\n\n")

_HAS_IMPORT = re.compile(r"^\s*import\s+\S", re.MULTILINE)
_HAS_DECL = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)
_UNKNOWN = re.compile(r"[Uu]nknown constant `([^`]+)`")

BASE_SYSTEM = (
    "You are a Lean 4 theorem prover using Mathlib. Output ONE fenced ```lean4 "
    "code block containing the theorem and its complete proof. `import Mathlib` "
    "is supplied automatically -- do not write import lines. "
    "NEVER use `sorry` or `admit`; they are rejected as incomplete proofs. "
    "Emit exactly one code block: your final, complete answer."
)

# Measured: across a full IMO suite run every success came from attempt 1 and no
# later attempt ever rescued a failure. Prompt-hint variation resamples the same
# basin. These strategies therefore change MECHANISM, not just wording:
#   sketch  -- a DIFFERENT MODEL proposes the informal plan first
#   search  -- real Mathlib names are injected before generation, not after failure
#   exact?  -- Lean's own proof search does the work
# Named so the bandit can learn which one works WHERE. The order below is only
# the cold-start order: once each has been tried in a given domain, UCB1 picks.
STRATEGIES = [
    ("direct", 0.3, "", None),
    ("leansearch", 0.7,
     "Use Lean's own proof search. For each goal, try `exact?`, `apply?`, "
     "`aesop`, or `simp?` and use what they find.", None),
    ("decompose", 0.8,
     "Decompose into several small `have` sub-lemmas, each independently "
     "provable, then combine. State each sub-lemma explicitly.", None),
    ("reroute", 0.9,
     "Your previous approaches failed. Abandon them entirely and use a "
     "different mathematical route to the same conclusion.", "sketch"),
    ("automation", 0.7,
     "Lean heavily on automation: `nlinarith`, `positivity`, `polyrith`, "
     "`omega`, `decide`, `norm_num`, `field_simp` then `ring_nf`.", "search"),
    ("generalize", 1.0,
     "Generalize: prove a STRONGER statement that implies the goal, which is "
     "often easier because induction/rewriting have more to work with.", "sketch"),
]
_BY_NAME = {s[0]: s for s in STRATEGIES}
STRATEGY_NAMES = tuple(s[0] for s in STRATEGIES)

_TACTIC = re.compile(r"\b(nlinarith|linarith|positivity|polyrith|omega|decide|aesop|"
                     r"norm_num|field_simp|ring_nf|simp|induction|rcases|exact\?|"
                     r"apply\?|constructor|intro|use|calc)\b")


def _signature(code: str) -> str:
    """Compact fingerprint of an approach: which tactics and lemmas it leaned on."""
    used = sorted(set(_TACTIC.findall(code)))[:6]   # not `tactics`: shadows the module
    lemmas = sorted({m.group(0) for m in mathlib_search._CITED.finditer(code)})[:6]
    return f"tactics[{','.join(used)}] lemmas[{','.join(lemmas)}]"


_TUNNEL = os.path.join(config.ROOT, "scripts", "goedel_tunnel.sh")
_SERVER_DOWN_UNTIL = [0.0]


def _answers(base: str) -> bool:
    try:
        requests.get(f"{base.rstrip('/')}/models", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


def _bring_up(base: str) -> bool:
    """Reachable box, stopped vLLM or closed tunnel: start both. A box that does
    not answer ssh in 8 s is down, and is not asked again for ten minutes, so a
    dead server costs one short wait rather than one per attempt."""
    host = os.environ.get("GOEDEL_HOST", "")
    if not host or time.time() < _SERVER_DOWN_UNTIL[0]:
        return False
    try:
        ok = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "-o", "RemoteCommand=none", host, "true"],
                            capture_output=True, timeout=20).returncode == 0
        if ok:
            _trace("[prove] server reachable, prover not answering: goedel_tunnel.sh up "
                   "(first load of the 32B takes a few minutes)")
            subprocess.run(["bash", _TUNNEL, "up"], capture_output=True, timeout=900)
            if _answers(base):
                return True
    except Exception:
        pass
    _SERVER_DOWN_UNTIL[0] = time.time() + 600
    return False


def _prover_target() -> tuple:
    """(base, model) for the next attempt. The lab server's 32B (PROVER_FALLBACK_*,
    named from when it was the fallback) whenever the box is reachable -- started
    if it is not serving; the local 8B only when the box itself is down."""
    base, model = config.PROVER_FALLBACK_BASE, config.PROVER_FALLBACK_MODEL
    if base and model:
        if _answers(base) or _bring_up(base):
            return base, model
        _trace(f"[prove] lab server unreachable; falling back to local {config.PROVER_MODEL}")
    return config.OLLAMA_BASE, config.PROVER_MODEL


def _sketch(statement: str, avoid: list) -> str:
    """Ask the ORCHESTRATOR (a different model) for an informal plan.

    Cross-model diversity is the one thing that reliably escapes a basin the
    prover keeps falling into; resampling the prover cannot.
    """
    prior = ("\n\nThese approaches have already FAILED -- propose something "
             "materially different:\n" + "\n".join(f"- {a}" for a in avoid[-4:])) if avoid else ""
    try:
        # native, not /v1: /v1 cannot ask for num_ctx, and a request for a
        # different window than the loaded one makes Ollama reload the model
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.ORCHESTRATOR_MODEL,
                  "messages": [{"role": "user", "content":
                                f"Give a SHORT informal proof plan (max 6 lines) for:\n\n"
                                f"{statement}{prior}\n\nName the key lemmas or identities "
                                f"to use. Do not write Lean."}],
                  "stream": False,
                  "options": {**config.ctx(config.ORCHESTRATOR_MODEL),
                              "num_predict": 3000, "temperature": 0.8}},
            timeout=900)
        r.raise_for_status()
        m = r.json().get("message") or {}
        return ((m.get("content") or "").strip() or (m.get("thinking") or "").strip())[-1200:]
    except Exception:
        return ""


_BODY = re.compile(r":=(?!.*\b(?:theorem|lemma)\b)", re.DOTALL)


def normalize_signature(sig: str) -> list:
    """Candidate spellings of a signature, best first.

    Rejecting outright is too brittle. `rstrip(':=')` is a character-class strip,
    so a signature that legitimately ends at the final `:` -- which is literally
    what derive_signature's own prompt asks for -- got its colon eaten and became
    malformed. And a model that helpfully appends `:= by sorry`, or wraps the
    thing in a code fence, was failing for reasons that have nothing to do with
    whether the mathematics typechecks. Try the obvious repairs before giving up.
    """
    s = re.sub(r"```(?:lean4?)?", "", sig or "").strip()
    out = []

    def add(x):
        x = x.strip()
        if x and x not in out:
            out.append(x)

    add(s)
    if s.endswith(":"):                       # "up to and including the final `:`"
        add(s[:-1].rstrip())
    m = _BODY.search(s)
    if m:                                     # a proof body came along for the ride
        add(s[:m.start()].rstrip())
    add(re.sub(r"\bsorry\b", "", s).rstrip().rstrip(":=").rstrip())
    return out


_JUNK = re.compile(r"\?\?\?|\bTODO\b|\bFIXME\b|<[a-z ]+>|\.\.\.")

# Binary Mathlib functions the orchestrator reaches for constantly. Applied to
# ONE argument they are partial applications of type `ℕ → ℕ`, and Lean will
# happily elaborate arithmetic on those pointwise via `Pi.instMul` -- so
#     Nat.gcd m * Nat.lcm m n = m * n
# TYPECHECKS, states something about functions, and has nothing to do with the
# claim. Observed from a real derivation, and it passed every gate we had:
# elaboration is necessary but nowhere near sufficient.
_BINARY = ("Nat.gcd", "Nat.lcm", "Nat.dist", "Nat.sub", "Nat.add", "Nat.mul",
           "gcd", "lcm", "min", "max")


def arity_lint(sig: str) -> str:
    """Catch binary functions applied to one argument. "" if clean."""
    body = sig.split(":", 1)[-1] if ":" in sig else sig
    body = re.sub(r"--[^\n]*", "", body)
    for fn in _BINARY:
        # `fn x` followed by an operator/end rather than a second argument.
        pat = re.compile(r"(?<![\w.])" + re.escape(fn) +
                         r"\s+([A-Za-z_][\w']*|\([^()]*\))\s*(?=[*+\-/=<>≤≥∣,)]|$)",
                         re.MULTILINE)
        m = pat.search(body)
        if m:
            return (f"`{fn}` is binary but is applied to one argument "
                    f"(`{fn} {m.group(1)}`). That is a partial application of type "
                    f"`ℕ → ℕ`, and Lean will elaborate arithmetic on it POINTWISE, "
                    f"so the statement typechecks while meaning something else. "
                    f"Write `{fn} x y`.")
    return ""


def _syntax_reject(sig: str) -> str:
    """Reject a hopeless signature WITHOUT paying for a Mathlib compile.

    Measured: three rejections in one run cost 291.7s + 293.4s + 320.7s with
    `attempts: 0`. Two of the three were rejectable by reading them --
    `m.val.gcd n p ??? -- need correct` and a signature that was only a comment.
    A cold `import Mathlib` under memory pressure is ~5 minutes; this is free.

    Returns a reason string, or "" if the signature is worth typechecking.
    """
    s = re.sub(r"```(?:lean4?)?", "", sig or "").strip()
    if not s:
        return "empty signature"
    # Strip comments before judging: a signature that is ONLY commentary has no
    # statement in it at all.
    bare = re.sub(r"--[^\n]*", "", re.sub(r"/-.*?-/", "", s, flags=re.S)).strip()
    if not bare:
        return "signature contains only comments, no declaration"
    if not re.search(r"\b(theorem|lemma|example)\b", bare):
        return "no `theorem`/`lemma`/`example` declaration"
    if ":" not in bare:
        return "no `:` -- a signature must state a type"
    m = _JUNK.search(bare)
    if m:
        return (f"placeholder {m.group(0)!r} left in the signature -- write the "
                f"statement out in full")
    if re.match(r"\s*import\b", bare, re.M):
        return ("`import` line inside lean_statement -- the harness supplies the "
                "preamble; send only the declaration")
    bad_arity = arity_lint(bare)
    if bad_arity:
        return bad_arity
    return ""


def _validate_signature(sig: str) -> tuple:
    """Typecheck a signature before ANY budget is spent on proving it.

    `derive_signature` has always validated its output; a caller-supplied
    `lean_statement` went straight through to the prover. That asymmetry cost
    2251 seconds on a statement written in Lean 3 syntax with invented constants
    and a `sorry` body -- it could never have compiled.

    A signature is good exactly when Lean's only complaint is the `sorry` we
    added: that means the STATEMENT elaborated and only the proof is missing.
    """
    last = {"ok": False, "errors": "empty signature", "elapsed": 0.0}
    for cand in normalize_signature(sig):
        verdict = check_lean(f"{PREAMBLE}{cand} := by sorry")
        errs = (verdict.get("errors", "") or "")
        if "sorry" in errs.lower() and not _ERROR_BESIDES_SORRY(errs):
            return {"ok": True, "elapsed": verdict.get("elapsed")}, cand
        last = {"ok": False, "errors": errs, "elapsed": verdict.get("elapsed")}
    return last, sig


def _premise_hint(statement: str, errors: str = "") -> str:
    """Real Mathlib names for this claim, searched on the claim AND on whatever
    the compiler said did not exist."""
    try:
        bad = _UNKNOWN.findall(errors or "") + re.findall(
            r"unknown (?:identifier|constant) '([^']+)'", errors or "")
        return mathlib_search.suggest_for_unknown(bad, statement).strip()
    except Exception:
        return ""


def _ERROR_BESIDES_SORRY(errors: str) -> bool:
    """True if Lean complained about something other than the injected `sorry`."""
    return any("error:" in ln and "sorry" not in ln.lower()
               for ln in (errors or "").splitlines())


_SIG = re.compile(r"^\s*(?:theorem|lemma)\s+[A-Za-z_][^\n]*", re.MULTILINE)

SIG_SCHEMA = {"type": "object",
              "properties": {"signature": {"type": "string"}},
              "required": ["signature"]}


def derive_signature(statement: str, tries: int = 3) -> str:
    """Derive a signature, retrying with the compiler error when it does not typecheck.

    Observed: for "sqrt 2 is irrational" the first attempt produced
    `¬ sqrt 2 ∈ Rat`, which is not well-typed. Feeding that error back is far
    cheaper than losing the pinned-signature fast path entirely.
    """
    feedback = ""
    for _ in range(max(1, tries)):
        sig = _derive_once(statement, feedback)
        if sig:
            return sig
        feedback = _LAST_SIG_ERROR.get("err", "")
    return ""


_LAST_SIG_ERROR: dict = {}


def _gen_timeout(budget: int) -> int:
    """Seconds to allow a generation of `budget` tokens.

    Derived, never hardcoded. The old fixed 900s meant the timeout, not the
    token budget, decided when a model stopped -- and only for models slower
    than budget/900 tok/s, so the handicap was invisible in any comparison
    against a fast one. Clamped to the problem deadline when a run has set one,
    so a single signature cannot eat a whole problem's time.
    """
    # Uncapped generation (budget <= 0) still needs a finite wall clock. The
    # real bound is the context window -- the model physically cannot emit more
    # than that -- so price the whole window at the floor throughput.
    tokens = budget if budget > 0 else config.CONTEXT_TOKENS
    secs = min(config.ORCH_TIMEOUT_CAP,
               max(300, int(tokens / max(config.ORCH_MIN_TOK_PER_SEC, 0.1))))
    if _PROBLEM_DEADLINE[0]:
        secs = max(60, min(secs, int(_PROBLEM_DEADLINE[0] - time.time())))
    return secs


def _think_opt() -> dict:
    """`think` for /api/chat, or {} to leave the model's own default alone."""
    mode = (config.ORCH_THINK or "").strip().lower()
    if not mode:
        return {}
    return {"think": False if mode in ("off", "false", "0", "none") else mode}


def _derive_once(statement: str, feedback: str = "") -> str:
    """Turn an informal claim into a pinned Lean signature.

    This is the highest-leverage step: a pinned signature unlocks BOTH the tactic
    pre-probe (7.7s vs 349.8s on sqrt-2) and proof-only mode, which removed the
    entire mis-formalization failure class and took P6 from 397.8s to 12.0s.
    """
    # Inject REAL Mathlib names before the model writes anything. The agent has
    # search_mathlib and never calls it: it wrote a signature around invented
    # `have_gcd`/`val` for a claim Mathlib already states as
    # `Nat.factorization_gcd`. Waiting for the compiler to reject an invented
    # name and only THEN searching is the loop this was built to break -- so
    # search up front, on the claim itself, rather than on the failure.
    # Stale stage from the PREVIOUS call would otherwise be reported as this
    # one's cause of death -- the same conflation this field exists to prevent.
    _LAST_SIG_ERROR.pop("stage", None)
    try:
        premises = mathlib_search.suggest_for_unknown([], statement)
    except Exception:
        premises = ""

    prompt = ("Write the Lean 4 (Mathlib) theorem signature for this claim: the "
              "declaration up to and including the final `:`, with NO `:=` and NO "
              "proof. Give the theorem a descriptive name.\n\n"
              f"CLAIM: {statement}\n\n"
              "Use Lean 4 syntax: `fun x => e` (NOT `λ x, e`), `let a := b;` (NOT "
              "`let a := b in`). Use real Mathlib definitions -- for p-adic "
              "valuations that is `Nat.factorization n p` or `padicValNat p n`, "
              "never a function you invent.\n\n"
              # The traps are stated BEFORE the model writes, not after the
              # compiler rejects. `|a - b|` over ℕ was the single most repeated
              # error in production and it is a SIGNATURE-time mistake: once it
              # is in the statement, no proof can rescue it.
              "ℕ IS NOT A GROUP. There is no negation and no `|·|` on ℕ, and "
              "subtraction is TRUNCATED (`3 - 5 = 0`). So:\n"
              "  - for |a - b| over naturals write `Nat.dist a b`, or "
              "`max a b - min a b`, or cast to ℤ with `((a : ℤ) - b)`\n"
              "  - never write `|a - b|`, `nat.abs`, or anything needing "
              "`AddGroup ℕ` / `Neg ℕ`\n"
              "  - prefer the concrete `Nat.gcd` / `Nat.lcm` over the "
              "`GCDMonoid`-generic `gcd` / `lcm`\n"
              "If the claim is about real numbers or requires negation, state it "
              "over ℤ or ℝ instead of ℕ.\n\n"
              # NAME, DO NOT UNFOLD. This is the top failure mode measured on
              # the Mathlib formalization benchmark: 6 of 7 genuinely wrong
              # statements came from hand-expanding a bundled notion or
              # retyping its context from the prose, and 5 of those COMPILED
              # while being false.
              #
              #   `U ∈ 𝓝[≠] x` unfolded by hand lost exactly the `∩ {x}ᶜ`
              #     factor -- a punctured filter silently became an ordinary
              #     neighbourhood, flipping a true theorem false at s = {x}
              #   `Summable f` rewritten as an equation between `tsum`s came
              #     out as `X = X`, provable by `rfl`
              #   `q.den` (canonical, reduced) replaced by raw `ℤ × ℕ` pairs
              #     made the fibres infinite, so a finiteness claim went false
              "NAME MATHLIB NOTIONS, NEVER UNFOLD THEM. If the claim mentions a "
              "notion Mathlib already bundles -- `Summable`, `𝓝[≠]`, "
              "`Nat.Primes`, `Rat.den`, `DifferentiableOn`, `Set.Finite`, "
              "`IsMaximal` -- use that head symbol VERBATIM. Do not expand it "
              "into its definition, do not re-encode it with your own indices "
              "or pairs, and do not swap a canonical form for a raw one. A "
              "hand-expansion that drops one factor still compiles and is "
              "false.\n\n"
              # The second mode: hypotheses that exist only in the Lean context,
              # never as English words, get dropped -- and because `deriv`,
              # `tsum` and a bare `Sub` are TOTAL functions with junk values,
              # the result compiles instead of erroring.
              "TAKE THE HYPOTHESES FROM MATHLIB, NOT FROM THE PROSE. Typeclass "
              "assumptions and side conditions are usually implied by the "
              "notions you used, not spelled out in English -- e.g. `OrderedSub` "
              "for truncated subtraction, `DifferentiableOn ℝ (deriv f)` for a "
              "second-derivative claim. Lean's `deriv`, `tsum` and `Sub` are "
              "TOTAL: omitting such a hypothesis does NOT cause an error, it "
              "produces a statement that compiles and is false. Inherit the "
              "WEAKEST context that supports the notions (do not demand "
              "`CommRing` where `Ring` suffices, or `LinearOrder` where "
              "`Preorder` does).\n\n"
              "SANITY CHECK BEFORE YOU ANSWER: if your statement could be "
              "closed by `rfl` or `simp` alone, or if both sides of an equation "
              "are the same expression, you have restated a definition rather "
              "than the claim. Start again.\n\n"
              "Example: theorem two_dvd_mul_succ (n : ℕ) : 2 ∣ n * (n + 1)"
              + premises
              + (f"\n\nYour previous attempt did NOT typecheck in Lean:\n{feedback[:500]}\n"
                 "Use correct Mathlib types and predicates (e.g. `Irrational (Real.sqrt 2)`)."
                 if feedback else ""))
    # THE budget bug, third occurrence. `num_predict: 3000` was hardcoded here
    # while a thinking orchestrator bills its reasoning against the same budget.
    # Measured on nemotron-3.5-lightning with this exact prompt:
    #
    #   num_predict= 3000 -> done_reason='length', content 0 chars, signature ''
    #   num_predict=16000 -> done_reason='stop',   content 88 chars, correct signature
    #
    # It did not produce a WRONG signature, it produced NOTHING -- 7 of 10
    # benchmark cases came back `empty`, SIG@1 = 20%. With no signature there is
    # no pinned statement, so no tactic probe and no proof-only mode, and the
    # prover is handed an informal claim to formalize itself: exactly the
    # mis-formalization class that costs 300-1800s a time. LIMITATIONS #3 was
    # fixed in `_call` and in `agent.call_orchestrator` and missed here, which is
    # why the system looked like it had a prover problem.
    # Uncapped (-1) passes straight through to ollama, which reads a negative
    # num_predict as "until context is full or the model stops".
    budget = -1 if config.ORCH_MAX_TOKENS < 0 else max(config.ORCH_MAX_TOKENS, 8000)
    # Specialist path: when FORMALIZER_BASE is set, the tuned 7B produces the
    # candidate line and every gate below (extract, typecheck, repair) is
    # unchanged -- so the specialist is judged by exactly the 27B's gates.
    from . import formalizer as _formalizer
    if _formalizer.available():
        cand = _formalizer.derive(statement)
        if cand:
            sig = _signature_line(cand)
            if sig:
                probe = _probe_statement(sig)
                if statement_ok(probe):
                    _LAST_SIG_ERROR.pop("err", None)
                    _LAST_SIG_ERROR.pop("stage", None)
                    return sig
                err = probe.get("errors", "") or ""
                patched = _repair_missing_instances(sig, err)
                if patched and statement_ok(_probe_statement(patched)):
                    return patched
                _LAST_SIG_ERROR["stage"] = "typecheck_failed"
                _LAST_SIG_ERROR["err"] = err
                _LAST_SIG_ERROR["sig"] = sig
        # Specialist produced nothing usable: fall through to the 27B so the
        # specialist can only ADD coverage, never subtract it.

    text, body, think = "", {}, _think_opt()
    for attempt in range(2):
        try:
            # Constrained decoding: the signature comes back in its own field
            # rather than needing to be scraped out of prose.
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content": prompt}], "stream": False,
                      "format": SIG_SCHEMA,
                      "options": {**config.ctx(config.ORCHESTRATOR_MODEL), "num_predict": budget, "temperature": _DERIVE_TEMP[0]},
                      # think=False, NOT `**think`. MEASURED, on the three
                      # formalization_v2 cases, three attempts each:
                      #
                      #   think=off    6/9 (67%)  mean   35s  range   22-57s
                      #   think=high   1/4 (25%)  mean  910s  range 224-2413s
                      #
                      # Reasoning and a `format` schema bill against ONE
                      # num_predict allowance, and the JSON is what loses: the
                      # 2413s and 2457s attempts each generated ~22,000 tokens
                      # and returned an empty signature. The rate difference is
                      # small-n and not the argument; the COST is, and it runs
                      # 26x the wrong way with a catastrophic tail.
                      #
                      # This also decides it for the scout, which only earns its
                      # place if observing is cheaper than committing -- against
                      # a 10-minute proof budget, a 40-minute observation is not
                      # an observation.
                      #
                      # Signatures produced with thinking off were checked to be
                      # FAITHFUL, not merely well-typed: the sum case returned
                      # `∑ i ∈ Finset.range n, (i+1) = n*(n+1)/2` and both
                      # triangle-inequality passes returned the same
                      # `‖x + y‖ ≤ ‖x‖ + ‖y‖`.
                      "think": False},
                timeout=_gen_timeout(budget))
            r.raise_for_status()
            body = r.json()
            text = str(json.loads(body["message"]["content"]).get("signature", "")).strip()
        except requests.Timeout:
            # A DISTINCT failure from "the model produced nothing": the model was
            # still generating and we hung up on it. Recording it as
            # generation_empty is what made a wall-clock ceiling look like a
            # capability limit for an entire benchmark run.
            _LAST_SIG_ERROR["stage"] = "timeout"
            body, text = {"done_reason": "timeout"}, ""
        except Exception:
            body, text = {}, ""
        if text:
            break
        if attempt == 0:
            # Truncated mid-thought: the budget is the problem, not the claim.
            # Meaningless when already uncapped -- there is nothing to double,
            # and `length` then means the CONTEXT filled, which more budget
            # cannot fix.
            if body.get("done_reason") == "length" and budget > 0:
                budget *= 2
                _trace(f"[sig] truncated at {budget // 2} tokens -- retrying at {budget}")
                continue
            # Ran out of WALL CLOCK, which is the opposite problem: a bigger
            # budget would only make it slower. Cap the reasoning instead --
            # what overran was the thinking, and this step needs recall more
            # than it needs deliberation.
            if body.get("done_reason") == "timeout" and not think:
                think = {"think": "low"}
                _trace("[sig] timed out mid-generation -- retrying with think=low")
                continue
        break

    if not text:                                    # fall back to free text
        try:
            r = requests.post(
                f"{config.OLLAMA_NATIVE}/api/chat",
                json={"model": config.ORCHESTRATOR_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False,
                      "options": {**config.ctx(config.ORCHESTRATOR_MODEL),
                                  "num_predict": budget, "temperature": _DERIVE_TEMP[0]}},
                timeout=_gen_timeout(budget))
            r.raise_for_status()
            m = r.json().get("message") or {}
            text = ((m.get("content") or "").strip() or (m.get("thinking") or "").strip())
        except requests.Timeout:
            _LAST_SIG_ERROR["stage"] = "timeout"
            return ""
        except Exception:
            return ""

    # WHY this returned nothing matters: "the model produced no text", "the
    # text held no theorem", and "the theorem did not typecheck" call for
    # opposite fixes (more budget / better prompt / better premises), and all
    # three used to surface identically as `empty`. That ambiguity produced two
    # wrong diagnoses in one session -- once blaming a token budget that was
    # fine, once blaming the model for what turned out to be a dice roll.
    if not (text or "").strip():
        _LAST_SIG_ERROR["stage"] = "generation_empty"
        return ""
    text = re.sub(r"```(?:lean4?)?", "", text)
    hits = _SIG.findall(text)
    if not hits:
        _LAST_SIG_ERROR["stage"] = "no_theorem_in_output"
        _LAST_SIG_ERROR["err"] = f"model wrote {len(text)} chars with no theorem: {text[:200]}"
        return ""
    sig = hits[-1].strip().split(":=")[0].rstrip()
    # Must typecheck as a statement before we pin it. `sorry` in the error is the
    # expected signal that the SIGNATURE is fine and only the proof is missing.
    probe = _probe_statement(sig)
    err = probe.get("errors", "") or ""
    if "sorry" in err.lower():
        _LAST_SIG_ERROR.pop("err", None)
        _LAST_SIG_ERROR.pop("stage", None)
        return sig
    # Before giving up, try adding the instance binders Lean named as missing.
    # This is the largest single derivation failure mode (56% of failures are
    # typeclass errors) and the repair is deterministic.
    # Single shot. Iterating was tried and MEASURED WORTHLESS: on eight real
    # rejected candidates, following the missing-instance chain for up to three
    # rounds fixed ZERO of them. The early `LE α` / `DecidableEq α` examples
    # that motivated this repair were not representative -- the bulk of the 56%
    # typeclass failures are not forgotten binders at all but wrong structures
    # over MetricSpace, cyclic groups, filters and the like, which no binder
    # addition can rescue.
    #
    # Kept anyway, because it is nearly free (one extra compile, fires on ~5% of
    # failures) and it does occasionally convert a rejection into a statement.
    # It is a small win, not the 25%-of-items lever it was hypothesised to be.
    patched = _repair_missing_instances(sig, err)
    if patched:
        probe2 = _probe_statement(patched)
        err2 = probe2.get("errors", "") or ""
        if "sorry" in err2.lower() and not _ERROR_BESIDES_SORRY(err2):
            _LAST_SIG_ERROR.pop("err", None)
            _LAST_SIG_ERROR.pop("stage", None)
            _trace(f"[prove] repaired signature by adding missing instances")
            # Mark the rescue in the trace. Attribution cannot be inferred from
            # a before/after comparison, because derivation is stochastic at
            # 40-67% per attempt -- a rerun rescues some items by luck alone.
            # Counting the times the repair itself turned a rejection into an
            # accepted statement needs no control group.
            _LAST_SIG_ERROR["repaired"] = True
            return patched

    _LAST_SIG_ERROR["stage"] = "typecheck_failed"
    _LAST_SIG_ERROR["err"] = err
    _LAST_SIG_ERROR["sig"] = sig
    return ""


_DECL_NAME = re.compile(r"^(\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?"
                        r"(?:theorem|lemma)\s+)([A-Za-z_][\w'\u2080-\u2089.]*)")


def _binder_section(sig: str) -> str:
    """Text before the conclusion. One implementation, in lean.py."""
    return split_signature(sig)[0]


_MISSING_INST = re.compile(
    r"failed to synthesize instance of type class\s*\n\s*([^\n]+)")


def _repair_missing_instances(sig: str, err: str) -> str:
    """Add instance binders Lean says are missing, when it is safe to do so.

    MEASURED: 56% of derivation failures are typeclass errors, and the most
    common shape is not a wrong class -- it is a class the model USED without
    DECLARING. It writes `≤` without `[LE α]`, or decidable equality without
    `[DecidableEq α]`, and Lean names the missing instance exactly:

        failed to synthesize instance of type class
          LE α

    That is a deterministic repair: splice `[LE α]` into the binder list and
    re-elaborate. No model call, one compile.

    CONSERVATIVE BY CONSTRUCTION. An instance is only added when every argument
    it mentions is ALREADY BOUND in the signature. That refuses the cases where
    the real fault is conceptual rather than a missing binder -- `SDiff (Type
    u_1)` (set difference applied to a Type), `HPow ℝ≥0 ℝ≥0 ?m.7` (unresolved
    metavariable), `OfNat (X i) 0` -- none of which a binder can fix.

    A repaired statement is not thereby FAITHFUL: adding `[LE α]` where the
    author meant `[PartialOrder α]` yields something well-formed but weaker.
    That is acceptable here because the alternative is an empty derivation,
    which scores zero and teaches nothing, whereas a weaker statement still
    faces the equivalence check.
    """
    if not sig or not err:
        return ""
    wanted, seen = [], set()
    for raw in _MISSING_INST.findall(err):
        cls = raw.strip().rstrip(",")
        if not cls or cls in seen:
            continue
        if "?" in cls or "✝" in cls:          # unresolved metavariable
            continue
        if len(cls) > 60 or "\n" in cls:
            continue
        # Greek letters MUST be in this character class: Lean type variables
        # are α, β, ι, and `[A-Za-z_]` matches none of them. With the ASCII-only
        # version, `LE α` yielded args=[] and every repair was silently refused.
        args = re.findall(r"[^\W\d_][\w'\u2080-\u2089]*", cls, re.UNICODE)[1:]
        if not args:                           # e.g. a bare `Nat` class: skip
            continue
        # Every argument must already be bound, or we would be inventing one.
        # The binder section ends at the first colon AT BRACKET DEPTH ZERO --
        # splitting on the first colon outright cuts inside `(α : Type*)` and
        # leaves `theorem t (α`, against which nothing matches and every repair
        # is refused. Caught by the unit cases, not by reading.
        binders = _binder_section(sig)
        if not all(re.search(r"\b" + re.escape(a) + r"\b", binders) for a in args):
            continue
        seen.add(cls)
        wanted.append(f"[{cls}]")
    if not wanted:
        return ""
    # Insert at the END of the binder section, NOT after the theorem name.
    # `theorem t [LE α] (α : Type*) ...` references α before it is bound and
    # cannot compile -- so every repair would have failed while the unit tests,
    # which only asked whether a repair was ATTEMPTED, passed.
    cut = len(_binder_section(sig))
    head, tail = sig[:cut].rstrip(), sig[cut:]
    return head + " " + " ".join(wanted) + " " + tail.lstrip()

def _probe_statement(sig: str) -> dict:
    """Typecheck a candidate SIGNATURE, immune to what it calls itself.

    The declared name is renamed to a fixed private one first. Without this, a
    model that names its theorem sensibly -- i.e. the same thing Mathlib already
    calls it -- is rejected with "`catalan_eq_centralBinom_div` has already been
    declared", and a CORRECT statement scores as a derivation failure. Observed
    three times in 77 recorded failures (`catalan_eq_centralBinom_div`,
    `Ideal.FG.map`, and one more), and it can only get worse as the library
    grows, because the better the name the likelier the clash.

    The name has no bearing on whether the statement elaborates, so renaming
    costs nothing and removes the whole failure class.
    """
    probe = _DECL_NAME.sub(lambda m: m.group(1) + "_sigprobe", sig, count=1)
    return check_lean(f"{PREAMBLE}{probe} := by sorry")


def _signature_line(text: str) -> str:
    """The bare `theorem ... : ...` line from a specialist that emits only that.

    Deliberately NOT _extract_lean: that function targets prose containing code
    fences and supplies imports when it finds none, so a one-line signature
    comes back with the preamble glued on. _probe_statement then adds the
    preamble a second time and appends `:= by sorry` to a whole file.
    """
    t = (text or "").strip()
    t = re.sub(r"^```(?:lean4?)?\s*|\s*```$", "", t).strip()
    for ln in t.splitlines():
        ln = ln.strip()
        if re.match(r"(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?"
                    r"(?:theorem|lemma)\b", ln):
            return re.sub(r"\s*:=.*$", "", ln).strip()
    return ""


def _extract_lean(text: str, narrow: bool = True) -> str:
    """Pull the last declaration-bearing code block; supply imports if absent.

    Narrow imports cost 1.9s/1.3 GB versus 7.4s/5.7 GB for `import Mathlib`
    (measured). `check_and_fix` falls back to the full import if narrow misses
    something, so speed never costs correctness.
    """
    blocks = re.findall(r"```(?:lean4?)?\s*\n(.*?)```", text, re.DOTALL)
    candidates = [b for b in blocks if _HAS_DECL.search(b)] or blocks
    code = (candidates[-1] if candidates else text).strip()
    if code and not _HAS_IMPORT.search(code):
        head = ""
        if narrow:
            try:
                head = mathlib_search.narrow_imports(code)
            except Exception:
                head = ""
        code = (f"{head}\nset_option maxHeartbeats 400000\n\n{code}" if head
                else PREAMBLE + code)
    return code


_CASCADE = re.compile(r"unknown tactic|unsolved goals|sorry", re.I)

# Structural facts about ℕ that the prover kept rediscovering the hard way. Each
# entry cost real rounds in run-20260813-220855: `AddGroup ℕ` twice, `nat.abs`
# (a Lean 3 name for something ℕ does not have) once, and several attempts to
# write |a-b| over ℕ. The compiler says what is wrong; it never says what to do
# instead, and the prover has no way to learn it between attempts.
_TRAPS = (
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


_FAILED: dict = {}
_PROBLEM_DEADLINE = [0.0]
# The problem currently being worked on. The quality-diversity archive needs it
# to tell a RESTATEMENT of the goal apart from a genuine sub-lemma -- the same
# claim is a dead end for one problem and real progress on another.
_GOAL = [""]
# The claim currently being proved. `_remember` is called from five return
# points that only carry the failure key, so the statement is stashed here
# rather than threaded through every one of them.
_CURRENT = ["", "", ""]
# Sampling temperature for SIGNATURE derivation, overridable so the stability
# harness can measure the noise floor. 0.2 is the production value: low enough
# to be mostly reproducible, non-zero enough that every case is a coin of
# unknown bias -- which is exactly why a single PASS/FAIL flip proves nothing.
_DERIVE_TEMP = [0.2]


def set_problem_deadline(ts: float) -> None:
    """Wall-clock instant past which no proof attempt may run. 0 disables."""
    _PROBLEM_DEADLINE[0] = ts or 0.0


def set_goal(statement: str) -> None:
    """The problem under attack, for archive role inference. "" disables."""
    _GOAL[0] = statement or ""


def _fail_key(statement: str, lean_statement: str = "") -> str:
    """Identity of an attempt: the maths, not the phrasing.

    Normalized so that re-typing the same claim with different whitespace or
    capitalisation is recognised as the repeat it is.
    """
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    return f"{norm(statement)}||{norm(lean_statement)}"


def _remember(key: str, out: dict) -> dict:
    """Record a failure so the same claim is not paid for twice in one session."""
    _FAILED[key] = {k: v for k, v in out.items() if k != "attempts"}
    # Failures belong in the archive too: they mark a cell as ATTEMPTED without
    # filling it, which is exactly the information "what have I not explored"
    # needs. Counting only successes would make an expensively-failed region
    # look untouched and invite the same dead end again.
    try:
        from . import archive
        if _CURRENT[0]:
            archive.record(_CURRENT[0], ok=False, seconds=out.get("elapsed", 0.0),
                           code=_CURRENT[1], goal=_GOAL[0])
    except Exception:
        pass
    return out


def reset_failures() -> None:
    """Clear the per-session failure memory. Called when a new problem opens."""
    _FAILED.clear()


def _trap_hint(errors: str) -> str:
    """Turn a recurring compiler complaint into the fix it never states."""
    hits = [msg for rx, msg in _TRAPS if rx.search(errors or "")]
    return ("\n\nIMPORTANT:\n" + "\n".join(f"- {h}" for h in hits)) if hits else ""


def _headline(errors: str) -> str:
    """The most INFORMATIVE compiler error, not the first one.

    Lean reports the tactic-block failure before the elaboration failure that
    caused it. Printing line one gave `unknown tactic` for 35 straight rounds
    while the actual message, three lines down, was `failed to synthesize
    GCDMonoid ℕ` -- a missing import. The trace named a symptom and hid the
    cause, and the strategy ladder was steered by the symptom.
    """
    lines = [l.strip() for l in (errors or "").splitlines() if l.strip()]
    real = [l for l in lines if "error" in l.lower() and not _CASCADE.search(l)]
    pick = (real or lines or ["?"])[0]
    # Instance failures put the class on the FOLLOWING line -- carry it along.
    if "synthesize" in pick.lower():
        i = lines.index(pick)
        if i + 1 < len(lines):
            pick = f"{pick} {lines[i + 1]}"
    return re.sub(r"^tmp\w+\.lean:", "", pick)[:110]


def check_and_fix(code: str) -> tuple:
    """Compile; if narrow imports were the problem, retry with full Mathlib.

    Without this, a missing import is indistinguishable from a wrong proof.
    """
    verdict = check_lean(code)
    if verdict["ok"] or "import Mathlib\n" in code:
        return code, verdict
    # A MISSING INSTANCE is an import problem too, and it was not in this list.
    # Measured cost of the omission: 35 consecutive rounds across 3 claims and 5
    # strategies, every one dying on `failed to synthesize GCDMonoid ℕ` because
    # bare `gcd` needs the instance that a narrow import had left out. The
    # retry-with-full-Mathlib that exists precisely for this never fired, and the
    # first error line said `unknown tactic` -- so it read as a broken proof.
    if re.search(r"[Uu]nknown (constant|identifier)|unknown namespace"
                 r"|synthInstanceFailed|failed to synthesize"
                 r"|[Ff]unction expected at", verdict["errors"]):
        body = re.sub(r"^\s*(import .*|set_option .*)$\n?", "", code, flags=re.MULTILINE)
        full = PREAMBLE + body.strip()
        v2 = check_lean(full)
        if v2["ok"]:
            return full, v2
    return code, verdict


def _fit(base: str, model: str, messages: list, max_tokens: int) -> int:
    """`max_tokens`, clamped to what the server's context has left after the prompt.

    vLLM rejects (HTTP 400) any request whose prompt plus max_tokens exceeds
    max_model_len. The truncation retry doubled 16k to 28k against a 16k window,
    so a reply that was merely cut off became a dead attempt ("prover error"), and
    a 16k batch budget plus its prompt can sink all n best-of-n samples at once.
    vLLM's /tokenize reports both numbers; a server without it (Ollama) passes
    through unclamped.
    """
    try:
        r = requests.post(re.sub(r"/v1/?$", "", base) + "/tokenize",
                          json={"model": model, "messages": messages}, timeout=10)
        r.raise_for_status()
        j = r.json()
        return max(1, min(max_tokens, j["max_model_len"] - j["count"]))
    except Exception:
        return max_tokens


def _call(messages: list, temperature: float, base: str, model: str,
          max_tokens: int = None, timeout: int = 900) -> tuple:
    """Returns (text, truncated). Truncation must be visible to the caller.

    A proof cut off at max_tokens yields a partial code block that fails with a
    SYNTAX error -- indistinguishable from a mathematical failure unless we
    check finish_reason here.
    """
    r = requests.post(
        f"{base}/chat/completions",
        json={"model": model, "messages": messages,
              "max_tokens": _fit(base, model, messages, max_tokens or config.PROVER_MAX_TOKENS),
              "temperature": temperature},
        timeout=timeout,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    msg = choice["message"]
    # Goedel-style provers put the whole answer in `reasoning`, leaving `content`
    # empty. Reading only `content` yields "" -> an empty file -> a vacuous pass.
    text = (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()
    return text, choice.get("finish_reason") == "length"


_BATCH_CHUNK = 4         # samples per request in sample_proofs


def sample_proofs(lean_statement: str, n: int, base: str, model: str,
                  temperature: float = 1.0, timeout: int = 900) -> list:
    """N independent proofs of ONE pinned goal, in ONE request.

    This is the change the whole architecture was missing. The current design is
    pass@1 repeated serially -- 5 attempts x 6 rounds, each a separate round
    trip -- and Goedel-V2-32B's headline 88.1% on MiniF2F is **pass@32**. The
    goal was never sampled more than once at a time, so the reported capability
    was never available to us.

    Temperature is deliberately HIGH. The compiler is a perfect filter: a wrong
    sample costs one compile and nothing else, so diversity is worth more than
    per-sample precision. We do not need the model to be right, only to be right
    once out of n.

    Requires a server that honours `n` (vLLM does; Ollama does not), so this is
    inert until PROVER_BATCH_BASE is configured. Returns [] when unavailable, and
    the caller falls back to the sequential path.
    """
    if not (base and model and n > 1):
        return []
    prompt = (f"{BASE_SYSTEM}\n\nProve this Lean 4 theorem. Keep the signature "
              f"VERBATIM and supply only the proof:\n\n{lean_statement}")
    messages = [{"role": "user", "content": prompt}]
    max_tokens = _fit(base, model, messages, config.PROVER_MAX_TOKENS)

    def chunk(k):
        r = requests.post(
            f"{base}/chat/completions",
            json={"model": model, "messages": messages, "n": k, "temperature": temperature,
                  "top_p": 0.95, "max_tokens": max_tokens},
            timeout=timeout)
        r.raise_for_status()
        return [(m.get("content") or "").strip() or (m.get("reasoning") or "").strip()
                for m in (ch.get("message", {}) for ch in r.json().get("choices", []))]

    # n as concurrent chunks, not one all-or-nothing request: the server runs the
    # same samples either way, but when the slowest overrun the timeout, the
    # finished ones -- correct proofs among them -- are kept instead of lost.
    sizes = [min(_BATCH_CHUNK, n - i) for i in range(0, n, _BATCH_CHUNK)]
    out, errors = [], []
    with futures.ThreadPoolExecutor(len(sizes)) as pool:
        for fut in futures.as_completed([pool.submit(chunk, k) for k in sizes]):
            try:
                out += [t for t in fut.result() if t]
            except Exception as exc:
                errors.append(exc)
    if not out:
        _trace(f"[prove] batch sampling unavailable ({errors[0] if errors else 'empty'})"
               " -- sequential fallback")
        return []
    _trace(f"[prove] sampled {len(out)}/{n} candidate proofs"
           + (f" ({len(errors)} of {len(sizes)} requests failed: {errors[0]})" if errors else ""))
    return out


def first_compiling(candidates: list, workers: int = None) -> tuple:
    """Compile candidates concurrently; the first that passes wins.

    Sampling is now cheap and COMPILING is the bottleneck, which is the whole
    reason the Lean farm has to live where there is RAM. Returns (code, verdict)
    or (None, None).
    """
    workers = workers or max(1, tactics.WORKERS)
    codes = [_extract_lean(c) for c in candidates]
    codes = [c for c in codes if c.strip()]
    if not codes:
        return None, None
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(check_and_fix, c): c for c in codes}
        try:
            for fut in futures.as_completed(jobs):
                try:
                    code, verdict = fut.result()
                except Exception:
                    continue
                if verdict.get("ok"):
                    for other in jobs:
                        other.cancel()
                    return code, verdict
        finally:
            pass
    return None, None


def _attempt(statement: str, temperature: float, hint: str, rounds: int,
             deadline: float, base: str, model: str, mode: str = None,
             avoid: list = None) -> dict:
    """One independent attempt: fresh context, up to `rounds` self-corrections."""
    avoid = avoid or []
    system = BASE_SYSTEM + (f"\n\nSTRATEGY: {hint}" if hint else "")

    extra = ""
    if avoid:
        # Naming the exhausted approaches is what stops attempt N repeating
        # attempt 1 in different words.
        extra += ("\n\nApproaches already tried and FAILED -- do NOT repeat them:\n"
                  + "\n".join(f"- {a}" for a in avoid[-5:]))
    if mode == "sketch":
        plan = _sketch(statement, avoid)
        if plan:
            extra += f"\n\nA proof plan from another model -- follow it:\n{plan}"
    elif mode == "search":
        hits = mathlib_search.suggest_for_unknown([], statement)
        if hits:
            extra += hits

    # What have WE already proved that bears on this goal? The library was
    # importable but not discoverable, so the system reproved its own results
    # instead of building on them -- the reason a growing library was not
    # actually compounding. Offered on every attempt, not just `search` mode:
    # a lemma we own is cheaper than any lemma we have to find.
    try:
        own = library.suggest(statement)
        if own:
            extra += own
    except Exception:
        pass

    messages = [{"role": "system", "content": system},
                {"role": "user", "content":
                    f"Prove in Lean 4 with Mathlib:\n\n{statement}{extra}"}]
    banned: list = []
    last_sig = ""
    log = []

    budget = config.PROVER_MAX_TOKENS
    truncations = 0
    rnd = 0
    while rnd < rounds:
        rnd += 1
        left = deadline - time.time()
        if left <= 0:
            return {"ok": False, "reason": "time budget", "rounds": rnd - 1, "log": log}
        try:
            # Cap the HTTP wait at the remaining budget. A fixed 900s timeout let
            # one call run past the deadline: a 1800s budget overshot to 2251s.
            raw, truncated = _call(messages, temperature, base, model, budget,
                                   timeout=max(30, min(900, int(left) + 30)))
        except Exception as exc:
            return {"ok": False, "reason": f"prover error: {exc}", "rounds": rnd - 1, "log": log}

        if truncated:
            # A cut-off reply is not a failed ROUND -- nothing was compiled, so
            # nothing was learned. Charging it against the self-correction budget
            # is how one claim burned 2251s across 5 rounds without Lean ever
            # seeing a single generated proof. Give it more room and retry
            # WITHOUT consuming a round; bail after a few so it cannot loop.
            truncations += 1
            rnd -= 1
            budget = min(budget * 2, config.PROVER_MAX_TOKENS_CAP)
            _trace(f"  generation truncated ({truncations}) -- raising budget to "
                   f"{budget} tokens and retrying (round not consumed)", indent=7)
            log.append({"round": rnd + 1, "ok": False, "truncated": True,
                        "errors": f"TRUNCATED at max_tokens; budget -> {budget}"})
            if truncations >= config.PROVER_MAX_TRUNCATIONS:
                return {"ok": False, "reason":
                        f"output truncated {truncations}x even at {budget} tokens -- "
                        f"the claim is too large to prove in one block; decompose it",
                        "rounds": rnd, "log": log}
            messages.append({"role": "user", "content":
                             "Your reply was cut off before the code block ended. Answer "
                             "again and be MUCH shorter: skip the prose, emit only the "
                             "final ```lean4 block."})
            continue

        code, verdict = check_and_fix(_extract_lean(raw))
        _trace(f"  round {rnd}/{rounds}: lean {'OK' if verdict['ok'] else 'rejected'} "
               f"({verdict.get('elapsed', 0)}s)"
               + ("" if verdict["ok"] else f" -- {_headline(verdict['errors'])}"),
               indent=7)
        if verdict["ok"] and config.AXIOM_AUDIT:
            # Compiling is not the same as proving. `sorry` and empty files were
            # already blocked; this blocks the next tier -- a locally declared
            # `axiom`, or `native_decide`, which bypasses the kernel. Fails open.
            audit = audit_axioms(code)
            if not audit["ok"]:
                verdict = {"ok": False, "errors": audit["errors"],
                           "elapsed": verdict.get("elapsed", 0.0)}
        # 300 chars used to be the whole record of a round, and it was not enough
        # to tell one failure from another later. This goes to the JOURNAL, not
        # to the model's context (see the `_`-prefix convention below), so it can
        # afford to be the detail that saves the next session a repeat.
        log.append({"round": rnd, "ok": verdict["ok"], "errors": verdict["errors"][:1500],
                    "elapsed": verdict.get("elapsed"),
                    # Keep the code on FAILURE too. It was dropped, so all 35
                    # failed rounds of run-20260813-212005 recorded an empty
                    # `lean` -- the one artifact that would have shown the
                    # missing import immediately. The record exists for the
                    # rounds that went wrong, not the ones that went right.
                    "lean": code})
        if verdict["ok"]:
            return {"ok": True, "lean": code, "rounds": rnd, "log": log,
                    "approach": _signature(code)}

        sig = re.sub(r"tmp\w+\.lean", "", verdict["errors"])[:300]
        if sig and sig == last_sig:
            # Feedback is not landing. Abandon this context -- the caller starts a
            # fresh attempt with a different strategy rather than giving up.
            return {"ok": False, "reason": "stalled", "rounds": rnd, "log": log}
        last_sig = sig

        new_bad = [b for b in _UNKNOWN.findall(verdict["errors"]) if b not in banned]
        banned.extend(new_bad)
        ban = (f"\n\nThese constants DO NOT EXIST in Mathlib -- never use them again: "
               f"{', '.join(banned)}." if banned else "")
        # Don't just say "wrong" -- say what IS right. Fires automatically on any
        # unknown constant, so the model never has to think to ask.
        suggestions = (mathlib_search.suggest_for_unknown(new_bad, statement)
                       if new_bad else "")
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"The Lean compiler rejected that:\n\n"
                                        f"{verdict['errors'][:1500]}{ban}{suggestions}"
                                        f"{_trap_hint(verdict['errors'])}\n\n"
                                        "Fix it and output the corrected complete Lean 4 code block."},
        ]

    return {"ok": False, "reason": "rounds exhausted", "rounds": rounds, "log": log}


def prove_theorem(statement: str, lean_statement: str = "", attempts: int = None,
                  rounds: int = None, time_budget: int = None,
                  obligation: str = "") -> dict:
    """Prove a claim in Lean, escalating across independent attempts.

    `lean_statement` (optional) pins the exact Lean theorem signature. When given,
    the prover supplies ONLY the proof, which makes mis-formalization structurally
    impossible and skips the review gate entirely -- prefer it whenever you can
    write the statement yourself.

    Returns proved=True only if the compiler accepted a complete proof AND (in
    informal mode) the statement passed back-translation review.
    """
    attempts = config.PROVE_ATTEMPTS if attempts is None else attempts
    rounds = rounds or config.PROVE_ROUNDS
    # Never run past the PROBLEM's deadline. Both P4 proof calls took the full
    # 1800s PROVE_TIME_BUDGET, and the second started with minutes left -- so the
    # run overran its 90-minute budget and the step budget could never be spent
    # on anything else. A proof budget is a cap, not an entitlement.
    # Refuse a claim the sandbox can refute. The scout falsifies plan
    # obligations, but prove_theorem is a TOOL the agent calls on anything it
    # likes, so a false claim otherwise walks straight into a 13-minute average
    # attempt -- and, before the faithfulness gate existed, into a banked proof
    # of a false statement. ~20s against 780s, and the result is memoised so a
    # scouted claim costs nothing here.
    if config.GT_FALSIFY and not lean_statement:
        try:
            from . import falsify as _falsify
            _f = _falsify.probe(statement)
        except Exception:
            _f = {}
        if _f.get("status") == "falsified":
            return {"proved": False, "attempts": [], "elapsed": 0.0,
                    "refuted": True,
                    "note": ("This claim is FALSE -- a counterexample was computed: "
                             + str(_f.get("counterexample", ""))[:200]
                             + ". Check those values against the claim. If they "
                               "hold, do not attempt a proof; restate the claim "
                               "or drop it.")}

    deadline = time.time() + (time_budget or config.PROVE_TIME_BUDGET)
    if _PROBLEM_DEADLINE[0]:
        deadline = min(deadline, _PROBLEM_DEADLINE[0])
        left = deadline - time.time()
        if left < 60:
            return {"proved": False, "attempts": [], "elapsed": 0.0,
                    "note": (f"Only {int(max(0, left))}s left in this problem's time "
                             "budget -- not enough to attempt a proof. Record what you "
                             "have and close the session.")}
        _trace(f"[prove] budget clamped to {int(left)}s left of the problem's time")
    started = time.time()
    history = []

    _trace(f"[prove] {statement.splitlines()[0][:100]}")

    # A claim that already failed this session fails the same way again. On
    # IMO26_P4 the SAME statement -- "if angle BAP = 90 - angle ABC then angle
    # APB = 90" -- was submitted three times and cost 1829.9s + 269.4s + 1829.2s:
    # 65 of the run's 90 minutes, for one claim, with zero new information. The
    # cache only remembers successes, so nothing stopped it. Operating rule 3
    # already forbids this; the harness has to enforce it.
    # Already proved? Say so instead of proving it again. Across 26 prove calls
    # the ONLY four successes were the same divisibility fact reworded four
    # ways -- the agent kept returning to the one claim it could formalize
    # cleanly, and each success looked like progress. Retreating to a solved
    # trivial lemma is the local minimum this system actually falls into.
    from . import journal as _journal
    already = _journal.known(statement)
    if already:
        _trace("[prove] ALREADY ESTABLISHED -- refusing to re-prove it")
        return {"proved": False, "attempts": [], "elapsed": 0.0, "redundant": True,
                "established": already,
                "note": ("This is already established and needs no reproof:\n  "
                         f"{already}\nBuild on it. Spend this turn on a claim that is "
                         "NOT yet known -- look at what the problem still requires and "
                         "prove the next missing link, not a fact you already hold.")}

    # journal.known() catches the same FACT reworded. This catches the same
    # KIND of result -- a different fact that nonetheless explores nothing new.
    # Both guards are needed: the five divisibility restatements were caught by
    # neither, because each was worded differently enough to look novel while
    # being behaviourally identical.
    try:
        from . import archive as _archive
        occ = _archive.occupied(statement, lean_statement, goal=_GOAL[0])
        if occ.get("filled"):
            _trace(f"[prove] cell {occ.get('cell')} already filled -- refusing")
            return {"proved": False, "attempts": [], "elapsed": 0.0, "redundant": True,
                    "note": ("A result of this KIND is already established:\n  "
                             f"{occ['by']}\nProving another one explores nothing new. "
                             + _archive.advice(_GOAL[0]))}
    except Exception:
        pass

    key = _fail_key(statement, lean_statement)
    _CURRENT[0], _CURRENT[1] = statement, lean_statement
    # Which checklist item this serves, NAMED rather than guessed. Fuzzy text
    # matching could not connect a helper lemma to its obligation: the agent
    # spent 35 minutes on "the exponent of p in the prime factorization of
    # gcd(m,n)" -- genuine work supporting `potential_decrease` -- and the
    # ledger scored the overlap against "product / blackboard / invariant /
    # move" below threshold and recorded nothing. Progress read as zero, the
    # stuck counter never reset, and closure could never fire.
    _CURRENT[2] = obligation or ""
    prior = _FAILED.get(key)
    if prior is not None:
        _trace("[prove] ALREADY FAILED this session with this exact statement "
               "-- returning the earlier result instead of re-spending the budget")
        return dict(prior, repeated=True,
                    note=("You already submitted this exact claim and signature this "
                          "session; it failed for the reason below and would fail "
                          "identically again. Change the mathematics, not the wording: "
                          "prove a weaker or special case, pick a different sub-lemma, "
                          "or pin a different signature. Original reason: "
                          + str(prior.get("note") or "see attempts")))

    hit = cache.get(statement, lean_statement)
    if hit.get("lean"):
        _trace("[prove] cache HIT -- returning the stored proof")
        return {"proved": True, "lean": hit["lean"], "attempts": [], "rounds": 0,
                "elapsed": 0.0, "solved_by": "cache",
                "formalization": {"faithful": True, "verdict": "cached"}}

    # Derive a signature when the caller did not pin one: it unlocks the cheap
    # tactic probe and removes the mis-formalization class entirely.
    if not lean_statement and config.SIGNATURE_FIRST:
        _trace("[prove] deriving a Lean signature (model call + typecheck, up to 3 tries)...")
        t0 = time.time()
        derived = derive_signature(statement)
        _trace(f"[prove] signature {'pinned' if derived else 'NOT derived'} "
               f"in {time.time() - t0:.0f}s" + (f": {derived[:90]}" if derived else ""))
        if derived:
            review = review_statement(statement, derived + " := by sorry")
            if review.get("faithful") is not False:
                lean_statement = derived
                history.append({"attempt": 0, "model": "signature", "mode": "derive",
                                "ok": True, "reason": f"pinned: {derived[:90]}", "rounds": 0})

    if lean_statement:
        # VALIDATE a caller-supplied signature before spending anything on it.
        # The derive path has always typechecked its output; this path never did,
        # so whatever the orchestrator wrote went straight to the prover. It sent
        #     theorem q_invariant ... : let g_p := λ p, have_gcd nums [...] in ... sorry
        # -- Lean 3 syntax, invented constants, and a `sorry` body -- and the
        # system spent 2251 seconds failing to prove something that could never
        # compile. Five seconds of typechecking buys that back.
        junk = _syntax_reject(lean_statement)
        if junk:
            _trace(f"[prove] REJECTED the supplied lean_statement on sight: {junk}")
            check = {"ok": False, "errors": junk, "elapsed": 0.0}
        else:
            check, lean_statement = _validate_signature(lean_statement)
        if not check["ok"]:
            if not junk:
                _trace(f"[prove] REJECTED the supplied lean_statement in "
                       f"{check.get('elapsed')}s -- it does not typecheck")
            derived = derive_signature(statement) if config.SIGNATURE_FIRST else ""
            if not derived:
                return _remember(key, {"proved": False, "attempts": history, "elapsed": round(time.time() - started, 1),
                        "lean_statement_rejected": True,
                        "errors": (check.get("errors", "") or "")[:1200],
                        "note": ("Your lean_statement does not typecheck, so nothing was "
                                 "attempted. Fix it and resubmit, or omit lean_statement "
                                 "and let the prover derive one. Common causes: Lean 3 "
                                 "syntax (`λ x, e` and `let .. in` are Lean 4 `λ x => e` "
                                 "and `let .. ;`), invented constants, or a proof body "
                                 "where only a signature was asked for -- end at the "
                                 "final `:`."),
                        # Don't tell it to "check with search_mathlib" -- it never
                        # does. Hand it the real names with the error.
                        "mathlib_candidates": _premise_hint(statement,
                                                            check.get("errors", ""))})
            _trace(f"[prove] recovered by deriving one instead: {derived[:90]}")
            lean_statement = derived
            history.append({"attempt": 0, "model": "signature", "mode": "recover",
                            "ok": True, "reason": "supplied signature rejected; derived one",
                            "rounds": 0, "_rejected": check.get("errors", "")[:600]})
        else:
            _trace(f"[prove] supplied signature typechecks ({check.get('elapsed')}s)")

        # Let Lean try first: `exact?` searches Mathlib natively and costs no GPU.
        # Measured 6.7s vs 349.8s on sqrt-2. Only spend prover attempts if it misses.
        #
        # The probe's budget is a SHARE of the time left, not a fixed 240s. With
        # TACTIC_WORKERS=1 the twelve rungs run sequentially, so on a hard goal
        # the probe hit its default cap and left almost nothing behind it:
        # measured on an AM-GM claim with a 420s budget, the probe consumed 330s
        # and the strategy ladder started with 90s -- one attempt, on a goal that
        # needed several. Cheap-and-often-decisive earns real time; it does not
        # earn most of the budget.
        probe_budget = max(30.0, min(240.0, (deadline - time.time()) * 0.4))
        _trace(f"[prove] tactic probe: {len(tactics.LADDER)} rungs, "
               f"{tactics.WORKERS} workers, {int(probe_budget)}s budget, "
               f"each pays `import Mathlib`...")
        hit = tactics.probe(lean_statement, budget=probe_budget)
        _trace(f"[prove] tactic probe {'SOLVED via ' + hit['tactic'] if hit['ok'] else 'found nothing'}"
               f" ({hit['elapsed']}s, {len(hit['tried'])} rungs completed)")
        if hit["ok"]:
            # This is the path that fired on both of the run's "successes", and
            # it skipped review because the signature was pinned. `exact?`
            # succeeding means Mathlib already had the theorem -- which says the
            # PROOF is fine and nothing at all about whether the signature
            # expresses the claim. Review it like any other compiling proof.
            _trace("[prove] tactic solved it -- back-translating for faithfulness...")
            review = review_statement(statement, hit["lean"])
            _trace(f"[prove] review: {str(review.get('verdict', ''))[:100]}")
            if review.get("faithful") is False:
                return _remember(key, {"proved": False, "attempts": history,
                        "elapsed": hit["elapsed"], "formalization_suspect": True,
                        "lean": hit["lean"], "formalization": review,
                        "solved_by": f"tactic:{hit['tactic']}",
                        "note": ("A tactic proved the pinned signature, but the "
                                 "signature does not express your claim: "
                                 + str(review.get("verdict", ""))
                                 + " Narrow the claim, or pin a signature covering it.")})
            out = {"proved": True, "lean": hit["lean"], "attempts": history,
                   "rounds": 0, "elapsed": hit["elapsed"],
                   "solved_by": f"tactic:{hit['tactic']}",
                   "formalization": review}
            _record(statement, lean_statement, out)
            return out
        history.append({"attempt": 0, "model": "lean-tactics", "mode": "probe",
                        "ok": False, "reason": f"no tactic worked ({len(hit['tried'])} tried)",
                        "rounds": 0, "_tried": hit["tried"],
                        "_not_run": hit.get("not_run", [])})
        statement = (f"{statement}\n\nProve EXACTLY this Lean 4 theorem, keeping the "
                     f"signature verbatim and supplying only the proof:\n\n{lean_statement}"
                     f"\n\nThese tactics already FAILED, do not retry them alone: "
                     f"{', '.join(hit['tried'])}")

    # BEST-OF-N, before the sequential ladder. One batched request beats the
    # 5x6 attempt matrix on both axes: it is pass@n rather than pass@1 repeated,
    # and it costs one round trip instead of thirty. Inert until a vLLM endpoint
    # is configured, so nothing changes on the Mac-only setup.
    if lean_statement and config.PROVER_BATCH_BASE and config.PROVER_BATCH_MODEL:
        _prover_target()            # the batch shares the server: bring it up first
        n = config.PROVER_BATCH_N
        _trace(f"[prove] best-of-{n}: sampling in one batched request...")
        t0 = time.time()
        cands = sample_proofs(lean_statement, n, config.PROVER_BATCH_BASE,
                              config.PROVER_BATCH_MODEL)
        if cands:
            _trace(f"[prove] compiling {len(cands)} candidates concurrently...")
            code, verdict = first_compiling(cands)
            took = round(time.time() - t0, 1)
            history.append({"attempt": 0, "model": config.PROVER_BATCH_MODEL,
                            "mode": f"best-of-{n}", "ok": bool(code),
                            "reason": f"{len(cands)} sampled", "rounds": 1})
            if code:
                _trace(f"[prove] best-of-{n} SOLVED in {took}s")
                review = review_statement(statement, code)
                if review.get("faithful") is not False:
                    out = {"proved": True, "lean": code, "attempts": history,
                           "rounds": 1, "elapsed": took,
                           "solved_by": f"best-of-{n}", "formalization": review}
                    _record(statement, lean_statement, out)
                    return out
                _trace("[prove] best-of-N proof MIS-FORMALIZED -- not counted")
            else:
                _trace(f"[prove] best-of-{n} found nothing in {took}s")

    avoid: list = []
    used: list = []
    # Bandit stats are kept PER DOMAIN: a strategy that wins on divisibility can
    # be useless on compactness, and a single global ranking would be wrong half
    # the time. The archive's object tag is the domain label.
    try:
        from . import archive as _arch
        _domain = _arch.cell(statement, lean_statement, goal=_GOAL[0])[1]
    except Exception:
        _domain = "general"
    i = 0
    # attempts <= 0 means "keep trying until the time budget is gone".
    while (attempts <= 0 or i < attempts) and time.time() < deadline:
        # WHICH strategy, chosen by measured success rather than by position in
        # a hard-coded list. Attempts cost minutes each, so spending them in a
        # fixed order regardless of what has been working is the most expensive
        # guess in the system. Arms already used in THIS call are excluded --
        # within one proof we want diversity; across calls the bandit learns.
        try:
            from . import bandit as _bandit
            name = _bandit.select(_domain, arms=STRATEGY_NAMES, exclude=tuple(used))
        except Exception:
            name = STRATEGY_NAMES[i % len(STRATEGY_NAMES)]
        used.append(name)
        if len(used) >= len(STRATEGY_NAMES):
            used.clear()                     # a full cycle: everything is fair again
        _, temp, hint, mode = _BY_NAME.get(name, STRATEGIES[0])
        # Past one full cycle of strategies, push temperature up so repeats
        # of the same strategy still explore somewhere new -- but cap at 1.0.
        # Above that a prover mostly emits syntactically broken Lean, so the
        # extra "diversity" is just wasted compiles. Diversify with strategies,
        # premises and the sketch model instead of with heat.
        temp = min(config.MAX_PROVER_TEMP, temp + 0.1 * (i // len(STRATEGIES)))

        attempt_started = time.time()
        left = int(deadline - time.time())
        _trace(f"[prove] attempt {i + 1}"
               f"{'/' + str(attempts) if attempts > 0 else ''} "
               f"temp={temp:.2f} strategy={(hint[:48] + '...') if hint else 'direct'}"
               f"{' mode=' + mode if mode else ''} ({left}s budget left)")

        # The lab server's 32B whenever it answers; the local 8B is a backup
        # for when it does not, never a second opinion. Asked every attempt, so
        # a tunnel that drops mid-run falls back and one that returns is used.
        base, model = _prover_target()
        res = _attempt(statement, temp, hint, rounds, deadline, base, model,
                       mode=mode, avoid=avoid)

        i += 1
        # Close the bandit's loop. Without this the selector is a random
        # permutation with extra machinery: it can only learn from outcomes it
        # is actually told about.
        try:
            from . import bandit as _bandit
            _bandit.update(name, won=bool(res.get("ok")), context=_domain,
                           seconds=time.time() - attempt_started)
        except Exception:
            pass
        # Keys starting with `_` are for the DURABLE RECORD only: journal.py
        # writes them in full, agent.py strips them before the result enters the
        # model's context. Round-by-round compiler output is exactly this shape
        # of data -- indispensable to the next session, ruinous to a 32k window.
        history.append({"attempt": i, "model": model, "temp": round(temp, 2),
                        "mode": mode, "strategy": name, "ok": res["ok"],
                        "reason": res.get("reason"), "rounds": res.get("rounds"),
                        "_strategy": hint or "direct",
                        "_rounds": res.get("log", [])})
        if not res["ok"]:
            # Record what this attempt actually did so the next one diverges.
            sig = next((_signature(l.get("errors", "")) for l in reversed(res.get("log", []))
                        if l.get("errors")), "")
            last_err = next((l.get("errors", "")[:120] for l in reversed(res.get("log", []))
                             if l.get("errors")), "")
            note = f"{hint[:60] or 'direct'} -> {res.get('reason')}: {last_err}"
            if note not in avoid:
                avoid.append(note)
            continue

        # Pinning does NOT make mis-formalization impossible -- it moves the risk
        # from the prover to whoever wrote the signature, where nothing was
        # checking it. Observed: claim "gcd(m,n) divides both m and n AND
        # lcm(m,n) is a multiple of both" was pinned to a signature covering only
        # the gcd half, and recorded proved=true with review skipped. The review
        # is the one gate between a compiling proof and an overstated result, so
        # it now runs on every path.
        _trace("[prove] compiled -- back-translating the statement for faithfulness...")
        review = review_statement(statement, res["lean"])
        _trace(f"[prove] review: {str(review.get('verdict', ''))[:100]}")
        if review.get("faithful") is False:
            history[-1]["formalization"] = "MISMATCH"
            if lean_statement:
                # The signature is pinned, so every further attempt would prove
                # the SAME mismatched statement. Retrying cannot help; the caller
                # has to change the signature or narrow the claim.
                return _remember(key, {"proved": False, "attempts": history,
                        "elapsed": round(time.time() - started, 1),
                        "formalization_suspect": True,
                        "lean": res["lean"], "formalization": review,
                        "note": ("The pinned lean_statement COMPILES but does not "
                                 "express your claim: " + str(review.get("verdict", ""))
                                 + " Narrow the claim to what the signature actually "
                                 "says, or pin a signature that covers all of it.")})
            # Not pinned: a later attempt may formalize it honestly.
            continue
        out = {"proved": True, "lean": res["lean"], "attempts": history,
               "rounds": res["rounds"], "elapsed": round(time.time() - started, 1),
               "formalization": review}
        _record(statement, lean_statement, out)
        return out

    suspect = any(h.get("formalization") == "MISMATCH" for h in history)
    return _remember(key, {"proved": False, "attempts": history,
            "elapsed": round(time.time() - started, 1),
            "formalization_suspect": suspect,
            "note": ("every compiling proof mis-formalized the claim" if suspect
                     else "not proved within attempt/time budget -- a real negative result")})


# ---------------------------------------------------------------------------
# TRACING. Applied at module scope, deliberately, rather than edited into each
# function: prove_theorem has TEN return statements and _derive_once is called
# through a module-global lookup, so wrapping the names here captures every
# exit path -- including the early redundancy and deadline returns, which are
# the ones that reveal budget being spent on nothing.
#
# Both wrappers are failure-transparent: the trace call cannot change what the
# caller receives, and cannot raise into a running proof.
# ---------------------------------------------------------------------------

_prove_theorem_untraced = prove_theorem
_derive_once_untraced = _derive_once


@functools.wraps(_prove_theorem_untraced)
def prove_theorem(statement, lean_statement="", *a, **kw):          # noqa: F811
    out = _prove_theorem_untraced(statement, lean_statement, *a, **kw)
    try:
        trace.proof(statement, out if isinstance(out, dict) else {}, lean_statement)
    except Exception:
        pass
    return out


@functools.wraps(_derive_once_untraced)
def _derive_once(statement, feedback=""):                            # noqa: F811
    t0 = time.time()
    sig = _derive_once_untraced(statement, feedback)
    try:
        # On failure record WHAT was produced, not merely that nothing was.
        # 77 failures were logged with an error string and no candidate, so the
        # dominant sub-mode (39% malformed syntax) could be counted but never
        # read -- the same blind spot the scout had.
        trace.signature(statement, sig, ok=bool(sig),
                        seconds=time.time() - t0,
                        err="" if sig else _LAST_SIG_ERROR.get("err", ""),
                        rejected="" if sig else _LAST_SIG_ERROR.get("sig", ""),
                        repaired=bool(_LAST_SIG_ERROR.pop("repaired", False)))
    except Exception:
        pass
    return sig


