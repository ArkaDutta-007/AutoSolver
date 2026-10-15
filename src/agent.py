"""Perpetual auto-research orchestrator.

Orchestrator: Lightning (1M context, tool-trained). Ground truth: the Lean
compiler. State is appended to JSONL after every step, so a crash, reboot, or
Ctrl-C loses at most the in-flight step.
"""

import json
import os
import re
import signal
import time

import requests

from . import config, gt, journal, library, plan, propose, prover, reward, search, tools, trace

KEEP_RECENT_MSGS = 20


# Compaction fires when the WHOLE prompt -- system, anchor, every tool result --
# reaches config.COMPACT_AT (75%) of the window. It is checked right before each
# call, so what the model is sent is at most that plus a nudge, and the rest of
# the window is generation. (The old 0.30-of-chars threshold was chosen when the
# tool result and anchor were counted on top of it; they are inside it now.)
#
# Tokens, not characters, when the server has told us: the native endpoint
# reports prompt_eval_count. It can UNDER-count when the prompt cache served a
# prefix, so a character floor at a deliberately dense 3 chars/token backs it.
_FILL = {"chars": 0, "tokens": 0}
DENSE_CHARS_PER_TOKEN = 3.0


def _chars(messages: list) -> int:
    return sum(len(json.dumps(m)) for m in messages)


def _note_fill(messages: list, body: dict) -> None:
    n = int(body.get("prompt_eval_count") or 0)
    if n:
        _FILL.update(chars=_chars(messages), tokens=n)


def prompt_tokens(messages: list) -> int:
    chars = _chars(messages)
    floor = chars / DENSE_CHARS_PER_TOKEN
    if _FILL["tokens"]:
        return int(max(floor, _FILL["tokens"] + (chars - _FILL["chars"]) / DENSE_CHARS_PER_TOKEN))
    return int(floor)


def compact_at_chars() -> int:
    """The character equivalent of the threshold, for budgets that think in chars."""
    return config.budget_chars(config.COMPACT_AT)


def tool_result_max_chars() -> int:
    """One tool result must never be able to fill the window on its own.

    `read_journal` defaults to 200,000 chars and the model called it with exactly
    that -- roughly 57k tokens, which by itself exceeded the entire window.
    """
    return max(4_000, config.budget_chars(0.10))

SYSTEM = """You are an autonomous research agent running unattended for a long time.

Your tools:
- prove_theorem: formally verify a mathematical claim in Lean 4
- search_mathlib: look up real Mathlib lemma names
- web_search / fetch_url: ONLY present for open (unsolved) problems. If you do
  not see them, this problem has a known solution somewhere and you must solve it
  yourself. Do not ask for them; do not work around their absence.

- open_problem / read_journal / journal_entry / rule_out / write_solution /
  close_session: your journal. Every problem gets its own resumable document.
  read_journal is the ONLY way to read it back -- you have no file access.

You are one session in a long chain working the same problem. The chain is a
loop, and your job is to advance it by one turn of it:

    inherit the PLAN left by the last session
      -> ACT on it (prove_theorem / run_python / search_mathlib)
      -> Lean VERIFIES or refutes
      -> REFLECT on what that result actually means
      -> leave the next PLAN for the session after you

Everything durable lives in the journal, and the block above every turn shows
you the current state of the loop: the inherited plan, what previous sessions
concluded, what is established, what is ruled out.

Operating rules:
0. Call open_problem FIRST for each problem. It returns `next_step` (what the
   last session told you to do -- start there unless you have a reason not to,
   and say the reason), `prior_sessions` (the accumulated reflections),
   `established` (proved -- build on it, never re-prove it) and `exhausted`
   (dead ends -- do NOT repeat them).
   Journal continuously: PLAN before acting, REASON for each choice, REFLECT on
   each result. A later session reads this to continue your work, so write for
   that reader. rule_out() anything you have definitively eliminated.
   On success call write_solution() with a clean step-by-step solution.
   Always finish with close_session(status, summary, next_step). `summary` is
   what you LEARNED; `next_step` is the single concrete thing the next session
   should try first. A session that ends without a next_step forces the one
   after it to start blind -- that is the failure this loop exists to prevent.
1. Work in small steps. Each turn: decide the single most valuable next action
   and CALL A TOOL. Think as long as the problem needs, but a turn that ends
   without a tool call has accomplished nothing -- journal_entry is always a
   valid way to bank thinking before you act.
2. Any mathematical claim you intend to rely on must go through prove_theorem.
   A claim that "seems obviously true" is exactly the kind that should be checked.
3. prove_theorem already retries hard internally. proved=false is a RESULT --
   record it, state what blocked it, and move to a different angle. Do not
   resubmit the identical statement.
4. If a proof fails, consider whether a WEAKER lemma or a special case is
   provable, and pursue that. Partial formal progress beats none.
5. State what you have established and what remains open. Be concrete.
6. You are not trying to finish quickly. Depth over speed.
7. run_python is for EXPLORING before you formalize. Simulate small cases,
   search for a pattern, test a conjecture numerically. A conjecture that fails
   at n=4 is not worth a single prover second. Observe, then conjecture, then
   prove -- in that order.

Every turn you are re-shown the problem statement, what is established, and what
is ruled out. That block is the truth. If it disagrees with your memory, the
block is right and your memory is stale.
"""

ANCHOR = """== PROBLEM ({title}) ==  [session {session}]
{statement}

== PLAN INHERITED FROM THE LAST SESSION ==
{next_step}

== PRIOR SESSIONS (reflect -> plan, oldest first) ==
{history}

== ESTABLISHED (proved, build on these) ==
{established}

== RULED OUT (do not repeat) ==
{exhausted}

== JOURNAL TAIL ==
{tail}
== END OF DURABLE STATE =="""


def _anchor_message() -> dict:
    """Rebuild durable state as a user message, fresh, every single turn.

    This is the fix for the failure that wasted every run so far: the agent
    assumed the conversation remembered the problem. When the window truncated it
    did not error -- it confabulated, opened a problem titled "Continuing
    interrupted work", and then invented "Fundamental Theorem of Arithmetic" to
    work on instead. Durable state is now re-stated every turn, so a truncated
    context costs recent chatter and never the problem itself.
    """
    a = journal.anchor(tail_chars=max(2000, config.budget_chars(0.08)))
    if not a.get("statement") and not a.get("title"):
        return {}
    fmt = lambda xs: "\n".join(f"- {x}" for x in xs[-12:]) if xs else "(nothing yet)"
    history = "\n".join(
        f"- s{e.get('session')} [{e.get('status')}]: {str(e.get('summary', ''))[:400]}"
        + (f"\n    -> next: {e['next_step'][:300]}" if e.get("next_step") else "")
        for e in (a.get("sessions_log") or [])[-4:]) or "(this is the first session)"
    # The exploration map rides along with the anchor because it is durable
    # state of exactly the same kind: `established` says WHAT is known, this
    # says what KINDS have been covered and which are untouched. Listing only
    # the former is what let five restatements of one fact read as progress.
    try:
        from . import archive
        emap = archive.advice(a.get("statement") or "")
    except Exception:
        emap = ""
    # The checklist is durable state of the same kind as `established`: it says
    # what FINISHING means. Without it every turn re-invents the plan, which is
    # how five problems produced zero write_solution calls.
    try:
        emap += plan.brief(a.get("statement") or "", _ELAPSED[0])
    except Exception:
        pass
    # The game-theoretic layer rides the anchor for the same reason everything
    # else here does: a mechanism the model must remember to invoke is a
    # mechanism that does not run. Measured -- search_mathlib went uncalled
    # while the agent invented names Mathlib already had. This arrives unasked.
    try:
        emap += gt.brief(a.get("statement") or "",
                         seconds_left=max(0.0, _SECONDS_LEFT[0]),
                         claim=a.get("next_step") or "")
    except Exception:
        pass
    return {"role": "user", "content": ANCHOR.format(
        title=a["title"], session=a.get("session", 1),
        statement=a["statement"] or "(not recorded)",
        next_step=a.get("next_step") or "(none named — you are starting fresh)",
        history=history,
        established=fmt(a["established"]), exhausted=fmt(a["exhausted"]),
        tail=(a["journal_tail"] or "(empty)") + emap)}

_stop = False
# Fraction of this problem's budget already spent, for the closure trigger.
_ELAPSED = [0.0]
# Wall-clock left in this problem, for the forecast-driven allocator.
_SECONDS_LEFT = [0.0]


def _sigint(signum, frame):
    global _stop
    print("\n[shutting down after current step -- state is saved]", flush=True)
    _stop = True


def log(path: str, record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _for_context(value):
    """Strip `_`-prefixed keys before a tool result enters the model's context.

    The convention: `_rounds`, `_tried`, `_strategy` and friends are for the
    DURABLE RECORD. journal.py writes them in full -- round-by-round compiler
    output is exactly what a later session needs and what the summary table
    cannot express -- but pasting them into a 32k window would push out the
    problem statement to make room for error text the model has already seen.
    Grow the record; protect the context.
    """
    if isinstance(value, dict):
        return {k: _for_context(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_for_context(v) for v in value]
    return value


def _brief(result, limit: int = 320) -> str:
    """One-line tool result for the terminal: every key, no wall of proof text."""
    if not isinstance(result, dict):
        return str(result)[:limit]
    out = {}
    for k, v in _for_context(result).items():
        s = str(v)
        out[k] = s[:70] + "..." if len(s) > 70 else v
    return json.dumps(out, ensure_ascii=False, default=str)[:limit]


def _code_version() -> str:
    """Which code produced this transcript.

    Fixes landed mid-run three times in one day, and the running process keeps
    its modules from import time -- so a transcript read afterwards was never
    the system it appeared to describe. Stamp it.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, timeout=5)
        rev = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{rev}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


EXPLORE_ONLY = {"run_python", "search_mathlib", "read_journal", "journal_entry"}

# What kind of problem this is decides what counts as checking it. The loop was
# built for proofs, and it treated everything as one: one run was Irodov 1.96
# (a projectile's momentum increment), a problem whose answer is an
# expression, sent into a harness that withholds run_python after
# twelve calls to force a Lean proof of... a physics formula.
_ASKS_PROOF = re.compile(r"\b(prove|show that|demonstrate|verify that|deduce that|"
                         r"for (all|every|each)|there (exists?|is|are) (infinitely|no|a unique))\b", re.I)
_ASKS_VALUE = re.compile(r"\b(find|calculate|compute|determine|evaluate|estimate|express|"
                         r"what (is|are)|how (many|much|long|far|fast|high))\b", re.I)
_PHYSICAL = re.compile(r"\b(velocity|speed|mass|force|momentum|energy|acceleration|charge|"
                       r"field|current|voltage|friction|gravity|pressure|temperature|particle|"
                       r"body|pendulum|spring|frequency|wavelength)\b|\bm/s\b|\bkg\b", re.I)


def kind_of(goal: str) -> str:
    """proof | compute | physics. Cheap on purpose: a regex, no model call."""
    text = goal.split("Open its journal first")[0]
    if _ASKS_PROOF.search(text) and not _ASKS_VALUE.search(text[:400]):
        return "proof"
    if _PHYSICAL.search(text):
        return "physics"
    return "compute" if _ASKS_VALUE.search(text) else "proof"


KIND_RULES = {
    "proof": "",
    "compute": """
THIS PROBLEM ASKS FOR A VALUE OR AN EXPRESSION, not a proof. Adjust rule 2:
- The deliverable is the answer, derived and CHECKED. run_python (sympy) is your
  verifier: substitute the answer back, check small or limiting cases, and
  confirm a numeric instance independently of the derivation.
- prove_theorem is for the pure-mathematics identities the derivation rests on
  that Lean can state cheaply; do not try to formalise the whole computation.
- write_solution once the answer agrees two independent ways. Put the final
  answer on its own line at the end, in closed form.""",
    "physics": """
THIS IS A PHYSICS PROBLEM: it asks for a quantity, not a theorem. Adjust rule 2:
- State the model first (frame, forces, conserved quantities, assumptions such
  as "drag negligible") in a journal_entry, then derive symbolically.
- run_python (sympy) is your verifier: check dimensions/units, limiting cases
  (angle -> 0, t -> 0, m -> infinity), and one numeric instance.
- prove_theorem only for a clean mathematical identity inside the derivation;
  Lean does not know physics, and a physical model is not a Lean statement.
- write_solution once the answer is derived and checked: every part (a), (b),
  ... answered, each final result on its own line with its units.""",
}

NUDGE = (
    "You have made {n} tool calls without a single prove_theorem. Exploration has "
    "told you something -- now commit to it. State the pattern you found as ONE "
    "precise mathematical claim and call prove_theorem on it. If you can write the "
    "Lean signature yourself, pass lean_statement too. A numerical pattern that is "
    "never formalized is not a result.")

FORCE = (
    "{n} tool calls, still no prove_theorem. run_python has been WITHHELD for this "
    "turn. Your next call must be prove_theorem on the most specific claim your "
    "exploration supports -- start with the smallest lemma, not the whole problem "
    "-- or close_session naming that claim as next_step.")


def _decoded_lines(response):
    """Decode an SSE stream as UTF-8, incrementally.

    `r.iter_lines(decode_unicode=True)` was corrupting every non-ASCII character
    the model produced. Two reasons, both real:

    1. requests picks the encoding from the Content-Type header, and HTTP says
       `text/*` without an explicit charset is ISO-8859-1. An SSE stream is
       `text/event-stream`, so every multi-byte UTF-8 sequence was decoded as
       latin-1 -- Omega became "Î©", and re-encoding on write produced the
       double-encoded bytes now sitting in the journal (24 mangled Omegas, 13
       mangled products in problems/001 alone).
    2. Even with the right charset, chunks are decoded independently, so a
       character split across a chunk boundary is mangled anyway.

    An incremental decoder over raw bytes fixes both. This matters more here
    than in most systems: the corrupted characters are exactly the ones a proof
    is written in -- the divides bar, the naturals, product, element-of -- and
    they were flowing into the journal, back into context, and into Lean.
    """
    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    for chunk in response.iter_content(chunk_size=None):
        if not chunk:
            continue
        buf += decoder.decode(chunk)
        while "\n" in buf:
            line, _, buf = buf.partition("\n")
            yield line.rstrip("\r")
    buf += decoder.decode(b"", final=True)
    if buf:
        yield buf.rstrip("\r")


def _native_payload(messages: list, max_tokens: int, suppress: set, stream: bool,
                    think: str = "") -> dict:
    """Body for ollama's NATIVE /api/chat.

    The orchestrator used to speak `/v1/chat/completions`, which has no `think`
    parameter -- so thinking effort was controllable for signature derivation
    (which already used the native endpoint) and NOT for the reasoning loop that
    decomposes the problem and picks strategies. That is backwards: the tool
    loop is where deliberation matters most. `/v1` also ignores `num_ctx`, which
    is what forced the `-32768` derived tags in the first place.

    Message shape stays OpenAI-style internally because the rest of the module
    depends on it; only the wire format is converted, here and in _from_native.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            # DEEP copy. `dict(m)` shares the nested tool_calls objects, so
            # rewriting `arguments` here also rewrote the caller's history --
            # after which tools.dispatch would json.loads() a dict and every
            # tool call from the second turn onward would fail.
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                # Native wants arguments as an OBJECT; we carry them as a JSON
                # string so tools.dispatch's json.loads keeps working.
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {}
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        m.pop("tool_call_id", None)          # native keys tool results by order
        out.append(m)
    body = {"model": config.ORCHESTRATOR_MODEL, "messages": out,
            "tools": tools.specs(suppress=suppress), "stream": stream,
            "options": {**config.ctx(config.ORCHESTRATOR_MODEL), "num_predict": max_tokens, "temperature": 0.7}}
    # An explicit override beats the configured effort. Asking the model to
    # "think briefly" in the PROMPT does not work -- effort is an API
    # parameter, and think=high stayed high while the retry truncated exactly
    # as the first attempt had.
    body.update({"think": think} if think else prover._think_opt())
    return body


def _from_native(calls: list) -> list:
    """Native tool calls -> the OpenAI shape the dispatcher expects."""
    out = []
    for i, tc in enumerate(calls or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        out.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                    "function": {"name": fn.get("name", ""),
                                 "arguments": args if isinstance(args, str)
                                 else json.dumps(args or {})}})
    return out


def _stream_orchestrator(messages: list, max_tokens: int, suppress: set = None,
                         think: str = "") -> tuple:
    """Stream the reply, printing reasoning as it is produced.

    Without this a step is a blank `thinking...` for minutes followed by a wall
    of text -- step 3 of the IMO P1 run was 386 seconds and 73,359 characters.
    Streaming makes the deliberation watchable while it happens, which is the
    only way to tell grinding apart from looping in real time.
    """
    r = requests.post(f"{config.OLLAMA_NATIVE}/api/chat",
                      json=_native_payload(messages, max_tokens, suppress, True, think),
                      timeout=prover._gen_timeout(max_tokens), stream=True)
    r.raise_for_status()

    content, reasoning, finish, printed = [], [], None, False
    calls: list = []
    for raw in _decoded_lines(r):
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except ValueError:
            continue
        m = chunk.get("message") or {}

        # Native names the reasoning channel `thinking`; /v1 called it
        # `reasoning`. Accept both so a server change cannot silently drop it.
        for field, sink in (("thinking", reasoning), ("reasoning", reasoning),
                            ("content", content)):
            piece = m.get(field)
            if piece:
                if not printed:
                    print("\n  ┌─ thinking ─────────────────────────────", flush=True)
                    printed = True
                sink.append(piece)
                print(piece, end="", flush=True)

        if m.get("tool_calls"):
            calls.extend(m["tool_calls"])
        if chunk.get("done"):
            finish = chunk.get("done_reason") or "stop"
            _note_fill(messages, chunk)

    if printed:
        print("\n  └────────────────────────────────────────", flush=True)

    msg = {"role": "assistant", "content": "".join(content)}
    if calls:
        msg["tool_calls"] = _from_native(calls)
    joined = "".join(reasoning).strip()
    reasoning_only = False
    if not msg["content"].strip() and joined:
        msg["content"] = joined
        reasoning_only = True
    return msg, finish, reasoning_only, bool(content or reasoning)


def call_orchestrator(messages: list, max_tokens: int = None, suppress: set = None,
                      think: str = "") -> tuple:
    """Returns (message, finish_reason, reasoning_only).

    Lightning is a thinking model: it puts deliberation in `reasoning` and can
    leave `content` empty. prover.py has folded that back since the Goedel bug,
    but the orchestrator never did -- so a reasoning-only turn was discarded AND
    appended to the context with empty content, leaving the model to re-derive
    the same thoughts next step. Observed as three consecutive ~2-minute steps
    producing nothing but "(no tool call -- nudging)".

    A turn cut off at the cap is NOT retried. Regenerating from scratch at a
    bigger budget discards the partial reasoning and pays for the whole thing
    again -- measured at ~7 minutes for one step across an 8k/16k/32k ladder.
    Since the partial reasoning now survives in `content`, it is cheaper and
    strictly more informative to keep it and continue from it next turn.
    """
    max_tokens = max_tokens or config.ORCH_MAX_TOKENS
    if config.STREAM_TRACE:
        try:
            msg, finish, reasoning_only, got = _stream_orchestrator(
                messages, max_tokens, suppress, think)
            if got:
                return msg, finish, reasoning_only, True
            print("  [stream produced nothing -- falling back]", flush=True)
        except Exception as exc:
            print(f"  [stream failed ({exc}) -- falling back]", flush=True)

    r = requests.post(f"{config.OLLAMA_NATIVE}/api/chat",
                      json=_native_payload(messages, max_tokens, suppress, False, think),
                      timeout=prover._gen_timeout(max_tokens))
    r.raise_for_status()
    body = r.json()
    _note_fill(messages, body)
    raw = body.get("message") or {}
    msg = {"role": "assistant", "content": raw.get("content") or ""}
    if raw.get("tool_calls"):
        msg["tool_calls"] = _from_native(raw["tool_calls"])
    finish = body.get("done_reason") or "stop"

    reasoning = ((raw.get("thinking") or raw.get("reasoning") or "")).strip()
    reasoning_only = False
    if not msg["content"].strip() and reasoning:
        msg["content"] = reasoning
        reasoning_only = True

    return msg, finish, reasoning_only, False


def compact(messages: list) -> list:
    """Trim context, but persist the digest to the notebook FIRST.

    Compaction therefore loses nothing durable: the context shrinks while the
    on-disk record keeps growing.
    """
    used = prompt_tokens(messages)
    if used < config.COMPACT_AT * config.CONTEXT_TOKENS:
        return messages
    head, tail = messages[1:-KEEP_RECENT_MSGS], messages[-KEEP_RECENT_MSGS:]
    if not head:
        return messages
    print(f"[context ~{used:,} of {config.CONTEXT_TOKENS:,} tokens "
          f"({used / config.CONTEXT_TOKENS:.0%}) -- compacting]", flush=True)
    _FILL.update(chars=0, tokens=0)          # the next call counts afresh
    digest = "\n".join(f"- {m.get('role')}: {str(m.get('content'))[:200]}"
                       for m in head if m.get("content"))[:8000]
    print(f"[compacting {len(head)} older messages -> notebook]", flush=True)
    journal.entry('SUMMARY', f'Auto-compaction of {len(head)} messages:\n\n{digest}')
    return [messages[0], {"role": "user", "content": f"Summary of earlier work:\n{digest}"}] + tail


def _strip_stale_reasoning(messages: list, keep_last: int = 2) -> list:
    """Drop raw chain-of-thought from older assistant turns.

    A single reasoning turn ran to 113,191 characters (~32k tokens). Kept in the
    transcript it crowds out everything else, and feeding a thinking model its own
    previous reasoning degrades the next turn rather than helping it. The full
    text is already on disk in the journal, so nothing is lost -- only the copy
    that was competing with the problem statement for window space.
    """
    out, seen = [], 0
    for m in reversed(messages):
        if (m.get("role") == "assistant" and not m.get("tool_calls")
                and len(str(m.get("content") or "")) > 4000):
            seen += 1
            if seen > keep_last:
                m = dict(m)
                m["content"] = (str(m["content"])[:1500]
                                + "\n\n[...reasoning trimmed; full text is in the journal]")
        out.append(m)
    return list(reversed(out))


def _score(gain: "reward.Tracker", path: str, step: int, kind: str, messages: list,
           stuck_steps: int) -> int:
    """Close the step's reward; on a stall, say so and bring the proposals forward."""
    got = gain.end()
    print(gain.line(got), flush=True)
    log(path, {"event": "reward", "step": step, **got})
    if got["stalled"]:
        print("  [implicit reward flat -- asking for a different line]", flush=True)
        messages.append({"role": "user", "content": reward.stall_message(got, kind)})
        return max(stuck_steps, config.ORCH_PASS_EVERY)    # pass@K runs next step
    return stuck_steps


def run(goal: str, max_steps: int = 0, time_budget: int = 0) -> str:
    """Work one problem. Returns closed | maxsteps | timebudget | interrupted."""
    signal.signal(signal.SIGINT, _sigint)
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    path = os.path.join(config.RUNS_DIR, f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    offered = [t["function"]["name"] for t in tools.specs()]
    rev = _code_version()
    print(f"goal:   {goal}\nsearch: {'ENABLED' if search.enabled() else 'disabled'}\n"
          f"code:   {rev}\n"
          f"tools:  {len(offered)} offered -- {', '.join(offered)}\n"
          f"limit:  {max_steps if max_steps > 0 else 'perpetual'} steps\n"
          f"log:    {path}\nstop:   Ctrl-C\n" + "-" * 60, flush=True)
    log(path, {"event": "start", "goal": goal, "search_enabled": search.enabled(),
               "tools_offered": offered, "code_version": rev,
               "orchestrator": config.ORCHESTRATOR_MODEL, "prover": config.PROVER_FALLBACK_MODEL or config.PROVER_MODEL})

    kind = kind_of(goal)
    print(f"kind:   {kind}", flush=True)
    messages = [{"role": "system", "content": SYSTEM + KIND_RULES[kind]},
                {"role": "user", "content": f"Research goal: {goal}\n\nBegin."}]
    # The implicit reward (reward.py): scored every step from what happened.
    gain = reward.Tracker()
    step = proved = 0
    # Observed in run-20260813-191856: 15 consecutive run_python calls, 15
    # distinct programs, zero prove_theorem. Giving the agent a cheap tool with
    # no forcing function swapped one failure mode (never explore) for another
    # (only explore). Count the drought and escalate.
    since_prove = 0
    recent_actions: list = []
    forced_think = ""
    stuck_steps = 0        # steps since the checklist last moved
    started = time.time()
    reason = "maxsteps"
    closed = False
    # Each problem gets a fresh failure memory, and the prover is told when this
    # problem's clock runs out so a single proof attempt cannot outlive it.
    prover.reset_failures()
    prover.set_problem_deadline(started + time_budget if time_budget else 0)
    # The quality-diversity archive needs the goal to tell a restatement of it
    # apart from a genuine sub-lemma: the SAME claim is a dead end for one
    # problem and real progress on another.
    prover.set_goal(goal)
    # Tag every Lean verdict from here on with the problem it belongs to.
    # Without this the trace is still usable for pass rates but nothing can
    # be attributed to a problem, and per-problem difficulty is what any
    # curriculum needs.
    try:
        trace.begin(goal=goal)
    except Exception:
        pass
    # Decompose ONCE into the obligations that would settle the problem. Built
    # here rather than on demand because a ledger rebuilt each turn is not a
    # checklist -- it is a new plan every turn, which is what the agent was
    # already doing when it closed nothing in ten hours.
    try:
        # The big cloud model plans first (plan.blueprint); from here on the
        # local model does every step. Off, keyless or failed = local plan.
        led = plan.build(goal, specs=tools.specs())
        bp = led.get("blueprint") or {}
        if bp:
            kind = bp["formulation"]["kind"] if bp["formulation"].get("kind") in KIND_RULES else kind
            # in the system prompt, because compaction keeps that one verbatim
            messages[0]["content"] = SYSTEM + KIND_RULES[kind] + "\n\n" + plan.blueprint_text(bp)
            print(f"[planner] {bp['model']} planned {len(bp['steps'])} step(s); kind: {kind}; "
                  "the local model does the work", flush=True)
            print("\n".join(plan.blueprint_text(bp).splitlines()[1:-1]), flush=True)
            log(path, {"event": "planner", "blueprint": bp, "kind": kind})
        n = len(led.get("obligations") or [])
        print(f"[plan] {n} obligation(s) to discharge", flush=True)
        for o in (led.get("obligations") or []):
            print(f"   [ ] {o['id']:<14} ({o['kind']}) {o['claim'][:88]}", flush=True)
    except Exception as exc:
        print(f"[plan] ledger unavailable ({exc})", flush=True)

    while not _stop and not closed and (max_steps <= 0 or step < max_steps):
        if time_budget and time.time() - started > time_budget:
            reason = "timebudget"
            print(f"\n[per-problem time budget of {_hms(time_budget)} reached]", flush=True)
            break
        step += 1
        stuck_steps += 1
        _ELAPSED[0] = ((time.time() - started) / time_budget) if time_budget else 0.0
        _SECONDS_LEFT[0] = max(0.0, time_budget - (time.time() - started)) if time_budget else 0.0
        messages = _strip_stale_reasoning(compact(messages))
        # Escalate out of an exploration loop. Asking is level one; WITHHOLDING
        # the tool is level two, the same lever this harness already uses on the
        # web tools, and for the same reason -- an offered tool gets reached for.
        # Only for proofs: for a value, run_python IS the check, not a detour.
        forcing = kind == "proof" and since_prove >= config.PROVE_FORCE_AFTER
        suppress = {"run_python"} if forcing else None

        # ENFORCED CLOSURE. Telling the agent to "close now" in the prompt is
        # advisory, and advisory did not work: write_solution was called zero
        # times across every run ever recorded. When the checklist says it is
        # time to finish, the tools that OPEN new lines are withheld, so the
        # only moves left are to attempt the closure obligation, write up what
        # is established, or close the session. Same mechanism that ended the
        # run_python exploration loop -- withholding beats asking.
        try:
            c = plan.should_close(goal, _ELAPSED[0])
        except Exception:
            c = {"close": False}
        if c.get("close"):
            suppress = (suppress or set()) | {"run_python", "conjecture", "hunches"}
            print(f"  [CLOSING: {c['why']} -- withholding exploration tools]",
                  flush=True)
            messages.append({"role": "user", "content":
                             f"STOP OPENING NEW LINES. {c['why']}. You have "
                             f"{c.get('proved', 0)}/{c.get('total', 0)} obligations "
                             "discharged. Do exactly one of: (a) prove_theorem the "
                             "`closure` obligation using the lemmas you already "
                             "hold, or (b) write_solution with the argument you can "
                             "actually support, stating explicitly which obligations "
                             "remain open and why. A partial solution that names its "
                             "gap is a result; an unassembled pile of lemmas is not."})
        if forcing:
            print(f"  [{since_prove} calls without prove_theorem -- "
                  f"withholding run_python this turn]", flush=True)
            messages.append({"role": "user", "content": FORCE.format(n=since_prove)})
        elif kind == "proof" and since_prove and since_prove % config.PROVE_NUDGE_AFTER == 0:
            print(f"  [{since_prove} calls without prove_theorem -- nudging]", flush=True)
            messages.append({"role": "user", "content": NUDGE.format(n=since_prove)})
        # Re-state durable truth every turn. Cheap; the alternative is an agent
        # that forgets which problem it is solving and invents a new one.
        anchor = _anchor_message()
        if anchor:
            messages = ([messages[0], anchor] + [m for m in messages[1:]
                        if not str(m.get("content", "")).startswith("== PROBLEM (")])

        # pass@K for the ORCHESTRATOR. Several structurally different next
        # actions are proposed by the FAST model and scored against the
        # exploration map, then offered as evidence. The orchestrator still
        # decides -- the score knows about novelty, repetition and cost, and
        # nothing about the mathematics -- but what it reliably catches is the
        # loop, which is what no amount of the orchestrator's own deliberation
        # ever caught.
        # GATED, not every step. Each proposal is a separate model call, and on
        # this machine four of them cost ~8 minutes -- measured as ~90% of a
        # step whose orchestrator turn took 31.6s. Proposals are worth that only
        # when the agent is NOT advancing: if the checklist moved last step, the
        # cheapest correct thing is to keep going.
        want_props = (config.ORCH_PASS_K > 1
                      and (step == 1 or stuck_steps >= config.ORCH_PASS_EVERY))
        if want_props:
            t_prop = time.time()
            try:
                picks = propose.choose(goal, recent_actions)
                block = propose.brief(picks)
                if block:
                    top = picks["best"]
                    print(f"  [pass@{len(picks['ranked'])}: {top['angle']}/"
                          f"{top['tool']} {top['score']:+.2f} "
                          f"in {time.time() - t_prop:.0f}s]", flush=True)
                    messages.append({"role": "user", "content": block})
                else:
                    # NEVER silent. An empty result printed nothing, so four
                    # failing calls per step were invisible for the whole run.
                    print(f"  [pass@K produced NO candidates in "
                          f"{time.time() - t_prop:.0f}s -- proposer is failing]",
                          flush=True)
                stuck_steps = 0
            except Exception as exc:
                print(f"  [proposals unavailable ({exc})]", flush=True)
                stuck_steps = 0
        gain.begin()
        # The orchestrator can think for minutes. Say so, or the terminal looks dead.
        print(f"\n[step {step}] ({_hms(time.time() - started)} elapsed) thinking...",
              end="", flush=True)
        t0 = time.time()
        try:
            msg, finish, reasoning_only, streamed = call_orchestrator(
                messages, suppress=suppress, think=forced_think)
            forced_think = ""            # one-shot: only the retry is degraded
        except Exception as exc:
            print(f"\r[step {step}] orchestrator error after {time.time() - t0:.1f}s: "
                  f"{exc}\n         retrying in 10s", flush=True)
            log(path, {"event": "error", "step": step, "error": str(exc)})
            time.sleep(10)
            continue
        print(f"\r[step {step}] ({_hms(time.time() - started)} elapsed) "
              f"orchestrator replied in {time.time() - t0:.1f}s", flush=True)

        messages.append(msg)
        content = (msg.get("content") or "").strip()
        if content:
            if not streamed:
                label = "reasoning" if reasoning_only else "says"
                print(f"  [{label}] {content}", flush=True)
            log(path, {"event": "assistant", "step": step, "content": content,
                       "reasoning_only": reasoning_only, "finish_reason": finish})
            # Persist BEFORE anything can drop it. Step 3 of the P1 run produced
            # 113k characters containing the correct invariant for the problem
            # and it was never written anywhere, because only TOOL CALLS were
            # journalled. A turn that thinks well and forgets to call a tool must
            # still leave its mathematics on disk.
            journal.entry("THINKING", content)

        calls = msg.get("tool_calls") or []
        gain.turn(finish, bool(calls))
        if not calls:
            if finish == "length":
                # NEVER tell the model to "continue from where you stopped".
                # When the window has already dropped the reasoning, that
                # instruction cannot be obeyed and is obeyed anyway: the model
                # confabulated a prior conversation, opened a problem called
                # "Continuing interrupted work", and then invented an entirely
                # different problem. Re-ground it in durable state instead, and
                # tell it plainly that the long think did not survive.
                print(f"  (cut off at {config.ORCH_MAX_TOKENS} tokens -- re-grounding; "
                      f"the full text was saved to the journal)", flush=True)
                messages.append({"role": "user", "content":
                                 "Your last turn hit the generation limit and was NOT "
                                 "completed. Its full text has been saved to the journal, "
                                 "so nothing is lost. Do not try to recall it from memory. "
                                 "Re-read the PROBLEM / ESTABLISHED / RULED OUT block above "
                                 "-- that is the authoritative state -- then take ONE small "
                                 "concrete action and end this turn with a tool call. "
                                 "Think briefly this time."})
                # And ENFORCE it. Measured across the IMO run: 24 of 69 steps
                # ended `length` + reasoning_only, producing nothing, because
                # the prompt-level "think briefly" cannot lower an API-level
                # effort setting. One degraded retry converts a dead step into
                # an action; the next step returns to full effort.
                forced_think = "low"
                stuck_steps = _score(gain, path, step, kind, messages, stuck_steps)
            else:
                print(f"  (no tool call, finish={finish} -- nudging)", flush=True)
                messages.append({"role": "user", "content":
                                 "Continue. What is the next most valuable action? "
                                 "End this turn with a tool call."})
                stuck_steps = _score(gain, path, step, kind, messages, stuck_steps)
            continue

        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  -> {name}({str(args)[:120]})", flush=True)
            # What the agent ACTUALLY did, for the proposer's repetition check.
            # Scoring against proposals would be circular -- it has to see the
            # real history, including the times the orchestrator overrode the
            # ranking.
            recent_actions.append(
                f"{name}: {str(args.get('statement') or args.get('claim') or args.get('text') or args.get('query') or '')[:140]}")
            del recent_actions[:-20]

            t1 = time.time()
            discharged = False
            result = tools.dispatch(name, args, suppress=suppress)
            since_prove = 0 if name == "prove_theorem" else since_prove + 1
            print(f"     <- [{time.time() - t1:.1f}s] {_brief(result)}", flush=True)
            if name == "prove_theorem":
                # Move the checklist, not just the journal. "Progress" has to
                # mean an obligation changed state; otherwise a turn that
                # happened looks the same as a turn that advanced.
                try:
                    # The agent NAMES the obligation; text overlap is only the
                    # fallback. A helper lemma supporting `potential_decrease`
                    # shares almost no words with it, so inference alone scored
                    # 35 minutes of real work as no progress at all.
                    d = plan.discharge(goal, args.get("statement", ""),
                                       bool(result.get("proved")),
                                       note=str(result.get("note") or "")[:150],
                                       obligation=args.get("obligation", ""))
                    if d.get("matched"):
                        if d.get("status") == plan.PROVED:
                            stuck_steps = 0
                            discharged = bool(result.get("proved"))
                        pr = plan.progress(goal)
                        print(f"     [checklist] {d['matched']} -> {d['status']} "
                              f"({pr['proved']}/{pr['total']} discharged)", flush=True)
                except Exception:
                    pass
                if result.get("proved"):
                    proved += 1
                    print(f"     [LEAN VERIFIED after {len(result.get('attempts', []))} "
                          f"attempt(s)]", flush=True)
                elif result.get("formalization_suspect"):
                    print("     [compiled but MIS-FORMALIZED -- not counted]", flush=True)

            gain.tool(name, args, result, discharged=discharged)
            journal.record_result(name, args, result)
            log(path, {"event": "tool", "step": step, "tool": name,
                       "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(_for_context(result))[:tool_result_max_chars()]})

            # close_session means the agent is DONE with this problem. Without
            # this the loop ran to max_steps regardless, so in a batch the first
            # problem consumed the whole budget and later ones never started.
            if name == "close_session" and result.get("ok"):
                closed = True
                reason = "closed"
                print(f"     [session closed: {args.get('status', '?')}]", flush=True)
                break
        stuck_steps = _score(gain, path, step, kind, messages, stuck_steps)

    # SLEEP PHASE. Wake proved things; sleep looks back over them for recurring
    # structure and tries to abstract it into one reusable lemma. Without it the
    # library only ever accumulates problem-specific results that never transfer
    # -- which is why a "growing" library was not actually compounding.
    #
    # Runs after the problem is finished, never during it: it costs a model call
    # and a full module recompile, and nothing it produces is needed to solve
    # the problem that just ended. Every candidate is verified by Lean inside
    # library.add() before it is kept.
    if config.GROW_LIBRARY and config.LIBRARY_SLEEP:
        try:
            out = library.sleep_phase()
            if out.get("kept"):
                print(f"[sleep] generalised: {', '.join(out['kept'])}", flush=True)
                journal.entry("SUMMARY",
                              "Sleep phase kept generalisations: " + ", ".join(out["kept"]))
            elif out.get("groups_tried"):
                print(f"[sleep] {out['groups_tried']} group(s) examined, "
                      "no verified generalisation", flush=True)
        except Exception as exc:
            print(f"[sleep] skipped ({exc})", flush=True)

    if _stop:
        reason = "interrupted"

    log(path, {"event": "stop", "steps": step, "proved": proved, "return": gain.total,
               "searches": search.used(), "reason": reason})
    print(f"\n{'-' * 60}\nended: {reason} | steps: {step} | elapsed: "
          f"{_hms(time.time() - started)} | lean-verified: {proved} | "
          f"return: {gain.total:+.2f} | "
          f"searches: {search.used()}\nlog: {path}", flush=True)
    return reason
