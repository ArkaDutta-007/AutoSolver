"""Resumable per-problem journal.

Two documents per problem:

  problems/NNN-slug.md    the working journal -- plan / reason / act / reflect,
                          every attempt, every dead end, and why it was abandoned
  solutions/NNN-slug.md   written ONLY on success: a clean step-by-step solution

The journal is the resume mechanism. A later session reads it, learns which
approaches are already exhausted, and continues from there instead of
rediscovering the same dead ends -- which is the failure mode that made attempts
2..N useless in the first measured run.
"""

import hashlib
import json
import os
import re
import time

from . import config

PROBLEMS_DIR = os.path.join(config.ROOT, "problems")
SOLUTIONS_DIR = os.path.join(config.ROOT, "solutions")
STATE_SUFFIX = ".state.json"

_cur: dict = {"path": None, "state": None, "title": None}


def _slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:maxlen].rstrip("-")) or "problem"


def _state_path(md_path: str) -> str:
    return md_path[:-3] + STATE_SUFFIX if md_path.endswith(".md") else md_path + STATE_SUFFIX


def problem_key(statement: str) -> str:
    """Identity comes from the STATEMENT, not from a title the model chose."""
    norm = re.sub(r"[^a-z0-9]+", " ", (statement or "").lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()[:16] if norm else ""


def find_existing(title: str, statement: str = "") -> str:
    """Locate a prior journal for this problem, so work resumes rather than restarts.

    Matching on the model's title forked the record: mid-run the agent renamed
    IMO26_P1 to "IMO26_P1: Game with gcd and lcm/gcd", the slug stopped
    matching, and it opened a SECOND journal for a problem it was already
    working -- splitting the plan from the exploration that produced it.

    So: match on the statement first, and only fall back to the title. A problem
    is what it asks, not what the model decided to call it this turn.
    """
    if not os.path.isdir(PROBLEMS_DIR):
        return ""
    key = problem_key(statement)
    if key:
        for f in sorted(os.listdir(PROBLEMS_DIR)):
            if not f.endswith(STATE_SUFFIX):
                continue
            try:
                st = json.load(open(os.path.join(PROBLEMS_DIR, f), encoding="utf-8"))
            except Exception:
                continue
            if st.get("key") == key or (st.get("statement")
                                        and problem_key(st["statement"]) == key):
                md = os.path.join(PROBLEMS_DIR, f[:-len(STATE_SUFFIX)] + ".md")
                if os.path.exists(md):
                    return md
    want = _slug(title)
    for f in sorted(os.listdir(PROBLEMS_DIR)):
        if f.endswith(".md") and f[4:-3] == want:
            return os.path.join(PROBLEMS_DIR, f)
    return ""


def open_problem(title: str, statement: str = "", source: str = "") -> dict:
    """Open or RESUME a problem journal. Returns prior context when resuming."""
    os.makedirs(PROBLEMS_DIR, exist_ok=True)
    path = find_existing(title, statement)
    resumed = bool(path)

    if not path:
        n = len([f for f in os.listdir(PROBLEMS_DIR) if f.endswith(".md")]) + 1
        path = os.path.join(PROBLEMS_DIR, f"{n:03d}-{_slug(title)}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {title}\n\n")
            if source:
                fh.write(f"*Source: `{source}`*\n\n")
            if statement:
                fh.write(f"## Statement\n\n{statement}\n\n")
            fh.write("## Journal\n\n")

    sp = _state_path(path)
    state = {"title": title, "statement": statement, "source": source, "key": "",
             "sessions": 0, "exhausted": [], "established": [], "solved": False,
             "solution": ""}
    if os.path.exists(sp):
        try:
            state.update(json.load(open(sp, encoding="utf-8")))
        except Exception:
            pass
    state["sessions"] = state.get("sessions", 0) + 1
    state.setdefault("established", [])
    # A resumed problem keeps its ORIGINAL statement. The model retyped the P1
    # statement three different ways across six sessions (`2026` vs `{2026}`,
    # LaTeX vs plain); letting each session overwrite it means the anchor drifts
    # away from the problem actually being solved.
    if resumed and state.get("statement"):
        statement = state["statement"]
    else:
        state["statement"] = statement
    state["key"] = state.get("key") or problem_key(state.get("statement") or statement)
    _cur.update({"path": path, "state": state, "title": title})
    _save()

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n---\n\n### Session {state['sessions']} "
                 f"<sub>{time.strftime('%Y-%m-%d %H:%M:%S')}</sub>\n\n")
        if resumed:
            fh.write(f"Resuming. {len(state['exhausted'])} approach(es) already ruled out.\n\n")

    # Hand the accumulated reflect->plan chain BACK. Without this the agent
    # writes a handoff summary every session and no session ever reads one.
    log_ = state.get("sessions_log", [])
    pending = next((e.get("next_step") for e in reversed(log_) if e.get("next_step")), "")
    return {"path": path, "resumed": resumed, "session": state["sessions"],
            "exhausted": list(state["exhausted"]),
            "established": list(state.get("established", [])),
            "solved": state.get("solved", False),
            "solution": state.get("solution", ""),
            "next_step": pending,
            "prior_sessions": [{k: (str(v)[:600] if k == "summary" else v)
                                for k, v in e.items()} for e in log_[-5:]]}


def _save() -> None:
    if not _cur["path"]:
        return
    with open(_state_path(_cur["path"]), "w", encoding="utf-8") as fh:
        json.dump(_cur["state"], fh, indent=1)


def _write(text: str) -> None:
    if _cur["path"]:
        with open(_cur["path"], "a", encoding="utf-8") as fh:
            fh.write(text)


def entry(kind: str, text: str) -> dict:
    """Record a PLAN / REASON / ACT / REFLECT / SUMMARY step."""
    kind = (kind or "NOTE").upper()
    _write(f"**{kind}** <sub>{time.strftime('%H:%M:%S')}</sub>\n\n{text}\n\n")
    return {"ok": True, "kind": kind, "path": _cur["path"]}


def read(max_chars: int = 0) -> dict:
    """Read the open journal back.

    Without this the agent has no way to see its own record: `open_problem`
    returns the path and the `exhausted` list but not the contents. Observed
    consequence -- it reached for `fetch_url` on a `file://` URL to read a local
    file, burning steps and poking a web tool in a closed-mode run.

    Returns the TAIL when truncated: the most recent work is the relevant part.

    The default used to be 200,000 chars (~57k tokens) and the model called it
    with exactly that -- a single tool result larger than the whole context
    window. The cap is now a fraction of the window actually measured at startup.
    """
    hard_cap = max(4000, config.budget_chars(0.20))
    max_chars = min(int(max_chars) or hard_cap, hard_cap)
    if not _cur["path"]:
        return {"ok": False, "reason": "no open problem -- call open_problem first"}
    try:
        text = open(_cur["path"], encoding="utf-8").read()
    except OSError as exc:
        return {"ok": False, "reason": f"could not read journal: {exc}"}
    truncated = len(text) > max_chars
    return {"ok": True, "path": _cur["path"], "truncated": truncated,
            "chars": len(text), "text": text[-max_chars:] if truncated else text}


def anchor(tail_chars: int = 6000) -> dict:
    """Durable state, re-injected into EVERY turn.

    The agent used to rely on the conversation to remember what it was doing.
    When the window truncated, it did not notice -- it reported "I don't have any
    context from a previous conversation", then invented a different problem to
    work on. Nothing durable is allowed to live only in the context any more.
    """
    st = _cur.get("state") or {}
    tail = ""
    if _cur.get("path"):
        # Seek to the tail rather than reading the file. Every assistant turn is
        # now journalled in full, and one of them ran to 113k characters -- over
        # a long run this file reaches megabytes, and it is re-read on EVERY
        # turn to build the anchor.
        try:
            with open(_cur["path"], "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - tail_chars * 4), os.SEEK_SET)
                tail = fh.read().decode("utf-8", "ignore")[-tail_chars:]
        except OSError:
            tail = ""
    log_ = st.get("sessions_log", [])
    return {"title": st.get("title", ""), "statement": st.get("statement", ""),
            "established": list(st.get("established", [])),
            "exhausted": list(st.get("exhausted", [])),
            "sessions_log": log_,
            "next_step": next((e.get("next_step") for e in reversed(log_)
                               if e.get("next_step")), ""),
            "session": st.get("sessions", 0), "journal_tail": tail}


_STOPWORDS = {"for", "the", "and", "any", "all", "positive", "integers", "integer",
              "non", "negative", "natural", "numbers", "number", "if", "then",
              "both", "of", "is", "a", "b", "m", "n", "that", "we", "let", "define"}


def _content_key(claim: str) -> frozenset:
    """What a claim is ABOUT, ignoring how it was phrased this time."""
    words = re.findall(r"[a-z_]+|[^\w\s]", (claim or "").lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def known(claim: str) -> str:
    """The established fact this claim restates, or "" if it is genuinely new."""
    st = _cur.get("state") or {}
    key = _content_key(claim)
    if not key:
        return ""
    for prior in st.get("established", []):
        pk = _content_key(prior.split("  [")[0])
        if not pk:
            continue
        overlap = len(key & pk) / max(len(key | pk), 1)
        if overlap >= 0.8:
            return prior
    return ""


def establish(claim: str, how: str = "") -> None:
    """Record something now known to be TRUE, so later turns build on it.

    Deduplicated by CONTENT. Five of P1's five `established` entries were the
    same divisibility fact reworded -- "gcd(m,n) divides both m and n", "if
    d = gcd(m,n) then d divides both m and n", and so on. Exact-string matching
    let every rephrasing through, so the list that is supposed to tell the next
    session what is known instead told it the same thing five times and hid how
    little had actually been proved.
    """
    st = _cur.get("state")
    if st is None:
        return
    st.setdefault("established", [])
    if known(claim):
        return
    # A PROVED claim cannot be a dead end. One live run ruled this exact
    # statement out four times and then proved it -- leaving the next session
    # instructed both to build on it and never to attempt it. That contradiction
    # is the documented local-minimum mechanism (the real invariant sitting in
    # `exhausted`, falsely eliminated), so proving something must RETRACT the
    # matching rule-out rather than sit beside it.
    key = _content_key(claim)
    if key:
        before = len(st.get("exhausted") or [])
        st["exhausted"] = [x for x in (st.get("exhausted") or [])
                           if _content_key(x.split("—")[0]) != key]
        if len(st["exhausted"]) != before:
            _write(f"**RETRACTED** <sub>{time.strftime('%H:%M:%S')}</sub>\n\n"
                   f"{before - len(st['exhausted'])} rule-out(s) withdrawn: the claim "
                   f"was PROVED.\n\n")
    note = f"{claim.strip()}" + (f"  [{how.strip()}]" if how else "")
    if note not in st["established"]:
        st["established"].append(note)
        _save()


def rule_out(approach: str, why: str) -> dict:
    """Mark an approach exhausted. Carried into every future session."""
    if not _cur["state"]:
        return {"ok": False, "reason": "no open problem"}
    note = f"{approach.strip()} — {why.strip()}"
    # CONTENT-keyed, not string-keyed. Exact-match dedup let one live run record
    # the SAME claim four times, twice over: eight `exhausted` entries encoding
    # two distinct claims, because each rule_out phrased the reason slightly
    # differently. An inflated dead-end list is not merely noisy -- every future
    # session is told "do NOT repeat these".
    key = _content_key(approach)
    dup = any(key and _content_key(x.split("—")[0]) == key
              for x in _cur["state"]["exhausted"])
    if not dup and note not in _cur["state"]["exhausted"]:
        _cur["state"]["exhausted"].append(note)
        _save()
    _write(f"**RULED OUT** <sub>{time.strftime('%H:%M:%S')}</sub>\n\n{note}\n\n")
    return {"ok": True, "exhausted_count": len(_cur["state"]["exhausted"])}


def _attempt_table(tries: list) -> str:
    """Every attempt, in full. This is what a later session reads to avoid
    re-walking the same basins, so it is deliberately not summarized."""
    if not tries:
        return "_no attempts recorded_\n"
    cols = ["attempt", "model", "temp", "mode", "ok", "rounds", "formalization", "reason"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in tries:
        row = [str(t.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n" + _rounds_detail(tries)


def _rounds_detail(tries: list) -> str:
    """What the Lean compiler actually said, round by round.

    The summary table says an attempt failed; it cannot say WHY, and "why" is
    the only part a later session can learn from. These `_`-prefixed fields are
    written here in full and stripped before the result reaches the model, so
    the record keeps growing while the context stays small.
    """
    blocks = []
    for t in tries:
        rounds = t.get("_rounds") or []
        tried = t.get("_tried")
        if tried is not None:
            blocks.append(f"\n<details><summary>Attempt {t.get('attempt')} — tactic probe, "
                          f"{len(tried)} rung(s) completed</summary>\n\n"
                          f"- tried: `{', '.join(tried) or 'none'}`\n"
                          + (f"- never ran (budget): `{', '.join(t['_not_run'])}`\n"
                             if t.get("_not_run") else "")
                          + "\n</details>\n")
            continue
        if not rounds:
            continue
        body = [f"\n<details><summary>Attempt {t.get('attempt')} — "
                f"{t.get('_strategy', 'direct')[:70]} — {len(rounds)} round(s)</summary>\n"]
        for r in rounds:
            if r.get("ok"):
                body.append(f"\n**round {r.get('round')}: COMPILED** "
                            f"({r.get('elapsed')}s)\n\n```lean\n{r.get('lean', '')}\n```\n")
            else:
                body.append(f"\n**round {r.get('round')}: rejected** ({r.get('elapsed')}s)\n\n"
                            f"```\n{r.get('errors', '')}\n```\n")
        body.append("\n</details>\n")
        blocks.append("".join(body))
    return "".join(blocks)


def record_result(name: str, args: dict, result: dict) -> None:
    """Auto-log a tool call, and rule out approaches that provably failed.

    Nothing durable is truncated: the journal is the whole point of the next
    session, and a clipped compiler error or half a claim is exactly the detail
    that would have saved it a repeat.
    """
    if not _cur["path"]:
        return
    if name != "prove_theorem":
        # read_journal returns the journal itself -- writing that back in would
        # double the file on every call. Record the metadata, not the echo.
        shown = ({k: v for k, v in result.items() if k != "text"}
                 if name == "read_journal" else result)
        _write(f"**ACT** `{name}` <sub>{time.strftime('%H:%M:%S')}</sub>\n\n"
               f"- args: `{args}`\n- result: `{shown}`\n\n")
        return

    claim = str(args.get("statement", ""))
    pinned = str(args.get("lean_statement", "") or "")
    head = f"**ACT** `prove_theorem` <sub>{time.strftime('%H:%M:%S')}</sub>\n\n"
    body = f"Claim: {claim}\n\n" + (f"Pinned signature:\n```lean\n{pinned}\n```\n\n" if pinned else "")

    if result.get("proved"):
        establish(claim[:300], f"Lean-verified via {result.get('solved_by') or 'prover'}")
        form = result.get("formalization") or {}
        _write(f"{head}**VERIFIED** via {result.get('solved_by') or 'prover'} in "
               f"{result.get('elapsed')}s ({result.get('rounds')} round(s))\n\n{body}"
               f"Formalization check: {form.get('verdict', 'n/a')}\n\n"
               f"```lean\n{result.get('lean', '')}\n```\n\n"
               f"{_attempt_table(result.get('attempts', []))}\n")
        return

    tries = result.get("attempts", [])
    reasons = sorted({str(t.get("reason")) for t in tries if t.get("reason")})
    _write(f"{head}**NOT PROVED** after {len(tries)} attempt(s), {result.get('elapsed')}s\n\n"
           f"{body}{result.get('note', '')}\n\n{_attempt_table(tries)}\n")
    for r in reasons:
        rule_out(f"prove_theorem on: {claim}", r)


def solve(solution_md: str, lean: str = "") -> dict:
    """Write the clean step-by-step solution document. Called on success."""
    if not _cur["path"]:
        return {"ok": False, "reason": "no open problem"}
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    name = os.path.basename(_cur["path"])
    out = os.path.join(SOLUTIONS_DIR, name)
    st = _cur["state"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Solution — {st.get('title')}\n\n")
        if st.get("source"):
            fh.write(f"*Source: `{st['source']}`*\n\n")
        if st.get("statement"):
            fh.write(f"## Problem\n\n{st['statement']}\n\n")
        fh.write(f"## Solution\n\n{solution_md.strip()}\n\n")
        if lean:
            fh.write(f"## Verified Lean\n\n```lean\n{lean.strip()}\n```\n\n")
        fh.write(f"*Solved after {st.get('sessions', 1)} session(s). "
                 f"Journal: `problems/{name}`*\n")

    st["solved"] = True
    st["solution"] = out
    _save()
    _write(f"**SOLVED** — clean solution written to `solutions/{name}`\n\n")
    return {"ok": True, "path": out}


def close(status: str, summary: str, next_step: str = "") -> dict:
    """End the session: REFLECT on what happened, PLAN what the next one does.

    This closes the reflect -> plan -> act loop that was previously open at both
    ends. `last_summary` OVERWROTE its predecessor, so session 6 destroyed
    session 5's reflection; and `open_problem` never returned it, so the handoff
    was written and then never read by anybody. Six sessions of IMO P1 each
    began from nothing as a result.

    Summaries now ACCUMULATE in `sessions_log`, and `open_problem` hands the
    chain back on resume. `next_step` is separate from `summary` on purpose:
    "what I learned" and "what to do about it" are different questions, and the
    second is the one the next session has to act on.
    """
    entry_ = {"session": (_cur["state"] or {}).get("sessions", 0),
              "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
              "status": status, "summary": summary.strip(),
              "next_step": next_step.strip()}

    _write(f"\n**SESSION SUMMARY ({status})** <sub>{time.strftime('%H:%M:%S')}</sub>\n\n"
           f"{summary}\n\n"
           + (f"**NEXT STEP** — {next_step}\n\n" if next_step else
              "_no next step named — the next session starts blind_\n\n"))

    if _cur["state"] is not None:
        _cur["state"].setdefault("sessions_log", []).append(entry_)
        _cur["state"]["last_summary"] = summary        # back-compat
        _save()
    path = _cur["path"]
    _cur.update({"path": None, "state": None, "title": None})
    return {"ok": True, "path": path, "status": status,
            "sessions_recorded": len((_cur["state"] or {}).get("sessions_log", []) or []) or None}
