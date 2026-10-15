#!/usr/bin/env python3
"""Entry point.

    python3 run.py "a research goal"                    # free-form goal
    python3 run.py --questions                          # every file in questions/
    python3 run.py --questions exam.pdf                 # one file
    python3 run.py --questions a.txt b.pdf shots/       # several, formats mixed
    python3 run.py --questions 'imo/*.png'              # a glob (quote it)
    python3 run.py --manifest set.list                  # a curated ordered list
    python3 run.py --questions --list                   # parse only, show what was found
    python3 run.py --resume                             # continue unsolved problems

By default the web tools are WITHHELD: anything with a published solution must be
solved by the model itself. Pass --open only for genuinely unsolved problems,
where there is no solution to leak and literature search is pure upside.
"""

import argparse
import json
import os

from src import agent, config, inputs, journal, preflight, search


def _list(problems: list) -> None:
    for p in problems:
        print(f"  [{p['title']}]  ({p['source']})")
        print(f"      {p['text'][:140].replace(chr(10), ' ')}...")


def _unsolved() -> list:
    """Problems with a journal that are not yet solved -- the resume set."""
    out = []
    if not os.path.isdir(journal.PROBLEMS_DIR):
        return out
    for f in sorted(os.listdir(journal.PROBLEMS_DIR)):
        if not f.endswith(journal.STATE_SUFFIX):
            continue
        try:
            st = json.load(open(os.path.join(journal.PROBLEMS_DIR, f), encoding="utf-8"))
        except Exception:
            continue
        if not st.get("solved"):
            out.append({"title": st.get("title", ""), "text": st.get("statement", ""),
                        "source": st.get("source", "")})
    return out


def _quarantine(slugs: list) -> None:
    """Park a problem journal so --resume stops picking it up.

    Needed because the agent could invent problems: a truncated context produced
    `002-continuing-interrupted-work` and `003-fundamental-theorem-of-arithmetic`,
    neither of which anyone asked for, and both were then permanently in the
    resume set. Moved, not deleted -- they are evidence.
    """
    dest = os.path.join(journal.PROBLEMS_DIR, "quarantine")
    if not slugs:
        print("problems available to quarantine:")
        for f in sorted(os.listdir(journal.PROBLEMS_DIR)):
            if f.endswith(".md"):
                print(f"  {f[:-3]}")
        print("\nusage: python3 run.py --quarantine 002-continuing-interrupted-work")
        return
    os.makedirs(dest, exist_ok=True)
    for slug in slugs:
        moved = 0
        for ext in (".md", journal.STATE_SUFFIX):
            src = os.path.join(journal.PROBLEMS_DIR, slug + ext)
            if os.path.exists(src):
                os.replace(src, os.path.join(dest, slug + ext))
                moved += 1
        print(f"  {slug}: {'quarantined' if moved else 'NOT FOUND'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", nargs="*", help="free-form research goal")
    ap.add_argument("--questions", nargs="*", metavar="TARGET",
                    help="files, directories, or globs to read problems from "
                         "(no argument = everything in questions/)")
    ap.add_argument("--manifest", metavar="FILE",
                    help="file listing targets, one per line (# comments allowed)")
    ap.add_argument("--resume", action="store_true",
                    help="continue every problem whose journal is not yet solved")
    ap.add_argument("--list", action="store_true", help="parse and list; do not run")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = perpetual")
    ap.add_argument("--problem-minutes", type=int, default=0,
                    help="wall-clock cap per problem in a batch (0 = unlimited). "
                         "Without it one problem can consume the whole overnight run.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the context-window check. You almost certainly do not "
                         "want this: an undersized window makes the agent confabulate "
                         "rather than fail.")
    ap.add_argument("--quarantine", nargs="*", metavar="SLUG",
                    help="move problem journals out of the resume set, into "
                         "problems/quarantine/. No argument = list candidates.")
    ap.add_argument("--open", dest="open_problem", action="store_true",
                    help="this is an OPEN (unsolved) problem: allow web search. "
                         "Omit for anything with a published solution -- the web "
                         "tools are then withheld entirely, not merely blocked.")
    ap.add_argument("--search", dest="open_problem", action="store_true",
                    help=argparse.SUPPRESS)   # back-compat alias
    a = ap.parse_args()

    if a.quarantine is not None:
        _quarantine(a.quarantine)
        return

    if a.open_problem:
        search.set_enabled(True)
        print("OPEN-PROBLEM mode: web search ENABLED (no published solution to leak)")
    else:
        search.set_enabled(False)
        print("closed mode: web tools withheld -- the model must solve it itself")

    problems = []
    if a.resume:
        problems = _unsolved()
        print(f"resuming {len(problems)} unsolved problem(s)")
    elif a.manifest or a.questions is not None:
        targets = list(a.questions or [])
        if a.manifest:
            targets += inputs.read_manifest(a.manifest)
        problems = inputs.load(targets or None)
        where = ", ".join(targets) if targets else inputs.QUESTIONS_DIR
        print(f"loaded {len(problems)} problem(s) from {where}")

    if a.list:
        _list(problems)
        return

    # Questions first (above): a screenshot or scan is read by the vision model
    # and that model unloaded BEFORE anything else takes the memory. Then a
    # cheap server check -- no context probes; the window is asked for, 128K.
    if not a.skip_preflight:
        info = preflight.run()
        if info.get("context_tokens"):
            config.set_context_tokens(info["context_tokens"])

    if problems:
        for i, p in enumerate(problems, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(problems)}] {p['title']}\n{'=' * 70}", flush=True)
            goal = (f"Solve this problem. Title: {p['title']}\n"
                    f"Source: {p.get('source', 'n/a')}\n\n{p['text']}\n\n"
                    "Open its journal first; if earlier sessions ruled approaches out, "
                    "do not repeat them. Journal your plan and reasoning as you go. "
                    "On success write the clean solution document.\n\n"
                    "Do NOT try to solve the whole problem in one turn. Take ONE "
                    "concrete step now and end the turn with a tool call -- "
                    "journal_entry to bank a plan, or prove_theorem / run_python on "
                    "the smallest claim worth checking. You have many turns.")
            reason = agent.run(goal, a.max_steps,
                               time_budget=a.problem_minutes * 60 or config.PROBLEM_TIME_BUDGET)
            # Ctrl-C means stop the batch, not "skip to the next problem with a
            # stop flag already set" -- which silently ran the rest at 0 steps.
            if reason == "interrupted":
                print(f"\ninterrupted -- {len(problems) - i} problem(s) not started",
                      flush=True)
                break
        return

    if not a.goal:
        ap.error("give a goal, or --questions, or --manifest, or --resume")
    agent.run(" ".join(a.goal), a.max_steps)


if __name__ == "__main__":
    main()
