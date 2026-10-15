#!/usr/bin/env python3
"""Regression tests for the 2026-08-14 hardening pass.

Every test here corresponds to something that actually went wrong in
`runs/run-20260813-171447.jsonl`. Run offline; no model, no Lean, no network.

    python3 tests/hardening.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import agent, cache, config, journal, lean, sandbox, tools  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


print("\ncache: cosmetic drift must not create a second cache entry")
a = "There are 2026 integers greater than 1"
b = "There are {2026} integers greater than {1}"
c = r"There are \displaystyle 2026 integers   greater than 1"
check("LaTeX braces normalize", cache._key(a) == cache._key(b))
check("whitespace + macros normalize", cache._key(a) == cache._key(c))
check("different claims stay different", cache._key(a) != cache._key("There are 7 integers"))
check("lean signature is NOT brace-stripped",
      cache._key(a, "theorem t {n : N} : P n") != cache._key(a, "theorem t (n : N) : P n"))

print("\nlean: 'error' as a substring must not fail a good proof")
check("real diagnostic detected",
      bool(lean._ERROR_LINE.search("tmp.lean:12:4: error: unknown constant `foo`")))
check("the word 'error' alone does not",
      not lean._ERROR_LINE.search("'my_error_bound' depends on axioms: [propext]"))
check("warnings are not errors",
      not lean._ERROR_LINE.search("tmp.lean:3:0: warning: declaration uses 'sorry'"))

print("\nlean: axiom audit")
check("axiom line parses",
      lean._AXIOM_LINE.findall("'foo' depends on axioms: [propext, Classical.choice]")
      == ["propext, Classical.choice"])
check("locally declared axiom is caught",
      not lean.audit_axioms("axiom cheat : False\ntheorem t : False := cheat")["ok"])
check("native_decide is caught",
      not lean.audit_axioms("theorem t : 2 + 2 = 4 := by native_decide")["ok"])
check("audit fails OPEN when there is nothing to check",
      lean.audit_axioms("-- just a comment")["ok"])

print("\nconfig: budgets track the measured window, not a fictional one")
config.set_context_tokens(4_000)
small_tool, small_compact = agent.tool_result_max_chars(), agent.compact_at_chars()
config.set_context_tokens(128_000)
big_tool, big_compact = agent.tool_result_max_chars(), agent.compact_at_chars()
check("tool-result cap scales with context", big_tool > small_tool * 4, f"{small_tool} -> {big_tool}")
check("compaction threshold scales", big_compact > small_compact * 4)
check("a single tool result can never fill the window",
      big_tool < config.CONTEXT_TOKENS * config.CHARS_PER_TOKEN * 0.5)

print("\nagent: oversized reasoning is trimmed, recent turns are kept")
msgs = [{"role": "system", "content": "s"}]
for i in range(4):
    msgs.append({"role": "assistant", "content": "X" * 50_000})
out = agent._strip_stale_reasoning(msgs, keep_last=2)
kept = [m for m in out if len(str(m["content"])) > 40_000]
check("only the 2 most recent long turns survive in full", len(kept) == 2, f"kept {len(kept)}")
check("trimmed turns keep a pointer to the journal",
      "journal" in out[1]["content"])
check("tool_calls turns are never trimmed",
      agent._strip_stale_reasoning(
          [{"role": "assistant", "content": "Y" * 50_000, "tool_calls": [1]}],
          keep_last=0)[0]["content"] == "Y" * 50_000)

print("\njournal: durable state survives a wiped context")
tmp = tempfile.mkdtemp()
journal.PROBLEMS_DIR = os.path.join(tmp, "problems")
journal.SOLUTIONS_DIR = os.path.join(tmp, "solutions")
journal.open_problem("IMO26_P1", statement="There are 2026 integers greater than 1.")
journal.establish("gcd of the exponent multiset is invariant", "Lean-verified")
journal.rule_out("brute force over all boards", "2026 numbers is not enumerable")
a = journal.anchor()
check("statement is recoverable", "2026 integers" in a["statement"])
check("established survives", any("invariant" in e for e in a["established"]))
check("ruled-out survives", any("brute force" in e for e in a["exhausted"]))

# The exact drift seen across sessions 3-6: the model retyped the statement.
journal.open_problem("IMO26_P1", statement="There are {2026} integers greater than {1}. GARBLED")
check("resuming does NOT let a retyped statement overwrite the original",
      "GARBLED" not in journal.anchor()["statement"])

anchor_msg = agent._anchor_message()
check("anchor message carries the problem", "2026 integers" in anchor_msg["content"])
check("anchor message is labelled for the stripper",
      anchor_msg["content"].startswith("== PROBLEM ("))

print("\njournal: read() cannot return more than the window holds")
config.set_context_tokens(32_000)
r = journal.read(max_chars=200_000)          # what the model actually asked for
check("oversized read request is clamped",
      len(r.get("text", "")) <= config.budget_chars(0.20) + 1,
      f"got {len(r.get('text',''))}")

print("\nsandbox: the agent can finally compute something")
r = sandbox.run_python("print(gcd(12, 18), lcm(4, 6))")
check("preloaded math helpers work", r["ok"] and r["stdout"].strip() == "6 12", str(r)[:200])
r = sandbox.run_python("import sympy; print(sympy.factorint(360))")
check("sympy available", r["ok"] and "3: 2" in r["stdout"], str(r)[:200])
r = sandbox.run_python("while True: pass", timeout=3)
check("runaway loop is killed", not r["ok"] and "timed out" in r.get("error", ""))
r = sandbox.run_python("raise ValueError('boom')")
check("errors are reported, not raised", not r["ok"] and "boom" in r["stderr"])

# Model-written code: computes, but cannot reach out or read the user's files.
if os.path.exists(sandbox.SANDBOX):
    for label, code in [
        ("network", "import socket\ntry:\n socket.create_connection(('1.1.1.1', 80), 3); print('OPEN')\n"
                    "except OSError: print('BLOCKED')"),
        ("spawn", "import os\nprint('OPEN' if os.system('/bin/echo x') == 0 else 'BLOCKED')"),
        ("home read", f"try:\n open({os.path.expanduser('~/.zshrc')!r}).read(1); print('OPEN')\n"
                      f"except OSError: print('BLOCKED')"),
        ("repo .env", f"try:\n open({os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')!r}).read(1); print('OPEN')\n"
                      f"except OSError: print('BLOCKED')"),
        ("write outside", f"try:\n open({os.path.expanduser('~/autosolver-escape')!r}, 'w'); print('OPEN')\n"
                          f"except OSError: print('BLOCKED')"),
        ("LaunchServices", "try:\n from AppKit import NSWorkspace\n from Foundation import NSURL\n"
                           " u = NSURL.URLWithString_('https://example.com')\n"
                           " print('OPEN' if NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(u) else 'BLOCKED')\n"
                           "except ImportError: print('BLOCKED')")]:
        r = sandbox.run_python(code, timeout=30)
        check(f"sandbox blocks {label}", r["stdout"].strip().endswith("BLOCKED"), str(r)[:200])
    check("no escape file", not os.path.exists(os.path.expanduser("~/autosolver-escape")))
    r = sandbox.run_python("open('scratch.txt','w').write('x'); print(open('scratch.txt').read())")
    check("sandbox writes in its own dir", r["stdout"].strip() == "x", str(r)[:200])

# The exploration that would have cracked P1 on day one.
r = sandbox.run_python("""
import random
def move(b, i, j):
    m, n = b[i], b[j]
    g = gcd(m, n)
    b = list(b); b[i], b[j] = g, (m * n // g) // g
    return b
def fixpoint(b):
    b = list(b)
    for _ in range(4000):
        opts = [(i, j) for i in range(len(b)) for j in range(len(b))
                if i != j and b[i] > 1 and b[j] > 1]
        if not opts: break
        b = move(b, *random.choice(opts))
    return sorted(b)
start = [12, 18, 50, 7]
outs = {tuple(fixpoint(start)) for _ in range(40)}
print("distinct fixpoints:", len(outs), outs)
""", timeout=45)
check("P1 invariant is discoverable empirically",
      r["ok"] and "distinct fixpoints: 1" in r["stdout"], str(r)[:300])

print("\ntools: run_python is offered and dispatchable")
names = [t["function"]["name"] for t in tools.specs(open_problem_mode=False)]
check("run_python offered in closed mode", "run_python" in names)
check("web tools still withheld in closed mode",
      "web_search" not in names and "fetch_url" not in names)
check("dispatch works", tools.dispatch("run_python", {"code": "print(6*7)"})["stdout"] == "42")
check("unoffered tools still rejected",
      "error" in tools.dispatch("web_search", {"query": "x"}))

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
