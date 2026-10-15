#!/usr/bin/env python3
"""No grammar-constrained call may also carry a reasoning budget.

WHY THIS IS A WHOLE TEST FILE
-----------------------------
This bug has now been found SIX times, in six different modules, each time by
watching a run produce nothing and then working backwards. The mechanism is
always identical: Ollama bills reasoning tokens and response tokens against ONE
`num_predict` allowance. When the response is additionally constrained by a JSON
schema, the model thinks, exhausts the budget, and the JSON is cut off
mid-object. `json.loads` throws, the caller's `except` swallows it, and the call
returns nothing -- silently, because an empty result prints nothing.

The occurrences:

  plan.py         an empty ledger cached permanently; closure disabled for a run
  propose.py      pass@K produced zero candidates
  intuition.py    burned ~8 of every 9 minutes producing no hunches
  prover.py       a signature derivation ran 2,457 seconds and returned ""
                  -- 22,000 tokens at the measured 9.2 tok/s, all of it thinking
  library.py      llm_generalizer, latent: sleep_phase had never run
  review.py       WORST: no `think` key at all, so the model's own default
                  applied. The truncated JSON fell through to a prose-scraping
                  fallback, which extracted a verdict from unfinished reasoning.
                  Three draws on one hand-checked faithful signature returned
                  MISMATCH, FAITHFUL, FAITHFUL -- a coin-flip on the gate that
                  decides whether a proof counts as real.

That last one is why ABSENCE of a think key is treated as a failure here and not
as a safe default. review.py passed the first version of this test.

Fixing them one at a time has not worked, because each fix only closed the site
that had already cost a run. This test closes the CLASS: it walks the AST of
every module and fails on any `requests.post` whose payload sets `format`
without an explicit `think: False`.

If a call site ever genuinely needs both -- a long-form reasoning task whose
answer must still be schema-shaped -- add it to ALLOWED with a comment saying
what budget makes it safe. Deliberate exceptions are fine. Silent ones are what
this exists to stop.

    python3 -m tests.test_schema_calls
"""

import ast
import glob
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (module, function) pairs deliberately exempt. Empty, and it should stay that
# way until someone has MEASURED that a given site survives both.
ALLOWED: set = set()


def _enclosing_function(tree, lineno: str) -> str:
    best = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno and (not best or node.lineno > best[1]):
                best = (node.name, node.lineno)
    return best[0] if best else "<module>"


def schema_calls() -> list:
    """Every requests.post that constrains output with a `format` schema."""
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "*.py"))):
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "post"):
                continue
            for kw in node.keywords:
                if kw.arg != "json" or not isinstance(kw.value, ast.Dict):
                    continue
                keys = [k.value if isinstance(k, ast.Constant) else "**"
                        for k in kw.value.keys]
                if "format" not in keys:
                    continue
                think, spread = None, "**" in keys
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and k.value == "think":
                        think = v
                found.append({
                    "module": os.path.basename(path),
                    "line": node.lineno,
                    "func": _enclosing_function(tree, node.lineno),
                    "think": think,
                    "spread": spread,
                })
    return found


class TestNoThinkingUnderASchema(unittest.TestCase):
    def test_at_least_one_schema_call_exists(self):
        # Guards the guard: if the AST walk silently stops matching (an API
        # change, a refactor to a helper), every other test here would pass
        # vacuously and the class would reopen unnoticed.
        self.assertGreater(len(schema_calls()), 3)

    def test_no_schema_call_carries_a_truthy_think(self):
        bad = []
        for c in schema_calls():
            if (c["module"], c["func"]) in ALLOWED:
                continue
            t = c["think"]
            if t is None:
                # NOT safe. An absent key defers to the model's default, and
                # Qwen3.8 reasons by default -- which is exactly how review.py
                # shipped a coin-flip faithfulness gate while passing this test.
                bad.append(f"{c['module']}:{c['line']} in {c['func']}() "
                           f"-> no think key; the model default applies")
                continue
            literal_false = isinstance(t, ast.Constant) and t.value is False
            if not literal_false:
                bad.append(f"{c['module']}:{c['line']} in {c['func']}() "
                           f"-> think={ast.unparse(t)}")
        self.assertEqual(bad, [], "grammar-constrained calls carrying a "
                                  "reasoning budget:\n  " + "\n  ".join(bad))

    def test_no_schema_call_splats_an_unknown_dict_that_could_inject_think(self):
        # `**_think_opt()` is exactly how prover.py acquired think=high without
        # any line of code visibly saying so -- which is why reading the call
        # site did not reveal the bug and the trace did.
        bad = [f"{c['module']}:{c['line']} in {c['func']}()"
               for c in schema_calls()
               if c["spread"] and (c["module"], c["func"]) not in ALLOWED]
        self.assertEqual(bad, [], "schema calls splatting a dict that may "
                                  "contain `think`:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
