# lean

The Lean 4 project AutoSolver checks proofs in. It pins Lean and Mathlib
v4.33.0.

- `Proofs/Verified.lean` is the library of lemmas the agent has proved. It
  grows as the agent runs, and later proofs can `import Proofs.Verified`.
- `Narrow.lean` and `Test.lean` hold the same small theorem with a narrow
  import and with `import Mathlib`, for timing the import cost.
- `lean_check.py` is a standalone `check_lean(code)` helper, independent of
  the agent.

Set up with `lake exe cache get && lake build`. The `.lake/` directory (about
7 GB of Mathlib build files) is rebuilt by those commands and is not committed.
