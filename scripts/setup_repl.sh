#!/usr/bin/env bash
# Build leanprover-community/repl so Lean checks stop paying `import Mathlib`.
#
# WHY
#   ~25-29s with `import Mathlib` vs ~1.2s narrow, paid PER `lake env lean`.
#   One prove_theorem call pays it 1x to validate the signature, up to 3x to
#   recover a bad one, and 12x for the tactic ladder. The REPL imports Mathlib
#   once at startup and reuses that environment for every later check.
#
# USAGE
#   ./scripts/setup_repl.sh              # build against lean/'s toolchain
#
# Safe to skip: src/lean.py falls back to `lake env lean` when the binary is
# absent, so nothing breaks if this never runs.

set -euo pipefail
PROOFS="${LEAN_PROJECT:-$(cd "$(dirname "$0")/../lean" && pwd)}"
REPL_DIR="$PROOFS/.lake/repl"

[ -d "$PROOFS" ] || { echo "no Lean project at $PROOFS (set LEAN_PROJECT)"; exit 1; }

TOOLCHAIN="$(cat "$PROOFS/lean-toolchain")"
echo "Lean project : $PROOFS"
echo "toolchain    : $TOOLCHAIN"

if [ ! -d "$REPL_DIR/.git" ]; then
  echo "cloning leanprover-community/repl ..."
  git clone --quiet https://github.com/leanprover-community/repl.git "$REPL_DIR"
fi

# The REPL must be built with the SAME toolchain as Mathlib, or the oleans it
# loads will be rejected.
echo "$TOOLCHAIN" > "$REPL_DIR/lean-toolchain"
echo "building (first time takes a few minutes) ..."
( cd "$REPL_DIR" && lake build repl >/dev/null )

BIN="$REPL_DIR/.lake/build/bin/repl"
[ -x "$BIN" ] || { echo "build finished but $BIN is missing"; exit 1; }

cd "$(dirname "$0")/.."
touch .env; chmod 600 .env
tmp="$(mktemp "${TMPDIR:-/tmp}/autoresearch-env.XXXXXX")"
grep -v -E '^LEAN_REPL_PATH=' .env > "$tmp" || true
printf 'LEAN_REPL_PATH=%s\n' "$BIN" >> "$tmp"
cat "$tmp" > .env
rm -f "$tmp"

echo
echo "built: $BIN"
echo ".env updated with LEAN_REPL_PATH"
echo
echo "verify (first call boots Mathlib, later ones should be ~1s):"
echo "  python3 -c \"from src.lean import check_lean; import time"
echo "for i in range(3):"
echo "    t=time.time(); r=check_lean('import Mathlib\\ntheorem t (n : Nat) : n = n := by rfl')"
echo "    print(i, r['ok'], round(time.time()-t,2), 's')\""
