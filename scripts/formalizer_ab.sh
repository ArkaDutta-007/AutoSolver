#!/usr/bin/env bash
# Bring up the tuned formalizer and run the paired A/B against the 27B baseline.
#
# THE DECISION RULE IS FIXED HERE, BEFORE THE NUMBER IS SEEN.
# The 210-item benchmark detects a 7-point absolute T2 shift (57 discordant
# pairs at the measured 27% per-item flip rate). The 27B baseline is T2 = 58/210
# = 28%. So the specialist keeps `derive_signature` only at T2 >= 65/210 (31%+
# is inside noise); anything less is reported as a null, exactly as the naming
# rule and the instance repair were.
#
# Writing the threshold into the script rather than deciding after the fact is
# the point -- this project has twice produced a promising partial signal that
# evaporated at full n.
#
#   ./scripts/formalizer_ab.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GOEDEL_HOST="${GOEDEL_HOST:-$(sed -n 's/^GOEDEL_HOST=//p' "$(dirname "$0")/../.env" 2>/dev/null | tail -1 || true)}"
REMOTE="${GOEDEL_HOST:?set GOEDEL_HOST (in the environment or .env) to the GPU server ssh host}"
BASE_T2=58                      # 27B, n=210, measured
NEED=7                          # percentage points, from the power calculation
THRESHOLD=$(( BASE_T2 + (210 * NEED + 99) / 100 ))

echo "== 1. adapter present on the remote? =="
ssh -o BatchMode=yes "$REMOTE" 'ls ~/formalizer/adapter_v3/adapter_model.safetensors' >/dev/null 2>&1 || {
  echo "no adapter on the remote yet -- training has not finished, or it has not been copied over"
  exit 1; }

echo "== 2. serve on GPU 0 (port 8002) and tunnel =="
SERVE_SCRIPT=serve_formalizer.sh GOEDEL_PORT=8002 ./scripts/goedel_tunnel.sh up

echo "== 3. smoke: one derivation, greedy =="
FORMALIZER_BASE=http://127.0.0.1:8002/v1 python3 - <<'PY'
from src import formalizer
out = formalizer.derive("The greatest common divisor of two natural numbers "
                        "divides each of them.")
print("  ->", out[:180] or "(EMPTY -- endpoint or prompt problem)")
assert out.strip(), "specialist returned nothing; do not start the A/B"
PY

echo "== 4. paired A/B on the same 210 items =="
echo "   baseline T2 = ${BASE_T2}/210 (28%)"
echo "   specialist keeps the job only at T2 >= ${THRESHOLD}/210"
FORMALIZER_BASE=http://127.0.0.1:8002/v1 \
  python3 -u evals/formalize_score.py --n 210 --workers 2 \
    --out evals/formalize_score_specialist.json

echo "== 5. verdict =="
python3 - "$THRESHOLD" <<'PY'
import json, sys
th = int(sys.argv[1])
a = json.load(open("evals/formalize_score_210.json"))
b = json.load(open("evals/formalize_score_specialist.json"))
bykey = {r["name"]: r for r in a}
pairs = [(bykey[r["name"]], r) for r in b if r["name"] in bykey]
t2a = sum(1 for x, _ in pairs if x.get("t2"))
t2b = sum(1 for _, y in pairs if y.get("t2"))
gain = sum(1 for x, y in pairs if y.get("t2") and not x.get("t2"))
loss = sum(1 for x, y in pairs if x.get("t2") and not y.get("t2"))
print(f"  paired on {len(pairs)} items")
print(f"  27B  T2 {t2a}   specialist T2 {t2b}")
print(f"  discordant: +{gain} / -{loss}   (McNemar needs |gain-loss| >~ "
      f"{2*(gain+loss)**0.5:.0f})")
verdict = ("SPECIALIST WINS -- route derive_signature through it"
           if t2b >= th and abs(gain - loss) >= 2 * (gain + loss) ** 0.5
           else "NULL -- the 27B keeps derive_signature")
print(f"  {verdict}")
PY
