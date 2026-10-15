#!/bin/bash
# Goedel prover tuning on the lab GPU box, driven from the Mac. One word per action:
#
#   bash scripts/oai_run.sh start [args]   sync scripts, launch the whole pipeline in the background
#                                          (refuses, starting nothing, while the GPUs are in use)
#   bash scripts/oai_run.sh iterate [args] the same, then rounds 2+ and the 16-sample final evaluation
#                                          (args: --rounds 3 --budget 110)
#   bash scripts/oai_run.sh status [run]   stages, progress, budget left, GPU memory; the report when done
#   bash scripts/oai_run.sh stop [run]     stop the pipeline and its vLLM server (recorded PIDs only)
#   bash scripts/oai_run.sh selftest       CPU-only end to end: mock server, 2-layer model, real Lean
#   bash scripts/oai_run.sh bundle [dest]  scripts + data for Colab (default ~/Downloads/oaimath_bundle.tgz)
#
# start args go to oai_pipeline.py:  start --gpus 0,1,2   start --train-gpus 0,1,2   start --scale 0.5
set -euo pipefail
GOEDEL_HOST="${GOEDEL_HOST:-$(sed -n 's/^GOEDEL_HOST=//p' "$(dirname "$0")/../.env" 2>/dev/null | tail -1 || true)}"
HOST="${GOEDEL_HOST:?set GOEDEL_HOST (in the environment or .env) to the GPU server ssh host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
PY='~/venvs/train/bin/python'
FILES=(oai_pipeline.py oai_sample.py oai_extract.py oai_mock_vllm.py train_prover.py oai_colab_setup.sh)
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$HOST" "$@"; }
push() { (cd "$DIR" && scp -q "${FILES[@]}" "$HOST:oaimath/"); }
act="${1:-status}"; shift || true
case "$act" in
  start|selftest|iterate)
    push
    profile=lab; out=run; cmd=run
    if [ "$act" = selftest ]; then profile=selftest; out=selftest; fi
    if [ "$act" = iterate ]; then cmd=iterate; fi
    on "cd ~/oaimath; mkdir -p $out; setsid nohup $PY oai_pipeline.py $cmd --profile $profile --out ~/oaimath/$out $* >> $out/pipeline.log 2>&1 < /dev/null & sleep 20; tail -n 12 ~/oaimath/$out/pipeline.log" ;;
  status|stop)
    run="${1:-run}"; profile=lab
    if [ "$run" = selftest ]; then profile=selftest; fi
    on "cd ~/oaimath; $PY oai_pipeline.py $act --profile $profile --out ~/oaimath/$run" ;;
  bundle)
    push
    dest="${1:-$HOME/Downloads/oaimath_bundle.tgz}"
    on "cd ~/oaimath; tar -czf bundle.tgz ${FILES[*]} data/pairs.jsonl data/comp_pairs.jsonl data/minif2f.jsonl env/lakefile.toml env/lake-manifest.json env/lean-toolchain"
    scp -q "$HOST:oaimath/bundle.tgz" "$dest"
    echo "$dest -- upload it to Google Drive as MyDrive/oaimath/oaimath_bundle.tgz, then run scripts/oai_colab_cell.py" ;;
  sync) push ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
