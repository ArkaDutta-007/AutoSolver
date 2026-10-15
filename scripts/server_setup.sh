#!/usr/bin/env bash
# Run ON the GPU server (2+ GPUs with 48 GB each). Stands up what a laptop
# cannot host: a 32B prover served by vLLM, which honours `n` (Ollama does not).
#
# Your workstation keeps the orchestrator and the journal and reaches this box
# through an SSH tunnel, so nothing here needs a public port.
#
#   ./server_setup.sh models     # download weights + build Lean (slow, do first)
#   ./server_setup.sh serve      # start both vLLM replicas
#   ./server_setup.sh status     # what is up
set -euo pipefail

PROVER_MODEL="${PROVER_MODEL:-Goedel-LM/Goedel-Prover-V2-32B}"
PORT_A="${PORT_A:-8001}"
PORT_B="${PORT_B:-8002}"
# the repo this script sits in, wherever it was cloned
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO/server/logs}"
mkdir -p "$LOG_DIR"

n_gpus() { nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '; }

cmd_models() {
  pip install -q --upgrade "vllm>=0.6.0" huggingface_hub

  # 32B in bf16 is ~64 GB -> tensor-parallel across 2 cards per replica.
  huggingface-cli download "$PROVER_MODEL"

  # Lean + Mathlib. The REPL is the whole point: `import Mathlib` costs ~25s
  # cold and a laptop pays it on EVERY check because it has no RAM to keep a
  # process warm. Here it is paid once.
  if ! command -v lake >/dev/null; then
    curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y
    # shellcheck disable=SC1090
    source "$HOME/.elan/env"
  fi
  if [ ! -d "$HOME/repl" ]; then
    git clone https://github.com/leanprover-community/repl "$HOME/repl"
  fi
  cd "$HOME/repl" && lake update && lake exe cache get && lake build
  echo "models + Lean ready"
}

cmd_serve() {
  local g; g=$(n_gpus)
  if [ "$g" -lt 2 ]; then echo "need >=2 GPUs, found $g" >&2; exit 1; fi

  # TWO replicas at TP=2 rather than ONE at TP=4. Best-of-N sampling is
  # embarrassingly parallel, so two independent replicas give ~2x the sampling
  # throughput at the same total memory. --enable-prefix-caching matters here:
  # all n samples for a goal share one long prompt prefix.
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$PROVER_MODEL" \
    --tensor-parallel-size 2 --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    --port "$PORT_A" > "$LOG_DIR/vllm_a.log" 2>&1 &
  echo "replica A -> :$PORT_A (GPUs 0,1)"

  if [ "$g" -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve "$PROVER_MODEL" \
      --tensor-parallel-size 2 --max-model-len 16384 \
      --gpu-memory-utilization 0.92 --enable-prefix-caching \
      --port "$PORT_B" > "$LOG_DIR/vllm_b.log" 2>&1 &
    echo "replica B -> :$PORT_B (GPUs 2,3)"
  fi
  echo "warming up; first load takes several minutes. tail $LOG_DIR/vllm_a.log"
}

cmd_status() {
  echo "== GPUs =="; nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "  no nvidia-smi"
  for p in "$PORT_A" "$PORT_B"; do
    printf "== vLLM :%s == " "$p"
    curl -s --max-time 5 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 \
      && echo "up" || echo "down"
  done
}

case "${1:-status}" in
  models) cmd_models ;;
  serve)  cmd_serve  ;;
  status) cmd_status ;;
  *) echo "usage: $0 {models|serve|status}" >&2; exit 1 ;;
esac
