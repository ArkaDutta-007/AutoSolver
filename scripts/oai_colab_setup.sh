#!/bin/bash
# Colab VM setup for `oai_pipeline.py run --profile colab`. A Colab VM starts
# empty after every disconnect, so this runs each session; ~15 min fresh.
set -euo pipefail
cd /content/oaimath
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /content | tail -1
if [ ! -x "$HOME/.elan/bin/lake" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none > /dev/null
fi
export PATH="$HOME/.elan/bin:$PATH"
(cd env && lake exe cache get > /dev/null)        # Mathlib oleans for the pinned revision
python -c "import vllm, peft" 2>/dev/null || pip install -q vllm peft
# Colab ships torchao 0.10, and peft refuses to build any LoRA layer while a torchao
# older than it supports is installed. Nothing here uses torchao (vLLM's fp8 is its own).
pip uninstall -y -q torchao 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU: the vllm wheel may need a newer driver than this VM has'; print('torch', torch.__version__, 'sees', torch.cuda.get_device_name(0))"
