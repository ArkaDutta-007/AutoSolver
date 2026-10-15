#!/bin/bash
# Serve the tuned formalizer on GPU 0 -- the one GPU Goedel does not use, so
# prover (GPUs 1,2) and formalizer coexist. Same no-nvcc constraints as
# serve_goedel.sh: Ampere ships precompiled FlashAttention, nothing JIT-compiles.
#
# BASE + LoRA, not a merged model. The training run produces a ~80 MB adapter;
# merging it would mean writing and shipping a 15 GB copy of a base model vLLM
# already has cached, to serve exactly the same weights. `--lora-modules` loads
# the adapter onto the cached base instead, so the only artefact that has to
# survive the Colab session is the adapter itself.
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
exec ~/venvs/vllm-goedel/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name formalizer-base \
  --enable-lora \
  --lora-modules formalizer-v1=${FORMALIZER_ADAPTER:-$HOME/formalizer/adapter_v3} \
  --max-lora-rank 16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.55 \
  --host 127.0.0.1 --port 8002
