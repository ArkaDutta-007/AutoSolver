#!/bin/bash
# Serve Goedel-Prover-V2-32B (bf16) on the GPU server. goedel_tunnel.sh copies
# this file to the server and runs it there, so edit it for your box: the GPU
# list and --tensor-parallel-size go together (32B bf16 needs 2 x 48 GB).
# GPUs 1,2 here because GPU 0 runs something else on the box this was built on.
export CUDA_VISIBLE_DEVICES=1,2
# FlashInfer JIT-compiles CUDA kernels at runtime and there is no nvcc on this
# box (no CUDA toolkit, only the driver + torch cu130 wheels), so it dies with
# "Could not find nvcc". A6000 is Ampere, where vLLM ships precompiled
# FlashAttention -- no JIT needed.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
VLLM=~/venvs/vllm-goedel/bin/vllm; [ -x "$VLLM" ] || VLLM=vllm
exec "$VLLM" serve Goedel-LM/Goedel-Prover-V2-32B \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --host 127.0.0.1 --port 8001
