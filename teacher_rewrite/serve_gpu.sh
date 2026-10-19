#!/bin/bash
# One Qwen3.8-27B server (the rewriter and judge) on one GPU (TP=1). Usage: serve_gpu.sh <gpu> <port>
# Needs PLANNER_ENV (an environment with vLLM 0.28) and the model in the Hugging Face cache.
set -u
G=$1; PORT=$2
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
exec $PLANNER_ENV/bin/vllm serve Qwen/Qwen3.8-27B --served-model-name qwen27 --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 --max-model-len 65536 --max-num-seqs 512 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0 \
  --enable-prefix-caching --generation-config auto --reasoning-parser qwen3
