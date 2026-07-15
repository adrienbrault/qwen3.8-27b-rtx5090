#!/usr/bin/env bash
# Serve Qwen3.6-27B: Unsloth NVFP4 weights + TurboQuant k8v4 KV (8-bit K / 4-bit V) + MTP + vision.
# 160K context (165K pool) on one RTX 5090. Requires the patched image (see ../patches/).
# Idempotent: removes the existing container first.
#
#   MODEL_DIR=/path/to/unsloth-Qwen3.6-27B-NVFP4 ./serve.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/unsloth-27b-nvfp4-v2}   # compressed-tensors NVFP4, MTP head, vision
IMAGE=${IMAGE:-vllm-turboquant:patched}   # build from ../patches/Dockerfile
PORT=8020
NAME=vllm-27b

# Persistent compile/triton cache => warm restarts. ALWAYS mount these.
CACHE_DIR=${CACHE_DIR:-/srv/qwen5090/cache}
CACHE="-v ${CACHE_DIR}/torch_compile:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
  -p ${PORT}:8000 --restart unless-stopped --shm-size 16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e TORCH_MATMUL_PRECISION=high \
  -e VLLM_TQ_PRESET=turboquant_k8v4 \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  --model /model --served-model-name qwen3.6-27b --trust-remote-code \
  --kv-cache-dtype turboquant_k8v4 \
  --gpu-memory-utilization 0.94 --max-model-len 160000 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'

echo "launching $NAME (Unsloth NVFP4 + TurboQuant KV + MTP) on :$PORT ..."
for i in $(seq 1 90); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "DIED — see: sudo docker logs $NAME"; exit 1; }
  sleep 5
done
sudo docker restart owui-proxy >/dev/null 2>&1 || true   # so Open WebUI re-discovers
echo "daily up. KV pool: $(sudo docker logs $NAME 2>&1 | grep 'GPU KV cache size' | tail -1)"

# ------------------------------------------------------------------------------
# NOTES / KNOBS
#   IMAGE                      : MUST be the patched build — stock nightly produces
#                                garbage with turboquant+MTP. Build: ../patches/.
#   VLLM_TQ_PRESET             : REQUIRED, MUST equal --kv-cache-dtype. The MTP draft runner
#                                doesn't inherit the KV dtype upstream; this is what it falls back to.
#   --kv-cache-dtype turboquant_k8v4
#                              : 8-bit Keys / 4-bit Values. 165K pool (vs 136K on fp8_e4m3) —
#                                33.8K tok/GiB vs 26.0K, in LESS KV memory (4.89 vs 5.25 GiB).
#                                8-bit keys preserve long-context retrieval (needle 8/8 vs the
#                                4-bit-key turboquant_4bit_nc's 0/8). Do NOT set --block-size:
#                                the hybrid allocator auto-resolves the unified block to 2112.
#   --speculative-config ns=3  : MTP. Mean acceptance length ~3.2 (identical to fp8).
#   --max-num-seqs 8           : also frees activation memory -> bigger KV pool.
#   --max-num-batched-tokens   : keep at 8192. Dropping to 4096 costs ~4x PREFILL. Tested.
#   VLLM_TQ_KV_SPLITS          : TQ decode split count (default 32). Lowering it hurts
#                                BOTH c1 and c8 — leave it alone. Tested.
# SPEED vs fp8_e4m3 (decode @512 tg-mean, t/s): c1 164 vs 130, c2 319 vs 251, c4 524 vs 482,
#   c8 516 vs 478. k8v4 wins single-stream + short-context everywhere; fp8 only wins DEEP
#   context (>=4K) at high concurrency (decode c4@4096 461 vs 277).
# ALTERNATIVES:
#   deep-context high-concurrency batch: stock vllm/vllm-openai:nightly, --kv-cache-dtype fp8_e4m3
#                          --max-model-len 131072 (no --block-size). See ../docs/CONFIG.md.
# ------------------------------------------------------------------------------
