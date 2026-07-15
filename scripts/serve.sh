#!/usr/bin/env bash
# Serve Qwen3.6-27B: Unsloth NVFP4 weights + TurboQuant 4bit_nc KV (4-bit K / 4-bit V + norm-corr) + MTP + vision.
# 200K context (~235K pool) on one RTX 5090. Requires the patched image (see ../patches/).
# CRITICAL: --no-async-scheduling — without it, 4bit_nc + MTP corrupts KV (0/8 retrieval). See NOTES.
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
  -e VLLM_TQ_PRESET=turboquant_4bit_nc \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  --model /model --served-model-name qwen3.6-27b --trust-remote-code \
  --kv-cache-dtype turboquant_4bit_nc \
  --no-async-scheduling \
  --gpu-memory-utilization 0.94 --max-model-len 200000 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}' \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'

echo "launching $NAME (Unsloth NVFP4 + TurboQuant 4bit_nc KV + MTP) on :$PORT ..."
for i in $(seq 1 90); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "DIED — see: sudo docker logs $NAME"; exit 1; }
  sleep 5
done
sudo docker restart owui-proxy >/dev/null 2>&1 || true   # so Open WebUI re-discovers
echo "daily up. KV pool: $(sudo docker logs $NAME 2>&1 | grep 'GPU KV cache size' | tail -1)"
echo "  ^ VERIFY this is ~235K, not ~165K — ~165K means a silent turboquant_k8v4 fallback (wrong config)."

# ------------------------------------------------------------------------------
# NOTES / KNOBS
#   IMAGE                      : MUST be the patched build — stock nightly produces
#                                garbage with turboquant+MTP. Build: ../patches/.
#   VLLM_TQ_PRESET             : REQUIRED, MUST equal --kv-cache-dtype. The MTP draft runner
#                                doesn't inherit the KV dtype upstream; this is what it falls back to.
#   --no-async-scheduling      : CRITICAL. vLLM's async scheduler desyncs the request-ID->batch-row
#                                mapping under MTP's multi-token verify batches (vllm#42655) and
#                                corrupts KV — 0/8 retrieval on 4bit_nc (4-bit keys are hypersensitive),
#                                ~10% intermittent on k8v4. With this flag: 8/8 @9K/20K/40K, 90/90
#                                high-pressure concurrent. This is what un-rejected 4bit_nc.
#   --kv-cache-dtype turboquant_4bit_nc
#                              : 4-bit Keys / 4-bit Values + norm-correction. ~235K pool (+42% vs
#                                k8v4's 165K) at ~48K tok/GiB, in ~the same ~4.89 GiB. Do NOT set
#                                --block-size: the hybrid allocator auto-resolves the unified block.
#   --structured-outputs-config: enables the reasoning gate for the #44993 graft — response_format
#                                json_schema with thinking-on otherwise returns EMPTY content
#                                (schema leaks into reasoning_content). Lifted tool-eval 85->89.
#                                Needs adequate max_tokens (reasoning + JSON).
#   --default-chat-template-kwargs preserve_thinking:true
#                              : keep historical <think> across turns. CLIENT must resend prior
#                                reasoning in the `reasoning` field (NOT reasoning_content).
#   --speculative-config ns=3  : MTP. Mean acceptance length ~3.2.
#   --max-num-seqs 8           : also frees activation memory -> bigger KV pool.
#   --max-num-batched-tokens   : keep at 8192. Dropping to 4096 costs ~4x PREFILL. Tested.
#   VLLM_TQ_KV_SPLITS          : TQ decode split count (default 32). Lowering it hurts
#                                BOTH c1 and c8 — leave it alone. Tested.
# SPEED vs k8v4 (decode @512 tg-mean, t/s, both async-off): 4bit_nc c1 133 c2 211 c4 432 c8 435
#   vs k8v4 137/250/426/467 — a small decode cost (-3% c1, -7% c8) for +42% pool. Deep single-stream
#   (pp4096 c1) is 4bit_nc's worst: 126 vs 145 (-13%).
# ALTERNATIVES (both still use --no-async-scheduling where TQ+MTP is involved):
#   decode-optimal middle ground: --kv-cache-dtype turboquant_k8v4 -e VLLM_TQ_PRESET=turboquant_k8v4
#                          --max-model-len 160000 (165K pool). See ../docs/CONFIG.md.
#   deep-context high-concurrency batch: stock vllm/vllm-openai:nightly, --kv-cache-dtype fp8_e4m3
#                          --max-model-len 131072 (no --block-size). See ../docs/CONFIG.md.
# ------------------------------------------------------------------------------
