#!/usr/bin/env bash
# Serve Qwen3.6-27B: Lorbus INT4-AutoRound weights + fp8_e4m3 KV + FlashInfer + MTP ns=4 + vision.
# 200K context (~287K pool at util 0.98) on one RTX 5090. Requires the patched image (see ../patches/).
# CRITICAL: MTP ns>=2 + fp8 KV IMA-crashes under concurrency on stock vLLM — the image's
# PR #42603 graft is what fixes it. And --no-async-scheduling (vllm#42655). See NOTES.
# Idempotent: removes the existing container first.
#
#   MODEL_DIR=/path/to/Qwen3.6-27B-int4-AutoRound ./serve.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/qwen3.6-27b-autoround-int4}   # Lorbus INT4-AutoRound, MTP head, vision
IMAGE=${IMAGE:-vllm-qwen36:patched}   # build from ../patches/Dockerfile
PORT=8020
NAME=vllm-27b

# Persistent compile/triton/flashinfer cache => warm restarts. ALWAYS mount these —
# FlashInfer 0.6.15 JIT-compiles its kernels on first run (one ~min build, warm forever).
CACHE_DIR=${CACHE_DIR:-/srv/qwen5090/cache}
CACHE="-v ${CACHE_DIR}/torch_compile_ar_fp8fi:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor \
       -v ${CACHE_DIR}/flashinfer:/root/.cache/flashinfer"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
  -p ${PORT}:8000 --restart unless-stopped --shm-size 16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e TORCH_MATMUL_PRECISION=high \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  --model /model --served-model-name qwen3.6-27b --trust-remote-code \
  --quantization auto-round \
  --kv-cache-dtype fp8_e4m3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.98 --max-model-len 200000 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}' \
  --structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}' \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'

echo "launching $NAME (Lorbus INT4-AutoRound + fp8 KV + FlashInfer + MTP ns=4) on :$PORT ..."
for i in $(seq 1 90); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "DIED — see: sudo docker logs $NAME"; exit 1; }
  sleep 5
done
sudo docker restart owui-proxy >/dev/null 2>&1 || true   # so Open WebUI re-discovers
echo "daily up. KV pool: $(sudo docker logs $NAME 2>&1 | grep 'GPU KV cache size' | tail -1)"
echo "  ^ VERIFY this is ~287K (util 0.98). A wrong pool size means a dtype/align/preset mismatch (wrong config)."

# ------------------------------------------------------------------------------
# NOTES / KNOBS
#   IMAGE                      : MUST be the patched build — stock vLLM IMA-crashes with
#                                MTP ns>=2 + fp8 KV under concurrency (vllm#40756/#35288).
#                                Build: ../patches/ (PR #42603 + FlashInfer 0.6.15 + #44993).
#   PR #42603 (in the image)   : one current-stream sync in the MTP draft loop; the whole reason
#                                ns=4 is usable here. Single-stream/ns=1 hide the bug; load-test
#                                with 3+ parallel streams to reproduce/verify.
#   --no-async-scheduling      : CRITICAL. vLLM's async scheduler desyncs the request-ID->batch-row
#                                mapping under MTP's multi-token verify batches (vllm#42655) and
#                                corrupts KV. Keep it off.
#   --quantization auto-round  : Lorbus INT4-AutoRound (compressed-tensors AutoRound), native in base.
#   --kv-cache-dtype fp8_e4m3  : flat-with-depth attention via FlashInfer. e5m2 is NOT usable on this
#                                checkpoint. --mamba-cache-mode align packs GDN state into the unified
#                                pool -> ~287K at util 0.98. Do NOT set --block-size (hybrid allocator resolves it).
#   --structured-outputs-config: enables the reasoning gate for the #44993 graft — response_format
#                                json_schema with thinking-on otherwise returns EMPTY content.
#                                Needs adequate max_tokens (reasoning + JSON).
#   --default-chat-template-kwargs preserve_thinking:true
#                              : keep historical <think> across turns. CLIENT must resend prior
#                                reasoning in the `reasoning` field (NOT reasoning_content).
#   --speculative-config ns=4  : MTP. --mamba-cache-mode all is same speed / same pool and does NOT
#                                avoid the crash — don't bother; PR #42603 is the fix.
#   util 0.98 + mnbt 4096      : util is the ONLY pool lever here (+8,450 tok/0.01 -> 287,323 at
#                                0.98); mnbt does NOT change the pool — chunked prefill bounds the
#                                transient either way, so 4096 is essentially free on prefill (and at
#                                high util it AVOIDS a ~9% deep-prefill slowdown that mnbt 8192 incurs
#                                from allocator pressure). Ceiling probed ABOVE 0.98: distinct-prompt
#                                cold-start bursts at ~98%/~104% of pool, 8x concurrent 4-image
#                                (2048^2) vision bursts, and mixed vision+deep-text all pass with no
#                                OOM. CAVEAT: burst tests need DISTINCT prompts — prefix caching
#                                silently voids identical-prompt tests. ~600MB VRAM margin at 0.98;
#                                fall back to 0.96 (270K) / 0.94 (253K) if fragmentation or a
#                                colocated sidecar ever bites.
# SPEED (decode t/s total, --pp 512 4096 --tg 128 -c 1 2 4 8): @512 114/212/355/496, @4096 129/164/198/157.
#   Long-context c1 decode is flat ~128-133 t/s from 30K to 180K. tool-eval-bench 90 (full 69x2).
# ------------------------------------------------------------------------------
