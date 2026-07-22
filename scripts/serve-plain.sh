#!/usr/bin/env bash
# The NO-LMCACHE profile. Qwen3.6-27B: natfii NVFP4 W4A4 + fp8_e4m3 KV + FlashInfer +
# MTP ns=4 + vision. 200K context, ~239K pool at util 0.98, on one RTX 5090.
#
# The DAILY is ./serve.sh (same engine + LMCache DRAM/NVMe tiers). Run THIS one when you
# want the biggest hot pool with no sidecar and no local LMCache patches — you trade
# ~2.4M tokens of tier capacity for +25K on-GPU and mnbt 4096. Full trade in
# "What removing LMCache changes" in ../docs/LMCACHE.md.
#
# Requires the base patched image (see ../patches/). CRITICAL: MTP ns>=2 + fp8 KV
# IMA-crashes under concurrency on stock vLLM — the image's PR #42603 graft is what fixes
# it. And --no-async-scheduling (vllm#42655). See NOTES. Idempotent.
#
#   MODEL_DIR=/path/to/Qwen3.6-27B-VLM-NVFP4-MTP ./serve-plain.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # llama-benchy (pre-warm) commonly lives here

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/natfii-27b-nvfp4}   # natfii/Qwen3.6-27B-VLM-NVFP4-MTP
IMAGE=${IMAGE:-vllm-qwen36:patched}   # build from ../patches/Dockerfile
PORT=${PORT:-8020}
BIND_ADDR=${BIND_ADDR:-127.0.0.1}    # loopback by default — the API has NO auth. Set 0.0.0.0
                                     # only behind a firewall/VPN or an authenticated proxy.
NAME=vllm-27b

# --- tokenizer truncation guard (gotcha #9) -----------------------------------
# The published checkpoint ships tokenizer.json with truncation baked at 8192
# (calibration leftover). Text works; multimodal requests expanding past 8192
# tokens hard-400. Null it at every launch — a re-download reintroduces the bug.
python3 - <<PYEOF
import json
p = "$MODEL_DIR/tokenizer.json"
t = json.load(open(p))
if t.get("truncation") is not None:
    import shutil; shutil.copy(p, p + ".orig")
    t["truncation"] = None
    json.dump(t, open(p, "w"), ensure_ascii=False)
    print("tokenizer guard: TRUNCATION BUG FIXED (re-download detected)")
else:
    print("tokenizer guard: clean")
PYEOF

# Persistent compile/triton/flashinfer cache => warm restarts. ALWAYS mount these —
# FlashInfer 0.6.15 JIT-compiles its kernels on first run (one ~min build, warm forever).
CACHE_DIR=${CACHE_DIR:-/srv/qwen5090/cache}
CACHE="-v ${CACHE_DIR}/torch_compile_natfii:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor \
       -v ${CACHE_DIR}/flashinfer:/root/.cache/flashinfer"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
  -p ${BIND_ADDR}:${PORT}:8000 --restart unless-stopped --shm-size 16g \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  --model /model --served-model-name qwen3.6-27b --trust-remote-code \
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

echo "launching $NAME (natfii NVFP4 W4A4 + fp8 KV + FlashInfer + MTP ns=4) on ${BIND_ADDR}:$PORT ..."
HEALTHY=0
for i in $(seq 1 90); do
  curl -sf http://${BIND_ADDR}:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; HEALTHY=1; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "FAILED: container died — see: sudo docker logs $NAME"; exit 1; }
  sleep 5
done
if [ "$HEALTHY" != 1 ]; then
  echo "FAILED: /health never came up within 7.5 min"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1
fi

# --- pool assertion: a wrong pool means a dtype/align/preset mismatch — the server
# looks healthy at the wrong config. Fail closed. Expected ~239,436 @0.98, seqs 8.
POOL=$(sudo docker logs "$NAME" 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE '[0-9,]+ tokens' | tr -d ', tokens')
echo "daily up. KV pool: ${POOL} tokens"
if [ -z "$POOL" ] || [ "$POOL" -lt 233000 ] || [ "$POOL" -gt 245000 ]; then
  echo "FAILED: pool ${POOL:-<missing>} outside expected 233K-245K (util 0.98, seqs 8)."
  [ -n "${POOL_MIN:-}" ] && [ -n "${POOL_MAX:-}" ] && [ "$POOL" -ge "$POOL_MIN" ] && [ "$POOL" -le "$POOL_MAX" ] || exit 1
fi

# --- autotune shape pre-warm (gotcha #8): LOAD-BEARING at util 0.98 — the 0.98 margin
# was validated WITH this warm-up; an un-warmed engine meets the ~266 MiB workspace
# allocation on its first real deep burst. Fail closed; ALLOW_NO_PREWARM=1 to override.
if command -v llama-benchy >/dev/null 2>&1; then
  echo "pre-warming autotune shapes (pp8192 c8, ~60s)..."
  if llama-benchy --base-url http://${BIND_ADDR}:${PORT}/v1 --model qwen3.6-27b \
      --pp 8192 --tg 16 --concurrency 8 --runs 1 >/dev/null 2>&1; then
    echo "pre-warm done. free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  else
    echo "FAILED: autotune pre-warm errored — 0.98 margin NOT validated on this boot."
    [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || exit 1
  fi
else
  echo "WARNING: llama-benchy not found — autotune shapes NOT pre-warmed (gotcha #8)."
  [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || { echo "Install llama-benchy or set ALLOW_NO_PREWARM=1."; exit 1; }
fi

sudo docker restart owui-proxy >/dev/null 2>&1 || true   # so Open WebUI re-discovers

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
#   (no --quantization flag)   : the checkpoint is a ModelOpt NVFP4 W4A4 export; vLLM auto-detects
#                                it and dispatches CUTLASS FP4 GEMM (FlashInferCutlassNvFp4Linear).
#                                W4A4 = the 3.4x prefill vs W4A16 (native FP4 tensor cores vs
#                                Marlin dequant-to-bf16). Forcing a flag selects the wrong path.
#   --kv-cache-dtype fp8_e4m3  : flat-with-depth attention via FlashInfer. e5m2 is NOT usable on this
#                                checkpoint. --mamba-cache-mode align packs GDN state into the unified
#                                pool -> ~239K at util 0.98. Do NOT set --block-size (hybrid allocator resolves it).
#   --structured-outputs-config: enables the reasoning gate for the #44993 graft — response_format
#                                json_schema with thinking-on otherwise returns EMPTY content.
#                                Needs adequate max_tokens (reasoning + JSON).
#   --default-chat-template-kwargs preserve_thinking:true
#                              : keep historical <think> across turns. CLIENT must resend prior
#                                reasoning in the `reasoning` field (NOT reasoning_content).
#   --speculative-config ns=4  : MTP. --mamba-cache-mode all is same speed / same pool and does NOT
#                                avoid the crash — don't bother; PR #42603 is the fix.
#   util 0.98 + mnbt 4096      : util is the ONLY pool lever (+~8.4K tok/0.01 -> 239,436 at 0.98,
#                                222,535 at 0.96). The ceiling is MODEL-SPECIFIC: 0.98 serve-time-
#                                OOM'd the previous heavier W4A16 daily (lazy autotune workspace,
#                                gotcha #8) but passes here WITH the 128MiB workspace cap env +
#                                the boot pre-warm above. Battery that earned it: needle, pp8192xc8,
#                                pp30000xc8, 8x text flood, 8x 4-image vision, then two SIMULTANEOUS
#                                combined waves on a cold engine. Steady-state floor ~130-190 MiB.
#                                Fall back to 0.96 (222K) if fragmentation or a sidecar ever bites.
#   tokenizer guard            : see gotcha #9 in the README — baked truncation:8192 in the
#                                shipped tokenizer.json breaks >8K-token multimodal requests.
# SPEED: prefill ~13.5K t/s @8K (c1). Deep-concurrent sustained (tg512): pp512xc8 778,
#   pp8192xc8 466, pp30000xc8 149 t/s aggregate. tool-eval-bench ~90 (full 69x2, 4 trials).
# ------------------------------------------------------------------------------
