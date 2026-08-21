#!/usr/bin/env bash
# Qwen3.8-27B saka on the NVFP4-KV image (FA2 NVFP4 KV cache on the 5090) —
# EXPERIMENT: :8029 / vllm-exp only. Never the daily port.
#
# What this is: the digest-pinned nightly (same engine as the daily) + the
# patches-nvfp4kv stack (PR #49891 rebase + linear-V-scale store overlay).
# Halves KV bytes/token vs fp8 -> expected pool ~330-360K @0.95 (fp8 plain @0.98
# = 207,042; the "4M+" figure is the 121 GB DGX Spark, not a 32 GB card).
# PLAIN profile: NO LMCache (0001/0002 rc4 patches are fp8-page-specific).
#
# Envs: NS=4 (MTP depth; NS=0 -> no spec), UTIL=0.95, MAXLEN=200000, MNBT=4096,
#       LINEAR_VSF=1 (0 = in-tree swizzled writer, DIAGNOSTIC ONLY: expected to
#       break long-context recall), TEMPLATE_KWARGS, IMAGE.
# Gauntlet: see patches-nvfp4kv/README.md (pool -> depth needles cold/warm,
# MTP on vs off -> concurrent 32K+ recall (the "sean gate") -> killer -> 69x2 ->
# benchy incl. deep pp30K c1 decode vs fp8).
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/saka-qwen3.8-27b-mtp-nvfp4}
IMAGE=${IMAGE:-vllm-qwen38:nvfp4kv}
PORT=8029
NAME=vllm-exp
NS=${NS:-4}
UTIL=${UTIL:-0.95}
MAXLEN=${MAXLEN:-200000}
MNBT=${MNBT:-4096}
LINEAR_VSF=${LINEAR_VSF:-1}
TEMPLATE_KWARGS_DEFAULT='{"preserve_thinking":true,"reasoning_effort":"medium"}'
TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-$TEMPLATE_KWARGS_DEFAULT}

SPEC_ARGS=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${NS}}")
[ "$NS" = 0 ] && SPEC_ARGS=()

# tokenizer truncation guard (R62 pattern)
python3 - <<PYEOF
import json
p = "$MODEL_DIR/tokenizer.json"
t = json.load(open(p))
if t.get("truncation") is not None:
    import shutil; shutil.copy(p, p + ".orig")
    t["truncation"] = None
    json.dump(t, open(p, "w"), ensure_ascii=False)
    print("tokenizer guard: TRUNCATION FIXED")
else:
    print("tokenizer guard: clean")
PYEOF

# separate torch.compile cache: KV dtype changes the captured graphs
CACHE="-v /srv/qwen5090/cache/torch_compile_qwen38_nvfp4kv:/root/.cache/vllm/torch_compile_cache \
       -v /srv/qwen5090/cache/triton:/root/.triton/cache \
       -v /srv/qwen5090/cache/inductor:/root/.cache/inductor \
       -v /srv/qwen5090/cache/flashinfer:/root/.cache/flashinfer"

sudo docker rm -f "$NAME" sglang-exp >/dev/null 2>&1 || true
if sudo docker ps --format '{{.Names}}' | grep -qxE "vllm-27b|vllm-lmcache|vllm-eval"; then
  echo "FATAL: a daily/eval engine is running — one GPU, one engine." >&2; exit 1
fi
sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: image $IMAGE missing — build patches-nvfp4kv/Dockerfile.nvfp4kv first" >&2; exit 1; }

sudo docker run -d --name "$NAME" --entrypoint python3 --runtime nvidia --gpus all --ipc=host \
  -p 0.0.0.0:${PORT}:8000 --shm-size 16g \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  -e VLLM_SM12X_NVFP4_LINEAR_VSF="$LINEAR_VSF" \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name qwen3.8-27b qwen3.6-27b --trust-remote-code \
  --kv-cache-dtype nvfp4 --attention-backend FLASHINFER \
  --no-async-scheduling \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --max-num-seqs 8 --max-num-batched-tokens "$MNBT" \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs "$TEMPLATE_KWARGS"

echo "nvfp4kv audition (NS=$NS util=$UTIL maxlen=$MAXLEN mnbt=$MNBT linear_vsf=$LINEAR_VSF) on :$PORT ..."
echo "(first boot JIT-compiles the FA2 NVFP4 sm120 kernels: allow ~15 min; cached in /srv/qwen5090/cache/flashinfer)"
for i in $(seq 1 240); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo HEALTHY; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x \
    || { echo "FAILED: container died"; sudo docker logs "$NAME" 2>&1 | tail -40; exit 1; }
  sleep 10
  [ "$i" = 240 ] && { echo "FAILED: no /health in 40min"; sudo docker logs "$NAME" 2>&1 | tail -40; exit 1; }
done

# positive checks (never grep -q a docker-logs pipe under pipefail: SIGPIPE false-negative)
LOGS=$(sudo docker logs "$NAME" 2>&1 || true)
echo "$LOGS" | grep -aiE "GPU KV cache size|Maximum concurrency" | tail -2
echo "$LOGS" | grep -aiE "kv_cache_dtype=|decode_backend=|Using .* attention backend|FLASHINFER" | tail -4
OVERLAY=$(echo "$LOGS" | grep -ac "linear-V-scale store overlay ACTIVE" || true)
SWIZZ=$(echo "$LOGS" | grep -ac "swizzled-V-scale writer" || true)
NVFP4=$(echo "$LOGS" | grep -aic "kv_cache_dtype=nvfp4" || true)
echo "checks: nvfp4-dtype-lines=$NVFP4 overlay-active=$OVERLAY swizzled-fallback=$SWIZZ"
if [ "$LINEAR_VSF" = 1 ] && [ "$OVERLAY" = 0 ]; then
  echo "WARNING: overlay ACTIVE line not seen yet (it logs on the first KV store) — verify after the first request:"
  echo "  sudo docker logs $NAME 2>&1 | grep -a 'overlay'"
fi
if [ "$LINEAR_VSF" = 0 ]; then
  echo "DIAGNOSTIC MODE (LINEAR_VSF=0): swizzled-V-scale writer requested on purpose — deep recall is EXPECTED to be wrong; never measure quality/perf on this engine."
else
  [ "$SWIZZ" = 0 ] || { echo "FATAL: engine fell back to the SWIZZLED writer — recall will be wrong; fix the overlay before measuring." >&2; exit 1; }
fi
