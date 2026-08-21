#!/usr/bin/env bash
# Qwen3.8-27B saka + DFlash2 drafter (PR #52816 on nightly ba07e4a48) — EXPERIMENT: :8029 /
# vllm-exp only. Plain profile (fp8 KV, no LMCache), DFlash2 instead of MTP.
# Envs: NS=7 (draft block 8 - 1; 0 = no spec, for the AR baseline), UTIL=0.95, MAXLEN=200000,
#       MNBT=4096, DRAFT_DIR, IMAGE, TEMPLATE_KWARGS, KV=fp8_e4m3, BACKEND (unset = vLLM picks).
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/saka-qwen3.8-27b-mtp-nvfp4}
DRAFT_DIR=${DRAFT_DIR:-/srv/qwen5090/models/dflash2-qwen38-incoai}
IMAGE=${IMAGE:-vllm-qwen38:dflash2}
PORT=8029; NAME=vllm-exp
NS=${NS:-7}; UTIL=${UTIL:-0.95}; MAXLEN=${MAXLEN:-200000}; MNBT=${MNBT:-4096}; KV=${KV:-fp8_e4m3}
TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{"preserve_thinking":true,"reasoning_effort":"medium"}'}
SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${NS}}")
[ "$NS" = 0 ] && SPEC_ARGS=()
BACKEND_ARGS=(); [ -n "${BACKEND:-}" ] && BACKEND_ARGS=(--attention-backend "$BACKEND")

CACHE="-v /srv/qwen5090/cache/torch_compile_qwen38_dflash2:/root/.cache/vllm/torch_compile_cache \
       -v /srv/qwen5090/cache/triton:/root/.triton/cache \
       -v /srv/qwen5090/cache/inductor:/root/.cache/inductor \
       -v /srv/qwen5090/cache/flashinfer:/root/.cache/flashinfer"
sudo docker rm -f "$NAME" sglang-exp >/dev/null 2>&1 || true
if sudo docker ps --format '{{.Names}}' | grep -qxE "vllm-27b|vllm-lmcache|vllm-eval"; then
  echo "FATAL: a daily/eval engine is running — one GPU, one engine." >&2; exit 1
fi
sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: image $IMAGE missing — build patches-dflash2/Dockerfile.dflash2" >&2; exit 1; }
[ -f "$DRAFT_DIR/model.safetensors" ] || { echo "FATAL: drafter missing at $DRAFT_DIR" >&2; exit 1; }

sudo docker run -d --name "$NAME" --entrypoint python3 --runtime nvidia --gpus all --ipc=host \
  -p 0.0.0.0:${PORT}:8000 --shm-size 16g \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  -v "$MODEL_DIR":/model -v "$DRAFT_DIR":/draft:ro $CACHE \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name qwen3.8-27b qwen3.6-27b --trust-remote-code \
  --kv-cache-dtype "$KV" "${BACKEND_ARGS[@]}" \
  --no-async-scheduling \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --max-num-seqs 8 --max-num-batched-tokens "$MNBT" \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs "$TEMPLATE_KWARGS"

echo "dflash2 audition (NS=$NS util=$UTIL maxlen=$MAXLEN mnbt=$MNBT kv=$KV) on :$PORT ..."
for i in $(seq 1 180); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo HEALTHY; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x \
    || { echo "FAILED: container died"; sudo docker logs "$NAME" 2>&1 | grep -aE "Error|error|raise|Traceback" | tail -30; exit 1; }
  sleep 10
  [ "$i" = 180 ] && { echo "FAILED: no /health in 30min"; sudo docker logs "$NAME" 2>&1 | tail -40; exit 1; }
done
LOGS=$(sudo docker logs "$NAME" 2>&1 || true)
echo "$LOGS" | grep -aiE "GPU KV cache size|Maximum concurrency" | tail -2
echo "$LOGS" | grep -aiE "model runner|Using .* backend|speculative|dflash|DFlash2|spec" | grep -av "non-default args" | head -12
