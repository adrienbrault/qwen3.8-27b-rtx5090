#!/usr/bin/env bash
# THE DAILY engine (re-platformed 2026-08-15): latest vLLM, zero patches.
# Qwen3.8-27B saka MTP-NVFP4 on vllm/vllm-openai:nightly (0.27.2rc1.dev77+gac7509e2b),
# PLAIN profile — ZERO patches/grafts.
#
# Patch audit vs the 0.23 lmcfix6 base (all measured 2026-08-15, results dir
# 2026-08-15-qwen38-nightly): #42603 MTP graft OBSOLETE (c8 killer passed ungrafted),
# #44993 SO graft OBSOLETE (native SO+reasoning), deepseek_r1 workaround OBSOLETE
# (qwen3 parser handles prefilled <think>; reasoning served as message.reasoning),
# async×spec crash OBSOLETE — but QUALITY still prefers async OFF (69x2: 91±0 off vs
# 90±1.4 on) and --mamba-cache-mode align is LOAD-BEARING (69x2 without: 87-88.5).
# Perf vs 0.23 tier daily: c1 123 vs 84.5, c8 322 vs 256, prefill@8K 13.0K vs 10.9K.
# TIERS TEMPORARILY RETIRED: LMCache must be rebuilt against this vLLM (patch audit
# 0001/0002/0007/0008/0003/0005/0010 vs LMCache 0.5.3) — until then no DRAM/NVMe KV.
# DSpark (vLLM-native, RadixArk draft, block 7): +12-40% over MTP but context caps at
# ~64-128K (draft KV) — opt-in profile via launch-qwen38-nightly.sh STAGE=D, not daily.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/saka-qwen3.8-27b-mtp-nvfp4}
MODEL_NAME=${MODEL_NAME:-qwen3.8-27b}
MODEL_ALIAS=${MODEL_ALIAS:-qwen3.6-27b}
# digest-pinned: the :nightly tag moves daily; the daily must not silently change engines.
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:c96082d33456ceeae7ec0d4faf2b5e47fb806a103decf94f9fbc9b35fd7d6b25}  # 0.27.2rc1.dev77+gac7509e2b, pulled 2026-08-14
PORT=8020
BIND_ADDR=${BIND_ADDR:-127.0.0.1}    # loopback default — the API has NO auth
NAME=vllm-27b

# tokenizer truncation guard (R62)
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

CACHE="-v /srv/qwen5090/cache/torch_compile_qwen38_nightly:/root/.cache/vllm/torch_compile_cache \
       -v /srv/qwen5090/cache/triton:/root/.triton/cache \
       -v /srv/qwen5090/cache/inductor:/root/.cache/inductor \
       -v /srv/qwen5090/cache/flashinfer:/root/.cache/flashinfer"

sudo docker rm -f "$NAME" vllm-lmcache vllm-exp sglang-exp >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" --entrypoint python3 --runtime nvidia --gpus all --ipc=host \
  -p ${BIND_ADDR}:${PORT}:8000 --restart unless-stopped --shm-size 16g \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  -v "$MODEL_DIR":/model $CACHE \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name $MODEL_NAME $MODEL_ALIAS --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.98 --max-model-len 200000 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs '{"preserve_thinking":true}'

echo "launching $NAME (saka 3.8 on vLLM NIGHTLY, MTP ns=4, align, async off) on ${BIND_ADDR}:$PORT ..."
HEALTHY=0
for i in $(seq 1 150); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; HEALTHY=1; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "FAILED: container died"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "FAILED: no /health in 25min"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }

POOL=$(sudo docker logs "$NAME" 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE '[0-9,]+ tokens' | tr -d ', tokens')
echo "daily up. KV pool: ${POOL} tokens"
# measured 207,042 @0.98/align/ns4 on 2026-08-15; band catches gross misconfig only
if [ -z "$POOL" ] || [ "$POOL" -lt "${POOL_MIN:-200000}" ] || [ "$POOL" -gt "${POOL_MAX:-214000}" ]; then
  echo "FAILED: pool ${POOL:-<missing>} outside ${POOL_MIN:-200000}-${POOL_MAX:-214000} (expect ~207,042)."; exit 1
fi

# autotune/compile pre-warm at the deep-concurrent shape (gotcha #8 discipline at 0.98)
if command -v llama-benchy >/dev/null 2>&1; then
  echo "pre-warming (pp8192 c8)..."
  llama-benchy --base-url http://localhost:${PORT}/v1 --model $MODEL_NAME --pp 8192 --tg 16 --concurrency 8 --runs 1 >/dev/null 2>&1 \
    && echo "pre-warm done. free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)" \
    || { echo "FAILED: pre-warm errored"; [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || exit 1; }
else
  [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || { echo "llama-benchy missing"; exit 1; }
fi

sudo docker restart owui-proxy >/dev/null 2>&1 || true
echo "NIGHTLY DAILY UP"
