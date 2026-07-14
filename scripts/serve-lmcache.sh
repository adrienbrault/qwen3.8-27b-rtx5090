#!/usr/bin/env bash
# Serve Qwen3.6-27B with MTP + LMCache tiered KV caching (24 GB pinned-RAM L1 + 150 GB SSD L2).
# COMPOSED PROFILE, validated 2026-07-14 (see ../docs/LMCACHE.md).
# c1 118 / c8 450 composed vs daily 126 / 488 (-6..-9%); revisit walls 0.64-1.4s vs 5.8s re-prefill.
# 150 GB L2 ~= 2M tokens of session history that persists across restarts. NO VISION (image:0).
# Every flag below is load-bearing — read ../docs/LMCACHE.md before changing ANYTHING.
#
#   MODEL_DIR=/path/to/unsloth-Qwen3.6-27B-NVFP4 ./serve-lmcache.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/unsloth-27b-nvfp4-v2}   # compressed-tensors NVFP4, MTP head
IMAGE=${IMAGE:-lmcache-vllm:fixed}   # 0.5.1 pairing + cudart13 fix; nightly pairing REJECTS this model (EngineKVFormat 10)
PORT=8030
NAME=vllm-lmcache
BLK=1600                             # unified block size (fp8 + MTP ns=3); chunk MUST equal it
BATCHED=$((2 * BLK - 1))            # LMCache MP requirement: batched tokens in [chunk, 2*chunk)

# L2 disk tier — a fast SSD directory. 150 GB ~= 2M tokens; survives container restarts.
L2DIR=${L2DIR:-/srv/qwen5090/lmcache-l2}
mkdir -p "$L2DIR"

# Persistent caches => warm restarts. flashinfer mount is MANDATORY on non-nightly images
# (sm120 fp4 kernels JIT-compile on first forward — uncapped, they eat all host RAM).
CACHE_DIR=${CACHE_DIR:-/srv/qwen5090/cache}
CACHE="-v ${CACHE_DIR}/flashinfer:/root/.cache/flashinfer \
       -v ${CACHE_DIR}/torch_compile:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null   # free page cache before the RAM tier claims it

sudo docker run -d --name "$NAME" --restart unless-stopped \
  --entrypoint bash --runtime nvidia --gpus all --ipc=host \
  -p ${PORT}:8000 --shm-size 8g --memory 52g --memory-swap 52g \
  -e LMCACHE_DISABLE_BANNER=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  $CACHE \
  -v "$L2DIR":/l2 \
  -v "$MODEL_DIR":/model \
  "$IMAGE" -c "
    lmcache server --host 0.0.0.0 --port 5555 --chunk-size $BLK \
      --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU \
      --worker-reap-timeout-seconds 0 \
      --l2-adapter '{\"type\":\"fs_native\",\"base_path\":\"/l2\",\"max_capacity_gb\":150,\"num_workers\":4}' \
      > /tmp/lmcache-server.log 2>&1 &
    sleep 8
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model /model --served-model-name qwen3.6-27b --trust-remote-code \
      --kv-cache-dtype fp8_e4m3 \
      --gpu-memory-utilization 0.93 --max-model-len 120000 \
      --max-num-seqs 8 --max-num-batched-tokens $BATCHED \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \
      --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}' \
      --kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}' \
      --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}'
  "

echo "launching $NAME (MTP + LMCache 24G RAM + 150G SSD tiers) on :$PORT ..."
for i in $(seq 1 150); do
  if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
    echo "HEALTHY"
    sudo docker logs "$NAME" 2>&1 | grep -aoE "GPU KV cache size: [0-9,]+ tokens" | tail -1
    exit 0
  fi
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "DIED — see: sudo docker logs $NAME"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 10
done
echo "TIMEOUT"; exit 1

# ------------------------------------------------------------------------------
# NOTES / KNOBS (each one earned by a failure — see ../docs/LMCACHE.md)
#   IMAGE lmcache-vllm:fixed   : 0.5.1 pairing + cudart13 multi-stage COPY. The nightly
#                                pairing REJECTS this model's KV layout (EngineKVFormat 10)
#                                UNLESS you build lmcache with the format-10 kernel patch
#                                (../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch),
#                                which recovers MTP c1 122 / c8 458.
#   NO expandable_segments     : NEVER set PYTORCH_ALLOC_CONF=expandable_segments — cuMem/VMM
#                                memory is not CUDA-IPC-exportable; the sidecar can't import the
#                                KV handles and registration silently times out at 300s.
#   --ipc=host, --entrypoint bash : CUDA-IPC needs host IPC; the image entrypoint is `vllm serve`
#                                and would swallow `bash -c` as a vLLM flag.
#   --memory 52g               : cgroup cap. The 24 GB L1 is PINNED host RAM; drop_caches first.
#   --worker-reap-timeout-seconds 0 : reaper OFF. Lazy-heartbeat reap turns the cache into an
#                                unrecoverable zombie (0 hits, stores silently dropped).
#   --l1-size-gb 24            : L1 must exceed the hot working set / 0.8, or an LRU head-chunk
#                                cascade drops the hit rate to 0% (partial caching does NOT
#                                degrade gracefully on this hybrid).
#   --gpu-memory-utilization 0.93 : the lmcache sidecar holds ~1.4 GB VRAM that util does NOT
#                                account for; 0.94 concurrent-prefill-OOMs (the whole "MTP crashes
#                                with LMCache" myth). 0.93 is the safe ceiling.
#   DEFAULT cudagraph mode     : do NOT force FULL_DECODE_ONLY — it doesn't cover MTP verify-step
#                                shapes on vLLM 0.24 and collapses decode to c1 46 / c8 179.
#   chunk 1600 / batched 3199  : tied to the unified block size (1600 with MTP ns=3, 1568 without);
#                                batched MUST be 2*chunk-1.
# ------------------------------------------------------------------------------
