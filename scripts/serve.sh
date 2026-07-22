#!/usr/bin/env bash
# THE DAILY. Qwen3.6-27B: natfii NVFP4 W4A4 + fp8_e4m3 KV + FlashInfer + MTP ns=4 + vision,
# with LMCache tiered KV offload: 214K on-GPU + ~245K pinned DRAM + ~2.13M NVMe
# = ~2.59M reusable tokens, and the NVMe tier survives restarts.
#
# Requires the TIER image (../patches/lmcache/) — six local patches on top of the base image.
# Running this profile on stock LMCache is WORSE THAN NOT CACHING: stores are silently
# wrong-addressed and retrieves restore garbage state. Read ../patches/lmcache/README.md.
#
# For the no-LMCache variant (bigger hot pool, no tiers, no patches), see ./serve-plain.sh
# and "What removing LMCache changes" in ../docs/LMCACHE.md.
#
#   MODEL_DIR=/path/to/Qwen3.6-27B-VLM-NVFP4-MTP ./serve.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/natfii-27b-nvfp4}   # natfii/Qwen3.6-27B-VLM-NVFP4-MTP
IMAGE=${IMAGE:-vllm-qwen36:tiers}    # build from ../patches/lmcache/Dockerfile
PORT=${PORT:-8020}
NAME=vllm-27b
BLK=1616                             # unified block size with MTP ns=4 (1568 without). NOT 16.
BATCHED=$((2 * BLK - 1))             # LMCache MP requires batched tokens in [chunk, 2*chunk)

# L2 NVMe tier. 200 GB ~= 2.13M tokens, survives container restarts.
# The cap is only real because of patch 0008 + the eviction block below — verify both.
L2DIR=${L2DIR:-/srv/qwen5090/lmcache-l2-natfii}
L2CAP=${L2CAP:-200}
sudo mkdir -p "$L2DIR"

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
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null   # the 24 GB L1 is PINNED RAM

sudo docker run -d --name "$NAME" --restart unless-stopped \
  --entrypoint bash --runtime nvidia --gpus all --ipc=host \
  -p ${PORT}:8000 --shm-size 8g --memory 52g --memory-swap 52g \
  -e LMCACHE_DISABLE_BANNER=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 \
  -e LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1 -e CUDA_MODULE_LOADING=LAZY \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  $CACHE -v "$L2DIR":/l2 -v "$MODEL_DIR":/model \
  "$IMAGE" -c "
    lmcache server --host 0.0.0.0 --port 5555 --chunk-size $BLK \
      --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU \
      --worker-reap-timeout-seconds 0 \
      --l2-adapter '{\"type\":\"fs_native\",\"base_path\":\"/l2\",\"max_capacity_gb\":$L2CAP,\"num_workers\":4,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}' \
      > /tmp/lmcache-server.log 2>&1 &
    sleep 8
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model /model --served-model-name qwen3.6-27b --trust-remote-code \
      --kv-cache-dtype fp8_e4m3 --no-async-scheduling \
      --gpu-memory-utilization 0.95 --max-model-len 200000 \
      --max-num-seqs 8 --max-num-batched-tokens $BATCHED \
      --limit-mm-per-prompt '{\"image\":4,\"video\":0}' \
      --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
      --kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}' \
      --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":4}' \
      --structured-outputs-config '{\"backend\":\"xgrammar\",\"reasoning_parser\":\"qwen3\",\"enable_in_reasoning\":false}' \
      --default-chat-template-kwargs '{\"preserve_thinking\":true}' \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}'
  "

echo "launching $NAME (natfii W4A4 + fp8 KV + MTP ns=4 + LMCache 24G DRAM / ${L2CAP}G NVMe) on :$PORT ..."
for i in $(seq 1 150); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "DIED — see: sudo docker logs $NAME"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 10
done
echo "daily up. KV pool: $(sudo docker logs $NAME 2>&1 | grep -a 'GPU KV cache size' | tail -1)"
echo "  ^ VERIFY this is ~214K (util 0.95 + sidecar, at --max-num-seqs 8). 239K means the"
echo "    connector did NOT attach (you booted the plain profile); 165K/185K/205K = stale util."
echo "    Raising --max-num-seqs shifts it slightly: seqs 16 measures 211,267 (-1.3%)."

# --- autotune shape pre-warm (gotcha #8) --------------------------------------
if command -v llama-benchy >/dev/null 2>&1; then
  echo "pre-warming autotune shapes (pp8192 c8, ~60s)..."
  llama-benchy --base-url http://localhost:${PORT}/v1 --model qwen3.6-27b \
    --pp 8192 --tg 16 --concurrency 8 --runs 1 >/dev/null 2>&1 || true
  echo "pre-warm done. free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
fi

sudo docker restart owui-proxy >/dev/null 2>&1 || true   # so Open WebUI re-discovers

# ------------------------------------------------------------------------------
# NOTES / KNOBS — the tier-specific ones. For everything shared with the plain
# profile (PR #42603, --no-async-scheduling, no --quantization flag, the tokenizer
# guard, util-vs-pool arithmetic) see ./serve-plain.sh, which annotates them all.
#
#   IMAGE vllm-qwen36:tiers    : MUST carry patches 0001/0002/0007/0008 (LMCache) and
#                                0003/0005 (vLLM). On stock LMCache this profile stores
#                                wrong-addressed pages and restores garbage recurrent
#                                state — fluent output, vanished facts, no errors logged.
#                                ../patches/lmcache/README.md has the full table.
#   --ipc=host, --entrypoint bash : CUDA-IPC needs host IPC; the image entrypoint is
#                                `vllm serve` and would swallow our `bash -c`.
#   NO expandable_segments     : NEVER set PYTORCH_ALLOC_CONF=expandable_segments — cuMem/VMM
#                                memory is not CUDA-IPC-exportable (pytorch#165685,
#                                vllm#29544); the sidecar can't import the KV handles and
#                                register_kv_caches silently times out at 300s.
#   chunk 1616 / batched 3231  : chunk MUST equal vLLM's unified block size (1616 with MTP
#                                ns=4, 1568 without — discovered, not documented) and
#                                batched MUST be 2*chunk-1. This ceiling is why the tier
#                                profile can't use the plain daily's mnbt 4096.
#   --l1-size-gb 24            : PINNED host RAM; drop_caches first (done above), and the
#                                cgroup --memory 52g must leave room for it. L1 must exceed
#                                hot-working-set/0.8 or an LRU head-chunk cascade drops the
#                                hit rate to 0% — partial caching does NOT degrade gracefully.
#   --worker-reap-timeout-seconds 0 : reaper OFF. Lazy-heartbeat reap turns the cache into an
#                                unrecoverable zombie (found_count=0, stores silently dropped).
#   L2 eviction block          : patch 0008 enforces max_capacity_gb; this JSON block is what
#                                actually evicts. You need BOTH. Unpatched + unset, L2 grew to
#                                876 GB against a 60 GB cap and filled the root filesystem.
#                                Monitor `du -sh $L2DIR` for the first day of any rollout.
#   staging=1 + LAZY modules   : sidecar VRAM 1412 -> 796 MiB, zero latency cost. That's what
#                                bought util 0.95 (pool 214,084) instead of 0.92 (185,538).
#   --gpu-memory-utilization 0.95 : the sidecar's ~796 MiB is invisible to this flag, so the
#                                tier ceiling sits BELOW the plain daily's 0.98. Validated by
#                                an 858-cycle soak (needle+killer+vision per cycle, free VRAM
#                                flat at 701 MiB, L2 oscillating 39-47G under the 60G cap).
#                                Fallbacks: 0.94 -> 205,633 (killer floor 255 MiB), 0.92 -> 185,538.
#   WIPE poisoned L2           : any namespace written by a pre-0005 build must be deleted.
#                                0005 stops new poisoning; it does not repair stored chunks.
#   --structured-outputs-config: carried over from the plain daily (the #44993 graft is in the
#                                base image). This is the ONE flag not covered by the tier
#                                battery — probe response_format+thinking after first boot.
# TIERS: GPU 214,084 tok (~1-2s revisit) / DRAM ~245K (~2s) / NVMe ~2.13M (~4.4-7.5s, survives
#   restarts) vs ~11-13s cold re-prefill. Quality 69x2 = 89 (baseline ~89.8).
# ------------------------------------------------------------------------------
