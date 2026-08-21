#!/usr/bin/env bash
# TIER-RC4 on the V2 MODEL RUNNER (PROMOTED 2026-08-21, user-approved, FINDINGS R79): vLLM nightly
# ba07e4a48 (08-21) + LMCache 0.5.4rc4 + rebased stack (patches-rc4/), VLLM_USE_V2_MODEL_RUNNER=1.
# vs the 08-15 V1 daily measured the same hour: pool 209,859 (+4%), decode c1 152 / c4 360 / deep 143
# (+19%/+16%/+20%), needles+L2+restart-proof+killer clean, tool-eval 91+-0. Fresh L2 namespace rc4-v2.
# Previous generation: IMAGE=vllm-qwen38:tiers-rc4 EXTRA_ENV= L2DIR=/srv/qwen5090/lmcache-l2-rc4.
# Derived from launch-tier-daily.sh; nightly deltas: qwen3 parser, effort=medium kwarg,
# no SO-config graft (native), no deepseek_r1, fresh rc4 L2 namespace.
# THE DAILY. Qwen3.8-27B (PROMOTED 2026-08-14, user-approved, replacing natfii 3.6):
# sakamakismile MTP-NVFP4 W4A4 (natfii recipe on 3.8) + fp8_e4m3 KV + FlashInfer + MTP ns=4
# + vision, with LMCache tiered KV offload (fresh L2 namespace — 3.6 chunks are NOT valid
# for 3.8). Why (results dir 2026-08-14-qwen38-nvfp4): 69x2 pairs 91/90 vs 3.6's 89.0+-1.4;
# pool at parity; killer cold round 8/8 @98% pool; needle/vision/tools clean.
# 3.8 serving deltas vs 3.6: template prefills <think> + reasoning-effort system line
# (default xhigh) -> deepseek_r1 parser; T=0.6 override kept (measured better than model-default T=1.0);
# API serves reasoning as message.reasoning. Alias qwen3.6-27b kept for existing clients.
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
export PATH="$HOME/.local/bin:$PATH"   # llama-benchy (pre-warm) commonly lives here

MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/saka-qwen3.8-27b-mtp-nvfp4}  # sakamakismile/Qwen3.8-27B-MTP-NVFP4
MODEL_NAME=${MODEL_NAME:-qwen3.8-27b}
MODEL_ALIAS=${MODEL_ALIAS:-qwen3.6-27b}   # legacy name — hermes/owui/prime configs still say 3.6
IMAGE=${IMAGE:-vllm-qwen38:tiers-rc4-ba07e4a}   # Dockerfile.rc4 --build-arg VLLM_BASE=<08-21 nightly digest>
PORT=${PORT:-8029}   # gauntlet default = experiment port; promotion passes PORT=8020
BIND_ADDR=${BIND_ADDR:-127.0.0.1}    # loopback by default — the API has NO auth. Set 0.0.0.0
                                     # only behind a firewall/VPN or an authenticated proxy.
NAME=${NAME:-vllm-exp}
KVDTYPE=${KVDTYPE:-fp8_e4m3}   # R81 audition: nvfp4 (needs IMAGE=vllm-qwen38:tiers-nvfp4kv = tier image + patches-nvfp4kv)
# Unified hybrid block (vLLM: "Setting attention block size to N tokens to ensure that attention page
# size is >= mamba page size"): fp8 KV + MTP ns=4 -> 1616 (1568 without MTP); nvfp4 KV -> 2864 (0.5625 B/elt
# inflates the attention block to cover the mamba page). LMCache chunk MUST equal this block (rc4 rule).
case "$KVDTYPE" in nvfp4*) BLK_DEFAULT=2864 ;; *) BLK_DEFAULT=1616 ;; esac
BLK=${BLK:-$BLK_DEFAULT}             # NOT 16. Override only after reading the engine's own line.
BATCHED=$((2 * BLK - 1))             # LMCache MP requires batched tokens in [chunk, 2*chunk)

# L2 NVMe tier. 200 GB ~= 2.13M tokens, survives container restarts.
# The cap is only real because of patch 0008 + the eviction block below — verify both.
L2DIR=${L2DIR:-/srv/qwen5090/lmcache-l2-rc4-v2}   # FRESH namespace per stack generation (R79: V2 + new nightly)
UTIL=${UTIL:-0.95}
EXTRA_ENV=${EXTRA_ENV-"-e VLLM_USE_V2_MODEL_RUNNER=1"}   # R79: V2 model runner is the daily. `${VAR-default}` on purpose: an EXPLICIT empty EXTRA_ENV= reverts to V1 (`:-` would re-apply the default)
L2CAP=${L2CAP:-200}
sudo mkdir -p "$L2DIR"
# Orphaned temp files from a crashed sidecar count against the L2 cap (patch 0008's
# restart accounting) but are never indexed or evictable — sweep them before boot.
sudo find "$L2DIR" -type f -path "*tmp*" -mmin +10 -delete 2>/dev/null || true

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
CACHE="-v ${CACHE_DIR}/torch_compile_qwen38_nightly:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor \
       -v ${CACHE_DIR}/flashinfer:/root/.cache/flashinfer"

sudo docker rm -f "$NAME" vllm-27b vllm-exp vllm-eval sglang-exp >/dev/null 2>&1 || true
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null   # the 24 GB L1 is PINNED RAM

sudo docker run -d --name "$NAME" --restart unless-stopped \
  --entrypoint bash --runtime nvidia --gpus all --ipc=host \
  -p ${BIND_ADDR}:${PORT}:8000 --shm-size 8g --memory 52g --memory-swap 52g \
  -e LMCACHE_DISABLE_BANNER=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 \
  -e LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1 -e CUDA_MODULE_LOADING=LAZY \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 $EXTRA_ENV \
  $CACHE -v "$L2DIR":/l2 -v "$MODEL_DIR":/model \
  "$IMAGE" -c "
    lmcache server --host 0.0.0.0 --port 5555 --chunk-size $BLK \
      --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU \
      --worker-reap-timeout-seconds 0 \
      --l2-adapter '{\"type\":\"fs_native\",\"base_path\":\"/l2\",\"max_capacity_gb\":$L2CAP,\"num_workers\":4,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}' \
      > /tmp/lmcache-server.log 2>&1 &
    sleep 8
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model /model --served-model-name $MODEL_NAME $MODEL_ALIAS --trust-remote-code \
      --kv-cache-dtype $KVDTYPE --no-async-scheduling \
      --gpu-memory-utilization $UTIL --max-model-len 200000 \
      --max-num-seqs 8 --max-num-batched-tokens $BATCHED \
      --limit-mm-per-prompt '{\"image\":4,\"video\":0}' \
      --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
      --kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}' \
      --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":4}' \
      --default-chat-template-kwargs '{\"preserve_thinking\":true,\"reasoning_effort\":\"medium\"}' \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \\
      --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}'
      # T=0.6 override RESTORED after the 2026-08-14 battery: tier 69x4 = 90.5+-2.1 @T0.6 vs 87.8+-1.3 @T1.0
      # (model-default T=1.0 costs ~2.7pt and doubles trial variance; per-request temperature still wins).
  "

echo "launching $NAME (saka 3.8 W4A4 + fp8 KV + MTP ns=4 + LMCache 24G DRAM / ${L2CAP}G NVMe) on ${BIND_ADDR}:$PORT ..."
HEALTHY=0
for i in $(seq 1 150); do
  curl -sf http://${BIND_ADDR}:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; HEALTHY=1; break; }
  sudo docker ps --filter name="$NAME" --format x | grep -q x || { echo "FAILED: container died — see: sudo docker logs $NAME"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 10
done
if [ "$HEALTHY" != 1 ]; then
  echo "FAILED: /health never came up within 25 min"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1
fi

# --- pool assertion: 239K here means the connector did NOT attach (plain engine booted
# by accident) and every "tier hit" you measure afterwards would be vLLM's own prefix
# cache. 165K/185K/205K = stale util. Fail closed instead of printing a warning nobody reads.
POOL=$(sudo docker logs "$NAME" 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
echo "daily up. KV pool: ${POOL} tokens"
POOL_MIN=${POOL_MIN:-195000}   # R79: V2 + ba07e4a48 tier boots at 209,859 (profiling varies ~6% boot to boot)
POOL_MAX=${POOL_MAX:-225000}
if [ -z "$POOL" ] || [ "$POOL" -lt "$POOL_MIN" ] || [ "$POOL" -gt "$POOL_MAX" ]; then
  echo "FAILED: pool ${POOL:-<missing>} outside expected ${POOL_MIN}-${POOL_MAX} (util 0.95, seqs 8, mnbt 3231)."
  echo "  Pool alone can NOT prove the connector attached at these settings — see the positive check below."
  exit 1
fi

# --- positive connector check: pool value cannot distinguish tiers-vs-plain at identical
# util/mnbt, so demand direct evidence the LMCache connector initialized AND the sidecar
# is alive with registered workers. Fail closed otherwise.
# NB: no `grep -q` here — with pipefail, -q's early exit SIGPIPEs `docker logs` (141)
# and reads as a false "no connector" (this exact bug fired on the 2026-08-14 promotion).
CONNECTOR_HITS=$(sudo docker logs "$NAME" 2>&1 | grep -ac "LMCacheMPConnector" || true)
if [ "${CONNECTOR_HITS:-0}" -lt 1 ]; then
  echo "FAILED: no LMCacheMPConnector init in engine logs — plain engine booted under the tier launcher."; exit 1
fi
if ! sudo docker exec "$NAME" sh -c 'test -s /tmp/lmcache-server.log'; then
  echo "FAILED: lmcache sidecar log missing/empty — sidecar not running."; exit 1
fi

# --- autotune shape pre-warm (gotcha #8): load-bearing for the OOM margin. Fail closed
# if it can't run — set ALLOW_NO_PREWARM=1 to accept the un-warmed margin knowingly.
if command -v llama-benchy >/dev/null 2>&1; then
  echo "pre-warming autotune shapes (pp8192 c8, ~60s)..."
  if llama-benchy --base-url http://${BIND_ADDR}:${PORT}/v1 --model $MODEL_NAME \
      --pp 8192 --tg 16 --concurrency 8 --runs 1 >/dev/null 2>&1; then
    echo "pre-warm done. free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  else
    echo "FAILED: autotune pre-warm errored — the first real deep burst will pay the workspace allocation."
    [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || exit 1
  fi
else
  echo "WARNING: llama-benchy not found — autotune shapes NOT pre-warmed (gotcha #8)."
  [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || { echo "Install llama-benchy or set ALLOW_NO_PREWARM=1."; exit 1; }
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
