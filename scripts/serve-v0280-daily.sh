#!/usr/bin/env bash
# DAILY launcher, v0.28 generation (R108 promotion, user "promote" 2026-08-28). Stack:
#   image vllm-qwen38:v0280-nvfp4kv = vLLM v0.28.0 + patches-v0280 (0101 sm120-nvfp4 FA2 routing,
#   0102 linear-V-scale writer overlay, 0103 XQA-NVFP4 decode, 0104 drafter full-cudagraph)
#   nvfp4 KV + XQA decode + MTP ns4 + async scheduling (v0.28 default — NEVER add
#   --no-async-scheduling here; it costs 20-40%, R104e) + native OffloadingConnector disk tier
#   on the hard-capped loopback fs (setup-native-l2.sh; replaces LMCache: no sidecar, no 24G
#   pinned DRAM, no chunk=block ceiling). Provenance: FINDINGS R104-R107; public repo
#   bench/RESULTS.md + patches-v0280/. Rollback: launch-daily-legacy-0826.sh (tiers image + L2 intact).
#   Promotion: PORT=8020 NAME=vllm-27b bash launch-daily-v0280.sh   (default = :8029 experiment)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODEL_DIR=${MODEL_DIR:-/srv/qwen5090/models/qwen3.8-27b-nvfp4-rtx5090}  # ≡ HF gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 (R105)
IMAGE=${IMAGE:-vllm-qwen38:v0280-nvfp4kv}
PORT=${PORT:-8029}
BIND_ADDR=${BIND_ADDR:-127.0.0.1}
NAME=${NAME:-vllm-exp}
NS=${NS:-4}
SPEC_JSON=${SPEC_JSON:-}   # full --speculative-config override (R113 suffix arm); empty = MTP ns=$NS
MAXLEN=${MAXLEN:-262144}
UTIL=${UTIL:-0.955}   # R113 U-arm: pool 381,300 (+36K vs 0.93; the cudagraph-profiling equivalence), burst needles + margin green
MNBT=${MNBT:-8192}
SEQS=${SEQS:-8}
L2MNT=${L2MNT:-/srv/qwen5090/native-l2}   # hard-capped loopback fs — cap by construction (R69 lesson)
CACHE_DIR=${CACHE_DIR:-/srv/qwen5090/cache}
POOL_MIN=${POOL_MIN:-340000}   # R113 @0.955: 381,300 expected
POOL_MAX=${POOL_MAX:-420000}
KVT='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":4294967296,"offload_prompt_only":true,"secondary_tiers":[{"type":"fs","root_dir":"/l2","n_read_threads":16,"n_write_threads":4}]}}'
KVT_LINE="--kv-transfer-config '$KVT'"
NO_TIER=${NO_TIER:-0}   # NO_TIER=1: plain engine (diagnostics only — the daily contract includes the tier)
[ "$NO_TIER" = 1 ] && KVT_LINE=""

SPEC_FINAL=${SPEC_JSON:-"{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$NS}"}
NOSPEC=${NOSPEC:-0}          # 1 = no speculative decoding at all (R115 A-arm)
PIP_ARM=${PIP_ARM:-0}
EXTRA_MOUNT=${EXTRA_MOUNT:-}
FIWS=${FIWS:-134217728}
PREFIX_CACHE=${PREFIX_CACHE:-1}   # 0 = --no-enable-prefix-caching (ReplaySSM A/B only)
MAMBA_MODE=${MAMBA_MODE:-align}   # ReplaySSM requires 'none' (loses hybrid prefix caching — R128)
KVD=${KVD_OVERRIDE:-nvfp4}   # fp8_e4m3 for diagnostics/dflash-fp8 arms
EXTRA_ENV=${EXTRA_ENV:-}     # extra docker -e flags (e.g. "-e VLLM_SM12X_DFLASH_ADAPTIVE=1")
TP=${TP:-1}                  # R130: tensor parallelism (dual 5090). TP=2 needs NCCL env via
                             # EXTRA_ENV (-e NCCL_P2P_LEVEL=SYS) + POOL band override (~2x)
TP_LINE=""; [ "$TP" -gt 1 ] && TP_LINE="--tensor-parallel-size $TP"
PP=${PP:-1}                  # R150: pipeline parallelism (codex idea 10; PP=2 = one activation hop, no per-layer allreduce)
PP_LINE=""; [ "$PP" -gt 1 ] && PP_LINE="--pipeline-parallel-size $PP --distributed-executor-backend mp"
CGMODE=${CGMODE:-}           # e.g. piecewise (R119 graph-mode A/B); empty = engine default
FUSIONS=${FUSIONS:-}         # e.g. '\"fuse_norm_quant\":true' extras merged into pass_config (R122)  # 268435456 for dflash-on-nvfp4 (XQA scale scratch for target+draft, R109b/R112)   # e.g. "-v /path/draft:/draft:ro" (R112 dflash arms)        # 1 = pip install arctic-inference before serve (R115 S-arm; the R112 image bakes it)
# R136: CGMODE/FUSIONS were comment-only stubs — wire them into --compilation-config.
# MUST live BELOW the CGMODE/FUSIONS default declarations (set -u; the first r136 launch
# died on 'CGMODE: unbound variable' when this block sat above them, taking the daily
# restore down with it). FUSIONS = raw pass_config pairs, e.g. '"enable_sp":true'.
CC_LINE=""
if [ -n "${CGMODE}" ] || [ -n "${FUSIONS}" ]; then
  CCJ="{"
  [ -n "${CGMODE}" ] && CCJ="${CCJ}\"cudagraph_mode\":\"${CGMODE}\","
  [ -n "${FUSIONS}" ] && CCJ="${CCJ}\"pass_config\":{${FUSIONS}},"
  CCJ="${CCJ%,}}"
  CC_LINE="--compilation-config '$CCJ'"
fi
SPEC_LINE="--speculative-config '$SPEC_FINAL'"
[ "$NOSPEC" = 1 ] && SPEC_LINE=""
PIP_PREFIX=""
[ "$PIP_ARM" = 1 ] && PIP_PREFIX="pip install --no-cache-dir arctic-inference==0.1.1 >/tmp/pip-arm.log 2>&1 && "

mountpoint -q "$L2MNT" || { echo "FAILED: $L2MNT not mounted (run setup-native-l2.sh) — refusing an uncapped tier"; exit 1; }
# Startup GC (R148/codex idea 4): if <40G free, delete namespace sets oldest-first,
# always keeping the most recently modified set. Engines are down at this point.
if [ "$(df -k --output=avail "$L2MNT" | tail -1 | tr -dc 0-9)" -lt 41943040 ]; then
  for ns in $(ls -1t "$L2MNT" | grep '^_model_' | sed 's/_r[0-9]*$//' | awk '!seen[$0]++' | tail -n +2 | tac); do
    [ "$(df -k --output=avail "$L2MNT" | tail -1 | tr -dc 0-9)" -ge 41943040 ] && break
    echo "tier GC: deleting stale namespace $ns"; sudo rm -rf "$L2MNT/$ns" "$L2MNT/${ns}"_r*   # sudo: tier files are root-owned (container-written) — R152 lesson
  done
fi
[ "$(df -k --output=avail "$L2MNT" | tail -1 | tr -dc 0-9)" -ge 5242880 ] || { echo "FAILED: <5G free on $L2MNT — native tier ENOSPC crashes engine-init (R130); wipe stale namespaces"; exit 1; }

# tokenizer truncation guard (gotcha #9, checkpoint-side — survives across engine generations)
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

CACHE="-v ${CACHE_DIR}/torch_compile_qwen38_v0280nv:/root/.cache/vllm/torch_compile_cache \
       -v ${CACHE_DIR}/triton:/root/.triton/cache \
       -v ${CACHE_DIR}/inductor:/root/.cache/inductor \
       -v ${CACHE_DIR}/flashinfer:/root/.cache/flashinfer"

timeout 60 sudo docker rm -f "$NAME" vllm-27b vllm-exp vllm-eval >/dev/null 2>&1 || true
sync

# OffloadingConnector shm-leak sweep (upstream bug, found 2026-08-28): the connector's 4G CPU
# staging mmap in /dev/shm survives `docker rm -f` — 4 leaked boots ate 16G and starved the
# memory gate (and contributed to the 02:07 OOM). Delete only orphans (fuser: no holder).
for f in /dev/shm/vllm_offload_*.mmap; do [ -e "$f" ] || continue; sudo fuser -s "$f" 2>/dev/null || sudo rm -f "$f"; done

# engine-swap memory gate (2026-08-28 OOM incident): wait for the old engine's RAM to be reaped
# threshold 28G: this stack has NO 24G pinned L1 (that was the legacy tiers daily) — engine
# needs ~25G host-side; 40G was the legacy calibration and false-refused a healthy 32G boot.
GATE_KB=${GATE_KB:-29360128}
AVAIL_KB=0
for i in $(seq 1 60); do
  AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$AVAIL_KB" -gt "$GATE_KB" ] && break
  [ "$i" = 1 ] && echo "memory gate: waiting for MemAvailable > $((GATE_KB/1048576))G (now $((AVAIL_KB/1048576))G)..."
  sleep 5
done
[ "$AVAIL_KB" -le "$GATE_KB" ] && { echo "FAILED: memory gate timed out at $((AVAIL_KB/1048576))G — refusing to boot into an OOM window"; exit 1; }

sudo docker run -d --name "$NAME" --restart unless-stopped \
  --entrypoint bash --runtime nvidia --gpus all --ipc=host \
  -p ${BIND_ADDR}:${PORT}:8000 --shm-size 8g --memory 52g --memory-swap 52g \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=${FIWS:-134217728} \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4 \
  -e PYTHONHASHSEED=0 $EXTRA_ENV \
  $CACHE -v "$L2MNT":/l2 -v "$MODEL_DIR":/model $EXTRA_MOUNT \
  "$IMAGE" -c "${PIP_PREFIX}exec python3 -m vllm.entrypoints.openai.api_server \
    --model /model --served-model-name qwen3.8-27b qwen3.6-27b --trust-remote-code \
    --kv-cache-dtype $KVD \
    --gpu-memory-utilization $UTIL --max-model-len $MAXLEN \
    --max-num-seqs $SEQS --max-num-batched-tokens $MNBT \
    --limit-mm-per-prompt '{\"image\":4,\"video\":0}' \
    --mamba-cache-mode "$MAMBA_MODE" $([ "$PREFIX_CACHE" = "1" ] && echo "--enable-prefix-caching" || echo "--no-enable-prefix-caching") \
    $SPEC_LINE \
    $KVT_LINE \
    $TP_LINE \
    $PP_LINE \
    $CC_LINE \
    --default-chat-template-kwargs '{\"preserve_thinking\":true,\"reasoning_effort\":\"medium\"}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}'"

echo "launching $NAME (v0.28 nvfp4+XQA+MTP+native-offload) on ${BIND_ADDR}:$PORT ..."
HEALTHY=0
for i in $(seq 1 150); do
  curl -sf -m 5 http://${BIND_ADDR}:${PORT}/health >/dev/null 2>&1 && { echo "HEALTHY"; HEALTHY=1; break; }
  timeout 15 sudo docker ps --filter name="$NAME" --format x 2>/dev/null | grep -q x || { echo "FAILED: container died"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  [ "$(timeout 15 sudo docker inspect "$NAME" --format '{{.RestartCount}}' 2>/dev/null || echo 0)" -gt 0 ] && { echo "FAILED: engine-init crash loop; log: /tmp/$NAME-crash.log"; sudo docker logs "$NAME" > "/tmp/$NAME-crash.log" 2>&1; grep -aE "Error|raise" "/tmp/$NAME-crash.log" | grep -av Qwen3VLVideo | tail -8; timeout 60 sudo docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "FAILED: /health never came up in 25 min"; sudo docker logs "$NAME" 2>&1 | tail -20; exit 1; }

BOOTLOG=$(sudo docker logs "$NAME" 2>&1)
# fail-closed asserts: every piece of the stack must positively identify itself.
# grep -c, NOT grep -q: with pipefail, -q's early exit SIGPIPEs the echo and fails the
# pipeline on a SUCCESSFUL match (the documented launch-tier-rc4 gotcha; refired here 2026-08-28).
[ "$KVD" != nvfp4 ] || [ "$(echo "$BOOTLOG" | grep -ac "linear-V-scale store overlay ACTIVE")" -ge 1 ] || { echo "FAILED: overlay ACTIVE line missing — swizzled writer would serve silently-wrong KV (R106 D: ΔNLL 8.8%)"; exit 1; }
[ "$KVD" != nvfp4 ] || [ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -ge 1 ] || { echo "FAILED: decode_backend is not xqa — 0103 route did not engage (R107: −28% code decode)"; exit 1; }
if [ "$NO_TIER" != 1 ]; then [ "$(echo "$BOOTLOG" | grep -ac "OffloadingConnector")" -ge 1 ] || { echo "FAILED: OffloadingConnector did not initialize — no disk tier"; exit 1; }; fi
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
echo "daily up. KV pool: ${POOL} tokens"
if [ -z "$POOL" ] || [ "$POOL" -lt "$POOL_MIN" ] || [ "$POOL" -gt "$POOL_MAX" ]; then
  echo "FAILED: pool ${POOL:-<missing>} outside ${POOL_MIN}-${POOL_MAX}"; exit 1
fi

# autotune pre-warm (gotcha #8)
if command -v llama-benchy >/dev/null 2>&1; then
  echo "pre-warming autotune shapes (pp8192 c8)..."
  llama-benchy --base-url http://${BIND_ADDR}:${PORT}/v1 --model qwen3.8-27b \
    --pp 8192 --tg 16 --concurrency 8 --runs 1 >/dev/null 2>&1 || { echo "FAILED: pre-warm errored"; [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || exit 1; }
  echo "pre-warm done. free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
else
  [ "${ALLOW_NO_PREWARM:-0}" = 1 ] || { echo "FAILED: llama-benchy not found (ALLOW_NO_PREWARM=1 to accept)"; exit 1; }
fi

sudo docker restart owui-proxy >/dev/null 2>&1 || true
echo "DAILY UP (v0.28 gen: nvfp4 KV + XQA decode + MTP ns$NS + native disk tier @ $L2MNT, image $IMAGE). Pool $POOL. R104-R107 gauntlet 2026-08-28."
