#!/usr/bin/env bash
# CANDIDATE daily (R158, 2026-09-02) — NOT promoted; the user flips by running this instead of
# launch-daily.sh. Same checkpoint/drafter/tier as the daily; the KV cache becomes NVFP4 and the DFlash2
# drafter runs in CUDA graphs (patch 0129) on the revival image chain (0116-0119 + 0129).
#   vs the fp8 daily (R158b/R158c numbers in flan/FINDINGS.md): pool ~1.03M @262K (+57%), single-stream
#   decode above the fp8 route at equal drafter settings, c8 parity, fidelity cost 0.4 pp top-1 (R156e hn).
# Shape constraints (R155 Bug B dodge, still required): XQA OFF (VLLM_SM12X_NVFP4_XQA=0 + ALLOW_NO_XQA=1)
#   and --max-num-batched-tokens 8192 with a 512 MiB FlashInfer workspace; util 0.90 (tier-on value
#   validated in R157/R158c; the fp8 daily runs 0.92 with a 256 MiB workspace).
# Rollback: bash /srv/qwen5090/launch-daily.sh (fp8 daily, unchanged).
set -uo pipefail
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT EXTRA_ARGS \
        POOL_MIN POOL_MAX TP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR ALLOW_NO_XQA 2>/dev/null || true
fi
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect vllm-qwen38:v0280-nvfp4kv-revival-graphs >/dev/null 2>&1 || { echo "FAILED: image vllm-qwen38:v0280-nvfp4kv-revival-graphs missing (build: dflash-nvfp4-revival/Dockerfile.graphs)"; exit 1; }
env PORT=8020 NAME=vllm-27b BIND_ADDR=0.0.0.0 MODEL_DIR="$MODEL" TP=2 \
    IMAGE=vllm-qwen38:v0280-nvfp4kv-revival-graphs KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=8192 SEQS=8 UTIL=0.90 MAXLEN=262144 POOL_MIN=950000 POOL_MAX=1050000 \
    EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}' \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "CANDIDATE FAILED — daily NOT up; run launch-daily.sh"; exit 1; }
BOOTLOG=$(sudo docker logs vllm-27b 2>&1)
[ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || { echo "FAILED: drafter graphs not captured (0129 inactive?)"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || { echo "FAILED: drafter fell back to eager"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || { echo "FAILED: checkpoint identity"; exit 1; }
echo "CANDIDATE UP (R158: RedHat NVFP4 weights + NVFP4 KV + DFlash2 ns9 draft_tp2 in CUDA graphs (0129) + native disk tier, dual 5090, image vllm-qwen38:v0280-nvfp4kv-revival-graphs). Rollback: launch-daily.sh"
