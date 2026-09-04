#!/usr/bin/env bash
# DAILY (since 2026-09-02, R156 promotion, user "Lets go redhat"): RedHatAI/Qwen3.8-27B-NVFP4 weights
#   on the R134 serving shape — DFlash2-fp8-TP2 on the dual 5090s, syvai W4A16 drafter, ns9, native
#   disk tier, util 0.92, FIWS 256M, NCCL_P2P_LEVEL=SYS. Only the checkpoint changed.
# Why (FINDINGS R156/R156e, flan/r156-DECISION.md): the gittensor daily was +4.46% PPL from bf16 on raw
#   text and diverged from bf16's own greedy agentic trajectories 2.5x more often than RedHat
#   (moderate-bucket flips 8.6% vs 3.4%, uniform across tool/code/reason/prose). RedHat = +0.38% raw.
# Cost (measured, same shape): spec-ON decode ~-6% c1 / -7% c8, prefill -14%, pool 654,491 (-12.4%).
# Gates on this checkpoint: tool-eval 90.2 +-1.0 (69x4, own template), needles 9/9, GSM 0.8583 (n=120).
# Drafter: syvai W4A16 kept — z-lab bf16 drafter costs RedHat -6.5% code c8 and 46K pool (R156 2x2).
# Pool band 620-690K: the profiler's peak-activation estimate is BIMODAL on this shape (1.38 vs 1.98 GiB,
#   identical weights) so the pool lands at 654,491 or 628,798 (the gittensor daily had the same 26K
#   lottery: 746,849 / 720,809). First promotion boot hit the low mode and failed a 640K floor.
# Rollback: bash /srv/qwen5090/launch-daily-gittensor-0831.sh  (previous daily, frozen copy)
#           bash /srv/qwen5090/launch-daily-mtp-0829.sh        (single-GPU nvfp4+MTP config)
set -uo pipefail
# The daily is DETERMINISTIC (2026-08-23 knob-leak incident): drop inherited tuning env unless
# DAILY_ALLOW_ENV=1. Everything the v0280 launcher reads is unset here, then set explicitly.
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT \
        POOL_MIN POOL_MAX TP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR MMLIMIT MMKW \
        TIER_CAP_GB TIER_EVICT_SCOPE TIER_MIN_FREE_GB CPUB 2>/dev/null || true
fi
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4   # HF RedHatAI/Qwen3.8-27B-NVFP4 (llm-compressor, 303 modules kept 8-bit, FP8 lm_head)
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: daily checkpoint missing at $MODEL"; exit 1; }
# BIND 0.0.0.0 is DELIBERATE (pre-R108 launcher's documented rule, restored 2026-08-31 after
# open-webui broke): owui-proxy and harbor task containers reach the engine via 172.17.0.1,
# which a 127.0.0.1 bind refuses; the LAN sits behind the UDM.
env PORT=8020 NAME=vllm-27b BIND_ADDR=0.0.0.0 MODEL_DIR="$MODEL" \
  TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=0 FIWS=268435456 UTIL=0.92 \
  MAXLEN=262144 POOL_MIN=620000 POOL_MAX=690000 \
  CPUB=17179869184 \
  EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
  SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' \
  EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
  bash /srv/qwen5090/launch-daily-v0280.sh || exit 1

# promotion-specific fail-closed asserts (the v0280 launcher's nvfp4/XQA asserts self-skip on fp8):
BOOTLOG=$(sudo docker logs vllm-27b 2>&1)
# CPU tier 16 GiB (2026-09-04, R169/r172): every disk-tier hit is promoted through the CPU tier, so a prompt only gets served if
# ALL its blocks fit there. At 4 GiB (252 blocks x 17 MB) a 131K fp8 prompt (331 blocks) could never be served — the daily's
# tier had never served anything above ~100K. 16 GiB = ~1,010 blocks = ~400K fp8 tokens (a 262K prompt is ~660 blocks).
# Host: 46 GB MemAvailable with the engine up before this change. Asserted here so a silent fallback to 4 GiB cannot happen.
[ "$(sudo docker inspect vllm-27b --format '{{join .Args " "}}' | grep -c '"cpu_bytes_to_use":17179869184')" -ge 1 ] || { echo "FAILED: CPU tier is not 16 GiB on the container"; exit 1; }
CPUBLK=$(echo "$BOOTLOG" | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1 | grep -oE '[0-9]+'); [ "${CPUBLK:-0}" -ge 900 ] || { echo "FAILED: CPU tier has ${CPUBLK:-?} blocks (expected ~1,010 at 16 GiB)"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -aic "dflash")" -ge 1 ] || { echo "FAILED: dflash speculator not engaged in boot log"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "tensor_parallel_size=2\|--tensor-parallel-size 2\|TP1")" -ge 1 ] || { echo "FAILED: TP=2 not engaged"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || { echo "FAILED: boot log does not identify the RedHat compressed-tensors checkpoint"; exit 1; }
echo "DAILY UP (R156 gen: RedHatAI NVFP4 weights on DFlash2-fp8-TP2 + native disk tier, dual 5090, image vllm-qwen38:v0280-nvfp4kv). Rollback: launch-daily-gittensor-0831.sh"
