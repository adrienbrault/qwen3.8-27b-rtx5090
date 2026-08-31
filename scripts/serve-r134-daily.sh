#!/usr/bin/env bash
# DAILY (since 2026-08-31, R134 promotion, user "Yes promote"): DFlash2-fp8-TP2 on the dual 5090s.
#   = launch-daily-v0280.sh with: TP=2, fp8_e4m3 KV, DFlash2 ns7 (syvai W4A16 drafter), native
#   disk tier ON (R133b: the LMCache-era no-tiers rule does not apply to OffloadingConnector),
#   util 0.90, FIWS 256M, NCCL_P2P_LEVEL=SYS (P2P driver 610.57.04 + QuixiAI modules, R129).
# Gauntlet basis: R132 sweep (260/963/1382, deep 172.6/130.8) + R133 quality (tool-eval 90.2,
#   GSM 0.8417, needles 0-false) + R133b tier (revisit green, 239.8 c1 tiered, pool 711,281).
# Known cost: prefill −15% @8–30K vs the MTP daily (parity→+14% by 100K).
# ns9 since R139 (2026-08-31): code c1 325 / prose 173 (beats ns7 both), tool-eval 90.0, needles 4/4.
# Rollback: bash /srv/qwen5090/launch-daily-mtp-0829.sh  (single-GPU nvfp4+MTP config)
set -uo pipefail
# The daily is DETERMINISTIC (2026-08-23 knob-leak incident): drop inherited tuning env unless
# DAILY_ALLOW_ENV=1. Everything the v0280 launcher reads is unset here, then set explicitly.
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT \
        POOL_MIN POOL_MAX TP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR 2>/dev/null || true
fi
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
env PORT=8020 NAME=vllm-27b BIND_ADDR=127.0.0.1 \
  TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=0 FIWS=268435456 UTIL=0.90 \
  MAXLEN=262144 POOL_MIN=650000 POOL_MAX=800000 \
  EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
  SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' \
  EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
  bash /srv/qwen5090/launch-daily-v0280.sh || exit 1

# promotion-specific fail-closed asserts (the v0280 launcher's nvfp4/XQA asserts self-skip on fp8):
BOOTLOG=$(sudo docker logs vllm-27b 2>&1)
[ "$(echo "$BOOTLOG" | grep -aic "dflash")" -ge 1 ] || { echo "FAILED: dflash speculator not engaged in boot log"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "tensor_parallel_size=2\|--tensor-parallel-size 2\|TP1")" -ge 1 ] || { echo "FAILED: TP=2 not engaged"; exit 1; }
echo "DAILY UP (R134 gen: DFlash2-fp8-TP2 + native disk tier, dual 5090, image vllm-qwen38:v0280-nvfp4kv). Rollback: launch-daily-mtp-0829.sh"
