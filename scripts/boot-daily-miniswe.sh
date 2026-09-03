#!/usr/bin/env bash
# mini-SWE-agent eval engine = the DAILY stack (R156f: RedHatAI NVFP4 weights, DFlash2 ns9 syvai drafter,
# fp8 KV, TP2, MNBT 8192, FIWS 256M, native disk tier, T0.6/top_p0.95/top_k20 generation override) on
# :8030 as vllm-eval, with two SERVING-knob deltas justified by R159 (2026-09-02):
#   SEQS=16  — the daily admits 8; c16 = +34% aggregate decode, the last cell before the ~17-request
#              admission floor (c32/c64 never reach a steady state on this pool).
#   UTIL=0.90 — the boot profiler does not budget the spec-decode sampler logits at max_num_seqs;
#              0.92 OOM'd the SEQS=64 ladder. Headroom over pool (0.90 → ~624K, still 2.4x 262K).
# Tier: its OWN hard-capped loopback image (setup-native-l2.sh with the EVAL vars), not the daily's
# 393G tier (66% full, no runtime eviction — a 500-task campaign would strand the daily again, 09-01).
# Caller owns GPU exclusivity (the v0280 launcher removes vllm-27b/exp/eval itself). Exit != 0 on failure.
#   usage: bash boot-daily-miniswe.sh   (env: SEQS, UTIL, EVAL_L2 overrides)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
EVAL_L2=${EVAL_L2:-/srv/qwen5090/eval-l2}
SEQS=${SEQS:-16}; UTIL=${UTIL:-0.90}
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo bash /srv/qwen5090/eval-l2-dio.sh || { echo "FAILED: eval-l2 loop direct-IO setup"; exit 1; }
mountpoint -q "$EVAL_L2" || { echo "FAILED: $EVAL_L2 not mounted — sudo env IMG=/srv/qwen5090/fast/eval-l2.img MNT=$EVAL_L2 SIZE=450G LABEL=eval-l2 bash setup-native-l2.sh"; exit 1; }
env PORT=8030 NAME=vllm-eval BIND_ADDR=127.0.0.1 MODEL_DIR="$MODEL" \
  TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=0 FIWS=268435456 UTIL="$UTIL" SEQS="$SEQS" \
  MAXLEN=262144 POOL_MIN=580000 POOL_MAX=690000 L2MNT="$EVAL_L2" \
  EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
  SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' \
  EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
  bash /srv/qwen5090/launch-daily-v0280.sh || exit 1
BOOTLOG=$(sudo docker logs vllm-eval 2>&1)
[ "$(echo "$BOOTLOG" | grep -aic dflash)" -ge 1 ] || { echo "FAILED: dflash speculator not engaged"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || { echo "FAILED: not the RedHat checkpoint"; exit 1; }
echo "$BOOTLOG" | grep -aoE "max_num_seqs=[0-9]+" | head -1
echo "EVAL UP (daily stack, SEQS=$SEQS UTIL=$UTIL, tier $EVAL_L2) on 127.0.0.1:8030"
