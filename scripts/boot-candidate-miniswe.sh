#!/usr/bin/env bash
# R163: boot the nvfp4-KV CANDIDATE (R158c shape) as the SWE-bench campaign engine on :8030 / vllm-eval.
# Same plumbing as boot-daily-miniswe.sh (eval-l2 tier, SEQS=16, util 0.90) so the run pairs with R160 (fp8).
# Shape = launch-daily-nvfp4-candidate.sh: RedHat NVFP4 weights, nvfp4 KV, DFlash2 ns9 draft_tp2 in CUDA graphs
# (0129), XQA off + MNBT 8192 + FIWS 512M (Bug B dodge), image vllm-qwen38:v0280-nvfp4kv-revival-graphs.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
IMG=vllm-qwen38:v0280-nvfp4kv-revival-graphs
EVAL_L2=${EVAL_L2:-/srv/qwen5090/eval-l2}
# UTIL 0.88, not the shape's 0.90 (2026-09-03 03:34 UTC): at SEQS=16 the nvfp4 pool came out at 984,411 — the same as at
# SEQS=8 — i.e. the profiler reserved nothing for the wider layout, and the first 131K prefill OOM'd in execute_model
# (161 MB wanted, 21 MB free; engine.log in results/2026-09-02-miniswe-rh-nvfp4). The fp8 daily shrinks its pool by
# 27K tokens (~0.47 GB/GPU) for SEQS=16; 0.88 frees ~0.63 GB/GPU here. Pool expected ≈ 910K.
# 04:34 UTC: at SEQS=32 the pool was again unchanged (983,314) and the engine OOM'd already at FULL graph capture
# (8.5 MB free) — the shortfall scales with SEQS and the profiler sees none of it ("Bug C" in PLAN-PROMOTION.md).
# 0.86 (~1.26 GB/GPU margin) is the safe campaign setting; pool ≈ 840K, still +34% over fp8's 627K at SEQS=16.
SEQS=${SEQS:-16}; UTIL=${UTIL:-0.86}
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (dflash-nvfp4-revival/Dockerfile.graphs)"; exit 1; }
sudo bash /srv/qwen5090/eval-l2-dio.sh || { echo "FAILED: eval-l2 loop direct-IO setup"; exit 1; }
mountpoint -q "$EVAL_L2" || { echo "FAILED: $EVAL_L2 not mounted"; exit 1; }
env PORT=8030 NAME=vllm-eval BIND_ADDR=127.0.0.1 MODEL_DIR="$MODEL" IMAGE="$IMG" \
  TP=2 KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 NO_TIER=0 FIWS=536870912 MNBT=8192 UTIL="$UTIL" SEQS="$SEQS" \
  MAXLEN=262144 POOL_MIN=800000 POOL_MAX=1100000 L2MNT="$EVAL_L2" \
  EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
  SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}' \
  EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1" \
  bash /srv/qwen5090/launch-daily-v0280.sh || exit 1
BOOTLOG=$(sudo docker logs vllm-eval 2>&1)
[ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || { echo "FAILED: drafter graphs not captured (0129 inactive?)"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || { echo "FAILED: drafter fell back to eager"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || { echo "FAILED: not the RedHat checkpoint"; exit 1; }
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || { echo "FAILED: XQA decode engaged — Bug B dodge not in force"; exit 1; }
echo "$BOOTLOG" | grep -aoE "max_num_seqs=[0-9]+" | head -1
echo "EVAL UP (nvfp4 CANDIDATE, SEQS=$SEQS UTIL=$UTIL, tier $EVAL_L2) on 127.0.0.1:8030"
