#!/usr/bin/env bash
# R175 (2026-09-04, user "Run swebench"): SWE-Bench campaign engine = the PROMOTED daily (R174: vLLM 0.29rc2 + FlashInfer
# 0.6.16.post3, NVFP4 KV pinned per SEQS, DFlash2 ns9 draft_tp2 in CUDA graphs, embed offload, 16 GiB CPU tier) on :8030 /
# vllm-eval through launch-daily.sh EXP=eval — the served launcher itself, asserts included. SEQS=16 (pin 13.98 GB) pairs
# with R160 (fp8 daily, SEQS=16). Tier = eval-l2 (442 GB image) with the daily's 0137 cap (300 GB / min-free 40 GB / scope
# root), so the runner's >=80% tier cycle (wipe + reboot, 4.5 min, twice in R160) never fires. Caller owns GPU exclusivity.
#   usage: bash boot-r174-miniswe.sh   (env: SEQS, BIND_ADDR, EVAL_L2, TIER_CAP_GB, TIER_MIN_FREE_GB overrides)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
EVAL_L2=${EVAL_L2:-/srv/qwen5090/eval-l2}
SEQS=${SEQS:-16}; BIND_ADDR=${BIND_ADDR:-127.0.0.1}
grep -q "0.29 nvfp4 DAILY" /srv/qwen5090/launch-daily.sh || { echo "FAILED: launch-daily.sh is not the 0.29 nvfp4 launcher"; exit 1; }
sudo bash /srv/qwen5090/eval-l2-dio.sh || { echo "FAILED: eval-l2 loop direct-IO setup"; exit 1; }
mountpoint -q "$EVAL_L2" || { echo "FAILED: $EVAL_L2 not mounted"; exit 1; }
env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=eval SEQS="$SEQS" EVAL_BIND="$BIND_ADDR" \
  TIER_CAP_GB="${TIER_CAP_GB:-300}" TIER_MIN_FREE_GB="${TIER_MIN_FREE_GB:-40}" TIER_EVICT_SCOPE=root \
  bash /srv/qwen5090/launch-daily.sh || exit 1
sudo docker logs vllm-eval 2>&1 | grep -aoE "max_num_seqs=[0-9]+" | head -1
echo "EVAL UP (0.29 nvfp4 DAILY config, SEQS=$SEQS, tier $EVAL_L2 cap ${TIER_CAP_GB:-300} GB) on $BIND_ADDR:8030"
