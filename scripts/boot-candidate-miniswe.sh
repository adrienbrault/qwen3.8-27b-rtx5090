#!/usr/bin/env bash
# R166 (2026-09-03): boot the nvfp4-KV CANDIDATE as the SWE-bench campaign engine on :8030 / vllm-eval — a thin wrapper
# over launch-daily-nvfp4-candidate.sh (EXP=eval), so the campaign runs the promotable launcher itself, asserts included
# (pinned KV budget per SEQS, 0131 image, XQA off, MNBT 8192, FIWS 512M, drafter graphs, min-free guard).
# Same plumbing as boot-daily-miniswe.sh (eval-l2 tier, SEQS=16) so the run pairs with R160 (fp8, SEQS=16).
# History: R163's version used util 0.86 at SEQS 16 as the Bug C workaround; the pin replaces it (FINDINGS R166).
#   usage: bash boot-candidate-miniswe.sh   (env: SEQS, BIND_ADDR, EVAL_L2 overrides)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
EVAL_L2=${EVAL_L2:-/srv/qwen5090/eval-l2}
SEQS=${SEQS:-16}
BIND_ADDR=${BIND_ADDR:-127.0.0.1}   # tb21-style runs set 172.17.0.1 (agents call the engine from task containers)
sudo bash /srv/qwen5090/eval-l2-dio.sh || { echo "FAILED: eval-l2 loop direct-IO setup"; exit 1; }
mountpoint -q "$EVAL_L2" || { echo "FAILED: $EVAL_L2 not mounted"; exit 1; }
env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=eval SEQS="$SEQS" EVAL_BIND="$BIND_ADDR" \
  bash /srv/qwen5090/launch-daily-nvfp4-candidate.sh || exit 1
sudo docker logs vllm-eval 2>&1 | grep -aoE "max_num_seqs=[0-9]+" | head -1
echo "EVAL UP (nvfp4 CANDIDATE, pinned KV budget, SEQS=$SEQS, tier $EVAL_L2) on $BIND_ADDR:8030"
