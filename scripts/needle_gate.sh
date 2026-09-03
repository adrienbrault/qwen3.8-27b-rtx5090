#!/usr/bin/env bash
# R163 needle gate on the campaign engine (:8030): 131K + 220K x2 must all HIT. Usage: needle_gate.sh <pre|post> <results dir>
# Reason: Bug B (nvfp4 FA2 graph replay) is layout-dependent; SEQS=16 is a layout no cell validated. Without a
# pre AND post gate on the exact campaign engine, a low SWE-bench score cannot be attributed to KV quality.
# R166c (2026-09-03): each needle is also re-asked after a 12 x 90K flood evicts it from the GPU pool (pool at SEQS 16 =
# 859K tokens), so the answer must come back through the CPU/disk tier — the only exact-match check of tier-served blocks.
# Every earlier "warm revisit" pass was a GPU prefix-cache hit; the campaign reloads evicted prefixes constantly.
# R166d: each evicted needle is re-asked EVICT_REASKS times (default 2) because the disk lookup is async — the first
# re-ask is recomputed on every KV dtype (R166c), the second can be served from the CPU tier; the SUMMARY's
# tier_served / tier_served_hits say whether the tier actually served blocks on this engine and whether they were right.
# Pass = every cold AND every evicted re-ask hits (probe exit code). EVICT=0 restores the R163 behaviour.
set -uo pipefail
TAG=$1; R=$2; U=${U:-http://127.0.0.1:8030}; EVICT=${EVICT:-12}; EVICT_CTX=${EVICT_CTX:-90000}; EVICT_REASKS=${EVICT_REASKS:-2}
python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 220000 --samples 2 \
  --evict "$EVICT" --evict-ctx "$EVICT_CTX" --evict-reasks "$EVICT_REASKS" --out "$R/needles-$TAG.jsonl" > "$R/needles-$TAG.out" 2>&1; rc=$?
hits=$(grep -ac "^\[HIT " "$R/needles-$TAG.out"); miss=$(grep -ac "MISS" "$R/needles-$TAG.out")
echo "$(date -Is) [needle-gate $TAG] hits=$hits miss=$miss rc=$rc $(grep -a SUMMARY "$R/needles-$TAG.out" | cut -c1-200)" | tee -a "$R/audit.log"
grep -a "^\[" "$R/needles-$TAG.out" | cut -c1-260 | sed "s/^/[needle-gate $TAG] /" >> "$R/audit.log"
[ "$rc" -eq 0 ]
