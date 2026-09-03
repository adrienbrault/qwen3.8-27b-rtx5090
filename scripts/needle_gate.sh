#!/usr/bin/env bash
# R163 needle gate on the campaign engine (:8030): 131K + 220K x2 must all HIT. Usage: needle_gate.sh <pre|post> <results dir>
# Reason: Bug B (nvfp4 FA2 graph replay) is layout-dependent; SEQS=16 is a layout no cell validated. Without a
# pre AND post gate on the exact campaign engine, a low SWE-bench score cannot be attributed to KV quality.
set -uo pipefail
TAG=$1; R=$2; U=${U:-http://127.0.0.1:8030}
python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 220000 --samples 2 \
  --out "$R/needles-$TAG.jsonl" > "$R/needles-$TAG.out" 2>&1
hits=$(grep -ac "HIT " "$R/needles-$TAG.out"); miss=$(grep -ac "MISS" "$R/needles-$TAG.out")
echo "$(date -Is) [needle-gate $TAG] hits=$hits miss=$miss $(grep -a SUMMARY "$R/needles-$TAG.out" | cut -c1-140)" | tee -a "$R/audit.log"
[ "$hits" -ge 4 ] && [ "$miss" -eq 0 ]
