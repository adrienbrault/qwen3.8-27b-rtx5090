#!/usr/bin/env bash
# R165b follow-up (2026-09-03): the fidelity ruler has no same-checkpoint v0.28 baseline — every run-*.jsonl on the
# 2026-08-23 corpus is gittensor-era (run-v0280-fp8kv = gittensor daily). Cell A of r165b measured the RedHat
# checkpoint on the rc1 image (ΔNLL +1.33% vs the FP8 reference), which cannot be read as an rc1 regression until the
# SAME checkpoint on the v0.28 daily is on the same ruler. This runs the ruler (--conc 1, R104c: prompt_logprobs OOM
# above that) against the restored daily on :8020 (read-only measurement, ~2 min) and re-runs the compare.
# Unit: sudo systemd-run --unit=r165b-ref --collect -p User=adrienbrault -p RuntimeMaxSec=3600 bash -c \
#   'while systemctl is-active --quiet r165b-audition; do sleep 30; done; exec bash /srv/qwen5090/r165b-ref-fidelity.sh'
set -uo pipefail
R=/srv/qwen5090/results/2026-09-03-r165b-audition; FD=/srv/qwen5090/results/2026-08-23-fidelity
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
img=$(sudo docker inspect --format '{{.Config.Image}}' vllm-27b 2>/dev/null); ver=$(curl -s -m 5 http://127.0.0.1:8020/version)
case "$img" in *v0280*) ;; *) log "[ref] ABORT: daily :8020 is not the v0.28 image (image='$img' version='$ver')"; exit 3 ;; esac
log "[ref] fidelity ruler on the daily (RedHat, fp8 KV, $img $ver), --conc 1"
python3 /srv/qwen5090/probes/fidelity.py run --url http://127.0.0.1:8020 --model qwen3.6-27b --corpus "$FD/corpus.jsonl" --out "$R/run-v0280-redhat-fp8kv.jsonl" --conc 1 > "$R/fidelity-ref-run.log" 2>&1 || { log "[ref] FAILED: ruler rc=$? (log $R/fidelity-ref-run.log)"; exit 1; }
python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$R/run-v0280-redhat-fp8kv.jsonl" "$R/run-v0290rc1-fp8kv.jsonl" 2>&1 | tee "$R/fidelity-compare-vs-daily.txt" | cut -c1-200 | sed "s/^/[ref fidelity] /" | tee -a "$R/audit.log"
python3 /srv/qwen5090/probes/fidelity.py compare --ref "$R/run-v0280-redhat-fp8kv.jsonl" "$R/run-v0290rc1-fp8kv.jsonl" 2>&1 | tee "$R/fidelity-compare-rc1-vs-v028.txt" | cut -c1-200 | sed "s/^/[ref fidelity rc1-vs-v028] /" | tee -a "$R/audit.log"
log "[ref] DONE"
