#!/usr/bin/env bash
# R189b (2026-09-05): the README decode rows re-measured on the promoted pcie_ipc daily (R189 gated it on :8020 but ran only code c8 /
# prose c1 / code c16; the README rows are code c1, prose c1, prose at 30K, code c8, code c16 from R183's BASE arm of 2026-09-04, before
# the promotion). Same instrument (probes/decode_ss.py, steady-state, R183 run counts), on the DAILY port, no engine change. Last unit of
# the chain: r193b's finish() skips its restore because this unit is queued, so this unit brings the daily up if it is not.
#   unit: sudo systemd-run --unit=r189b-readme-decode --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r189b-readme-decode bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r193b-bss-artifact; do sleep 30; done; exec bash /srv/qwen5090/r189b-readme-decode.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r189b-readme-decode; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
[ -f "$PR/decode_ss.py" ] || { log "ABORT: decode_ss.py missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R189b start (lock held): README decode rows on the :8020 daily ==="
if ! curl -sf -m 5 $U/health >/dev/null; then
  log "daily not up: booting it from launch-daily.sh"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "FAILED: daily boot: $(grep -aE 'FAILED' "$R/boot-daily.log" | tail -1 | cut -c1-220)"; exit 1; }
  sleep 20
fi
BOOT=$(sudo docker logs vllm-27b 2>&1)
img=$(sudo docker inspect --format '{{.Config.Image}}' vllm-27b 2>/dev/null)
log "daily image=$img $([ "$img" = "$IMG" ] && echo OK || echo 'NOT THE R189 IMAGE') pcie=$(echo "$BOOT" | grep -ac 'PCIe IPC all-reduce enabled') pool=$(curl -s -m 5 $U/metrics | grep -aoE 'kv_cache_size_tokens="[0-9]+"' | tr -dc 0-9)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-300 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED: $(tail -1 "$R/probe-$name.err" | cut -c1-140)"; fi; }
p1 code-c1     --conc 1  --tokens 1024 --runs 3 --kind code
p1 prose-c1    --conc 1  --tokens 1024 --runs 3 --kind prose
p1 prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
p1 code-c8     --conc 8  --tokens 1024 --runs 2 --kind code
p1 code-c16    --conc 16 --tokens 1024 --runs 2 --kind code
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "daily image|RESULT|PROBE FAILED|error lines|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R189b DONE — daily left UP ==="
