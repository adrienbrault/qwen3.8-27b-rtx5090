#!/usr/bin/env bash
# R195b (2026-09-05, user: "in readme, for these measurements, i want both code and prose"): the README 8- and 16-stream decode rows
# carry code only; this unit reads PROSE at c8 and c16 on the promoted BSS daily (R195), same instrument (probes/decode_ss.py, steady
# state, 2 runs), on the DAILY port, no engine change. Boots the daily from launch-daily.sh if :8020 is down. r196 waits on this unit.
#   unit: sudo systemd-run --unit=r195b-readme-prose --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r195b-readme-prose bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r195b-readme-prose.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r195b-readme-prose; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
[ -f "$PR/decode_ss.py" ] || { log "ABORT: decode_ss.py missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R195b start (lock held): README prose c8/c16 rows on the :8020 BSS daily ==="
if ! curl -sf -m 5 $U/health >/dev/null; then
  log "daily not up: booting it from launch-daily.sh"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "FAILED: daily boot: $(grep -aE 'FAILED' "$R/boot-daily.log" | tail -1 | cut -c1-220)"; exit 1; }
  sleep 20
fi
BOOT=$(sudo docker logs vllm-27b 2>&1)
img=$(sudo docker inspect --format '{{.Config.Image}}' vllm-27b 2>/dev/null)
log "daily image=$img $([ "$img" = "$IMG" ] && echo OK || echo 'NOT THE R195 IMAGE') pcie=$(echo "$BOOT" | grep -ac 'PCIe IPC all-reduce enabled') pool=$(curl -s -m 5 $U/metrics | grep -aoE 'kv_cache_size_tokens="[0-9]+"' | tr -dc 0-9)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-300 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED: $(tail -1 "$R/probe-$name.err" | cut -c1-140)"; fi; }
p1 prose-c8    --conc 8  --tokens 1024 --runs 2 --kind prose
p1 prose-c16   --conc 16 --tokens 1024 --runs 2 --kind prose
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "daily image|RESULT|PROBE FAILED|error lines|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R195b DONE — daily left UP ==="
