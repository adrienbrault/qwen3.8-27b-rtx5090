#!/usr/bin/env bash
# R166b (2026-09-03): gate G12 "restart revisit" done properly — the battery's version salted a fresh prompt per run, and a
# manual re-run with the SEQS-8 salt overlapped the flood needles (both sends queued, deltas polluted). Here: boot the
# candidate launcher (EXP=1, SEQS 8) on the eval-l2 tier that still holds the R166 SEQS-8 boot's blocks, engine otherwise
# idle, and send the SAME 32K prompt (salt from results/2026-09-03-r166-gates/warm-revisit-N8.log). Pass = send1 shows
# tier hits (hits_delta ≫ 0, TTFT ≪ the 7.6 s cold prefill) — nvfp4 KV blocks written before a restart are read after it.
# Control: a fresh salt (cold) right after, same boot. Queue-registered; restores the daily at the end (unless queued).
# Unit: sudo systemd-run --unit=r166b-restart --collect -p User=adrienbrault -p RuntimeMaxSec=2400 -p TimeoutStopSec=900 bash /srv/qwen5090/r166b-restart-revisit.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r166-gates
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
source /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "[R166b] waiting for the GPU-exclusive lock"; flock 9; }
log "=== R166b restart revisit start (lock held) ==="
U=http://127.0.0.1:8029
SALT=$(grep -ao '"salt": "[0-9a-f]*"' "$R/warm-revisit-N8.log" | head -1 | grep -oE '[0-9a-f]{32}')
[ -n "$SALT" ] || { log "[R166b] FAILED: no salt in warm-revisit-N8.log"; exit 1; }
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R166b $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
teardown
ls -d /srv/qwen5090/eval-l2/_model_* >/dev/null 2>&1 || log "[R166b] WARNING: no namespace sets on eval-l2 — tier was wiped, send1 cannot hit"
if env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 bash /srv/qwen5090/launch-daily-nvfp4-candidate.sh > "$R/boot-R166b.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then
  log "[R166b] BOOT OK $(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-R166b.log" | tail -1)"
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 --salt "$SALT" > "$R/warm-revisit-R166b-restart.log" 2>&1
  grep -a RESULT "$R/warm-revisit-R166b-restart.log" | cut -c1-260 | sed "s/^/[R166b restart revisit, SEQS-8 salt] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-R166b-cold.log" 2>&1
  grep -a RESULT "$R/warm-revisit-R166b-cold.log" | cut -c1-260 | sed "s/^/[R166b cold control] /" | tee -a "$R/audit.log"
  sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[R166b engine error-lines] /" | tee -a "$R/audit.log"
else log "[R166b] BOOT FAILED: $(grep -aE 'FAILED|Error' "$R/boot-R166b.log" | tail -1 | cut -c1-160)"; fi
finish DONE
