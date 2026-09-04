#!/usr/bin/env bash
# R178 (2026-09-04, user "we could go from 8 to 12 instead of 8 to 16?"): admission + decode + pool at --max-num-seqs 12, 13 and 17
# on the daily image, :8029 (EXP=1), daily down for the chain. R177 found the daily at SEQS 16 runs 15 and queues the 16th; SEQS 8
# ran 8. Questions: does SEQS 12 admit 12 (or 11)? does 13 admit 12? does 17 admit 16 (i.e. is the rule N-1 above 8)? What do
# c8 / c(N-1) / cN aggregate decode and the pool read at each? Pins interpolated from the launcher table (8 -> 14.5 GB, 16 ->
# 13.98 GB, 32 -> 13.44 GB); each arm steps the pin down if Bug C headroom is missing.
# Unit: sudo systemd-run --unit=r178-seqs-ladder --collect -p User=adrienbrault -p RuntimeMaxSec=5400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r178-seqs-ladder bash /srv/qwen5090/r178-seqs-ladder.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r178-seqs-ladder; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R178 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R178 SEQS admission ladder start (lock held): image=$IMG arms 12/13/17 on :8029 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
declare -A POOL FREE
boot_arm(){ local seqs=$1 kv rc; shift
  for kv in "$@"; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=$seqs KV_BYTES=$kv bash $CAND > "$R/boot-seqs$seqs.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      POOL[$seqs]=$(grep -aoE 'Pool [0-9]+' "$R/boot-seqs$seqs.log" | tail -1 | tr -dc 0-9); FREE[$seqs]=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-seqs$seqs.log" | tail -1 | tr -dc 0-9)
      log "[seqs$seqs] BOOT OK pin=$kv pool=${POOL[$seqs]} min_free=${FREE[$seqs]}MiB max_num_seqs=$(sudo docker logs vllm-exp 2>&1 | grep -aoE "max_num_seqs': [0-9]+" | tail -1)"
      return 0; fi
    log "[seqs$seqs] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-seqs$seqs.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-seqs$seqs.log" || break; teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
adm(){ log "[$1 admission c=$2] $(python3 /srv/qwen5090/probes/admission_probe.py --url $U --conc $2 2>&1 | tail -1 | cut -c1-260)"; }
arm(){ local seqs=$1; shift; local tag=seqs$seqs
  boot_arm $seqs "$@" || { log "[$tag] BOOT FAILED on every pin"; teardown; return 1; }
  sleep 30
  adm $tag $seqs; adm $tag $((seqs+1))
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $tag code-c$((seqs-1)) --conc $((seqs-1)) --tokens 1024 --runs 2 --kind code
  p1 $tag code-c$seqs --conc $seqs --tokens 1024 --runs 2 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  log "[$tag engine error-lines] $(sudo docker logs vllm-exp 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
  teardown; }
teardown
arm 12 14240000000 14000000000 13500000000
arm 13 14200000000 14000000000 13500000000
arm 17 13920000000 13500000000
log "--- sheet ---"
for s in 12 13 17; do log "[SHEET seqs$s] pool=${POOL[$s]:-?} free=${FREE[$s]:-?}MiB $(grep -a "seqs$s admission" "$R/audit.log" | grep -aoE '"conc": [0-9]+, "max_running": [0-9]+' | tr '\n' ';')"; done
grep -aE "BOOT OK|admission|RESULT|FAILED|SHEET|error-lines" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
