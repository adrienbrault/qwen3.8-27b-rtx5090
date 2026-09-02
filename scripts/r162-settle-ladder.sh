#!/usr/bin/env bash
# R162 (2026-09-03): how long must the GPUs sit idle after a TP2 teardown before the next TP2 boot succeeds?
# Background: 2026-09-02 boots within 45 s of a teardown died in kernel_warmup (Worker CUDA "invalid argument")
# and hung; 300 s settles booted first try. Every restart cycle since pays 300 s. The launcher now fails fast on
# that signature, so a ladder is cheap. Cells: settle 60, 120, 180 s (each: teardown → idle → sleep S → boot the
# daily shape on :8029 with HEALTH_TRIES=45). Ends by launching the SWE-bench campaign resume (its own engine),
# which restores the daily when it finishes. Results: results/2026-09-03-r162-settle/audit.log.
# Unit: sudo systemd-run --unit=r162 --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=600 bash /srv/qwen5090/r162-settle-ladder.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-03-r162-settle; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
idle(){ for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " "); case "$u" in *[1-9][0-9][0-9][0-9]*) sleep 2;; *) break;; esac; done; echo "$u"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; }
finish(){ log "chaining into the SWE-bench campaign resume (START=240); it restores the daily at its end"
  sudo systemd-run --unit=miniswe --collect -p User=adrienbrault -p RuntimeMaxSec=100000 -p TimeoutStopSec=900 -E START=240 bash /srv/qwen5090/miniswe-full.sh 500 2>&1 | tee -a "$R/audit.log"
  log "=== R162 $1 ==="; }
trap 'log "### SIGTERM ###"; teardown; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED" | tee -a "$R/audit.log"; exit 4' TERM
log "=== R162 start ==="
for S in 60 120 180; do
  teardown; t0=$(date +%s); u=$(idle); log "cell settle=$S: gpus idle ($u MiB) $(( $(date +%s)-t0 )) s after teardown; sleeping $S"
  sleep "$S"; t1=$(date +%s)
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MODEL_DIR="$MODEL" \
    TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=0 FIWS=268435456 UTIL=0.92 MAXLEN=262144 POOL_MIN=560000 POOL_MAX=760000 \
    L2MNT=/srv/qwen5090/eval-l2 EXTRA_MOUNT="-v $DRAFT:/draft:ro" HEALTH_TRIES=45 \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
    bash /srv/qwen5090/launch-daily-v0280.sh > "$R/boot-settle$S.log" 2>&1
  rc=$?; dt=$(( $(date +%s)-t1 ))
  if [ $rc -eq 0 ] && curl -sf -m 5 http://127.0.0.1:8029/health >/dev/null; then log "cell settle=$S: BOOT OK in ${dt}s"
  else log "cell settle=$S: BOOT FAILED rc=$rc after ${dt}s: $(grep -aE 'FAILED' "$R/boot-settle$S.log" | tail -1 | cut -c1-140)"; fi
done
teardown
finish DONE
