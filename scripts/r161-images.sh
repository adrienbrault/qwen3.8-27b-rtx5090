#!/usr/bin/env bash
# R161 (2026-09-03, user "take over daily, look into supporting more images"): what does raising
# --limit-mm-per-prompt cost, and what does a per-image pixel cap buy, on the daily shape?
#   Cells (each a fresh TP2 boot on :8029, daily shape: RedHat NVFP4, fp8 KV, DFlash2 ns9, tier ON on eval-l2):
#     A count4      = the daily itself (probe-smoke.json against :8020 + daily-predown.log boot facts)
#     B count16     MMLIMIT {"image":16}                   (boot facts + 12-image probe)
#     C count16px1M MMLIMIT {"image":16} + MMKW {"max_pixels":1048576}  (1024 tok/image cap; same probe)
#   Boot facts: KV pool, "Encoder cache ... budget ... profiled with N image items", memory profiling lines.
#   Probe: flan/probes/mm_probe.py (TTFT cold/warm/+1 image, decode t/s + acceptance with 12 images vs text).
#   Takes the daily DOWN (both GPUs) and restores it unchanged at the end via daily-restore-retry.sh.
# Unit: sudo systemd-run --unit=r161 --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 bash /srv/qwen5090/r161-images.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-03-r161-images; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
PROBE=/srv/qwen5090/probes/mm_probe.py
BASE=http://127.0.0.1:8029/v1

settle(){ for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " "); case "$u" in *[1-9][0-9][0-9][0-9]*) sleep 5;; *) break;; esac; done; log "gpus idle ($u MiB); settling ${1:-60}s"; sleep "${1:-60}"; }
teardown(){ # $1 = container
  sudo docker logs "$1" > "$R/engine-$1-$(date +%H%M%S).log" 2>&1 || true; sudo docker rm -f "$1" >/dev/null 2>&1 || true; }
finish(){ log "restoring daily"; settle 60; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R161 $1 ==="; }
trap 'log "### SIGTERM ###"; teardown vllm-exp; finish ABORTED; exit 4' TERM

boot(){ # $1 = cell label, rest = extra env assignments
  local cell=$1; shift
  log "--- cell $cell boot: $*"
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MODEL_DIR="$MODEL" \
    TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=0 FIWS=268435456 UTIL=0.92 MAXLEN=262144 POOL_MIN=560000 POOL_MAX=760000 \
    L2MNT=/srv/qwen5090/eval-l2 EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
    "$@" bash /srv/qwen5090/launch-daily-v0280.sh > "$R/boot-$cell.log" 2>&1
  local rc=$?
  sudo docker logs vllm-exp 2>&1 | grep -aE "GPU KV cache size|Encoder cache will be initialized|Memory profiling|Available KV cache memory|peak|non_torch|limit_mm_per_prompt|mm_processor_kwargs" | cut -c1-220 | tee -a "$R/audit.log" > "$R/bootfacts-$cell.txt"
  [ $rc -eq 0 ] && curl -sf -m 5 http://127.0.0.1:8029/health >/dev/null && { log "cell $cell UP"; return 0; }
  log "cell $cell BOOT FAILED rc=$rc: $(grep -aE FAILED "$R/boot-$cell.log" | tail -1 | cut -c1-140)"; return 1
}

log "=== R161 start: taking the daily down ==="
sudo docker logs vllm-27b > "$R/daily-predown.log" 2>&1 || true
sudo docker rm -f vllm-27b >/dev/null 2>&1 || true
settle 60

# A (count 4) = the daily itself: probe-smoke.json (3 images, live daily) + daily-predown.log carry its boot facts.
# B: count 16
if boot B MMLIMIT='{"image":16,"video":0}'; then
  python3 "$PROBE" "$BASE" "$R/img" 12 B "$R/probe-B.json" 2>&1 | tee -a "$R/audit.log"
fi
teardown vllm-exp
settle 60
# C: count 16 + 1 Mpx cap
if boot C MMLIMIT='{"image":16,"video":0}' MMKW='{"max_pixels":1048576}'; then
  python3 "$PROBE" "$BASE" "$R/img" 12 C "$R/probe-C.json" 2>&1 | tee -a "$R/audit.log"
fi
teardown vllm-exp
finish DONE
