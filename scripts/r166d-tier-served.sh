#!/usr/bin/env bash
# R166d (2026-09-03): G12 tier half, take two. R166c showed that with the daily tier config the FIRST touch of an
# evicted prefix is never served on EITHER KV dtype (fp8 and nvfp4 both: ext_hits 0, 2.7-3.5 GB read from the fs tier
# in the background) — the disk lookup is asynchronous (vllm/v1/kv_offload/tiering/async_lookup.py) and promotes into
# the 4 GiB CPU tier while the request is already being recomputed. So "evicted re-ask HIT" in R166c proved recompute,
# not tier-served correctness. This run re-asks the evicted needle 3x in a row (needle_depth.py --evict-reasks 3): the
# 2nd/3rd touch can be served from the CPU tier (ext_hits ≈ prompt tokens) and THAT pass is the exact-match check of
# nvfp4 blocks read back from the tier. Per arm: wipe eval-l2 → boot → needles 131K x2 --evict 12 --evict-reasks 3 →
# docker restart → same seed again (restart revisit + evicted re-asks) → teardown. Arm order F (fp8, method control),
# N (candidate, daily CPU tier 4 GiB), then N24 (candidate, CPUB 24 GiB) ONLY if arm N still had no tier-served pass.
# Pass per arm = every needle HIT (rc 0) AND tier_served_hits == tier_served >= 1 in at least one of the two probes.
# Queue-registered; restores the daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r166d-tier --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r166d-tier bash /srv/qwen5090/r166d-tier-served.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r166d-tier; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
source /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R166d tier-served gate start (lock held) ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-nvfp4-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
L2=/srv/qwen5090/eval-l2
CPUB24=25769803776
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R166d tier-served gate $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 namespace sets wiped (engine down; cold tier)"; }
FP8_COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 IMAGE=vllm-qwen38:v0280-nvfp4kv KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
boot(){ # $1 arm (F|N|N24)
  local rc
  case "$1" in
    F)   env -i PATH="$PATH" HOME="$HOME" USER="$USER" $FP8_COMMON EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" bash $LAUNCH > "$R/boot-$1.log" 2>&1; rc=$?;;
    N)   env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 bash $CAND > "$R/boot-$1.log" 2>&1; rc=$?;;
    N24) env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 DAILY_ALLOW_ENV=1 CPUB=$CPUB24 bash $CAND > "$R/boot-$1.log" 2>&1; rc=$?;;
  esac
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    log "[$1] BOOT OK $(grep -aoE 'KV pool: [0-9]+ tokens|Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$1.log" | tail -1) cpu_tier=$(sudo docker logs vllm-exp 2>&1 | grep -aoE 'cpu_bytes_to_use[^,}]*' | head -1)"; return 0; fi
  log "[$1] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$1.log" | tail -1 | cut -c1-160)"; return 1; }
nd(){ # $1 tag; seed fixed → identical prompts across invocations (restart revisit)
  local tag=$1; shift
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 --samples 2 --seed r166d "$@" \
    --evict 12 --evict-ctx 90000 --evict-reasks 3 --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1; local rc=$?
  grep -a "^\[" "$R/needles-$tag.out" | cut -c1-420 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  log "[$tag] $(grep -a SUMMARY "$R/needles-$tag.out" | cut -c1-260) rc=$rc"; return $rc; }
served(){ python3 -c "import json,sys; d=json.loads(open('$R/needles-$1.out').read().split('SUMMARY',1)[1].splitlines()[0]); print(d.get('tier_served') or 0, d.get('tier_served_hits') or 0)" 2>/dev/null || echo "0 0"; }
ARM_SERVED=0
arm(){ # $1 = F|N|N24
  local A=$1 s1 s2 rc1 rc2
  wipe_l2
  boot $A || { log "[$A] arm skipped"; teardown; return 1; }
  nd "$A-evict"; rc1=$?
  log "[$A] docker restart"; sudo docker restart vllm-exp >/dev/null 2>&1; sleep 20
  for i in $(seq 90); do curl -sf -m 5 $U/health >/dev/null && break; sleep 10; done
  curl -sf -m 5 $U/health >/dev/null || { log "[$A] FAILED: engine did not come back after docker restart"; teardown; return 1; }
  log "[$A] engine back; pool $(sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9)"
  nd "$A-restart"; rc2=$?
  s1=$(served "$A-evict"); s2=$(served "$A-restart")
  read -r sv1 sh1 <<<"$s1"; read -r sv2 sh2 <<<"$s2"
  ARM_SERVED=$((sv1 + sv2))
  if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ "$ARM_SERVED" -ge 1 ] && [ "$sh1" = "$sv1" ] && [ "$sh2" = "$sv2" ]; then log "[$A] PASS: all needles hit, tier-served passes $ARM_SERVED (evict $sv1/$sh1, restart $sv2/$sh2 served/hit)"
  elif [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]; then log "[$A] INCONCLUSIVE: all needles hit but NO tier-served pass (recompute only) — served evict=$sv1 restart=$sv2"
  else log "[$A] FAIL: needle miss (rc $rc1/$rc2), tier-served evict $sv1/$sh1 restart $sv2/$sh2"; fi
  sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$A engine error-lines] /" | tee -a "$R/audit.log"
  teardown; }
log "taking the daily down"; teardown
arm F
arm N
if [ "$ARM_SERVED" -eq 0 ]; then log "arm N had no tier-served pass at CPUB 4 GiB — running N24 (CPUB 24 GiB)"; arm N24; else log "N24 skipped (arm N had $ARM_SERVED tier-served passes)"; fi
finish DONE
