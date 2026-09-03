#!/usr/bin/env bash
# R166c (2026-09-03): promotion gate G12 for the nvfp4 candidate, done right — the disk tier must serve KV blocks written
# in nvfp4 format (a) in-session after GPU eviction and (b) across a container restart — PAIRED against the fp8 daily
# shape with the same probe. Why: R166b's restart revisit had a cold TTFT (7.98 s) while the connector log showed fs-tier
# reads in the same interval as the boot pre-warm; the old probe only read GPU prefix-cache counters, so hits could not be
# attributed. warm-revisit.py now reports external_prefix_cache_hits and the tier's chunk-hit/read-bytes deltas per send.
# G12 is a CORRECTNESS gate, so the pass criterion is an EXACT-MATCH needle through the tier path, not TTFT: every
# revisit that passed before R166c was a GPU prefix-cache hit — no test had ever checked that nvfp4 blocks read back
# from CPU/disk decode to the right secret. needle_depth.py --evict (R166c) re-asks the needle after a 12 x 90K flood
# evicts it from the GPU pool; the same seed after `docker restart` re-asks byte-identical prompts (restart revisit).
# Per arm (eval-l2 wiped first): boot → needles 131K x2 --evict 12 (cold HIT + evicted HIT, /metrics ext-hit deltas)
# → warm-revisit run 1 (salt S, --flood 12x90K: TTFT + counters) → `docker restart` → health → needles again, same seed
# (cold pass = restart revisit, must HIT with ext hits) → warm-revisit run 2 (salt S) → run 3 fresh salt (cold control).
# Pass = 4/4 needle hits per arm (2 evicted + 2 restart) with ext_hits > 0 on the tier-served passes; TTFT is diagnostic.
# Queue-registered; restores the daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r166c-tier --collect -p User=adrienbrault -p RuntimeMaxSec=60000 -p TimeoutStopSec=900 bash /srv/qwen5090/r166c-tier-gate.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r166c-tier; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
source /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R166c tier gate start (lock held) ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-nvfp4-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R166c tier gate $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 namespace sets wiped (engine down; cold tier)"; }
FP8_COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 IMAGE=vllm-qwen38:v0280-nvfp4kv KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
boot(){ # $1 arm (F|N)
  local rc
  if [ "$1" = F ]; then env -i PATH="$PATH" HOME="$HOME" USER="$USER" $FP8_COMMON EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" bash $LAUNCH > "$R/boot-$1.log" 2>&1; rc=$?
  else env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 bash $CAND > "$R/boot-$1.log" 2>&1; rc=$?; fi
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$1] BOOT OK $(grep -aoE 'KV pool: [0-9]+ tokens|Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$1.log" | tail -1)"; return 0; fi
  log "[$1] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$1.log" | tail -1 | cut -c1-160)"; return 1; }
nd(){ # $1 tag, rest = probe args (seed fixed → identical prompts across invocations)
  local tag=$1; shift
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 --samples 2 --seed r166c "$@" \
    --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1; local rc=$?
  grep -a "^\[" "$R/needles-$tag.out" | cut -c1-300 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  log "[$tag] $(grep -a SUMMARY "$R/needles-$tag.out" | cut -c1-200) rc=$rc"; return $rc; }
rv(){ # $1 tag, rest = probe args
  local tag=$1; shift
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 "$@" > "$R/rv-$tag.log" 2>&1
  grep -a RESULT "$R/rv-$tag.log" | cut -c1-600 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"; }
arm(){ # $1 = F|N
  local A=$1 S; S=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
  wipe_l2
  boot $A || { log "[$A] arm skipped"; teardown; return 1; }
  nd "$A-evict" --evict 12 --evict-ctx 90000                          # cold HIT + evicted re-ask HIT (tier, in-session)
  rv "$A-insession" --salt "$S" --flood 12 --flood-ctx 90000        # send1 cold, send2 after eviction = tier (in-session)
  sudo docker logs vllm-exp 2>&1 | grep -a "KV Transfer metrics" | tail -3 | sed "s/.*KV Transfer metrics: //" | cut -c1-400 > "$R/kvt-$A-pre-restart.txt"
  log "[$A] docker restart"; sudo docker restart vllm-exp >/dev/null 2>&1; sleep 20
  for i in $(seq 90); do curl -sf -m 5 $U/health >/dev/null && break; sleep 10; done
  curl -sf -m 5 $U/health >/dev/null || { log "[$A] FAILED: engine did not come back after docker restart"; teardown; return 1; }
  log "[$A] engine back; pool $(sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9)"
  nd "$A-restart"                                                    # same seed → restart revisit exact-match (tier)
  rv "$A-restart" --salt "$S"                                        # send1 = restart revisit (tier), send2 = GPU
  rv "$A-cold-control"                                               # fresh salt
  sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$A engine error-lines] /" | tee -a "$R/audit.log"
  teardown; }
log "taking the daily down"; teardown
arm F
arm N
finish DONE
