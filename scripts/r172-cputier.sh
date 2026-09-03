#!/usr/bin/env bash
# R172 (2026-09-04): CPU-tier sizing for the disk tier — the r169 finding: the fs tier only serves a prompt whose blocks ALL fit
# the CPU (primary) tier, because every fs hit is promoted through it (tiering/manager.py _initiate_promotion → primary
# prepare_write → None = MISS). CPUB=4 GiB is 252 blocks of 17 MB: a 131K prompt is 187 nvfp4 blocks (fits, served) but 331
# fp8 blocks (79 promotions fail, served 0 — r169 arm F, and the v0.28 fp8 daily has never been seen to serve). A 262K fp8
# prompt needs ~11.3 GB of CPU tier; nvfp4 ~6.4 GB. R153's "CPU tier size is irrelevant" was measured on v0.28 (async
# lookup never serves the first touch) and at ≤12G with the fp8 100K prompt = 252 blocks exactly — reopened here.
# Arms (CPUB=16 GiB each, RAM-gated: MemAvailable must exceed CPUB + 8 GiB; 46 GB available with an engine up on 2026-09-04):
#   N16 = nvfp4 candidate (launch-daily-v0290-candidate.sh EXP=1, DAILY_ALLOW_ENV=1 so CPUB passes), split-KV as r169 decided
#   F16 = rc2 fp8 daily shape + embed offload (r169 arm F)
#   D16 = v0.28 fp8 daily image (the current daily; SKIP_D16=1 skips)
# Per arm: wipe eval-l2, boot, needles 131K + 220K x2 with --evict 12x90K --evict-reasks 3 (in-session tier path), offload
# metrics snapshot (promotion/allocation failures, chunk queries/hits, fs read bytes), teardown, FRESH boot over the existing
# tier, same seed again (restart revisit), metrics again. Served = any pass with ext hits > 0 at that depth.
# Unit: sudo systemd-run --unit=r172-cputier --collect -p User=adrienbrault -p RuntimeMaxSec=36000 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r172-cputier bash /srv/qwen5090/r172-cputier.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r172-cputier; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
RC2_IMG=${RC2_IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}
V28_IMG=${V28_IMG:-vllm-qwen38:v0280-nvfp4kv}
CPUB=${CPUB:-17179869184}   # 16 GiB
sudo docker image inspect "$RC2_IMG" >/dev/null 2>&1 || { log "ABORT: image $RC2_IMG missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R172 CPU-tier sizing start (lock held): CPUB=$CPUB rc2=$RC2_IMG v28=$V28_IMG SKIP_D16=${SKIP_D16:-0} ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
L2=/srv/qwen5090/eval-l2
R169=/srv/qwen5090/results/2026-09-03-r169-rc2
split_from_r169(){ grep -aoE "SPLIT_KV auto-decided from r168b: [01]" "$R169/audit.log" 2>/dev/null | tail -1 | grep -oE "[01]$" || echo 1; }
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R172 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 wiped (engine down; cold tier): $(df -h "$L2" | tail -1 | awk '{print $3" used of "$2}')"; }
ram_ok(){ local avail_kb need_kb; avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo); need_kb=$(( CPUB/1024 + 8*1024*1024 ))
  log "[$1] host RAM: MemAvailable=$((avail_kb/1024/1024)) GiB, need CPUB+8=$((need_kb/1024/1024)) GiB, Shmem=$(awk '/^Shmem:/{print int($2/1024)}' /proc/meminfo) MiB"
  [ "$avail_kb" -ge "$need_kb" ]; }
COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 CPUB=$CPUB"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
EMBED_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
pool(){ sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9; }
cpu_blocks(){ sudo docker logs vllm-exp 2>&1 | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1; }
boot_N16(){ local rc SPLIT; SPLIT=$(split_from_r169)
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$SPLIT DAILY_ALLOW_ENV=1 CPUB=$CPUB bash $CAND > "$R/boot-N16.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[N16] BOOT OK split_kv=$SPLIT pool=$(pool) $(cpu_blocks) $(grep -aoE 'min free VRAM [0-9]+ MiB' "$R/boot-N16.log" | tail -1)"; return 0; fi
  log "[N16] BOOT FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-N16.log" | tail -1 | cut -c1-200)"; return 1; }
boot_F16(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$RC2_IMG" $FP8_ENV EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="$FP8_X" EXTRA_ARGS="$EMBED_ARGS" bash $LAUNCH > "$R/boot-F16.log" 2>&1; rc=$?
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[F16] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-F16.log" | tail -1 | cut -c1-200)"; return 1; fi
  [ "$(sudo docker logs vllm-exp 2>&1 | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || { log "[F16] ASSERT FAILED: embed shard not offloaded"; return 1; }
  log "[F16] BOOT OK pool=$(pool) $(cpu_blocks) min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; return 0; }
boot_D16(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$V28_IMG" $FP8_ENV EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="$FP8_X" bash $LAUNCH > "$R/boot-D16.log" 2>&1; rc=$?
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[D16] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-D16.log" | tail -1 | cut -c1-200)"; return 1; fi
  log "[D16] BOOT OK pool=$(pool) $(cpu_blocks) (v0.28 fp8 daily image)"; return 0; }
# CPUB assert: the connector JSON on the container must carry the requested cpu_bytes_to_use
cpub_assert(){ local got; got=$(sudo docker inspect vllm-exp --format '{{join .Args " "}}' | grep -oE '"cpu_bytes_to_use":[0-9]+' | grep -oE '[0-9]+$')
  [ "$got" = "$CPUB" ] && log "[$1] CPUB asserted on the container: $got" || { log "[$1] CPUB ASSERT FAILED: container has '${got:-none}', wanted $CPUB — arm NOT measured"; return 1; }; }
nd_tier(){ local tag=$1; shift
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 220000 --samples 2 --seed r172 "$@" --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1; local rc=$?
  grep -a "^\[" "$R/needles-$tag.out" | cut -c1-300 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  log "[$tag] $(grep -a SUMMARY "$R/needles-$tag.out" | cut -c1-220) rc=$rc"; }
offload_metrics(){ curl -s -m 10 $U/metrics | grep -a "kv_offload" > "$R/metrics-$1.txt"
  log "[$1 offload] $(grep -aE 'promotion_allocation_failures|kv_offload_allocation_failure|tiering_chunk_(queries|hits)_total|tiering_read_bytes' "$R/metrics-$1.txt" | grep -v '^#' | sed -E 's/vllm:kv_offload_//; s/\{[^}]*\}//' | tr '\n' ' ' | cut -c1-300)"; }
served_sheet(){ # per depth: hits/total and whether any pass at that depth had ext hits > 0, across the given .out files
  python3 - "$@" <<'PY' 2>&1 | sed "s/^/[$1 SHEET] /" | tee -a "$R/audit.log"
import re,sys
files=sys.argv[2:]; tag=sys.argv[1]
by={}
for f in files:
    for line in open(f,errors="replace"):
        m=re.match(r"\[(HIT |MISS)\]\s+depth=\s*(\d+)",line)
        if not m: continue
        d=int(m.group(2)); hit=m.group(1).startswith("HIT")
        exts=[int(x) for x in re.findall(r"ext=(\d+)/",line)]
        e=by.setdefault(d,{"hit":0,"tot":0,"served":0})
        e["tot"]+=1; e["hit"]+=hit; e["served"]+= 1 if any(x>0 for x in exts) else 0
for d in sorted(by):
    e=by[d]; print(f"depth {d}: hits {e['hit']}/{e['tot']} served-passes {e['served']}/{e['tot']} -> {'SERVED' if e['served'] else 'NOT SERVED'}{' (MISS)' if e['hit']<e['tot'] else ''}")
PY
}
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
arm(){ local A=$1
  wipe_l2; ram_ok "$A" || { log "[$A] SKIPPED: not enough host RAM for CPUB"; return; }
  boot_$A || { teardown; return; }
  cpub_assert "$A" || { teardown; return; }
  nd_tier "$A-evict" --evict 12 --evict-ctx 90000 --evict-reasks 3 --evict-reask-gap 15
  offload_metrics "$A-evict"; errlines "$A-evict"
  log "[$A] restart revisit: teardown + fresh boot over the existing tier ($(df -h "$L2" | tail -1 | awk '{print $3}') on the tier)"; teardown
  boot_$A || { log "[$A] restart boot FAILED"; return; }
  nd_tier "$A-restart" --evict-reasks 3 --evict-reask-gap 15
  offload_metrics "$A-restart"; errlines "$A-restart"
  served_sheet "$A" "$R/needles-$A-evict.out" "$R/needles-$A-restart.out"
  teardown; }
log "taking the daily down"; teardown
arm N16
arm F16
[ "${SKIP_D16:-0}" = 1 ] || arm D16
log "SHEET:"; grep -aE "BOOT OK|SHEET\]|offload\]|error-lines|SKIPPED|ASSERT" "$R/audit.log" | cut -c1-220 > "$R/sheet.txt"; wc -l "$R/sheet.txt" | tee -a "$R/audit.log"
finish DONE
