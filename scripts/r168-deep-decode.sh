#!/usr/bin/env bash
# R168 (2026-09-03, user "focus on .29 and all related improvements, including nvfp4"): isolate the rc1 nvfp4 deep-context
# decode regression (R167: 30K-context c1 decode 29 tok/s on the rc1 chain vs 144 on the v0.28 candidate; short context
# normal; fp8 on rc1 fine at 135; both chains on the FlashInfer FA2 fallback with XQA off; ≈85 ms/step vs ≈15). torch is
# 2.13.0+cu130 on both, FlashInfer is the one library delta (0.6.16.post3 vs 0.6.18), the vLLM side is rc1 + the rebased
# 0101 chain. One boot per cell, every cell measures the same thing: deep30k (decode_ss --ctx 30000 c1, 2 runs), a short
# c1 control, then a torch-profiler capture of ~32 decode steps at 30K (kernel table = probes/prof_summary.py).
#   V28   v0.28 candidate image, spec ON                        reference 144 tok/s
#   V28F  v0.28 candidate + FlashInfer 0.6.18 (fiswap)           DECISIVE: drops → the library; stays → vLLM side
#   RC1   rc1 …-prs-embed (no offload), spec ON                  reference 29 tok/s
#   RC1F  rc1 …-prs-embed + FlashInfer 0.6.16.post3 (fiswap)     mirror of V28F
#   RC1N  rc1, NOSPEC=1                                          splits target attention from the drafter (0105/0109 rebase)
#   V28N  v0.28, NOSPEC=1                                        control for RC1N
#   RC1X  rc1, masked NVFP4 XQA on (0132, VLLM_SM12X_NVFP4_XQA=1) is the rc1 XQA path deep-context fast? (R165c: −24% short)
#   RC1P  rc1 fp8 daily shape                                    cross-dtype control (135)
# nvfp4 cells: SEQS 8, XQA off unless stated, MNBT 8192, FIWS 512M, drafter graphs; v0.28 pool pinned like the candidate
# (13.8 GB/GPU), rc1 util 0.88 (R165c). Draft kv_cache_dtype dropped on rc1 (regression #3). Traces land in
# $R/prof/<cell>/ (mounted at /prof). Cells whose image is missing are skipped and logged. Queue-registered; restores the
# daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r168-deep --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r168-deep bash /srv/qwen5090/r168-deep-decode.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r168-deep-decode; mkdir -p "$R/prof"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
V28_IMG=vllm-qwen38:v0280-nvfp4kv-revival-graphs-ws
V28F_IMG=$V28_IMG-fi0618
RC1_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed
RC1F_IMG=$RC1_IMG-fi0616
CELLS=${CELLS:-"V28 V28F RC1 RC1F RC1N V28N RC1P RC1X"}
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R168 deep-decode isolation start (lock held): cells $CELLS ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-45}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R168 deep-decode isolation $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
NV_ENV="KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912"
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
NV_XQA_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=1 -e VLLM_SM12X_DFLASH_GRAPHS=1"
V28_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
RC1_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER"}'
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
PROF="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof"

boot(){ # $1 tag, $2 image, $3 arm env, $4 spec ("" = NOSPEC), $5 extra env, $6 extra args
  local tag=$1 img=$2 arm=$3 spec=$4 x=$5 xa=$6 rc BOOTLOG nospec=0
  [ -n "$spec" ] || nospec=1
  sudo docker image inspect "$img" >/dev/null 2>&1 || { log "[$tag] SKIPPED: image $img missing"; return 1; }
  mkdir -p "$R/prof/$tag"
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$img" $arm NOSPEC=$nospec EXTRA_MOUNT="-v $DRAFT:/draft:ro -v $R/prof/$tag:/prof" SPEC_JSON="$spec" EXTRA_ENV="$x" EXTRA_ARGS="$xa $PROF" \
    bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
  BOOTLOG=$(sudo docker logs vllm-exp 2>&1)
  echo "$BOOTLOG" | grep -aE "KV cache layout|Available KV cache memory|GPU KV cache size|Capturing dflash|running the draft eagerly|decode_backend=|use_fa2_nvfp4|FA2|flashinfer version|Profiling enabled|torch profiler" | cut -c1-220 > "$R/bootfacts-$tag.txt"
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error|Exception' "$R/boot-$tag.log" | tail -1 | cut -c1-200); last py exception: $(echo "$BOOTLOG" | grep -aE '^[A-Za-z]*(Error|Exception):' | tail -1 | cut -c1-200)"; return 1; fi
  local pool fi; pool=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9)
  fi=$(sudo docker exec vllm-exp python3 -c 'import flashinfer, vllm; print(vllm.__version__, "flashinfer", flashinfer.__version__)' 2>/dev/null)
  log "[$tag] BOOT OK pool=$pool image=$img ($fi) nospec=$nospec route: $(grep -a 'decode_backend=' "$R/bootfacts-$tag.txt" | tail -1 | sed 's/.*decode_backend=/decode_backend=/' | cut -c1-80)"
  return 0; }
dss(){ # $1 tag, $2 label, rest = decode_ss args
  local tag=$1 lab=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --kind prose "$@" --out "$R/decode-$tag-$lab.json" > "$R/decode-$tag-$lab.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$lab.out" | sed "s/^/[$tag $lab] /" | cut -c1-230 | tee -a "$R/audit.log"; }
profile30k(){ # $1 tag — ~32 decode steps at 30K under the torch profiler, then the kernel table
  local tag=$1
  curl -sf -m 30 -X POST $U/start_profile >/dev/null || { log "[$tag] start_profile FAILED"; return 1; }
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --kind prose --ctx 30000 --tokens 32 --runs 1 --out "$R/decode-$tag-prof.json" > "$R/decode-$tag-prof.out" 2>&1
  curl -sf -m 600 -X POST $U/stop_profile >/dev/null || log "[$tag] stop_profile FAILED"
  sleep 20; sudo chown -R "$USER" "$R/prof/$tag" 2>/dev/null
  local n; n=$(ls "$R/prof/$tag" 2>/dev/null | wc -l); log "[$tag] profile: $n trace file(s) in $R/prof/$tag ($(du -sh "$R/prof/$tag" 2>/dev/null | cut -f1))"
  [ "$n" -gt 0 ] && python3 /srv/qwen5090/probes/prof_summary.py "$R/prof/$tag" --top 25 --steps 32 --json "$R/prof-$tag.json" > "$R/prof-$tag.txt" 2>&1 \
    && { grep -aE "^== " "$R/prof-$tag.txt" | sed "s/^/[$tag prof] /" | cut -c1-230 | tee -a "$R/audit.log"; sed -n '/ALL RANKS merged/,$p' "$R/prof-$tag.txt" | head -14 | tail -12 | sed "s/^/[$tag prof top] /" | cut -c1-200 | tee -a "$R/audit.log"; }; }
metrics_line(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$1] /" | cut -c1-160 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
measure(){ local T=$1
  dss $T deep30k --ctx 30000 --tokens 512 --runs 2
  dss $T short --tokens 256 --runs 2
  profile30k $T
  metrics_line $T; errlines $T; }
cell(){ # $1 tag ... boot args
  local T=$1; shift
  if boot "$T" "$@"; then measure "$T"; fi; teardown; }

log "taking the daily down"; teardown
for C in $CELLS; do case $C in
  V28)  cell V28  "$V28_IMG"  "$NV_ENV UTIL=0.90" "$V28_SPEC" "$NV_X" "--kv-cache-memory-bytes 13800000000";;
  V28F) cell V28F "$V28F_IMG" "$NV_ENV UTIL=0.90" "$V28_SPEC" "$NV_X" "--kv-cache-memory-bytes 13800000000";;
  V28N) cell V28N "$V28_IMG"  "$NV_ENV UTIL=0.90" ""          "$NV_X" "--kv-cache-memory-bytes 13800000000";;
  RC1)  cell RC1  "$RC1_IMG"  "$NV_ENV UTIL=0.88" "$RC1_SPEC" "$NV_X" "";;
  RC1F) cell RC1F "$RC1F_IMG" "$NV_ENV UTIL=0.88" "$RC1_SPEC" "$NV_X" "";;
  RC1N) cell RC1N "$RC1_IMG"  "$NV_ENV UTIL=0.88" ""          "$NV_X" "";;
  RC1X) cell RC1X "$RC1_IMG"  "$NV_ENV UTIL=0.88" "$RC1_SPEC" "$NV_XQA_X" "";;
  RC1P) cell RC1P "$RC1_IMG"  "$FP8_ENV"          "$FP8_SPEC" "$FP8_X" "";;
  *) log "unknown cell $C";; esac; done
log "SHEET (deep30k tok/s per cell):"; grep -a "deep30k\] RESULT" "$R/audit.log" | cut -c1-200 | tee -a "$R/audit.log" >/dev/null
finish DONE
