#!/usr/bin/env bash
# R164 (2026-09-03): Bug C ladder on the nvfp4 DFlash-graphs candidate (dflash-nvfp4-revival/PLAN-PROMOTION.md, G13).
#   L1 ledger  : ledger image (0130, VLLM_BUGC_LEDGER=1), SEQS=16, util 0.90 — per-descriptor capture allocation ledger
#                for BOTH graph managers; then one 131K needle (documents the post-capture OOM if it still happens).
#   L2 cap-s16 : graphs image, SEQS=16, util 0.90, --cudagraph-capture-sizes 10 20 40 80 (≤8 request shapes for target
#                AND drafter, zero-patch: cudagraph_utils.py:173-264 reads the shared compilation config) — boot facts,
#                needles 131K/220K, benchy c8/c16, decode_ss code 8/16.
#   L3 cap-s32 : same cap at SEQS=32 — boot facts, needles, benchy c8/c16/c32, decode_ss code 8/16/32.
#   L4 fp8-cap-s32 (control): fp8 daily shape, SEQS=32, util 0.92, same cap — benchy c8/c16/c32 vs R163's uncapped fp8-s32.
#   Boot facts per cell: KV pool, "Available KV cache memory", graph counts, "Actual usage" line (G = graph memory).
# Takes the GPU-exclusive flock; takes the daily down; restores it at the end.
# Unit: sudo systemd-run --unit=r164-bugc --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 bash /srv/qwen5090/r164-bugc.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r164-bugc; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R164 start (lock held) ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R164 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
NV_ENV="IMAGE=vllm-qwen38:v0280-nvfp4kv-revival-graphs KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912 UTIL=0.90"
NV_LEDGER_ENV="IMAGE=vllm-qwen38:v0280-nvfp4kv-revival-ledger KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912 UTIL=0.90"
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
CAP="--cudagraph-capture-sizes 10 20 40 80"

boot(){ # $1 tag, $2 arm env, $3 spec json, $4 extra env, $5 SEQS, $6 extra args
  local tag=$1 arm=$2 spec=$3 x=$4 seqs=$5 xa=$6 att rc
  for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $arm SEQS="$seqs" EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" EXTRA_ARGS="$xa" \
      bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
    sudo docker logs vllm-exp 2>&1 | grep -aE "Available KV cache memory|GPU KV cache size|Capturing CUDA graphs|Capturing dflash|Graph capturing finished|CUDA graph|Actual usage|cudagraph_capture_sizes|max_cudagraph_capture_size|OutOfMemoryError|memory allocation failed" | cut -c1-240 > "$R/bootfacts-$tag.txt"
    grep -aE "Actual usage|GPU KV cache size|Graph capturing finished" "$R/bootfacts-$tag.txt" | grep -a "TP0\|EngineCore" | cut -c1-200 | sed "s/^/[$tag facts] /" | tee -a "$R/audit.log"
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK att=$att $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-$tag.log" | tail -1)"; return 0; fi
    log "[$tag] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-140)"; teardown; [ $att = 1 ] && sleep 240
  done; return 1; }
needles(){ python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles] hits=$(grep -ac 'HIT ' "$R/needles-$1.out") miss=$(grep -ac MISS "$R/needles-$1.out") $(grep -a MISS "$R/needles-$1.out" | head -2 | cut -c1-120 | tr '\n' ' ')"; }
benchy(){ timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1.json" > "$R/benchy-$1.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$1.json" "benchy.$1" | tee -a "$R/audit.log"; }
dss(){ local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind.json" > "$R/decode-$tag-$kind.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind.out" | sed "s/^/[$tag decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }

log "taking the daily down"; teardown
# L1 ledger
if boot ledger-s16 "$NV_LEDGER_ENV" "$NV_SPEC" "$NV_X -e VLLM_BUGC_LEDGER=1" 16 ""; then
  sudo docker logs vllm-exp 2>&1 | grep -a BUGC_LEDGER > "$R/ledger-s16.txt"
  log "[ledger-s16] ledger lines=$(wc -l < "$R/ledger-s16.txt") $(grep -a 'BUGC_LEDGER summary' "$R/ledger-s16.txt" | grep -a TP0 | cut -c1-200 | tr '\n' ' ')"
  needles ledger-s16 "131000"; errlines ledger-s16
else sudo docker logs vllm-exp 2>&1 | grep -a BUGC_LEDGER > "$R/ledger-s16.txt" 2>/dev/null; log "[ledger-s16] boot failed; ledger lines=$(wc -l < "$R/ledger-s16.txt")"; fi
teardown
# L2 / L3 capped nvfp4
for s in 16 32; do
  if boot "cap-s$s" "$NV_ENV" "$NV_SPEC" "$NV_X" $s "$CAP"; then
    needles "cap-s$s" "131000 220000"
    [ $s = 16 ] && { benchy cap-s16-c8c16 "8 16" 2; dss cap-s16 code "8 16"; }
    [ $s = 32 ] && { benchy cap-s32-c8c16c32 "8 16 32" 2; dss cap-s32 code "8 16 32"; }
    errlines "cap-s$s"
  else log "[cap-s$s] skipped (boot failed twice)"; fi
  teardown
done
# L4 fp8 control with the cap
if boot fp8-cap-s32 "$FP8_ENV" "$FP8_SPEC" "$FP8_X" 32 "$CAP"; then
  benchy fp8-cap-s32-c8c16c32 "8 16 32" 2; dss fp8-cap-s32 code "8 16 32"; errlines fp8-cap-s32
else log "[fp8-cap-s32] skipped (boot failed twice)"; fi
finish DONE
