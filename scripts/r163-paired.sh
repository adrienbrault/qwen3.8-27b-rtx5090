#!/usr/bin/env bash
# R163 paired battery (2026-09-03): fp8 daily shape vs nvfp4 candidate, SAME DAY, same boot order, same probes,
# on :8029 / vllm-exp — the promotion sheet's gates G1/G6-G10/G12 (dflash-nvfp4-revival/PLAN-PROMOTION.md).
# Per arm: 3 boots (pool band, G9) → on boot 3: needles 9K..262K ×2 (G1), warm-revisit 32K (G12), benchy c1/c8 ×3
# (G6/G7), decode_ss code/prose c8 + deep30k (G7), tool-eval 69×4 (G4) → reboot at SEQS=32: needles 262K ×2 (G1 at the
# wide layout) + benchy c8/16/32 + decode_ss code c8/16/32 (G8 admission). nvfp4 extras: drafter-KV=fp8 cell (benchy c1),
# ns7 / ns11 cells (benchy c1). Takes the daily down; restores
# it at the end. Takes the GPU-exclusive flock at start, so it queues behind a running campaign unit by itself.
# Unit: sudo systemd-run --unit=r163-paired --collect -p User=adrienbrault -p RuntimeMaxSec=60000 -p TimeoutStopSec=900 bash /srv/qwen5090/r163-paired.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r163-paired; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R163 paired $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

# arm definitions: common env + per-arm env
COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
NV_IMG=vllm-qwen38:v0280-nvfp4kv-revival-graphs
NV_ENV="IMAGE=$NV_IMG KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912"
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"

boot(){ # $1 tag, $2 arm env, $3 spec json, $4 extra env, $5 SEQS
  local tag=$1 arm=$2 spec=$3 x=$4 seqs=$5 att rc
  for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $arm SEQS="$seqs" EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" \
      bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK att=$att $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-$tag.log" | tail -1)"; return 0; fi
    log "[$tag] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-140)"; teardown; [ $att = 1 ] && sleep 240
  done; return 1; }
pool(){ grep -aoE 'KV pool: [0-9]+' "$R/boot-$1.log" | tail -1 | tr -dc 0-9; }
needles(){ # $1 tag, $2 depths
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles] hits=$(grep -ac 'HIT ' "$R/needles-$1.out") miss=$(grep -ac MISS "$R/needles-$1.out") $(grep -a MISS "$R/needles-$1.out" | head -2 | cut -c1-120 | tr '\n' ' ')"; }
benchy(){ # $1 tag, $2 concurrency list, $3 runs
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1.json" > "$R/benchy-$1.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$1.json" "benchy.$1" | tee -a "$R/audit.log"; }
dss(){ # $1 tag, $2 kind, $3 conc list, extra...
  local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind.json" > "$R/decode-$tag-$kind.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind.out" | sed "s/^/[$tag decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 namespace sets wiped (engine down; both arms start from a cold tier, no cross-dtype block reads)"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }

arm(){ # $1 name, $2 arm env, $3 spec, $4 extra env
  local A=$1 aenv=$2 spec=$3 x=$4 pools=""
  log "=== arm $A ==="; wipe_l2
  for b in 1 2 3; do boot "$A-b$b" "$aenv" "$spec" "$x" 8 || { log "[$A] boot $b failed twice — arm aborted"; return 1; }; pools="$pools $(pool "$A-b$b")"; [ $b -lt 3 ] && teardown; done
  log "[$A pool band] boots:$pools"
  needles "$A" "9000 20000 131000 220000 262000"
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-$A.log" 2>&1; tail -2 "$R/warm-revisit-$A.log" | cut -c1-200 | sed "s/^/[$A revisit] /" | tee -a "$R/audit.log"
  benchy "$A-c1c8" "1 8" 3
  dss "$A" code 8; dss "$A" prose 8
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 --out "$R/decode-$A-deep30k.json" > "$R/decode-$A-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$A-deep30k.out" | sed "s/^/[$A decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"
  log "[$A] tool-eval x4 start"
  ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$A.json" > "$R/tooleval-$A.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$A.json" "$A" 2>&1 | tee -a "$R/audit.log"
  errlines "$A"; teardown
  # wide layout: SEQS=32 admission ladder + 262K needles on that layout
  if boot "$A-s32" "$aenv" "$spec" "$x" 32; then
    needles "$A-s32" "262000"
    benchy "$A-s32-c8c16c32" "8 16 32" 2
    dss "$A-s32" code "8 16 32"
    curl -s -m 5 $U/metrics | grep -aE "^vllm:(num_requests_(running|waiting)|num_preemptions_total|spec_decode_num_(accepted|draft)_tokens_total)" | sed "s/^/[$A-s32] /" | cut -c1-160 | tee -a "$R/audit.log"
    errlines "$A-s32"
  else log "[$A-s32] ladder skipped (boot failed)"; fi
  teardown; return 0; }

# GPU-exclusive lock (see miniswe-full.sh): blocks while the candidate campaign runs, then runs; no log-watching.
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R163 paired start: fp8 daily shape then nvfp4 candidate, :8029, tier $L2 ==="
sudo docker logs vllm-27b > "$R/daily-predown.log" 2>&1 || true
teardown
arm fp8 "$FP8_ENV" "$FP8_SPEC" "$FP8_X"
arm nvfp4 "$NV_ENV" "$NV_SPEC" "$NV_X"
# nvfp4 extras (benchy c1 only): drafter KV in fp8 (target stays nvfp4), ns7, ns11
for cell in dfp8 ns7 ns11; do
  case $cell in
    dfp8) spec='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"fp8_e4m3"}';;
    ns7)  spec='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}';;
    ns11) spec='{"method":"dflash","model":"/draft","num_speculative_tokens":11,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}';;
  esac
  if boot "nvfp4-$cell" "$NV_ENV" "$spec" "$NV_X" 8; then needles "nvfp4-$cell" "131000"; benchy "nvfp4-$cell-c1" "1" 3; dss "nvfp4-$cell" code 8; errlines "nvfp4-$cell"; fi
  teardown
done
finish DONE
