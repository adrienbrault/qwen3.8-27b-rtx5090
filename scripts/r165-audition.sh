#!/usr/bin/env bash
# R165 audition (2026-09-03): the vLLM v0.29.0rc1 image chain (patches-v0290, codex rebase + PR ports) on :8029.
# Order = cheapest decisive facts first (helpers inherited from r164c-ws.sh / r163-paired.sh):
#   A  fp8 daily shape on the base tag (SEQS 8, util 0.92): boot facts, needles 9K/131K/262K, warm-revisit 32K,
#      benchy c1/c8 x3, decode_ss code/prose c8, fidelity ruler (vs the FP8 reference; non-negotiable gate), tool-eval x4.
#   B  nvfp4 candidate shape on the -revival-prs tag (XQA off, drafter graphs, 0131 default 1 MiB), util 0.90:
#      SEQS 16 -> boot facts (G, pool: #53306/#53955 now reserve graph memory before sizing, so pool should DROP by ~G),
#      needles 131K/262K, benchy c1/c8/c16; SEQS 32 -> boot facts, needles 262K, benchy c8/c32 (the Bug C question);
#      SEQS 16 with VLLM_SM12X_POOLED_INT_WS_MIB=0 -> boot facts only (upstream accounting alone vs with 0131).
#   C  0132 (PR #53543 masked NVFP4 XQA) A/B on the nvfp4 shape, SEQS 8: XQA ON + XQA_VERIFY=1 -> needles 131K,
#      benchy c1 x3, decode_ss code c1/c8; then + VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM=1 -> benchy c1 x3.
#      Baseline = cell B's XQA-off c1 (same image, same boot shape apart from SEQS) and R164c ws-s16 c1 244.9 on v0280.
#   D  0133 (PR #54181 GDN packed decode BV) A/B on the fp8 daily shape, -revival-prs tag: VLLM_SM12X_GDN_PACKED_BV=16
#      -> needles 131K, benchy c1 x3, decode_ss code c1/c8. Baseline = cell A (unset = upstream launch config).
# Refuses to start (before touching the daily) if the images are missing. Takes the flock; takes the daily down; restores it.
# Unit: sudo systemd-run --unit=r165-audition --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 bash /srv/qwen5090/r165-audition.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r165-audition; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
BASE_IMG=vllm-qwen38:v0290rc1-nvfp4kv
PRS_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs
for i in $BASE_IMG $PRS_IMG; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing (build first: build-v0290rc1.sh)"; exit 3; }; done
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R165 audition start (lock held): $BASE_IMG / $PRS_IMG ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R165 audition $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192"
FP8_ENV="IMAGE=$BASE_IMG KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_PRS_ENV="IMAGE=$PRS_IMG KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
NV_ENV="IMAGE=$PRS_IMG KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912 UTIL=0.90"
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
NV_XQA_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=1 -e VLLM_SM12X_XQA_VERIFY=1 -e VLLM_SM12X_DFLASH_GRAPHS=1"

boot(){ # $1 tag, $2 arm env, $3 spec json, $4 extra env, $5 SEQS
  local tag=$1 arm=$2 spec=$3 x=$4 seqs=$5 att rc
  for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $arm SEQS="$seqs" EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" \
      bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
    sudo docker logs vllm-exp 2>&1 | grep -aE "Available KV cache memory|GPU KV cache size|Capturing CUDA graphs|Capturing dflash|Graph capturing finished|CUDA graph|Actual usage|cudagraph_capture_sizes|max_cudagraph_capture_size|OutOfMemoryError|memory allocation failed|overlay ACTIVE|decode_backend|use_fa2_nvfp4|SM12x|XQA|packed decode|Model runner|ModelRunner|is_neox|DFlash2|dflash2|KV connector|OffloadingConnector|deprecat|WARNING.*(flag|arg)" | cut -c1-240 > "$R/bootfacts-$tag.txt"
    grep -aE "Actual usage|GPU KV cache size|Graph capturing finished|Capturing CUDA graphs.*[0-9]+/[0-9]+" "$R/bootfacts-$tag.txt" | grep -a "TP0\|EngineCore" | tail -4 | cut -c1-200 | sed "s/^/[$tag facts] /" | tee -a "$R/audit.log"
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK att=$att $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-$tag.log" | tail -1) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')MiB"; return 0; fi
    log "[$tag] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-160)"; teardown; [ $att = 1 ] && sleep 120
  done; return 1; }
needles(){ python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles] hits=$(grep -ac 'HIT ' "$R/needles-$1.out") miss=$(grep -ac MISS "$R/needles-$1.out") $(grep -a MISS "$R/needles-$1.out" | head -2 | cut -c1-120 | tr '\n' ' ')"; }
benchy(){ timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1.json" > "$R/benchy-$1.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$1.json" "benchy.$1" | tee -a "$R/audit.log"; }
dss(){ local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind.json" > "$R/decode-$tag-$kind.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind.out" | sed "s/^/[$tag decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 namespace sets wiped"; }

sudo docker logs vllm-27b > "$R/daily-predown.log" 2>&1 || true
teardown; wipe_l2

# ---- A: fp8 daily shape on the base tag
if boot A-fp8 "$FP8_ENV" "$FP8_SPEC" "$FP8_X" 8; then
  needles A-fp8 "9000 131000 262000"
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-A.log" 2>&1; tail -2 "$R/warm-revisit-A.log" | cut -c1-200 | sed "s/^/[A-fp8 revisit] /" | tee -a "$R/audit.log"
  benchy A-fp8-c1c8 "1 8" 3
  dss A-fp8 code 8; dss A-fp8 prose 8
  log "[A-fp8] fidelity ruler start"
  python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-v0290rc1-fp8kv.jsonl" > "$R/fidelity-run.log" 2>&1
  REF_RH=$(ls -t /srv/qwen5090/results/*/run-*redhat*.jsonl 2>/dev/null | head -1)
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" ${REF_RH:+"$REF_RH"} "$R/run-v0290rc1-fp8kv.jsonl" 2>&1 | tee "$R/fidelity-compare.txt" | tail -8 | cut -c1-200 | sed "s/^/[A-fp8 fidelity] /" | tee -a "$R/audit.log"
  log "[A-fp8] tool-eval x4 start"
  ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-A.json" > "$R/tooleval-A.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-A.json" A-fp8 2>&1 | tee -a "$R/audit.log"
  curl -s -m 5 $U/metrics | grep -aE "^vllm:(num_preemptions_total|spec_decode_num_(accepted|draft)_tokens_total)" | sed "s/^/[A-fp8] /" | cut -c1-160 | tee -a "$R/audit.log"
  errlines A-fp8
else log "[A-fp8] cell skipped (boot failed twice) — the rc1 image does not serve the daily shape; read boot-A-fp8.log + bootfacts"; fi
teardown

# ---- B: nvfp4 candidate shape, Bug C on upstream accounting
if boot B-nv-s16 "$NV_ENV" "$NV_SPEC" "$NV_X" 16; then
  needles B-nv-s16 "131000 262000"; benchy B-nv-s16-c1c8c16 "1 8 16" 2; dss B-nv-s16 code 8; errlines B-nv-s16
else log "[B-nv-s16] skipped"; fi
teardown
if boot B-nv-s32 "$NV_ENV" "$NV_SPEC" "$NV_X" 32; then
  needles B-nv-s32 "262000"; benchy B-nv-s32-c8c32 "8 32" 2; curl -s -m 5 $U/metrics | grep -aE "^vllm:num_preemptions_total" | sed "s/^/[B-nv-s32] /" | tee -a "$R/audit.log"; errlines B-nv-s32
else log "[B-nv-s32] skipped"; fi
teardown
if boot B-nv-s16-ws0 "$NV_ENV" "$NV_SPEC" "$NV_X -e VLLM_SM12X_POOLED_INT_WS_MIB=0" 16; then benchy B-nv-s16-ws0-c1 "1" 2; errlines B-nv-s16-ws0; else log "[B-nv-s16-ws0] skipped"; fi
teardown

# ---- C: 0132 masked NVFP4 XQA A/B (XQA on + verify)
if boot C-nv-xqa "$NV_ENV" "$NV_SPEC" "$NV_XQA_X" 8; then
  grep -aE "decode_backend|XQA|xqa" "$R/bootfacts-C-nv-xqa.txt" | head -4 | cut -c1-200 | sed "s/^/[C-nv-xqa route] /" | tee -a "$R/audit.log"
  needles C-nv-xqa "131000"; benchy C-nv-xqa-c1 "1" 3; dss C-nv-xqa code 1; dss C-nv-xqa-c8 code 8; errlines C-nv-xqa
else log "[C-nv-xqa] skipped"; fi
teardown
if boot C-nv-xqa-iso "$NV_ENV" "$NV_SPEC" "$NV_XQA_X -e VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM=1" 8; then
  needles C-nv-xqa-iso "131000"; benchy C-nv-xqa-iso-c1 "1" 3; errlines C-nv-xqa-iso
else log "[C-nv-xqa-iso] skipped"; fi
teardown

# ---- D: 0133 GDN packed decode BV=16 forced, fp8 daily shape on the prs tag
if boot D-fp8-bv16 "$FP8_PRS_ENV" "$FP8_SPEC" "$FP8_X -e VLLM_SM12X_GDN_PACKED_BV=16" 8; then
  needles D-fp8-bv16 "131000"; benchy D-fp8-bv16-c1 "1" 3; dss D-fp8-bv16 code 1; dss D-fp8-bv16-c8 code 8; errlines D-fp8-bv16
else log "[D-fp8-bv16] skipped"; fi
finish DONE
