#!/usr/bin/env bash
# R165c (2026-09-03): the nvfp4 + XQA cells of the R165 audition, re-run after finding rc1 regression #3: every
# nvfp4 cell of r165b failed at rc1's new graph-memory profiling with "KV cache layout has not been resolved yet".
# Cause (source-verified): vllm/v1/worker/gpu/spec_decode/dflash/utils.py copies cache_config
# (replace(..., cache_dtype=spec.kv_cache_dtype)) when the speculative config sets kv_cache_dtype; the layout
# resolved by the engine core (#51718 refactor) is recorded on the worker's ORIGINAL cache_config only
# (worker_base.set_kv_cache_layout, gpu_worker.initialize_from_config), so the drafter's FlashInfer impl holds an
# unresolved copy and raises in forward. The fp8 cells never set the field, hence booted. Our "kv_cache_dtype":"nvfp4"
# equals the target's --kv-cache-dtype, so the field is dropped here (the draft inherits the same dtype on the
# shared object); the mixed-dtype arm (r163 dfp8) needs an upstream fix and is out of scope.
# Cells: R = v0.28 image, RedHat, fp8 KV, util 0.90: the fidelity ruler's missing same-checkpoint baseline (the daily at
# util 0.92 has no headroom: the ruler OOM-killed it at --conc 1 at 12:33 UTC, container auto-restarted in 50 s).
# B/C as r165b (nvfp4 SEQS 16/32/ws0, XQA on/iso) at NV_UTIL (default 0.88; the 0.90 pass OOMed, see NV_UTIL). boot() now quotes the last Python exception line on failure.
# Takes the flock; takes the daily down; restores it.
# Unit: sudo systemd-run --unit=r165c-audition --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 bash /srv/qwen5090/r165c-audition.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r165c-audition; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
BASE_IMG=vllm-qwen38:v0290rc1-nvfp4kv
V028_IMG=vllm-qwen38:v0280-nvfp4kv
PRS_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs
for i in $BASE_IMG $PRS_IMG $V028_IMG; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing (build first: build-v0290rc1.sh)"; exit 3; }; done
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R165c audition start (lock held): $BASE_IMG / $PRS_IMG; NV_UTIL=${NV_UTIL:-0.88} SKIP_R=${SKIP_R:-0} ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R165c audition $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192"
FP8_ENV="IMAGE=$PRS_IMG KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_PRS_ENV="IMAGE=$PRS_IMG KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
NV_UTIL=${NV_UTIL:-0.88}  # 0.90 = v0.28 candidate value; on rc1 the nvfp4 route boots at 0.90 (pool 905,438, layout fix works) then OOMs
                          # in the launcher pre-warm (pp8192 c8): 368 MiB aux-hidden-state cat in the drafter propose + 242 MiB, 32 MiB free.
                          # v0.28 same shape survives with 3 MiB reported free (r164c ws-s16). Second pass runs at 0.88 (NV_UTIL=0.88 SKIP_R=1).
NV_ENV="IMAGE=$PRS_IMG KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912 UTIL=$NV_UTIL"
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER"}'  # no draft kv_cache_dtype: see header (rc1 regression #3)
V028_ENV="IMAGE=$V028_IMG KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.90"
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
NV_XQA_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=1 -e VLLM_SM12X_XQA_VERIFY=1 -e VLLM_SM12X_DFLASH_GRAPHS=1"

boot(){ # $1 tag, $2 arm env, $3 spec json, $4 extra env, $5 SEQS
  local tag=$1 arm=$2 spec=$3 x=$4 seqs=$5 att rc
  for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $arm SEQS="$seqs" EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" \
      bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
    sudo docker logs vllm-exp 2>&1 | grep -aE "KV cache layout|Available KV cache memory|GPU KV cache size|Capturing CUDA graphs|Capturing dflash|Graph capturing finished|CUDA graph|Actual usage|cudagraph_capture_sizes|max_cudagraph_capture_size|OutOfMemoryError|memory allocation failed|overlay ACTIVE|decode_backend|use_fa2_nvfp4|SM12x|XQA|packed decode|Model runner|ModelRunner|is_neox|DFlash2|dflash2|KV connector|OffloadingConnector|deprecat|WARNING.*(flag|arg)" | cut -c1-240 > "$R/bootfacts-$tag.txt"
    grep -aE "Actual usage|GPU KV cache size|Graph capturing finished|Capturing CUDA graphs.*[0-9]+/[0-9]+" "$R/bootfacts-$tag.txt" | grep -a "TP0\|EngineCore" | tail -4 | cut -c1-200 | sed "s/^/[$tag facts] /" | tee -a "$R/audit.log"
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK att=$att $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-$tag.log" | tail -1) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')MiB"; return 0; fi
    log "[$tag] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-160)"
    log "[$tag] last exception: $(sudo docker logs vllm-exp 2>&1 | grep -aE '(ValueError|AttributeError|RuntimeError|AssertionError|NotImplementedError|TypeError|KeyError|OutOfMemoryError): ' | tail -1 | sed 's/.*multiproc_executor.py:[0-9]*\] //' | cut -c1-200)"; teardown; [ $att = 1 ] && sleep 120
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

SKIP_R=${SKIP_R:-0}
# ---- R: the ruler's same-checkpoint baseline on the v0.28 image (RedHat, fp8 KV), util 0.90 (never on the daily at 0.92)
if [ "$SKIP_R" = 1 ]; then log "[R-v028-fp8] skipped (SKIP_R=1, done in the first pass)"
elif boot R-v028-fp8 "$V028_ENV" "$FP8_SPEC" "$FP8_X" 8; then
  log "[R-v028-fp8] fidelity ruler start (--conc 1)"
  python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-v0280-redhat-fp8kv.jsonl" --conc 1 > "$R/fidelity-ref-run.log" 2>&1 || log "[R-v028-fp8] ruler FAILED rc=$? ($R/fidelity-ref-run.log)"
  RC1=/srv/qwen5090/results/2026-09-03-r165b-audition/run-v0290rc1-fp8kv.jsonl
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$R/run-v0280-redhat-fp8kv.jsonl" "$RC1" 2>&1 | tee "$R/fidelity-compare-vs-fp8ref.txt" | cut -c1-200 | sed "s/^/[R fidelity vs fp8ref] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$R/run-v0280-redhat-fp8kv.jsonl" "$RC1" 2>&1 | tee "$R/fidelity-compare-rc1-vs-v028.txt" | cut -c1-200 | sed "s/^/[R fidelity rc1 vs v0.28] /" | tee -a "$R/audit.log"
  errlines R-v028-fp8
else log "[R-v028-fp8] skipped"; fi
teardown

# ---- B: nvfp4 candidate shape, Bug C on upstream accounting
if boot B-nv-s16 "$NV_ENV" "$NV_SPEC" "$NV_X" 16; then
  needles B-nv-s16 "131000 262000"; python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-B.log" 2>&1; tail -1 "$R/warm-revisit-B.log" | cut -c1-200 | sed "s/^/[B-nv-s16 revisit] /" | tee -a "$R/audit.log"
  benchy B-nv-s16-c1c8c16 "1 8 16" 2; dss B-nv-s16 code 8; errlines B-nv-s16
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

finish DONE
