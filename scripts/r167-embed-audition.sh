#!/usr/bin/env bash
# R167 (2026-09-03, user "I want us to try the embeddings dram offload"): the input-embedding table in pinned host RAM.
# vLLM PR #53981 (ported as patches-v0290/0135, image …-revival-prs-embed) lets `--offload-backend uva --cpu-offload-gb 1
# --cpu-offload-params embed_tokens` reach Qwen3.8's untied BF16 table (248,320 x 5,120 = 2.37 GiB; 1.18 GiB per TP rank):
# the shard lives in pinned host memory behind a UVA view and the gather reads rows over PCIe (source review: NOTES18.md).
# Arms, all rc1 chain, :8029, TP2, SEQS 8, util-sized pools so the freed VRAM shows up in the boot log by itself:
#   A = nvfp4 route on …-prs (no offload)          — the control (R165c shape: util 0.88, XQA off, drafter graphs)
#   B = same on …-prs-embed + offload               — the cell: pool delta, fidelity vs A (expected ~identical), needles,
#                                                      decode c1/c8, prefill (131K cold TTFT = 8192-row gathers over PCIe)
#   C = fp8 daily shape on …-prs-embed + offload    — what the daily would gain (R165b: rc1 fp8 pool 619,076 without)
# Boot asserts for the offload arms (NOTES18 §5, fail closed): "Offloader set to UVAOffloader", "Total CPU offloaded
# parameters: 1.18", NO "matched no parameters" warning, the args present, UVA/pinning not disabled by env.
# --cpu-offload-gb 1: the target shard (1.18 GiB) crosses the budget first, so the drafter's own same-width table
# (replaced by the target's after load anyway) is skipped — only one pinned table per rank.
# Queue-registered; restores the daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r167-embed --collect -p User=adrienbrault -p RuntimeMaxSec=12000 -p TimeoutStopSec=900 bash /srv/qwen5090/r167-embed-audition.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r167-embed; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
PRS_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs
EMBED_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed
for i in $PRS_IMG $EMBED_IMG; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R167 embed-offload audition start (lock held): $PRS_IMG vs $EMBED_IMG; SKIP_C=${SKIP_C:-0} ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
FP8_RUN=/srv/qwen5090/results/2026-09-03-r165c-audition/run-v0280-redhat-fp8kv.jsonl
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R167 embed-offload audition $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
NV_ENV="KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912 UTIL=0.88"   # R165c: rc1 nvfp4 route boots at 0.88 (0.90 OOMs in pre-warm)
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER"}'  # no draft kv_cache_dtype (rc1 regression #3)
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
EMBED_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"

boot(){ # $1 tag, $2 image, $3 arm env, $4 spec, $5 extra env, $6 extra args ("" = none), $7 = 1 if offload asserts apply
  local tag=$1 img=$2 arm=$3 spec=$4 x=$5 xa=$6 chk=$7 rc BOOTLOG ARGS
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$img" $arm EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" EXTRA_ARGS="$xa" \
    bash $LAUNCH > "$R/boot-$tag.log" 2>&1; rc=$?
  BOOTLOG=$(sudo docker logs vllm-exp 2>&1); ARGS=$(sudo docker inspect vllm-exp --format '{{json .Args}} {{json .Config.Env}}' 2>/dev/null)
  echo "$BOOTLOG" | grep -aE "Offloader set to|CPU offloaded parameters|matched no parameters|Available KV cache memory|GPU KV cache size|Graph capturing finished|Capturing dflash|running the draft eagerly|decode_backend=" | cut -c1-220 > "$R/bootfacts-$tag.txt"
  grep -aE "Offloader set to|CPU offloaded parameters|matched no parameters|GPU KV cache size" "$R/bootfacts-$tag.txt" | grep -a "TP0\|EngineCore\|(Worker_TP0" | tail -5 | sed "s/^/[$tag facts] /" | tee -a "$R/audit.log"
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error|Exception' "$R/boot-$tag.log" | tail -1 | cut -c1-200); last py exception: $(echo "$BOOTLOG" | grep -aE '^[A-Za-z]*(Error|Exception):' | tail -1 | cut -c1-200)"; return 1; fi
  local pool free; pool=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9); free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  log "[$tag] BOOT OK pool=$pool min_free_after_prewarm=${free}MiB image=$img"
  if [ "$chk" = 1 ]; then  # NOTES18 §5 — fail closed
    local bad=0
    [ "$(echo "$BOOTLOG" | grep -ac 'Offloader set to UVAOffloader')" -ge 1 ] || { log "[$tag] ASSERT FAILED: UVAOffloader not selected"; bad=1; }
    [ "$(echo "$BOOTLOG" | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || { log "[$tag] ASSERT FAILED: target embed shard not offloaded (no 'Total CPU offloaded parameters: 1.18')"; bad=1; }
    [ "$(echo "$BOOTLOG" | grep -ac 'matched no parameters')" -eq 0 ] || { log "[$tag] ASSERT FAILED: selector matched no parameters (UVA unavailable / hook inactive)"; bad=1; }
    [ "$(echo "$ARGS" | grep -ac -- '--offload-backend uva')" -ge 1 ] && [ "$(echo "$ARGS" | grep -ac -- '--cpu-offload-params embed_tokens')" -ge 1 ] || { log "[$tag] ASSERT FAILED: offload args missing from the container"; bad=1; }
    [ "$(echo "$ARGS" | grep -ac 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1\|VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1')" -eq 0 ] || { log "[$tag] ASSERT FAILED: UVA/pinning disabled by env"; bad=1; }
    [ "$(echo "$BOOTLOG" | grep -aci 'pin_memory\|cudaHostAlloc\|cudaHostRegister' | head -1)" -eq 0 ] || log "[$tag] note: pin/host-alloc mentions in the log: $(echo "$BOOTLOG" | grep -ai 'pin_memory\|cudaHostAlloc\|cudaHostRegister' | grep -ai 'error\|fail' | head -1 | cut -c1-160)"
    [ $bad = 0 ] || { log "[$tag] offload asserts failed — arm NOT measured"; return 1; }
    log "[$tag] offload asserts OK: $(echo "$BOOTLOG" | grep -a 'Total CPU offloaded parameters' | tail -1 | sed 's/.*INFO[^]]*] //' | cut -c1-80)"
  else
    [ "$(echo "$BOOTLOG" | grep -ac 'CPU offloaded parameters')" -eq 0 ] || log "[$tag] WARNING: control arm shows offload lines"
  fi
  return 0; }
needles(){ # $1 tag, $2 depths, rest = extra args
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --warm "${@:3}" --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles $2] $(grep -a SUMMARY "$R/needles-$1.out" | cut -c1-120) prefill tok/s per row: $(python3 -c "import json,sys;print(' '.join(f\"{r['depth']//1000}K:{r['prompt_tokens']/r['cold_s']:.0f}\" for r in map(json.loads,open('$R/needles-$1.jsonl')) if 'cold_s' in r and r.get('prompt_tokens')))")"; }
benchy(){ # $1 tag, $2 conc, $3 runs
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1-c$2.json" > "$R/benchy-$1-c$2.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_runs.py "$R/benchy-$1-c$2.json" 2>&1 | sed "s/^/[$1 benchy c$2] /" | cut -c1-200 | tee -a "$R/audit.log"; }
dss(){ local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind-c$conc.json" > "$R/decode-$tag-$kind-c$conc.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind-c$conc.out" | sed "s/^/[$tag decode_ss $kind c$conc] /" | cut -c1-230 | tee -a "$R/audit.log"; }
deep30k(){ python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 --out "$R/decode-$1-deep30k.json" > "$R/decode-$1-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$1-deep30k.out" | sed "s/^/[$1 decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"; }
fidelity(){ # $1 tag
  python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-$1.jsonl" --conc 1 > "$R/fidelity-run-$1.log" 2>&1 || log "[$1] ruler FAILED rc=$? ($R/fidelity-run-$1.log)"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$R/run-$1.jsonl" 2>&1 | tee "$R/fidelity-$1-vs-fp8ref.txt" | cut -c1-200 | sed "s/^/[$1 fidelity vs fp8ref] /" | tee -a "$R/audit.log"; }
metrics_line(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:(num_preemptions_total|spec_decode_num_(accepted|draft)_tokens_total)" | sed "s/^/[$1] /" | cut -c1-160 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
measure(){ # $1 tag
  local T=$1
  needles $T "9000 131000"
  fidelity $T
  benchy $T 1 3; benchy $T 8 2
  dss $T code 1; dss $T code 8; dss $T prose 8
  deep30k $T
  metrics_line $T; errlines $T; }

log "taking the daily down"; teardown
# A: control
if boot A "$PRS_IMG" "$NV_ENV" "$NV_SPEC" "$NV_X" "" 0; then measure A; fi; teardown
# B: the cell
if boot B "$EMBED_IMG" "$NV_ENV" "$NV_SPEC" "$NV_X" "$EMBED_ARGS" 1; then measure B
  [ -f "$R/run-A.jsonl" ] && python3 /srv/qwen5090/probes/fidelity.py compare --ref "$R/run-A.jsonl" "$R/run-B.jsonl" 2>&1 | tee "$R/fidelity-B-vs-A.txt" | cut -c1-200 | sed "s/^/[B fidelity vs A (direct)] /" | tee -a "$R/audit.log"
fi; teardown
# C: fp8 daily shape + offload
if [ "${SKIP_C:-0}" != 1 ]; then
  if boot C "$EMBED_IMG" "$FP8_ENV" "$FP8_SPEC" "$FP8_X" "$EMBED_ARGS" 1; then measure C
    [ -f "$FP8_RUN" ] && python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FP8_RUN" "$R/run-C.jsonl" 2>&1 | tee "$R/fidelity-C-vs-v0280fp8.txt" | cut -c1-200 | sed "s/^/[C fidelity vs v0.28 fp8 daily shape] /" | tee -a "$R/audit.log"
  fi; teardown
fi
finish DONE
