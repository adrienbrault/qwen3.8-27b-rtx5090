#!/usr/bin/env bash
# R163c (2026-09-03): isolate the nvfp4 candidate's single-stream jitter (R163 paired: benchy c1 runs 220/180/241 vs
# fp8 275/277/278 same hour; c8 parity). Cells on :8029, each = boot → benchy c1 ×5 → decode_ss c1 code ×3:
#   A nvfp4 as-is (tier on)        B nvfp4 tier OFF        C nvfp4 drafter EAGER (no DFLASH graphs)
#   D nvfp4 EPP=performance        E fp8 EPP=performance   (EPP restored to balance_performance at the end)
# Takes the GPU-exclusive flock; restores the daily at the end.
# Unit: sudo systemd-run --unit=r163c --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 bash /srv/qwen5090/r163c-c1-jitter.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r163c-c1-jitter; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock"; flock 9; }
U=http://127.0.0.1:8029; LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4; DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'; L2=/srv/qwen5090/eval-l2
EPP0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)
epp(){ for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo "$1" | sudo tee "$f" >/dev/null; done; log "EPP=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"; }
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 60; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ epp "$EPP0"; teardown; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R163c $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
NV="IMAGE=vllm-qwen38:v0280-nvfp4kv-revival-graphs KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 FIWS=536870912"
NV_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
NV_X="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
NV_X_EAGER="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0"
FP8="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456"; FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'; FP8_X="-e NCCL_P2P_LEVEL=SYS"
cell(){ # $1 tag, $2 arm env, $3 spec, $4 extra env, $5 NO_TIER
  local tag=$1 arm=$2 spec=$3 x=$4 nt=$5
  if ! env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $arm NO_TIER="$nt" EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$spec" EXTRA_ENV="$x" bash $LAUNCH > "$R/boot-$tag.log" 2>&1; then
    log "[$tag] BOOT FAILED: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-120)"; teardown; return 1; fi
  log "[$tag] BOOT OK $(grep -aoE 'KV pool: [0-9]+' "$R/boot-$tag.log" | tail -1) graphs=$(sudo docker logs vllm-exp 2>&1 | grep -ac 'Capturing dflash2') eager=$(sudo docker logs vllm-exp 2>&1 | grep -ac 'eagerly') gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' '/')MiB $(sudo docker logs vllm-exp 2>&1 | grep -aoE 'Actual usage is [0-9.]+ GiB for consumed memory \(weights \+ non-torch\), [0-9.]+ GiB for peak activation' | head -1)"
  timeout 1800 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency 1 --runs 5 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$tag.json" > "$R/benchy-$tag.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$tag.json" "benchy.$tag" | tee -a "$R/audit.log"
  python3 -c "import json;d=json.load(open('$R/benchy-$tag.json'));b=[x for x in d['benchmarks'] if x['concurrency']==1][0];print('[$tag c1 runs]',[round(v,1) for v in (b.get('tg_throughput') or b.get('tg_req_throughput'))['values']])" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 256 --runs 3 --kind code --out "$R/decode-$tag.json" > "$R/decode-$tag.out" 2>&1
  grep -a RESULT "$R/decode-$tag.out" | sed "s/^/[$tag decode_ss c1 code] /" | cut -c1-200 | tee -a "$R/audit.log"
  uptime | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  teardown; }
log "=== R163c start: EPP=$EPP0 ==="
sudo docker logs vllm-27b > "$R/daily-predown.log" 2>&1 || true; teardown
cell A-nvfp4-tier   "$NV" "$NV_SPEC" "$NV_X" 0
cell B-nvfp4-notier "$NV" "$NV_SPEC" "$NV_X" 1
cell C-nvfp4-eager  "$NV" "$NV_SPEC" "$NV_X_EAGER" 0
epp performance
cell D-nvfp4-eppperf "$NV" "$NV_SPEC" "$NV_X" 0
cell E-fp8-eppperf   "$FP8" "$FP8_SPEC" "$FP8_X" 0
finish DONE
