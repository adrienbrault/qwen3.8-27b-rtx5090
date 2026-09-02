#!/usr/bin/env bash
# R158b — re-admit FULL CUDA graphs for the DFlash drafter on the nvfp4 route (patch 0129, env
# VLLM_SM12X_DFLASH_GRAPHS=1, image revival-graphs). R158 profile: the whole nvfp4-vs-fp8 single-stream
# gap at equal ns7/draft_tp1 is GPU idle (busy 17.6 vs 16.8 ms/step, span 26.2 vs 22.6) because 0116
# forces the drafter eager ("does not support full CUDA graphs; running the draft eagerly").
#   ng  = nvfp4 DFlash2 ns7 draft_tp1, XQA off, MNBT 8192, graphs ON  -> speed (benchy c1/c8) + needles 9K/20K/131K/250K
#   ng2 = same with draft_tp=2 (Bug A shape)                          -> needles 9K/20K + acceptance + c1 (fresh Bug A data)
# Yardsticks (R158, same ns7/draft_tp1): n (eager drafter) c1 212.4 / c8 649.6; f (fp8 XQA) c1 249.1 / c8 679.7.
#   sudo systemd-run --unit=r158b-graphs --collect -p User=adrienbrault -p RuntimeMaxSec=7200 bash /srv/qwen5090/r158b-graphs.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-02-r158-nvfp4-profile; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
M=/srv/qwen5090/models
MODEL=$M/qwen3.8-27b-redhat-nvfp4
DRAFT=$M/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
IMG=vllm-qwen38:v0280-nvfp4kv-revival-graphs
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }

boot(){ local tag=$1; shift
  settle
  for att in 1 2; do
    if env "$@" IMAGE=$IMG MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 NO_TIER=1 \
       KVD_OVERRIDE=nvfp4 FIWS=536870912 ALLOW_NO_XQA=1 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 \
       EXTRA_MOUNT="-v $DRAFT:/draft:ro" bash $LAUNCH > "$R/boot-$tag.log" 2>&1; then
      grep -aE "KV pool" "$R/boot-$tag.log" | tail -1 | sed "s/^/[$tag] POOL /" | tee -a "$R/audit.log"
      sudo docker logs vllm-exp 2>&1 | grep -aE "DFLASH_GRAPHS|running the draft eagerly|Capturing dflash2|0129" | head -4 | cut -c1-200 | sed "s/^/[$tag graphs] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-150)"
    timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
  done; return 1
}
needles(){ local tag=$1; shift
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths "$@" --samples 2 \
    --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1
  grep -a SUMMARY "$R/needles-$tag.out" | head -1 | sed "s/^/[$tag needles] /" | cut -c1-160 | tee -a "$R/audit.log" || echo "[$tag needles] FAILED" | tee -a "$R/audit.log"
}
benchy(){ local tag=$1 conc=$2
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency $conc --runs 3 --no-cache --latency-mode api --format json $EB \
    --save-result "$R/benchy-$tag-nat.json" > "$R/benchy-$tag-nat.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$tag-nat.json" "benchy.$tag.nat.t0.6" | tee -a "$R/audit.log"
}
metrics(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$1] /" | tee -a "$R/audit.log"; }

SPEC1='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
SPEC2='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
ENVG="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
log "=== R158b start: ng (drafter graphs, draft_tp1) -> ng2 (draft_tp2, Bug A probe) ==="

if boot ng SPEC_JSON="$SPEC1" EXTRA_ENV="$ENVG"; then
  needles ng 9000 20000 131000 250000
  benchy ng "1 8"
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 --tokens 512 --runs 3 --kind code \
    --out "$R/decode-ng-code.json" > "$R/decode-ng-code.out" 2>&1
  grep -a RESULT "$R/decode-ng-code.out" | sed "s/^/[ng decode_ss code] /" | cut -c1-230 | tee -a "$R/audit.log"
  metrics ng
  sudo docker logs vllm-exp > "$R/engine-ng.log" 2>&1 || true
else log "[ng] BOOT FAILED"; fi

if boot ng2 SPEC_JSON="$SPEC2" EXTRA_ENV="$ENVG"; then
  needles ng2 9000 20000
  metrics ng2
  benchy ng2 1
  metrics ng2
  sudo docker logs vllm-exp > "$R/engine-ng2.log" 2>&1 || true
else log "[ng2] BOOT FAILED"; fi

settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R158b DONE ==="
