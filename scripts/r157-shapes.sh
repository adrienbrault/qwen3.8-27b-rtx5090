#!/usr/bin/env bash
# R157 — the two nvfp4-KV TP2 shapes on the RedHat daily checkpoint, measured with the daily's instruments
# (user 2026-09-02: "The nvfp4 kv sounds great. Find ways to make it work? what are the speeds with mtp?
# Compared to dflash fp8?").  Arms, both MODEL=RedHat, TP=2, tier ON, util 0.90:
#   m = MTP ns4 + nvfp4 KV + XQA (the R142c "tp2mtp" capacity shape; daily image)
#   d = DFlash2 ns7 + nvfp4 KV, draft_tp=1 (Bug A dodge), MNBT 8192 + XQA OFF (Bug B dodge, R155g2a promote
#       candidate; revival image 0116+0117+0118b+0119), FIWS 512M
# Yardstick = the fp8 DFlash2 daily on RedHat (benchy.h T=0.6: c1 318.8 / c8 656; decode_ss code c8 1,212 /
# prose 925; deep30k 157; pool 654,491).  Heavy-TP2 cadence: settle + 2 attempts per boot, 3 boots total.
#   sudo systemd-run --unit=r157-shapes --collect -p User=adrienbrault -p RuntimeMaxSec=10800 bash /srv/qwen5090/r157-shapes.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-02-r157-nvfp4-shapes; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
M=/srv/qwen5090/models
MODEL=$M/qwen3.8-27b-redhat-nvfp4
DRAFT=$M/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }

battery(){ local tag=$1
  grep -aE "KV pool" "$R/boot-$tag.log" | tail -1 | sed "s/^/[$tag] POOL /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 9000 20000 131000 --samples 2 \
    --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1
  grep -a SUMMARY "$R/needles-$tag.out" | head -1 | sed "s/^/[$tag needles] /" | cut -c1-160 | tee -a "$R/audit.log" || echo "[$tag needles] FAILED" | tee -a "$R/audit.log"
  for mode in exact nat; do
    ex=""; [ $mode = exact ] && ex="--exact-tg"
    timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
      --concurrency 1 8 --runs 3 $ex --no-cache --latency-mode api --format json $EB \
      --save-result "$R/benchy-$tag-$mode.json" > "$R/benchy-$tag-$mode.out" 2>&1
    python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$tag-$mode.json" "benchy.$tag.$mode.t0.6" | tee -a "$R/audit.log"
  done
  for kind in code prose; do
    python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 --tokens 512 --runs 3 --kind $kind \
      --out "$R/decode-$tag-$kind.json" > "$R/decode-$tag-$kind.out" 2>&1
    grep -a RESULT "$R/decode-$tag-$kind.out" | sed "s/^/[$tag decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"
  done
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 \
    --out "$R/decode-$tag-deep30k.json" > "$R/decode-$tag-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$tag-deep30k.out" | sed "s/^/[$tag decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"
  curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  sudo docker logs vllm-exp > "$R/engine-$tag.log" 2>&1 || true
}

boot(){ local tag=$1; shift
  settle
  for att in 1 2; do
    if env "$@" MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 \
       PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 bash $LAUNCH > "$R/boot-$tag.log" 2>&1; then return 0; fi
    log "[$tag] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-150)"
    timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
  done; return 1
}

log "=== R157 start (RedHat; m = MTP ns4 nvfp4 XQA; d = DFlash2 ns7 nvfp4 draft_tp1 MNBT8192 XQA-off) ==="
if boot m EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS"; then battery m; else log "[m] BOOT FAILED"; fi
if boot d IMAGE=vllm-qwen38:v0280-nvfp4kv-revival KVD_OVERRIDE=nvfp4 SEQS=8 MNBT=8192 FIWS=536870912 ALLOW_NO_XQA=1 \
     EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
     SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}' \
     EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0"; then battery d; else log "[d] BOOT FAILED"; fi
settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R157 DONE ==="
