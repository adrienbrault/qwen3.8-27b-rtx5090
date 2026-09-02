#!/usr/bin/env bash
# R157b — decompose the nvfp4-KV single-stream gap on RedHat into its two suspected components.
#   k4f: nvfp4 KV, spec OFF, XQA OFF -> forward-pass rate on the FA2-over-nvfp4 path (what DFlash2 verify
#        batches pay; yardstick = kern.h fp8 spec-OFF c1 105.5 / c8 ~665).
#   f1 : the fp8 DFlash2 daily config with draft_tensor_parallel_size=1 -> the cost of an UNSHARDED drafter
#        on a shape where both modes work = upper bound on what fixing R155 Bug A would return to the
#        nvfp4 revival shape (yardstick = benchy.h c1 318.8 nat / decode_ss code c8 1,212).
#   sudo systemd-run --unit=r157b-levers --collect -p User=adrienbrault -p RuntimeMaxSec=7200 bash /srv/qwen5090/r157b-levers.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-02-r157-nvfp4-shapes; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
M=/srv/qwen5090/models; MODEL=$M/qwen3.8-27b-redhat-nvfp4; DRAFT=$M/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }
boot(){ local tag=$1; shift; settle
  for att in 1 2; do
    if env "$@" MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=1 \
       PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 bash $LAUNCH > "$R/boot-$tag.log" 2>&1; then
      grep -aE "KV pool" "$R/boot-$tag.log" | tail -1 | sed "s/^/[$tag] POOL /" | tee -a "$R/audit.log"; return 0; fi
    log "[$tag] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-150)"
    timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
  done; return 1
}
log "=== R157b start (k4f = nvfp4 spec-OFF XQA-off kernel; f1 = fp8 DFlash2 ns9 draft_tp=1) ==="
if boot k4f UTIL=0.90 KVD_OVERRIDE=nvfp4 NOSPEC=1 FIWS=536870912 MNBT=8192 ALLOW_NO_XQA=1 \
     EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0"; then
  timeout 3600 python3 /srv/qwen5090/probes/decode_bench.py --url $U --model qwen3.8-27b \
    --corpus /srv/qwen5090/r156-corpus.jsonl --kinds code,prose --conc 1 8 --tokens 512 --reps 3 --label kern-k4f \
    --out "$R/kern-k4f.json" > "$R/kern-k4f.out" 2>&1
  grep -a "^RESULT" "$R/kern-k4f.out" | sed "s/^/[kern.k4f] /" | tee -a "$R/audit.log"
else log "[k4f] BOOT FAILED"; fi
if boot f1 UTIL=0.92 KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 MNBT=8192 EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
     SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":1}' \
     EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS"; then
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency 1 8 --runs 3 --no-cache --latency-mode api --format json $EB \
    --save-result "$R/benchy-f1-nat.json" > "$R/benchy-f1-nat.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-f1-nat.json" "benchy.f1.nat.t0.6" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 --tokens 512 --runs 3 --kind code \
    --out "$R/decode-f1-code.json" > "$R/decode-f1-code.out" 2>&1
  grep -a RESULT "$R/decode-f1-code.out" | sed "s/^/[f1 decode_ss code] /" | cut -c1-230 | tee -a "$R/audit.log"
else log "[f1] BOOT FAILED"; fi
settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R157b DONE ==="
