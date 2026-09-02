#!/usr/bin/env bash
# R157c — can the XQA verify route close the nvfp4 DFlash2 verify-path gap?  R157b decomposed the −28% c1:
#   spec-OFF kernel nvfp4/XQA-off = fp8 (108.9 vs 105.5)  -> plain decode is free
#   fp8 with draft_tp=1: 318.8 -> 285.1 (−11%)              -> unsharded drafter (Bug A dodge) = ~11 pts
#   remainder ≈ −19% = the q_len=1+7 verify batches on FA2-over-nvfp4 (XQA off).
# Arm xv: revival shape with XQA ON + VLLM_SM12X_XQA_VERIFY=1 (0112: verify batches through XQA), at MNBT 4096
# (< M*=4929 so Bug B cannot trigger with the 256 MiB XQA workspace present), FIWS 512M (TP2 XQA scratch).
# Needles are the correctness gate for both the XQA-verify route and the MNBT-4096 Bug-B dodge.
#   sudo systemd-run --unit=r157c-xqaverify --collect -p User=adrienbrault -p RuntimeMaxSec=5400 bash /srv/qwen5090/r157c-xqaverify.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
while systemctl is-active -q r157b-levers.service; do sleep 15; done; sleep 60
R=/srv/qwen5090/results/2026-09-02-r157-nvfp4-shapes; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
M=/srv/qwen5090/models; MODEL=$M/qwen3.8-27b-redhat-nvfp4; DRAFT=$M/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }
log "=== R157c start (xv = nvfp4 DFlash2 ns7 draft_tp1, XQA ON + XQA_VERIFY=1, MNBT 4096) ==="
settle; ok=0
for att in 1 2; do
  if IMAGE=vllm-qwen38:v0280-nvfp4kv-revival MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 \
     KVD_OVERRIDE=nvfp4 SEQS=8 MNBT=4096 FIWS=536870912 MAXLEN=262144 NO_TIER=1 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 \
     EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
     SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}' \
     EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_XQA_VERIFY=1" bash $LAUNCH > "$R/boot-xv.log" 2>&1; then ok=1; break; fi
  log "[xv] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-xv.log" | tail -1 | cut -c1-150)"; timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
done
if [ "$ok" = 1 ]; then
  grep -aE "KV pool" "$R/boot-xv.log" | tail -1 | sed "s/^/[xv] POOL /" | tee -a "$R/audit.log"
  sudo docker logs vllm-exp 2>&1 | grep -aE "xqa_nvfp4_verify|XQA.?VERIFY|decode_backend" | head -3 | sed "s/^/[xv route] /" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 9000 20000 131000 --samples 2 --out "$R/needles-xv.jsonl" > "$R/needles-xv.out" 2>&1
  grep -a SUMMARY "$R/needles-xv.out" | head -1 | sed "s/^/[xv needles] /" | cut -c1-160 | tee -a "$R/audit.log" || echo "[xv needles] FAILED" | tee -a "$R/audit.log"
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency 1 8 --runs 3 \
    --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-xv-nat.json" > "$R/benchy-xv-nat.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-xv-nat.json" "benchy.xv.nat.t0.6" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 --tokens 512 --runs 3 --kind code --out "$R/decode-xv-code.json" > "$R/decode-xv-code.out" 2>&1
  grep -a RESULT "$R/decode-xv-code.out" | sed "s/^/[xv decode_ss code] /" | cut -c1-230 | tee -a "$R/audit.log"
  sudo docker logs vllm-exp > "$R/engine-xv.log" 2>&1 || true
else log "[xv] BOOT FAILED"; fi
settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R157c DONE ==="
