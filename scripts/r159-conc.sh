#!/usr/bin/env bash
# R159 — high-concurrency ladder for the DAILY config (user 2026-09-02: "code/prefill numbers at concurrency
# 16 ? 32 ? 64 ? prose and code"). The daily admits 8 streams (SEQS=8), so this boots the identical shape
# (RedHat NVFP4 weights, fp8 KV, DFlash2 ns9 syvai drafter, TP2, util 0.92, FIWS 256M, MNBT 8192) on :8029
# with SEQS=64 and NO tier (cold prefill, no cross-boot namespace hits). c8 is re-measured in the same boot
# as the anchor against the daily's known numbers (acceptance is boot-level state, R115).
#   prefill + generation: llama-benchy pp2048/tg256 at c8/16/32/64 (3 runs, tokenizer-counted)
#   steady-state decode:  probes/decode_ss.py code + prose at c8/16/32/64 (512 tokens, 3 runs)
# Ends with the daily restore (daily-restore-retry.sh handles the heavy-TP2 warmup transient).
# Attempt 1 (util 0.92, pool 626,367) OOM'd at the FIRST c64 step: sample_tokens tried to allocate 392,167,424 B
# (= 64 seqs x 10 spec positions x vocab x fp32 logits) with 233 MB free on GPU 1 — the boot profiler does not
# budget the spec-decode sampler at max_num_seqs, so SEQS=64 needs headroom: util 0.90 for the ladder (pool is
# irrelevant to these throughput numbers at pp2048/short prompts).
#   sudo systemd-run --unit=r159-conc --collect -p User=adrienbrault -p RuntimeMaxSec=3600 bash /srv/qwen5090/r159-conc.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-02-r159-conc${RUN_SUFFIX:-}; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 45; }
metrics(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$1] /" | tee -a "$R/audit.log"; }

log "=== R159 start: daily shape @ SEQS=64 on :8029, benchy c8/16/32/64 then decode_ss code+prose ==="
settle
booted=0
for att in 1 2; do
  if env PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MODEL_DIR="$MODEL" TP=2 KVD_OVERRIDE=fp8_e4m3 NO_TIER=1 \
       FIWS=268435456 UTIL=${UTIL_LADDER:-0.90} MAXLEN=262144 POOL_MIN=1 POOL_MAX=9999999 SEQS=64 MNBT=8192 \
       EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' \
       EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" bash $LAUNCH > "$R/boot.log" 2>&1; then
    grep -aE "KV pool" "$R/boot.log" | tail -1 | sed "s/^/[boot] POOL /" | tee -a "$R/audit.log"
    sudo docker logs vllm-exp 2>&1 | grep -aE "max_num_seqs|Capturing|cudagraph" | head -3 | cut -c1-160 | sed "s/^/[boot] /" | tee -a "$R/audit.log"
    booted=1; break
  fi
  log "[boot] attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot.log" | tail -1 | cut -c1-150)"
  timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 45
done
if [ $booted = 1 ]; then
  timeout 2400 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency 8 16 32 64 --runs 3 --no-cache --latency-mode api --format json $EB \
    --save-result "$R/benchy.json" > "$R/benchy.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy.json" "benchy.seqs64.t0.6" | tee -a "$R/audit.log"
  metrics after-benchy
  for kind in code prose; do
    python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 16 32 64 --tokens 512 --runs 3 --kind $kind \
      --out "$R/decode-$kind.json" > "$R/decode-$kind.out" 2>&1
    grep -a RESULT "$R/decode-$kind.out" | sed "s/^/[decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"
    metrics after-$kind
  done
  sudo docker logs vllm-exp > "$R/engine.log" 2>&1 || true
  grep -aciE "error|traceback" "$R/engine.log" | sed "s/^/[engine] error-lines: /" | tee -a "$R/audit.log"
else log "[boot] BOOT FAILED twice"; fi
settle
bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R159 DONE ==="
