#!/usr/bin/env bash
# R158c — the nvfp4 PROMOTION CANDIDATE at the daily contract, one arm (advisor: cadence budget, and the
# promotion-relevant number is the daily-contract shape, not ns tuning):
#   RedHat TP2, nvfp4 KV, XQA off (Bug B dodge), MNBT 8192, FIWS 512M, DFlash2 ns9 draft_tp=2, drafter
#   graphs (0129, image revival-graphs), tier ON, util 0.90 (R157 tier-on value; daily fp8 runs 0.92/256M).
# Battery = the promotion sheet: needles 9K/20K/131K/220K (Bug B is untested on the new graph layout —
# nothing near max-len has run with drafter graphs; 250K was HTTP 400), warm-revisit (tier + prefix cache
# with draft_tp=2), benchy c1/c8 natural, decode_ss code/prose c8 + deep30k (does the FA2-nvfp4 verify
# win survive depth?), tool-eval x4. Yardstick = fp8 daily (R156f/R157): c1 318.8 / c8 656, decode_ss code
# 1212 / prose 925 / deep30k 157, tool-eval 90.8+-0.5, needles 9/9, warm-revisit 7.49->0.45 s.
#   sudo systemd-run --unit=r158c-candidate --collect -p User=adrienbrault -p RuntimeMaxSec=10800 bash /srv/qwen5090/r158c-candidate.sh
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
T=cand
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }
SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
ENVG="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1"
log "=== R158c start: candidate = nvfp4 + DFlash2 ns9 draft_tp2 + drafter graphs + XQA off + MNBT 8192 + tier ==="
settle; ok=0
for att in 1 2; do
  if IMAGE=$IMG MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 NO_TIER=0 \
     KVD_OVERRIDE=nvfp4 FIWS=536870912 ALLOW_NO_XQA=1 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 \
     SPEC_JSON="$SPEC" EXTRA_ENV="$ENVG" EXTRA_MOUNT="-v $DRAFT:/draft:ro" bash $LAUNCH > "$R/boot-$T.log" 2>&1; then ok=1; break; fi
  log "[$T] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-$T.log" | tail -1 | cut -c1-150)"
  timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
done
if [ "$ok" = 1 ]; then
  grep -aE "KV pool" "$R/boot-$T.log" | tail -1 | sed "s/^/[$T] POOL /" | tee -a "$R/audit.log"
  sudo docker logs vllm-exp 2>&1 | grep -aE "DFLASH_GRAPHS=1|Capturing dflash2|running the draft eagerly" | head -3 | cut -c1-160 | sed "s/^/[$T graphs] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 9000 20000 131000 220000 --samples 2 \
    --out "$R/needles-$T.jsonl" > "$R/needles-$T.out" 2>&1
  grep -a SUMMARY "$R/needles-$T.out" | head -1 | sed "s/^/[$T needles] /" | cut -c1-160 | tee -a "$R/audit.log"
  grep -a "MISS" "$R/needles-$T.out" | head -4 | cut -c1-200 | sed "s/^/[$T needles] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-$T.log" 2>&1
  tail -2 "$R/warm-revisit-$T.log" | cut -c1-200 | sed "s/^/[$T revisit] /" | tee -a "$R/audit.log"
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency 1 8 --runs 3 --no-cache --latency-mode api --format json $EB \
    --save-result "$R/benchy-$T-nat.json" > "$R/benchy-$T-nat.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$T-nat.json" "benchy.$T.nat.t0.6" | tee -a "$R/audit.log"
  for kind in code prose; do
    python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 8 --tokens 512 --runs 3 --kind $kind \
      --out "$R/decode-$T-$kind.json" > "$R/decode-$T-$kind.out" 2>&1
    grep -a RESULT "$R/decode-$T-$kind.out" | sed "s/^/[$T decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"
  done
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 \
    --out "$R/decode-$T-deep30k.json" > "$R/decode-$T-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$T-deep30k.out" | sed "s/^/[$T decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"
  curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$T] /" | tee -a "$R/audit.log"
  log "[$T] tool-eval x4 start"
  ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 \
      --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$T.json" > "$R/tooleval-$T.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$T.json" "$T" 2>&1 | tee -a "$R/audit.log"
  sudo docker logs vllm-exp > "$R/engine-$T.log" 2>&1 || true
  grep -ac "illegal memory\|CUDA error\|Traceback" "$R/engine-$T.log" | sed "s/^/[$T engine error-lines] /" | tee -a "$R/audit.log"
else log "[$T] BOOT FAILED"; fi
settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R158c DONE ==="
