#!/usr/bin/env bash
# R158 — where do the nvfp4-KV DFlash2 single-stream points go? Kernel-level attribution with the
# torch profiler (user 2026-09-02: "make nvfp4 work"). All arms: RedHat, TP2, ns7, draft_tp=1
# (drafter identical across arms), MNBT 8192, util 0.90, NO tier — only the KV route differs:
#   n  = nvfp4 KV, XQA off (revival image): q_len=1 and q_len=8 verify both on FA2-over-nvfp4
#   f  = fp8 KV (daily image): dedicated XQA fp8 decode + XQA uniform-spec verify (the daily's route)
#   fa = fp8 KV + --attention-config.use_trtllm_attention=false: fp8 through FA2 (verify becomes a
#        "prefill", likely PIECEWISE) — kernel-time-only control; its wall numbers are NOT a route
#        comparison (advisor caveat)
# Between n and f (GPU free): the Bug A differential harness (probes/nvfp4_fa2_harness.py) sweeps
# hd x H x page x gqa x causal outside vLLM.
# Profiler: --profiler-config.* dotted flags (v0.28), delay 5 iterations (skips the prefill step),
# 60 active iterations, stack off; one trace per TP rank in $R/prof-<tag>/<c1|c8>/.
#   sudo systemd-run --unit=r158-profile --collect -p User=adrienbrault -p RuntimeMaxSec=9000 bash /srv/qwen5090/r158-profile.sh
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
PROF_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true --profiler-config.delay_iterations=5 --profiler-config.max_iterations=60"
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 30; }

boot(){ local tag=$1; shift
  settle; mkdir -p "$R/prof-$tag"; chmod 777 "$R/prof-$tag"
  for att in 1 2; do
    if env "$@" MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=0.90 MAXLEN=262144 NO_TIER=1 \
       PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 \
       EXTRA_MOUNT="-v $DRAFT:/draft:ro -v $R/prof-$tag:/prof" bash $LAUNCH > "$R/boot-$tag.log" 2>&1; then
      grep -aE "KV pool" "$R/boot-$tag.log" | tail -1 | sed "s/^/[$tag] POOL /" | tee -a "$R/audit.log"; return 0; fi
    log "[$tag] BOOT attempt $att failed: $(grep -aE 'FAILED|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-150)"
    timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1; sleep 40
  done; return 1
}

# capture <tag> <conc>: start profiler, one natural benchy run at that concurrency, stop, collect traces
capture(){ local tag=$1 conc=$2; local d="$R/prof-$tag/c$conc"; mkdir -p "$d"
  curl -s -m 10 -X POST $U/start_profile >/dev/null 2>&1 || log "[$tag c$conc] start_profile returned non-2xx (continuing)"
  timeout 900 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency $conc --runs 1 --no-cache --latency-mode api --format json $EB \
    --save-result "$d/benchy.json" > "$d/benchy.out" 2>&1
  curl -s -m 60 -X POST $U/stop_profile >/dev/null 2>&1 || true
  for i in $(seq 60); do n=$(ls "$R/prof-$tag"/*.json* 2>/dev/null | wc -l); [ "$n" -ge 2 ] && break; sleep 3; done
  sleep 15   # let the async dump finish writing
  mv "$R/prof-$tag"/*.json* "$d/" 2>/dev/null
  ls -la "$d" | sed "s/^/[$tag c$conc] /" | tee -a "$R/audit.log"
  for t in "$d"/*.json*; do [ -f "$t" ] || continue
    python3 /srv/qwen5090/probes/prof_summary.py "$t" --steps 60 --label "$tag c$conc $(basename $t)" > "$t.summary.txt" 2>&1
    head -16 "$t.summary.txt" | sed "s/^/[$tag c$conc] /" | tee -a "$R/audit.log"
  done
}

battery(){ local tag=$1
  capture $tag 1
  capture $tag 8
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency 1 8 --runs 3 --no-cache --latency-mode api --format json $EB \
    --save-result "$R/benchy-$tag-nat.json" > "$R/benchy-$tag-nat.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$tag-nat.json" "benchy.$tag.nat.t0.6" | tee -a "$R/audit.log"
  curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  sudo docker logs vllm-exp > "$R/engine-$tag.log" 2>&1 || true
  grep -a "decode_backend\|use_fa2_nvfp4_kv\|XQA\|cudagraph_mode\|CUDAGraphMode" "$R/engine-$tag.log" | head -8 | cut -c1-200 | sed "s/^/[$tag route] /" | tee -a "$R/audit.log"
}

SPEC7='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1}'
SPEC7N='{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}'
log "=== R158 start: n (nvfp4 FA2, XQA off) -> harness -> f (fp8 XQA) -> fa (fp8 forced FA2); all ns7 draft_tp1 ==="

if boot n IMAGE=vllm-qwen38:v0280-nvfp4kv-revival KVD_OVERRIDE=nvfp4 FIWS=536870912 ALLOW_NO_XQA=1 \
     SPEC_JSON="$SPEC7N" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0" EXTRA_ARGS="$PROF_ARGS"; then battery n; else log "[n] BOOT FAILED"; fi

# Bug A harness on a free GPU (revival image; flashinfer JIT cache mounted so the fp4 FA2 variants are warm)
settle
log "[harness] start (full sweep)"
timeout 1800 sudo docker run --rm --gpus '"device=0"' --entrypoint python3 \
  -v /srv/qwen5090/cache/flashinfer:/root/.cache/flashinfer -v /srv/qwen5090/probes:/probes:ro -v "$R":/out \
  vllm-qwen38:v0280-nvfp4kv-revival /probes/nvfp4_fa2_harness.py --out /out/harness.jsonl > "$R/harness.out" 2>&1
log "[harness] exit=$? $(grep -a '^SUMMARY\|^cells' "$R/harness.out" | tr '\n' ' ')"
grep -a "RED\|ERROR" "$R/harness.out" | head -40 | sed "s/^/[harness] /" | tee -a "$R/audit.log"

if boot f KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 SPEC_JSON="$SPEC7" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" EXTRA_ARGS="$PROF_ARGS"; then battery f; else log "[f] BOOT FAILED"; fi

if boot fa KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 SPEC_JSON="$SPEC7" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" \
     EXTRA_ARGS="$PROF_ARGS --attention-config.use_trtllm_attention=false"; then battery fa; else log "[fa] BOOT FAILED"; fi

settle
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R158 DONE ==="
