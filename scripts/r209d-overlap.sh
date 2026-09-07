#!/usr/bin/env bash
# R209d -- MEASURE the prefill/decode overlap that R209/R209b/R209c only asserted (2026-09-08).
#
# WHY THIS EXISTS. R209 claimed the mixed regime was "proven, not asserted" from a 1 Hz sample of vllm:prompt_tokens_total and
# vllm:generation_tokens_total: prefill_only == 0.00 in every row was read as "chunked prefill never ran without decode".
# THAT PROOF WAS STRUCTURAL, NOT EVIDENCE. In v1 the prompt counter does not advance per prefill chunk:
#   scheduler.py:935  request.prefill_stats.set(num_prompt_tokens=request.num_prompt_tokens, ...)   # FULL prompt, once,
#                     under "Track first scheduled prefill", i.e. at the request's first scheduling
#   scheduler.py:2055 prefill_stats = request.take_prefill_stats()  # consumed only under should_emit_output
# so the whole prompt is booked in one shot at the step that emits the request's FIRST TOKEN -- an event that by construction
# also advances generation_tokens_total. prefill_only == 0 was therefore impossible to violate, and `both` merely counted
# first-token events: 8K 0.79 req/s vs both 0.63-0.66, 32K 0.21 req/s vs both 0.19-0.20. The metric measured arrival rate.
# vllm:iteration_tokens_total is NOT a way out -- loggers.py:1202 observes prompt_token_stats.computed + num_generation_tokens
# and `computed` is fed by that same once-per-request PrefillStats, which is why the live histogram shows 586 steps above
# 16,384 tokens under --max-num-batched-tokens 8192. Both counter routes are dead.
#
# WHAT THIS DOES INSTEAD: reconstruct every request's phase timeline CLIENT-SIDE, where no engine accounting is involved.
# serve.py:1287 saves per-request start_times, and --save-detailed keeps them (serve.py:2352) alongside ttfts and itls. With
# (start, ttft, itls) per request, request i is in its PREFILL phase over [start_i, start_i+ttft_i) and in DECODE over
# [start_i+ttft_i, +sum(itls)). Sampling those intervals gives n_prefill(t) and n_decode(t) directly, so the overlap fraction
# is measured rather than argued -- and the per-token itl timestamps say whether long decode stalls land while a prefill is
# actually in flight, which is the mechanism.
# Little's law (sum(TTFT)/window) is NOT proof and is not used as such: R208's pf-c8-30k is eight lockstep prefills followed by
# eight lockstep decodes and would score occupancy ~4 with ZERO overlap. Occupancy is a magnitude, the timeline is the proof.
#
# R209's --save-result raised FileNotFoundError because it wrote under the root-owned /l2 mount. --result-dir /tmp is inside
# the container and writable, and serve.py:1556 makedirs it; the JSON is copied out with docker exec cat.
#
# BASELINE ONLY -- no power arms. R209b/R209c already measured the 400 W cap; this unit re-measures nothing but the regime,
# so both cards stay at their defaults and the run is ~4 min of GPU.
#   unit: sudo systemd-run --unit=r209d-overlap --collect -p User=adrienbrault -p RuntimeMaxSec=21600 -p TimeoutStopSec=300 \
#         -E GPU_QUEUE_NAME=r209d-overlap bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r209d-overlap.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-08-r209d-overlap; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
# per-invocation seed nonce: a killed run leaves its prompts in the prefix cache AND the KV tier, so reusing seeds reads as a
# fake speedup (2026-09-07: 13,984 vs 6,658 tok/s, all cache). numpy caps --seed at 2**32-1.
NONCE=$(( $(date +%s) % 100000 ))
U=http://127.0.0.1:8020
CU=http://127.0.0.1:8000
M=qwen3.8-27b

command -v nvidia-smi >/dev/null || { log "ABORT: nvidia-smi missing"; exit 3; }
docker inspect vllm-27b >/dev/null 2>&1 || { log "ABORT: container vllm-27b not present"; exit 3; }
curl -sf -m 5 $U/health >/dev/null || { log "ABORT: the daily is not up on :8020 — this unit measures the live daily"; exit 3; }
curl -sf -m 5 $U/v1/models | grep -q "$M" || { log "ABORT: :8020 does not serve $M"; exit 3; }
docker exec vllm-27b sh -c "curl -sf -m 5 $CU/health" >/dev/null || { log "ABORT: bench client cannot reach $CU"; exit 3; }
systemctl is-active --quiet tier-evict.timer || { log "ABORT: tier-evict.timer is not active and this unit writes ~30 GB of unique KV"; exit 3; }
docker exec vllm-27b sh -c 'vllm bench serve --help=save-detailed 2>&1 | grep -q -- --save-detailed' \
  || { log "ABORT: this vllm build has no --save-detailed, so per-request start_times cannot be saved"; exit 3; }
log "preflight OK: daily up, bench reachable, tier-evict.timer active, --save-detailed present"

cleanup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; [ -n "${SAMP:-}" ] && kill "$SAMP" 2>/dev/null; }
trap 'log "### SIGTERM/INT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT

# gpu-queue.sh only registers the unit; the flock is what serializes. r209b and r209c overlapped on 2026-09-07 for want of this
# and both datasets were discarded -- read-only units that only measure the daily feel safe and are not.
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R209d start (lock held; the daily stays UP on :8020) ==="

nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$R/power.csv" & SAMP=$!
echo "arm,row,t_start,t_end" > "$R/marks.csv"
tier_gb(){ du -sb /srv/qwen5090/native-l2 2>/dev/null | awk '{printf "%.1f", $1/1e9}'; }

# $1 row  $2 isl  $3 osl  $4 nprompts  $5 range-ratio json  $6 seed  $7 extra args
mix_row(){
  local row=$1 isl=$2 osl=$3 n=$4 rr=$5 seed=$6 extra=$7
  local out="$R/${row}.txt" jf="/tmp/r209d-${row}.json"
  log "  row $row: isl=$isl osl=$osl n=$n conc=8 rr=$rr seed=$seed extra='$extra' (nonce $NONCE)"
  local t0=$(date +%s.%N)
  docker exec vllm-27b vllm bench serve \
    --backend openai --endpoint /v1/completions --base-url "$CU" \
    --model "$M" --tokenizer /model \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio "$rr" --num-prompts "$n" --max-concurrency 8 \
    --seed "$seed" --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir /tmp --result-filename "r209d-${row}.json" \
    $extra > "$out" 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "$row,$row,$t0,$t1" >> "$R/marks.csv"
  [ $rc -eq 0 ] || log "  WARN row $row exited $rc (see $out)"
  docker exec vllm-27b cat "$jf" > "$R/${row}.json" 2>/dev/null
  if [ -s "$R/${row}.json" ]; then
    log "  saved detailed JSON: $(wc -c < "$R/${row}.json") bytes"
  else
    log "  ERROR: detailed JSON for $row is EMPTY — the overlap timeline cannot be built for this row"
  fi
  grep -E "Successful requests|Benchmark duration|Total input|Total generated|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|Mean TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL" "$out" | sed 's/^/    /' | tee -a "$R/audit.log"
}

log "### R209d prefill/decode overlap timeline, live daily on :8020, tier=$(tier_gb) GB ###"
# identical parameters to the R209c and R209b BASELINE rows, so the timelines describe those published numbers
mix_row mix-8k        8000  600 48 '{"input":0,"output":0.6}' "$(( NONCE * 1000 + 41 ))" ""
log "  tier=$(tier_gb) GB"
mix_row mix-32k-deep 32000  800 32 '{"input":0,"output":0.5}' "$(( NONCE * 1000 + 51 ))" "--request-rate 0.6"
log "  tier=$(tier_gb) GB"

kill "$SAMP" 2>/dev/null; SAMP=
python3 /srv/qwen5090/r209d-overlap.py "$R" 2>&1 | tee -a "$R/audit.log"
log "### R209d DONE — results in $R ###"
