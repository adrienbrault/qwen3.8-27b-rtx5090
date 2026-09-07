#!/usr/bin/env bash
# R209c mixed prefill+decode at 8K -- control-clean re-run of R209's mix-8k row (input length fixed) (2026-09-07, user "have you measured perf impact when many requests are prefilling/decoding?
# a c8 of both prefill and decode?").
#
# WHY THIS EXISTS. R208's rows are each ONE phase. The decode rows (code c1/c8/c16) use short prompts, so they are almost pure
# decode. pf-c8-30k is a SYNCHRONIZED BURST: eight identical-length prompts prefill in lockstep, then all eight decode in lockstep.
# Neither is the serving regime, where chunked prefill batches an arriving request's prefill chunks alongside other requests'
# ongoing decode steps. This unit measures that regime and PROVES it held, rather than asserting it.
#
# INSTRUMENT: `vllm bench serve --dataset-name random` (repo-standard, R156d: do not write a bespoke probe for a solved problem).
# It runs INSIDE the container: netmode is bridge and vLLM listens on :8000 internally, mapped to host :8020. The random dataset
# emits unique prompts, and --random-range-ratio spreads BOTH input and output length so completions destagger after the opening
# wave — that destaggering is what produces continuous mixing instead of R208's lockstep waves.
#
# THE R208 TRAP, FENCED. decode_ss.py seeded prompts deterministically, so every arm after the first was served from the prefix
# cache and read a fake -94% TTFT. Here each arm passes its own --seed, so no arm can reuse another's prompts. The lengths and
# distribution are identical across arms and the tokens are random, so content is not a confound (R156c: kernel speed is
# content-independent).
#
# ARMS (the user's power question, carried into the mixed regime):
#   A0   default limits  (GPU0 600 W, GPU1 575 W — asymmetric, restored per index)
#   A1   -pl 400 on both (the lowest cap the hardware accepts; 300 W is below power.min_limit and CANNOT be set)
#   A0b  default again   (same-config control, R203 discipline — decode's noise floor is +-4%, so a cost smaller than the control
#                         swing is not a result)
# No clock-lock arm: R208 already settled that -lgc is the wrong lever (it drew LESS than the 400 W cap and cost ~4x more).
#
# TIER BUDGET. Every prefill writes the KV tier at ~19.6 KB/token and random prompts dedupe against nothing, so this is the
# no-reuse worst case. Planned writes are ~1.15M prompt tokens/arm x 3 arms ~= 68 GB against a 300 GB cap that sat at 287 GB.
# That is safe ONLY because tier-evict.timer is active (it was not on 2026-09-01, when an uncapped tier stranded the daily), so
# this unit ABORTS if the timer is not running and logs tier size at every arm boundary.
#
# This measures THE LIVE DAILY on :8020 unmodified — the lever is host-level, not an engine change (R207/R208 gate precedent).
# The daily stays up and is SLOWER while the unit runs; other clients hitting :8020 skew the rows.
#   unit: sudo systemd-run --unit=r209c-mixed-8k --collect -p User=adrienbrault -p RuntimeMaxSec=21600 -p TimeoutStopSec=300 \
#         -p 'ExecStopPost=-/bin/bash /srv/qwen5090/r208-restore-power.sh' \
#         -E GPU_QUEUE_NAME=r209c-mixed-8k bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r209c-mixed-8k.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-07-r209c-mixed-8k; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
# PER-INVOCATION SEED NONCE. Per-ARM seeds are not enough: a killed run leaves its prompts in the prefix cache and the KV
# tier, so a re-run with the same seeds is served from cache and reads as a huge fake speedup. On 2026-09-07 the first r209b/c
# launch was killed mid-flight, and the re-run's A0 came back at 13,984 total tok/s against its own control's 6,658 -- 2x, all
# cache. Seeding from the clock makes every invocation's prompts unique, so no run can ever inherit another's cache.
# numpy caps --seed at 2**32-1, so the nonce is the clock mod 100000 and seeds are built arithmetically
NONCE=$(( $(date +%s) % 100000 ))
U=http://127.0.0.1:8020          # host-side, for health/metrics
CU=http://127.0.0.1:8000         # container-side, for the bench client
M=qwen3.8-27b
# No --save-result: native-l2 is root-owned, so the container could not create a subdir for it and R209's bench raised
# FileNotFoundError AFTER printing a complete table (exit 1 on every row). The summary parses the stdout table instead.

command -v nvidia-smi >/dev/null || { log "ABORT: nvidia-smi missing"; exit 3; }
docker inspect vllm-27b >/dev/null 2>&1 || { log "ABORT: container vllm-27b not present"; exit 3; }
curl -sf -m 5 $U/health >/dev/null || { log "ABORT: the daily is not up on :8020 — this unit measures the live daily, it does not boot one"; exit 3; }
curl -sf -m 5 $U/v1/models | grep -q "$M" || { log "ABORT: :8020 does not serve $M"; exit 3; }
docker exec vllm-27b sh -c "curl -sf -m 5 $CU/health" >/dev/null || { log "ABORT: bench client cannot reach $CU from inside the container"; exit 3; }
systemctl is-active --quiet tier-evict.timer || { log "ABORT: tier-evict.timer is not active — an uncapped tier stranded the daily on 2026-09-01 and this unit writes ~68 GB of unique KV"; exit 3; }
log "preflight OK: daily up, bench client reaches $CU, tier-evict.timer active"

# --- per-card defaults, read once, restored unconditionally -------------------------------------------------------------------
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.min_limit,clocks.max.sm,persistence_mode \
  --format=csv > "$R/gpu-caps.csv"; log "GPU caps:"; cat "$R/gpu-caps.csv" | tee -a "$R/audit.log"
DEF0=$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
DEF1=$(nvidia-smi -i 1 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
MIN0=$(nvidia-smi -i 0 --query-gpu=power.min_limit --format=csv,noheader,nounits | tr -d ' ')
MIN1=$(nvidia-smi -i 1 --query-gpu=power.min_limit --format=csv,noheader,nounits | tr -d ' ')
[ -n "$DEF0" ] && [ -n "$DEF1" ] || { log "ABORT: could not read power.default_limit"; exit 3; }
FLOOR=$(python3 -c "print(int(max(float('$MIN0'),float('$MIN1'))))")
log "defaults: GPU0 ${DEF0} W, GPU1 ${DEF1} W; floor ${MIN0}/${MIN1} W -> A1 caps both at ${FLOOR} W"
cat > /srv/qwen5090/r208-restore-power.sh <<EOR
#!/usr/bin/env bash
sudo -n nvidia-smi -rgc >/dev/null 2>&1
sudo -n nvidia-smi -i 0 -pl ${DEF0} >/dev/null 2>&1
sudo -n nvidia-smi -i 1 -pl ${DEF1} >/dev/null 2>&1
EOR
chmod +x /srv/qwen5090/r208-restore-power.sh

restore(){ sudo -n nvidia-smi -rgc >/dev/null 2>&1; sudo -n nvidia-smi -i 0 -pl "$DEF0" >/dev/null 2>&1; sudo -n nvidia-smi -i 1 -pl "$DEF1" >/dev/null 2>&1; }
cleanup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; restore; [ -n "${SAMP:-}" ] && kill "$SAMP" 2>/dev/null; [ -n "${ESAMP:-}" ] && kill "$ESAMP" 2>/dev/null
  nvidia-smi --query-gpu=index,power.limit,clocks.max.sm --format=csv | tee -a "$R/audit.log"; }
trap 'log "### SIGTERM/INT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT

# GPU-EXCLUSIVE LOCK. Without this two of these units run against the same daily at once and BOTH datasets are garbage: on
# 2026-09-07 r209b and r209c overlapped from 21:30 and had to be discarded. Sourcing lib/gpu-queue.sh only registers the unit
# in the queue so a chain pays one daily down/up -- it does NOT serialize anything. The flock is what serializes.
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R209c start (lock held; the daily stays UP on :8020) ==="

# --- samplers ------------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$R/power.csv" & SAMP=$!
# engine state at 1 Hz: this is the EVIDENCE that prefill and decode ran concurrently. prompt_tokens_total and
# generation_tokens_total both advancing within the same second is what "mixed" means; num_requests_running says how deep.
( echo "t,running,waiting,prompt_tokens_total,generation_tokens_total"
  while :; do
    curl -s -m 3 $U/metrics 2>/dev/null | awk -v t="$(date +%s.%N)" '
      /^vllm:num_requests_running/{r=$NF} /^vllm:num_requests_waiting\{/{w=$NF}
      /^vllm:prompt_tokens_total/{p=$NF} /^vllm:generation_tokens_total/{g=$NF}
      END{if(p!="")printf "%s,%s,%s,%s,%s\n",t,r,w,p,g}'
    sleep 1
  done ) > "$R/engine.csv" & ESAMP=$!
echo "arm,row,t_start,t_end" > "$R/marks.csv"
mark(){ echo "$1,$2,$3,$4" >> "$R/marks.csv"; }

tier_gb(){ du -sb /srv/qwen5090/native-l2 2>/dev/null | awk '{printf "%.1f", $1/1e9}'; }

# --- one mixed row -------------------------------------------------------------------------------------------------------------
# $1 arm  $2 row  $3 isl  $4 osl  $5 nprompts  $6 range-ratio json  $7 seed
mix_row(){
  local arm=$1 row=$2 isl=$3 osl=$4 n=$5 rr=$6 seed=$7
  local out="$R/${arm}-${row}.txt"
  log "  row $arm/$row: isl=$isl osl=$osl n=$n conc=8 rr=$rr seed=$seed (nonce $NONCE: prompts unique to this invocation)"
  local t0=$(date +%s.%N)
  docker exec vllm-27b vllm bench serve \
    --backend openai --endpoint /v1/completions --base-url "$CU" \
    --model "$M" --tokenizer /model \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio "$rr" --num-prompts "$n" --max-concurrency 8 \
    --seed "$seed" --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    > "$out" 2>&1
  local rc=$? t1=$(date +%s.%N)
  mark "$arm" "$row" "$t0" "$t1"
  [ $rc -eq 0 ] || log "  WARN row $arm/$row exited $rc (see $out)"
  grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT|Mean ITL|P99 ITL" "$out" | sed 's/^/    /' | tee -a "$R/audit.log"
}

arm_rows(){
  local arm=$1 seedbase=$2
  log "== ARM $arm == tier=$(tier_gb) GB"
  # R209's mix-32k was DISQUALIFIED by its own control (A0b output tok/s -29.3%, TPOT p50 +33.2%). Cause: n=24 with
  # --random-range-ratio 0.3 on the INPUT never averaged out, so arms drew different amounts of prefill work (A0b 805K input
  # tokens vs A0 745K, 8% more) and that swamped the power effect. Three fixes here:
  #   * input range ratio 0 -> every arm does IDENTICAL prefill work. Prompts stay unique (random tokens, per-arm seed), so
  #     the R208 prefix-cache trap is still fenced; only the length confound is removed.
  #   * n 24 -> 32 and osl 400 -> 800, so the run is longer than its own opening wave.
  #   * --request-rate 0.6 (Poisson arrivals) instead of pure max-concurrency waves. R209 showed prefill_only=0.00 in every
  #     row -- prefill NEVER ran without decode alongside -- but with lockstep waves the tail was 66% decode-only. Staggered
  #     arrivals keep new prefills landing while earlier requests are still decoding, which is the regime being asked about.
  # Identical to R209's mix-8k row in every parameter EXCEPT input range ratio 0.4 -> 0. R209 read the 400 W cap as
  # +19.2% TTFT p50 there, but A1 happened to draw 3.3% more input tokens than A0 (393K vs 381K) from 48 draws of a +-40%
  # length spread, so the magnitude was inflated. Fixing input length isolates the cap; it also doubles as a second sample of
  # the noise floor, which R209 put at ~5% on this row.
  mix_row "$arm" mix-8k 8000 600 48 '{"input":0,"output":0.6}' "$(( NONCE * 1000 + seedbase * 10 + 1 ))"
  log "== ARM $arm done == tier=$(tier_gb) GB"
}

log "### R209c mixed prefill+decode at 8K (input length fixed), live daily on :8020 ###"
arm_rows A0 10
log "--- A1: capping both cards at ${FLOOR} W ---"
sudo -n nvidia-smi -i 0 -pl "$FLOOR" | tee -a "$R/audit.log"
sudo -n nvidia-smi -i 1 -pl "$FLOOR" | tee -a "$R/audit.log"
nvidia-smi --query-gpu=index,power.limit --format=csv | tee -a "$R/audit.log"
arm_rows A1 20
log "--- A0b: restoring defaults for the control ---"
restore; nvidia-smi --query-gpu=index,power.limit --format=csv | tee -a "$R/audit.log"
arm_rows A0b 30

kill "$SAMP" 2>/dev/null; kill "$ESAMP" 2>/dev/null; SAMP=; ESAMP=
python3 /srv/qwen5090/r209-mixed-summary.py "$R" 2>&1 | tee -a "$R/audit.log"
log "### R209c DONE — results in $R ###"
restore
nvidia-smi --query-gpu=index,power.limit,clocks.max.sm --format=csv | tee -a "$R/audit.log"
