#!/usr/bin/env bash
# R211b — the KNEE of the prefill-chunk curve, plus the admission cap at the good chunk size. Follow-up to R211, same rows,
# same prompts, same instrument. R211 found: 2 blocks costs 7.5% output throughput and 2.6% steady E2EL against 5 blocks and
# is worse on every ramp metric; 11 blocks TIES 5 blocks because at 75% prefix reuse a request only has ~8K uncached, so a
# chunk above 5 blocks has nothing left to fill (the stall is min(chunk, remaining prefill), which is why arm D broke the
# 1:2:5:11 scaling on the steady row and kept it on the cold ramp). So the optimum is at or below 5 blocks and the untested
# ground is 3 and 4. Separately R211's arm E showed SEQS 32 cuts steady TTFT 32% (queue 36%) but costs 12% E2EL -- arm H asks
# whether that trade is better at 5 blocks than it was at 2.
#
# WHY. R209d established that a decoding stream's stall IS one prefill step (0 stalls in 7,045 decode steps with no prefill in
# flight; 26.2 % / 44.1 % with one) and the 2026-09-08 daily change from --max-num-batched-tokens 8192 to 4096 was made on that
# basis. The live omp run afterwards is consistent with it (stall median ~453 ms against R209d's 904 ms at comparable 32K depth)
# but is NOT a controlled comparison: different harness, different concurrency, cold-vs-warm cache. This unit is the controlled
# ladder, and it also answers whether 4096 is the right rung or an arbitrary one.
#
# THE THING THAT MAKES THIS LADDER NON-OBVIOUS: MNBT IS NOT THE CHUNK SIZE ON THIS HYBRID.
# scheduler.py:438-446 snaps every non-final prefill chunk DOWN to a multiple of cache_config.block_size, because in "align"
# mamba cache mode the SSM state is materialised at block boundaries. The escape hatch (mamba_has_prefill_checkpoint_blocks,
# scheduler.py:335) requires `not use_eagle`, and SpeculativeConfig.use_eagle() returns True for method "mtp" — which is the
# daily's route since R207 (the boot log's "SM12X eagle-drop replay boundary retained" line confirms it live). So on the daily,
# block_size = 1,472 and the effective chunk is floor(MNBT / 1472) blocks:
#     MNBT  1536 ->  1 block  =  1,472      MNBT  4096 ->  2 blocks =  2,944   (the daily since 2026-09-08)
#     MNBT  8192 ->  5 blocks =  7,360      MNBT 16384 -> 11 blocks = 16,192   (8192 was the daily before it)
# So the change already made was a 5x -> 2x chunk reduction, not the 2x it was described as, and MNBT values between block
# multiples buy nothing. THIS IS THE FALSIFIABLE PREDICTION THIS UNIT TESTS: if the stall IS the chunk step, the measured stall
# medians across arms C/B/A/D must scale as 1 : 2 : 5 : 11. If they do not, the R209d mechanism is incomplete.
#
# ARMS (all EXP=1 on :8029 / vllm-exp, daily image, MTP ns3, PCIE_IPC=1, BSS=1 — i.e. the daily's own route, so a winner is
# directly promotable; SEQS 16 unless stated):
#   A   MNBT  8192            5 blocks   the pre-2026-09-08 daily            (control)
#   B   MNBT  4096            2 blocks   the daily as it stands now
#   C   MNBT  1536            1 block    the floor
#   D   MNBT 16384           11 blocks   the opposite direction, to bound the curve
#   E   MNBT  4096, SEQS 32   2 blocks   admission cap raised above the offered load (a DIFFERENT question: the user's
#                                        complaint is a slow START, and their run showed ~3.3 s of every 4.89 s TTFT was
#                                        QUEUE, not prefill compute — chunk size cannot touch that, a seq cap can)
#   A'  MNBT  8192            5 blocks   same-config repeat of A = the boot-to-boot noise floor (R203 discipline: an effect
#                                        smaller than the A/A' swing is not a result)
#
# ROWS per arm, in this order, on a freshly wiped tier:
#   plant   n=1   prefix 24000 + input 8000. Not scored. Without it the "steady" row's first 20 requests all arrive with an
#                 UNCACHED shared prefix and redundantly prefill it, baking a cold burst into the steady-state stall count.
#   steady  n=60  prefix 24000 + input 8000, osl 900, conc 20 -> 75 % prefix reuse at ~32K prompts. THE DECISION ROW: the live
#                 daily ran 82.5 % prefix hit at a 49K mean prompt, so this is the regime that matters.
#   ramp    n=24  no prefix, input 32000, osl 900, conc 20 -> 0 % reuse. The cold-burst / worst case, and the row that speaks to
#                 "the start is slow". NOT steady state: one burst of 20 plus 4 stragglers, and it is scored as a ramp.
#
# SEEDS: the SAME seed for every arm, deliberately, which INVERTS the usual rule. benchmark-seed-uniqueness exists because a
# killed run's prompts survive in the live daily's caches and serve the re-run a fake speedup. Here every arm is a fresh boot
# (empty GPU cache) onto a WIPED eval-l2 tier, so there is nothing to inherit — and holding the seed fixed removes the last work
# confound: get_sampling_params draws output lengths from the same rng as the prefix, so per-arm seeds would give each arm a
# few % different total decode work, which is exactly the size of the effect being measured (this is the R209 input-length bug
# in output form). A per-INVOCATION nonce still guards against a killed R211 feeding its own re-run.
#
# INSTRUMENT: `vllm bench serve` (repo-standard, R156d) for the client-side table, PLUS per-row deltas of the engine's own
# histograms scraped before and after each row. The ITL bucket delta is the primary score: it gives the stall RATE and the stall
# LENGTH distribution directly, per row, with no client-side reconstruction. request_queue_time vs request_prefill_time deltas
# split TTFT into admission and compute, which is what separates arm E's question from arms A-D's.
#
# COST: ~1.27M unique prompt tokens per arm; eval-l2 (415 G free) is wiped per arm so peak tier is ~25 G. Six boots, ~15 min
# each. The daily is DOWN for the duration and is restored by this unit at the end (one down/up for the chain).
#
#   unit: sudo systemd-run --unit=r211b-mnbt-knee --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=300 \
#         -E GPU_QUEUE_NAME=r211b-mnbt-knee bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r211b-mnbt-knee.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-08-r211b-mnbt-knee; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }

# PINNED to R211's nonce: these arms are compared against R211's A/B/A2, so they must draw the IDENTICAL prompts. Safe despite
# benchmark-seed-uniqueness: every arm boots fresh onto a WIPED eval-l2, so there is no cache to inherit -- that rule guards
# against a killed run's prompts surviving in a LIVE daily, which is not this shape. A3 re-runs R211's A config and is the
# CROSS-RUN ANCHOR: if A3 does not land on A/A2, the two runs are not on one scale and the joined table is void.
NONCE=63681
SEED_STEADY=$(( NONCE * 10 + 1 ))
SEED_RAMP=$(( NONCE * 10 + 2 ))
U=http://127.0.0.1:8029          # host-side, health/metrics
CU=http://127.0.0.1:8000         # container-side, bench client
NAME=vllm-exp
M=qwen3.8-27b
L2=/srv/qwen5090/eval-l2

command -v nvidia-smi >/dev/null || { log "ABORT: nvidia-smi missing"; exit 3; }
mountpoint -q "$L2" || { log "ABORT: $L2 not mounted — refusing to wipe/write an unmounted tier path"; exit 3; }
systemctl is-active --quiet tier-evict.timer || { log "ABORT: tier-evict.timer is not active"; exit 3; }
[ -x /srv/qwen5090/launch-daily.sh ] || [ -f /srv/qwen5090/launch-daily.sh ] || { log "ABORT: launch-daily.sh missing"; exit 3; }
log "preflight OK (nonce $NONCE; seeds steady=$SEED_STEADY ramp=$SEED_RAMP, IDENTICAL across arms by design)"

nvidia-smi --query-gpu=index,name,power.limit,power.default_limit --format=csv > "$R/gpu-caps.csv"

DAILY_RESTORED=0
restore_daily(){
  [ "$DAILY_RESTORED" = 1 ] && return 0
  DAILY_RESTORED=1
  log "restoring the daily (env -i, canonical invocation)"
  timeout 120 sudo docker rm -f "$NAME" >/dev/null 2>&1
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/daily-restore.log" 2>&1 \
    && log "DAILY RESTORED" || log "DAILY RESTORE FAILED — see $R/daily-restore.log"
}
cleanup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; [ -n "${SAMP:-}" ] && kill "$SAMP" 2>/dev/null; restore_daily; }
trap 'log "### SIGTERM/INT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT

exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock"; flock 9; }
log "=== R211b start (lock held; the daily goes DOWN for the ladder) ==="

nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 500 > "$R/power.csv" & SAMP=$!

echo "arm,row,t_start,t_end,rc" > "$R/marks.csv"
tier_gb(){ du -sb "$L2" 2>/dev/null | awk '{printf "%.1f", $1/1e9}'; }
snap(){ curl -s -m 10 "$U/metrics" > "$R/metrics-$1-$2-$3.prom" 2>/dev/null; }

wipe_tier(){
  log "  wiping $L2 (was $(tier_gb) GB)"
  sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + 2>/dev/null
  sync; log "  tier now $(tier_gb) GB"
}

# $1 arm  $2 mnbt  $3 seqs
boot(){
  local arm=$1 mnbt=$2 seqs=$3
  log "== ARM $arm: MNBT=$mnbt SEQS=$seqs (predicted chunk $(( mnbt / 1472 )) x 1472 = $(( mnbt / 1472 * 1472 )) tokens) =="
  timeout 180 sudo docker rm -f "$NAME" vllm-27b >/dev/null 2>&1
  wipe_tier
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" \
      EXP=1 EXP_MNBT="$mnbt" SEQS="$seqs" PCIE_IPC=1 BSS=1 \
      bash /srv/qwen5090/launch-daily.sh > "$R/boot-$arm.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  BOOT FAILED for $arm (rc $rc): $(grep -aE 'FAILED' "$R/boot-$arm.log" | tail -1 | cut -c1-160)"
    return 1
  fi
  local pool; pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$arm.log" | tail -1 | grep -oE '[0-9]+')
  log "  boot OK: pool ${pool:-?}"
  curl -sf -m 10 "$U/health" >/dev/null || { log "  ABORT-ARM: :8029 health not up"; return 1; }
  sudo docker exec "$NAME" sh -c "curl -sf -m 5 $CU/health" >/dev/null 2>&1 \
    || { log "  ABORT-ARM: bench client cannot reach $CU inside $NAME"; return 1; }
  return 0
}

# $1 arm  $2 row  $3 prefix_len  $4 input_len  $5 output_len  $6 nprompts  $7 seed  $8 scored(0/1)
row(){
  local arm=$1 rw=$2 plen=$3 ilen=$4 olen=$5 n=$6 seed=$7 scored=$8
  local out="$R/${arm}-${rw}.txt"
  [ "$scored" = 1 ] && snap "$arm" "$rw" pre
  log "  row $arm/$rw: prefix=$plen input=$ilen output=$olen n=$n conc=20 seed=$seed"
  local t0; t0=$(date +%s.%N)
  sudo docker exec "$NAME" vllm bench serve \
    --backend openai --endpoint /v1/completions --base-url "$CU" \
    --model "$M" --tokenizer /model \
    --dataset-name random --random-prefix-len "$plen" \
    --random-input-len "$ilen" --random-output-len "$olen" \
    --random-range-ratio '{"input":0,"output":0.6}' \
    --num-prompts "$n" --max-concurrency 20 \
    --seed "$seed" --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /tmp --result-filename "${arm}-${rw}.json" \
    > "$out" 2>&1
  local rc=$? t1; t1=$(date +%s.%N)
  echo "$arm,$rw,$t0,$t1,$rc" >> "$R/marks.csv"
  [ "$scored" = 1 ] && snap "$arm" "$rw" post
  sudo docker cp "$NAME:/tmp/${arm}-${rw}.json" "$R/detail-${arm}-${rw}.json" >/dev/null 2>&1 \
    || log "  (no detailed json for $arm/$rw)"
  [ $rc -eq 0 ] || log "  WARN row $arm/$rw exited $rc (see $out)"
  grep -aE "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P90 TTFT|P99 TTFT|Mean ITL|Median ITL|P99 ITL|Mean TPOT|Median TPOT" "$out" \
    | sed 's/^/    /' | tee -a "$R/audit.log"
}

arm_rows(){
  local arm=$1
  row "$arm" plant  24000 8000  16  1 "$SEED_STEADY" 0
  row "$arm" steady 24000 8000 900 60 "$SEED_STEADY" 1
  row "$arm" ramp       0 32000 900 24 "$SEED_RAMP"  1
  log "== ARM $arm done == tier=$(tier_gb) GB"
}

run_arm(){ # $1 arm $2 mnbt $3 seqs
  if boot "$1" "$2" "$3"; then arm_rows "$1"; else log "  SKIPPING rows for $1 (boot failed)"; fi
}

log "### R211b knee: F(4480=3blk) G(5952=4blk) H(8192,SEQS32) A3(8192 anchor) ###"
run_arm F  4480 16
run_arm G  5952 16
run_arm H  8192 32
run_arm A3 8192 16

log "### R211b rows done — restoring the daily ###"
restore_daily
log "### R211b DONE ### results in $R"
