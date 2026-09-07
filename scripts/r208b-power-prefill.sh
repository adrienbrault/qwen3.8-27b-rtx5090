#!/usr/bin/env bash
# R208b prefill under the power lever (2026-09-07). R208 measured DECODE cleanly but its prefill signal was void: decode_ss seeds
# prompts deterministically (seed=f"{c}-{i}"), so arm 2 re-sent arm 1's bytes and hit the prefix cache — 100K TTFT read 16.58 s on
# A0 and 0.93 s on A1. Decode is unaffected (steady state is measured after prefill), so only prefill needs re-running.
# Fix: probes/decode_ss.py now takes --seed-prefix (default "" = every historical invocation byte-identical); this unit passes the
# arm name, so no two arms — and no two runs within an arm — share a prompt.
# Prefill is compute-bound, which is where a power cap should bite hardest; R208 showed decode barely notices the 400 W floor.
# Arms mirror R208 (default -> 400 W floor -> the calibrated ~300 W clock lock from R208's calib.csv -> default control).
#   unit: sudo systemd-run --unit=r208b-power-prefill --collect -p User=adrienbrault -p RuntimeMaxSec=21600 -p TimeoutStopSec=300 \
#         -p 'ExecStopPost=-/bin/bash /srv/qwen5090/r208-restore-power.sh' \
#         -E GPU_QUEUE_NAME=r208b-power-prefill bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r208b-power-prefill.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-07-r208b-power-prefill; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; M=qwen3.8-27b
[ -f "$PR/decode_ss.py" ] || { log "ABORT: decode_ss.py missing"; exit 3; }
grep -q -- "--seed-prefix" "$PR/decode_ss.py" || { log "ABORT: decode_ss.py has no --seed-prefix; arms would share prompts and the prefill signal would be void again"; exit 3; }
curl -sf -m 5 $U/health >/dev/null || { log "ABORT: the daily is not up on :8020"; exit 3; }
curl -sf -m 5 $U/v1/models | grep -q "$M" || { log "ABORT: :8020 does not serve $M"; exit 3; }
DEF0=$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
DEF1=$(nvidia-smi -i 1 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
FLOOR=$(nvidia-smi -i 0 --query-gpu=power.min_limit --format=csv,noheader,nounits | tr -d ' ' | cut -d. -f1)
log "defaults GPU0 ${DEF0} W / GPU1 ${DEF1} W; floor ${FLOOR} W"
# A2's clock is calibrated HERE, under a prefill load, not reused from R208. R208 calibrated under c8 DECODE, which draws only
# 250 W even at a 2400 MHz lock — decode never approaches 300 W on this box, so a decode-calibrated clock cannot land on a 300 W
# prefill. Prefill is the only workload that exceeds 400 W (R208 A0: 480/457 W mean, 524/511 W p95 at 100K), so it is the load
# a "what would 300 W cost" proxy has to be calibrated against.

restore(){ sudo nvidia-smi -rgc >/dev/null 2>&1; sudo nvidia-smi -i 0 -pl "$DEF0" >/dev/null 2>&1; sudo nvidia-smi -i 1 -pl "$DEF1" >/dev/null 2>&1
           log "RESTORED limits: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"; }
SAMP=""; LOAD=""
cleanup(){ [ -n "$LOAD" ] && kill "$LOAD" 2>/dev/null; pkill -f "decode_ss.py --url $U --model $M --conc 1 --ctx 100000" 2>/dev/null
           [ -n "$SAMP" ] && kill "$SAMP" 2>/dev/null; restore; }
trap 'log "### SIGTERM/EXIT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock"; flock 9; }
log "=== R208b prefill sweep start (lock held; the daily stays UP) ==="

nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$R/power.csv" 2>"$R/power.err" & SAMP=$!
sleep 2; [ -s "$R/power.csv" ] || { log "ABORT: power sampler produced nothing"; exit 3; }
echo "arm,row,t_start,t_end" > "$R/marks.csv"

row(){ local arm="$1" name="$2"; shift 2
  local t0 t1; t0=$(date +%s.%N)
  python3 "$PR/decode_ss.py" --url $U --model $M --seed-prefix "$arm-" "$@" --out "$R/decode-$arm-$name.jsonl" \
    > "$R/probe-$arm-$name.out" 2> "$R/probe-$arm-$name.err"
  t1=$(date +%s.%N); echo "$arm,$name,$t0,$t1" >> "$R/marks.csv"
  log "  $arm/$name: $(grep -aoE '"(ttft_s_median|ss_agg_tps_median)": *[0-9.]+' "$R/probe-$arm-$name.out" | tr '\n' ' ')"
}
arm_rows(){ local arm="$1"; log "--- arm $arm ---"
  row "$arm" pf-c1-30k  --conc 1 --ctx 30000  --tokens 512 --runs 2 --kind code
  row "$arm" pf-c1-100k --conc 1 --ctx 100000 --tokens 512 --runs 2 --kind code
  row "$arm" pf-c8-30k  --conc 8 --ctx 30000  --tokens 512 --runs 2 --kind code
}

restore; sleep 20; arm_rows A0-default
sudo nvidia-smi -i 0 -pl "$FLOOR" >/dev/null 2>&1; sudo nvidia-smi -i 1 -pl "$FLOOR" >/dev/null 2>&1
log "A1 limits now: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
sleep 20; arm_rows A1-pl${FLOOR}
restore
log "--- A2 calibration: stepping the SM clock under a 100K PREFILL load, picking the step whose hotter card sits nearest 300 W ---"
( i=0; while [ $i -lt 40 ]; do i=$((i+1)); python3 "$PR/decode_ss.py" --url $U --model $M --conc 1 --ctx 100000 --tokens 64 \
    --runs 1 --kind code --seed-prefix "calib$i-" --out /dev/null >/dev/null 2>&1; done ) & LOAD=$!
sleep 25
LGC=""; BESTD=99999
for c in 3000 2700 2400 2100 1800; do
  sudo nvidia-smi -lgc 0,$c >/dev/null 2>&1 || { log "  lgc $c rejected"; continue; }
  sleep 8
  read -r p0 p1 <<< "$(timeout 14 nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 250 \
    | awk -F', *' '$2>150{s[$1]+=$2; n[$1]++} END{if(n[0]&&n[1]) printf "%.1f %.1f", s[0]/n[0], s[1]/n[1]}')"
  [ -n "${p1:-}" ] || { log "  lgc $c: no loaded power samples, skipped"; continue; }
  hi=$(python3 -c "print(max($p0,$p1))"); d=$(python3 -c "print(abs($hi-300))")
  log "  lgc $c MHz under prefill -> GPU0 ${p0} W, GPU1 ${p1} W (hotter ${hi} W)"
  echo "$c,$p0,$p1,$hi" >> "$R/calib.csv"
  [ "$(python3 -c "print(1 if $d < $BESTD else 0)")" = 1 ] && { LGC=$c; BESTD=$d; }
done
kill "$LOAD" 2>/dev/null; LOAD=""; pkill -f "decode_ss.py --url $U --model $M --conc 1 --ctx 100000" 2>/dev/null; sleep 25
if [ -n "${LGC:-}" ]; then
  sudo nvidia-smi -lgc 0,$LGC >/dev/null 2>&1 && { log "A2 clock lock 0,$LGC MHz (hotter card ${BESTD} W from 300 under prefill)"; sleep 10
    arm_rows A2-lgc${LGC}; sudo nvidia-smi -rgc >/dev/null 2>&1; } || log "A2 SKIPPED: lgc $LGC rejected"
else
  log "A2 SKIPPED: calibration produced no usable step"
fi
restore; sleep 30; arm_rows A0b-default
kill "$SAMP" 2>/dev/null; SAMP=""; sleep 1; restore
python3 /srv/qwen5090/r208-power-summary.py "$R" 2>&1 | tee -a "$R/audit.log"
log "=== R208b done; results in $R ==="
