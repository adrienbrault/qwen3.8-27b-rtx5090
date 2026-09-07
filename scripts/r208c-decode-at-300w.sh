#!/usr/bin/env bash
# R208c (2026-09-07): decode cost of the ~300 W clock. R208 measured decode at a 2400 MHz lock, but R208b's prefill calibration
# showed 2400 MHz draws 361 W prefilling — the step that actually lands near 300 W under prefill is 2100 MHz. Clock->power is
# workload-dependent (2100 MHz: ~318 W prefill, ~225 W decode; 2400 MHz: ~361 W prefill, ~250 W decode), so no single clock is
# "300 W" for both. This unit measures the DECODE rows at 2100 MHz against a same-unit baseline, so the ~300 W configuration has
# a cost on both halves from one coherent setting.
#   unit: sudo systemd-run --unit=r208c-decode-at-300w --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=300 \
#         -p 'ExecStopPost=-/bin/bash /srv/qwen5090/r208-restore-power.sh' \
#         -E GPU_QUEUE_NAME=r208c-decode-at-300w bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r208c-decode-at-300w.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-07-r208c-decode-at-300w; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; M=qwen3.8-27b; LGC=2100
grep -q -- "--seed-prefix" "$PR/decode_ss.py" || { log "ABORT: decode_ss.py has no --seed-prefix"; exit 3; }
curl -sf -m 5 $U/health >/dev/null || { log "ABORT: the daily is not up on :8020"; exit 3; }
DEF0=$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
DEF1=$(nvidia-smi -i 1 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
restore(){ sudo nvidia-smi -rgc >/dev/null 2>&1; sudo nvidia-smi -i 0 -pl "$DEF0" >/dev/null 2>&1; sudo nvidia-smi -i 1 -pl "$DEF1" >/dev/null 2>&1
           log "RESTORED: $(nvidia-smi --query-gpu=index,power.limit,clocks.max.sm --format=csv,noheader | tr '\n' ' ')"; }
SAMP=""
cleanup(){ [ -n "$SAMP" ] && kill "$SAMP" 2>/dev/null; restore; }
trap 'log "### SIGTERM/EXIT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock"; flock 9; }
log "=== R208c decode at ${LGC} MHz start (daily stays UP) ==="
nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$R/power.csv" 2>"$R/power.err" & SAMP=$!
sleep 2; echo "arm,row,t_start,t_end" > "$R/marks.csv"
row(){ local arm="$1" name="$2"; shift 2
  local t0 t1; t0=$(date +%s.%N)
  python3 "$PR/decode_ss.py" --url $U --model $M --seed-prefix "$arm-" "$@" --out "$R/decode-$arm-$name.jsonl" \
    > "$R/probe-$arm-$name.out" 2> "$R/probe-$arm-$name.err"
  t1=$(date +%s.%N); echo "$arm,$name,$t0,$t1" >> "$R/marks.csv"
  log "  $arm/$name: $(grep -aoE '"(ss_agg_tps_median|ttft_s_median|accept_per_draft_median)": *[0-9.]+' "$R/probe-$arm-$name.out" | tr '\n' ' ')"
}
arm_rows(){ local arm="$1"; log "--- arm $arm ---"
  row "$arm" code-c1      --conc 1  --tokens 1024 --runs 3 --kind code
  row "$arm" code-c8      --conc 8  --tokens 1024 --runs 2 --kind code
  row "$arm" code-c16     --conc 16 --tokens 1024 --runs 2 --kind code
  row "$arm" prose-c1-30k --conc 1  --tokens 1024 --runs 2 --kind prose --ctx 30000
}
restore; sleep 20; arm_rows A0-default
sudo nvidia-smi -lgc 0,$LGC >/dev/null 2>&1 && { log "locked 0,$LGC MHz"; sleep 15; arm_rows A2-lgc${LGC}; } || log "ABORT-ish: lgc $LGC rejected"
sudo nvidia-smi -rgc >/dev/null 2>&1; restore; sleep 25; arm_rows A0b-default
kill "$SAMP" 2>/dev/null; SAMP=""; sleep 1; restore
python3 /srv/qwen5090/r208-power-summary.py "$R" 2>&1 | tee -a "$R/audit.log"
log "=== R208c done ==="
