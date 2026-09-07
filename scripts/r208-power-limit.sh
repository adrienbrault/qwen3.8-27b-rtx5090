#!/usr/bin/env bash
# R208 power-limit sweep (2026-09-07, user "Could get me daily perf report when both cards are each power limited to 300w ?").
#
# 300 W IS NOT SETTABLE. Both 5090s report power.min_limit = 400.00 W; nvidia-smi -pl rejects anything below it. The defaults are
# also asymmetric (GPU0 600 W, GPU1 575 W), so a hardcoded restore is wrong — this unit reads power.default_limit per index and
# restores per index. Arms:
#   A0   default limits                      (baseline)
#   A1   -pl 400 on both                     (the lowest real cap the hardware offers)
#   A2   -lgc 0,<mhz> calibrated to ~300 W   (a PROXY: a clock lock is a fixed ceiling, a power cap lets the card boost on light
#                                             work and throttle only on heavy work, so A2 over-constrains decode vs a real 300 W cap)
#   A0b  default limits again                (same-config control, R203 discipline: also says whether A0 was thermally limited)
# Rows per arm, the R207 decode_ss set trimmed to five (code c1/c8/c16, prose c1 @30K, code c1 @100K). The 100K row's ttft_s_median
# is the prefill signal — decode_ss already records TTFT, so no bespoke probe (R156d).
# A continuous nvidia-smi sampler runs for the whole unit and every row writes its epoch bounds to marks.csv; the summary joins them,
# so each cell gets mean/p95 draw per card even on rows the cap never bound.
#
# This measures THE LIVE DAILY on :8020 unmodified — the lever is host-level, not an engine change (R207 gate precedent). The daily
# stays up throughout and is SLOWER while the unit runs; other clients hitting :8020 skew the rows.
# The EXIT trap and ExecStopPost both restore limits+clocks, and the unit verifies the restore before it exits.
#   unit: sudo systemd-run --unit=r208-power-limit --collect -p User=adrienbrault -p RuntimeMaxSec=21600 -p TimeoutStopSec=300 \
#         -p 'ExecStopPost=-/bin/bash /srv/qwen5090/r208-restore-power.sh' \
#         -E GPU_QUEUE_NAME=r208-power-limit bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r208-power-limit.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-07-r208-power-limit; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; M=qwen3.8-27b
for t in decode_ss.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
command -v nvidia-smi >/dev/null || { log "ABORT: nvidia-smi missing"; exit 3; }
curl -sf -m 5 $U/health >/dev/null || { log "ABORT: the daily is not up on :8020 — this unit measures the live daily, it does not boot one"; exit 3; }
curl -sf -m 5 $U/v1/models | grep -q "$M" || { log "ABORT: :8020 does not serve $M"; exit 3; }

# --- per-card defaults, read once, restored unconditionally -------------------------------------------------------------------
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,persistence_mode \
  --format=csv > "$R/gpu-caps.csv"; log "GPU caps:"; cat "$R/gpu-caps.csv" | tee -a "$R/audit.log"
DEF0=$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
DEF1=$(nvidia-smi -i 1 --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
MIN0=$(nvidia-smi -i 0 --query-gpu=power.min_limit --format=csv,noheader,nounits | tr -d ' ')
MIN1=$(nvidia-smi -i 1 --query-gpu=power.min_limit --format=csv,noheader,nounits | tr -d ' ')
[ -n "$DEF0" ] && [ -n "$DEF1" ] || { log "ABORT: could not read power.default_limit"; exit 3; }
FLOOR=$(python3 -c "print(int(max(float('$MIN0'),float('$MIN1'))))")
log "defaults: GPU0 ${DEF0} W, GPU1 ${DEF1} W; floor ${MIN0}/${MIN1} W -> A1 caps both at ${FLOOR} W (300 W is below the floor and cannot be set)"
# ship the same restore as a standalone script so systemd ExecStopPost can run it after a SIGKILL
cat > /srv/qwen5090/r208-restore-power.sh <<EOR
#!/usr/bin/env bash
sudo -n nvidia-smi -rgc >/dev/null 2>&1
sudo -n nvidia-smi -i 0 -pl $DEF0 >/dev/null 2>&1
sudo -n nvidia-smi -i 1 -pl $DEF1 >/dev/null 2>&1
echo "\$(date -Is) ExecStopPost restore: \$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\\n' ' ')" >> $R/restore-verify.log
EOR
chmod +x /srv/qwen5090/r208-restore-power.sh

restore(){ sudo nvidia-smi -rgc >/dev/null 2>&1; sudo nvidia-smi -i 0 -pl "$DEF0" >/dev/null 2>&1; sudo nvidia-smi -i 1 -pl "$DEF1" >/dev/null 2>&1
           local now; now=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' '); log "RESTORED limits: $now"; }
SAMP=""; LOAD=""
cleanup(){ [ -n "$LOAD" ] && kill "$LOAD" 2>/dev/null; pkill -f "decode_ss.py --url $U --model $M --conc 8 --tokens 2048" 2>/dev/null
           [ -n "$SAMP" ] && kill "$SAMP" 2>/dev/null; restore; }
trap 'log "### SIGTERM/EXIT ###"; cleanup; exit 4' TERM INT
trap 'cleanup' EXIT

exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R208 power-limit sweep start (lock held; the daily stays UP on :8020) ==="

# --- continuous power sampler + row marks -------------------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$R/power.csv" 2>"$R/power.err" & SAMP=$!
sleep 2; [ -s "$R/power.csv" ] || { log "ABORT: power sampler produced nothing"; exit 3; }
echo "arm,row,t_start,t_end" > "$R/marks.csv"

row(){ # row <arm> <name> <decode_ss args...>
  local arm="$1" name="$2"; shift 2
  local t0 t1; t0=$(date +%s.%N)
  python3 "$PR/decode_ss.py" --url $U --model $M "$@" --out "$R/decode-$arm-$name.jsonl" \
    > "$R/probe-$arm-$name.out" 2> "$R/probe-$arm-$name.err"
  t1=$(date +%s.%N); echo "$arm,$name,$t0,$t1" >> "$R/marks.csv"
  log "  $arm/$name: $(grep -aoE '"(ss_agg_tps_median|ss_per_stream_tps_median|ttft_s_median|accept_per_draft_median)": *[0-9.]+' "$R/probe-$arm-$name.out" | tr '\n' ' ')"
}
arm_rows(){ local arm="$1"
  log "--- arm $arm rows ---"
  row "$arm" code-c1      --conc 1  --tokens 1024 --runs 3 --kind code
  row "$arm" code-c8      --conc 8  --tokens 1024 --runs 2 --kind code
  row "$arm" code-c16     --conc 16 --tokens 1024 --runs 2 --kind code
  row "$arm" prose-c1-30k --conc 1  --tokens 1024 --runs 2 --kind prose --ctx 30000
  row "$arm" code-c1-100k --conc 1  --tokens 512  --runs 2 --kind code  --ctx 100000
}

# --- A0 baseline --------------------------------------------------------------------------------------------------------------
restore; sleep 20; arm_rows A0-default

# --- A1: the 400 W floor ------------------------------------------------------------------------------------------------------
sudo nvidia-smi -i 0 -pl "$FLOOR" 2>&1 | tee -a "$R/audit.log" >/dev/null
sudo nvidia-smi -i 1 -pl "$FLOOR" 2>&1 | tee -a "$R/audit.log" >/dev/null
log "A1 limits now: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
sleep 20; arm_rows A1-pl${FLOOR}
restore

# --- A2: clock lock calibrated to ~300 W --------------------------------------------------------------------------------------
log "--- A2 calibration: stepping the SM clock under a c8 code load, picking the step whose hotter card sits nearest 300 W ---"
( for i in 1 2 3 4 5 6 7 8; do python3 "$PR/decode_ss.py" --url $U --model $M --conc 8 --tokens 2048 --runs 1 --kind code \
    --out /dev/null >/dev/null 2>&1; done ) & LOAD=$!
sleep 20
BEST=""; BESTD=99999
for c in 2400 2100 1800 1500 1200; do
  sudo nvidia-smi -lgc 0,$c >/dev/null 2>&1 || { log "  lgc $c rejected"; continue; }
  sleep 8
  read -r p0 p1 <<< "$(timeout 12 nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 250 \
    | awk -F', *' '{s[$1]+=$2; n[$1]++} END{if(n[0]&&n[1]) printf "%.1f %.1f", s[0]/n[0], s[1]/n[1]}')"
  [ -n "${p1:-}" ] || { log "  lgc $c: no power samples, skipped"; continue; }
  hi=$(python3 -c "print(max($p0,$p1))"); d=$(python3 -c "print(abs($hi-300))")
  log "  lgc $c MHz -> GPU0 ${p0} W, GPU1 ${p1} W (hotter ${hi} W)"
  echo "$c,$p0,$p1,$hi" >> "$R/calib.csv"
  keep=$(python3 -c "print(1 if $d < $BESTD else 0)"); [ "$keep" = 1 ] && { BEST=$c; BESTD=$d; }
done
kill "$LOAD" 2>/dev/null; LOAD=""; pkill -f "decode_ss.py --url $U --model $M --conc 8 --tokens 2048" 2>/dev/null; sleep 20
if [ -n "$BEST" ]; then
  sudo nvidia-smi -lgc 0,$BEST >/dev/null 2>&1
  log "A2 clock lock: 0,$BEST MHz (nearest 300 W of the steps tried; PROXY for a 300 W cap, not the same lever)"
  arm_rows A2-lgc${BEST}
  sudo nvidia-smi -rgc >/dev/null 2>&1
else
  log "A2 SKIPPED: no clock step could be applied"
fi

# --- A0b control --------------------------------------------------------------------------------------------------------------
restore; sleep 30; arm_rows A0b-default

kill "$SAMP" 2>/dev/null; SAMP=""; sleep 1
restore
nvidia-smi --query-gpu=index,power.limit,power.default_limit,clocks.max.sm --format=csv | tee -a "$R/audit.log"

# --- summary ------------------------------------------------------------------------------------------------------------------
python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import csv, json, os, re, sys, datetime
R = sys.argv[1]
samples = []
with open(f"{R}/power.csv") as f:
    for row in csv.reader(f):
        if len(row) < 7: continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append((t, int(row[1]), float(row[2]), float(row[3])))
        except Exception: pass
def win(a, b, idx):
    v = [(p, c) for (t, i, p, c) in samples if a <= t <= b and i == idx]
    if not v: return None
    ps = sorted(p for p, _ in v); cs = [c for _, c in v]
    return dict(mean=round(sum(ps)/len(ps), 1), p95=round(ps[int(0.95*(len(ps)-1))], 1),
                max=round(ps[-1], 1), sm_mean=round(sum(cs)/len(cs)))
out = []
with open(f"{R}/marks.csv") as f:
    for m in csv.DictReader(f):
        a, b = float(m["t_start"]), float(m["t_end"])
        rec = {"arm": m["arm"], "row": m["row"], "dur_s": round(b-a, 1),
               "gpu0": win(a, b, 0), "gpu1": win(a, b, 1)}
        p = f"{R}/probe-{m['arm']}-{m['row']}.out"
        if os.path.exists(p):
            txt = open(p, errors="ignore").read()
            for k in ("ss_agg_tps_median", "ss_per_stream_tps_median", "ttft_s_median", "accept_per_draft_median"):
                mm = re.search(rf'"{k}"\s*:\s*([0-9.]+)', txt)
                if mm: rec[k] = float(mm.group(1))
        out.append(rec)
json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
hdr = f"{'arm':<14}{'row':<14}{'agg tps':>9}{'per-str':>9}{'ttft s':>9}{'acc':>6}   {'GPU0 mean/p95 W':>17} {'GPU1 mean/p95 W':>17}  sm MHz"
print(hdr); print("-"*len(hdr))
for r in out:
    g0, g1 = r["gpu0"] or {}, r["gpu1"] or {}
    print(f"{r['arm']:<14}{r['row']:<14}{r.get('ss_agg_tps_median',0):>9.1f}{r.get('ss_per_stream_tps_median',0):>9.1f}"
          f"{r.get('ttft_s_median',0):>9.2f}{r.get('accept_per_draft_median',0):>6.2f}   "
          f"{g0.get('mean',0):>8.0f}/{g0.get('p95',0):<8.0f} {g1.get('mean',0):>8.0f}/{g1.get('p95',0):<8.0f}  "
          f"{g0.get('sm_mean',0)}/{g1.get('sm_mean',0)}")
PY
log "=== R208 done; results in $R ==="
