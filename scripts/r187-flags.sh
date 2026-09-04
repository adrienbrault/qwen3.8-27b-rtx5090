#!/usr/bin/env bash
# R187 (2026-09-05): the flag/env-level arms of the R186 program analysis, on the DAILY image (fi0616) so every number is
# comparable with the R183 replicate band (c1 285–308, c8 1,327–1,421, c16 1,738–1,804) and with R185's arms.
#   BASE-a   served flags; nvidia-smi clocks/power/throttle sampled at 1 Hz through the battery (R186 #11)      -> band check + drift anchor
#   OMP1/2   OMP_NUM_THREADS=1 / 2 inside the container (R186 #5, CPU contention; the container default is logged)
#   CPUSET   --cpuset-cpus 0-7 = one SMT thread per physical core (siblings are N,N+8 on the 9800X3D; R186 #5). Rides on
#            EXTRA_MOUNT_APPEND, which launch-daily splices raw into `docker run` — a documented abuse of that knob.
#   BSS      --enable-batch-sharded-sampling (R186 #2 first step); if it refuses to coexist with DFlash, BOOT FAILED + reason is the result
#   M4096 / M12288  EXP_MNBT prefill chunk (R186 §4; 8192 served, 16K/32K failed in R183) with the ttft ladder + c8 + dense ruler
#   BASE-b   served flags again at the end -> within-unit drift; an arm is a result only if it clears BOTH bases
#   PROF-deep torch profiler at 30K and 100K single-stream decode (R186 #9 step 1: does attention become a lever at depth?)
# Every arm: bat_decode (decode_ss code c1×2 / prose c1×2 / code c8×2 / code c16×1) + decode ruler ctx0 vs bf16 + smi summary;
# BASE/MNBT arms add ttft 8k/36k/120k; MNBT arms add the dense ruler. Queued behind miniswe-r183 (GPU_QUEUE_NAME registers it so
# the campaign's finish() skips the daily restore); finish restores the daily unless another unit is queued (R188 arms will be).
#   unit: sudo systemd-run --unit=r187-flags --collect -p User=adrienbrault -p RuntimeMaxSec=64800 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r187-flags bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q miniswe-r183; do sleep 30; done; exec bash /srv/qwen5090/r187-flags.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r187-flags; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
PROF_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true --profiler-config.delay_iterations=50 --profiler-config.max_iterations=60"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
grep -q "$IMG" "$CAND" || { log "ABORT: $IMG is not launch-daily.sh's DAILY_IMG any more — R187 must run on the daily image"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: references missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in prof_summary.py prof_cpu.py prof_decode_split.py decode_ss.py kv_capacity_probe.py decode_fidelity.py fidelity_ladder.py fidelity_compare.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
SMI_PID=
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ [ -n "$SMI_PID" ] && { kill "$SMI_PID" 2>/dev/null; SMI_PID=; }; for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R187 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R187 start (lock held): flag/env arms on $IMG — BASE-a, OMP1, OMP2, CPUSET, BSS, M4096, M12288, BASE-b, PROF-deep ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
layout(){ curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' '; }
boot_arm(){ local tag=$1 pins=$2 kv rc; shift 2
  for kv in $pins; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB $(layout)"
      ELOG > "$R/engine-boot-$tag.log"
      # proof of path: what the container actually got (OMP default, visible CPUs, cpuset), and the resolved engine args the arm touches
      log "[$tag cpu] $(sudo docker exec vllm-exp bash -c 'echo OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} nproc=$(nproc) cpuset=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo n/a)' 2>&1 | tail -1)"
      log "[$tag args] $(grep -aoE 'max_num_batched_tokens=[0-9]+|batch_sharded_sampling=[A-Za-z]+|enable_batch_sharded_sampling=[A-Za-z]+|Using [A-Za-z0-9]+ for NVFP4 GEMM' "$R/engine-boot-$tag.log" | sort -u | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|unsupported|not supported|not compatible|JointFailure|ValueError" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
# nvidia-smi sampler (R186 #11): 1 Hz through an arm's battery; summary = per GPU mean/min SM clock (all samples and samples with
# power > 150 W, i.e. the engine working), memory clock, mean/max power, max temp, and the throttle-reason bitmasks seen.
smi_start(){ nvidia-smi --query-gpu=index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -l 1 > "$R/smi-$1.csv" 2>/dev/null & SMI_PID=$!; }
smi_stop(){ local tag=$1; [ -n "$SMI_PID" ] && { kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null; SMI_PID=; }
  log "[$tag smi] $(python3 - "$R/smi-$tag.csv" <<'PY'
import sys, csv, collections
d = collections.defaultdict(list)
for row in csv.reader(open(sys.argv[1])):
    if len(row) < 7: continue
    try: d[row[0].strip()].append((float(row[2]), float(row[3]), float(row[4]), float(row[5]), row[6].strip()))
    except ValueError: pass
out = []
for g in sorted(d):
    v = d[g]; sm = [x[0] for x in v]; pw = [x[2] for x in v]; tp = [x[3] for x in v]
    busy = [x[0] for x in v if x[2] > 150] or sm
    rs = collections.Counter(x[4] for x in v)
    out.append(f"gpu{g} n={len(v)} sm_mean={sum(sm)/len(sm):.0f} sm_busy_mean={sum(busy)/len(busy):.0f} sm_busy_min={min(busy):.0f} mem={v[0][1]:.0f} pw_mean={sum(pw)/len(pw):.0f} pw_max={max(pw):.0f} temp_max={max(tp):.0f} reasons={dict(rs.most_common(3))}")
print(' | '.join(out) if out else 'no samples')
PY
)"; }
bat_decode(){ local T=$1
  p1 $T code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
  p1 $T prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $T code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $T code-c16 --conc 16 --tokens 1024 --runs 1 --kind code; }
ttft_ladder(){ local T=$1
  cap $T ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41
  cap $T ttft36k --ctx 36000 --conc 1 --tokens 8 --seed 42
  cap $T ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43; }
# arm TAG EXTRAS [ENV=VAL ...]   EXTRAS: substrings "ttft" and/or "dense" select the extra probes
arm(){ local tag=$1 extras=$2; shift 2
  teardown; wipe_l2
  if boot_arm "$tag" "$PINS" "$@"; then
    sleep 20; smi_start "$tag"
    bat_decode "$tag"
    case "$extras" in *ttft*) ttft_ladder "$tag";; esac
    dfid "$tag" 0
    case "$extras" in *dense*) ruler_dense "$tag";; esac
    smi_stop "$tag"
    log "[$tag engine error-lines] $(errs)"
  else log "[$tag] BOOT FAILED on every pin"; fi; }

arm BASE-a ttft
arm OMP1   ""   EXTRA_ENV_APPEND="-e OMP_NUM_THREADS=1"
arm OMP2   ""   EXTRA_ENV_APPEND="-e OMP_NUM_THREADS=2"
arm CPUSET ""   EXTRA_MOUNT_APPEND="--cpuset-cpus 0-7"
arm BSS    ""   EXTRA_ARGS_APPEND="--enable-batch-sharded-sampling"
arm M4096  ttft+dense EXP_MNBT=4096
arm M12288 ttft+dense EXP_MNBT=12288
arm BASE-b ttft

# ---------- PROF-deep: torch profiler around single-stream decode at 30K and 100K ----------
# delay_iterations=50 skips the prompt's prefill chunks (4 at 30K, 13 at 100K with the 8192 chunk) and the first decode steps;
# max_iterations=60 then captures 60 decode steps; decode_ss --tokens 512 gives ~300 steps so the window lands inside decode.
# Reader check (2026-09-05): decode_ss sends no warm-up request, so the window lands in the measured decode; still, attention time per
# step MUST grow from ctx30000 to ctx100000 — near-identical summaries mean the window missed and the PROF-deep arm is void.
teardown; wipe_l2; mkdir -p "$R/prof-deep"; chmod 777 "$R/prof-deep"
capture_deep(){ local ctx=$1; local d="$R/prof-deep/ctx$ctx"; mkdir -p "$d"
  curl -s -m 10 -X POST $U/start_profile >/dev/null 2>&1 || log "[prof ctx$ctx] start_profile returned non-2xx (continuing)"
  p1 PROF-deep prof-ctx$ctx --conc 1 --tokens 512 --runs 1 --kind prose --ctx "$ctx"
  curl -s -m 60 -X POST $U/stop_profile >/dev/null 2>&1 || true
  for i in $(seq 60); do n=$(ls "$R/prof-deep"/*.json* 2>/dev/null | wc -l); [ "$n" -ge 2 ] && break; sleep 3; done
  sleep 15; mv "$R/prof-deep"/*.json* "$d/" 2>/dev/null
  ls -la "$d" | grep -v "^total" | sed "s/^/[prof ctx$ctx] /" | tee -a "$R/audit.log"
  for t in "$d"/*.json*; do [ -f "$t" ] || continue
    python3 $PR/prof_summary.py "$t" --steps 60 --top 22 > "$t.summary.txt" 2>&1; head -18 "$t.summary.txt" | sed "s/^/[prof ctx$ctx] /" | tee -a "$R/audit.log"
    python3 $PR/prof_decode_split.py "$t" > "$t.split.txt" 2>&1; head -14 "$t.split.txt" | sed "s/^/[prof-split ctx$ctx] /" | tee -a "$R/audit.log"
    python3 $PR/prof_cpu.py "$t" --steps 60 > "$t.cpu.txt" 2>&1; head -8 "$t.cpu.txt" | sed "s/^/[prof-cpu ctx$ctx] /" | tee -a "$R/audit.log"
  done; }
if boot_arm PROF-deep "$PINS" EXTRA_ARGS_APPEND="$PROF_ARGS" EXTRA_MOUNT_APPEND="-v $R/prof-deep:/prof"; then
  sleep 20
  capture_deep 30000
  capture_deep 100000
  log "[PROF-deep engine error-lines] $(errs)"
else log "[PROF-deep] BOOT FAILED on every pin"; fi
grep -aE "BOOT OK|BOOT FAILED|boot-err| cpu\]| args\]|RESULT|PROBE FAILED|cap |decode ctx|vs bf16|smi\]|error-lines|prof" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
