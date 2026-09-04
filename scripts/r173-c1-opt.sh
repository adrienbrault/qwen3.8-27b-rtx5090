#!/usr/bin/env bash
# R173 (2026-09-04, user: "do the opt tweaks i approved earlier on the pending future daily"): the c1 tuning chain on the 0.29
# nvfp4 candidate, run on the attention path r168e elects (PICK=auto reads r168e's bf16 sheet: ON = rc2 + split-KV knob,
# S = rc2 + FlashInfer 0.6.16.post3 swap; OFF cannot win — it decodes 30K at 25 tok/s). Levers, in expected-value order:
#   1. BASE boot (ns9, draft_tp2): the first daily-shaped boot of the winner — pool, free VRAM, cold needles 131K/220K,
#      cudagraph-mode lines from the boot log; then CPU pinning as A/B/A ON THE SAME BOOT (unpinned → pinned → unpinned):
#      no vLLM affinity env exists, so it is host taskset on the three hot processes (Worker_TP0/TP1/EngineCore each get a
#      physical core + its SMT sibling; the api_server gets the rest). A torch-profiler capture of ~32 decode steps at 30K
#      in both states gives the rank0/rank1 kernel table (allreduce-wait asymmetry = the pinning signal).
#   2. ns7 and ns11 (ns9 = BASE) at ctx0 and 30K: acceptance per draft drops with depth, so the best length may differ.
#   3. draft_tp=1 last (prior evidence loses: fp8 c1 -11%, R156; ng 270.7 vs ng2 276.9).
#   Dropped: an explicit full-graph arm — steady-state verify steps already replay FULL graphs on this route (FINDINGS census);
#   the BASE boot log records the effective cudagraph mode so that claim is re-checked on this image rather than assumed.
# Per measurement block: decode_ss prose c1 / code c1 (ctx0), prose c1 @30K, code c8 (steady-state tok/s + accept-per-draft
# from /metrics), llama-benchy c1 pp2048/tg256 (tokenizer-counted). Tokens/step = accepted/drafts + 1 from the counter deltas.
# Any arm below 1 GB free VRAM is disqualified as a daily config regardless of speed (r169 arm D OOM under logprobs).
# Gate: waits for r168e's verdict BEFORE touching the GPU lock (flock is not FIFO). r168e ABORTED → PICK=ON (r169's verdict).
# Unit: sudo systemd-run --unit=r173-c1-opt --collect -p User=adrienbrault -p RuntimeMaxSec=18000 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r173-c1-opt bash /srv/qwen5090/r173-c1-opt.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r173-c1-opt; mkdir -p "$R/prof"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
R168E=/srv/qwen5090/results/2026-09-04-r168e-splitkv-bf16
RC2_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs
RC2FI16_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh; L2=/srv/qwen5090/eval-l2
PROF="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof"
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
# --- gate on r168e (before the lock) ---
log "=== R173 queued: waiting for r168e's verdict (no lock held) ==="
until grep -aqE "=== R168e (DONE|ABORTED) ===" "$R168E/audit.log" 2>/dev/null; do sleep 30; done
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R173 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
PICK=${PICK:-auto}
# pre-lock abort: r168e's finish skipped the daily restore because this unit was queued, so if nobody else is queued, bring it back
abort_prelock(){ log "ABORT: $1"; if [ -z "$(gpu_queue_others)" ]; then log "no other unit queued: taking the lock to restore the daily"; flock 9; HAVE_LOCK=1; finish ABORTED; fi; exit 3; }
top1(){ python3 -c 'import json,sys; print(round(100*json.load(open(sys.argv[1]))["top1_agreement"],3))' "$1" 2>/dev/null; }
if [ "$PICK" = auto ]; then
  if grep -aq "=== R168e ABORTED ===" "$R168E/audit.log"; then log "r168e ABORTED — falling back to PICK=ON (r169's bf16 verdict on the same config)"; PICK=ON
  else on=$(top1 "$R168E/bf16-ON.json"); s=$(top1 "$R168E/bf16-S.json")
    [ -n "$on" ] && [ -n "$s" ] || abort_prelock "r168e finished but bf16-ON.json / bf16-S.json unreadable (ON='$on' S='$s') — set PICK=ON|S"
    # dense top-1 vs bf16 decides; the floor is exact (r168d) so any gap is real. Tie within 0.05 pp → S (r168d: S is 4 pp closer to F at 160K–262K).
    PICK=$(python3 -c "import sys; on,s=float(sys.argv[1]),float(sys.argv[2]); print('ON' if on-s>0.05 else 'S')" "$on" "$s")
    log "r168e sheet: dense top-1 vs bf16 ON=$on% S=$s% → PICK=$PICK"; fi
fi
case "$PICK" in ON) IMG=$RC2_IMG; SPLIT=1;; S) IMG=$RC2FI16_IMG; SPLIT=0;; *) abort_prelock "PICK=$PICK is not ON|S";; esac
sudo docker image inspect "$IMG" >/dev/null 2>&1 || abort_prelock "image $IMG missing (no fallback: S needs the rc2+fi0616 build)"
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R173 c1 tuning start (lock held): PICK=$PICK image=$IMG split_kv=$SPLIT ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
declare -A POOL FREE
boot_cand(){ local tag=$1 ns=$2 dtp=$3 kv rc; mkdir -p "$R/prof/$tag"
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$SPLIT CAND_IMG=$IMG SPEC_NS=$ns SPEC_DTP=$dtp ${kv:+KV_BYTES=$kv} \
      EXTRA_ARGS_APPEND="$PROF" EXTRA_MOUNT_APPEND="-v $R/prof/$tag:/prof" bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      POOL[$tag]=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag.log" | tail -1 | tr -dc 0-9); FREE[$tag]=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag.log" | tail -1 | tr -dc 0-9)
      log "[$tag] BOOT OK ns$ns draft_tp$dtp pin=${kv:-default} pool=${POOL[$tag]} min_free=${FREE[$tag]}MiB flashinfer=$(sudo docker exec vllm-exp python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null)"
      sudo docker logs vllm-exp 2>&1 | grep -aiE "cudagraph_mode|CUDAGraphMode|cudagraph.*(FULL|PIECEWISE|not supported)|capturing" | cut -c1-200 | sort -u | head -12 > "$R/cudagraph-$tag.txt"
      log "[$tag] cudagraph lines: $(wc -l < "$R/cudagraph-$tag.txt"); mode: $(grep -aoiE 'cudagraph_mode[^,)]{0,60}' "$R/cudagraph-$tag.txt" | sort -u | tr '\n' ' ' | cut -c1-200)"
      [ "${FREE[$tag]:-0}" -ge 1024 ] || log "[$tag] NOTE: only ${FREE[$tag]} MiB free — below the 1 GB daily floor (r169 arm D OOM); speed still measured"
      return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || break; teardown
  done; return 1; }
boxstate(){ sudo docker logs vllm-exp 2>&1 | grep -aq "CUDA error: invalid argument" || grep -aq "CUDA error: invalid argument" "$R/boot-$1.log" 2>/dev/null; }
boot_retry(){ local tag=$1; shift; "$@" && return 0
  if boxstate "$tag"; then log "[$tag] box-state boot failure (invalid argument in warm-up): teardown + 300 s idle, retrying once"; teardown; settle 300; "$@" && return 0; fi
  return 1; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
# --- CPU pinning (host taskset; 9800X3D: physical 0-7, SMT sibling = core+8) ---
pids(){ W0=$(pgrep -f '^VLLM::Worker_TP0$' | head -1); W1=$(pgrep -f '^VLLM::Worker_TP1$' | head -1); EC=$(pgrep -f '^VLLM::EngineCore$' | head -1); API=$(pgrep -f '^python3 -m vllm.entrypoints.openai.api_server' | head -1)
  [ -n "$W0" ] && [ -n "$W1" ] && [ -n "$EC" ] && [ -n "$API" ]; }
psr_sample(){ local t; for t in 1 2 3; do ps -o pid=,psr=,comm= -p "$W0,$W1,$EC,$API" 2>/dev/null | tr -s ' ' | tr '\n' ';'; sleep 1; done; }
pin(){ local mode=$1; pids || { log "[pin] engine PIDs not found"; return 1; }
  if [ "$mode" = on ]; then sudo taskset -a -pc 2,10 $W0 >/dev/null; sudo taskset -a -pc 4,12 $W1 >/dev/null; sudo taskset -a -pc 6,14 $EC >/dev/null; sudo taskset -a -pc 0,1,3,5,7,8,9,11,13,15 $API >/dev/null
  else for p in $W0 $W1 $EC $API; do sudo taskset -a -pc 0-15 $p >/dev/null; done; fi
  log "[pin $mode] masks: W0=$(taskset -p $W0 | awk '{print $NF}') W1=$(taskset -p $W1 | awk '{print $NF}') EC=$(taskset -p $EC | awk '{print $NF}') API=$(taskset -p $API | awk '{print $NF}'); psr over 3 s: $(psr_sample)"; }
# --- measurement ---
mval(){ curl -s -m 5 $U/metrics | grep -a "^$1{" | awk '{s+=$NF} END{printf "%d", s}'; }
dss(){ # $1 label, rest = decode_ss args ; logs RESULT + tokens/step from the counter deltas
  local lab=$1; shift; local d0 a0 d1 a1
  d0=$(mval vllm:spec_decode_num_drafts_total); a0=$(mval vllm:spec_decode_num_accepted_tokens_total)
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$lab.json" > "$R/decode-$lab.out" 2>&1
  d1=$(mval vllm:spec_decode_num_drafts_total); a1=$(mval vllm:spec_decode_num_accepted_tokens_total)
  grep -a RESULT "$R/decode-$lab.out" | sed "s/^/[$lab] /" | cut -c1-230 | tee -a "$R/audit.log"
  log "[$lab] steps=$((d1-d0)) tokens/step=$(python3 -c "d=$((d1-d0)); a=$((a1-a0)); print(round(a/d+1,2) if d else 'n/a')")"; }
benchy(){ local lab=$1
  timeout 1200 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency 1 --runs 3 --no-cache --latency-mode api --format json --extra-body '{"temperature":0.6}' --save-result "$R/benchy-$lab.json" > "$R/benchy-$lab.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$lab.json" "$lab benchy" 2>/dev/null | tee -a "$R/audit.log" || log "[$lab benchy] FAILED: $(tail -1 "$R/benchy-$lab.out" | cut -c1-160)"; }
measure(){ local lab=$1 full=${2:-1}   # full=0: the short reversibility block
  dss "$lab-prose-c1" --conc 1 --kind prose --tokens 512 --runs 3
  dss "$lab-code-c8"  --conc 8 --kind code  --tokens 512 --runs 2
  if [ "$full" = 1 ]; then
    dss "$lab-code-c1"  --conc 1 --kind code  --tokens 512 --runs 3
    dss "$lab-deep30k"  --conc 1 --kind prose --tokens 512 --runs 2 --ctx 30000
    benchy "$lab"; fi; }
profile30k(){ local tag=$1 lab=$2 d0 d1 steps   # ~32 decode tokens at 30K under the torch profiler; steps from the drafts counter
  d0=$(mval vllm:spec_decode_num_drafts_total)
  curl -sf -m 30 -X POST $U/start_profile >/dev/null || { log "[$lab] start_profile FAILED"; return 1; }
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --kind prose --ctx 30000 --tokens 32 --runs 1 --out "$R/decode-$lab-prof.json" > "$R/decode-$lab-prof.out" 2>&1
  curl -sf -m 600 -X POST $U/stop_profile >/dev/null || log "[$lab] stop_profile FAILED"
  d1=$(mval vllm:spec_decode_num_drafts_total); steps=$((d1-d0)); [ "$steps" -gt 0 ] || steps=32
  sleep 20; sudo chown -R "$USER" "$R/prof/$tag" 2>/dev/null; mkdir -p "$R/prof/$lab"; find "$R/prof/$tag" -maxdepth 1 -type f -exec mv {} "$R/prof/$lab/" \; 2>/dev/null   # the boot dir holds only this capture's traces
  local n; n=$(ls "$R/prof/$lab" 2>/dev/null | wc -l); log "[$lab] profile: $n trace file(s), $steps spec steps ($(du -sh "$R/prof/$lab" 2>/dev/null | cut -f1))"
  [ "$n" -gt 0 ] && python3 /srv/qwen5090/probes/prof_summary.py "$R/prof/$lab" --top 25 --steps "$steps" --json "$R/prof-$lab.json" > "$R/prof-$lab.txt" 2>&1 \
    && { grep -aE "^== " "$R/prof-$lab.txt" | sed "s/^/[$lab prof] /" | cut -c1-230 | tee -a "$R/audit.log"; sed -n '/ALL RANKS merged/,$p' "$R/prof-$lab.txt" | head -10 | tail -8 | sed "s/^/[$lab prof top] /" | cut -c1-200 | tee -a "$R/audit.log"; }; }
needles(){ local lab=$1
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 220000 --samples 1 --out "$R/needles-$lab.jsonl" > "$R/needles-$lab.out" 2>&1
  log "[$lab needles] hits=$(grep -ac '^\[HIT ' "$R/needles-$lab.out") miss=$(grep -ac MISS "$R/needles-$lab.out") $(grep -a SUMMARY "$R/needles-$lab.out" | cut -c1-160)"; }
# --- arms ---
log "taking the daily down"; teardown
wipe_l2
if boot_retry BASE boot_cand BASE 9 2; then
  needles BASE
  pin off; measure BASE-unpin;  profile30k BASE BASE-unpin
  pin on;  measure BASE-pin;    profile30k BASE BASE-pin
  pin off; measure BASE-unpin2 0
  errlines BASE
fi; teardown
for arm in "NS7 7 2" "NS11 11 2" "DTP1 9 1"; do set -- $arm
  wipe_l2
  if boot_retry "$1" boot_cand "$1" "$2" "$3"; then measure "$1"; errlines "$1"; fi; teardown
done
# --- sheet ---
log "--- sheet (steady-state tok/s @ median; acc = accepted per draft token; tokens/step from counters) ---"
sheet_row(){ local lab=$1 f; for f in prose-c1 code-c1 deep30k code-c8; do
    python3 - "$R/decode-$lab-$f.json" "$f" <<'PY' 2>/dev/null
import json,sys
try: s=json.loads(open(sys.argv[1]).readline())["summary"]; print(f"{sys.argv[2]}={s['ss_agg_tps_median']} acc={s['accept_per_draft_median']}", end='  ')
except Exception: print(f"{sys.argv[2]}=n/a", end='  ')
PY
  done; echo; }
for lab in BASE-unpin BASE-pin BASE-unpin2 NS7 NS11 DTP1; do
  tag=${lab%%-*}; log "[SHEET $lab] pool=${POOL[$tag]:-?} free=${FREE[$tag]:-?}MiB $(sheet_row $lab)$(grep -a "\[$lab benchy\] RESULT" "$R/audit.log" | tail -1 | grep -oE 'tg_tps=[^ ]+ \(n=[0-9]+\)' )"
done
grep -aE "BOOT OK|cudagraph lines|needles|pin (on|off)\]|steps=|SHEET|error-lines|prof\]|NOTE" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
