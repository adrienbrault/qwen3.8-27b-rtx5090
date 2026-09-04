#!/usr/bin/env bash
# R173b (2026-09-04): paired ns7/ns9 confirmation on the elected 0.29 nvfp4 path (S image, split_kv=0). r173 measured ns7 once
# (code c8 1,216 vs BASE 1,018–1,135 across three blocks, pool +42.7K) and wrote it into the recommended config on speed alone;
# c1 on this box is ±20% within a boot, so one boot per arm cannot separate ns7 from ns9. Two things are missing before ns7
# becomes the launcher default:
#   1. speed, paired: four boots alternating ns9/ns7/ns9/ns7 at the same 13.5 GB pin, code c8 x3 + deep30k x2 + prose c1 x2 each;
#   2. decode numerics: ns7 changes the verify batch (q=8 vs q=10) on the path whose attention kernels made ON-vs-OFF diverge
#      18/20 chunks (r168c). The bf16 rulers are prefill-path and cannot see this. decode_fidelity (greedy, 20 chunks x 256 tokens,
#      ctx0 and ctx30000) on every boot: NS9B-vs-NS9A and NS7B-vs-NS7A are the cross-boot floors (r168d: EXACT, 20/20),
#      NS7-vs-NS9 on both pairs is the verdict. 20/20 = the speculative length is numerically invisible; otherwise quantified.
# Reads: r173 BASE/NS7 numbers for the sheet; r168d's dec-S-*.jsonl (rc1+fi0616 image) as an informative cross-image floor.
# Unit: sudo systemd-run --unit=r173b-ns-confirm --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r173b-ns-confirm bash /srv/qwen5090/r173b-ns-confirm.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r173b-ns-confirm; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616; SPLIT=0
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh; L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity; R168D=/srv/qwen5090/results/2026-09-04-r168d-splitkv-ref
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$FD/corpus.jsonl" ] || { log "ABORT: decode corpus missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R173b $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R173b ns7/ns9 paired confirmation start (lock held): image=$IMG split_kv=$SPLIT ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
declare -A POOL FREE
boot_cand(){ local tag=$1 ns=$2 kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$SPLIT CAND_IMG=$IMG SPEC_NS=$ns SPEC_DTP=2 ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      POOL[$tag]=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag.log" | tail -1 | tr -dc 0-9); FREE[$tag]=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag.log" | tail -1 | tr -dc 0-9)
      log "[$tag] BOOT OK ns$ns draft_tp2 pin=${kv:-default} pool=${POOL[$tag]} min_free=${FREE[$tag]}MiB kv=$(sudo docker logs vllm-exp 2>&1 | grep -aoE 'reserved [0-9.]+ GiB memory for KV Cache' | tail -1)"
      return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || break; teardown
  done; return 1; }
boxstate(){ sudo docker logs vllm-exp 2>&1 | grep -aq "CUDA error: invalid argument" || grep -aq "CUDA error: invalid argument" "$R/boot-$1.log" 2>/dev/null; }
boot_retry(){ local tag=$1; shift; "$@" && return 0
  if boxstate "$tag"; then log "[$tag] box-state boot failure (invalid argument in warm-up): teardown + 300 s idle, retrying once"; teardown; settle 300; "$@" && return 0; fi
  return 1; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
mval(){ curl -s -m 5 $U/metrics | grep -a "^$1{" | awk '{s+=$NF} END{printf "%d", s}'; }
dss(){ local lab=$1; shift; local d0 a0 d1 a1
  d0=$(mval vllm:spec_decode_num_drafts_total); a0=$(mval vllm:spec_decode_num_accepted_tokens_total)
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$lab.json" > "$R/decode-$lab.out" 2>&1
  d1=$(mval vllm:spec_decode_num_drafts_total); a1=$(mval vllm:spec_decode_num_accepted_tokens_total)
  grep -a RESULT "$R/decode-$lab.out" | sed "s/^/[$lab] /" | cut -c1-230 | tee -a "$R/audit.log"
  log "[$lab] steps=$((d1-d0)) tokens/step=$(python3 -c "d=$((d1-d0)); a=$((a1-a0)); print(round(a/d+1,2) if d else 'n/a')")"; }
dfid(){ python3 /srv/qwen5090/probes/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$1-ctx$2.jsonl" --chunks 20 --ctx "$2" --tokens 256 > "$R/dec-$1-ctx$2.out" 2>&1
  log "[$1 decode ctx$2] $(tail -1 "$R/dec-$1-ctx$2.out" | cut -c1-160)"; }
cmp_dec(){ [ -f "$2" ] && [ -f "$3" ] || { log "[decode $1] skipped (dump missing)"; return; }
  log "[decode $1] $(python3 /srv/qwen5090/probes/decode_fidelity.py compare "$2" "$3" 2>&1 | tail -1 | cut -c1-300)"; }
arm(){ local tag=$1 ns=$2
  wipe_l2
  if boot_retry "$tag" boot_cand "$tag" "$ns"; then
    dfid "$tag" 0; dfid "$tag" 30000
    dss "$tag-code-c8"  --conc 8 --kind code  --tokens 512 --runs 3
    dss "$tag-deep30k"  --conc 1 --kind prose --tokens 512 --runs 2 --ctx 30000
    dss "$tag-prose-c1" --conc 1 --kind prose --tokens 512 --runs 2
    errlines "$tag"
  fi; teardown; }
log "taking the daily down"; teardown
arm NS9A 9; arm NS7A 7; arm NS9B 9; arm NS7B 7
log "--- decode numerics (greedy, 20 chunks x 256 tokens): floors first, then ns7 vs ns9 ---"
for c in 0 30000; do
  cmp_dec "floor NS9B-vs-NS9A ctx$c" "$R/dec-NS9A-ctx$c.jsonl" "$R/dec-NS9B-ctx$c.jsonl"
  cmp_dec "floor NS7B-vs-NS7A ctx$c" "$R/dec-NS7A-ctx$c.jsonl" "$R/dec-NS7B-ctx$c.jsonl"
  cmp_dec "NS7A-vs-NS9A ctx$c"       "$R/dec-NS9A-ctx$c.jsonl" "$R/dec-NS7A-ctx$c.jsonl"
  cmp_dec "NS7B-vs-NS9B ctx$c"       "$R/dec-NS9B-ctx$c.jsonl" "$R/dec-NS7B-ctx$c.jsonl"
  cmp_dec "info NS9A-vs-r168d-S(rc1 image) ctx$c" "$R168D/dec-S-ctx$c.jsonl" "$R/dec-NS9A-ctx$c.jsonl"
done
log "--- sheet (steady-state tok/s @ median [min-max over runs]; acc = accepted per draft token) ---"
sheet_row(){ local lab=$1 f; for f in code-c8 deep30k prose-c1; do
    python3 - "$R/decode-$lab-$f.json" "$f" <<'PY' 2>/dev/null
import json,sys
try: s=json.loads(open(sys.argv[1]).readline())["summary"]; mm=s.get('ss_agg_tps_min_max',['?','?']); print(f"{sys.argv[2]}={s['ss_agg_tps_median']} [{mm[0]}-{mm[1]}] acc={s['accept_per_draft_median']}", end='  ')
except Exception: print(f"{sys.argv[2]}=n/a", end='  ')
PY
  done; echo; }
for lab in NS9A NS7A NS9B NS7B; do log "[SHEET $lab] pool=${POOL[$lab]:-?} free=${FREE[$lab]:-?}MiB $(sheet_row $lab)"; done
grep -aE "BOOT OK|decode|steps=|SHEET|error-lines" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
