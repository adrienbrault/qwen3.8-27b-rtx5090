#!/usr/bin/env bash
# R168e (2026-09-04): the bf16 ruler on the three nvfp4 attention paths of the 0.29 route. r168d settled the floor (same config,
# two boots = EXACT: 20/20 decode chunks, depth2 100%, KL 0) and measured every path against the fp8 arm F, but F is a proxy;
# the target is bf16 (user rule). r169 measured only ON (split_kv=1) against bf16: 92.773% top-1 / +0.789% PPL / KL 0.0144
# on 724,781 positions. This unit runs the same dense + agentic rulers on OFF (rc2, split_kv=0) and S (FlashInfer 0.6.16.post3
# swap image) and repeats ON as the cross-boot control (must reproduce r169 N to the floor). 724K positions decide what 1,000
# depth2 records cannot; depth2-vs-F (r168d) stays the deep-context half of the verdict.
# Unit: sudo systemd-run --unit=r168e-splitkv-bf16 --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r168e-splitkv-bf16 bash /srv/qwen5090/r168e-splitkv-bf16.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r168e-splitkv-bf16; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
RC2_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs
FI16_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed-fi0616
for i in $RC2_IMG $FI16_IMG; do sudo docker image inspect $i >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] || { log "ABORT: bf16 references missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R168e split-KV bf16 ruler start (lock held) ==="
U=http://127.0.0.1:8029
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R168e $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
boot_cand(){ local tag=$1 split=$2 img=$3 kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$split CAND_IMG=$img ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT OK split_kv=$split image=$img $(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$tag.log" | tail -1) flashinfer=$(sudo docker exec vllm-exp python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null)"; return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || break; teardown
  done; return 1; }
boxstate(){ sudo docker logs vllm-exp 2>&1 | grep -aq "CUDA error: invalid argument" || grep -aq "CUDA error: invalid argument" "$R/boot-$1.log" 2>/dev/null; }
boot_retry(){ local tag=$1; shift; "$@" && return 0
  if boxstate "$tag"; then log "[$tag] box-state boot failure (invalid argument in warm-up): teardown + 300 s idle, retrying once"; teardown; settle 300; "$@" && return 0; fi
  return 1; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
bf16_ladder(){ local T=$1
  timeout 3600 python3 /srv/qwen5090/probes/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 /srv/qwen5090/probes/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall|ALL|agent|code|prose|PPL|KL|top-1" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"
  timeout 3600 python3 /srv/qwen5090/probes/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  log "[$T bf16 agentic] $(grep -aE "^DONE|FAIL" "$R/score-$T-agentic.out" | tail -1 | cut -c1-160)"
  python3 /srv/qwen5090/probes/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "^===|scored positions|top-1 agreement|corpus PPL|certain|confident|moderate|near-tie|FATAL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
arm(){ local tag=$1 split=$2 img=$3
  wipe_l2
  if boot_retry "$tag" boot_cand "$tag" "$split" "$img"; then bf16_ladder "$tag"; errlines "$tag"; fi
  teardown; }
log "taking the daily down"; teardown
arm ON  1 $RC2_IMG      # r169 N repeat: cross-boot control against 92.773% / +0.789% / KL 0.0144
arm OFF 0 $RC2_IMG      # knob off = FlashInfer 0.6.18's guarded (non-split) nvfp4 path, 5x slower at depth
arm S   0 $FI16_IMG     # FlashInfer 0.6.16.post3 swap layer (split path on by default there)
log "--- sheet (bf16 dense: top-1 / PPL delta / KL; agentic: top-1 / PPL delta) ---"
for T in ON OFF S; do
  log "[SHEET $T] dense: $(grep -aE 'overall top-1|corpus PPL|truncated KL' "$R/bf16-$T.txt" 2>/dev/null | sed -E 's/ +/ /g' | cut -c1-90 | tr '\n' '|')  agentic: $(grep -aE 'overall top-1|corpus PPL' "$R/bf16-$T-agentic.txt" 2>/dev/null | sed -E 's/ +/ /g' | cut -c1-90 | tr '\n' '|')"
done
grep -aE "BOOT OK|SHEET|error-lines" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
