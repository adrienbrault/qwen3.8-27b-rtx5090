#!/usr/bin/env bash
# R191 (2026-09-05): batch-sharded sampling numerics. R187's BSS arm was the only flag arm outside the band in steps/s (+3.5% c16,
# +1.6% c8 — the size the R190 audit predicted from halving the one 19.9 MB logits all-gather per step), but its acceptance per draft
# token differed from the base seeds everywhere (code c1 0.276 vs 0.363), i.e. the sharded sampler draws different tokens for the same
# seed. Before the step rate counts: (1) greedy equivalence — decode_fidelity.py is T=0 with logprobs; BSS ON vs OFF must agree
# 20/20 chunks with median |Δlogprob| 0 at ctx 0 and 30K if vLLM's "same result, sharded" claim holds on this stack; (2) both arms vs
# the r173c bf16 decode reference (the promotion ruler); (3) the agentic ruler vs the bf16 dump; (4) acceptance repeated 3× at code
# c1 / c8 on each arm (plus a second OFF read for the run-to-run spread). Image = the daily (pcieipc) with PCIE_IPC=1, pin 13.98 GB,
# SEQS 16, :8029. Arms: OFF-a (control), ON, OFF-b. Queued behind r190-microbench; r189 re-issued to wait on this unit (chain restore).
#   unit: sudo systemd-run --unit=r191-bss-numerics --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r191-bss-numerics bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r190-microbench; do sleep 30; done; exec bash /srv/qwen5090/r191-bss-numerics.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r191-bss-numerics; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q 'then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}' "$CAND" || { log "ABORT: launch-daily.sh lacks the R187 PCIE_IPC default fix"; exit 3; }
for t in decode_ss.py decode_fidelity.py fidelity_compare.py agentic_ref.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R191 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R191 start (lock held): batch-sharded sampling numerics on $IMG (PCIE_IPC=1) — OFF-a, ON, OFF-b ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG EXPECT_BSS_LINES [ENV=VAL ...]
boot_arm(){ local tag=$1 expect=$2 kv rc n; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$n expected=$expect $([ "$n" = "$expect" ] && echo OK || echo MISMATCH)"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|sharded" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
# arm TAG EXPECT [ENV=VAL...]
arm(){ local tag=$1 expect=$2; shift 2
  teardown; wipe_l2
  if boot_arm "$tag" "$expect" "$@"; then
    sleep 20
    dfid "$tag" 0
    dfid "$tag" 30000
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 3 --kind code
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    ruler_agentic "$tag"
    log "[$tag engine error-lines] $(errs)"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
arm OFF-a 0
arm ON    2 EXTRA_ARGS_APPEND=--enable-batch-sharded-sampling
arm OFF-b 0
# the direct question: BSS ON vs OFF at temperature 0 (same image, same night) — 20/20 agreeing chunks and median 0 = numerically equivalent
for ctx in 0 30000; do
  [ -f "$R/dec-OFF-a-ctx$ctx.jsonl" ] && [ -f "$R/dec-ON-ctx$ctx.jsonl" ] && log "[ON vs OFF-a decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-OFF-a-ctx$ctx.jsonl" "$R/dec-ON-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
  [ -f "$R/dec-OFF-a-ctx$ctx.jsonl" ] && [ -f "$R/dec-OFF-b-ctx$ctx.jsonl" ] && log "[OFF-b vs OFF-a decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-OFF-a-ctx$ctx.jsonl" "$R/dec-OFF-b-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
done
grep -aE "BOOT OK|BOOT FAILED|boot-err|RESULT|PROBE FAILED|decode ctx|vs bf16|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
