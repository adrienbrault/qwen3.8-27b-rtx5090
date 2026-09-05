#!/usr/bin/env bash
# R190e (2026-09-05): engine A/B of patch 0144 (pcie_ipc all-reduce fused with residual add + Gemma RMSNorm, codex gpt-6-astra R190,
# derivative of FlashInfer's ipc_tp2_remote_push_kernel). R190b probe (results/2026-09-05-r190-microbench/fused-norm-b.out, 2 GPUs,
# --atol 0): bitwise at M=1 and M=10 (the c1 decode shape), saving 9 / 12 µs per call; at M=80 / 160 it differs by one bf16 ulp
# (0.0078) on 3 / 4 elements of the norm output (residual bitwise), saving 16 / 19 µs; M ≥ 321 falls back to the unfused path.
# Every decoder layer has two all-reduce → residual → norm sequences, so ≈112 calls/step ≈ 1.4 ms of the 17 ms c1 step if the probe's
# saving transfers into the graph-captured engine. Arms: CTRL (PCIE_IPC=1, fused norm off), FN (VLLM_SM12X_PCIE_IPC_AR_FUSED_NORM=1),
# FN-b (repeat). Per arm: proof line count (2 ranks × "PCIe IPC fused norm ACTIVE"), decode_ss code c1/c8/c16 + prose c1, decode ruler
# vs bf16 at ctx 0/30K (and FN vs CTRL directly — bitwise at c1 is the claim), agentic ruler vs bf16, error lines.
# Image ...-pcieipc-fusednorm (0144 on the pcieipc base), PCIE_IPC=1, SEQS 16, :8029. Queued behind r190d; r189 re-issued to wait on it.
#   unit (re-issued 2026-09-05 after the 0144 stream fix; chain r190c → r192 → r190e → r190d → r189): sudo systemd-run --unit=r190e-fusednorm --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r190e-fusednorm bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r192-humming; do sleep 30; done; exec bash /srv/qwen5090/r190e-fusednorm.sh'
#   First run (05:44 UTC): CTRL arm measured, FN arm failed to boot on both pins — 0144's strict stream-ownership check raised in capture_model's
#   eager warmup (NOTES27 addendum); unit stopped 05:58, image rebuilt with ws._check_stream(), re-queued. The audit log carries both runs.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r190e-fusednorm; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-fusednorm
STG=/srv/qwen5090/patches-v0290/r190-staging
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
sudo docker run --rm --entrypoint cat "$IMG" /opt/prs-markers/0144 2>/dev/null | grep -q APPLIED || { log "ABORT: $IMG lacks the 0144 marker"; exit 3; }
[ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in decode_ss.py decode_fidelity.py fidelity_compare.py agentic_ref.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R190e $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R190e start (lock held): 0144 pcie_ipc fused residual+RMSNorm on $IMG (PCIE_IPC=1) — CTRL, FN, FN-b ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG EXPECT_PROOF_LINES [ENV=VAL ...]
boot_arm(){ local tag=$1 expect=$2 kv rc n; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'PCIe IPC fused norm ACTIVE' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") fusednorm_lines=$n expected=$expect $([ "$n" = "$expect" ] && echo OK || echo MISMATCH)"
      grep -a 'PCIe IPC fused norm' "$R/engine-boot-$tag.log" | head -1 | sed -E 's/^.*(PCIe IPC fused norm)/\1/' | cut -c1-200 | sed "s/^/[$tag proof] /" | tee -a "$R/audit.log"
      log "[$tag allreduce] $(grep -aoE 'Using \[[^]]*\] all-reduce backends' "$R/engine-boot-$tag.log" | head -1)"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|fused norm" | grep -av "^\[rank" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
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
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    dfid "$tag" 0
    dfid "$tag" 30000
    ruler_agentic "$tag"
    log "[$tag engine error-lines] $(errs) min_free_now=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
arm CTRL 0
arm FN   2 EXTRA_ENV_APPEND="-e VLLM_SM12X_PCIE_IPC_AR_FUSED_NORM=1"
arm FN-b 2 EXTRA_ENV_APPEND="-e VLLM_SM12X_PCIE_IPC_AR_FUSED_NORM=1"
for ctx in 0 30000; do
  [ -f "$R/dec-CTRL-ctx$ctx.jsonl" ] && [ -f "$R/dec-FN-ctx$ctx.jsonl" ] && log "[FN vs CTRL decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-CTRL-ctx$ctx.jsonl" "$R/dec-FN-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
  [ -f "$R/dec-FN-ctx$ctx.jsonl" ] && [ -f "$R/dec-FN-b-ctx$ctx.jsonl" ] && log "[FN-b vs FN decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-FN-ctx$ctx.jsonl" "$R/dec-FN-b-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
done
grep -aE "BOOT OK|BOOT FAILED|boot-err|proof|allreduce\]|RESULT|PROBE FAILED|decode ctx|vs bf16|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
