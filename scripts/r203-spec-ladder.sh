#!/usr/bin/env bash
# R203 (2026-09-06, R202 M1; user "ok go"): does speculative decoding change the prompt-logprob ladder? vllm#53488 (open, 2026-08-23) reports
# `prompt_logprobs` silently corrupted "for a subset of requests" under spec decode on Qwen3.8-27B (MTP-reported; DFlash untested). Every
# dense/agentic ruler number since R156 was taken spec ON (the served drafter) against a bf16 reference dumped spec OFF; no spec-ON-vs-OFF
# ladder pair exists in FINDINGS. Two arms on the daily image, same route (EXP=1 SEQS=16, PCIE_IPC=1, BSS=1, VLLM_TRITON_FORCE_FIRST_CONFIG=1
# on both per R193d): ON = the daily's DFlash ns7 draft_tp2; OFF = SPEC_METHOD=none (new launch-daily.sh passthrough → NOSPEC=1, no drafter,
# no --speculative-config). Both arms run the dense ruler (R156 corpus, 693 docs, --logprobs 20) and the agentic ruler, scored against the
# R156 bf16 dumps AND against each other: fidelity_compare.py (corpus PPL / top-1 / KL) plus probes/ladder_doc_compare.py (per-doc PPL
# deltas, worst docs, positions moved by >1 nat) — the per-doc view is what would show a #53488-shaped subset corruption that a corpus
# mean hides. Confound stated up front: spec OFF changes the attention block (mamba page without draft slots) and therefore the compile
# hash, so the two arms are two compile artifacts; the R193 per-boot draw (~0.004-0.006 on the decode ruler at 30K) is the noise floor
# for small uniform deltas, not for per-doc outliers. Outcome table: CLEAN (median doc delta ~0, no doc beyond ±2 %, no position beyond
# 1 nat beyond the two-boot control) = the rulers stand; OUTLIERS = the ladder must be re-taken spec OFF for every candidate and #53488 applies to
# DFlash too. Control: R196 arm H (results/2026-09-05-r196-minima-audition/dump-H-*.jsonl, the same daily config spec ON on another boot)
# gives the per-doc spread of two boots of ONE config; ON-vs-OFF outliers absent from ON-vs-H are the spec signal. One code-c1 decode_ss run per arm proves the arm is what it says (spec OFF ≈ half the tok/s). Daily restored at the end.
#   unit: sudo systemd-run --unit=r203-spec-ladder --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r203-spec-ladder bash /srv/qwen5090/r203-spec-ladder.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r203-spec-ladder; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder
BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
PINS="13980000000 13500000000"
IMG=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1)
[ -n "$IMG" ] || { log "ABORT: cannot read DAILY_IMG from $CAND"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] || { log "ABORT: reference dumps/corpus missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q "R203 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R203 SPEC_METHOD=none passthrough"; exit 3; }
for t in fidelity_ladder.py fidelity_compare.py agentic_ref.py ladder_doc_compare.py decode_ss.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R203 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R203 start (lock held): spec ON vs OFF prompt-logprob ladder on $IMG (EXP=1 SEQS=16 PCIE_IPC=1 BSS=1 FORCE_FIRST_CONFIG=1) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG [ENV=VAL ...]
boot_arm(){ local tag=$1 kv rc; shift 1
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1" "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      local saved loaded; saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) dflash_graphs=$(grep -ac 'Capturing dflash2 CUDA graphs' "$R/engine-boot-$tag.log") spec_cfg=$(sudo docker inspect vllm-exp --format '{{json .Args}}' | grep -ac speculative-config) force_first=$(sudo docker inspect vllm-exp --format '{{json .Config.Env}}' | grep -ac VLLM_TRITON_FORCE_FIRST_CONFIG=1) aot_saved=$saved aot_loaded=$loaded $([ "$saved" = 0 ] && [ "$loaded" -ge 1 ] && echo SAME-ARTIFACT || echo FRESH-COMPILE) compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense --resume > "$R/score-$T-dense.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the dense ruler ($(grep -ac '^\[warn\]' "$R/score-$T-dense.out") docs failed): $(sudo dmesg -T | grep -a Xid | tail -1 | cut -c1-160)"; finish ENGINE-DIED; exit 5; }
  log "[$T dense] docs failed=$(grep -ac '^\[warn\]' "$R/score-$T-dense.out") records=$(wc -l < "$R/dump-$T-dense.jsonl")"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
H_DIR=/srv/qwen5090/results/2026-09-05-r196-minima-audition   # R196 arm H = the daily config spec ON, another boot (2026-09-05 14:22, aot_saved=3 = its own artifact, no FORCE_FIRST): two-boot noise-floor control
doc_cmp(){ local kind=$1 a=$2 b=$3 la=$4 lb=$5
  python3 $PR/ladder_doc_compare.py --a "$R/dump-$a-$kind.jsonl" --b "$b" --label-a $la --label-b $lb 2>&1 | tee "$R/doc-compare-$kind-$la-vs-$lb.txt" | grep -aE "^docs|^corpus|^per-doc|^positions|^DOC-COMPARE" | cut -c1-260 | sed "s/^/[$kind per-doc $la vs $lb] /" | tee -a "$R/audit.log"; }
# arm TAG [ENV=VAL ...]
arm(){ local tag=$1; shift 1
  teardown; wipe_l2
  if boot_arm "$tag" "$@"; then
    sleep 20
    ruler_dense "$tag"
    ruler_agentic "$tag"
    p1 $tag code-c1 --conc 1 --tokens 512 --runs 1 --kind code
    log "[$tag] engine error lines: $(errs)"
    ELOG > "$R/engine-$tag-final.log"
  else log "[$tag] ARM FAILED: could not boot at any pin"; fi; }
arm OFF SPEC_METHOD=none POOL_MIN=400000 POOL_MAX=2500000
arm ON
if [ -s "$R/dump-ON-dense.jsonl" ] && [ -s "$R/dump-OFF-dense.jsonl" ]; then
  python3 $PR/fidelity_compare.py --ref "$R/dump-OFF-dense.jsonl" --arm "$R/dump-ON-dense.jsonl" --label "ON-vs-OFF" --json "$R/on-vs-off-dense.json" 2>&1 | tee "$R/on-vs-off-dense.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[ON vs OFF dense] /" | tee -a "$R/audit.log"
  for kind in dense agentic; do
    doc_cmp $kind ON "$R/dump-OFF-$kind.jsonl" ON OFF                       # the M1 question
    [ -s "$H_DIR/dump-H-$kind.jsonl" ] && { doc_cmp $kind ON "$H_DIR/dump-H-$kind.jsonl" ON H; doc_cmp $kind OFF "$H_DIR/dump-H-$kind.jsonl" OFF H; }   # two-boot control
  done
else log "COMPARE SKIPPED: one arm has no dense dump"; fi
grep -aE "BOOT OK|ARM FAILED|vs bf16|ON vs OFF|per-doc|code-c1|error lines|ENGINE DIED|restor" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
