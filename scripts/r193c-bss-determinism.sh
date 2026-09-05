#!/usr/bin/env bash
# R193c (2026-09-05): is batch-sharded sampling deterministic on a LOADED artifact, and does R193's B1≡D1 reproduce? R193c (one artifact,
# BSS-h loaded OFF-h's compile cd95c5/d5e217/fd65aa) read BSS-h vs OFF-h = 7/20 chunks bitwise, median 0 on agreed positions, p99 0.094 at
# ctx 0 (rare single-token flips) and 2/20 / median 0.00565 at 30K (a class-sized difference) — while R193's B1 (BSS ON, fresh compile) was
# bitwise ≡ D1 (BSS OFF, fresh compile) 20/20 at ctx 0 and 17/20 at 30K. The two do not fit one story. This unit boots OFF-h2 and BSS-h2
# on the SAME host artifact (both must show aot_saved=0 aot_loaded>=1 and the r193b hash set) and compares each with its r193b twin:
# OFF-h2 ≡ OFF-h 20/20 at both contexts = the loaded path is deterministic for OFF; BSS-h2 ≡ BSS-h = BSS-on-loaded is deterministic
# (then BSS is a real, reproducible numerics change of sub-lottery size at ctx 0 and lottery size at 30K); BSS-h2 ≠ BSS-h = BSS is
# non-deterministic run to run (a sharded-sampler ordering effect), which R193's B1 coincidence would then illustrate. Speed re-read: c8 x2,
# c16 x2, c1 x2 per arm. Own chain after r189b; restores the daily at the end.
#   unit: sudo systemd-run --unit=r193c-bss-determinism --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r193c-bss-determinism bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r189b-readme-decode; do sleep 30; done; exec bash /srv/qwen5090/r193c-bss-determinism.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r193c-bss-determinism; mkdir -p "$R"
RB=/srv/qwen5090/results/2026-09-05-r193b-bss-artifact
for f in dec-OFF-h-ctx0 dec-OFF-h-ctx30000 dec-BSS-h-ctx0 dec-BSS-h-ctx30000; do [ -f "$RB/$f.jsonl" ] || { echo "ABORT: $RB/$f.jsonl missing"; exit 3; }; done
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
sudo docker run --rm --entrypoint cat "$IMG" /opt/prs-markers/0147 2>/dev/null | grep -q PRS-0147 || { log "ABORT: $IMG lacks the 0147 marker"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q 'then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}' "$CAND" || { log "ABORT: launch-daily.sh lacks the R187 PCIE_IPC default fix"; exit 3; }
for t in decode_ss.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R193c $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R193c start (lock held): BSS on the daily host artifact, $IMG (PCIE_IPC=1) — OFF-h, BSS-h ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG EXPECT_BSS_LINES [ENV=VAL ...]
boot_arm(){ local tag=$1 expect=$2 kv rc n saved loaded; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log")
      saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$n expected=$expect $([ "$n" = "$expect" ] && echo OK || echo MISMATCH) aot_saved=$saved aot_loaded=$loaded $([ "$saved" = 0 ] && [ "$loaded" -ge 1 ] && echo SAME-ARTIFACT || echo FRESH-COMPILE) compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
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
# arm TAG EXPECT [ENV=VAL...]
arm(){ local tag=$1 expect=$2; shift 2
  teardown; wipe_l2
  if boot_arm "$tag" "$expect" "$@"; then
    sleep 20
    dfid "$tag" 0
    dfid "$tag" 30000
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
    log "[$tag engine error-lines] $(errs)"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
arm OFF-h2 0
arm BSS-h2 2 EXTRA_ARGS_APPEND=--enable-batch-sharded-sampling
# the direct question: BSS ON vs OFF on ONE artifact at temperature 0 — 20/20 agreeing chunks and median 0 = numerically equivalent
h_b=$(grep -a '\[BSS-h\] BOOT OK' "$RB/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*'); h_o2=$(grep -a '\[OFF-h2\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*'); h_b2=$(grep -a '\[BSS-h2\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*')
if grep -aq '\[OFF-h2\] BOOT OK.*SAME-ARTIFACT' "$R/audit.log" && grep -aq '\[BSS-h2\] BOOT OK.*SAME-ARTIFACT' "$R/audit.log" && [ -n "$h_b" ] && [ "$h_b" = "$h_o2" ] && [ "$h_b" = "$h_b2" ]; then log "SAME-ARTIFACT quartet ($h_b): OFF-h, BSS-h (r193b), OFF-h2, BSS-h2 all on the one host artifact"
else log "CONFOUNDED: an arm fresh-compiled or the hash sets differ (r193b $h_b / OFF-h2 $h_o2 / BSS-h2 $h_b2)"; fi
cmp(){ log "[$1 vs $2 decode ctx$3] $(python3 $PR/decode_fidelity.py compare "$4" "$5" 2>&1 | tail -1 | cut -c1-300)"; }
for ctx in 0 30000; do
  [ -f "$R/dec-OFF-h2-ctx$ctx.jsonl" ] && cmp OFF-h2 OFF-h $ctx "$RB/dec-OFF-h-ctx$ctx.jsonl" "$R/dec-OFF-h2-ctx$ctx.jsonl"
  [ -f "$R/dec-BSS-h2-ctx$ctx.jsonl" ] && cmp BSS-h2 BSS-h $ctx "$RB/dec-BSS-h-ctx$ctx.jsonl" "$R/dec-BSS-h2-ctx$ctx.jsonl"
  [ -f "$R/dec-OFF-h2-ctx$ctx.jsonl" ] && [ -f "$R/dec-BSS-h2-ctx$ctx.jsonl" ] && cmp BSS-h2 OFF-h2 $ctx "$R/dec-OFF-h2-ctx$ctx.jsonl" "$R/dec-BSS-h2-ctx$ctx.jsonl"
done
grep -aE "BOOT OK|BOOT FAILED|boot-err|RESULT|PROBE FAILED|decode ctx|vs bf16|error-lines|SAME-ARTIFACT|CONFOUNDED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
