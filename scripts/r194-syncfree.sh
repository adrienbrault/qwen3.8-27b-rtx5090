#!/usr/bin/env bash
# R194 (2026-09-05): patch 0146 (codex BRIEF29) — one asynchronous D2H copy of the target seq_lens per decode step instead of the
# five blocking `seq_lens.to("cpu")` reads the R190d census found (deprecated CommonAttentionMetadata.seq_lens_cpu; the draft keeps
# its own exact copy, so the census should read 9 → 5, not 4). Target = the 4.09 ms/step no-kernel gap R183 measured at c1.
# Four boots on :8029 (PCIE_IPC=1, SEQS 16, pins 13.98/13.5):
#   CEN-OFF / CEN-ON  image ...-r190diag-syncfree (0142c census + 0143 + 0146), VLLM_SM12X_SYNC_CENSUS=1, knob off/on, c1 code decode
#                     → per-step sync table: target api:to backend.py:473 expected 5 → 0, seqlens_copy.py synchronize 0 → 1, total 9 → 5;
#                     proof line "SM12X seq_lens one-copy active: 5 target seq_lens_cpu consumers ..." on the ON boot only.
#   OFF / ON          image ...-pcieipc-syncfree, knob off/on. The knob is excluded from the compile key (operator hunk in 0146), so ON
#                     must LOAD the artifact OFF saved (aot_loaded ≥ 1, aot_saved 0 = SAME-ARTIFACT) — the decode ruler ctx 0 / 30K ON vs
#                     OFF is then a same-artifact bitwise gate (20/20, median 0 required; R193 showed fresh artifacts differ on their own).
#                     Decode probes code c1 (3 runs) / prose c1 / code c8 / code c16 → steps/s ON vs OFF; the c1 step is the number.
# Falsifier (NOTES29): syncs gone but the c1 step unchanged → the gap is launch-bound, not sync-bound. Own chain: waits for r189 (the
# daily restore + pcie_ipc promotion), takes the daily down, restores it at the end via daily-restore-retry.sh.
#   unit: sudo systemd-run --unit=r194-syncfree --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r194-syncfree bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r189-promote-pcieipc; do sleep 30; done; exec bash /srv/qwen5090/r194-syncfree.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r194-syncfree; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
BASE=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
IMG=$BASE-syncfree; IMG_CEN=$BASE-r190diag-syncfree
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
has0146(){ sudo docker image inspect "$1" >/dev/null 2>&1 && sudo docker run --rm --entrypoint cat "$1" /opt/prs-markers/0146 2>/dev/null | grep -q PRS-0146; }
has0146 "$IMG" || { log "ABORT: image $IMG missing or without the 0146 marker"; exit 3; }
CEN=1; has0146 "$IMG_CEN" || { CEN=0; log "WARNING: census image $IMG_CEN missing or without the 0146 marker — census arms SKIPPED, speed/ruler arms run"; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in decode_ss.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R194 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R194 start (lock held): 0146 one-copy seq_lens — CEN-OFF, CEN-ON ($IMG_CEN); OFF, ON ($IMG) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG IMAGE EXPECT_PROOF_LINES [ENV=VAL ...]
boot_arm(){ local tag=$1 img=$2 expect=$3 kv rc n saved loaded; shift 3
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$img "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'SM12X seq_lens one-copy active' "$R/engine-boot-$tag.log")
      saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") proof_lines=$n expected=$expect $([ "$n" = "$expect" ] && echo OK || echo MISMATCH) aot_saved=$saved aot_loaded=$loaded compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
      grep -a 'SM12X seq_lens one-copy active' "$R/engine-boot-$tag.log" | head -1 | sed -E 's/^.*(SM12X seq_lens)/\1/' | cut -c1-200 | sed "s/^/[$tag proof] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|SEQLENS|seqlens_copy" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
# census TAG: the 0142c table prints once after 20 warm-up + 50 steps of decode; aggregate per phase/site over the measured steps
census(){ local tag=$1
  p1 $tag code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
  sleep 5; ELOG > "$R/engine-$tag-full.log"
  awk '/SM12X sync census complete/{p=1} p{print} p && ++n>600{exit}' "$R/engine-$tag-full.log" > "$R/sync-census-$tag.txt"
  log "[$tag census] table lines=$(wc -l < "$R/sync-census-$tag.txt")"
  sed -E 's/^.*sync_census.py:[0-9]+\] //' "$R/sync-census-$tag.txt" | awk '$2!="total" && NF>=5 {c[$2" "$4" "$5]+=$3; n[$2" "$4" "$5]++} END{for(k in c) printf "%6d over %3d steps  %s\n", c[k], n[k], k}' | sort -rn | head -12 | sed "s/^/[$tag census] /" | tee -a "$R/audit.log"
  log "[$tag census totals] $(sed -E 's/^.*sync_census.py:[0-9]+\] //' "$R/sync-census-$tag.txt" | awk '$2=="total"{s+=$3; n++} END{printf "%d syncs over %d steps = %.2f/step", s, n, (n?s/n:0)}')"
  log "[$tag engine error-lines] $(errs)"; }
if [ "$CEN" = 1 ]; then
  teardown; wipe_l2
  if boot_arm CEN-OFF $IMG_CEN 0 EXTRA_ENV_APPEND="-e VLLM_SM12X_SYNC_CENSUS=1"; then sleep 15; census CEN-OFF; else log "[CEN-OFF] BOOT FAILED on every pin"; fi
  teardown; wipe_l2
  if boot_arm CEN-ON $IMG_CEN 2 EXTRA_ENV_APPEND="-e VLLM_SM12X_SYNC_CENSUS=1 -e VLLM_SM12X_SEQLENS_ONE_COPY=1"; then sleep 15; census CEN-ON; else log "[CEN-ON] BOOT FAILED on every pin"; fi
fi
# same-artifact pair: OFF first (saves or loads), then ON must load — the knob is not a compile-key factor
speed_arm(){ local tag=$1
  sleep 20
  dfid "$tag" 0
  dfid "$tag" 30000
  p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 3 --kind code
  p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
  log "[$tag engine error-lines] $(errs)"; }
teardown; wipe_l2
if boot_arm OFF $IMG 0; then speed_arm OFF; else log "[OFF] BOOT FAILED on every pin"; fi
teardown; wipe_l2
if boot_arm ON $IMG 2 EXTRA_ENV_APPEND="-e VLLM_SM12X_SEQLENS_ONE_COPY=1"; then
  # same artifact = ON loaded (saved nothing) AND its compile-hash set equals OFF's; R191's ON arm had loaded=6/saved=0 of a DIFFERENT hash set
  h_off=$(grep -a '\[OFF\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*'); h_on=$(grep -a '\[ON\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*')
  if grep -aq 'aot_saved=0 aot_loaded=[1-9]' <(grep -a '\[ON\] BOOT OK' "$R/audit.log") && [ -n "$h_off" ] && [ "$h_off" = "$h_on" ]; then log "[ON] SAME-ARTIFACT as OFF ($h_off; loaded, nothing saved)"
  else log "[ON] WARNING: not the same artifact as OFF (OFF $h_off / ON $h_on) — the ruler below is confounded by the compile lottery (R193)"; fi
  speed_arm ON
else log "[ON] BOOT FAILED on every pin"; fi
for ctx in 0 30000; do
  [ -f "$R/dec-OFF-ctx$ctx.jsonl" ] && [ -f "$R/dec-ON-ctx$ctx.jsonl" ] && log "[ON vs OFF decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-OFF-ctx$ctx.jsonl" "$R/dec-ON-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
done
grep -aE "BOOT OK|BOOT FAILED|boot-err|proof|census|RESULT|PROBE FAILED|decode ctx|SAME-ARTIFACT|WARNING|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
