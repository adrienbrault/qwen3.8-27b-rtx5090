#!/usr/bin/env bash
# R205b (2026-09-06, memo item "GDN metadata overhead", vllm#52297 ported by codex as patch 0157, BRIEF31): the GDN speculative-decode
# metadata (spec masks, decode/prefill split, spec/non-spec token indices, three query_start_loc cumsums, the accepted-tokens gather) is
# built ONCE per step in MambaHybridModelState.prepare_attn and handed to the 3 GDN groups' build() instead of being recomputed per group.
# Pure refactor, no knob: the served numerics must be bitwise identical. Two arms on :8029, the daily launch config (DFlash ns7, PCIE_IPC=1,
# BSS=1, SEQS 16, pins 13.98/13.5), VLLM_TRITON_FORCE_FIRST_CONFIG=1 on both (R193d protocol):
#   OFF = the daily image ($DAILY)   ON = $DAILY-gdncm (build-r205b-gdn-common-metadata.sh; marker 0157)
#   gate 1: ON's compile artifact must be the one OFF used (aot_loaded ≥ 1, aot_saved 0, same hash set) — the diff touches worker code
#           outside the compiled graph, so the key should not move; if it does, the ruler is confounded by the compile lottery (R193).
#   gate 2: decode ruler ctx 0 / 30K ON vs OFF bitwise 20/20 (median 0) — any token that moves is a bug, not a trade.
#   gate 3: proof line "SM12X GDN common metadata hoisted: 3 GDN groups share one spec-metadata build per step" on ON (TP2: 2 processes).
#   numbers: decode_ss code c1 (runs 3) / prose c1 / code c8 / code c16 steps/s ON vs OFF. NOTES31 predicts ≤ 0.08 ms/step at c1
#           (≈ 0.4 % steps/s, below the run-to-run noise) and up to ~0.28 ms on mixed batches; the falsifier is nothing moving at all.
#   unit: sudo systemd-run --unit=r205b-gdncm-gate --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r205b-gdncm-gate bash /srv/qwen5090/r205b-gdncm-gate.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r205b-gdncm-gate; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); IMG_OFF=$DAILY; IMG_ON=$DAILY-gdncm
PINS="13980000000 13500000000"
PROOF='SM12X GDN common metadata hoisted'
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
sudo docker image inspect "$IMG_OFF" >/dev/null 2>&1 || { log "ABORT: daily image $IMG_OFF missing"; exit 3; }
sudo docker image inspect "$IMG_ON" >/dev/null 2>&1 && sudo docker run --rm --entrypoint cat "$IMG_ON" /opt/prs-markers/0157 2>/dev/null | grep -q 0157 || { log "ABORT: $IMG_ON missing or without the 0157 marker"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in decode_ss.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R205b $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R205b start (lock held): 0157 GDN common metadata — OFF=$IMG_OFF ON=$IMG_ON, daily launch config, FORCE_FIRST on both ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG IMAGE EXPECT_PROOF_LINES
boot_arm(){ local tag=$1 img=$2 expect=$3 kv rc n saved loaded
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$img EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac "$PROOF" "$R/engine-boot-$tag.log")
      saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv image=$img pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") proof_lines=$n expected=$expect $([ "$n" = "$expect" ] && echo OK || echo MISMATCH) aot_saved=$saved aot_loaded=$loaded compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
      grep -a "$PROOF" "$R/engine-boot-$tag.log" | head -1 | sed -E 's/^.*(SM12X GDN)/\1/' | cut -c1-200 | sed "s/^/[$tag proof] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|common_gdn|compute_common" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
arm(){ local tag=$1
  sleep 20
  dfid "$tag" 0
  dfid "$tag" 30000
  p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 3 --kind code
  p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
  log "[$tag engine error-lines] $(errs)"; }
teardown; wipe_l2
if boot_arm OFF $IMG_OFF 0; then arm OFF; else log "[OFF] BOOT FAILED on every pin"; fi
teardown; wipe_l2
if boot_arm ON $IMG_ON 2; then
  h_off=$(grep -a '\[OFF\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*'); h_on=$(grep -a '\[ON\] BOOT OK' "$R/audit.log" | grep -aoE 'compile_hashes=[0-9a-f,]*')
  if grep -aq 'aot_saved=0 aot_loaded=[1-9]' <(grep -a '\[ON\] BOOT OK' "$R/audit.log") && [ -n "$h_off" ] && [ "$h_off" = "$h_on" ]; then log "[ON] SAME-ARTIFACT as OFF ($h_off; loaded, nothing saved)"
  else log "[ON] WARNING: not the same artifact as OFF (OFF $h_off / ON $h_on) — the ruler below is confounded by the compile lottery (R193)"; fi
  arm ON
else log "[ON] BOOT FAILED on every pin"; fi
for ctx in 0 30000; do
  [ -f "$R/dec-OFF-ctx$ctx.jsonl" ] && [ -f "$R/dec-ON-ctx$ctx.jsonl" ] && log "[ON vs OFF decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-OFF-ctx$ctx.jsonl" "$R/dec-ON-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
done
grep -aE "BOOT OK|BOOT FAILED|boot-err|proof|RESULT|PROBE FAILED|decode ctx|SAME-ARTIFACT|WARNING|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
