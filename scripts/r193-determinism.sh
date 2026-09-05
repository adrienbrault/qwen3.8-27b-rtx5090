#!/usr/bin/env bash
# NOTE (post-run, 2026-09-05 08:50 UTC): the automated best_config diff below reported "differing=0 only-in-one=0 common=0" — it tars
# /tmp/torchinductor_root, but this image writes the runtime-autotune *.best_config files elsewhere; the evidence is the per-arm
# bestconfig-<arm>-manual/ dumps taken from the inductor cache while each engine was up (34 files per arm, 9 differing D1 vs D2). Fix the
# path before any rerun.
# R193 (2026-09-05): are the T=0 decode rulers a compile-artifact lottery? R190c found CTRL vs B12X (same image, same flags, two fresh
# torch.compile artifacts) disagreeing on 19/20 chunks at ctx 0 (median |Δlogprob| 0.00042; vs bf16 medians 0.0004 vs 0.00062), while
# every pair that reused the SAME saved AOT artifact was bitwise identical (R191 OFF-a/OFF-b; R190c B12X/B12X-b). Two explanations fit:
# (a) a lottery — inductor's runtime Triton config autotune (pointwise/reduction block sizes; torch/_inductor/runtime/triton_heuristics.py
# picks by benchmarking unless autotune_pointwise is off) chooses differently on each fresh compile and the choice is bundled into the
# artifact; (b) deterministic but config-dependent numerics (each env/patch changes the traced graph). If (a), R191's "BSS not
# T=0-equivalent / farther from bf16" verdict is confounded (ON came from a different artifact than OFF). Note VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
# is INERT here: set_inductor_config() only applies it to single-size compile ranges and the daily has compile_sizes [] — so the
# determinism knob is inductor_compile_config {"triton.autotune_pointwise": false} (single config per kernel, no benchmarking).
# Arms (daily image, PCIE_IPC=1, SEQS 16, :8029; every arm is a FRESH compile: VLLM_CACHE_ROOT points at a container-local, empty
# dir so no saved AOT artifact can be loaded; inductor's cache is /tmp/torchinductor_root = container-local already):
#   L1, L2  defaults, two independent fresh compiles → L2 vs L1 at ctx 0 / 30K (20/20 + median 0 = no lottery). Dense ruler on every arm
#           (the R188 promotion ruler, 192 s; its fresh-compile spread is unmeasured). The *.best_config files inductor wrote are saved per arm
#           and diffed L1 vs L2 (the mechanism, not just the symptom).
#   D1, D2  autotune_pointwise off, two fresh compiles → must be 20/20 (else the lottery is elsewhere); expect 0 best_config files.
#   B1      batch-sharded sampling ON under the D settings → vs D1: the R191 question with the confound removed.
#   Every arm also vs the R191 OFF-a dump (the daily's own numerics class) and vs the r173c bf16 decode reference; steps/s c1/c8/c16
#   per arm (D vs L = the price of determinism). If L1 ≡ L2 the D arms are skipped and B1 runs with default autotune (vs L1).
# Chain: waits for r190d-diag; r189-promote-pcieipc re-issued to wait for this unit (r189 stays the chain's daily restore).
#   unit: sudo systemd-run --unit=r193-determinism --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r193-determinism bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r190d-diag; do sleep 30; done; exec bash /srv/qwen5090/r193-determinism.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r193-determinism; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
DAILY_REF=/srv/qwen5090/results/2026-09-05-r191-bss-numerics   # dec-OFF-a-ctx{0,30000}.jsonl = the daily's numerics class
PINS="13980000000 13500000000"
NOAUTO='"inductor_compile_config":{"triton.autotune_pointwise":false}'
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] && [ -f "$DAILY_REF/dec-OFF-a-ctx0.jsonl" ] && [ -f "$DAILY_REF/dec-OFF-a-ctx30000.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q 'CC_EXTRA' "$CAND" || { log "ABORT: launch-daily.sh lacks the CC_EXTRA passthrough"; exit 3; }
grep -q 'EXTRA_ENV_APPEND' "$CAND" || { log "ABORT: launch-daily.sh lacks EXTRA_ENV_APPEND"; exit 3; }
for t in decode_ss.py decode_fidelity.py fidelity_compare.py fidelity_ladder.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R193 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R193 start (lock held): compile-artifact determinism on $IMG (PCIE_IPC=1) — L1, L2, [D1, D2], B1, every arm a fresh compile ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG [ENV=VAL ...]  — VLLM_CACHE_ROOT is container-local and empty, so the boot cannot load a saved AOT artifact
boot_arm(){ local tag=$1 kv rc saved loaded; shift
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG \
      EXTRA_ENV_APPEND="-e VLLM_CACHE_ROOT=/root/.cache/vllm-r193-$tag" "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log") aot_saved=$saved aot_loaded=$loaded $([ "$loaded" = 0 ] && [ "$saved" -ge 1 ] && echo FRESH-COMPILE || echo NOT-FRESH) compile_hash=$(grep -aoE 'torch_compile_cache/[0-9a-f]{10}/' "$R/engine-boot-$tag.log" | head -1 | tr -d / | cut -d/ -f2)"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|autotune_pointwise|inductor_compile_config" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
# the mechanism evidence: inductor writes one <kernel>.best_config per runtime-autotuned Triton kernel (autotune_local_cache); with
# autotune_pointwise off there is nothing to benchmark and no file. Saved per arm before teardown, diffed pairwise at the end.
grab_bestconfig(){ local tag=$1
  sudo docker exec vllm-exp bash -c 'cd /tmp/torchinductor_root 2>/dev/null && find . -name "*.best_config" | sort | tar cf - -T -' > "$R/bestconfig-$tag.tar" 2>/dev/null
  mkdir -p "$R/bestconfig-$tag" && tar xf "$R/bestconfig-$tag.tar" -C "$R/bestconfig-$tag" 2>/dev/null
  log "[$tag best_config] files=$(find "$R/bestconfig-$tag" -name '*.best_config' | wc -l) kernels=$(sudo docker exec vllm-exp bash -c 'ls /tmp/torchinductor_root 2>/dev/null | wc -l')"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
  log "[$T decode ctx$ctx vs daily(R191 OFF-a)] $(python3 $PR/decode_fidelity.py compare "$DAILY_REF/dec-OFF-a-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
# pair A B: A's dump compared against B's (B = the reference of the pair)
pair(){ local a=$1 b=$2 ctx
  for ctx in 0 30000; do [ -f "$R/dec-$a-ctx$ctx.jsonl" ] && [ -f "$R/dec-$b-ctx$ctx.jsonl" ] && log "[$a vs $b decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-$b-ctx$ctx.jsonl" "$R/dec-$a-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; done; }
same(){ local a=$1 b=$2 ctx ok=1   # 0 = identical at both ctx (20 fully agreeing chunks, median 0)
  for ctx in 0 30000; do python3 $PR/decode_fidelity.py compare "$R/dec-$b-ctx$ctx.jsonl" "$R/dec-$a-ctx$ctx.jsonl" 2>/dev/null | tail -1 | grep -q '"fully_agreeing": 20,.*"median_abs_dlogprob_on_agreed": 0\.0\b' || ok=0; done
  [ $ok = 1 ]; }
# arm TAG DENSE(0/1) [ENV=VAL...]
arm(){ local tag=$1 dense=$2; shift 2
  teardown; wipe_l2
  if boot_arm "$tag" "$@"; then
    sleep 20
    dfid "$tag" 0
    dfid "$tag" 30000
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 3 --kind code
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    [ "$dense" = 1 ] && ruler_dense "$tag"
    grab_bestconfig "$tag"
    log "[$tag engine error-lines] $(errs)"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
arm L1 1
arm L2 1
pair L2 L1
if same L2 L1; then
  log "L1 ≡ L2 at both contexts: no lottery on independent fresh compiles — skipping D1/D2; B1 runs with default autotune vs L1"
  arm B1 1 EXTRA_ARGS_APPEND=--enable-batch-sharded-sampling
  pair B1 L1
else
  log "L1 ≠ L2: lottery reproduced on independent fresh compiles — D arms (autotune_pointwise off) next"
  arm D1 1 CC_EXTRA="$NOAUTO"
  arm D2 1 CC_EXTRA="$NOAUTO"
  pair D2 D1; pair D1 L1
  arm B1 1 CC_EXTRA="$NOAUTO" EXTRA_ARGS_APPEND=--enable-batch-sharded-sampling
  pair B1 D1
fi
# mechanism: which runtime-autotuned kernels chose differently between the two default fresh compiles
if [ -d "$R/bestconfig-L1" ] && [ -d "$R/bestconfig-L2" ]; then
  diff -rq "$R/bestconfig-L1" "$R/bestconfig-L2" > "$R/bestconfig-L1-vs-L2.diff" 2>&1
  log "[best_config L1 vs L2] differing=$(grep -c '^Files .* differ' "$R/bestconfig-L1-vs-L2.diff") only-in-one=$(grep -c '^Only in' "$R/bestconfig-L1-vs-L2.diff") common=$(comm -12 <(cd "$R/bestconfig-L1" && find . -name '*.best_config' | sort) <(cd "$R/bestconfig-L2" && find . -name '*.best_config' | sort) | wc -l)"
fi
grep -aE "BOOT OK|BOOT FAILED|boot-err|RESULT|PROBE FAILED|decode ctx|vs bf16|best_config|error-lines|lottery" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
