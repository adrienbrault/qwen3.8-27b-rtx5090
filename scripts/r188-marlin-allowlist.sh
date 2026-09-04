#!/usr/bin/env bash
# R188 (2026-09-05): per-layer NVFP4 GEMM kernel selection (patch 0139, codex gpt-6-astra from BRIEF22, image ...-fi0616-marlinlist).
# R183b showed the GEMM kernel is a fidelity knob: MarlinNvFp4LinearKernel on every NVFP4 layer is 94.051% top-1 / +0.207% PPL vs
# bf16 against FlashInferCutlass's 92.771% / +0.744%, at −7.8% c8 / −17.3% c16 / +32% TTFT@100K. 0139 adds VLLM_SM12X_NVFP4_MARLIN_LAYERS
# (Python regex over the vLLM module prefix; unset = byte-identical selection) so the fidelity can be bought per layer. Seven arms:
#   CTRL      knob unset on the 0139 image. With the knob unset the diff cannot reach the original ladder (matches_allowlist(None) is
#             False), so the equivalence proof is the two proof lines: 0 allowlist lines AND the same "Using X for NVFP4 GEMM" selection
#             as R183 BASE. A nonzero count or a changed selection line voids the sheet. The dense ruler should then reproduce R183
#             BASE (92.771% / 40,162 flips / KL 0.014082; R183's replicates were bit-identical) — a few flips off with the proofs
#             intact is image-layer/boot drift of the R185 kind (92.774 on the pcieipc image); CTRL, not R183 BASE, is the
#             within-unit yardstick for the six Marlin arms either way.
#   M-ALL     every NVFP4 module (layers 0–55, gate_up + down) through the allowlist = R183b's DK-Cutlass arm by another route; MUST
#             reproduce 94.051% / 33,047 / 0.010046. Proves the allowlist path ≡ the ladder's Marlin selection.
#   M-DOWN    down_proj only, layers 0–55          M-GATEUP  gate_up_proj only, layers 0–55
#   M-L0-18 / M-L19-37 / M-L38-55   both projections, one third of the NVFP4 layers each
# Every arm: proof of path (allowlist line count vs expectation, both TP ranks log into one container so counts are 2× per-rank;
# the env as seen inside the container; the "Using X for NVFP4 GEMM" selection lines) + decode_ss c1/c8/c16 + ttft 8k/120k + decode
# ruler ctx0 and 30K vs the r173c bf16 reference + dense ruler + agentic ruler vs the R156 bf16 dumps. Queued behind r187-flags
# (GPU_QUEUE_NAME registers it so R187's finish() skips the daily restore); finish restores the daily unless another unit is queued.
# Rank arms on dense PPL / top-1 first, then agentic, then the decode cost; the interesting result is an arm that keeps most of
# M-ALL's fidelity gain at a fraction of its c8/c16/TTFT cost.
#   unit: sudo systemd-run --unit=r188-marlin --collect -p User=adrienbrault -p RuntimeMaxSec=86400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r188-marlin bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r187-flags; do sleep 30; done; exec bash /srv/qwen5090/r188-marlin-allowlist.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r188-marlin-allowlist; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-marlinlist
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
KNOB=VLLM_SM12X_NVFP4_MARLIN_LAYERS
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: references missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in decode_ss.py kv_capacity_probe.py decode_fidelity.py fidelity_ladder.py fidelity_compare.py agentic_ref.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R188 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R188 start (lock held): per-layer Marlin allowlist (0139) on $IMG — CTRL, M-ALL, M-DOWN, M-GATEUP, M-L0-18, M-L19-37, M-L38-55 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
layout(){ curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' '; }
# boot_arm TAG EXPECTED_ALLOWLIST_LINES [ENV=VAL ...]
boot_arm(){ local tag=$1 expect=$2 kv rc n; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB $(layout)"
      ELOG > "$R/engine-boot-$tag.log"
      # proof of path: the knob as the container sees it, the 0139 helper module present, allowlist selection lines (2 ranks), GEMM selections
      log "[$tag env] $(sudo docker exec vllm-exp bash -c "echo $KNOB=\${$KNOB:-unset}; python3 -c 'import vllm.nvfp4_marlin_allowlist as m; print(\"0139 helper present\")' 2>&1 | tail -1" 2>&1 | tr '\n' ' ')"
      n=$(grep -ac 'NVFP4 Marlin allowlist: MarlinNvFp4LinearKernel for ' "$R/engine-boot-$tag.log")
      if [ "$n" = "$expect" ]; then log "[$tag proof] allowlist lines $n = expected $expect OK"; else log "[$tag proof] allowlist lines $n != expected $expect MISMATCH"; fi
      log "[$tag proof-first] $(grep -a 'NVFP4 Marlin allowlist' "$R/engine-boot-$tag.log" | head -2 | sed -E 's/^.*(NVFP4 Marlin allowlist)/\1/' | cut -c1-140 | tr '\n' ' ')"
      log "[$tag gemm] $(grep -aoE 'Using [A-Za-z0-9]+ for NVFP4 GEMM' "$R/engine-boot-$tag.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|unsupported|not supported|not compatible|JointFailure|ValueError|allowlist" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
bat_decode(){ local T=$1
  p1 $T code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
  p1 $T prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $T code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $T code-c16 --conc 16 --tokens 1024 --runs 1 --kind code; }
# arm TAG EXPECTED_LINES [REGEX]   (no regex = knob unset)
arm(){ local tag=$1 expect=$2 rx=${3:-}; local -a ev=()
  [ -n "$rx" ] && ev=(EXTRA_ENV_APPEND="-e $KNOB=$rx")
  teardown; wipe_l2
  if boot_arm "$tag" "$expect" "${ev[@]}"; then
    sleep 20
    bat_decode "$tag"
    cap $tag ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41
    cap $tag ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
    dfid "$tag" 0
    dfid "$tag" 30000
    ruler_dense "$tag"
    ruler_agentic "$tag"
    log "[$tag engine error-lines] $(errs)"
  else log "[$tag] BOOT FAILED on every pin"; fi; }

# module names are model.language_model.layers.<n>.mlp.{gate_up_proj,down_proj}; layers 56–63 are FP8 and never enter the branch.
# expected line counts = codex's per-rank counts × 2 ranks (56/56/112 per rank; thirds 19/19/18 layers × 2 modules per rank)
# reader check: MISMATCH at exactly half everywhere (112/56/56/38/38/36) = rank-0-only logging, not a patch failure — the proof-first
# and gemm lines disambiguate; MISMATCH at 0 with a regex set = the knob did not reach the container (see the env line).
arm CTRL     0
arm M-ALL    224 'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.(gate_up|down)_proj$'
arm M-DOWN   112 'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.down_proj$'
arm M-GATEUP 112 'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.gate_up_proj$'
arm M-L0-18  76  'layers\.([0-9]|1[0-8])\.mlp\.(gate_up|down)_proj$'
arm M-L19-37 76  'layers\.(19|2[0-9]|3[0-7])\.mlp\.(gate_up|down)_proj$'
arm M-L38-55 72  'layers\.(3[89]|4[0-9]|5[0-5])\.mlp\.(gate_up|down)_proj$'
grep -aE "BOOT OK|BOOT FAILED|boot-err| env\]| proof| gemm\]|RESULT|PROBE FAILED|cap |decode ctx|vs bf16|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
