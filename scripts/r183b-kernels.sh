#!/usr/bin/env bash
# R183b (2026-09-04): follow-up to r183-next-levers.sh, same results dir and helpers (copied verbatim; r183 is running and must not be
# edited). Two findings from the R183 BASE profile drive it:
#   (a) `--linear-backend X` filters EVERY layer type, and the RedHat checkpoint keeps FP8 modules: b12x died on "Failed to find a kernel
#       that can implement the ScaledMM linear layer". The clean ladder keeps the auto path and disables NVFP4 kernel classes one at a
#       time through VLLM_DISABLED_KERNELS (comma list; envs.py), cumulatively, in vLLM's priority order (FlashInferCutlass ->
#       FlashInferB12x -> Cutlass -> Marlin -> FlashInferTrtllm -> FlashInferCudnn -> Fbgemm -> B12x -> Emulation). The boot log's
#       "Using <kernel> for NVFP4 GEMM" names the selection; a selection already measured (by r183 or earlier in this ladder) is skipped,
#       Emulation stops the ladder. Decode-only profile: target GEMMs are 7.1 ms of a 14.4 ms busy c1 step at 27 us per call.
#   (b) 22% of the c1 decode step is inter-kernel gap inside graph replay (~1,400 small kernels/step: 551 elementwise, 292 triton fused,
#       229 fp4 quant, 169 other, 135 memcpy). The fusion passes reduce the count: fuse_norm_quant+fuse_act_quant (with and without
#       custom_ops +rms_norm,+silu_and_mul — the pass match counts are debug-level only, so both variants run), enable_qk_norm_rope_fusion,
#       fuse_rope_kvcache (touches the KV write path: the nvfp4 writer overlay is 0102, so the dense ruler gates it), and all three.
# The all-reduce arms of r183 are known no-ops on this box (BASE engine log: SymmMemCommunicator "capability 12.0 not supported",
# FlashInfer all-reduce "not supported for world_size=2"); they double as BASE replicates for boot-to-boot noise.
#   unit: sudo systemd-run --unit=r183b-kernels --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r183b-kernels bash /srv/qwen5090/r183b-kernels.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r183-next-levers; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; V0280=/srv/qwen5090/launch-daily-v0280.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4; DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
EB='--extra-body {"temperature":0.6}'
PROF_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true --profiler-config.delay_iterations=5 --profiler-config.max_iterations=60"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: references missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R183b $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R183b start (lock held): NVFP4 kernel ladder via VLLM_DISABLED_KERNELS (auto path, FP8 layers untouched) -> kernel-count fusion arms ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError'; }
layout(){ curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' '; }
# boot_arm TAG "pins" ENV=VAL... : launch-daily.sh EXP=1 SEQS=16 with the arm's env, first pin that boots; logs pool/layout/kernel/all-reduce/fusion lines
boot_arm(){ local tag=$1 pins=$2 kv rc; shift 2
  for kv in $pins; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB $(layout)"
      ELOG > "$R/engine-boot-$tag.log"
      log "[$tag gemm] $(grep -aoE 'Using \S+ for NVFP4 GEMM' "$R/engine-boot-$tag.log" | sort | uniq -c | tr '\n' ';')"
      log "[$tag allreduce] $(grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" "$R/engine-boot-$tag.log" | head -1)"
      grep -aiE "replaced [0-9]+ pattern|fusion pass|allreduce.?fusion|sequence.?parallel|async.?tp|local argmax|dynamic spec|eagerly" "$R/engine-boot-$tag.log" | head -8 | cut -c1-200 | sed "s/^/[$tag passes] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|unsupported|not supported|fall" | head -4 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
bat_decode(){ local T=$1
  p1 $T code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
  p1 $T prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $T code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $T code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  log "[$T errors after c16] $(errs)"; }
bat_ttft(){ local T=$1
  cap $T ttft8k   --ctx 8000   --conc 1 --tokens 8 --seed 41
  cap $T ttft36k  --ctx 36000  --conc 1 --tokens 8 --seed 42
  cap $T ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
  cap $T ttft240k --ctx 240000 --conc 1 --tokens 8 --seed 44; }
# arm TAG KINDS ENV... ; KINDS letters: d decode, t ttft, n numerics (dfid ctx0 + dense ruler), m MNBT extras (deep30k, dfid 30K, agentic), g needle gate
arm(){ local tag=$1 kinds=$2; shift 2
  wipe_l2
  boot_arm $tag "$PINS" "$@" || { log "[$tag] BOOT FAILED on every pin"; return 1; }
  sleep 20
  case $kinds in *d*) bat_decode $tag;; esac
  case $kinds in *t*) bat_ttft $tag;; esac
  case $kinds in *m*) p1 $tag deep30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000;; esac
  case $kinds in *n*) dfid $tag 0; ruler_dense $tag;; esac
  case $kinds in *m*) dfid $tag 30000; ruler_agentic $tag;; esac
  case $kinds in *g*) U=$U EVICT=12 bash $PR/needle_gate.sh "post-$tag" "$R"; log "[$tag needle gate] rc=$?";; esac
  log "[$tag engine error-lines] $(errs)"
  teardown; }


teardown
# ---------- kernel ladder ----------
declare -A SEEN
for k in $(grep -ahoE 'Using \S+ for NVFP4 GEMM' "$R"/engine-boot-*.log 2>/dev/null | awk '{print $2}' | sort -u); do SEEN[$k]=1; done
log "[ladder] already measured: ${!SEEN[*]}"
DIS=""
for k in FlashInferCutlassNvFp4LinearKernel FlashInferB12xNvFp4LinearKernel CutlassNvFp4LinearKernel MarlinNvFp4LinearKernel \
         FlashInferTrtllmNvFp4LinearKernel FlashInferCudnnNvFp4LinearKernel FbgemmNvFp4LinearKernel B12xNvFp4LinearKernel; do
  DIS="${DIS:+$DIS,}$k"; T="DK-${k%NvFp4LinearKernel}"; wipe_l2
  if boot_arm $T "$PINS" EXTRA_ENV_APPEND="-e VLLM_DISABLED_KERNELS=$DIS"; then
    sel=$(grep -aoE 'Using \S+ for NVFP4 GEMM' "$R/engine-boot-$T.log" | awk '{print $2}' | sort -u | head -1)
    if [ -z "$sel" ]; then log "[$T] no NVFP4 GEMM selection line — skipped"
    elif [ "$sel" = EmulationNvFp4LinearKernel ]; then log "[$T] selection is Emulation — ladder exhausted"; teardown; break
    elif [ -n "${SEEN[$sel]:-}" ]; then log "[$T] selected $sel, already measured — skipped"
    else SEEN[$sel]=1; log "[$T] MEASURING $sel"; sleep 20; bat_decode $T
      cap $T ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41; cap $T ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
      dfid $T 0; ruler_dense $T; log "[$T engine error-lines] $(errs)"; fi
    teardown
  else log "[$T] BOOT FAILED (disabled: $DIS)"; fi
done
# ---------- kernel-count fusion arms ----------
arm FU-nq    dn FUSIONS_APPEND='"fuse_norm_quant":true,"fuse_act_quant":true'
arm FU-nq-co dn FUSIONS_APPEND='"fuse_norm_quant":true,"fuse_act_quant":true' CC_EXTRA='"custom_ops":["+rms_norm","+silu_and_mul"]'
arm FU-qk    dn FUSIONS_APPEND='"enable_qk_norm_rope_fusion":true'
arm FU-rk    dn FUSIONS_APPEND='"fuse_rope_kvcache":true'
arm FU-kern  dn FUSIONS_APPEND='"fuse_norm_quant":true,"fuse_act_quant":true,"enable_qk_norm_rope_fusion":true,"fuse_rope_kvcache":true'
grep -aE "BOOT OK|BOOT FAILED|FELL BACK|MEASURING|skipped|exhausted|gemm\]|allreduce\]|passes\]|cap |RESULT|PROBE FAILED|error-lines|errors after|vs bf16|decode ctx|needle-gate" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
