#!/usr/bin/env bash
# R183 (2026-09-04, user "Ok go, try all" / "Not 7"): the next-lever chain on the promoted bf16-SSM daily config (R182), one unit,
# :8029 via launch-daily.sh EXP=1 (SEQS 16, pin 13.98 GB -> 13.5 fallback), eval-l2 wiped before every arm so no arm serves another
# arm's KV. Every arm reads the same battery against BASE (same chain, same hour). Stages, in order:
#   1. BASE + torch profiler: c1/c8/c16 kernel tables per TP rank (R158 recipe: benchy pp2048 tg256, 60 steps after 5) -> where the
#      13 ms step goes now that the drafter is graphed, and whether rank 0 carries more than rank 1 (GPU0 hotter under load).
#   2. NVFP4 GEMM ladder (item 6): --linear-backend b12x | flashinfer_b12x | cutlass | marlin | flashinfer_cudnn | flashinfer_trtllm |
#      humming vs the daily's FlashInferCutlassNvFp4LinearKernel. R158: GEMM = 10.2 ms/step at c1 vs a ~5 ms weight-bandwidth floor.
#      An arm whose boot log does not name the expected kernel FELL BACK (vLLM falls back at startup) and is not measured.
#   3. TP2 all-reduce tax (item 2; R158: NCCL 16 ms of a 46 ms step at c8): VLLM_ALLREDUCE_USE_FLASHINFER=1, VLLM_ALLREDUCE_USE_SYMM_MEM=1,
#      pass fuse_allreduce_rms (+FI env), enable_sp+fuse_gemm_comms (async TP), all three, and use_local_argmax_reduction (drafter
#      argmax without the O(vocab) all-gather at draft_tp2). The "all-reduce backends (in dispatch order)" line and the pattern-
#      replacement counts are logged: zero replacements = the pass did nothing. fuse_allreduce_rms with 0 matches re-runs with
#      custom_ops +rms_norm (CC_EXTRA).
#   4. Prefill chunk (item 3): EXP_MNBT 16384 / 32768 (the 8192 cap was the Bug B dodge; Bug B was XQA-specific and XQA is off).
#      TTFT ladder 8K/36K/120K/240K, deep30k decode, code c8, dense+agentic rulers, decode ruler at ctx0/ctx30000, needle gate on 32768.
#   5. Speculation policy (item 4): num_speculative_tokens_per_batch_size [[1,4,9],[5,12,5],[13,64,3]] and [[1,8,9],[9,64,1]]
#      (ns9 at c1, shorter drafts when the verify batch is wide), and draft_tp1 (R173: +8% c1 under the old headroom floor).
#      Error lines are counted right after c16 (dynamic ns crosses a graph-capture boundary the boot asserts cannot see).
#   6. DP2 (item 2, structural): two one-card engines (TP=1, --data-parallel-size 2, draft_tp1, no tier) through launch-daily-v0280.sh
#      directly; c8/c16 aggregate vs BASE, per-rank pool. Boot failure is acceptable; last arm.
# Fidelity: the dense ruler (724,781 positions vs the R156 bf16 dump) on every numerics-changing arm (GEMM, fusion, FI all-reduce,
# MNBT), the 20-chunk decode ruler (vs the r173c bf16 decode reference) as a screen; agentic on the MNBT arms.
# The campaign (item 5: miniswe W=16, RUN_SUFFIX -r183-bf16-w16, cold tier) is a separate unit queued on the GPU lock behind this one,
# so finish() skips the daily restore and the campaign's own finish restores it.
#   unit: sudo systemd-run --unit=r183-next-levers --collect -p User=adrienbrault -p RuntimeMaxSec=21600 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r183-next-levers bash /srv/qwen5090/r183-next-levers.sh
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
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R183 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R183 next-levers start (lock held): BASE+prof -> GEMM ladder -> all-reduce -> MNBT -> spec policy -> DP2 ==="
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

# ---------- 1. BASE + profiler ----------
teardown
mkdir -p "$R/prof-BASE"; chmod 777 "$R/prof-BASE"
capture(){ local conc=$1; local d="$R/prof-BASE/c$conc"; mkdir -p "$d"
  curl -s -m 10 -X POST $U/start_profile >/dev/null 2>&1 || log "[prof c$conc] start_profile returned non-2xx (continuing)"
  timeout 900 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 \
    --concurrency $conc --runs 1 --no-cache --latency-mode api --format json $EB --save-result "$d/benchy.json" > "$d/benchy.out" 2>&1
  curl -s -m 60 -X POST $U/stop_profile >/dev/null 2>&1 || true
  for i in $(seq 60); do n=$(ls "$R/prof-BASE"/*.json* 2>/dev/null | wc -l); [ "$n" -ge 2 ] && break; sleep 3; done
  sleep 15; mv "$R/prof-BASE"/*.json* "$d/" 2>/dev/null
  ls -la "$d" | grep -v "^total" | sed "s/^/[prof c$conc] /" | tee -a "$R/audit.log"
  for t in "$d"/*.json*; do [ -f "$t" ] || continue
    python3 $PR/prof_summary.py "$t" --steps 60 --top 22 > "$t.summary.txt" 2>&1
    head -18 "$t.summary.txt" | sed "s/^/[prof c$conc] /" | tee -a "$R/audit.log"
    python3 $PR/prof_cpu.py "$t" --steps 60 > "$t.cpu.txt" 2>&1; head -12 "$t.cpu.txt" | sed "s/^/[prof-cpu c$conc] /" | tee -a "$R/audit.log"
  done; }
wipe_l2
if boot_arm BASE "$PINS" EXTRA_ARGS_APPEND="$PROF_ARGS" EXTRA_MOUNT_APPEND="-v $R/prof-BASE:/prof"; then
  sleep 20
  capture 1; capture 8; capture 16
  bat_decode BASE; bat_ttft BASE; dfid BASE 0; ruler_dense BASE
  p1 BASE deep30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
  log "[BASE engine error-lines] $(errs)"; teardown
else log "[BASE] BOOT FAILED on every pin"; fi

# ---------- 2. NVFP4 GEMM ladder ----------
declare -A KERN=([b12x]=B12xNvFp4LinearKernel [flashinfer_b12x]=FlashInferB12xNvFp4LinearKernel [cutlass]=CutlassNvFp4LinearKernel [marlin]=MarlinNvFp4LinearKernel
                 [flashinfer_cudnn]=FlashInferCudnnNvFp4LinearKernel [flashinfer_trtllm]=FlashInferTrtllmNvFp4LinearKernel [humming]=HummingNvFp4LinearKernel)
for lb in b12x flashinfer_b12x cutlass marlin flashinfer_cudnn flashinfer_trtllm humming; do
  T="LB-$lb"; wipe_l2
  if boot_arm $T "$PINS" EXTRA_ARGS_APPEND="--linear-backend $lb"; then
    if grep -aq "Using ${KERN[$lb]} for NVFP4 GEMM" "$R/engine-boot-$T.log"; then
      sleep 20; bat_decode $T; cap $T ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41; cap $T ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
      dfid $T 0; ruler_dense $T; log "[$T engine error-lines] $(errs)"
    else log "[$T] FELL BACK (expected ${KERN[$lb]}; got: $(grep -aoE 'Using \S+ for NVFP4 GEMM' "$R/engine-boot-$T.log" | sort -u | tr '\n' ';')) — not a measurement, skipped"; fi
    teardown
  else log "[$T] BOOT FAILED"; fi
done

# ---------- 3. all-reduce / fusion / argmax ----------
FI_ENV="-e VLLM_ALLREDUCE_USE_FLASHINFER=1"
arm AR-fi     dn EXTRA_ENV_APPEND="$FI_ENV"
arm AR-symm   dn EXTRA_ENV_APPEND="-e VLLM_ALLREDUCE_USE_SYMM_MEM=1"
arm FU-arrms  dn EXTRA_ENV_APPEND="$FI_ENV" FUSIONS_APPEND='"fuse_allreduce_rms":true'
if [ "$(grep -aoE 'replaced [0-9]+' "$R/engine-boot-FU-arrms.log" 2>/dev/null | awk '{s+=$2} END{print s+0}')" = 0 ]; then
  log "[FU-arrms] 0 pattern replacements — re-running with custom_ops +rms_norm"
  arm FU-arrms2 dn EXTRA_ENV_APPEND="$FI_ENV" FUSIONS_APPEND='"fuse_allreduce_rms":true' CC_EXTRA='"custom_ops":["+rms_norm"]'
fi
arm FU-sptp   dn FUSIONS_APPEND='"enable_sp":true,"fuse_gemm_comms":true'
arm FU-all    dn EXTRA_ENV_APPEND="$FI_ENV" FUSIONS_APPEND='"fuse_allreduce_rms":true,"enable_sp":true,"fuse_gemm_comms":true'
arm SP-argmax d  SPEC_EXTRA='"use_local_argmax_reduction":true'

# ---------- 4. prefill chunk ----------
arm M16 dtnm  EXP_MNBT=16384
arm M32 dtnmg EXP_MNBT=32768

# ---------- 5. speculation policy ----------
arm SP-dyn1 d SPEC_EXTRA='"num_speculative_tokens_per_batch_size":[[1,4,9],[5,12,5],[13,64,3]]'
arm SP-dyn2 d SPEC_EXTRA='"num_speculative_tokens_per_batch_size":[[1,8,9],[9,64,1]]'
arm SP-dtp1 d SPEC_DTP=1

# ---------- 6. DP2 (two one-card engines) ----------
wipe_l2
SPEC_DP='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER"}'
if env -i PATH="$PATH" HOME="$HOME" USER="$USER" IMAGE="$IMG" MODEL_DIR="$MODEL" TP=1 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 \
     UTIL=0.90 MAXLEN=262144 NO_TIER=1 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 KVD_OVERRIDE=nvfp4 FIWS=536870912 ALLOW_NO_XQA=1 \
     EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$SPEC_DP" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1" \
     EXTRA_ARGS="--data-parallel-size 2 --mamba-ssm-cache-dtype bfloat16" bash $V0280 > "$R/boot-DP2.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then
  ELOG > "$R/engine-boot-DP2.log"
  log "[DP2] BOOT OK pools: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$R/engine-boot-DP2.log" | tr '\n' ';') seqs/rank 8 $(layout)"
  sleep 20; bat_decode DP2; cap DP2 ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41; cap DP2 ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
  cap DP2 short16 --ctx 0 --conc 16 --tokens 400 --ignore-eos --seed 45
  log "[DP2 engine error-lines] $(errs)"
else log "[DP2] BOOT FAILED: $(grep -aE 'FAILED|Error' "$R/boot-DP2.log" | tail -1 | cut -c1-200)"; ELOG 2>/dev/null | grep -aiE "error|exception" | head -4 | cut -c1-200 | sed 's/^/[DP2 boot-err] /' | tee -a "$R/audit.log"; fi
teardown
grep -aE "BOOT OK|BOOT FAILED|FELL BACK|gemm\]|allreduce\]|passes\]|cap |RESULT|PROBE FAILED|error-lines|errors after|vs bf16|decode ctx|needle-gate|prof c[0-9]+\] (BASE|wall|GPU|kernel|rank)" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
