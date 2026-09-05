#!/usr/bin/env bash
# R201 (2026-09-06, user: "Wdyt of https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga ?"): audition of the Kearuga checkpoint on the daily
# route, R196 template. What it is (card + hf_quant_config + requant/materialize reports, rev 1a7f4231): ModelOpt MIXED_PRECISION export of
# Qwen3.8-27B — attention q/k/v/o (15 of 16 layers) + GDN in_proj_qkv/z/out_proj (45 of 48 layers) FP8 W8A8 static; MLP gate/up/down of
# layers 2-61 W4A16_NVFP4 (weight-only 4-bit, bf16 activations -> the Marlin path on our route, R60c/R188/R192) re-quantized by GPTQ from bf16
# with Hessians ("4o6": per-16 block scale exponent chosen per block; down_proj from an AWQ donor, pre_quant_scale dropped); boundary MLPs
# (layers 0, 1, 62, 63) FP8; embed/lm_head/norms/vision/MTP bf16; NO KV-cache scales (kv_cache_quant_algo null). 24.85 GB (RedHat 22 GB:
# ~1.4 GB more per card at TP2, so the 13.98 GB pin may miss the 512 MiB floor and fall back to 13.5 GB — that pool delta is part of the price).
# Card claims (DGX Spark, SGLang, bf16 KV, K=10 DFlash2): held-out full-vocab KL 0.0208, top-1 95.0 % vs bf16; not our instrument. Prior from
# our own ladder: the same protection pattern as RedHat plus W4A16 MLPs is the R188/R192 Marlin class — dense gap +0.67 -> +0.21 % on RedHat's
# weights, at c8 -4 % / c16 -7 % and prefill +10-31 %. So the expected outcome is "closer to bf16 than the daily, slower at c8/c16 and prefill";
# the open questions are the size of each, the pool at the pin, the nvfp4-KV route without checkpoint KV scales (engine-quant log lines), and
# the drafter's acceptance. Single arm K; the H control is R196's arm H (same image, same route, 2026-09-05 14:22 UTC). Rulers vs the R156
# bf16 dumps; decode_ss code-c1 x3 / prose-c1 / code-c8 / prose-c8 / code-c16; tool-eval 69x4; needle gate (131K + 220K + evicted re-asks
# through the eval-l2 tier). Chat template: Kearuga copies the base's; if it differs from RedHat's it is swapped to RedHat's for the run
# (R196 rule: the agentic ruler and tool-eval measure the quantization, not the template), original kept as chat_template.jinja.kearuga-orig.
#   unit run 5 (2026-09-05 23:45 UTC, differential): add -E XARGS_K='--linear-backend torch' to the line below (run 4 'cutlass' cannot boot, see above).
#   unit (re-issued 2026-09-06 after the cfgfix): sudo systemd-run --unit=r201-kearuga --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r201-kearuga bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r201-kearuga-audition.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r201-kearuga; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
# R201 boot 1 (2026-09-05 22:41 UTC) died in load_weights: `no module or parameter named 'layers.0.mlp.down_proj.input_scale'` — the
# checkpoint's config.json quantized_layers (375, what vLLM reads first) lacks the 10 FP8 boundary MLPs that hf_quant_config.json (385)
# and the tensors declare, so vLLM built them unquantized. probes/merge_quantized_layers.py writes the -cfgfix sibling (hard links,
# config.json = union of both maps); that sibling is what boots here.
# R201 runs 2 and 3 (2026-09-05 23:02 and 23:34 UTC) both lost the engine in the dense ruler's prefill (Xid 13 "illegal instruction" in the
# GDN core at doc 523; CUDA "unknown error" in a piecewise-graph replay at doc 619): two faults, two docs, two GPUs. The one kernel this
# checkpoint exercises that the daily route never runs is FlashInferFP8ScaledMMLinearKernel (the ModelOpt static-FP8 attention/GDN/boundary
# projections; vLLM gates it on compute capability >= 100, so sm120 gets the sm100 kernel). Run 4 (--linear-backend cutlass) could not
# boot: the "cutlass" set contains CutlassW4A8LinearKernel, so the WNA16 filter does not fall back and the W4A16 MLPs find no kernel.
# Run 5 sets XARGS_K="--linear-backend torch": that set holds only PerTensorTorchFP8ScaledMMLinearKernel (torch._scaled_mm, cuBLASLt,
# per-tensor FP8 — the same numerics class as shipped), every other layer type falls back to auto (Marlin for W4A16). A clean full
# battery there pins the faults on the FlashInfer FP8 GEMM on sm120 (which matters for every FP8-attention candidate).
XARGS_K=${XARGS_K:-}
KEARUGA=/srv/qwen5090/models/qwen3.8-27b-kearuga-cfgfix; REDHAT=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$KEARUGA/model.safetensors.index.json" ] && [ -f "$KEARUGA/config.json" ] && [ -f "$KEARUGA/model-00003-of-00003.safetensors" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q 'then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}' "$CAND" || { log "ABORT: launch-daily.sh lacks the R187 PCIE_IPC default fix"; exit 3; }
grep -q 'CAND_MODEL' "$CAND" || { log "ABORT: launch-daily.sh lacks the R196 CAND_MODEL passthrough"; exit 3; }
for t in decode_ss.py decode_fidelity.py fidelity_compare.py fidelity_ladder.py agentic_ref.py tooleval_summary.py needle_gate.sh; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R201 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R201 start (lock held): Kearuga (0xWhiteMage, rev 1a7f4231) audition on the daily route, $IMG (PCIE_IPC=1 BSS=1) — arm K; H control = R196 arm H ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG MODEL_DIR [ENV=VAL ...]
boot_arm(){ local tag=$1 mdir=$2 kv rc n; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG CAND_MODEL=$mdir EXTRA_ARGS_APPEND="${XARGS_K:-}" "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK model=$(basename $mdir) pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB weights=$(grep -aoE 'Loading weights took [0-9.]+ seconds|Model loading took [0-9.]+ GiB[^\n]{0,40}' "$R/engine-boot-$tag.log" | head -1) pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$n aot_saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log") aot_loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log") compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
      grep -aiE "quantization|modelopt|marlin|nvfp4|kv[_ ]?scale|k_scale|scaling factor|W4A16|fp8" "$R/engine-boot-$tag.log" | grep -avE "deprecated|Unknown vLLM env" | sed -E 's/^[^ ]* //' | sort -u | head -14 | cut -c1-220 | sed "s/^/[$tag engine-quant] /" | tee -a "$R/audit.log"
      log "[$tag layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|quantiz|kv_cache_scheme|k_scale" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
ruler_dense(){ local T=$1
  # R201 run 2 (2026-09-05 23:02 UTC): the engine died at doc 523 with Xid 13 / Triton "illegal instruction" inside the GDN core
  # (qwen_gdn_attention_core_fused_norm_packed, piecewise graph); the 522 scored docs stay in the dump and --resume continues at 523.
  # If the engine dies again the ruler is void: log it loudly (ENGINE DIED) instead of scoring a partial dump.
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense --resume > "$R/score-$T-dense.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the dense ruler ($(grep -ac '^\[warn\]' "$R/score-$T-dense.out") docs failed): $(sudo dmesg -T | grep -a Xid | tail -1 | cut -c1-160); $(sudo docker logs vllm-exp 2>&1 | grep -a 'Error \[CUDA\]\|illegal' | head -1 | cut -c1-160)"; finish ENGINE-DIED; exit 5; }
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
tooleval(){ local T=$1
  ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$T.json" > "$R/tooleval-$T.log" 2>&1 )
  python3 $PR/tooleval_summary.py "$R/tooleval-$T.json" "$T" 2>&1 | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
# arm TAG MODEL_DIR
arm(){ local tag=$1 mdir=$2; shift 2
  teardown; wipe_l2
  if boot_arm "$tag" "$mdir" "$@"; then
    sleep 20
    ruler_dense "$tag"
    ruler_agentic "$tag"
    dfid "$tag" 0
    dfid "$tag" 30000
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
    p1 $tag prose-c8 --conc 8 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    tooleval "$tag"
    log "[$tag needle] gate start (131K + 220K cold, evicted re-asks through the eval-l2 tier)"
    U=$U bash $PR/needle_gate.sh "$tag" "$R" > "$R/needle-$tag.log" 2>&1; rc=$?
    log "[$tag needle] rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-$tag.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
    log "[$tag engine error-lines] $(errs)  preemptions=$(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
if cmp -s "$KEARUGA/chat_template.jinja" "$REDHAT/chat_template.jinja"; then log "[K] chat_template.jinja identical to RedHat's: served as is"
else
  [ -f "$KEARUGA/chat_template.jinja.kearuga-orig" ] || cp "$KEARUGA/chat_template.jinja" "$KEARUGA/chat_template.jinja.kearuga-orig"
  cp "$REDHAT/chat_template.jinja" "$KEARUGA/chat_template.jinja"
  log "[K] chat_template.jinja := RedHat's (original $(wc -c < "$KEARUGA/chat_template.jinja.kearuga-orig") B differs; kept as chat_template.jinja.kearuga-orig)"
fi
arm K "$KEARUGA"
grep -aE "BOOT OK|BOOT FAILED|boot-err|engine-quant|layout|RESULT|PROBE FAILED|decode ctx|vs bf16|tool-eval|needle|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
