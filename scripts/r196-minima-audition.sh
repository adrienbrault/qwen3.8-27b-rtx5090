#!/usr/bin/env bash
# R196 (2026-09-05, user "We should try https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4"): audition of the minima-ai NVFP4 checkpoint
# on the daily route. What it is (card + config + paper arXiv:2609.04098): llm-compressor NVFP4 W4A4 group 16 on ALL 496 backbone linears —
# the 48 GDN layers' in_proj_qkv/z/a/b + out_proj included, with the vLLM-fused groups (qkv+z, b+a) rewritten to shared global scales —
# lm_head / embeddings / conv1d / norms bf16, static fp8 KV scales, 128×32K calibration, no QAT. 18.8 GB single safetensors, text-only
# Qwen3_5ForCausalLM, no MTP head. Card numbers vs bf16: WikiText-2 PPL@4K 7.68 vs 6.95, @32K 10.50 vs 10.35, MMLU-Pro 79.7 vs 80.4,
# GPQA-Diamond 85.1 vs 86.5, GSM8K/AIME matched. The daily RedHat checkpoint is MIXED: attention + GDN qkv/z/out + lm_head + the last 8
# MLP layers in FP8, the rest NVFP4 (22 GB); GDN in_proj_a/b bf16. So this is "all-NVFP4 incl. GDN" vs "FP8 where it hurts", ~3.5 GB less
# weight VRAM. Rulers = the R180 set vs the R156 bf16 dumps (dense 724,781 positions, agentic 57,972, decode ctx0/30K), then drafter fit
# (decode_ss acceptance: the syvai DFlash2 drafter was trained on the bf16 target; a farther checkpoint accepts less), then tool-eval 69×4.
# Two arms, same image, same session, control first: H = RedHat (the daily checkpoint), M = minima. Both on the daily route exactly
# (EXP=1 :8029, SEQS 16, PCIE_IPC=1, BSS=1, pin 13.98 GB → 13.5 GB fallback, nvfp4 KV, DFlash2 ns9 syvai). Numerics caveat (R193/R193d):
# each arm compiles or loads its own artifact (different weights = different hash), so the bf16 distances carry the lottery noise
# (±0.035 % dense PPL, ±0.003 at 30K); checkpoint differences of the R156 size (RedHat +0.37 % vs gittensor +4.46 %) dwarf it.
# Weights: /srv/qwen5090/models/qwen3.8-27b-minima-nvfp4 (fetch-minima unit, hf download, 2026-09-05 11:30 UTC). Restores the daily at the end.
# Pre-flight (advisor, 2026-09-05 11:47 UTC): download complete (model.safetensors 18,788,354,104 B, "Downloaded" in .fetch.log);
# Qwen3_5ForCausalLM is registered in the image (first text-only checkpoint on this route: every other one is the ForConditionalGeneration
# wrapper); --limit-mm-per-prompt is only consumed on the supports_multimodal branch (vllm/config/model.py:784); the DFlash target taps
# fall back to the model itself when get_language_model() is absent (spec_decode/dflash/utils.py:46-54). generation_config.json is
# identical to RedHat's; chat_template.jinja is NOT (see the swap before arm M). Prior: the card's own WikiText number is +10.5 % PPL
# from bf16 at 4K (+1.4 % at 32K) against RedHat's +0.37 % on our dense ruler — expect M to lose the fidelity rulers; the informative
# outputs are WHERE (dense vs agentic vs decode 30K), the drafter's acceptance on it, and the R156 arm G (QUASAR, all-GDN-NVFP4 +
# bf16 lm_head) row for comparison. Arm M's min_free at the same pin is the headroom a larger pin would have on this checkpoint.
#   unit: sudo systemd-run --unit=r196-minima-audition --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r196-minima-audition bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r196-minima-audition.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r196-minima-audition; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
MINIMA=/srv/qwen5090/models/qwen3.8-27b-minima-nvfp4; REDHAT=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$MINIMA/model.safetensors" ] && [ -f "$MINIMA/config.json" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q 'then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}' "$CAND" || { log "ABORT: launch-daily.sh lacks the R187 PCIE_IPC default fix"; exit 3; }
grep -q 'CAND_MODEL' "$CAND" || { log "ABORT: launch-daily.sh lacks the R196 CAND_MODEL passthrough"; exit 3; }
for t in decode_ss.py decode_fidelity.py fidelity_compare.py fidelity_ladder.py agentic_ref.py tooleval_summary.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R196 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R196 start (lock held): minima-ai NVFP4 audition on the daily route, $IMG (PCIE_IPC=1 BSS=1) — H (RedHat control), M (minima) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG MODEL_DIR [ENV=VAL ...]
boot_arm(){ local tag=$1 mdir=$2 kv rc n; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG CAND_MODEL=$mdir "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      n=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log")
      log "[$tag] BOOT OK model=$(basename $mdir) pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB weights=$(grep -aoE 'Loading weights took [0-9.]+ seconds|Model loading took [0-9.]+ GiB[^\n]{0,40}' "$R/engine-boot-$tag.log" | head -1) pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$n aot_saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log") aot_loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log") compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
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
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
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
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    tooleval "$tag"
    log "[$tag engine error-lines] $(errs)  preemptions=$(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
arm H "$REDHAT"
# Chat-template confound (pre-flight): minima ships an older Qwen template (no system-message merging, no reasoning_effort high→xhigh
# mapping; 8,952 vs 9,993 bytes, first difference at line 45). The launcher passes no --chat-template and the checkpoint dir is the only
# model mount, so arm M serves RedHat's template from its own dir (the original is kept as chat_template.jinja.minima-orig): the agentic
# ruler and tool-eval then measure the quantization, not the template.
[ -f "$MINIMA/chat_template.jinja.minima-orig" ] || cp "$MINIMA/chat_template.jinja" "$MINIMA/chat_template.jinja.minima-orig"
cp "$REDHAT/chat_template.jinja" "$MINIMA/chat_template.jinja"
log "[M] chat_template.jinja := RedHat's ($(cmp -s "$MINIMA/chat_template.jinja" "$REDHAT/chat_template.jinja" && echo identical || echo DIFFERS)); original kept as chat_template.jinja.minima-orig"
arm M "$MINIMA"
for ctx in 0 30000; do [ -f "$R/dec-H-ctx$ctx.jsonl" ] && [ -f "$R/dec-M-ctx$ctx.jsonl" ] && log "[M vs H decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-H-ctx$ctx.jsonl" "$R/dec-M-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; done
grep -aE "BOOT OK|BOOT FAILED|boot-err|RESULT|PROBE FAILED|decode ctx|vs bf16|tool-eval|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
