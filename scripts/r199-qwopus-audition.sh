#!/usr/bin/env bash
# R199 (2026-09-05, backlog "Qwopus3.8-27B-Flash audition"): Jackrong/Qwopus3.8-27B-Flash is a fine-tune of Qwen3.8-27B (bf16 rev 072b8c27,
# 18 shards, config identical to the base but for the model_name/pad_token/torch_dtype keys; tokenizer vocab + merges + added tokens
# byte-identical to the base, so every position-aligned ruler transfers). Two community NVFP4 quants of it were fetched (r199-fetch-qwopus.sh):
#   X = Shiftedx/Qwopus3.8-27B-Flash-NVFP4-MTP (e46dcfbe): ModelOpt NVFP4 W4A4 g16 on EVERY linear except lm_head/embed/conv1d/in_proj_a/b
#       (GDN qkv/z/out_proj quantized like gittensor/minima), fp8 KV scales, MTP head kept (15 mtp.* tensors), 19 GB. The like-for-like arm.
#   J = sojufx/Qwopus3.8-27B-Flash-NVFP4 (bdd5036f): ModelOpt MIXED_PRECISION = attention + GDN projections FP8 W8A8, MLP gate/up/down
#       W4A16_NVFP4 (weight-only 4-bit, bf16 activations -> Marlin path, R60c/R188), MTP + vision bf16, 23 GB. The fidelity arm.
# Both share the Qwopus chat template (Qwen3-style: tools in the system turn, <think> handling, no vision/reasoning_effort blocks); it is
# byte-identical across the bf16 and both quants, so every Qwopus arm serves the MODEL'S OWN template (unlike R196, no swap): tool-eval and
# the agentic ruler then measure the checkpoint, not a template mismatch. The Qwen3.8 bf16 rulers cannot judge a fine-tune's quant, so this
# unit generates the Qwopus bf16 references with the R156/r173c method (v0280 launcher, bf16 KV, spec OFF, no tier, no prefix cache):
# dense dump (MAXLEN 3072 UTIL 0.88 SEQS 4), agentic gen on the R156 prompt set (max-tokens 2048, conc 4), greedy decode at ctx0 + 30K.
# Order (advisor 2026-09-05): smoke-boot the quants FIRST (a ModelOpt loader miss makes the bf16 hours worthless) and read the acceptance
# off decode_ss code-c1 (the syvai DFlash2 drafter was trained on the base target; the headline number of this audition), then the bf16
# references, then one full battery boot per quant on the daily route (EXP=1 :8029, SEQS 16, PCIE_IPC=1, BSS=1, pin 13.98 -> 13.5, nvfp4 KV,
# DFlash2 ns7 syvai): dense + agentic (Qwopus ids primary, Qwen ids as context) + decode ctx0/30K rulers, decode_ss rows, tool-eval, needle
# gate (X only). Last, X with its own MTP head (SPEC_METHOD=mtp ns3 on the mtppcie image) for c1/c8 speed only: R198 disqualified MTP from the
# daily (tiers never serve under it), so that arm is a speed reading, not a candidate. Two questions kept apart in the write-up: "is the
# quant faithful to Qwopus bf16" (rulers) and "is Qwopus worth serving" (tool-eval, agentic completion lengths, later SWE-bench paired vs R175).
# Context rows: Qwopus bf16 vs Qwen3.8 bf16 (dense/decode/agentic) = the size of the fine-tune's own shift, the yardstick the quant deltas
# are read against. Numerics caveat as R196 (each arm = its own compile artifact, lottery noise ±0.035 % dense PPL). Restores the daily at the end.
#   unit: sudo systemd-run --unit=r199-qwopus --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r199-qwopus bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r199-qwopus-audition.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r199-qwopus; mkdir -p "$R/report"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
MTP_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie
M=/srv/qwen5090/models; QB=$M/qwopus-27b-flash-bf16; SX=$M/qwopus-27b-flash-shiftedx-nvfp4; SJ=$M/qwopus-27b-flash-sojufx-nvfp4
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; LAUNCH_BF16=/srv/qwen5090/launch-daily-v0280.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
FD=/srv/qwen5090/results/2026-08-23-fidelity; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl; P=/srv/qwen5090/r156-agentic-prompts.jsonl
PYT=/srv/qwen5090/venv-lmeval/bin/python
PINS="13980000000 13500000000"
MTP_PIN_ENV="EXTRA_ENV_APPEND=-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1"
for i in "$IMG" "$MTP_IMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
for d in "$QB" "$SX" "$SJ"; do [ -f "$d/config.json" ] && [ -f "$d/model.safetensors.index.json" ] || { log "ABORT: checkpoint incomplete: $d"; exit 3; }; done
for f in "$BF16_DIR/dump-a1-dense.jsonl" "$BF16_DIR/dump-a1-agentic.jsonl" "$BF16_DIR/agentic-ids.jsonl" "$LADDER_CORPUS" "$P" "$FD/corpus.jsonl" "$DREF/dec-bf16-ctx0.jsonl" "$DREF/dec-bf16-ctx30000.jsonl" "$LAUNCH_BF16" "$PYT"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
for t in decode_ss.py decode_fidelity.py fidelity_compare.py fidelity_ladder.py agentic_ref.py tooleval_summary.py needle_gate.sh; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
grep -q 'CAND_MODEL' "$CAND" || { log "ABORT: launch-daily.sh lacks the R196 CAND_MODEL passthrough"; exit 3; }
grep -qF 'SPEC_METHOD_=${SPEC_METHOD:-dflash}' "$CAND" || { log "ABORT: launch-daily.sh SPEC_METHOD wiring missing"; exit 3; }
cmp -s "$QB/chat_template.jinja" "$SX/chat_template.jinja" && cmp -s "$QB/chat_template.jinja" "$SJ/chat_template.jinja" || { log "ABORT: Qwopus chat templates differ across bf16/X/J"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  grep -aE "BOOT OK|BOOT FAILED|boot-err|engine-quant|layout|RESULT|PROBE FAILED|decode ctx|vs |tool-eval|needle|completion|error-lines|SKIP|ABORT" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"; log "=== R199 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R199 start (lock held): Qwopus3.8-27B-Flash audition — bf16 072b8c27e0d555741c3075079af10f0b521da46f, X=Shiftedx e46dcfbe1aef581743509edb3a3c2c8934c3942d, J=sojufx bdd5036f276555a0385e08907e8e3829e51c6f5d; $IMG (PCIE_IPC=1 BSS=1), model's own chat template on every Qwopus arm ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }   # tier content is token-hash keyed, NOT model keyed: always wipe between checkpoints
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG MODEL_DIR [ENV=VAL ...]  (daily route; the extra assignments reach the launcher: CAND_IMG, SPEC_METHOD, SPEC_NS, EXTRA_ENV_APPEND)
boot_arm(){ local tag=$1 mdir=$2 kv rc; shift 2
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG CAND_MODEL=$mdir "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      log "[$tag] BOOT OK model=$(basename $mdir) pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB weights=$(grep -aoE 'Loading weights took [0-9.]+ seconds' "$R/engine-boot-$tag.log" | tail -1) bss=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log")"
      grep -aiE "quantization|modelopt|marlin|nvfp4|kv[_ ]?scale|k_scale|ForConditional|ForCausal|mtp|W4A16|fp8" "$R/engine-boot-$tag.log" | grep -avE "^\s*$|WARNING.*deprecated" | sed -E 's/^[^ ]* //' | sort -u | head -14 | cut -c1-220 | sed "s/^/[$tag engine-quant] /" | tee -a "$R/audit.log"
      log "[$tag layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|quantiz|kv_cache_scheme|k_scale|not supported|unsupported" | head -10 | cut -c1-240 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    ELOG > "$R/engine-bootfail-$tag-$kv.log" 2>&1; teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-300 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
# rulers. cmp_dense/cmp_agentic are offline (dump vs dump); dense_dump/agentic_score/dfid_run need the arm up.
dense_dump(){ local T=$1; timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T dense dump] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"; }
cmp_dense(){ local label=$1 ref=$2 arm=$3; [ -s "$ref" ] && [ -s "$arm" ] || { log "[$label dense] SKIP (dump missing)"; return; }
  python3 $PR/fidelity_compare.py --ref "$ref" --arm "$arm" --label "$label" --json "$R/report/$label-dense.json" 2>&1 | tee "$R/report/$label-dense.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$label dense] /" | tee -a "$R/audit.log"; }
agentic_score(){ local T=$1 ids=$2 suffix=$3; timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$ids" --out "$R/dump-$T-agentic$suffix.jsonl" > "$R/score-$T-agentic$suffix.out" 2>&1
  grep -aE "^DONE|FAIL" "$R/score-$T-agentic$suffix.out" | tail -1 | sed "s/^/[$T agentic$suffix score] /" | tee -a "$R/audit.log"; }
cmp_agentic(){ local label=$1 ref=$2 arm=$3; [ -s "$ref" ] && [ -s "$arm" ] || { log "[$label agentic] SKIP (dump missing)"; return; }
  python3 $PR/fidelity_compare.py --ref "$ref" --arm "$arm" --label "$label" --json "$R/report/$label-agentic.json" 2>&1 | tee "$R/report/$label-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$label agentic] /" | tee -a "$R/audit.log"; }
dfid_run(){ local T=$1 ctx=$2; python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  [ -s "$R/dec-$T-ctx$ctx.jsonl" ] || { log "[$T decode ctx$ctx] EMPTY dump: $(tail -1 "$R/dec-$T-ctx$ctx.out" | cut -c1-160)"; rm -f "$R/dec-$T-ctx$ctx.jsonl"; }; }
cmp_dec(){ local label=$1 ref=$2 arm=$3; [ -s "$ref" ] && [ -s "$arm" ] || { log "[$label] SKIP (dump missing)"; return; }
  log "[$label] $(python3 $PR/decode_fidelity.py compare "$ref" "$arm" 2>&1 | tail -1 | cut -c1-300)"; }
tooleval(){ local T=$1; ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$T.json" > "$R/tooleval-$T.log" 2>&1 )
  python3 $PR/tooleval_summary.py "$R/tooleval-$T.json" "$T" 2>&1 | tee -a "$R/audit.log"; }
completion_stats(){ local label=$1 f=$2; [ -s "$f" ] || return; python3 - "$f" "$label" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, sys, statistics
f, label = sys.argv[1], sys.argv[2]; acc = {}
for line in open(f):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    for k, v in d.items():
        if isinstance(v, list): acc.setdefault(k, []).append(len(v))
print(f"[{label} completion lengths] n={max((len(v) for v in acc.values()), default=0)} " + " ".join(f"{k}: mean={statistics.mean(v):.0f} median={statistics.median(v):.0f} max={max(v)}" for k, v in acc.items()))
PY
}

# ---------- phase A: smoke boots (loader + drafter fit), cheapest kill first ----------
smoke(){ local tag=$1 mdir=$2; teardown; wipe_l2
  if boot_arm "$tag" "$mdir"; then sleep 15; p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code; log "[$tag smoke] engine error-lines=$(errs)"; teardown; return 0; fi
  log "[$tag] BOOT FAILED on every pin (smoke)"; return 1; }
OK_X=0; OK_J=0
smoke X "$SX" && OK_X=1
smoke J "$SJ" && OK_J=1
if [ $OK_X = 0 ] && [ $OK_J = 0 ]; then log "neither quant boots on the daily route: loader problem, engine logs in $R/engine-bootfail-*.log — bf16 references NOT generated"; finish "ABORTED (no quant boots)"; exit 1; fi

# ---------- phase B: Qwopus bf16 references (R156 arm a / r156-agentic a / r173c method, v0280 launcher) ----------
boot_bf16(){ local maxlen=$1 util=$2 seqs=$3 mnbt=$4 att
  for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" MODEL_DIR="$QB" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 UTIL=$util KVD_OVERRIDE=bfloat16 NOSPEC=1 NO_TIER=1 PREFIX_CACHE=0 \
      SEQS=$seqs MNBT=$mnbt MAXLEN=$maxlen FIWS=268435456 POOL_MIN=1 POOL_MAX=99999999 EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" bash $LAUNCH_BF16 > "$R/boot-bf16-$maxlen.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null && { log "[bf16] BOOT OK maxlen=$maxlen util=$util seqs=$seqs $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-bf16-$maxlen.log" | tail -1)"; return 0; }
    log "[bf16] boot maxlen=$maxlen attempt $att FAILED: $(grep -aE 'FAILED|Error' "$R/boot-bf16-$maxlen.log" | tail -1 | cut -c1-200)"; teardown; done; return 1; }
teardown
if boot_bf16 3072 0.88 4 2048; then
  dense_dump QB
  cmp_dense "QB-vs-QwenBF16" "$BF16_DIR/dump-a1-dense.jsonl" "$R/dump-QB-dense.jsonl"
  dfid_run QB 0; cmp_dec "QB-vs-QwenBF16 decode ctx0" "$DREF/dec-bf16-ctx0.jsonl" "$R/dec-QB-ctx0.jsonl"
  log "[bf16 dense boot] engine error-lines=$(errs)"; teardown
else log "[bf16] dense reference SKIPPED (boot failed twice)"; fi
if boot_bf16 40960 0.92 4 2048 || boot_bf16 6144 0.92 4 2048; then
  timeout 7200 $PYT $PR/agentic_ref.py gen --url $U --model qwen3.8-27b --prompts $P --tokenizer $QB --out-ref "$R/dump-QB-agentic.jsonl" --out-ids "$R/agentic-ids-QB.jsonl" --max-tokens 2048 --conc 4 > "$R/score-QB-agentic.out" 2>&1
  grep -aE "^\[smoke\]|^DONE|FAIL" "$R/score-QB-agentic.out" | tail -2 | sed "s/^/[bf16 agentic gen] /" | tee -a "$R/audit.log"
  completion_stats "QB agentic (Qwopus bf16 gen)" "$R/agentic-ids-QB.jsonl"; completion_stats "Qwen bf16 agentic (R156 gen)" "$BF16_DIR/agentic-ids.jsonl"
  # Qwopus bf16 scoring the Qwen bf16 generations = the fine-tune's agentic distance from the base, same regime as the quant rulers
  agentic_score QB "$BF16_DIR/agentic-ids.jsonl" "-qwenids"; cmp_agentic "QB-vs-QwenBF16" "$BF16_DIR/dump-a1-agentic.jsonl" "$R/dump-QB-agentic-qwenids.jsonl"
  if grep -aq "maxlen=40960" "$R/audit.log"; then dfid_run QB 30000; cmp_dec "QB-vs-QwenBF16 decode ctx30000" "$DREF/dec-bf16-ctx30000.jsonl" "$R/dec-QB-ctx30000.jsonl"; else log "[bf16 decode ctx30000] SKIPPED (maxlen 6144 fallback)"; fi
  log "[bf16 agentic boot] engine error-lines=$(errs)"; teardown
else log "[bf16] agentic + decode-30K references SKIPPED (boot failed)"; fi

# ---------- phase C: full battery per quant, one boot each, DFlash2 ns7 syvai drafter ----------
battery(){ local tag=$1 mdir=$2 full=$3; teardown; wipe_l2
  boot_arm "$tag" "$mdir" || { log "[$tag] BOOT FAILED on every pin (battery)"; return 1; }
  sleep 20
  dense_dump $tag
  cmp_dense "$tag-vs-QB" "$R/dump-QB-dense.jsonl" "$R/dump-$tag-dense.jsonl"
  cmp_dense "$tag-vs-QwenBF16" "$BF16_DIR/dump-a1-dense.jsonl" "$R/dump-$tag-dense.jsonl"
  if [ -s "$R/agentic-ids-QB.jsonl" ]; then agentic_score $tag "$R/agentic-ids-QB.jsonl" ""; cmp_agentic "$tag-vs-QB" "$R/dump-QB-agentic.jsonl" "$R/dump-$tag-agentic.jsonl"; fi
  agentic_score $tag "$BF16_DIR/agentic-ids.jsonl" "-qwenids"; cmp_agentic "$tag-vs-QwenBF16" "$BF16_DIR/dump-a1-agentic.jsonl" "$R/dump-$tag-agentic-qwenids.jsonl"
  for ctx in 0 30000; do dfid_run $tag $ctx; cmp_dec "$tag-vs-QB decode ctx$ctx" "$R/dec-QB-ctx$ctx.jsonl" "$R/dec-$tag-ctx$ctx.jsonl"; cmp_dec "$tag-vs-QwenBF16 decode ctx$ctx" "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$tag-ctx$ctx.jsonl"; done
  p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $tag prose-c8 --conc 8 --tokens 1024 --runs 2 --kind prose
  p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
  tooleval "$tag"
  if [ "$full" = 1 ]; then tooleval "$tag-2"
    log "[$tag needle] gate start (131K + 220K cold, evicted re-asks through the eval-l2 tier)"
    U=$U bash $PR/needle_gate.sh "$tag" "$R" > "$R/needle-$tag.log" 2>&1; rc=$?
    log "[$tag needle] rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-$tag.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"; fi
  log "[$tag battery] engine error-lines=$(errs) preemptions=$(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"; }
[ $OK_X = 1 ] && battery X "$SX" 1
[ $OK_J = 1 ] && battery J "$SJ" 0
# ---------- phase D: X with its own MTP head, speed only (R198: MTP is not a daily candidate) ----------
if [ $OK_X = 1 ]; then teardown; wipe_l2
  if boot_arm XM "$SX" CAND_IMG=$MTP_IMG SPEC_METHOD=mtp SPEC_NS=3 "$MTP_PIN_ENV"; then sleep 15
    p1 XM code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 XM code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
    log "[XM] engine error-lines=$(errs)"
  else log "[XM] MTP arm BOOT FAILED on every pin"; fi
fi
[ -s "$R/dec-X-ctx0.jsonl" ] && [ -s "$R/dec-J-ctx0.jsonl" ] && for ctx in 0 30000; do cmp_dec "J-vs-X decode ctx$ctx" "$R/dec-X-ctx$ctx.jsonl" "$R/dec-J-ctx$ctx.jsonl"; done
finish DONE
