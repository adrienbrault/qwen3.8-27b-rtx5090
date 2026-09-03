#!/usr/bin/env bash
# R166 (2026-09-03, user "make nvfp4 work"): the nvfp4-KV candidate with the PINNED KV budget (Bug C fix at the operator
# level, launch-daily-nvfp4-candidate.sh EXP=1) against the fp8 daily shape, PAIRED same hour on :8029, every gate of
# dflash-nvfp4-revival/PLAN-PROMOTION.md except G2/G3 (SWE-bench: only once everything here is green).
#   arm F  fp8 daily shape (RedHat, fp8 KV, DFlash2 ns9, TP2, util 0.92, SEQS 8, tier eval-l2 wiped): needles
#          9K/20K/131K/220K/258K x2 (G1), warm-revisit 32K (G12), benchy c1 x5 (per-run values: the box is bimodal,
#          R163c) + c8 x3 (G6/G7), decode_ss code/prose c8 + deep30k (G7), tool-eval 69x4 (G4), error lines (G10).
#          Fidelity for this arm = today's cell R (results/2026-09-03-r165c-audition/run-v0280-redhat-fp8kv.jsonl).
#   arm N8 candidate launcher, SEQS 8 (the daily contract): boot facts (pinned line, pool, free after pre-warm), the same
#          probes as F, plus the fidelity ruler --conc 1 (G5; the pin guarantees the >=0.5 GiB it needs) compared to the
#          FP8 reference AND directly to cell R (same checkpoint, fp8 KV -> nvfp4 KV isolated).
#   arm N16 candidate, SEQS 16, tier NOT wiped: restart-revisit (G12: nvfp4 blocks survive a restart), needles 131K/220K
#          x2 and 131K under 8 concurrent 20K loaders (G1 at the wider layout), benchy c16 x2, decode_ss code c16,
#          preemptions.
#   arm N32 candidate, SEQS 32 (the Bug C shape): needles 258K x2, benchy c8/c16/c32 x2 with a 1 s /metrics sampler
#          (max num_requests_running = admission, G8; fp8 reference = R159's 17 on 624K), preemptions.
# Boot order F -> N (plan rule). Every boot's pool/free is a gate line. Takes the GPU-exclusive flock, registers in the
# gpu-queue, takes the daily down, restores it at the end unless another unit is queued (OPERATIONS §12).
# Unit: sudo systemd-run --unit=r166-gates --collect -p User=adrienbrault -p RuntimeMaxSec=18000 -p TimeoutStopSec=900 bash /srv/qwen5090/r166-candidate-gates.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r166-gates; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
source /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R166 candidate gates start (lock held) ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-nvfp4-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
FP8_RUN=/srv/qwen5090/results/2026-09-03-r165c-audition/run-v0280-redhat-fp8kv.jsonl
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R166 candidate gates $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 namespace sets wiped (engine down; cold tier)"; }

FP8_COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8 IMAGE=vllm-qwen38:v0280-nvfp4kv KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'

facts(){ # $1 tag: boot facts from the engine log
  sudo docker logs vllm-exp 2>&1 | grep -aE "Available KV cache memory|GPU KV cache size|as specified by kv_cache_memory_bytes|Actual usage|Graph capturing finished|int workspace shrunk|Capturing dflash2|decode_backend" | cut -c1-240 > "$R/bootfacts-$1.txt"
  grep -aE "GPU KV cache size|kv_cache_memory_bytes|Graph capturing finished" "$R/bootfacts-$1.txt" | grep -a "TP0\|EngineCore" | sed -E 's/.*(INFO|WARNING) [0-9: -]+\[[a-z_.:0-9]+\] //' | cut -c1-170 | sed "s/^/[$1 facts] /" | tee -a "$R/audit.log"
  log "[$1 facts] free VRAM after pre-warm: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"; }
boot_fp8(){ local att rc; for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" $FP8_COMMON EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" bash $LAUNCH > "$R/boot-F.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[F] BOOT OK att=$att $(grep -aoE 'KV pool: [0-9]+ tokens' "$R/boot-F.log" | tail -1)"; facts F; return 0; fi
    log "[F] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-F.log" | tail -1 | cut -c1-160)"; teardown; [ $att = 1 ] && sleep 180
  done; return 1; }
boot_cand(){ # $1 tag, $2 SEQS
  local att rc; for att in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=$2 bash $CAND > "$R/boot-$1.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$1] BOOT OK att=$att $(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$1.log" | tail -1)"; facts "$1"; return 0; fi
    log "[$1] BOOT FAILED att=$att rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-$1.log" | tail -1 | cut -c1-160)"
    log "[$1] last exception: $(sudo docker logs vllm-exp 2>&1 | grep -aE '(ValueError|AttributeError|RuntimeError|AssertionError|OutOfMemoryError): ' | tail -1 | cut -c1-200)"
    teardown; [ $att = 1 ] && sleep 180
  done; return 1; }
needles(){ # $1 tag, $2 depths, rest = extra args
  local tag=$1 depths=$2; shift 2
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $depths --samples 2 "$@" --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1
  log "[$tag needles] hits=$(grep -ac 'HIT ' "$R/needles-$tag.out") miss=$(grep -ac MISS "$R/needles-$tag.out") $(grep -aE 'MISS|HTTP|Error' "$R/needles-$tag.out" | head -2 | cut -c1-120 | tr '\n' ' ')"; }
revisit(){ python3 /srv/qwen5090/probes/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/warm-revisit-$1.log" 2>&1; grep -a RESULT "$R/warm-revisit-$1.log" | cut -c1-260 | sed "s/^/[$1 revisit] /" | tee -a "$R/audit.log"; }
benchy(){ # $1 tag, $2 conc list, $3 runs
  timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1.json" > "$R/benchy-$1.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_summary.py "$R/benchy-$1.json" "benchy.$1" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/benchy_runs.py "$R/benchy-$1.json" "benchy.$1" | tee -a "$R/audit.log"; }
dss(){ local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind.json" > "$R/decode-$tag-$kind.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind.out" | sed "s/^/[$tag decode_ss $kind] /" | cut -c1-230 | tee -a "$R/audit.log"; }
deep30k(){ python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 --out "$R/decode-$1-deep30k.json" > "$R/decode-$1-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$1-deep30k.out" | sed "s/^/[$1 decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"; }
tooleval(){ log "[$1] tool-eval x4 start"
  ( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$1.json" > "$R/tooleval-$1.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$1.json" "$1" 2>&1 | tee -a "$R/audit.log"; }
metrics_line(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:(num_preemptions_total|spec_decode_num_(accepted|draft)_tokens_total)" | sed "s/^/[$1] /" | cut -c1-160 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
core(){ # $1 tag: the shared SEQS-8 probe set (G1 G12 G6 G7 G4 G10)
  needles "$1" "9000 20000 131000 220000 258000"
  revisit "$1"
  benchy "$1-c1" 1 5; benchy "$1-c8" 8 3
  dss "$1" code 8; dss "$1" prose 8; deep30k "$1"
  tooleval "$1"
  metrics_line "$1"; errlines "$1"; }

log "taking the daily down"; teardown

# ---- arm F: fp8 daily shape ----
wipe_l2
if boot_fp8; then core F; else log "[F] arm skipped (boot failed twice)"; fi
teardown

# ---- arm N8: candidate, SEQS 8 ----
wipe_l2
if boot_cand N8 8; then
  core N8
  log "[N8] fidelity ruler start (--conc 1)"
  python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-nvfp4-candidate.jsonl" --conc 1 > "$R/fidelity-run.log" 2>&1 || log "[N8] ruler FAILED rc=$? ($R/fidelity-run.log)"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$FP8_RUN" "$R/run-nvfp4-candidate.jsonl" 2>&1 | tee "$R/fidelity-compare-vs-fp8ref.txt" | cut -c1-200 | sed "s/^/[N8 fidelity vs fp8ref] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FP8_RUN" "$R/run-nvfp4-candidate.jsonl" 2>&1 | tee "$R/fidelity-compare-nvfp4-vs-fp8kv.txt" | cut -c1-200 | sed "s/^/[N8 fidelity nvfp4 vs fp8 KV] /" | tee -a "$R/audit.log"
  log "[N8 facts] free VRAM after battery: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"; errlines N8-end
else log "[N8] arm skipped (boot failed twice)"; fi
teardown   # tier kept: N16 proves the restart revisit

# ---- arm N16: candidate, SEQS 16 ----
if boot_cand N16 16; then
  revisit N16-restart
  needles N16 "131000 220000"
  needles N16-flood8 "131000" --loaders 8 --parallel 2
  benchy N16-c16 16 2; dss N16 code 16
  metrics_line N16; errlines N16
else log "[N16] arm skipped (boot failed twice)"; fi
teardown

# ---- arm N32: candidate, SEQS 32 ----
if boot_cand N32 32; then
  needles N32 "258000"
  ( while :; do curl -s -m 2 $U/metrics | grep -aE "^vllm:num_requests_(running|waiting)" | tr '\n' ' '; echo; sleep 1; done > "$R/admission-N32.txt" ) & SAMP=$!
  benchy N32-c8c16c32 "8 16 32" 2
  kill $SAMP 2>/dev/null
  log "[N32 admission] max num_requests_running=$(grep -aoE 'num_requests_running[^ ]* [0-9.]+' "$R/admission-N32.txt" | awk '{if($2+0>m)m=$2+0} END{print m+0}') max waiting=$(grep -aoE 'num_requests_waiting[^ ]* [0-9.]+' "$R/admission-N32.txt" | awk '{if($2+0>m)m=$2+0} END{print m+0}') (fp8 R159: 17 running on 624K)"
  dss N32 code 32
  metrics_line N32; errlines N32
else log "[N32] arm skipped (boot failed twice)"; fi
finish DONE
