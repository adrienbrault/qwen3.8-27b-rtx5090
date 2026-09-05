#!/usr/bin/env bash
# R200 (2026-09-05, user: "Try c32/c64 decode/prefill code/prose"): the daily serves SEQS 16 and every published concurrency row stops at c16.
# This unit boots the daily configuration on :8029 at SEQS 32 and SEQS 64 (EXP=1, same image/flags/pin as launch-daily.sh: nvfp4 KV,
# DFlash2 ns7 syvai, PCIE_IPC=1, BSS=1, pin 13.98 GB -> 13.5 GB fallback) and measures, per boot:
#   decode  probes/decode_ss.py (steady-state window = every stream running, engine counters): code and prose at c=SEQS, tokens 1024, runs 2;
#           the SEQS 64 boot also runs c32 (same-boot read of what the larger SEQS costs c32) and the SEQS 32 boot runs c16 (ties to the daily's c16 rows).
#   prefill llama-benchy (repo-standard, tokenizer-counted): --pp 2048 8192 --tg 32 --concurrency SEQS [32 on the SEQS 64 boot], runs 2, --no-cache,
#           temperature 0.6 like the README rows. c64 x pp 8192 = 524K prompt tokens per run, inside the pool.
# Read-outs to keep: pool at each SEQS (the GDN state lives in the pool: SEQS 8 -> 16 cost 34K tokens, R176/R178), min free VRAM after
# capture (graph sizes grow with SEQS), accept_per_draft at c32/c64, and steps/s = tps / (1 + 7 x accept). Restores the daily at the end.
#   unit: sudo systemd-run --unit=r200-c32c64 --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r200-c32c64 bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r200-c32c64.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r200-c32c64; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=$(grep -oE '^DAILY_IMG=[^ ]+' /srv/qwen5090/launch-daily.sh | cut -d= -f2)
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
TOK=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4; PINS="13980000000 13500000000"; TEMP=0.6
[ -n "$IMG" ] && sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: daily image '$IMG' missing"; exit 3; }
for t in decode_ss.py benchy_summary.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
[ -x "$HOME/.local/bin/llama-benchy" ] || { log "ABORT: llama-benchy missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  grep -aE "BOOT OK|BOOT FAILED|layout|RESULT|PROBE FAILED|error-lines|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"; log "=== R200 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R200 start (lock held): daily config on :8029 at SEQS 32 and 64, $IMG, PCIE_IPC=1 BSS=1 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
boot(){ local s=$1 kv rc
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=$s KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG bash $CAND > "$R/boot-s$s-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-s$s.log"
      log "[s$s] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-s$s-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-s$s-$kv.log" | tail -1 | tr -dc 0-9)MiB free_now=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB graphs=$(grep -aoE 'Capturing CUDA graphs[^\n]{0,80}' "$R/engine-boot-s$s.log" | tail -1 | cut -c1-90)"
      log "[s$s layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') max_num_seqs=$(grep -aoE 'max_num_seqs=[0-9]+' "$R/engine-boot-s$s.log" | head -1)"
      return 0; fi
    log "[s$s] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-s$s-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|OutOfMemory" | head -6 | cut -c1-220 | sed "s/^/[s$s boot-err] /" | tee -a "$R/audit.log"; teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-330 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-160)"; fi; }
benchy(){ local tag=$1; shift
  timeout 5400 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$TOK" --pp 2048 8192 --tg 32 --concurrency "$@" --runs 2 --exact-tg --no-cache \
    --latency-mode api --format json --extra-body "{\"temperature\":$TEMP}" --save-result "$R/benchy-$tag.json" > "$R/benchy-$tag.out" 2>&1
  [ -s "$R/benchy-$tag.json" ] && python3 $PR/benchy_summary.py "$R/benchy-$tag.json" "$tag prefill" 2>&1 | cut -c1-330 | tee -a "$R/audit.log" || log "[$tag prefill] BENCHY FAILED: $(tail -2 "$R/benchy-$tag.out" | tr '\n' ' ' | cut -c1-200)"; }
arm(){ local s=$1; shift; teardown
  boot $s || { log "[s$s] BOOT FAILED on every pin"; return 1; }
  sleep 20
  local c; for c in "$@"; do
    p1 s$s code-c$c  --conc $c --tokens 1024 --runs 2 --kind code
    p1 s$s prose-c$c --conc $c --tokens 1024 --runs 2 --kind prose
  done
  benchy s$s "$@"
  log "[s$s] engine error-lines=$(errs) preemptions=$(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}') free_after=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; }
arm 32 32 16
arm 64 64 32
finish DONE
