#!/usr/bin/env bash
# R182 (2026-09-04, user "Ok promote"): promote the bf16 SSM state to the daily. launch-daily.sh now passes
# --mamba-ssm-cache-dtype bfloat16 (block 2,944 → 1,584, pool 1,020,596 at the unchanged 13.98 GB pin) and asserts it.
# Chain: daily down → native-l2 wiped (every tier hash changes with the block size) → boot the daily from launch-daily.sh
# (KV_BYTES=13.5 GB fallback if Bug C headroom is missing; then the launcher table must follow) → reads on :8020: cache layout,
# kv_capacity short / 100K / five 100K, needle gate (131K + 220K cold and evicted re-asks through the tiers), decode_ss code c8 +
# prose c1, tool-eval 69x4. If the bf16 daily cannot boot, the frozen rollback launcher brings the fp32 daily back.
# Unit: sudo systemd-run --unit=r182-promote-ssm-bf16 --collect -p User=adrienbrault -p RuntimeMaxSec=5400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r182-promote-ssm-bf16 bash /srv/qwen5090/r182-promote-ssm-bf16.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r182-promote-ssm-bf16; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/native-l2; ROLLBACK=/srv/qwen5090/launch-daily-r174-ssm-fp32-0904.sh
[ -f "$ROLLBACK" ] || { log "ABORT: rollback launcher missing"; exit 3; }
grep -q "mamba-ssm-cache-dtype" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh has no SSM dtype flag (ship the R182 launcher first)"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ teardown; log "ROLLBACK: booting the fp32-SSM daily from $ROLLBACK"; env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash $ROLLBACK > "$R/boot-rollback.log" 2>&1 && log "ROLLBACK daily up: $(grep -aoE 'Pool [0-9]+' "$R/boot-rollback.log" | tail -1)" || log "ROLLBACK FAILED: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; log "=== R182 ABORTED (rolled back) ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then curl -sf -m 5 $U/health >/dev/null || rollback; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R182 promote SSM-bf16 start (lock held) ==="
log "taking the fp32-SSM daily down (pool $(curl -s -m 5 $U/metrics | grep -aoE 'kv_cache_size_tokens="[0-9]+"' | head -1))"; teardown
log "native-l2 before wipe: $(du -sh $L2 2>/dev/null | cut -f1)"; sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
log "native-l2 after wipe: $(du -sh $L2 2>/dev/null | cut -f1)"
PIN=13980000000
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then :
else
  log "boot at the table pin FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"
  if grep -aq "Bug C headroom missing" "$R/boot-daily-$PIN.log"; then
    teardown; PIN=13500000000; log "retrying with KV_BYTES=$PIN (launcher table must be updated to this value if it boots)"
    env -i HOME="$HOME" USER="$USER" PATH="$PATH" KV_BYTES=$PIN bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "boot at $PIN FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"; rollback; exit 1; }
  else rollback; exit 1; fi
fi
log "DAILY UP pin=$PIN $(tail -1 "$R/boot-daily-$PIN.log" | cut -c1-200)"
log "[layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|mamba_ssm_cache_dtype="[a-z0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB cpu_tier=$(sudo docker logs vllm-27b 2>&1 | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1)"
sleep 20
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
cap ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
cap five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
log "needle gate start (131K + 220K cold, then evicted re-asks through the tiers)"
U=$U bash $PR/needle_gate.sh post "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-gate.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-260 | tee -a "$R/audit.log" || log "[daily $name] PROBE FAILED"; }
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-daily.json" > "$R/tooleval-daily.log" 2>&1 )
python3 $PR/tooleval_summary.py "$R/tooleval-daily.json" daily-ssm-bf16 2>&1 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "DAILY UP|layout|cap |needle|RESULT|tool-eval|error lines|FAILED|ROLLBACK" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R182 DONE — daily = SSM bf16 (pin $PIN); rollback $ROLLBACK ==="
