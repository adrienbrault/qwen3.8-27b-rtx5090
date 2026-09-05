#!/usr/bin/env bash
# R197 promotion gate (2026-09-05, user "I validate switching from d9 to d7"): the first boot of the ns7 daily ON THE SERVING PORT, gated the
# way R182/R189 were. Every ns7 number so far is from :8029 (EXP=1, MIN_FREE floor 384 MiB); launch-daily.sh now serves DFlash ns7 (block
# 1,552, band 1.00–1.085M, native-l2 `.block` stamp + `_model_*` wipe on mismatch) and this unit exercises those edits with a rollback:
#   teardown → boot from launch-daily.sh at 13.98 GB (13.5 GB retry on ANY failure, R191) → assert the tier wipe + stamp 1552 → layout/pool
#   → kv_capacity short / 100K / five 100K → needle gate (fully cold: the tier was just wiped, as at R182) → decode_ss README rows
#   (code c1 ×3, prose c1 ×2, prose 30K ×2, code c8, prose c8, code c16, prose c16 — the last one was unmeasured at ns7) → tool-eval 69×4
#   → error lines. On a boot failure at both pins the frozen ns9 launcher (launch-daily-r195-ns9-0905.sh) brings the daily back.
# The daily stays UP at the end. Queued behind r196 with GPU_QUEUE_NAME so r196 skips its own restore: this unit IS the restore.
#   unit: sudo systemd-run --unit=r197-promote-ns7 --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r197-promote-ns7 bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r196-minima-audition; do sleep 30; done; exec bash /srv/qwen5090/r197-promote-ns7.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r197-promote-ns7; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/native-l2; ROLLBACK=/srv/qwen5090/launch-daily-r195-ns9-0905.sh
[ -f "$ROLLBACK" ] || { log "ABORT: rollback launcher missing"; exit 3; }
grep -q "^DAILY_NS=7; DAILY_BLOCK=1552" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh is not the ns7 launcher"; exit 3; }
for t in kv_capacity_probe.py needle_gate.sh decode_ss.py tooleval_summary.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ teardown; log "ROLLBACK: booting the ns9 daily from $ROLLBACK"; env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash $ROLLBACK > "$R/boot-rollback.log" 2>&1 && log "ROLLBACK daily up: $(grep -aoE 'Pool [0-9]+' "$R/boot-rollback.log" | tail -1)" || log "ROLLBACK FAILED too: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then curl -sf -m 5 $U/health >/dev/null || rollback; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R197 promote ns7 start (lock held) ==="
teardown
log "native-l2 before: $(du -sh $L2 2>/dev/null | cut -f1), stamp '$(cat $L2/.block 2>/dev/null || echo none)' (a mismatch wipes _model_* inside the launcher)"
PIN=13980000000
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then :
else
  log "boot at the table pin FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"
  teardown; PIN=13500000000; log "retrying with KV_BYTES=$PIN (R191: warmup 'invalid argument' flake at 13.98; launcher table must be updated if this pin sticks)"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" KV_BYTES=$PIN bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "boot at $PIN FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"; rollback; exit 1; }
fi
log "DAILY UP pin=$PIN $(tail -1 "$R/boot-daily-$PIN.log" | cut -c1-260)"
grep -aE "block stamp|wiping" "$R/boot-daily-$PIN.log" | cut -c1-240 | sed 's/^/[wipe] /' | tee -a "$R/audit.log"
log "native-l2 after: $(du -sh $L2 2>/dev/null | cut -f1), stamp '$(cat $L2/.block 2>/dev/null || echo none)' (expected 1552)"
[ "$(cat $L2/.block 2>/dev/null)" = 1552 ] || log "WARN: block stamp is not 1552 after the boot"
BOOT=$(sudo docker logs vllm-27b 2>&1); echo "$BOOT" > "$R/engine-boot-daily.log"
log "[block] $(echo "$BOOT" | grep -aoE 'Setting attention block size to [0-9]+ tokens' | head -1)  [spec] $(echo "$BOOT" | grep -aoE 'num_spec_tokens=[0-9]+' | head -1)"
log "[allreduce] $(echo "$BOOT" | grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" | head -1)"
log "[layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
sleep 20
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
cap ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
cap five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
log "needle gate start (131K + 220K cold on the wiped tier, then the evicted re-asks through the tiers)"
U=$U bash $PR/needle_gate.sh post "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-gate.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED"; fi; }
p1 code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
p1 prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c8 --conc 8 --tokens 1024 --runs 2 --kind prose
p1 code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
p1 prose-c16 --conc 16 --tokens 1024 --runs 2 --kind prose
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-daily.json" > "$R/tooleval-daily.log" 2>&1 )
python3 $PR/tooleval_summary.py "$R/tooleval-daily.json" daily-ns7 2>&1 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "DAILY UP|wipe\]|native-l2|block\]|allreduce|layout|cap |needle|RESULT|PROBE FAILED|tool-eval|error lines|FAILED|ROLLBACK|WARN" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R197 promote ns7 DONE — daily = DFlash ns7 (pin $PIN); rollback $ROLLBACK ==="
