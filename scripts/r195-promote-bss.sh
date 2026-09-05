#!/usr/bin/env bash
# R195 (2026-09-05, user "yes" on the R193e sheet): promote batch-sharded sampling to the daily. launch-daily.sh now boots
# `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash` (pcieipc + patch 0147) with --enable-batch-sharded-sampling on the daily
# port and asserts the "Batch-sharded sampling enabled" line + the 0147 marker. The engine numbers exist (R193b/c/e on :8029, bitwise vs OFF
# under the protocol, +2.6% c8 / +4.5% c16 steps/s); this unit adds the R166-style gate set ON THE LIVE :8020 DAILY, the way R189 did, plus the
# R189b README probe list and the bf16 decode ruler (the daily's compile artifact changes with the image: 0147 changes the hash string, so the
# daily now loads r193b's OFF-h artifact cd95c505/d5e217de/fd65aadf — a different compile-lottery draw (R193) whose bf16 position must be on record):
#   boot from launch-daily.sh (13.5 GB pin fallback) → artifact identity (aot_saved/aot_loaded/hashes) → cache layout + pool → decode ruler ctx0/30K
#   vs the r173c bf16 dumps → kv_capacity short / 100K / five 100K → needle gate (131K + 220K cold + tier re-asks; block unchanged, native-l2 kept)
#   → decode_ss code c1 ×3, prose c1 ×3, prose 30K ×2, code c8 ×2, code c16 ×2 → tool-eval 69×4 → error lines. Daily stays UP at the end.
# If the bsshash daily cannot boot, the frozen R189 launcher brings the pcieipc daily back.
#   unit: sudo systemd-run --unit=r195-promote-bss --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r195-promote-bss bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r195-promote-bss.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r195-promote-bss; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/native-l2; ROLLBACK=/srv/qwen5090/launch-daily-r189-nobss-0905.sh
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
EXPECT_HASHES=cd95c50561a5,d5e217dec357,fd65aadffd91,
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash
[ -f "$ROLLBACK" ] || { log "ABORT: rollback launcher missing"; exit 3; }
grep -q "DAILY_IMG=$IMG" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh does not serve $IMG (ship the R185 launcher first)"; exit 3; }
grep -q "PCIe IPC all-reduce enabled" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh has no pcie_ipc assert"; exit 3; }
grep -q "Batch-sharded sampling enabled" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh has no BSS assert (ship the R195 launcher first)"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: bf16 decode reference missing"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
for t in kv_capacity_probe.py needle_gate.sh decode_ss.py tooleval_summary.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ teardown; log "ROLLBACK: booting the R189 pcieipc (no 0147, BSS off) daily from $ROLLBACK"; env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash $ROLLBACK > "$R/boot-rollback.log" 2>&1 && log "ROLLBACK daily up: $(grep -aoE 'Pool [0-9]+' "$R/boot-rollback.log" | tail -1)" || log "ROLLBACK FAILED too: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then curl -sf -m 5 $U/health >/dev/null || rollback; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R195 promote batch-sharded sampling start (lock held): $IMG, --enable-batch-sharded-sampling ==="
teardown
log "native-l2 kept (block size unchanged; the tier's pre-existing content is the restart-survival test): $(du -sh $L2 2>/dev/null | cut -f1)"
PIN=13980000000
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then :
else
  log "boot at the table pin FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"
  # R191/R190e: a 13.98 GB boot can die in the Triton warmup ("CUDA error: invalid argument", no Bug C line); retry at 13.5 GB on ANY failure.
  teardown; PIN=13500000000; log "retrying with KV_BYTES=$PIN (launcher table must be updated to this value if it boots)"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" KV_BYTES=$PIN bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "boot at $PIN FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"; rollback; exit 1; }
fi
log "DAILY UP pin=$PIN $(tail -1 "$R/boot-daily-$PIN.log" | cut -c1-260)"
BOOT=$(sudo docker logs vllm-27b 2>&1); echo "$BOOT" > "$R/engine-boot-daily.log"
log "[allreduce] $(echo "$BOOT" | grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" | head -1)"
echo "$BOOT" | grep -aE "PCIe IPC" | sed -E 's/^.*(PCIe IPC)/\1/' | sort -u | head -4 | cut -c1-220 | sed "s/^/[pcie] /" | tee -a "$R/audit.log"
log "[bss] lines=$(echo "$BOOT" | grep -ac 'Batch-sharded sampling enabled') marker=$(sudo docker exec vllm-27b cat /opt/prs-markers/0147 2>/dev/null)"
saved=$(echo "$BOOT" | grep -ac "saved AOT compiled function"); loaded=$(echo "$BOOT" | grep -ac "Directly load AOT"); hashes=$(echo "$BOOT" | grep -aoE 'torch_aot_compile/[0-9a-f]{12}' | cut -d/ -f2 | sort -u | tr '\n' ',')
log "[artifact] aot_saved=$saved aot_loaded=$loaded compile_hashes=$hashes $([ "$saved" = 0 ] && [ "$loaded" -ge 1 ] && [ "$hashes" = "$EXPECT_HASHES" ] && echo "= r193b OFF-h artifact (loaded, nothing saved)" || echo "DIFFERENT from r193b OFF-h ($EXPECT_HASHES) — fresh compile-lottery draw; the ruler below is its position")"
log "[layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
sleep 20
dfid(){ local ctx=$1
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-daily-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-daily-ctx$ctx.out" 2>&1
  log "[daily decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-daily-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
dfid 0
dfid 30000
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
cap ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
cap five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
log "needle gate start (131K + 220K cold, then evicted re-asks through the tiers)"
U=$U bash $PR/needle_gate.sh post "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-gate.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED"; fi; }
p1 code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 3 --kind prose
p1 prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-daily.json" > "$R/tooleval-daily.log" 2>&1 )
python3 $PR/tooleval_summary.py "$R/tooleval-daily.json" daily-bss 2>&1 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "DAILY UP|allreduce|pcie\]|bss\]|artifact\]|layout|decode ctx|cap |needle|RESULT|PROBE FAILED|tool-eval|error lines|FAILED|ROLLBACK" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R195 DONE — daily = batch-sharded sampling on $IMG (pin $PIN); rollback $ROLLBACK ==="
