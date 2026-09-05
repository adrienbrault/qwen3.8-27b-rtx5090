#!/usr/bin/env bash
# R189 (2026-09-05, user on the R185 sheet: "seems like no brainer? Lets use?!"): promote FlashInfer's pcie_ipc all-reduce (patch 0138) to the
# daily. launch-daily.sh now boots `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc` with VLLM_SM12X_PCIE_IPC_AR=1 on the daily
# port and asserts the "PCIe IPC all-reduce enabled" line and the PCIE_IPC, CUSTOM, PYNCCL backend order. The engine numbers exist
# (R185/R185b on :8029, SEQS 16, same flags); what this unit adds is the R166-style gate set ON THE LIVE :8020 DAILY, the way R182 did:
#   boot from launch-daily.sh (13.5 GB pin fallback on a Bug C headroom miss) → cache layout + pool → kv_capacity short / 100K / five 100K
#   → needle gate (131K + 220K cold, then the evicted re-asks through the CPU/disk tiers; block size unchanged so native-l2 is NOT wiped and
#   the pre-existing tier content is the restart-survival test) → decode_ss code c8, prose c1, code c16 → tool-eval 69×4 → error lines.
# If the pcieipc daily cannot boot, the frozen pre-R185 launcher brings the fi0616 daily back. The daily stays UP at the end (no teardown).
# Queued behind r188-marlin (GPU_QUEUE_NAME registers it so R188's finish() skips its own restore; this unit IS the restore).
#   unit (re-issued 2026-09-05 06:50 UTC; chain r192 → r190e → r190d → r193 → r189): sudo systemd-run --unit=r189-promote-pcieipc --collect -p User=adrienbrault -p RuntimeMaxSec=86400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r189-promote-pcieipc bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r193-determinism; do sleep 30; done; exec bash /srv/qwen5090/r189-promote-pcieipc.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r189-promote-pcieipc; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/native-l2; ROLLBACK=/srv/qwen5090/launch-daily-r182-nopcie-0905.sh
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
[ -f "$ROLLBACK" ] || { log "ABORT: rollback launcher missing"; exit 3; }
grep -q "DAILY_IMG=$IMG" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh does not serve $IMG (ship the R185 launcher first)"; exit 3; }
grep -q "PCIe IPC all-reduce enabled" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh has no pcie_ipc assert"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
for t in kv_capacity_probe.py needle_gate.sh decode_ss.py tooleval_summary.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ teardown; log "ROLLBACK: booting the fi0616 (no 0138) daily from $ROLLBACK"; env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash $ROLLBACK > "$R/boot-rollback.log" 2>&1 && log "ROLLBACK daily up: $(grep -aoE 'Pool [0-9]+' "$R/boot-rollback.log" | tail -1)" || log "ROLLBACK FAILED too: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then curl -sf -m 5 $U/health >/dev/null || rollback; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R189 promote pcie_ipc start (lock held): $IMG, VLLM_SM12X_PCIE_IPC_AR=1 ==="
teardown
log "native-l2 kept (block size unchanged; the tier's pre-existing content is the restart-survival test): $(du -sh $L2 2>/dev/null | cut -f1)"
PIN=13980000000
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then :
else
  log "boot at the table pin FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"
  # R191 (2026-09-05 04:13 UTC): on this image + PCIE_IPC=1 the 13.98 GB pin left 725 MiB free on the first boot and the next two boots
  # died in the Triton warmup with "CUDA error: invalid argument" (no Bug C line); both booted at 13.5 GB (pool 985,621, 1.5-1.9 GB
  # free). So retry at 13.5 GB on ANY boot failure, not only on the Bug C signature.
  teardown; PIN=13500000000; log "retrying with KV_BYTES=$PIN (launcher table must be updated to this value if it boots)"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" KV_BYTES=$PIN bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "boot at $PIN FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"; rollback; exit 1; }
fi
log "DAILY UP pin=$PIN $(tail -1 "$R/boot-daily-$PIN.log" | cut -c1-260)"
BOOT=$(sudo docker logs vllm-27b 2>&1); echo "$BOOT" > "$R/engine-boot-daily.log"
log "[allreduce] $(echo "$BOOT" | grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" | head -1)"
echo "$BOOT" | grep -aE "PCIe IPC" | sed -E 's/^.*(PCIe IPC)/\1/' | sort -u | head -4 | cut -c1-220 | sed "s/^/[pcie] /" | tee -a "$R/audit.log"
log "[layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
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
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED"; fi; }
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
p1 code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-daily.json" > "$R/tooleval-daily.log" 2>&1 )
python3 $PR/tooleval_summary.py "$R/tooleval-daily.json" daily-pcieipc 2>&1 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "DAILY UP|allreduce|pcie\]|layout|cap |needle|RESULT|PROBE FAILED|tool-eval|error lines|FAILED|ROLLBACK" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R189 DONE — daily = pcie_ipc all-reduce on $IMG (pin $PIN); rollback $ROLLBACK ==="
