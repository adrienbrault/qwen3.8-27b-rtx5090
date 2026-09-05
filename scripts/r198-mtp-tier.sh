#!/usr/bin/env bash
# R198 (2026-09-05, follows R197): does the MTP head cost the KV tiers and the prefix cache on this hybrid? The R197 MTP boots print
# "no KV cache group could be identified as the draft model's, so every group -- including Mamba groups [0, 1, 2] -- will be treated as a
# draft group ... prefix-cache reuse across requests will be disabled and any external KV offload tier will store without ever serving a
# hit". M3p (MTP ns3 on the pcie_ipc all-reduce, patch 0148) is the fastest c8/c16 arm and the best 30K single stream of the ladder, so
# that warning is the only thing standing between it and a promotion sheet. This unit boots M3p exactly as ladder5 did (EXP=1 :8029,
# SEQS 16, eval-l2 wiped, forced first config) and measures the warning: prefix-cache queries/hits and tier CPU→GPU bytes from /metrics
# across (a) the same 120K prompt asked twice (a warm revisit = GPU prefix-cache hit on the daily), (b) the R166d needle gate
# (131K + 220K cold, then the evicted re-asks that only the CPU/disk tier can serve; SUMMARY tier_served / tier_served_hits).
# The DFlash ns7 control is the same needle gate on the daily in r197-promote-ns7 (results/2026-09-05-r197-promote-ns7).
# Restores the daily at the end through daily-restore-retry.sh (queue-aware).
#   unit: sudo systemd-run --unit=r198-mtp-tier --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r198-mtp-tier bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r196-minima-audition || systemctl is-active -q r197-promote-ns7; do sleep 30; done; exec bash /srv/qwen5090/r198-mtp-tier.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r198-mtp-tier; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
IMG=${IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie}
PIN_ENV="EXTRA_ENV_APPEND=-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
grep -qF 'SPEC_METHOD_=${SPEC_METHOD:-dflash}' "$CAND" || { log "ABORT: launch-daily.sh SPEC_METHOD wiring missing"; exit 3; }
for t in kv_capacity_probe.py needle_gate.sh; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R198 ${1:-DONE} ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R198 start (lock held): MTP ns3 on pcie_ipc ($IMG) — tier + prefix-cache hit census ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
teardown; sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
booted=0
for kv in 13980000000 13500000000; do
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG SPEC_METHOD=mtp SPEC_NS=3 "$PIN_ENV" bash $CAND > "$R/boot-$kv.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null && { booted=1; log "[M3p] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$kv.log" | tail -1 | tr -dc 0-9)MiB"; break; }
  log "[M3p] boot pin=$kv FAILED: $(grep -aE 'FAILED' "$R/boot-$kv.log" | tail -1 | cut -c1-220)"; teardown
done
[ "$booted" = 1 ] || { log "[M3p] BOOT FAILED on every pin"; finish ABORTED; exit 1; }
sudo docker logs vllm-exp > "$R/engine-boot-M3p.log" 2>&1
log "[warn] $(grep -aoE 'no KV cache group could be identified[^\"]{0,160}' "$R/engine-boot-M3p.log" | head -1)"
log "[mtp-pcie] $(grep -aoE 'SM12X PCIe IPC: MTP drafter admitted[^\"]{0,80}' "$R/engine-boot-M3p.log" | head -1)  [layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ')"
snap(){ curl -s -m 5 $U/metrics | awk -v t="$1" '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} /^vllm:kv_offload_total_bytes_total.*CPU_to_GPU/ {up+=$NF} /^vllm:kv_offload_total_bytes_total.*GPU_to_CPU/ {down+=$NF} /^vllm:num_preemptions_total/ {pre=$NF} END {printf "[metrics %s] prefix_queries=%d prefix_hits=%d tier_GPU_to_CPU=%.3f GB tier_CPU_to_GPU=%.3f GB preemptions=%s\n", t, q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"], down/1e9, up/1e9, pre}' | tee -a "$R/audit.log"; }
sleep 20; snap boot
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
snap after-short
cap ctx120k-cold --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
snap after-120k-cold
cap ctx120k-revisit --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
snap after-120k-revisit
log "needle gate start (131K + 220K cold, then 12x90K flood evicts them, then two re-asks each through the tier)"
U=$U bash $PR/needle_gate.sh mtp "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate.log" | tail -1 | cut -c1-300)"
snap after-needles
log "[M3p engine error-lines] $(sudo docker logs vllm-exp 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  eval-l2 now $(du -sh $L2 2>/dev/null | cut -f1)"
grep -aE "BOOT|warn\]|mtp-pcie|metrics|cap |needle|error-lines|FAILED|DAILY|===" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
