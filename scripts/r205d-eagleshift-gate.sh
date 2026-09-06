#!/usr/bin/env bash
# R205d (2026-09-06): gate patch 0158 (eagle-drop Mamba replay boundary) live on the R205 MTP route. R205c found the mechanism behind
# R205's prefix_hits=0: under MTP the scheduler checkpoints the Mamba state one block before the FullAttn drop, and reachable_block_mask
# (live prefix_cache_retention_interval=0) retains only the block after it, so no Mamba state block gets a hash and every first revisit
# misses (send1 0 / send2 0 with the junction pinned / send3 hits). 0158 retains the backed-off boundary too. Offline the fix hits
# 51,216 of 53,468 (= dense-retention control). This unit boots <mtpcache>-eagleshift on :8029 with the R205 launch config (MTP ns3,
# PCIe MTP, FORCE_FIRST) and runs the caching gates that R205 failed:
#   warm_equal gpu-warm + tier-flood (T=0 cold-vs-warm token equality WITH hits: the retained state must be the right state),
#   warm-revisit 32K gpu-warm + tier-flood (revisit ttft ≪ cold; R205: 7.8 s = cold),
#   reask 6K x3 (send 2 must hit ≈4416; R205c: 0 / 0 / 4416),
#   needle gate (tier_served must leave 0/4; R205 0/4 although 2.4 GB tier reads happened),
#   decode_ss code-c8 / prose-c1 sanity (R205 M3c: 1,501.8 / 165.1).
# Waits for the image (build unit r205d-build runs in parallel, CPU only) before taking the GPU lock.
#   unit: sudo systemd-run --unit=r205d-eagleshift-gate --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r205d-eagleshift-gate bash /srv/qwen5090/r205d-eagleshift-gate.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r205d-eagleshift-gate; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); IMG=$DAILY-mtppcie-mtpcache-eagleshift
PIN=13980000000; PIN_FALLBACK=13500000000
PROOF='SM12X eagle-drop replay boundary retained'
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in warm-revisit.py warm_equal.py reask.py needle_gate.sh decode_ss.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
for i in $(seq 120); do sudo docker image inspect "$IMG" >/dev/null 2>&1 && break; [ $i = 1 ] && log "waiting for image $IMG (build unit)"; sleep 30; done
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG never appeared"; exit 3; }
sudo docker run --rm --entrypoint cat "$IMG" /opt/prs-markers/0158 | grep -q 0158 || { log "ABORT: marker 0158 missing in $IMG"; exit 3; }

. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R205d $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R205d start (lock held): $IMG (0158 on mtpcache), MTP ns3, same route as R205/R205c ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
teardown; wipe_l2
boot_once(){ local tag=$1 kv=$2 rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG SPEC_METHOD=mtp SPEC_NS=3 EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) proof_lines=$(grep -ac "$PROOF" "$R/engine-boot-$tag.log") min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
    log "[$tag proof] $(grep -ao "$PROOF[^\"]{0,160}" "$R/engine-boot-$tag.log" | head -1)"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
if boot_once ES $PIN; then :; elif boot_once ESb $PIN_FALLBACK; then :; else log "BOOT FAILED at both pins"; finish ABORTED; exit 1; fi
snap(){ curl -s -m 5 $U/metrics | awk -v t="$1" '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} /^vllm:kv_offload_total_bytes_total.*CPU_to_GPU/ {up+=$NF} /^vllm:kv_offload_total_bytes_total.*GPU_to_CPU/ {down+=$NF} /^vllm:num_preemptions_total/ {pre=$NF} END {printf "[metrics %s] prefix_queries=%d prefix_hits=%d tier_GPU_to_CPU=%.3f GB tier_CPU_to_GPU=%.3f GB preemptions=%s\n", t, q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"], down/1e9, up/1e9, pre}' | tee -a "$R/audit.log"; }
revisit(){ local name=$1; shift 1
  python3 $PR/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 "$@" > "$R/revisit-$name.log" 2>&1
  log "[revisit $name] $(grep -a RESULT "$R/revisit-$name.log" | cut -c1-330)"; }
equal(){ local name=$1; shift 1
  python3 $PR/warm_equal.py --url $U --model qwen3.8-27b --n 3 --ctx 6000 --max-tokens 48 "$@" > "$R/warm-equal-$name.log" 2>&1
  log "[warm-equal $name] $(tail -1 "$R/warm-equal-$name.log") $(grep -a RESULT "$R/warm-equal-$name.log" | cut -c1-420)"; }
reask(){ local name=$1; shift 1
  python3 $PR/reask.py --url $U --model qwen3.8-27b --ctx 6000 "$@" > "$R/reask-$name.log" 2>&1
  log "[reask $name] $(grep -a RESULT "$R/reask-$name.log" | cut -c1-360)"; }
p1(){ local name=$1; shift 1
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[$name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[$name] PROBE FAILED: $(tail -1 "$R/probe-$name.err" | cut -c1-160)"; fi; }
sleep 10; snap boot
reask compl-mt8-gap0 --api completions --max-tokens 8 --gap 0 --sends 3 --seed a
snap after-reask
equal gpu-warm
revisit gpu-warm
snap after-gpu-warm
revisit tier-flood --flood 12 --flood-ctx 90000
equal tier-flood --flood 12 --flood-ctx 90000
snap after-tier-flood
log "needle gate start (131K + 220K cold, 12x90K flood, two re-asks each through the tier; R205: tier_served 0/4)"
U=$U bash $PR/needle_gate.sh eagleshift "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate.log" | tail -1 | cut -c1-330)"
grep -aE "^\[needle-gate" "$R/needle-gate.log" | grep -aE "depth=" | cut -c1-200 | sed "s/^/[needle] /" | tee -a "$R/audit.log"
snap after-needles
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
ELOG > "$R/engine-full.log"
log "[engine error-lines] $(grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError' "$R/engine-full.log")"
{ echo "R205d 0158 eagle-shift gate — $IMG"; grep -a "reask\|warm-equal\|revisit\|needle gate rc\|metrics\|RESULT\|proof\|BOOT OK\|error-lines" "$R/audit.log" | cut -c1-360; } > "$R/sheet.txt"
finish DONE
