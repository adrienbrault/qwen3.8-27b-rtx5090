#!/usr/bin/env bash
# R206c (2026-09-06, user: "Can the new MTP run c32 c64?"): the R200b pool arithmetic says MTP ns3 admits 74 requests (17,584 tok-equiv of
# 1,309,368) where DFlash ns7 admits 36 (c64 impossible, R200). This unit boots M3 (<daily>-mtppcie-mtpcache-eagleshift, MTP ns3) at SEQS 64
# and measures decode_ss code/prose at c16 (anchor to R206), c32 and c64, then the DF control (daily image, DFlash ns7) at SEQS 64 for c16/c32
# (its c64 is the R200 no-result). Read-outs: pool at SEQS 64, max "Running: N reqs" the engine ever reached, min free VRAM, accept per draft.
# Boots at the 13.98 pin, EXP=1 on :8029, gpu-queue chained (no daily bounce), restores the daily at the end.
#   unit: sudo systemd-run --unit=r206c-mtp-c32c64 --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r206c-mtp-c32c64 bash /srv/qwen5090/r206c-mtp-c32c64.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r206c-mtp-c32c64; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); MTP=$DAILY-mtppcie-mtpcache-eagleshift
PIN=13980000000; PIN_FALLBACK=13500000000
PROOF='SM12X eagle-drop replay boundary retained'
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
for i in "$DAILY" "$MTP"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
sudo docker run --rm --entrypoint cat "$MTP" /opt/prs-markers/0158 2>/dev/null | grep -q 0158 || { log "ABORT: marker 0158 missing in $MTP"; exit 3; }
grep -q "R197 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R197 SPEC_METHOD passthrough"; exit 3; }
for t in decode_ss.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done

. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R206c $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R206c start (lock held): c32/c64 at SEQS 64, M3=$MTP (mtp ns3) then DF control=$DAILY (dflash ns7) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_once TAG IMG PIN [SPEC_METHOD SPEC_NS] → 0 = up
boot_once(){ local tag=$1 img=$2 kv=$3 method=${4:-dflash} ns=${5:-7} rc extra="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1"
  [ "$method" = mtp ] && extra="$extra -e VLLM_SM12X_PCIE_IPC_MTP=1"
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=64 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$img SPEC_METHOD=$method SPEC_NS=$ns EXTRA_ENV_APPEND="$extra" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    log "[$tag] BOOT OK pin=$kv spec=$method/ns$ns pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) proof_0158=$(grep -ac "$PROOF" "$R/engine-boot-$tag.log") min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB aot_loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log") compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception|headroom" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
boot_arm(){ local tag=$1; shift 1; boot_once "$tag" "$@" || boot_once "${tag}b" "$@" ; }   # positional: IMG PIN [method ns]; retry once at the same pin
layout(){ log "[$1 layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
arm_report(){ local tag=$1
  log "[$tag] max running: $(ELOG | grep -aoE 'Running: [0-9]+ reqs' | sort -t' ' -k2 -n | tail -1) max waiting: $(ELOG | grep -aoE 'Waiting: [0-9]+ reqs' | sort -t' ' -k2 -n | tail -1) min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB engine error lines: $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')  prefix: $(curl -s -m 5 $U/metrics | awk '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} END {printf "queries=%d hits=%d", q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"]}')"
  ELOG > "$R/engine-$tag-final.log"; }

conc_rows(){ local tag=$1 c64=$2
  p1 $tag code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  p1 $tag code-c32 --conc 32 --tokens 1024 --runs 2 --kind code
  p1 $tag prose-c32 --conc 32 --tokens 1024 --runs 2 --kind prose
  if [ "$c64" = 1 ]; then
    p1 $tag code-c64 --conc 64 --tokens 1024 --runs 2 --kind code
    p1 $tag prose-c64 --conc 64 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c32-after64 --conc 32 --tokens 1024 --runs 1 --kind code
  fi; }

### M3 arm (the question): SEQS 64, c16/c32/c64
teardown; wipe_l2
if boot_arm M3 "$MTP" $PIN mtp 3; then sleep 20; layout M3; conc_rows M3 1; arm_report M3; else log "[M3] ARM FAILED"; fi

### DF control: SEQS 64, c16/c32 (c64 = R200 no-result, not re-run)
teardown; wipe_l2
if boot_arm DF "$DAILY" $PIN dflash 7; then sleep 20; layout DF; conc_rows DF 0; arm_report DF; else log "[DF] ARM FAILED"; fi

{ echo "R206c c32/c64 at SEQS 64 — M3=$MTP (mtp ns3) vs DF control=$DAILY (dflash ns7)"
  grep -aE "BOOT OK|FAILED|layout|revisit|RESULT|PROBE FAILED|tool-eval|needle|max running|error lines|proof|ENGINE DIED|ARM FAILED" "$R/audit.log" | cut -c1-330; } > "$R/sheet.txt"
finish DONE
