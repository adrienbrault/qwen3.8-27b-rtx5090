#!/usr/bin/env bash
# R185b (2026-09-05): the fidelity control R185 lacked. R185's PCIE-on arm scored 95.572% / +2.680% on the agentic ruler vs bf16
# (r168e's S image: 95.669% / +2.612%) and 0.00631 median |dlogprob| at 30K decode; neither cell has a same-image, same-day
# control (R183's BASE ran only ctx0 + the dense ruler, R185's PCIE-off arm only decode + ctx0). One arm: the pcieipc image
# with the knob UNSET (CustomAllreduce path) — deep30k, decode fidelity at 30K, agentic ruler. If OFF reproduces ON's numbers,
# the gap is the R182 SSM-bf16 daily vs the r168e image, not the kernel; if OFF matches r168e, the kernel moves the numerics.
# Helpers are r185-pcieipc.sh's (same rulers, same probes, same launch-daily EXP passthrough); writes into R185's results dir.
#   unit: sudo systemd-run --unit=r185b-pcieoff-fid --collect -p User=adrienbrault -p RuntimeMaxSec=5400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r185b-pcieoff-fid bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r185-pcieipc; do sleep 20; done; exec bash /srv/qwen5090/r185b-pcieoff-fid.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r185-pcieipc; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] || { log "ABORT: references missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval ar-bench; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R185b $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R185b start (lock held): PCIE-off fidelity control (deep30k, dfid 30K, agentic ruler) on $IMG ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
layout(){ curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' '; }
boot_arm(){ local tag=$1 pins=$2 kv rc; shift 2
  for kv in $pins; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB $(layout)"
      ELOG > "$R/engine-boot-$tag.log"
      log "[$tag allreduce] $(grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" "$R/engine-boot-$tag.log" | head -1)"
      grep -aE "PCIe IPC" "$R/engine-boot-$tag.log" | sed -E 's/^.*(PCIe IPC)/\1/' | sort -u | head -8 | cut -c1-220 | sed "s/^/[$tag pcie] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|unsupported|not supported|PCIe IPC|JointFailure" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }

teardown; wipe_l2
if boot_arm PCIE-off-fid "$PINS"; then
  sleep 20
  p1 PCIE-off-fid deep30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
  dfid PCIE-off-fid 30000
  ruler_agentic PCIE-off-fid
  log "[PCIE-off-fid engine error-lines] $(errs)"
  teardown
else log "[PCIE-off-fid] BOOT FAILED on every pin"; fi
grep -aE "arbench|BOOT OK|BOOT FAILED|ENABLED|NOT enabled|pcie\]|allreduce\]|cap |RESULT|PROBE FAILED|error-lines|errors after|vs bf16|decode ctx|pcie-tail|GATE" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
