#!/usr/bin/env bash
# R190d (2026-09-05): rerun of R190 item D. The first run (03:56 UTC) booted the r190diag image with 0142 sync census + 0143 collective
# tags, printed both proof lines, then the first decode killed both ranks: "SYNC_CENSUS does not support concurrent runner/output
# calls" — 0142's exclusive scope lock rejected the async-scheduling overlap (execute N+1 on the main thread while get_output N drains
# on the output thread) that is the served regime. 0142c (codex, BRIEF25b/NOTES25b) replaces it with thread-owned scopes and
# reference-counted global hooks; image ...-pcieipc-r190diag rebuilt from 0142c + 0143. Same procedure as R190 D: boot on :8029 with
# the daily flags + PCIE_IPC=1, both knobs on, one c1 code decode so the census window (20 warm-up + 50 steps) completes, save the
# per-call-site table. Chain: waits on r190c; r189 re-issued to wait on this unit.
#   unit: sudo systemd-run --unit=r190d-diag --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=600 \
#         -E GPU_QUEUE_NAME=r190d-diag bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r190c-dispatch; do sleep 30; done; exec bash /srv/qwen5090/r190d-diag.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r190-microbench; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
PR=/srv/qwen5090/probes; CAND=/srv/qwen5090/launch-daily.sh; U=http://127.0.0.1:8029
IMG_DG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-r190diag
sudo docker image inspect "$IMG_DG" >/dev/null 2>&1 || { log "ABORT: image $IMG_DG missing"; exit 3; }
sudo docker run --rm --entrypoint grep "$IMG_DG" -c "_Hooks" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sync_census.py 2>/dev/null | grep -qv '^0$' || { log "ABORT: $IMG_DG does not carry 0142c (no _Hooks in sync_census.py)"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-30}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval r190-bench; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R190d $1 ==="; }
trap 'log "### SIGTERM ###"; finish KILLED; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R190d diag rerun start (lock held): 0142c sync census + 0143 tags on $IMG_DG ==="
teardown
booted=0
for kv in 13980000000 13500000000; do
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG_DG \
    EXTRA_ENV_APPEND="-e VLLM_SM12X_SYNC_CENSUS=1 -e VLLM_SM12X_COLLECTIVE_TAGS=1" bash $CAND > "$R/boot-diag2-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then booted=1; log "[diag2] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-diag2-$kv.log" | tail -1 | tr -dc 0-9)"; break; fi
  log "[diag2] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-diag2-$kv.log" | tail -1 | cut -c1-220)"
  sudo docker logs vllm-exp 2>&1 | grep -aiE "error|exception|census|COLLECTIVE_TAGS" | head -6 | cut -c1-200 | sed "s/^/[diag2 boot-err] /" | tee -a "$R/audit.log"; teardown
done
if [ $booted = 1 ]; then
  sudo docker logs vllm-exp > "$R/engine-boot-diag2.log" 2>&1
  log "[diag2 proof] census=$(grep -ac 'SM12X sync census active' "$R/engine-boot-diag2.log") tags=$(grep -ac 'VLLM_SM12X_COLLECTIVE_TAGS=1 active' "$R/engine-boot-diag2.log") pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-diag2.log")"
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 1024 --runs 2 --kind code --out "$R/decode-diag2-c1.jsonl" > "$R/probe-diag2-c1.out" 2> "$R/probe-diag2-c1.err"
  grep -a "RESULT\|error" "$R/probe-diag2-c1.out" | sed "s/^/[diag2 c1] /" | cut -c1-260 | tee -a "$R/audit.log"
  sleep 5; sudo docker logs vllm-exp > "$R/engine-diag2-full.log" 2>&1
  awk '/SM12X sync census complete/{p=1} p{print} /^[^ ]/ && p && ++n>400{exit}' "$R/engine-diag2-full.log" > "$R/sync-census-table.txt"
  log "[diag2 census] table lines=$(wc -l < "$R/sync-census-table.txt"); $(grep -a 'sync census complete' "$R/engine-diag2-full.log" | head -1 | cut -c1-160)"
  grep -aE "^ *[0-9]+ +(target|sample\+draft|output) " "$R/sync-census-table.txt" | awk '{c[$2" "$4" "$5]+=$3} END{for(k in c) print c[k], k}' | sort -rn | head -20 | sed "s/^/[diag2 census top] /" | tee -a "$R/audit.log"
  grep -aE "^ *[0-9]+ total " "$R/sync-census-table.txt" | awk '{s+=$3; n++} END{if(n) printf "%d steps, mean %.1f syncs/step\n", n, s/n}' | sed "s/^/[diag2 census] /" | tee -a "$R/audit.log"
  log "[diag2 errors] $(grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError' "$R/engine-diag2-full.log")"
fi
grep -aE "fused-norm|census\]|dispatch|gdn-spec|diag" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
