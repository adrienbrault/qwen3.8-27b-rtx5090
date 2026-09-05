#!/usr/bin/env bash
# R200b (2026-09-05, follow-up to R200): the daily's pool admits ~36 requests, not 64 — each request occupies 28,613 tokens-equivalent
# (2.72 % of the 1,052,277 pool) the moment it is scheduled, constant over 1,200 generated tokens (probes/pool_cost_probe.py on :8020).
# That is 380 MB at 13.3 KB/token, five times the bf16 GDN state of one request (48 layers x 48 heads x 128 x 128 x 2 B = 75 MB), so
# something multiplies the state per request. This unit isolates the multiplier on :8029 with the daily image/flags/pin, one boot per arm:
#   A  SPEC_NS=1             ns1 (NOSPEC is not wired on the EXP path): with B (ns3) and the daily ns7 this gives the slope in ns
#   B  SPEC_NS=3             ns3: cost scales with ns?
#   C  SSM_DTYPE=float32     fp32 state: cost doubles if it is the state itself (R182 halved the state dtype and gained 117K pool tokens at SEQS 16)
#   D  SPEC_METHOD=mtp ns3   the MTP path (mtppcie image) for comparison
# Each arm: boot, pool_cost_probe --conc 8 14 --tokens 600, layout line, teardown. Restores the daily at the end (~25 min total).
#   unit: sudo systemd-run --unit=r200b-pool-cost --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r200b-pool-cost bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r200b-pool-cost.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r200b-pool-cost; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=$(grep -oE '^DAILY_IMG=[^ ]+' /srv/qwen5090/launch-daily.sh | cut -d= -f2); MTP_IMG=$IMG-mtppcie
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes; PINS="13980000000 13500000000"
MTP_PIN_ENV="EXTRA_ENV_APPEND=-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1"
for i in "$IMG" "$MTP_IMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
[ -f "$PR/pool_cost_probe.py" ] || { log "ABORT: pool_cost_probe.py missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  grep -aE "BOOT OK|BOOT FAILED|layout|pool-cost|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"; log "=== R200b $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R200b start (lock held): per-request pool cost by spec method / ns / state dtype, $IMG ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
boot(){ local tag=$1 kv rc; shift
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB"
      log "[$tag layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | tr ',' '\n' | grep -aE 'block_size=|kv_cache_size_tokens|num_gpu_blocks=|mamba_block|mamba_cache_mode|mamba_ssm' | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"; teardown
  done; return 1; }
arm(){ local tag=$1; shift; teardown
  boot "$tag" "$@" || { log "[$tag] BOOT FAILED on every pin"; return 1; }
  sleep 15
  python3 $PR/pool_cost_probe.py --url $U --conc 8 14 --tokens 600 2>&1 | sed "s/^/[$tag pool-cost] /" | cut -c1-330 | tee -a "$R/audit.log"; }
arm A-ns1 SPEC_NS=1
arm B-ns3 SPEC_NS=3
arm C-fp32state SSM_DTYPE=float32
arm D-mtp3 CAND_IMG=$MTP_IMG SPEC_METHOD=mtp SPEC_NS=3 "$MTP_PIN_ENV"
finish DONE
