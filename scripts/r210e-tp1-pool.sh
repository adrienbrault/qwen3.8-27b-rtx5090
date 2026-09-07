#!/usr/bin/env bash
# R210e (2026-09-08, R210 series re-run, user question: "what kv pool with 1 card of the current daily?"): measure the served MTP-ns3
# configuration on ONE RTX 5090 instead of two, everything else held.
#
# The arithmetic says the second card is worth far more than 2x. From the daily's own boot log: the checkpoint is
# 21.81 GiB, model loading takes 10.19 GiB PER CARD at TP=2 (the embedding is offloaded to host RAM), the KV pin is
# 13.98 GB/card and yields 1,309,368 tokens -- so 10,677 B/token/card, 21,354 B/token in total. At TP=1 the weights
# stop being sharded (+10.19 GiB on the one card) while each token now costs the full 21,354 B, which squeezes the KV
# budget from ~26 GiB across two cards to ~5 GiB on one and predicts 215-280K tokens, ~19 % of the TP=2 pool. That is
# below the daily's own 262,144-token max-model-len, so the one-card shape may not admit a single full-length request.
# This unit replaces that estimate with a number.
#
#   C-tp2-control  TP=2, daily pin, EXP path -- confirms the instrument reproduces the daily's 1,309,368 on :8029
#   A-tp1-262k     TP=1, PCIE_IPC=0 (pcie_ipc is a two-card all-reduce), descending KV pins, max-model-len 262,144
#   B-tp1-131k     TP=1 at max-model-len 131,072, run only if A cannot boot at any pin -- separates "no KV budget"
#                  from "pool smaller than one request"
# R210 (first attempt) could not answer it: the TP=2 control's engine booted and reported the daily's 1,309,368 but was
# rejected by a stale EXP pool band (850K-1.1M, fixed in launch-daily.sh), and the first TP=1 boot four minutes later died
# in the Qwen Triton warmup on `torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32)` with cudaErrorInvalidValue -- a
# one-element int32 allocation, which cannot fail for a legitimate reason. That is the heavy-TP2 teardown transient
# (OPERATIONS.md sec 9, daily-restore-retry.sh's 300 s second attempt), not a TP=1 incompatibility and not an OOM, so
# descending the pin ladder against it would have burned 40 minutes on the same failure. This version settles 300 s
# before each TP=1 arm and retries a pin ONCE after another 300 s when the failure carries the transient's signature --
# and only then, so a genuine OOM or a pool-band rejection still falls straight through to the next pin.
#
# R210b then failed for two reasons of its own, both fixed here. Its `teardown 300` never settled 300 s: teardown()
# ignored its argument and always called settle with the 60 s default, so the transient recurred on the first TP=1 boot.
# And launch-daily.sh read $TP after its unset list -- which wipes TP, POOL_MIN and POOL_MAX on the experiment path too
# -- so TP=1 silently became TP=2 and the arm booted two cards at a 6.5 GB pin (pool 608,065 = 6.5e9 / 10,677 B/token,
# the TP=2 rate). The knob is EXP_TP now, alongside EXP_MAXLEN, because EXP_-prefixed names survive that unset.
# Largest pin that boots wins; the arm reports pool, min free VRAM and the cache layout. ~20 min including the restore.
#   unit: sudo systemd-run --unit=r210e-tp1-pool --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r210e-tp1-pool bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r210e-tp1-pool.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-08-r210e-tp1-pool; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=$(grep -oE '^DAILY_IMG=[^ ]+' /srv/qwen5090/launch-daily.sh | cut -d= -f2)
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2
TP1_PINS="6500000000 6000000000 5500000000 5000000000 4500000000 4000000000 3500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
grep -q 'TP_=${EXP_TP:-2}' "$CAND" || { log "ABORT: launch-daily.sh has no EXP-only TP passthrough"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle "${1:-60}"; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  grep -aE "BOOT OK|BOOT FAILED|layout|weights|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"; log "=== R210e $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R210e start (lock held): one-card KV pool of the served MTP ns3 shape, image $IMG ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }

transient(){ grep -aqE "hang signature|CUDA error: invalid argument|cudaErrorInvalidValue" "$1"; }
boot(){ local tag=$1 pins=$2 kv rc try; shift 2
  for kv in $pins; do
   for try in 1 2; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB"
      log "[$tag weights] $(sudo docker logs vllm-exp 2>&1 | grep -aoE 'Model loading took [0-9.]+ GiB' | tail -1) | $(sudo docker logs vllm-exp 2>&1 | grep -aoE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' | tail -1)"
      log "[$tag layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | tr ',' '\n' | grep -aE 'block_size=|kv_cache_size_tokens|num_gpu_blocks=|mamba_block|mamba_ssm' | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv try=$try FAILED rc=$rc: $(grep -aE 'FAILED|Error|error' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    mv "$R/boot-$tag-$kv.log" "$R/boot-$tag-$kv-try$try.log"
    if [ $try = 1 ] && transient "$R/boot-$tag-$kv-try$try.log"; then
      log "[$tag] pin=$kv carries the heavy-TP2 warmup transient — teardown + 300 s settle, retrying the SAME pin"; teardown 300
    else
      teardown; break
    fi
   done
  done; return 1; }

teardown
boot C-tp2-control 13980000000 PCIE_IPC=1 BSS=1 || log "[C-tp2-control] BOOT FAILED — the instrument itself is suspect, read the TP=1 arms with that in mind"
teardown 300
if boot A-tp1-262k "$TP1_PINS" EXP_TP=1 PCIE_IPC=0 BSS=0; then
  log "[A-tp1-262k] one card serves the daily's 262,144-token contract"
else
  log "[A-tp1-262k] BOOT FAILED on every pin down to 3.5 GB — trying half the max-model-len"
  teardown 300
  boot B-tp1-131k "$TP1_PINS" EXP_TP=1 PCIE_IPC=0 BSS=0 EXP_MAXLEN=131072 || log "[B-tp1-131k] BOOT FAILED on every pin"
fi
finish DONE
