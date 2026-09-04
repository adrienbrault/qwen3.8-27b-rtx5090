#!/usr/bin/env bash
# R179 (2026-09-04, user "what about the optimization ideas"): can the per-request pool cost (R178: ~6.4% fixed + ~0.2%/1K tokens,
# so 15 short or four 100K requests fill the 903K "token" pool) be cut without losing prefix caching? Three knobs on the daily
# image, :8029 (EXP=1), accounting + speed only (fidelity gate later for the dtype arm):
#   MB2   --mamba-block-size 5888   (2 x 2,944: half as many GDN state snapshots per context; coarser mamba prefix reuse)
#   MB4   --mamba-block-size 11776  (4 x)
#   BF16  --mamba-ssm-cache-dtype bfloat16 (half-size SSM state; the attention block may shrink; pool may leave the EXP band, so
#         the pin ladder steps down to keep it < 1M)
# Per arm: pool + cache_config_info; kv_capacity_probe short x1, short x4, 30K, 100K (usage %), 5 x 100K ignore_eos co-residency
# (max_running is the answer), a 100K revisit (2nd send same seed: TTFT tells whether the mamba prefix still hits), decode_ss code c8 x2
# + prose c1 x2. Restores the daily at the end (queue-aware).
# Unit: sudo systemd-run --unit=r179-pool-cost --collect -p User=adrienbrault -p RuntimeMaxSec=5400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r179-pool-cost bash /srv/qwen5090/r179-pool-cost.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r179-pool-cost; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R179 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R179 pool-cost knobs start (lock held): image=$IMG arms MB2/MB4/BF16 on :8029 (SEQS 16) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
boot_arm(){ local tag=$1 xargs=$2 kv rc; shift 2
  for kv in "$@"; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv EXTRA_ARGS_APPEND="$xargs" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv args='$xargs' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB"
      log "[$tag] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|mamba_block_size="[0-9]+"|mamba_ssm_cache_dtype="[a-z0-9]+"|mamba_cache_mode="[a-z]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"|kv_cache_max_concurrency="[0-9.]+"' | tr '\n' ' ')"
      log "[$tag] $(sudo docker logs vllm-exp 2>&1 | grep -aoE 'Setting attention block size to [0-9]+ tokens[^.]*|Add [0-9]+ padding layers, may waste at most [0-9.]+%' | sort -u | tr '\n' ';')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"; teardown
  done; return 1; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
arm(){ local tag=$1 xargs=$2; shift 2
  boot_arm $tag "$xargs" "$@" || { log "[$tag] BOOT FAILED on every pin"; return 1; }
  sleep 20
  cap $tag short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 11
  cap $tag short4 --ctx 0 --conc 4 --tokens 400 --ignore-eos --seed 12
  cap $tag ctx30k --ctx 36000 --conc 1 --tokens 200 --ignore-eos --seed 13
  cap $tag ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 14
  cap $tag revisit100k --ctx 120000 --conc 1 --tokens 32 --seed 14
  cap $tag five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 15
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  log "[$tag engine error-lines] $(sudo docker logs vllm-exp 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
  teardown; }
teardown
arm MB2  "--mamba-block-size 5888"  13980000000 13500000000
arm MB4  "--mamba-block-size 11776" 13980000000 13500000000
arm BF16 "--mamba-ssm-cache-dtype bfloat16" 13980000000 12000000000 10000000000
grep -aE "BOOT OK|block_size|padding|cap |RESULT|FAILED|error-lines" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
