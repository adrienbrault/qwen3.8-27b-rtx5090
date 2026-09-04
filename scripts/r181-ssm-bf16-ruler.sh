#!/usr/bin/env bash
# R181 (2026-09-04): the SSM-bf16 decision needs a bigger decode ruler at 30K. R180 (20 chunks): dense/agentic bf16 rulers neutral
# (92.771% / +0.744% vs BASE 92.716% / +0.770%), but the greedy decode ruler at 30K read median |dlogprob| 0.0063 vs 0.0037 and
# BF16 was farther on 10 of 15 comparable chunks (closer on 11 of 17 at ctx0). The dense ruler is a single prefill and barely reads
# a stored bf16 state; decode at 30K reads it every step. 20 chunks cannot settle a 1.7x. This run: a fresh bf16 decode reference
# with 80 chunks at ctx30000 (r173c recipe: bf16 weights, bf16 KV, no drafter, MAXLEN 40960), then BASE and BF16 arms on the daily
# image with the same 80 chunks, compared per chunk (probes/decode_paired.py). ~55 min, daily restored at the end.
# Unit: sudo systemd-run --unit=r181-ssm-bf16-ruler --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r181-ssm-bf16-ruler bash /srv/qwen5090/r181-ssm-bf16-ruler.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r181-ssm-bf16-ruler; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
CHUNKS=${CHUNKS:-80}; CTX=30000
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16MODEL=/srv/qwen5090/models/qwen3.8-27b-bf16; LAUNCH28=/srv/qwen5090/launch-daily-v0280.sh; FD=/srv/qwen5090/results/2026-08-23-fidelity
[ -f "$BF16MODEL/model.safetensors.index.json" ] && [ -f "$FD/corpus.jsonl" ] || { log "ABORT: bf16 checkpoint or corpus missing"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R181 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R181 SSM-bf16 decode ruler start (lock held): $CHUNKS chunks at ctx$CTX, bf16 reference regenerated ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
dfid(){ local T=$1
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$CTX.jsonl" --chunks $CHUNKS --ctx $CTX --tokens 256 > "$R/dec-$T-ctx$CTX.out" 2>&1
  log "[$T decode ctx$CTX] $(tail -1 "$R/dec-$T-ctx$CTX.out" | cut -c1-200)"; }
teardown
# --- bf16 reference ---
env -i PATH="$PATH" HOME="$HOME" USER="$USER" MODEL_DIR="$BF16MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 \
   UTIL=0.92 KVD_OVERRIDE=bfloat16 NOSPEC=1 NO_TIER=1 PREFIX_CACHE=0 SEQS=1 MNBT=2048 MAXLEN=40960 FIWS=268435456 POOL_MIN=1 POOL_MAX=99999999 \
   bash $LAUNCH28 > "$R/boot-bf16.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "bf16 reference boot FAILED: $(grep -aE 'FAILED|Error' "$R/boot-bf16.log" | tail -1 | cut -c1-200)"; finish ABORTED; exit 1; }
log "[bf16] BOOT OK $(grep -aoE 'Pool [0-9]+' "$R/boot-bf16.log" | tail -1)"
dfid bf16
[ -s "$R/dec-bf16-ctx$CTX.jsonl" ] || { log "bf16 reference dump EMPTY"; finish ABORTED; exit 1; }
log "[bf16 vs r173c 20-chunk reference, same config two boots] $(python3 $PR/decode_fidelity.py compare /srv/qwen5090/results/2026-09-04-r173c-bf16-decode/dec-bf16-ctx30000.jsonl "$R/dec-bf16-ctx$CTX.jsonl" 2>&1 | tail -1 | cut -c1-200)"
teardown
# --- arms on the daily image ---
boot_arm(){ local tag=$1 xargs=$2 kv=$3
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv EXTRA_ARGS_APPEND="$xargs" bash $CAND > "$R/boot-$tag.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "[$tag] BOOT FAILED: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"; return 1; }
  log "[$tag] BOOT OK pin=$kv args='$xargs' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag.log" | tail -1 | tr -dc 0-9) $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|mamba_ssm_cache_dtype="[a-z0-9]+"' | tr '\n' ' ')"; }
for arm in "BASE|" "BF16|--mamba-ssm-cache-dtype bfloat16"; do tag=${arm%%|*}; xargs=${arm#*|}
  wipe_l2; boot_arm $tag "$xargs" 13500000000 || { teardown; continue; }
  dfid $tag
  log "[$tag vs bf16 ctx$CTX] $(python3 $PR/decode_fidelity.py compare "$R/dec-bf16-ctx$CTX.jsonl" "$R/dec-$tag-ctx$CTX.jsonl" 2>&1 | tail -1 | cut -c1-260)"
  log "[$tag engine error-lines] $(sudo docker logs vllm-exp 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
  teardown
done
python3 $PR/decode_paired.py "$R/dec-bf16-ctx$CTX.jsonl" "$R/dec-BASE-ctx$CTX.jsonl" "$R/dec-BF16-ctx$CTX.jsonl" --label-a BASE --label-b BF16 > "$R/paired-ctx$CTX.txt" 2>&1
log "[PAIRED ctx$CTX BASE vs BF16] $(tail -1 "$R/paired-ctx$CTX.txt" | cut -c1-330)"
grep -aE "BOOT|decode|PAIRED|error-lines|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
