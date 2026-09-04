#!/usr/bin/env bash
# R173c (2026-09-04): the bf16 ruler for the DECODE path. r173b showed the speculative length is not numerically invisible:
# ns7-vs-ns9 greedy continuations diverge 19/20 chunks at ctx0 and 30K (floors exact, both pairs identical), the same magnitude
# as nvfp4-KV-vs-fp8-KV (r168d ON-vs-F). The R156 bf16 rulers are prefill-path and cannot rank ns7 against ns9. This unit
# generates the missing reference: bf16 weights + bf16 KV, no drafter, greedy decode_fidelity at ctx0 and (VRAM permitting)
# ctx30000 on the same corpus, then compares every existing decode dump against it — NS9A/NS7A (r173b, S image), F (rc2 fp8),
# S/ON/OFF2 (r168d). The ranking metric is median |dlogprob| on agreed tokens and the first-divergence positions, since any
# quantized path diverges from bf16 in nearly every 256-token greedy chunk.
# bf16 = 52 GB of 64 GB (R156 arm a): the pool at util 0.92 / SEQS 1 is 48,787 tokens, so MAXLEN_TRY (default 40960) fits; falls back to
# MAXLEN 3072 (ctx0 only) if the boot fails. First run (05:04 UTC) used 32768 and the 30K probe got HTTP 400: the probe pads by
# words (ctx/1.3) so its 30K prompt tokenizes past 32768-256. Rerun with DEPTHS=30000 MAXLEN_TRY=40960 (unit r173c2).
# Unit: sudo systemd-run --unit=r173c-bf16-decode --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r173c-bf16-decode bash /srv/qwen5090/r173c-bf16-decode.sh
# Rerun (30K only): sudo systemd-run --unit=r173c2-bf16-decode --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r173c2-bf16-decode -E DEPTHS=30000 -E MAXLEN_TRY=40960 bash /srv/qwen5090/r173c-bf16-decode.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode; mkdir -p "$R"
DEPTHS=${DEPTHS:-0 30000}; MAXLEN_TRY=${MAXLEN_TRY:-40960}
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
MODEL=/srv/qwen5090/models/qwen3.8-27b-bf16; LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
U=http://127.0.0.1:8029; FD=/srv/qwen5090/results/2026-08-23-fidelity
R173B=/srv/qwen5090/results/2026-09-04-r173b-ns-confirm; R168D=/srv/qwen5090/results/2026-09-04-r168d-splitkv-ref; R168C=/srv/qwen5090/results/2026-09-04-r168c-splitkv-decode
[ -f "$MODEL/model.safetensors.index.json" ] || { log "ABORT: bf16 checkpoint missing"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$R173B/dec-NS9A-ctx0.jsonl" ] || { log "ABORT: corpus or r173b dumps missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R173c${DEPTHS// /-} $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R173c bf16 decode ruler start (lock held) ==="
log "taking the daily down"; teardown
boot_bf16(){ local maxlen=$1
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" MODEL_DIR="$MODEL" TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 \
     UTIL=0.92 KVD_OVERRIDE=bfloat16 NOSPEC=1 NO_TIER=1 PREFIX_CACHE=0 SEQS=1 MNBT=2048 MAXLEN=$maxlen FIWS=268435456 POOL_MIN=1 POOL_MAX=99999999 \
     bash $LAUNCH > "$R/boot-bf16-$maxlen.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; }
MAXLEN=$MAXLEN_TRY
if boot_bf16 $MAXLEN_TRY; then :; else
  log "bf16 boot at MAXLEN $MAXLEN_TRY FAILED: $(grep -aE 'FAILED|Error' "$R/boot-bf16-$MAXLEN_TRY.log" | tail -1 | cut -c1-200); retrying at 3072 (ctx0 only)"; teardown
  MAXLEN=3072; boot_bf16 3072 || { log "bf16 boot at MAXLEN 3072 FAILED: $(grep -aE 'FAILED|Error' "$R/boot-bf16-3072.log" | tail -1 | cut -c1-200)"; finish ABORTED; exit 1; }
fi
log "[bf16] BOOT OK maxlen=$MAXLEN $(grep -aoE 'Pool [0-9]+' "$R/boot-bf16-$MAXLEN.log" | tail -1) kv=$(sudo docker logs vllm-exp 2>&1 | grep -aoE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)"
dfid(){ python3 /srv/qwen5090/probes/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-bf16-ctx$1.jsonl" --chunks 20 --ctx "$1" --tokens 256 > "$R/dec-bf16-ctx$1.out" 2>&1
  log "[bf16 decode ctx$1] $(tail -1 "$R/dec-bf16-ctx$1.out" | cut -c1-160)"; }
for d in $DEPTHS; do [ "$d" -gt 0 ] && [ "$MAXLEN" -lt 32768 ] && { log "[bf16 decode ctx$d] skipped (maxlen $MAXLEN)"; continue; }; rm -f "$R/dec-bf16-ctx$d.jsonl"; dfid "$d"; [ -s "$R/dec-bf16-ctx$d.jsonl" ] || { log "[bf16 decode ctx$d] EMPTY dump — removed"; rm -f "$R/dec-bf16-ctx$d.jsonl"; }; done
sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[bf16 engine error-lines] /" | tee -a "$R/audit.log"
teardown
cmp_dec(){ [ -f "$2" ] && [ -f "$3" ] || { log "[decode $1] skipped (dump missing)"; return; }
  log "[decode $1] $(python3 /srv/qwen5090/probes/decode_fidelity.py compare "$2" "$3" 2>&1 | tail -1 | cut -c1-300)"; }
log "--- every arm vs bf16 (ref = bf16; median |dlogprob| on agreed tokens ranks) ---"
for c in 0 30000; do [ -s "$R/dec-bf16-ctx$c.jsonl" ] || { log "[decode ctx$c] no bf16 dump — skipped"; continue; }
  cmp_dec "NS9A(S,ns9)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R173B/dec-NS9A-ctx$c.jsonl"
  cmp_dec "NS7A(S,ns7)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R173B/dec-NS7A-ctx$c.jsonl"
  cmp_dec "F(rc2 fp8 KV,ns9)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R168D/dec-F-ctx$c.jsonl"
  cmp_dec "S(rc1+fi0616,ns9)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R168D/dec-S-ctx$c.jsonl"
  cmp_dec "ON(rc2 split,ns9)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R168C/dec-ON-ctx$c.jsonl"
  cmp_dec "OFF(rc2 knob off,ns9)-vs-bf16 ctx$c" "$R/dec-bf16-ctx$c.jsonl" "$R168C/dec-OFF-ctx$c.jsonl"
done
grep -aE "BOOT OK|decode|error-lines" "$R/audit.log" | cut -c1-300 > "$R/sheet.txt"
finish DONE
