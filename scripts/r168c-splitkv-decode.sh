#!/usr/bin/env bash
# R168c (2026-09-04): the 0136 split-KV knob is INVISIBLE to the teacher-forced rulers — r168b's ON-vs-OFF direct comparison was
# bit-identical (top-1 1.0000, KL 0.00000) because prompt-logprob scoring is a long-query prefill, which FlashInfer never splits
# either way. The knob only changes the short-query/long-KV path (spec verification, q = 10 against a 30K+ KV), so the
# adjudication must run THAT path:
#   decode_fidelity.py (R107b, built for decode-only kernels): T=0 greedy generations with per-token logprobs at ctx 0 and
#     30,000, compared ON vs OFF token-for-token; noise floor = ON run twice on the same boot (spec decoding at T=0 is exact
#     rejection sampling, so a correct kernel pair agrees except near-tie flips with tiny |Δlogprob|).
#   fidelity_ladder.py --mode depth2 (R156 powered depth: 5 bases x 50 continuations sharing a cached prefix at 1K/32K/128K/200K):
#     with prefix caching the continuation is a SHORT query against a long cached KV — exactly the guarded regime — and it is
#     teacher-forced, so fidelity_compare.py gives top-1 flips / PPL / truncated KL ON vs OFF per depth bin.
# Pass = decode agreement ON-vs-OFF at the noise floor (ON-vs-ON) and depth2 ON-vs-OFF within R156's same-arm noise floor.
# Unit (RuntimeMaxSec counts the flock wait): sudo systemd-run --unit=r168c-splitkv-decode --collect -p User=adrienbrault -p RuntimeMaxSec=28800 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r168c-splitkv-decode bash /srv/qwen5090/r168c-splitkv-decode.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r168c-splitkv-decode; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R168c split-KV decode-path adjudication start (lock held) ==="
U=http://127.0.0.1:8029
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
FD=/srv/qwen5090/results/2026-08-23-fidelity
CORPUS=/srv/qwen5090/r156-corpus.jsonl
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R168c $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
boot(){ local tag=$1 kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$2 ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT OK split_kv=$2 $(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$tag.log" | tail -1)"; return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || break; teardown
  done; return 1; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
dfid(){ # $1 tag $2 ctx
  python3 /srv/qwen5090/probes/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$1-ctx$2.jsonl" --chunks 20 --ctx "$2" --tokens 256 > "$R/dec-$1-ctx$2.out" 2>&1
  log "[$1 decode ctx$2] $(grep -ac ' ok ' "$R/dec-$1-ctx$2.out")/20 chunks"; }
d2(){ local tag=$1
  timeout 7200 python3 /srv/qwen5090/probes/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$CORPUS" --out "$R/dump-$tag-depth2.jsonl" --logprobs 20 --mode depth2 --depths "1000,32000,128000,200000" --depth-bases 5 --depth-conts 50 > "$R/score-$tag-depth2.out" 2>&1
  log "[$tag depth2] $(tail -1 "$R/score-$tag-depth2.out" | cut -c1-160)"; }
cmp_dec(){ log "[decode $1] $(python3 /srv/qwen5090/probes/decode_fidelity.py compare "$2" "$3" 2>&1 | tail -1 | cut -c1-300)"; }
cmp_d2(){ python3 /srv/qwen5090/probes/fidelity_compare.py --ref "$2" --arm "$3" --label "$1" --depth-bins "0,2000,64000,160000,262144" --json "$R/d2-$1.json" 2>&1 | grep -aE "^===|scored positions|top-1 agreement|corpus PPL|certain|confident|moderate|near-tie|FATAL|bin|depth" | cut -c1-200 | sed "s/^/[depth2 $1] /" | tee -a "$R/audit.log"; }

log "taking the daily down"; teardown; wipe_l2
if boot OFF 0; then dfid OFF 0; dfid OFF 30000; d2 OFF; errlines OFF; fi; teardown; wipe_l2
if boot ON 1; then dfid ON 0; dfid ON 30000; dfid ONb 30000; d2 ON; errlines ON; fi; teardown
log "--- comparisons ---"
[ -f "$R/dec-ON-ctx30000.jsonl" ] && [ -f "$R/dec-ONb-ctx30000.jsonl" ] && cmp_dec "noise floor ON-vs-ON ctx30000" "$R/dec-ON-ctx30000.jsonl" "$R/dec-ONb-ctx30000.jsonl"
[ -f "$R/dec-OFF-ctx30000.jsonl" ] && [ -f "$R/dec-ON-ctx30000.jsonl" ] && cmp_dec "ON-vs-OFF ctx30000" "$R/dec-OFF-ctx30000.jsonl" "$R/dec-ON-ctx30000.jsonl"
[ -f "$R/dec-OFF-ctx0.jsonl" ] && [ -f "$R/dec-ON-ctx0.jsonl" ] && cmp_dec "ON-vs-OFF ctx0" "$R/dec-OFF-ctx0.jsonl" "$R/dec-ON-ctx0.jsonl"
[ -s "$R/dump-OFF-depth2.jsonl" ] && [ -s "$R/dump-ON-depth2.jsonl" ] && cmp_d2 "ON-vs-OFF" "$R/dump-OFF-depth2.jsonl" "$R/dump-ON-depth2.jsonl"
grep -aE "BOOT OK|decode|depth2|error-lines" "$R/audit.log" | cut -c1-220 > "$R/sheet.txt"
finish DONE
