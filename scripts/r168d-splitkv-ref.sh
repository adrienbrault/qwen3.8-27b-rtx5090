#!/usr/bin/env bash
# R168d (2026-09-04): r168c could not adjudicate the 0136 split-KV knob on its own. ON-vs-ON on one boot is exact (20/20 chunks,
# |Δlogprob| 0.0), but ON-vs-OFF = different boots: decode diverged 18/20 at 30K AND 17/20 at ctx 0, and the depth2 ladder read
# 92.8% top-1 with 7/110 flips where the reference was >0.9 sure, flips 7.2% / 1.2% / 10.4% / 10.0% by depth bin. That is far
# below R156's cross-boot same-arm floor (99.2%) — but that floor was dense long-query scoring, not depth2. Two missing
# pieces, both measured here on the same image as r168c (rc2 prs):
#   OFF2 = knob OFF booted again: OFF2-vs-OFF is the cross-boot floor for depth2 AND for the decode probe.
#   F    = fp8 KV shape (r169 arm F: rc2 + embed offload): the guard never applies to fp8, and fp8 KV is 0.13 pp from bf16
#          (R156), so it is the reference — whichever of ON / OFF sits closer to F is the more faithful path.
#   S    = the alternative fix, FlashInfer 0.6.16.post3 swapped under the rc1 prs image (split-KV on by default there): the
#          same probes tell whether its split path matches ON, OFF or neither.
# Comparisons: decode_fidelity (ctx0, ctx30000) and depth2 for every pair against F and against OFF; sheet per pair with the
# certain-bucket flip rate and the by-depth rates. Verdict rule: ON is acceptable if its distance to F (certain-bucket flips,
# by-depth flips) is within the OFF2-vs-OFF floor of OFF's distance to F; otherwise the knob corrupts the short-query path
# (FlashInfer's stated reason for the guard) and the 0.29 route ships with the 0.6.16 swap layer (if S is clean) or not at all.
# Unit: sudo systemd-run --unit=r168d-splitkv-ref --collect -p User=adrienbrault -p RuntimeMaxSec=36000 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r168d-splitkv-ref bash /srv/qwen5090/r168d-splitkv-ref.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r168d-splitkv-ref; mkdir -p "$R"
C=/srv/qwen5090/results/2026-09-04-r168c-splitkv-decode
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
RC2_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs
FI16_IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed-fi0616
for i in $RC2_IMG $FI16_IMG; do sudo docker image inspect $i >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
[ -s "$C/dump-OFF-depth2.jsonl" ] && [ -s "$C/dump-ON-depth2.jsonl" ] || { log "ABORT: r168c dumps missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R168d split-KV reference arms start (lock held) ==="
U=http://127.0.0.1:8029
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
FD=/srv/qwen5090/results/2026-08-23-fidelity
CORPUS=/srv/qwen5090/r156-corpus.jsonl
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R168d $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
boot_cand(){ local tag=$1 split=$2 img=$3 kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$split CAND_IMG=$img ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT OK split_kv=$split image=$img $(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$tag.log" | tail -1) flashinfer=$(sudo docker exec vllm-exp python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)"; return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || break; teardown
  done; return 1; }
# Box-state boot retry (flan-heavy-tp2-cadence; r170 2026-09-04 lost both arms to "CUDA error: invalid argument" in the
# TP1 warm-up after a chain of TP2 boots). If a boot fails with that signature: teardown, GPUs idle + 300 s, ONE retry.
boxstate(){ sudo docker logs vllm-exp 2>&1 | grep -aq "CUDA error: invalid argument" || grep -aq "CUDA error: invalid argument" "$R/boot-$1.log" 2>/dev/null; }
boot_retry(){ local tag=$1; shift   # usage: boot_retry <tag> <boot fn> [args]
  "$@" && return 0
  if boxstate "$tag"; then log "[$tag] box-state boot failure (invalid argument in warm-up): teardown + 300 s idle, retrying once"; teardown; settle 300; "$@" && return 0; fi
  return 1; }
COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
boot_F(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$RC2_IMG" KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92 EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9}' EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS" EXTRA_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens" bash $LAUNCH > "$R/boot-F.log" 2>&1; rc=$?
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[F] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-F.log" | tail -1 | cut -c1-200)"; return 1; fi
  log "[F] BOOT OK fp8 KV, pool=$(sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9)"; return 0; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
dfid(){ python3 /srv/qwen5090/probes/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$1-ctx$2.jsonl" --chunks 20 --ctx "$2" --tokens 256 > "$R/dec-$1-ctx$2.out" 2>&1
  log "[$1 decode ctx$2] $(grep -ac ' ok ' "$R/dec-$1-ctx$2.out")/20 chunks"; }
d2(){ local tag=$1
  timeout 7200 python3 /srv/qwen5090/probes/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$CORPUS" --out "$R/dump-$tag-depth2.jsonl" --logprobs 20 --mode depth2 --depths "1000,32000,128000,200000" --depth-bases 5 --depth-conts 50 > "$R/score-$tag-depth2.out" 2>&1
  log "[$tag depth2] $(tail -1 "$R/score-$tag-depth2.out" | cut -c1-160)"; }
measure(){ dfid $1 0; dfid $1 30000; d2 $1; errlines $1; }
cmp_dec(){ [ -f "$2" ] && [ -f "$3" ] || return; log "[decode $1] $(python3 /srv/qwen5090/probes/decode_fidelity.py compare "$2" "$3" 2>&1 | tail -1 | cut -c1-300)"; }
cmp_d2(){ [ -s "$2" ] && [ -s "$3" ] || return
  python3 /srv/qwen5090/probes/fidelity_compare.py --ref "$2" --arm "$3" --label "$1" --depth-bins "0,2000,64000,160000,262144" --json "$R/d2-$1.json" > "$R/d2-$1.txt" 2>&1
  python3 - "$R/d2-$1.json" "$1" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys; d=json.load(open(sys.argv[1])); b=d["buckets"]; dp=d.get("depth",{})
print(f"[depth2 {sys.argv[2]}] top-1 {d['top1_agreement']*100:.2f}%  certain {b['certain']['flips']}/{b['certain']['n']} ({b['certain']['rate']*100:.2f}%)  confident {b['confident']['rate']*100:.2f}%  moderate {b['moderate']['rate']*100:.2f}%  near-tie {b['near-tie']['rate']*100:.2f}%  KL {d.get('kl_trunc_mean')}  by-depth " + " ".join(f"{k}:{v['rate']*100:.1f}%" for k,v in sorted(dp.items(), key=lambda kv:int(kv[0]))))
PY
}

log "taking the daily down"; teardown; wipe_l2
if boot_retry OFF2 boot_cand OFF2 0 $RC2_IMG; then measure OFF2; fi; teardown; wipe_l2
if boot_retry F boot_F; then measure F; fi; teardown; wipe_l2
if boot_retry S boot_cand S 0 $FI16_IMG; then measure S; fi; teardown
log "--- comparisons (ref first) ---"
cmp_dec "floor OFF2-vs-OFF ctx0"      "$C/dec-OFF-ctx0.jsonl"     "$R/dec-OFF2-ctx0.jsonl"
cmp_dec "floor OFF2-vs-OFF ctx30000"  "$C/dec-OFF-ctx30000.jsonl" "$R/dec-OFF2-ctx30000.jsonl"
for a in OFF ON; do cmp_dec "$a-vs-F ctx0" "$R/dec-F-ctx0.jsonl" "$C/dec-$a-ctx0.jsonl"; cmp_dec "$a-vs-F ctx30000" "$R/dec-F-ctx30000.jsonl" "$C/dec-$a-ctx30000.jsonl"; done
for a in OFF2 S; do cmp_dec "$a-vs-F ctx0" "$R/dec-F-ctx0.jsonl" "$R/dec-$a-ctx0.jsonl"; cmp_dec "$a-vs-F ctx30000" "$R/dec-F-ctx30000.jsonl" "$R/dec-$a-ctx30000.jsonl"; done
cmp_d2 "floor-OFF2-vs-OFF" "$C/dump-OFF-depth2.jsonl" "$R/dump-OFF2-depth2.jsonl"
cmp_d2 "OFF-vs-F"  "$R/dump-F-depth2.jsonl" "$C/dump-OFF-depth2.jsonl"
cmp_d2 "ON-vs-F"   "$R/dump-F-depth2.jsonl" "$C/dump-ON-depth2.jsonl"
cmp_d2 "OFF2-vs-F" "$R/dump-F-depth2.jsonl" "$R/dump-OFF2-depth2.jsonl"
cmp_d2 "S-vs-F"    "$R/dump-F-depth2.jsonl" "$R/dump-S-depth2.jsonl"
cmp_d2 "S-vs-ON"   "$C/dump-ON-depth2.jsonl" "$R/dump-S-depth2.jsonl"
grep -aE "BOOT OK|decode|depth2|error-lines" "$R/audit.log" | cut -c1-260 > "$R/sheet.txt"
finish DONE
