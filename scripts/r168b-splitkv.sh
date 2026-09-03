#!/usr/bin/env bash
# R168b (2026-09-03): validate 0136 on the rc2 chain — codex NOTES19 found the rc1 nvfp4 deep-context decode collapse in
# FlashInfer 0.6.18's `_nvfp4_kv_requires_disabled_split_kv` (forces split-KV OFF for NVFP4 KV in the FA2 prefill wrapper;
# spec verification is q=10 → prefill path → one CTA walks the 30K KV per step). 0136 = opt-in VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1
# that restores the 0.6.16 behaviour. FlashInfer added the guard after seeing short-Q/long-KV corruption with split-KV (the Bug B
# family — v0.28 ran split-KV ON with our dodge and passed every gate), so speed alone does not clear it:
#   OFF = launch-daily-v0290-candidate.sh EXP=1 (SPLIT_KV=0): deep30k + short + profile   (expected ≈ rc1's 29 tok/s)
#   ON  = same with SPLIT_KV=1: deep30k + short + profile, THEN the correctness set — needles 9K/131K/220K, fidelity ruler vs
#         the FP8 reference (and direct vs OFF), decode_ss code c1/c8, error lines. Pass = ON ≈ v0.28's 144 tok/s at 30K AND
#         needles all hit AND fidelity within the v0.28 candidate's band (0.29 pp top-1, R166).
# Queue-registered; restores the daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r168b-splitkv --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r168b-splitkv bash /srv/qwen5090/r168b-splitkv.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r168b-splitkv; mkdir -p "$R/prof"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=${CAND_IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}
sudo docker run --rm --entrypoint grep "$IMG" -q _maybe_enable_sm12x_nvfp4_prefill_split_kv /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py 2>/dev/null \
  || sudo docker run --rm --entrypoint python3 "$IMG" -c 'import vllm,os,sys; sys.exit(0 if "_maybe_enable_sm12x_nvfp4_prefill_split_kv" in open(os.path.join(os.path.dirname(vllm.__file__),"v1/attention/backends/flashinfer.py")).read() else 1)' \
  || { log "ABORT: $IMG has no 0136"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R168b split-KV validation start (lock held): $IMG ==="
U=http://127.0.0.1:8029
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-45}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R168b split-KV validation $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
PROF="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/prof"
boot(){ # $1 tag, $2 SPLIT_KV
  local tag=$1 kv rc
  mkdir -p "$R/prof/$tag"
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 SPLIT_KV=$2 EXTRA_ARGS_APPEND="$PROF" EXTRA_MOUNT_APPEND="-v $R/prof/$tag:/prof" ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-$tag.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[$tag] BOOT OK ${kv:+(retry KV_BYTES=$kv) }$(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-$tag.log" | tail -1) split_kv=$2 $(sudo docker logs vllm-exp 2>&1 | grep -a 're-enabled FlashInfer split-KV' | head -1 | sed 's/.*WARNING[^]]*] //' | cut -c1-80)"; return 0; fi
    log "[$tag] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-$tag.log" || { teardown; return 1; }
    teardown
  done; return 1; }
dss(){ local tag=$1 lab=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$lab.json" > "$R/decode-$tag-$lab.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$lab.out" | sed "s/^/[$tag $lab] /" | cut -c1-230 | tee -a "$R/audit.log"; }
profile30k(){ local tag=$1
  curl -sf -m 30 -X POST $U/start_profile >/dev/null || { log "[$tag] start_profile FAILED"; return 1; }
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --kind prose --ctx 30000 --tokens 32 --runs 1 --out "$R/decode-$tag-prof.json" > "$R/decode-$tag-prof.out" 2>&1
  curl -sf -m 600 -X POST $U/stop_profile >/dev/null || log "[$tag] stop_profile FAILED"
  sleep 20; sudo chown -R "$USER" "$R/prof/$tag" 2>/dev/null
  local n; n=$(ls "$R/prof/$tag" 2>/dev/null | wc -l); log "[$tag] profile: $n trace file(s) ($(du -sh "$R/prof/$tag" 2>/dev/null | cut -f1))"
  [ "$n" -gt 0 ] && python3 /srv/qwen5090/probes/prof_summary.py "$R/prof/$tag" --top 25 --steps 32 --json "$R/prof-$tag.json" > "$R/prof-$tag.txt" 2>&1 \
    && { grep -aE "^== " "$R/prof-$tag.txt" | sed "s/^/[$tag prof] /" | cut -c1-230 | tee -a "$R/audit.log"; sed -n '/ALL RANKS merged/,$p' "$R/prof-$tag.txt" | head -14 | tail -12 | sed "s/^/[$tag prof top] /" | cut -c1-200 | tee -a "$R/audit.log"; }; }
needles(){ python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --warm --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles $2] $(grep -a SUMMARY "$R/needles-$1.out" | cut -c1-120)"; }
fidelity(){ python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-$1.jsonl" --conc 1 > "$R/fidelity-run-$1.log" 2>&1 || log "[$1] ruler FAILED rc=$?"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$R/run-$1.jsonl" 2>&1 | tee "$R/fidelity-$1-vs-fp8ref.txt" | cut -c1-200 | sed "s/^/[$1 fidelity vs fp8ref] /" | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
speed(){ local T=$1
  dss $T deep30k --conc 1 --kind prose --ctx 30000 --tokens 512 --runs 2
  dss $T short --conc 1 --kind prose --tokens 256 --runs 2
  profile30k $T; errlines $T; }

log "taking the daily down"; teardown
sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 wiped"
if boot OFF 0; then speed OFF; fidelity OFF; fi; teardown
if boot ON 1; then speed ON
  needles ON "9000 131000 220000"
  fidelity ON
  [ -f "$R/run-OFF.jsonl" ] && python3 /srv/qwen5090/probes/fidelity.py compare --ref "$R/run-OFF.jsonl" "$R/run-ON.jsonl" 2>&1 | tee "$R/fidelity-ON-vs-OFF.txt" | cut -c1-200 | sed "s/^/[ON fidelity vs OFF (direct)] /" | tee -a "$R/audit.log"
  dss ON code-c1 --conc 1 --kind code --tokens 512 --runs 3; dss ON code-c8 --conc 8 --kind code --tokens 512 --runs 3
  errlines "ON final"
fi; teardown
log "SHEET:"; grep -aE "BOOT OK|deep30k\] RESULT|short\] RESULT|code-c[18]\] RESULT|needles|fidelity|error-lines|prof\]" "$R/audit.log" | cut -c1-200 > "$R/sheet.txt"; wc -l "$R/sheet.txt" | tee -a "$R/audit.log"
finish DONE
