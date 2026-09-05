#!/usr/bin/env bash
# R190 (2026-09-05, user: "'the larger items are kernel or training work.' Have codex astra agent write these!"): first GPU numbers for
# the codex (gpt-6-astra) deliverables in patches-v0290/ (briefs BRIEF23..28, notes NOTES23..28, diffs 0140..0145). Everything here is
# offline-verified only so far (scratch apply at fuzz 0, py_compile, real apply in the served image); this unit is the first execution.
#   C. 0144 pcie_ipc fused residual+RMSNorm: torchrun 2-GPU probe in image ...-fusednorm — bitwise gate (atol 0) + saving_us per M.
#   A. 0140 GEMM census: every NVFP4/FP8 kernel class at the per-rank served shapes, M 1..8192, graph replay medians (image = daily
#      pcieipc, GPU 0, no engine) → census-graph.json/.md; make_dispatch_table.py turns >5% winners into dispatch.json (else empty).
#   B. 0145 GDN spec-update microbench: the ns9 kernel (fused_sigmoid_gating_delta_rule_update) baseline vs tiled/split configs at
#      N=1/8/16 × T=10, correctness vs baseline + fp32 reference, graph-replay µs (image ...-gdnspec, GPU 0).
#   D. 0142 sync census + 0143 collective tags: boot the ...-r190diag image on :8029 (daily flags, PCIE_IPC=1 to match the daily) with
#      both knobs on, drive a c1 code decode so the census window (20 warm-up + 50 steps) completes, and save the per-call-site table.
# Reads: NOTES23/24b/25/27 falsification criteria. Not a promotion of anything; results feed FINDINGS + the next briefs.
# Chain: waits for r188-marlin; r189-promote-pcieipc is re-issued to wait for this unit, so r189 remains the chain's daily restore.
#   unit: sudo systemd-run --unit=r190-microbench --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r190-microbench bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r188-marlin; do sleep 30; done; exec bash /srv/qwen5090/r190-microbench.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r190-microbench; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
STG=/srv/qwen5090/patches-v0290/r190-staging; PR=/srv/qwen5090/probes; CAND=/srv/qwen5090/launch-daily.sh; U=http://127.0.0.1:8029
BASE=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
IMG_FN=$BASE-fusednorm; IMG_GS=$BASE-gdnspec; IMG_DG=$BASE-r190diag
CD=/srv/qwen5090/cache
CACHE="-v $CD/triton:/root/.triton/cache -v $CD/inductor:/root/.cache/inductor -v $CD/flashinfer:/root/.cache/flashinfer"
for i in $BASE $IMG_FN $IMG_GS $IMG_DG; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
for f in w23/nvfp4_gemm_census.py w23/make_dispatch_table.py w24/gdn_spec_microbench.py; do [ -f "$STG/$f" ] || { log "ABORT: $STG/$f missing"; exit 3; }; done
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-30}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval r190-bench; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R190 $1 ==="; }
trap 'log "### SIGTERM ###"; finish KILLED; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R190 microbench start (lock held) ==="
teardown
# bench TAG IMAGE GPUS TIMEOUT -- cmd...   (one-shot container, results dir mounted at /out, staging at /stg)
bench(){ local tag=$1 img=$2 gpus=$3 to=$4; shift 4
  timeout "$to" sudo docker run --rm --name r190-bench --runtime nvidia --gpus "$gpus" --ipc=host --shm-size 8g \
    -e CUDA_MODULE_LOADING=LAZY -e PYTHONHASHSEED=0 -e MAX_JOBS=4 $CACHE -v "$STG":/stg:ro -v "$R":/out --entrypoint bash "$img" \
    -c "cd /out && $*" > "$R/$tag.out" 2> "$R/$tag.err"; local rc=$?
  log "[$tag] rc=$rc out=$(wc -l < "$R/$tag.out") lines err=$(grep -aciE 'error|exception|traceback' "$R/$tag.err") err-lines; last: $(grep -a . "$R/$tag.out" | tail -1 | cut -c1-200)"
  [ $rc -ne 0 ] && grep -aiE "error|exception|Traceback|assert" "$R/$tag.err" | head -4 | cut -c1-200 | sed "s/^/[$tag err] /" | tee -a "$R/audit.log"
  settle 10; return $rc; }
# ---- C. 0144 fused norm (2 GPUs) ----
bench fused-norm "$IMG_FN" all 1200 torchrun --standalone --nproc-per-node=2 /opt/pcie_fused_norm_check.py --atol 0
grep -aE "M=|rows|saving|mismatch|max_abs|PASS|FAIL" "$R/fused-norm.out" | head -24 | cut -c1-200 | sed "s/^/[fused-norm] /" | tee -a "$R/audit.log"
# ---- A. 0140 GEMM census (GPU 0, daily image) ----
bench census "$BASE" device=0 3000 python3 /stg/w23/nvfp4_gemm_census.py --json /out/census-graph.json --mode graph
cp "$R/census.out" "$R/census-graph.md" 2>/dev/null
grep -aE "^\| *(gate_up|down|qkv|o_proj|gdn|lm_head|in_proj|out_proj)" "$R/census-graph.md" | awk -F'|' '$0 ~ /\| *(10|80|160) *\|/' | head -40 | cut -c1-200 | sed "s/^/[census] /" | tee -a "$R/audit.log"
if [ -s "$R/census-graph.json" ]; then
  sudo docker run --rm -v "$STG":/stg:ro -v "$R":/out --entrypoint python3 "$BASE" /stg/w23/make_dispatch_table.py /out/census-graph.json > "$R/dispatch.json" 2> "$R/dispatch.stderr"
  log "[dispatch] rc=$? rules=$(python3 -c "import json;print(len(json.load(open('$R/dispatch.json'))['rules']))" 2>/dev/null || echo ?) $(grep -a . "$R/dispatch.stderr" | head -6 | tr '\n' ';' | cut -c1-400)"
else log "[dispatch] no census json"; fi
# ---- B. 0145 GDN spec microbench (GPU 0) ----
bench gdn-spec "$IMG_GS" device=0 1800 python3 /stg/w24/gdn_spec_microbench.py --device 0
grep -aE "baseline|variant|bv=|PASS|FAIL|mismatch|us|µs" "$R/gdn-spec.out" | head -40 | cut -c1-200 | sed "s/^/[gdn-spec] /" | tee -a "$R/audit.log"
# ---- D. diag engine: 0142 sync census + 0143 collective tags ----
booted=0
for kv in 13980000000 13500000000; do
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 CAND_IMG=$IMG_DG \
    EXTRA_ENV_APPEND="-e VLLM_SM12X_SYNC_CENSUS=1 -e VLLM_SM12X_COLLECTIVE_TAGS=1" bash $CAND > "$R/boot-diag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then booted=1; log "[diag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-diag-$kv.log" | tail -1 | tr -dc 0-9)"; break; fi
  log "[diag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-diag-$kv.log" | tail -1 | cut -c1-220)"
  sudo docker logs vllm-exp 2>&1 | grep -aiE "error|exception|census|COLLECTIVE_TAGS" | head -6 | cut -c1-200 | sed "s/^/[diag boot-err] /" | tee -a "$R/audit.log"; teardown
done
if [ $booted = 1 ]; then
  sudo docker logs vllm-exp > "$R/engine-boot-diag.log" 2>&1
  log "[diag proof] census=$(grep -ac 'SM12X sync census active' "$R/engine-boot-diag.log") tags=$(grep -ac 'VLLM_SM12X_COLLECTIVE_TAGS=1 active' "$R/engine-boot-diag.log") pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-diag.log") $(grep -a 'sync census active' "$R/engine-boot-diag.log" | head -1 | sed -E 's/^.*(SM12X sync)/\1/' | cut -c1-120)"
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 1024 --runs 2 --kind code --out "$R/decode-diag-c1.jsonl" > "$R/probe-diag-c1.out" 2> "$R/probe-diag-c1.err"
  grep -a RESULT "$R/probe-diag-c1.out" | sed "s/^/[diag c1] /" | cut -c1-260 | tee -a "$R/audit.log"
  sleep 5; sudo docker logs vllm-exp > "$R/engine-diag-full.log" 2>&1
  awk '/SM12X sync census complete/{p=1} p{print} /^[^ ]/ && p && ++n>400{exit}' "$R/engine-diag-full.log" > "$R/sync-census-table.txt"
  log "[diag census] table lines=$(wc -l < "$R/sync-census-table.txt"); $(grep -a 'sync census complete' "$R/engine-diag-full.log" | head -1 | cut -c1-160)"
  grep -aE "^ *[0-9]+ +(target|sample\+draft|output) " "$R/sync-census-table.txt" | awk '{c[$2" "$4" "$5]+=$3} END{for(k in c) print c[k], k}' | sort -rn | head -20 | sed "s/^/[diag census top] /" | tee -a "$R/audit.log"
  log "[diag errors] $(grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError' "$R/engine-diag-full.log")"
fi
grep -aE "fused-norm|census\]|dispatch|gdn-spec|diag" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
