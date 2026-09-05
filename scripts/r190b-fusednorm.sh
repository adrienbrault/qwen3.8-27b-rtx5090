#!/usr/bin/env bash
# R190b (2026-09-05): rerun of R190 item C only. The first run (r190-microbench, 03:51 UTC) failed before touching the kernel:
# pcie_fused_norm_check.py built GemmaRMSNorm (a vLLM CustomOp) outside set_current_vllm_config() → AssertionError on 0.29.0rc2.
# The probe is fixed in patches-v0290/r190/pcie/ and mounted from staging (/stg/w27) so the ...-fusednorm image needs no rebuild.
# Gate unchanged: --atol 0 (bitwise) on 2 GPUs, saving_us per M. Chain: waits on r191; r188 (Marlin, rebuilt image) then r189 follow.
#   unit: sudo systemd-run --unit=r190b-fusednorm --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=600 \
#         -E GPU_QUEUE_NAME=r190b-fusednorm bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r191-bss-numerics; do sleep 30; done; exec bash /srv/qwen5090/r190b-fusednorm.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r190-microbench; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
STG=/srv/qwen5090/patches-v0290/r190-staging
IMG_FN=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-fusednorm
CD=/srv/qwen5090/cache
CACHE="-v $CD/triton:/root/.triton/cache -v $CD/inductor:/root/.cache/inductor -v $CD/flashinfer:/root/.cache/flashinfer"
sudo docker image inspect "$IMG_FN" >/dev/null 2>&1 || { log "ABORT: image $IMG_FN missing"; exit 3; }
[ -f "$STG/w27/pcie_fused_norm_check.py" ] || { log "ABORT: $STG/w27/pcie_fused_norm_check.py missing"; exit 3; }
grep -q set_current_vllm_config "$STG/w27/pcie_fused_norm_check.py" || { log "ABORT: staged probe lacks the R190b config-context fix"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-30}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval r190-bench; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R190b $1 ==="; }
trap 'log "### SIGTERM ###"; finish KILLED; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R190b fused-norm rerun start (lock held): probe from staging, image $IMG_FN ==="
teardown
timeout 1200 sudo docker run --rm --name r190-bench --runtime nvidia --gpus all --ipc=host --shm-size 8g \
  -e CUDA_MODULE_LOADING=LAZY -e PYTHONHASHSEED=0 -e MAX_JOBS=4 $CACHE -v "$STG":/stg:ro -v "$R":/out --entrypoint bash "$IMG_FN" \
  -c "cd /out && torchrun --standalone --nproc-per-node=2 /stg/w27/pcie_fused_norm_check.py --atol 0" > "$R/fused-norm-b.out" 2> "$R/fused-norm-b.err"; rc=$?
log "[fused-norm-b] rc=$rc out=$(wc -l < "$R/fused-norm-b.out") lines err=$(grep -aciE 'error|exception|traceback' "$R/fused-norm-b.err") err-lines; last: $(grep -a . "$R/fused-norm-b.out" | tail -1 | cut -c1-200)"
[ $rc -ne 0 ] && grep -aiE "error|exception|Traceback|assert" "$R/fused-norm-b.err" | grep -av "^\[rank[01]\]: *\^" | head -6 | cut -c1-220 | sed "s/^/[fused-norm-b err] /" | tee -a "$R/audit.log"
grep -aE "M=|rows|saving|mismatch|max_abs|PASS|FAIL" "$R/fused-norm-b.out" | head -24 | cut -c1-200 | sed "s/^/[fused-norm-b] /" | tee -a "$R/audit.log"
grep -aE "fused-norm|census\]|dispatch|gdn-spec|diag" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
