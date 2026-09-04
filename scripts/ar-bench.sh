#!/usr/bin/env bash
# R184 (2026-09-04): all-reduce microbenchmark on the two 5090s — NCCL vs vLLM's custom all-reduce (the daily's decode
# path) vs flashinfer main's pcie_ipc all-reduce (flashinfer PR #4393, merged 2026-08-20, not in any 0.6.x release).
# Runs in a scratch container of the FI-0.6.18 rc2 image with the flashinfer main checkout mounted over the package
# (PYTHONPATH), so the pcie_ipc kernel JITs from source; nothing on the host or in the daily image changes.
# Needs both GPUs for ~2 min; runs alongside a booting engine (see the runner below), never during a measurement.
set -euo pipefail
IMG=${IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}      # flashinfer 0.6.18 + vllm 0.29.0rc2 (+ nvcc)
FI=${FI:-/srv/qwen5090/src/flashinfer-main}
R=${R:-/srv/qwen5090/results/2026-09-04-r184-arbench}
CACHE=/srv/qwen5090/cache/flashinfer-main
MODE=${1:-bench}     # jit | bench
ROWS=${ROWS:-1,8,10,16,20,40,80,160,320,2048,8192}
TUNE=${TUNE:-pcie-tune.json}   # pcie_ipc autotune cache under $R; tune on a FREE card (a co-resident engine skews it)
mkdir -p "$R" "$CACHE"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }

# flashinfer's JIT resolves its sources under <package>/data/{csrc,include,cutlass,spdlog,cccl}; a wheel ships them,
# a checkout does not. Link them once.
D=$FI/flashinfer/data; mkdir -p "$D"
for pair in csrc:../../csrc include:../../include cutlass:../../3rdparty/cutlass spdlog:../../3rdparty/spdlog cccl:../../3rdparty/cccl; do
  n=${pair%%:*}; t=${pair#*:}; [ -e "$D/$n" ] || ln -s "$t" "$D/$n"
done

EXTRA=""; [ "$MODE" = jit ] && EXTRA="--jit-only --backends pcie"
log "R184 ar-bench mode=$MODE rows=$ROWS tune=$TUNE image=$IMG fi=$(git -C $FI log --oneline -1)"
docker run --rm --gpus all --ipc=host --shm-size 8g --name ar-bench \
  -v "$FI":/fi -v /srv/qwen5090/probes:/probes:ro -v "$CACHE":/ficache -v "$R":/out \
  -e PYTHONPATH=/fi -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/ficache \
  -e NCCL_P2P_LEVEL=SYS -e VLLM_LOGGING_LEVEL=WARNING -e MAX_JOBS=8 \
  --entrypoint torchrun "$IMG" --nproc_per_node=2 --standalone /probes/ar_bench.py \
    --hidden 5120 --rows "$ROWS" --json /out/arbench-$MODE-$(date +%H%M%S).json --tune-cache /out/$TUNE $EXTRA \
  2>&1 | grep -av "Warning\|warn(" | tee -a "$R/run-$MODE.log" | tail -40
log "R184 ar-bench mode=$MODE done rc=${PIPESTATUS[0]}"
