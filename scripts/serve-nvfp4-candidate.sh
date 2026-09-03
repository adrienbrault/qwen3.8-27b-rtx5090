#!/usr/bin/env bash
# CANDIDATE daily (R158 shape, R166 sizing; 2026-09-03) — NOT promoted; the user flips by running this instead
# of launch-daily.sh. Same checkpoint/drafter/tier as the daily; the KV cache becomes NVFP4 and the DFlash2 drafter
# runs in CUDA graphs (0129) on the revival image chain (0116-0119 + 0129 + 0131).
#
# KV pool = PINNED BYTES, not util (R166, Bug C fix at the operator level): vLLM v0.28 sizes the KV pool from
#   gpu_memory_utilization BEFORE CUDA-graph capture and its pre-capture graph estimate is zero, so on this route the
#   graphs (1.20 GiB @SEQS 16 / ~1.7 @32 with 0131; 2.04/OOM without) must fit in the util headroom — at util 0.90 the
#   candidate booted with 3 MiB free (R164c) and could not boot at SEQS 32. `--kv-cache-memory-bytes` skips that
#   profiling and takes the pool size verbatim (gpu_worker.py:489). Measured basis (R164c, util 0.90): 14.18 GiB
#   available → 984,411 tokens = 15,466 B/token/GPU. Pin 12.85 GiB at SEQS 8 (table below) → pool ≈ 892K tokens
#   (+36% over the fp8 daily's 657K) with ≈650 MiB free after pre-warm (the fp8 daily runs with ≈370).
#   MIN_FREE_MIB below fails the boot if that headroom is not there (the Bug C guard).
#   The pool is deterministic by construction (no bimodal sizing).
# Shape constraints (R155 Bug B dodge, still required, ASSERTED below — G11): XQA OFF (VLLM_SM12X_NVFP4_XQA=0 +
#   ALLOW_NO_XQA=1), --max-num-batched-tokens 8192, 512 MiB FlashInfer workspace, 0131 pooled int workspace shrunk.
# EXP=1: experiment mode for the gate battery (r166-candidate-gates.sh) — :8029 / vllm-exp / eval-l2 tier, SEQS
#   overridable; everything else identical, so what the battery measures is this launcher, asserts included.
# Rollback: bash /srv/qwen5090/launch-daily.sh (fp8 daily, unchanged).
set -uo pipefail
EXP=${EXP:-0}; EXP_SEQS=${SEQS:-8}; KV_BYTES=${KV_BYTES:-}; MIN_FREE_MIB=${MIN_FREE_MIB:-384}
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT EXTRA_ARGS \
        POOL_MIN POOL_MAX TP PP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE EAGER MMLIMIT MMKW CPUB \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR ALLOW_NO_XQA ALLOW_NO_PREWARM HEALTH_TRIES 2>/dev/null || true
fi
IMG=vllm-qwen38:v0280-nvfp4kv-revival-graphs-ws
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
if [ "$EXP" = 1 ]; then PORT=8029; NAME=vllm-exp; BIND=127.0.0.1; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
elif [ "$EXP" = eval ]; then PORT=8030; NAME=vllm-eval; BIND=${EVAL_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS   # campaign engine (boot-candidate-miniswe.sh)
else PORT=8020; NAME=vllm-27b; BIND=0.0.0.0; L2=/srv/qwen5090/native-l2; SEQS=8; fi
# Pin per SEQS (R166 boot 1: 13.41 GiB at SEQS 8 → pool 931,214, graphs 0.72 GiB, only 101 MiB free after pre-warm —
# the pinned path skips the profiler's activation-peak reserve too, so the pin must absorb graphs AND pre-warm peak).
# 12.85 GiB at SEQS 8 targets ≈650 MiB free; 16/32 subtract R164c's graph increments (+0.48 / +0.50 GiB).
if [ -z "$KV_BYTES" ]; then case "$SEQS" in
  8) KV_BYTES=13800000000;; 16) KV_BYTES=13280000000;; 32) KV_BYTES=12740000000;;
  *) echo "FAILED: no pinned KV budget for SEQS=$SEQS (set KV_BYTES)"; exit 1;; esac; fi
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (build: dflash-nvfp4-revival/Dockerfile.ws)"; exit 1; }
env PORT=$PORT NAME=$NAME BIND_ADDR=$BIND MODEL_DIR="$MODEL" TP=2 L2MNT="$L2" \
    IMAGE="$IMG" KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=8192 SEQS=$SEQS UTIL=0.90 MAXLEN=262144 POOL_MIN=800000 POOL_MAX=920000 \
    EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
    EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES" \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}' \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "CANDIDATE FAILED — engine NOT up$([ "$EXP" = 1 ] || echo '; run launch-daily.sh')"; exit 1; }
BOOTLOG=$(sudo docker logs "$NAME" 2>&1)
ARGS=$(sudo docker inspect "$NAME" --format '{{json .Args}} {{json .Config.Env}}')
fail(){ echo "FAILED: $1"; exit 1; }
# G11 asserts: every dodge/fix must positively identify itself (grep -c, not -q: pipefail + -q SIGPIPE gotcha)
[ "$(echo "$BOOTLOG" | grep -ac "as specified by kv_cache_memory_bytes")" -ge 1 ] || fail "pinned KV budget not honoured (no kv_cache_memory_bytes line)"
[ "$(echo "$BOOTLOG" | grep -ac "int workspace shrunk 8 MiB -> 1 MiB")" -ge 1 ] || fail "0131 pooled int workspace not active (image/env drift)"
[ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || fail "drafter graphs not captured (0129 inactive?)"
[ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || fail "drafter fell back to eager"
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || fail "XQA decode engaged — Bug B dodge not in force"
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || fail "checkpoint identity"
[ "$(echo "$ARGS" | grep -ac -- "--max-num-batched-tokens 8192")" -ge 1 ] || fail "MNBT is not 8192 (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912")" -ge 1 ] || fail "FlashInfer workspace is not 512 MiB (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_NVFP4_XQA=0")" -ge 1 ] || fail "VLLM_SM12X_NVFP4_XQA=0 missing"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$FREE" -ge "$MIN_FREE_MIB" ] || fail "only $FREE MiB free after pre-warm (< $MIN_FREE_MIB) — Bug C headroom missing; lower KV_BYTES"
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
echo "CANDIDATE UP on ${BIND}:${PORT} (RedHat NVFP4 weights + NVFP4 KV pinned $KV_BYTES B/GPU + DFlash2 ns9 draft_tp2 in CUDA graphs + 0131 + native disk tier, dual 5090, image $IMG, SEQS $SEQS). Pool $POOL, min free VRAM $FREE MiB.$([ "$EXP" = 1 ] || echo ' Rollback: launch-daily.sh')"
