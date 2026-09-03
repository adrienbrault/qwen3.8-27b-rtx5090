#!/usr/bin/env bash
# 0.29 CANDIDATE daily (R168, 2026-09-03, user "focus on .29 and all related improvements, including nvfp4") — NOT
# promoted; the user flips by running this instead of launch-daily.sh. Same checkpoint/drafter/tier as the daily on the
# vLLM 0.29 chain (rc2 now, 0.29.0 when it ships: CAND_IMG=...): NVFP4 KV, DFlash2 ns9 draft_tp2 in CUDA graphs, 0131,
# 0134 (prefix-cache reuse under DFlash — without it rc1 gets 0 prefix hits and the tier never serves, R165 cell A) and
# 0135 (embed-table UVA offload, R167: +9.3% pool at zero decode/fidelity cost; OFFLOAD=0 disables).
# Shape = launch-daily-nvfp4-candidate.sh (the v0.28 candidate) with the rc1-era differences:
#   - draft spec WITHOUT kv_cache_dtype (rc1 regression #3: the draft dtype in the spec JSON breaks the boot, NOTES17)
#   - KV pool PINNED per SEQS like the v0.28 candidate (the util path sizes the pool before graph capture, Bug C), values
#     PROVISIONAL from R167 arm B (util 0.88 + offload: 971,797 tokens = 15.03 GB/GPU at 365 MiB free; the pinned path
#     skips the activation reserve, R166): 14.50 / 13.98 / 13.44 GB at SEQS 8 / 16 / 32 → target ≥ 650 MiB free.
#     r169 re-derives them on rc2; MIN_FREE_MIB fails the boot below 384 MiB (Bug C guard).
#   - Bug B dodge still in force and ASSERTED (XQA off + ALLOW_NO_XQA, MNBT 8192, FIWS 512 MiB) until r168 says otherwise.
# OPEN (r168-deep-decode.sh): the rc1 nvfp4 route decodes 30K-context prompts at 29 tok/s vs 144 on v0.28 (R167). This
# launcher is the shape; it is NOT daily-grade until that regression is understood — see FINDINGS R168.
# EXP=1 → :8029 / vllm-exp / eval-l2 (batteries); EXP=eval → :8030 / vllm-eval (campaign engine); default → :8020 daily.
# Rollback: bash /srv/qwen5090/launch-daily.sh (fp8 daily, v0.28, unchanged).
set -uo pipefail
EXP=${EXP:-0}; EXP_SEQS=${SEQS:-8}; KV_BYTES=${KV_BYTES:-}; MIN_FREE_MIB=${MIN_FREE_MIB:-384}; OFFLOAD=${OFFLOAD:-1}
IMG=${CAND_IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT EXTRA_ARGS \
        POOL_MIN POOL_MAX TP PP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE EAGER MMLIMIT MMKW CPUB \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR ALLOW_NO_XQA ALLOW_NO_PREWARM HEALTH_TRIES 2>/dev/null || true
fi
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
if [ "$EXP" = 1 ]; then PORT=8029; NAME=vllm-exp; BIND=127.0.0.1; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
elif [ "$EXP" = eval ]; then PORT=8030; NAME=vllm-eval; BIND=${EVAL_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
else PORT=8020; NAME=vllm-27b; BIND=0.0.0.0; L2=/srv/qwen5090/native-l2; SEQS=8; fi
if [ -z "$KV_BYTES" ]; then case "$SEQS" in
  8) KV_BYTES=14500000000;; 16) KV_BYTES=13980000000;; 32) KV_BYTES=13440000000;;
  *) echo "FAILED: no pinned KV budget for SEQS=$SEQS (set KV_BYTES)"; exit 1;; esac; fi
OFFLOAD_ARGS=""; [ "$OFFLOAD" = 1 ] && OFFLOAD_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (build: build-v0290rc2.sh)"; exit 1; }
env PORT=$PORT NAME=$NAME BIND_ADDR=$BIND MODEL_DIR="$MODEL" TP=2 L2MNT="$L2" \
    IMAGE="$IMG" KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=8192 SEQS=$SEQS UTIL=0.88 MAXLEN=262144 POOL_MIN=850000 POOL_MAX=1000000 \
    EXTRA_MOUNT="-v $DRAFT:/draft:ro" \
    EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES $OFFLOAD_ARGS" \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER"}' \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "0.29 CANDIDATE FAILED — engine NOT up$([ "$EXP" = 1 ] || echo '; run launch-daily.sh')"; exit 1; }
BOOTLOG=$(sudo docker logs "$NAME" 2>&1)
ARGS=$(sudo docker inspect "$NAME" --format '{{json .Args}} {{json .Config.Env}}')
fail(){ echo "FAILED: $1"; exit 1; }
VER=$(sudo docker exec "$NAME" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
case "$VER" in 0.29*) ;; *) fail "engine is vllm '$VER', not 0.29.x (image drift)";; esac
# G11 asserts (grep -c, not -q: pipefail + -q SIGPIPE gotcha)
[ "$(echo "$BOOTLOG" | grep -ac "as specified by kv_cache_memory_bytes")" -ge 1 ] || fail "pinned KV budget not honoured (no kv_cache_memory_bytes line)"
[ "$(echo "$BOOTLOG" | grep -ac "int workspace shrunk 8 MiB -> 1 MiB")" -ge 1 ] || fail "0131 pooled int workspace not active (image/env drift)"
[ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || fail "drafter graphs not captured (0129 inactive?)"
[ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || fail "drafter fell back to eager"
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || fail "XQA decode engaged — Bug B dodge not in force"
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || fail "checkpoint identity"
[ "$(echo "$ARGS" | grep -ac -- "--max-num-batched-tokens 8192")" -ge 1 ] || fail "MNBT is not 8192 (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912")" -ge 1 ] || fail "FlashInfer workspace is not 512 MiB (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_NVFP4_XQA=0")" -ge 1 ] || fail "VLLM_SM12X_NVFP4_XQA=0 missing"
if [ "$OFFLOAD" = 1 ]; then  # R167 / NOTES18 §5, fail closed
  [ "$(echo "$BOOTLOG" | grep -ac 'Offloader set to UVAOffloader')" -ge 1 ] || fail "UVAOffloader not selected (0135 inactive?)"
  [ "$(echo "$BOOTLOG" | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || fail "target embed shard not offloaded (no 'Total CPU offloaded parameters: 1.18')"
  [ "$(echo "$BOOTLOG" | grep -ac 'matched no parameters')" -eq 0 ] || fail "offload selector matched no parameters"
  [ "$(echo "$ARGS" | grep -ac 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1\|VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1')" -eq 0 ] || fail "UVA/pinning disabled by env"
else
  [ "$(echo "$BOOTLOG" | grep -ac 'CPU offloaded parameters')" -eq 0 ] || fail "OFFLOAD=0 but the engine offloaded parameters"
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$FREE" -ge "$MIN_FREE_MIB" ] || fail "only $FREE MiB free after pre-warm (< $MIN_FREE_MIB) — Bug C headroom missing; lower KV_BYTES"
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
echo "0.29 CANDIDATE UP on ${BIND}:${PORT} (vllm $VER, RedHat NVFP4 weights + NVFP4 KV pinned $KV_BYTES B/GPU + DFlash2 ns9 draft_tp2 in CUDA graphs + 0131/0134 + embed offload=$OFFLOAD + native disk tier, dual 5090, image $IMG, SEQS $SEQS). Pool $POOL, min free VRAM $FREE MiB.$([ "$EXP" = 1 ] || echo ' Rollback: launch-daily.sh')"
