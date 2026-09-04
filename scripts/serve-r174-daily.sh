#!/usr/bin/env bash
# Public copy of the private repo's flan/launch-daily-r174-ssm-fp32-0904.sh: the R174/R176 daily frozen before R182 (fp32 SSM state).
# ROLLBACK (frozen 2026-09-04 before the R182 SSM-bf16 promotion): the R174/R176 daily exactly — fp32 SSM state, 2,944-token
# blocks, pool 903,793 at SEQS 16. Tear the running daily down first; the native-l2 tier written by the bf16 daily has
# different hashes (block 1,584) so this launcher starts cold on the tier.
# DAILY (since 2026-09-04, R168 promotion, user "Promote, go"; sheet flan/r168-DECISION.md): the vLLM 0.29 nvfp4-KV route.
#   RedHatAI/Qwen3.8-27B-NVFP4 weights (unchanged since R156) + NVFP4 KV + DFlash2 ns9 draft_tp2 in CUDA graphs on the dual
#   5090s (TP2), syvai W4A16 drafter, native disk tier with 0137 LRU eviction, 16 GiB CPU tier, embed-table UVA offload (0135).
#   Image S = rc2 + patches-v0290 + FlashInfer 0.6.16.post3 swap (`vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616`, split_kv=0).
# Why (r168e/r169/r170/r172/r173, all vs the R156 bf16 rulers): pool 937,795 tokens pinned (fp8 daily 654,491 / 628,798,
#   +43%); disk tier actually SERVES 131K/220K prompts (4/4; the v0.28 daily never served above ~100K, 0/4 even at 16 GiB);
#   tier eviction holds a cap under floods and restarts (the fp8 daily's tier stranded at 100% on 2026-09-01); S image is the
#   closest of the three 0.29 attention paths to bf16 (dense top-1 92.797%, +0.753% PPL; fp8 daily 93.07% / +0.376%), and the
#   bf16 DECODE reference (r173c) puts it in the same 0.0051–0.0062 band as fp8 KV at 30K.
# Cost (r173/r173b, steady-state): code c8 ≈ −5% (1,134–1,146 vs 1,143–1,153), deep30k prose c1 ≈ −5% (147 vs 150–164),
#   short-request admission lower at c32 (nvfp4 blocks are 2,944 tokens). ns7 (+12.5% c8) was RETRACTED on the bf16 decode
#   ruler (median 2× ns9's at 30K) — do not flip NS on the daily without a new r173c-style ruler run.
# Pin: KV pool is PINNED (kv_cache_memory_bytes) because the util path sizes the pool before graph capture (Bug C). 14.5 GB at
#   SEQS 8 leaves 1,311–1,809 MiB free after pre-warm across boots (r173b); the guard below fails the boot under MIN_FREE_MIB.
# Bug B dodge ASSERTED (XQA off, MNBT 8192, FIWS 512 MiB): nvfp4 prefill above MNBT≈4,929 corrupts under XQA (R155).
# EXP=1 → :8029 / vllm-exp / eval-l2 (batteries); EXP=eval → :8030 / vllm-eval; default → :8020 daily. Experiments may pass
#   CAND_IMG / SPLIT_KV / OFFLOAD / KV_BYTES / SEQS / SPEC_NS / SPEC_DTP / TIER_* / CPUB; the daily port ignores the env. Since the
#   promotion, experiments default to the DAILY image (fi0616) — pass CAND_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs for the 0.6.18 rc2 image.
# Rollback: bash /srv/qwen5090/launch-daily-redhat-fp8-0902.sh  (v0.28 fp8 daily, frozen copy; tear this one down first)
set -uo pipefail
EXP=${EXP:-0}; EXP_SEQS=${SEQS:-8}; KV_BYTES=${KV_BYTES:-}; OFFLOAD=${OFFLOAD:-1}; SPLIT_KV=${SPLIT_KV:-0}
# Daily: pool band 870–950K around the deterministic 903,793 (SEQS 16 pin; 937,795 at SEQS 8) and a 512 MiB free floor. Experiments keep the candidate launcher's
# 850K–1M band (the SEQS 16/32 pins land at ~904K / ~869K) and its 384 MiB Bug C floor (the r17x boot_cand retry ladders expect it).
if [ "$EXP" = 0 ]; then MIN_FREE_MIB=${MIN_FREE_MIB:-512}; P_MIN=870000; P_MAX=950000; else MIN_FREE_MIB=${MIN_FREE_MIB:-384}; P_MIN=850000; P_MAX=1000000; fi
DAILY_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
# Tier knobs (0137): the daily boots with a 300 GB LRU cap on the 393 GB native-l2 fs, evict_scope root (stale namespaces from
# other configs are evicted too), 40 GB min-free (the launcher's own GC threshold). Experiments (EXP≠0) honour the env, unset = no cap.
if [ "$EXP" != 0 ] || [ "${DAILY_ALLOW_ENV:-0}" = 1 ]; then T_CAP=${TIER_CAP_GB:-}; T_SCOPE=${TIER_EVICT_SCOPE:-}; T_MINFREE=${TIER_MIN_FREE_GB:-}; CPU_B=${CPUB:-17179869184}; IMG=${CAND_IMG:-$DAILY_IMG}
else T_CAP=300; T_SCOPE=root; T_MINFREE=40; CPU_B=17179869184; IMG=$DAILY_IMG; fi
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT EXTRA_ARGS \
        POOL_MIN POOL_MAX TP PP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE EAGER MMLIMIT MMKW CPUB \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR ALLOW_NO_XQA ALLOW_NO_PREWARM HEALTH_TRIES \
        TIER_CAP_GB TIER_EVICT_SCOPE TIER_MIN_FREE_GB 2>/dev/null || true
fi
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
if [ "$EXP" = 1 ]; then PORT=8029; NAME=vllm-exp; BIND=127.0.0.1; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
elif [ "$EXP" = eval ]; then PORT=8030; NAME=vllm-eval; BIND=${EVAL_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
else PORT=8020; NAME=vllm-27b; BIND=0.0.0.0; L2=/srv/qwen5090/native-l2; SEQS=16; fi   # SEQS 16 since 2026-09-04 (R176, user): R159 c16 = +34% aggregate on the fp8 shape; pin 13.98 GB → pool 903,793 (−3.6% vs SEQS 8)   # BIND 0.0.0.0 deliberate: owui-proxy/harbor reach the engine via 172.17.0.1
if [ -z "$KV_BYTES" ]; then case "$SEQS" in
  8) KV_BYTES=14500000000;; 16) KV_BYTES=13980000000;; 32) KV_BYTES=13440000000;;
  *) echo "FAILED: no pinned KV budget for SEQS=$SEQS (set KV_BYTES)"; exit 1;; esac; fi
OFFLOAD_ARGS=""; [ "$OFFLOAD" = 1 ] && OFFLOAD_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
# 0136: FlashInfer 0.6.18 forces split-KV OFF for NVFP4 KV in the FA2 prefill wrapper (= the spec-verification path); SPLIT_KV=1
# re-enables it. The S image ships FlashInfer 0.6.16.post3 whose split path is on by default, so the daily runs SPLIT_KV=0 (r168e:
# closest to bf16 of the three paths; r173c: its decode dumps are identical to rc2 ON's at ctx0 and 30K).
SPLIT_ENV=""; [ "$SPLIT_KV" = 1 ] && SPLIT_ENV="-e VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1"
XARGS=""; XMOUNT=""; if [ "$EXP" != 0 ]; then XARGS="${EXTRA_ARGS_APPEND:-}"; XMOUNT="${EXTRA_MOUNT_APPEND:-}"; fi
NS_=9; DTP_=2; if [ "$EXP" != 0 ]; then NS_=${SPEC_NS:-9}; DTP_=${SPEC_DTP:-2}; fi
case "$NS_$DTP_" in *[!0-9]*) echo "FAILED: SPEC_NS/SPEC_DTP must be integers (got $NS_/$DTP_)"; exit 1;; esac
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (build: build-v0290rc2.sh + the fi0616 swap layer)"; exit 1; }
env PORT=$PORT NAME=$NAME BIND_ADDR=$BIND MODEL_DIR="$MODEL" TP=2 L2MNT="$L2" CPUB=$CPU_B ${T_CAP:+TIER_CAP_GB=$T_CAP} ${T_SCOPE:+TIER_EVICT_SCOPE=$T_SCOPE} ${T_MINFREE:+TIER_MIN_FREE_GB=$T_MINFREE} \
    IMAGE="$IMG" KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=8192 SEQS=$SEQS UTIL=0.88 MAXLEN=262144 POOL_MIN=$P_MIN POOL_MAX=$P_MAX \
    EXTRA_MOUNT="-v $DRAFT:/draft:ro $XMOUNT" \
    EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES $OFFLOAD_ARGS $XARGS" \
    SPEC_JSON='{"method":"dflash","model":"/draft","num_speculative_tokens":'$NS_',"draft_tensor_parallel_size":'$DTP_',"attention_backend":"FLASHINFER"}' \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1 $SPLIT_ENV" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "0.29 nvfp4 DAILY FAILED — engine NOT up$([ "$EXP" != 0 ] || echo '; rollback: launch-daily-redhat-fp8-0902.sh')"; exit 1; }
BOOTLOG=$(sudo docker logs "$NAME" 2>&1)
ARGS=$(sudo docker inspect "$NAME" --format '{{json .Args}} {{json .Config.Env}}')
fail(){ echo "FAILED: $1"; exit 1; }
VER=$(sudo docker exec "$NAME" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
FIVER=$(sudo docker exec "$NAME" python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)
case "$VER" in 0.29*) ;; *) fail "engine is vllm '$VER', not 0.29.x (image drift)";; esac
if [ "$IMG" = "$DAILY_IMG" ]; then case "$FIVER" in 0.6.16*) ;; *) fail "S image should carry FlashInfer 0.6.16.x, got '$FIVER' (image drift)";; esac; fi
# fail-closed asserts (grep -c, not -q: pipefail + -q SIGPIPE gotcha)
[ "$(echo "$BOOTLOG" | grep -ac "as specified by kv_cache_memory_bytes")" -ge 1 ] || fail "pinned KV budget not honoured (no kv_cache_memory_bytes line)"
[ "$(echo "$BOOTLOG" | grep -ac "int workspace shrunk 8 MiB -> 1 MiB")" -ge 1 ] || fail "0131 pooled int workspace not active (image/env drift)"
[ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || fail "drafter graphs not captured (0129 inactive?)"
[ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || fail "drafter fell back to eager"
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || fail "XQA decode engaged — Bug B dodge not in force"
[ "$(echo "$BOOTLOG" | grep -ac "redhat-nvfp4\|compressed-tensors")" -ge 1 ] || fail "checkpoint identity"
[ "$(echo "$ARGS" | grep -ac -- "--max-num-batched-tokens 8192")" -ge 1 ] || fail "MNBT is not 8192 (Bug B dodge)"
[ "$(echo "$ARGS" | grep -acE "num_speculative_tokens.{1,5}$NS_[,}]")" -ge 1 ] && [ "$(echo "$ARGS" | grep -acE "draft_tensor_parallel_size.{1,5}$DTP_[,}]")" -ge 1 ] || fail "speculative config is not ns$NS_ draft_tp$DTP_ on the container"
[ "$(echo "$ARGS" | grep -ac "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912")" -ge 1 ] || fail "FlashInfer workspace is not 512 MiB (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_NVFP4_XQA=0")" -ge 1 ] || fail "VLLM_SM12X_NVFP4_XQA=0 missing"
# CPU tier (r172): every disk-tier hit is promoted through the CPU tier, so 16 GiB (~1,010 blocks) is what lets 131K–262K prompts be served.
[ "$(echo "$ARGS" | grep -acE "cpu_bytes_to_use.{1,5}$CPU_B[,}]")" -ge 1 ] || fail "CPU tier is not $CPU_B bytes on the container"
CPUBLK=$(echo "$BOOTLOG" | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1 | grep -oE '[0-9]+')
[ "$EXP" != 0 ] || [ "${CPUBLK:-0}" -ge 900 ] || fail "CPU tier has ${CPUBLK:-?} blocks (expected ~1,010 at 16 GiB)"
if [ -n "$T_CAP" ]; then [ "$(echo "$ARGS" | grep -acE "max_capacity_gb.{1,5}$T_CAP[,}]")" -ge 1 ] || fail "0137 tier cap $T_CAP GB not on the container"; fi
if [ "$OFFLOAD" = 1 ]; then  # R167 / NOTES18 §5, fail closed
  [ "$(echo "$BOOTLOG" | grep -ac 'Offloader set to UVAOffloader')" -ge 1 ] || fail "UVAOffloader not selected (0135 inactive?)"
  [ "$(echo "$BOOTLOG" | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || fail "target embed shard not offloaded (no 'Total CPU offloaded parameters: 1.18')"
  [ "$(echo "$BOOTLOG" | grep -ac 'matched no parameters')" -eq 0 ] || fail "offload selector matched no parameters"
  [ "$(echo "$ARGS" | grep -ac 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1\|VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1')" -eq 0 ] || fail "UVA/pinning disabled by env"
else
  [ "$(echo "$BOOTLOG" | grep -ac 'CPU offloaded parameters')" -eq 0 ] || fail "OFFLOAD=0 but the engine offloaded parameters"
fi
if [ "$SPLIT_KV" = 1 ]; then [ "$(echo "$BOOTLOG" | grep -ac "re-enabled FlashInfer split-KV")" -ge 1 ] || fail "SPLIT_KV=1 but 0136 did not engage (image without 0136?)"
else [ "$(echo "$BOOTLOG" | grep -ac "re-enabled FlashInfer split-KV")" -eq 0 ] || fail "split-KV re-enabled although SPLIT_KV=0"; fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$FREE" -ge "$MIN_FREE_MIB" ] || fail "only $FREE MiB free after pre-warm (< $MIN_FREE_MIB) — Bug C headroom missing; lower KV_BYTES"
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
LABEL="0.29 nvfp4 DAILY UP"; [ "$EXP" = 0 ] || LABEL="0.29 nvfp4 EXP UP"
echo "$LABEL on ${BIND}:${PORT} (vllm $VER, FlashInfer $FIVER, RedHat NVFP4 weights + NVFP4 KV pinned $KV_BYTES B/GPU + DFlash2 ns$NS_ draft_tp$DTP_ in CUDA graphs + 0131/0134 + embed offload=$OFFLOAD + split_kv=$SPLIT_KV + native disk tier${T_CAP:+ cap ${T_CAP} GB} + CPU tier $CPU_B B, dual 5090, image $IMG, SEQS $SEQS). Pool $POOL, min free VRAM $FREE MiB.$([ "$EXP" != 0 ] || echo ' Rollback: launch-daily-redhat-fp8-0902.sh')"
