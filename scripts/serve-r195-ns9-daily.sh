#!/usr/bin/env bash
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
#   promotion, experiments default to the DAILY image (pcieipc-bsshash since R195; knobs off unless PCIE_IPC=1 / BSS=1) — pass CAND_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs for the 0.6.18 rc2 image.
# R182 (2026-09-04, user "Ok promote"): SSM state cached in bf16 (`--mamba-ssm-cache-dtype bfloat16`). The hybrid allocator stores a GDN
#   state snapshot per block (mamba_cache_mode=align) and sizes the attention block to that page, so the fp32 state made a request cost
#   6.4% of the pool at admission + ~0.2%/1K tokens (R178: 15 short or four 100K requests fill 903K "tokens"). bf16 halves the page:
#   block 2,944 → 1,584, pool 1,020,596 at the same 13.98 GB pin (R179), fixed cost 3.5%, 100K prompt 16.5%, five 100K co-resident
#   (R180); rulers vs bf16 neutral (dense 92.771%/+0.744% vs 92.716%/+0.770%), 80-chunk decode ruler at 30K 0.00576 vs 0.00443
#   median |Δlogprob|, 31/31 per chunk, tails equal (R181). Block change = every tier hash changes; native-l2 wiped at promotion.
# R185 (2026-09-05, user "seems like no brainer? Lets use?!"): FlashInfer main's pcie_ipc all-reduce (patch 0138 + package pcie_ipc_ar21;
#   image `...-fi0616-pcieipc` = the S image + one 359 KB layer, Dockerfile.pcieipc) behind VLLM_SM12X_PCIE_IPC_AR=1, ASSERTED at boot
#   ("PCIe IPC all-reduce enabled" + backend order PCIE_IPC, CUSTOM, PYNCCL). R185/R185b (results/2026-09-05-r185-pcieipc, knob on vs
#   off on the same image, same night): code c1 +4.6%, prose c1 +5.4%, prose 30K +3.1%, c16 +2.6%, c8 flat in tok/s = +4.9% steps/s;
#   numerics identical on every paired ruler (decode ctx0/30K, agentic). Experiments: PCIE_IPC=1 opts in (default OFF so batteries stay
#   comparable with the R183 band); the daily port forces it on. Gates on the live daily: r189-promote-pcieipc.sh.
# Rollback: bash /srv/qwen5090/launch-daily-r182-nopcie-0905.sh (fi0616 image, no 0138, frozen pre-R185 launcher; tear this one down
#   first); older: launch-daily-r174-ssm-fp32-0904.sh (fp32 SSM), launch-daily-redhat-fp8-0902.sh (v0.28 fp8 daily)
# R195 (2026-09-05, user "yes" on the R193e sheet): batch-sharded sampling (`--enable-batch-sharded-sampling`: each TP rank samples its
#   half of the batch from local logits, an all-to-all replaces the 19.9 MB per-step logits all-gather). Image `...-pcieipc-bsshash` =
#   the pcieipc image + patch 0147 (the flag removed from ParallelConfig.compute_hash, so ON/OFF share one AOT compile artifact; marker
#   /opt/prs-markers/0147 asserted). R193e (results/2026-09-05-r193e-pin-bss; one artifact + VLLM_TRITON_FORCE_FIRST_CONFIG=1 on both arms):
#   ON vs OFF bitwise 20/20 median 0 at ctx0 and 30K; steps/s +2.6% c8, +4.5% c16, +1.5% c1 (R193b/R193c same size). Seeded T>0 requests
#   draw a different sample stream than the unsharded sampler (R187). Experiments: BSS=1 (or the flag in EXTRA_ARGS_APPEND) opts in;
#   the daily port forces it on and asserts the "Batch-sharded sampling enabled" line. Gates on the live daily: r195-promote-bss.sh.
#   NOTE the daily's compile artifact changed with the image (0147 changes the hash string): cd95c505/d5e217de/fd65aadf (r193b OFF-h),
#   another compile-lottery draw (R193) — its bf16 ruler position is read by r195.
# Rollback: bash /srv/qwen5090/launch-daily-r189-nobss-0905.sh (pcieipc image, no 0147, sharded sampling off; frozen pre-R195 launcher;
#   same block size so the native-l2 tier is shared; tear this one down first).
set -uo pipefail
EXP=${EXP:-0}; EXP_SEQS=${SEQS:-8}; KV_BYTES=${KV_BYTES:-}; OFFLOAD=${OFFLOAD:-1}; SPLIT_KV=${SPLIT_KV:-0}; SSM_DTYPE=${SSM_DTYPE:-bfloat16}  # R182; experiments may pass SSM_DTYPE=float32 (NOT in the unset list below: R183 found every EXP boot dying on "SSM_DTYPE: unbound"; the daily branch forces bfloat16)
# Daily: pool band 990K–1.05M around the deterministic 1,020,596 (SEQS 16 pin, bf16 SSM; fp32 SSM gave 903,793) and a 512 MiB free floor. Experiments
# keep an 850K–1.1M band (fp32-SSM pins land at ~904K / ~869K, bf16-SSM at ~986K–1.02M) and the 384 MiB Bug C floor (the r17x boot_cand retry ladders expect it).
if [ "$EXP" = 0 ]; then MIN_FREE_MIB=${MIN_FREE_MIB:-512}; P_MIN=990000; P_MAX=1050000; else MIN_FREE_MIB=${MIN_FREE_MIB:-384}; P_MIN=850000; P_MAX=1100000; fi
# R197: the pool follows the speculative configuration (each slot holds a GDN state snapshot; the MTP head needs no drafter weights): MTP ns3
# booted at 1,309,368 and the 1.1M ceiling rejected a healthy engine. EXP arms with another spec method/length get a wide band and log the pool.
if [ "$EXP" != 0 ] && { [ "${SPEC_METHOD:-dflash}" != dflash ] || [ "${SPEC_NS:-9}" != 9 ]; }; then P_MIN=${POOL_MIN:-600000}; P_MAX=${POOL_MAX:-1500000}; fi
DAILY_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash   # R195: S image + patch 0138 (Dockerfile.pcieipc) + patch 0147 (Dockerfile.bss-not-a-compile-factor)
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
# R196 EXP-only passthrough: CAND_MODEL=<dir> audits another checkpoint on the daily route (same KV dtype, drafter, knobs); the daily port ignores it.
if [ "$EXP" != 0 ] && [ -n "${CAND_MODEL:-}" ]; then MODEL=$CAND_MODEL; fi
if [ "$EXP" = 1 ]; then PORT=8029; NAME=vllm-exp; BIND=127.0.0.1; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
elif [ "$EXP" = eval ]; then PORT=8030; NAME=vllm-eval; BIND=${EVAL_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
else PORT=8020; NAME=vllm-27b; BIND=0.0.0.0; L2=/srv/qwen5090/native-l2; SEQS=16; SSM_DTYPE=bfloat16; fi   # SEQS 16 since 2026-09-04 (R176, user): R159 c16 = +34% aggregate on the fp8 shape; pin 13.98 GB → pool 903,793 (−3.6% vs SEQS 8)   # BIND 0.0.0.0 deliberate: owui-proxy/harbor reach the engine via 172.17.0.1
# R197 rollback copy (2026-09-05): frozen ns9 launcher + the same native-l2 block stamp as the ns7 launcher, at ITS block (1,584), so a rollback
# wipes the 1,552-token content and a later return to ns7 wipes the 1,584-token content. Only addition to the frozen copy.
DAILY_BLOCK=1584
if [ "$EXP" = 0 ]; then ST_=$(cat "$L2/.block" 2>/dev/null || true); if [ "$ST_" != "$DAILY_BLOCK" ]; then
  echo "native-l2 block stamp '${ST_:-none}' != $DAILY_BLOCK: wiping the tier's _model_* content (was $(du -sh "$L2" 2>/dev/null | cut -f1))"
  sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; fi; fi
if [ -z "$KV_BYTES" ]; then case "$SEQS" in
  8) KV_BYTES=14500000000;; 16) KV_BYTES=13980000000;; 32) KV_BYTES=13440000000;;   # bytes unchanged by R182; the 8/32 pools were read with fp32 SSM (937,795 / ~869K) and are ~+13% with bf16
  *) echo "FAILED: no pinned KV budget for SEQS=$SEQS (set KV_BYTES)"; exit 1;; esac; fi
OFFLOAD_ARGS=""; [ "$OFFLOAD" = 1 ] && OFFLOAD_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
case "$SSM_DTYPE" in bfloat16|float32|float16|auto) ;; *) echo "FAILED: SSM_DTYPE must be bfloat16|float32|float16|auto (got $SSM_DTYPE)"; exit 1;; esac
SSM_ARGS="--mamba-ssm-cache-dtype $SSM_DTYPE"
# 0136: FlashInfer 0.6.18 forces split-KV OFF for NVFP4 KV in the FA2 prefill wrapper (= the spec-verification path); SPLIT_KV=1
# re-enables it. The S image ships FlashInfer 0.6.16.post3 whose split path is on by default, so the daily runs SPLIT_KV=0 (r168e:
# closest to bf16 of the three paths; r173c: its decode dumps are identical to rc2 ON's at ctx0 and 30K).
SPLIT_ENV=""; [ "$SPLIT_KV" = 1 ] && SPLIT_ENV="-e VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1"
# R185: the daily forces the pcie_ipc all-reduce on; experiments opt in with PCIE_IPC=1 (an image without 0138 then fails the assert below).
# (R187 fix 2026-09-05: the first version assigned 1 and then read its own value, so every experiment saw PCIE_IPC=1 and images without 0138 failed the assert)
if [ "$EXP" = 0 ]; then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}; fi
case "$PCIE_IPC" in 0|1) ;; *) echo "FAILED: PCIE_IPC must be 0 or 1 (got $PCIE_IPC)"; exit 1;; esac
PCIE_ENV=""; [ "$PCIE_IPC" = 1 ] && PCIE_ENV="-e VLLM_SM12X_PCIE_IPC_AR=1"
XARGS=""; XMOUNT=""; MNBT_=8192; FUS_=""; SPX_=""; XENV=""; CCX_=""
# R183 EXP-only passthrough (the EXP=0 path is unchanged): EXP_MNBT (chunk size; the 8192 assert follows it), FUSIONS_APPEND (raw
# pass_config pairs -> v0280 FUSIONS), SPEC_EXTRA (raw JSON pairs appended inside --speculative-config), EXTRA_ENV_APPEND (-e pairs), CC_EXTRA (raw top-level
# compilation-config pairs -> v0280 CCEXTRA).
if [ "$EXP" != 0 ]; then XARGS="${EXTRA_ARGS_APPEND:-}"; XMOUNT="${EXTRA_MOUNT_APPEND:-}"; MNBT_=${EXP_MNBT:-8192}; FUS_="${FUSIONS_APPEND:-}"; SPX_="${SPEC_EXTRA:-}"; XENV="${EXTRA_ENV_APPEND:-}"; CCX_="${CC_EXTRA:-}"; fi
case "$MNBT_" in *[!0-9]*|"") echo "FAILED: EXP_MNBT must be an integer (got $MNBT_)"; exit 1;; esac
# R195: the daily forces batch-sharded sampling on; experiments opt in with BSS=1 or by passing the flag in EXTRA_ARGS_APPEND (r191/r193* do).
if [ "$EXP" = 0 ]; then BSS=1; else BSS=${BSS:-0}; case "${XARGS:-}" in *enable-batch-sharded-sampling*) BSS=1;; esac; fi
case "$BSS" in 0|1) ;; *) echo "FAILED: BSS must be 0 or 1 (got $BSS)"; exit 1;; esac
BSS_ARGS=""; if [ "$BSS" = 1 ]; then case "${XARGS:-}" in *enable-batch-sharded-sampling*) ;; *) BSS_ARGS="--enable-batch-sharded-sampling";; esac; fi
NS_=9; DTP_=2; SPEC_METHOD_=dflash; if [ "$EXP" != 0 ]; then NS_=${SPEC_NS:-9}; DTP_=${SPEC_DTP:-2}; SPEC_METHOD_=${SPEC_METHOD:-dflash}; fi
# R197 EXP-only passthrough: SPEC_METHOD=mtp swaps the DFlash2 drafter for the checkpoint's own MTP head (vLLM method qwen3_5_mtp, SPEC_NS
# tokens, no /draft mount; the drafter-graph asserts below apply to dflash only). The daily port always runs dflash ns9 draft_tp2.
case "$SPEC_METHOD_" in dflash|mtp) ;; *) echo "FAILED: SPEC_METHOD must be dflash or mtp (got $SPEC_METHOD_)"; exit 1;; esac
case "$NS_$DTP_" in *[!0-9]*) echo "FAILED: SPEC_NS/SPEC_DTP must be integers (got $NS_/$DTP_)"; exit 1;; esac
if [ "$SPEC_METHOD_" = mtp ]; then
  SPEC_JSON_='{"method":"qwen3_5_mtp","num_speculative_tokens":'$NS_"${SPX_:+,$SPX_}"'}'; DRAFT_MOUNT=""; SPEC_DESC="MTP head ns$NS_"
else
  SPEC_JSON_='{"method":"dflash","model":"/draft","num_speculative_tokens":'$NS_',"draft_tensor_parallel_size":'$DTP_',"attention_backend":"FLASHINFER"'"${SPX_:+,$SPX_}"'}'; DRAFT_MOUNT="-v $DRAFT:/draft:ro"; SPEC_DESC="DFlash2 ns$NS_ draft_tp$DTP_ in CUDA graphs"
fi
[ -f "$MODEL/model.safetensors.index.json" ] || [ -f "$MODEL/model.safetensors" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (build: build-v0290rc2.sh + the fi0616 swap layer + patches-v0290/Dockerfile.pcieipc + Dockerfile.bss-not-a-compile-factor)"; exit 1; }
env PORT=$PORT NAME=$NAME BIND_ADDR=$BIND MODEL_DIR="$MODEL" TP=2 L2MNT="$L2" CPUB=$CPU_B ${T_CAP:+TIER_CAP_GB=$T_CAP} ${T_SCOPE:+TIER_EVICT_SCOPE=$T_SCOPE} ${T_MINFREE:+TIER_MIN_FREE_GB=$T_MINFREE} \
    IMAGE="$IMG" KVD_OVERRIDE=nvfp4 ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=$MNBT_ SEQS=$SEQS ${FUS_:+FUSIONS="$FUS_"} ${CCX_:+CCEXTRA="$CCX_"} UTIL=0.88 MAXLEN=262144 POOL_MIN=$P_MIN POOL_MAX=$P_MAX \
    EXTRA_MOUNT="$DRAFT_MOUNT $XMOUNT" \
    EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES $OFFLOAD_ARGS $SSM_ARGS $BSS_ARGS $XARGS" \
    SPEC_JSON="$SPEC_JSON_" \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1 $SPLIT_ENV $PCIE_ENV $XENV" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "0.29 nvfp4 DAILY FAILED — engine NOT up$([ "$EXP" != 0 ] || echo '; rollback: launch-daily-r189-nobss-0905.sh')"; exit 1; }
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
if [ "$SPEC_METHOD_" = dflash ]; then
  [ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || fail "drafter graphs not captured (0129 inactive?)"
  [ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || fail "drafter fell back to eager"
else
  [ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -eq 0 ] || fail "SPEC_METHOD=mtp but the DFlash2 drafter captured graphs"
fi
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || fail "XQA decode engaged — Bug B dodge not in force"
[ "$(echo "$BOOTLOG" | grep -ac "$(basename "$MODEL")\|compressed-tensors")" -ge 1 ] || fail "checkpoint identity"
[ "$(echo "$ARGS" | grep -ac -- "--max-num-batched-tokens $MNBT_")" -ge 1 ] || fail "MNBT is not $MNBT_ (Bug B dodge = 8192 on the daily)"
[ "$(echo "$ARGS" | grep -ac -- "--mamba-ssm-cache-dtype $SSM_DTYPE")" -ge 1 ] || fail "SSM cache dtype is not $SSM_DTYPE on the container"
# R197: the attention block is sized to the mamba page, which grows with the number of speculative slots (ns9 → 1,584; ns6 → 1,536), so the
# exact-value assert holds for the daily's dflash ns9 only; EXP arms with another ns/method must still show the sizing line, and log its value.
if [ "$SSM_DTYPE" = bfloat16 ]; then
  if [ "$EXP" = 0 ] || { [ "$SPEC_METHOD_" = dflash ] && [ "$NS_" = 9 ]; }; then
    [ "$(echo "$BOOTLOG" | grep -ac 'Setting attention block size to 1584 tokens')" -ge 1 ] || fail "bf16 SSM state should give a 1,584-token attention block (R180); block line missing or different"
    [ "$EXP" != 0 ] || echo "$DAILY_BLOCK" | sudo tee "$L2/.block" >/dev/null
  else
    BLK_=$(echo "$BOOTLOG" | grep -aoE 'Setting attention block size to [0-9]+ tokens' | head -1 | tr -dc 0-9)
    [ -n "$BLK_" ] || fail "attention block sizing line missing"; echo "EXP $SPEC_DESC: attention block $BLK_ tokens (daily ns9 = 1,584)"
  fi
fi
if [ "$SPEC_METHOD_" = dflash ]; then
  [ "$(echo "$ARGS" | grep -acE "num_speculative_tokens.{1,5}$NS_[,}]")" -ge 1 ] && [ "$(echo "$ARGS" | grep -acE "draft_tensor_parallel_size.{1,5}$DTP_[,}]")" -ge 1 ] || fail "speculative config is not ns$NS_ draft_tp$DTP_ on the container"
else
  [ "$(echo "$ARGS" | grep -acE "num_speculative_tokens.{1,5}$NS_[,}]")" -ge 1 ] && [ "$(echo "$ARGS" | grep -ac "qwen3_5_mtp")" -ge 1 ] || fail "speculative config is not MTP ns$NS_ on the container"
fi
[ "$(echo "$ARGS" | grep -ac "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912")" -ge 1 ] || fail "FlashInfer workspace is not 512 MiB (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_NVFP4_XQA=0")" -ge 1 ] || fail "VLLM_SM12X_NVFP4_XQA=0 missing"
# CPU tier (r172): every disk-tier hit is promoted through the CPU tier, so 16 GiB (~1,010 blocks of 2,944; ~1,880 of 1,584) is what lets 131K–262K prompts be served.
[ "$(echo "$ARGS" | grep -acE "cpu_bytes_to_use.{1,5}$CPU_B[,}]")" -ge 1 ] || fail "CPU tier is not $CPU_B bytes on the container"
CPUBLK=$(echo "$BOOTLOG" | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1 | grep -oE '[0-9]+')
[ "$EXP" != 0 ] || [ "${CPUBLK:-0}" -ge 900 ] || fail "CPU tier has ${CPUBLK:-?} blocks (expected ~1,010 at 16 GiB with 2,944-token blocks, ~1,880 with 1,584)"
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
if [ "$PCIE_IPC" = 1 ]; then  # R185, fail closed: a silent fallback to CustomAllreduce is the failure mode this guards
  [ "$(echo "$BOOTLOG" | grep -ac "PCIe IPC all-reduce enabled")" -ge 1 ] || fail "PCIE_IPC=1 but no 'PCIe IPC all-reduce enabled' line (image without 0138, or the kernel fell back to CUSTOM)"
  [ "$(echo "$BOOTLOG" | grep -acF "Using ['PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends")" -ge 1 ] || fail "all-reduce backend order is not PCIE_IPC, CUSTOM, PYNCCL"
  [ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_PCIE_IPC_AR=1")" -ge 1 ] || fail "VLLM_SM12X_PCIE_IPC_AR=1 missing on the container"
else [ "$(echo "$BOOTLOG" | grep -ac "PCIe IPC all-reduce enabled")" -eq 0 ] || fail "pcie_ipc all-reduce engaged although PCIE_IPC=0"; fi
if [ "$BSS" = 1 ]; then  # R195, fail closed (-ge 1: rank 1 does not always log INFO, R190e)
  [ "$(echo "$BOOTLOG" | grep -ac "Batch-sharded sampling enabled")" -ge 1 ] || fail "BSS=1 but no 'Batch-sharded sampling enabled' line"
  [ "$(echo "$ARGS" | grep -ac -- "--enable-batch-sharded-sampling")" -ge 1 ] || fail "--enable-batch-sharded-sampling missing on the container"
else [ "$(echo "$BOOTLOG" | grep -ac "Batch-sharded sampling enabled")" -eq 0 ] || fail "sharded sampling engaged although BSS=0"; fi
if [ "$IMG" = "$DAILY_IMG" ]; then sudo docker exec "$NAME" test -f /opt/prs-markers/0147 || fail "daily image lacks the 0147 marker (image drift: sharded sampling would fork the compile artifact)"; fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$FREE" -ge "$MIN_FREE_MIB" ] || fail "only $FREE MiB free after pre-warm (< $MIN_FREE_MIB) — Bug C headroom missing; lower KV_BYTES"
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
LABEL="0.29 nvfp4 DAILY UP"; [ "$EXP" = 0 ] || LABEL="0.29 nvfp4 EXP UP"
echo "$LABEL on ${BIND}:${PORT} (vllm $VER, FlashInfer $FIVER, RedHat NVFP4 weights + NVFP4 KV pinned $KV_BYTES B/GPU + $SPEC_DESC + SSM $SSM_DTYPE + 0131/0134 + embed offload=$OFFLOAD + split_kv=$SPLIT_KV + pcie_ipc=$PCIE_IPC + bss=$BSS + native disk tier${T_CAP:+ cap ${T_CAP} GB} + CPU tier $CPU_B B, dual 5090, image $IMG, SEQS $SEQS). Pool $POOL, min free VRAM $FREE MiB.$([ "$EXP" != 0 ] || echo ' Rollback: launch-daily-r182-nopcie-0905.sh')"
