#!/usr/bin/env bash
# Serves the measured configuration on any host with two RTX 5090 cards: the engine of scripts/serve-r231-nvidia-daily.sh
# (image, vLLM flags, container env, MTP ns3, NVFP4 KV pinned at 14.86 GB per card, 16 sequences) without the serving
# host's /srv/qwen5090 layout. Settings: serve.env in the repo root, or SERVE_ENV=<file>; environment variables override the
# file; serve.env.example lists every setting and its default.
#   bash scripts/serve.sh           start, wait for /health, check the boot log, print the endpoint
#   bash scripts/serve.sh --print   print the docker run command and exit; runs and creates nothing (also DRY_RUN=1)
#   bash scripts/serve.sh --stop    stop and remove the container
# Left out on purpose: the serving host's experiment ports, rollback launchers, clock offsets, the llama-benchy autotune
# pre-warm with its free-VRAM floor, and its host maintenance (orphaned /dev/shm segment sweep, host-RAM gate before boot,
# boot-time tier eviction by tier-evict.sh). The disk KV tier is off unless KV_TIER_DIR is set.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SERVED_LAUNCHER=$ROOT/scripts/serve-r231-nvidia-daily.sh
die(){ echo "serve.sh: FAILED: $*" >&2; exit 1; }

# Settings: remember what the environment sets (set-but-empty counts, so KV_TIER_DIR= turns off a tier the file enables),
# source the file, then put the environment values back on top.
KNOBS=(MODEL_DIR CACHE_DIR PORT BIND GPUS NAME IMAGE SERVED_NAME SERVED_ALIAS KV_TIER_DIR KV_TIER_CAP_GB POWER_LIMIT_W
       DOCKER HEALTH_TIMEOUT)
declare -A FROM_ENV=()
for k in "${KNOBS[@]}"; do if [ -n "${!k+x}" ]; then FROM_ENV[$k]=${!k}; fi; done
ENV_FILE=${SERVE_ENV:-$ROOT/serve.env}
if [ -r "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  . "$ENV_FILE"
elif [ -n "${SERVE_ENV:-}" ]; then die "SERVE_ENV=$SERVE_ENV does not exist"; fi
for k in "${!FROM_ENV[@]}"; do printf -v "$k" '%s' "${FROM_ENV[$k]}"; done

# The image tag is read from the served launcher, the same way build-served-image.sh reads it, so the two cannot drift.
SERVED_IMG=$(sed -nE '/^DAILY_IMG=/{s/^DAILY_IMG=([^ ]+).*/\1/p;q;}' "$SERVED_LAUNCHER")
[ -n "$SERVED_IMG" ] || die "no DAILY_IMG= line in $SERVED_LAUNCHER"
MODEL_DIR=${MODEL_DIR:-}
CACHE_DIR=${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/qwen38-27b-vllm}
PORT=${PORT:-8020}
BIND=${BIND:-127.0.0.1}                  # the serving host binds 0.0.0.0: its proxies reach the engine over the docker bridge
GPUS=${GPUS:-0,1}                        # two indices, or "all" (what the serving host passes; it has exactly two cards)
NAME=${NAME:-qwen38-27b}
IMAGE=${IMAGE:-$SERVED_IMG}
SERVED_NAME=${SERVED_NAME:-qwen3.8-27b}
SERVED_ALIAS=${SERVED_ALIAS-qwen3.6-27b} # space-separated extra model names; empty for none
KV_TIER_DIR=${KV_TIER_DIR:-}
KV_TIER_CAP_GB=${KV_TIER_CAP_GB:-300}
POWER_LIMIT_W=${POWER_LIMIT_W:-}
DOCKER=${DOCKER:-docker}                 # "sudo docker" where the user is not in the docker group
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1500}   # seconds (the serving host waits 25 minutes); a first boot compiles kernels
read -r -a DK <<< "$DOCKER"
read -r -a ALIASES <<< "$SERVED_ALIAS"

MODE=run
case "${1:-}" in --print) MODE=print;; --stop) MODE=stop;; "") ;; *) die "usage: serve.sh [--print|--stop]";; esac
[ "${DRY_RUN:-0}" != 1 ] || MODE=print
if [ "$MODE" = stop ]; then
  "${DK[@]}" container inspect "$NAME" >/dev/null 2>&1 || { echo "no container named $NAME"; exit 0; }
  "${DK[@]}" stop -t 60 "$NAME" >/dev/null 2>&1 || true
  "${DK[@]}" rm -f "$NAME" >/dev/null; echo "stopped and removed $NAME"; exit 0
fi

[ -n "$MODEL_DIR" ] || die "MODEL_DIR is not set (the nvidia/Qwen3.8-27B-NVFP4 download; see serve.env.example)"
for v in PORT KV_TIER_CAP_GB HEALTH_TIMEOUT; do case "${!v}" in ''|*[!0-9]*) die "$v must be an integer (got '${!v}')";; esac; done
case "$POWER_LIMIT_W" in *[!0-9]*) die "POWER_LIMIT_W must be empty or watts (got '$POWER_LIMIT_W')";; esac
if [ "$GPUS" = all ]; then GPU_ARG=all; GPU_LIST=()
else
  IFS=, read -r -a GPU_LIST <<< "$GPUS"
  for i in "${GPU_LIST[@]}"; do case "$i" in ''|*[!0-9]*) die "GPUS must be 'all' or comma-separated indices (got '$GPUS')";; esac; done
  [ "${#GPU_LIST[@]}" -ge 2 ] || die "GPUS=$GPUS names ${#GPU_LIST[@]} card(s); the configuration is tensor-parallel over two"
  GPU_ARG="\"device=$GPUS\""             # the inner quotes keep docker from splitting the list at the comma
fi

if [ "$MODE" = run ]; then               # --print creates nothing, so these directories appear on the first real start only
  mkdir -p "$CACHE_DIR"/{torch_compile_qwen38_v0280nv,triton,inductor,flashinfer}
  [ -z "$KV_TIER_DIR" ] || mkdir -p "$KV_TIER_DIR"
fi
abs(){ if [ -d "$1" ]; then (cd "$1" && pwd); else echo "$1"; fi; }
MODEL_DIR=$(abs "$MODEL_DIR"); CACHE_DIR=$(abs "$CACHE_DIR"); [ -z "$KV_TIER_DIR" ] || KV_TIER_DIR=$(abs "$KV_TIER_DIR")
for v in MODEL_DIR CACHE_DIR KV_TIER_DIR; do case "${!v}" in ''|/*) ;; *) die "$v=${!v} must be an absolute path";; esac; done

# The disk tier (off by default). The JSON is the served launcher's: a 16 GiB pinned CPU tier that every disk hit is promoted
# through (R172), a filesystem tier at /l2, and patch 0137's LRU eviction with the served cap, scope and floor. Off means no
# connector at all, the same switch as NO_TIER=1 in serve-v0280-daily.sh: the GPU prefix cache is then the only KV cache.
TIER_MOUNT=(); TIER_ARGS=()
if [ -n "$KV_TIER_DIR" ]; then
  TIER_MOUNT=(-v "$KV_TIER_DIR":/l2)
  TIER_ARGS=(--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":17179869184,"offload_prompt_only":true,"secondary_tiers":[{"type":"fs","root_dir":"/l2","n_read_threads":16,"n_write_threads":4,"max_capacity_gb":'"$KV_TIER_CAP_GB"',"evict_scope":"root","min_free_gb":40}]}}')
fi

# The engine, flag for flag and in the order of serve-v0280-daily.sh with the values serve-r231-nvidia-daily.sh passes it.
# docs/CONFIG.md explains each one. The served launcher runs the same command through `bash -c "exec python3 ..."`.
RUN=("${DK[@]}" run -d --name "$NAME" --restart unless-stopped --oom-score-adj -800
  --entrypoint python3 --runtime nvidia --gpus "$GPU_ARG" --ipc=host
  -p "$BIND:$PORT:8000" --shm-size 8g --memory 52g --memory-swap 52g
  -e VLLM_ATTENTION_BACKEND=FLASHINFER
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912      # 512 MiB: with XQA off, the Bug B dodge
  -e CUDA_MODULE_LOADING=LAZY
  -e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4   # bounds the first boot's JIT host RAM
  -e PYTHONHASHSEED=0                                     # stable prefix-cache and tier hashes across boots
  -e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1
  -e VLLM_SM12X_PCIE_IPC_AR=1 -e VLLM_SM12X_PCIE_IPC_MTP=1   # pcie_ipc all-reduce (0138), MTP drafter admitted on it (0148)
  -v "$CACHE_DIR/torch_compile_qwen38_v0280nv":/root/.cache/vllm/torch_compile_cache   # subdir names match the serving host
  -v "$CACHE_DIR/triton":/root/.triton/cache -v "$CACHE_DIR/inductor":/root/.cache/inductor
  -v "$CACHE_DIR/flashinfer":/root/.cache/flashinfer
  "${TIER_MOUNT[@]}" -v "$MODEL_DIR":/model
  "$IMAGE" -m vllm.entrypoints.openai.api_server
  --model /model --served-model-name "$SERVED_NAME" "${ALIASES[@]}" --trust-remote-code
  --kv-cache-dtype nvfp4
  --gpu-memory-utilization 0.88 --max-model-len 262144    # the byte pin below sets the pool, not the utilization
  --max-num-seqs 16 --max-num-batched-tokens 8192         # 8192 = 5 blocks of 1,472 per prefill chunk (R211)
  --limit-mm-per-prompt '{"image":32,"video":0}'
  --mamba-cache-mode align --enable-prefix-caching
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  "${TIER_ARGS[@]}"
  --tensor-parallel-size 2
  --default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"medium"}'
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --enable-prompt-tokens-details
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
  --kv-cache-memory-bytes 14860000000                     # R234 pin: pool 1,391,795 tokens
  --offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens   # embedding table in pinned host RAM (0135)
  --mamba-ssm-cache-dtype bfloat16
  --enable-batch-sharded-sampling)

if [ "$MODE" = print ]; then
  printf '%q ' "${RUN[@]}"; echo
  [ -n "$KV_TIER_DIR" ] || echo "# disk KV tier off (set KV_TIER_DIR to enable)" >&2
  [ -z "$POWER_LIMIT_W" ] || echo "# after the boot checks: sudo nvidia-smi -i <each of ${GPUS}> -pl $POWER_LIMIT_W" >&2
  exit 0
fi

# Preflight: each failure names its fix.
[ "$IMAGE" = "$SERVED_IMG" ] || echo "WARN: IMAGE=$IMAGE is not the served tag $SERVED_IMG" >&2
"${DK[@]}" image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "image $IMAGE not found; build it with: bash scripts/build-served-image.sh (about 20 minutes, 60 GB of disk)"
[ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ] \
  || die "MODEL_DIR=$MODEL_DIR has no config.json/tokenizer.json; download: huggingface-cli download nvidia/Qwen3.8-27B-NVFP4 --local-dir <dir>"
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
[ "${NGPU:-0}" -ge 2 ] || die "nvidia-smi sees ${NGPU:-0} GPU(s); this configuration needs two"
for i in "${GPU_LIST[@]}"; do [ "$i" -lt "$NGPU" ] 2>/dev/null || die "GPUS=$GPUS: no GPU $i (nvidia-smi sees $NGPU)"; done
[ "$GPUS" != all ] || mapfile -t GPU_LIST < <(seq 0 $((NGPU - 1)))
# Tokenizer guard, as serve-v0280-daily.sh does it: a tokenizer.json that carries a truncation setting cuts prompts short,
# so the setting is removed in place and the original kept next to it.
if ! grep -qE '"truncation": *null' "$MODEL_DIR/tokenizer.json"; then
  command -v python3 >/dev/null || die "$MODEL_DIR/tokenizer.json sets truncation; python3 is needed to clear it"
  python3 - "$MODEL_DIR/tokenizer.json" <<'PY'
import json, shutil, sys
p = sys.argv[1]; t = json.load(open(p))
if t.get("truncation") is not None:
    shutil.copy(p, p + ".orig"); t["truncation"] = None
    json.dump(t, open(p, "w"), ensure_ascii=False); print("tokenizer guard: truncation cleared, original in tokenizer.json.orig")
PY
fi
if "${DK[@]}" container inspect "$NAME" >/dev/null 2>&1; then
  echo "replacing the existing container $NAME"; "${DK[@]}" rm -f "$NAME" >/dev/null
fi
port_busy(){ if command -v ss >/dev/null; then [ -n "$(ss -Hltn "sport = :$1" 2>/dev/null)" ]; else (: </dev/tcp/127.0.0.1/"$1") 2>/dev/null; fi; }
if port_busy "$PORT"; then die "port $PORT is in use; set PORT to a free one"; fi
if [ -n "$KV_TIER_DIR" ]; then
  # The tier's namespace hash covers the mount path, dtype and parallelism, not the weights or the block size, so blocks written
  # under another checkpoint or block size would be read back as this one's. The serving host wipes them; here the choice is yours.
  CKPT=$(basename "$MODEL_DIR"); BLOCK=1472   # MTP ns3 with the bf16 SSM state gives a 1,472-token attention block (R207)
  if ls -d "$KV_TIER_DIR"/_model_* >/dev/null 2>&1 && { [ "$(cat "$KV_TIER_DIR/.checkpoint" 2>/dev/null)" != "$CKPT" ] \
       || [ "$(cat "$KV_TIER_DIR/.block" 2>/dev/null)" != "$BLOCK" ]; }; then
    die "$KV_TIER_DIR holds KV blocks of another checkpoint or block size (stamps .checkpoint/.block); empty it with: sudo find $KV_TIER_DIR -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +"
  fi
  { echo "$CKPT" > "$KV_TIER_DIR/.checkpoint" && echo "$BLOCK" > "$KV_TIER_DIR/.block"; } 2>/dev/null \
    || die "cannot write the tier stamps .checkpoint/.block in $KV_TIER_DIR"
  FREE_KB=$(df -Pk "$KV_TIER_DIR" | awk 'NR==2{print $4}')
  [ "$FREE_KB" -ge 5242880 ] || die "under 5 GB free on $KV_TIER_DIR; the tier fails engine init with ENOSPC (R130)"
fi

"${RUN[@]}" >/dev/null || die "docker run failed (NVIDIA container runtime installed?)"
HOST=$BIND; [ "$BIND" != 0.0.0.0 ] || HOST=127.0.0.1
echo "starting $NAME on $BIND:$PORT; a first boot also compiles kernels into $CACHE_DIR (waiting up to ${HEALTH_TIMEOUT}s)"
abort(){ echo "--- last 40 log lines of $NAME" >&2; "${DK[@]}" logs --tail 40 "$NAME" >&2 2>&1 || true
         [ "${2:-}" != rm ] || "${DK[@]}" rm -f "$NAME" >/dev/null 2>&1 || true; die "$1"; }
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -sf -m 5 "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  # --restart unless-stopped would retry a failed init forever; the first exit or restart ends the boot instead.
  [ "$("${DK[@]}" inspect -f '{{.State.Running}} {{.RestartCount}}' "$NAME" 2>/dev/null)" = "true 0" ] \
    || abort "the container exited during startup; removed" rm
  # A worker that dies in kernel warmup leaves the other waiting forever; stop at its first error line. The log is captured
  # before grep so that neither a -q SIGPIPE nor docker's own exit status can turn a match into a miss under pipefail.
  TAIL=$("${DK[@]}" logs --tail 600 "$NAME" 2>&1 || true)
  if grep -qaE "Worker failed with error|torch.AcceleratorError|CUDA error: (invalid argument|an illegal)" <<< "$TAIL"; then
    abort "a worker hit a CUDA error during warmup; removed. Retry once before reporting it" rm
  fi
  [ "$SECONDS" -lt "$deadline" ] || abort "/health did not answer within ${HEALTH_TIMEOUT}s; the container is still running (--stop removes it)"
  sleep 10
done

# Boot checks: the served launcher's asserts, as the log lines each patch or setting prints when it is in force.
LOG=$("${DK[@]}" logs "$NAME" 2>&1)
NEED=("linear-V-scale store overlay ACTIVE"             # 0102: without it NVFP4 KV is written wrong and nothing else notices
      "as specified by kv_cache_memory_bytes"           # the byte pin, not utilization, sized the pool
      "int workspace shrunk 8 MiB -> 1 MiB"             # 0131
      "Setting attention block size to 1472 tokens"
      "Offloader set to UVAOffloader" "Total CPU offloaded parameters: 1.18"
      "PCIe IPC all-reduce enabled" "Using ['PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends"
      "SM12X PCIe IPC: MTP drafter admitted" "Batch-sharded sampling enabled"
      "SM12X eagle-drop replay boundary retained")        # 0158: without it every MTP prefix revisit re-prefills
[ -z "$KV_TIER_DIR" ] || NEED+=("OffloadingConnector")
bad=0
for l in "${NEED[@]}"; do grep -qaF -- "$l" <<< "$LOG" || { echo "boot log lacks: $l" >&2; bad=1; }; done
for l in "decode_backend=xqa" "re-enabled FlashInfer split-KV"; do
  if grep -qaF -- "$l" <<< "$LOG"; then echo "boot log shows: $l" >&2; bad=1; fi
done
VERS=$("${DK[@]}" exec "$NAME" python3 -c 'import vllm, flashinfer; print(vllm.__version__, flashinfer.__version__)' 2>/dev/null || true)
case "$VERS" in "0.29"*" 0.6.16"*) ;; *) echo "vllm/flashinfer are '$VERS', expected 0.29.x / 0.6.16.x" >&2; bad=1;; esac
"${DK[@]}" exec "$NAME" test -f /opt/prs-markers/0147 -a -f /opt/prs-markers/0148 -a -f /opt/prs-markers/0158 \
  || { echo "image lacks a patch marker (0147/0148/0158)" >&2; bad=1; }
[ "$bad" = 0 ] || die "the engine answers but differs from the served configuration (lines above); it is still running (--stop removes it)"
POOL=$(grep -a 'GPU KV cache size' <<< "$LOG" | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9 || true)
[ "$POOL" = 1391795 ] || echo "WARN: KV pool ${POOL:-?} tokens; the served boots read 1,391,795 at this pin (R234)" >&2
if [ -n "$KV_TIER_DIR" ]; then   # a host that cannot pin 16 GiB gets fewer CPU-tier blocks, and long prompts stop being tier-served
  CPUBLK=$(grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' <<< "$LOG" | tail -1 | tr -dc 0-9 || true)
  [ "${CPUBLK:-0}" -ge 550 ] || echo "WARN: CPU tier holds ${CPUBLK:-?} blocks; the served boots hold 595 of 1,472 tokens (R207)" >&2
fi

# Power cap, applied as the serving host applies it: after the checks pass, per card, a warning rather than a failure.
# nvidia-smi -pl lasts until reboot or driver reload; 400 W is the lowest value these cards accept.
if [ -n "$POWER_LIMIT_W" ]; then
  for i in "${GPU_LIST[@]}"; do
    sudo nvidia-smi -i "$i" -pl "$POWER_LIMIT_W" >/dev/null || echo "WARN: could not set GPU $i to $POWER_LIMIT_W W" >&2
  done
fi
echo "UP: http://$HOST:$PORT/v1, model $SERVED_NAME${SERVED_ALIAS:+ (alias $SERVED_ALIAS)}, KV pool ${POOL:-?} tokens," \
     "disk tier ${KV_TIER_DIR:-off}${POWER_LIMIT_W:+, power limit $POWER_LIMIT_W W}"
