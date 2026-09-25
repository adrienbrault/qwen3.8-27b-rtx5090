#!/usr/bin/env bash
# Builds the image scripts/serve-r231-nvidia-daily.sh serves, from a clean clone, in one command:
#   bash scripts/build-served-image.sh
# Nine layers, each tagged the way the launcher expects (context patches-v0290/, each tag = previous tag + suffix):
#   vllm-qwen38:v0290rc2-nvfp4kv   Dockerfile                          vLLM v0.29.0rc2 wheel over a pinned nightly, 0101-0113, the sm_120a NVFP4-KV overlay op
#     -revival                     Dockerfile.revival                  0116-0119, 0129, 0131
#     -prs                         Dockerfile.prs                      0132-0137
#     -fi0616                      Dockerfile.fiswap                   FlashInfer 0.6.16.post3 in place of the base's 0.6.18
#     -pcieipc                     Dockerfile.pcieipc                  0138 + the pcie_ipc all-reduce extension, compiled for sm_120a at build time
#     -bsshash                     Dockerfile.bss-not-a-compile-factor 0147
#     -mtppcie                     Dockerfile.pcie-mtp                 0148
#     -mtpcache                    Dockerfile.mtp-cache                0152, 0154, 0155, 0156
#     -eagleshift                  Dockerfile.mtp-eagle-shift          0158  <- the served tag
# The build args are the ones recorded in the served image's `docker history` and in the build logs of 2026-09-03 to 09-06
# (results/2026-09-03-r168-v0290rc2-image, 2026-09-05-r185-pcieipc, 2026-09-06-r205-mtp-cache-image, 2026-09-06-r205d-eagleshift-image).
# One difference: the base is pinned by digest, because the nightly tag it was built from has been deleted from Docker Hub (below).
#
# Requirements: x86_64 Linux, Docker with BuildKit through the buildx plugin, and the active builder on the `docker` driver
# (`docker buildx use default`). Dockerfile.pcieipc and Dockerfile.bss-not-a-compile-factor use `RUN --network=none`, which the
# legacy builder rejects; a docker-container builder cannot see the locally built `FROM vllm-qwen38:*` tags. No GPU is used:
# CUDA code compiles for sm_120a with nvcc from the base image. CHECK=1 adds an identity check that needs the NVIDIA container
# runtime (it loads libcuda and the compiled ops, and runs nothing on the GPU).
#
# Cost, from the 2026-09-03 and 09-04 builds on a Ryzen 7 9800X3D (results/2026-09-03-r168-v0290rc2-image, build niced beside
# other work): the first layer about 10 minutes once the base is local, the FlashInfer swap about 5 minutes, the other seven
# layers under a minute together. Downloads: the base image (8.65 GB compressed), the vLLM wheel (316 MB), FlashInfer
# 0.6.16.post3 with its cubin and cu130 jit-cache wheels, and a vLLM source checkout (csrc/ only). Disk: the served image
# occupies 47.5 GB under the containerd image store, 30.5 GB of it the unpacked base; plan for 60 GB free with build cache.
#
# Pinned upstream inputs. Each can disappear upstream; the build then fails at that step, and nothing here substitutes silently.
#   base   vllm/vllm-openai@sha256:383e409fc7695d6e40cd40d452f3ec277a3d1c462d7b1510034768d26f2cd397, the multi-arch index of
#          nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb (2026-09-01; torch 2.13.0+cu130, FlashInfer 0.6.18). On 2026-09-25 the
#          tag returned 404 on Docker Hub and the digest still resolved. patches-v0290/Dockerfile defaults to the tag, so this
#          script passes the digest. If the digest goes too, the image cannot be reproduced exactly: VLLM_BASE=<other image>
#          builds on another base, the Dockerfile asserts reject one without torch cu13x and FlashInfer 0.6.18, and
#          Dockerfile.pcieipc rejects one without torch 2.13.0+cu130. What such a base yields is a different image from the one
#          measured.
#   wheel  https://wheels.vllm.ai/586f1d6d2da011744e1bae26c8686dc206bf648c/ (the v0.29.0rc2 tag commit, cu130), installed --no-deps.
#          If it goes, build the wheel from vLLM at tag v0.29.0rc2 and pass VLLM_WHEEL_URL pointing at a copy of it.
#   FlashInfer 0.6.16.post3 from https://flashinfer.ai/whl (flashinfer-python, flashinfer-cubin) and /whl/cu130 (jit-cache).
#          Without the jit-cache wheel the layer still builds and says JIT-ONLY in its log; kernels then JIT on first boot.
#   vLLM source at tag v0.29.0rc2 from GitHub (csrc/ for the overlay op), arctic-inference==0.1.1 from PyPI.
#
# Knobs:
#   DRY_RUN=1   print the docker commands and check the COPY sources; runs no docker command
#   FORCE=1     rebuild tags that already exist (default: an existing tag is skipped). This replaces the tag a running
#               launcher serves; the launcher checks the tag, not the image ID, so its next boot uses the rebuilt image.
#   CHECK=1     identity check of the final image (NVIDIA container runtime required)
#   NO_CACHE=1  pass --no-cache to every layer, so nothing is reused from an earlier build on this host
#   IMAGE_REPO  repository name for the nine tags (default vllm-qwen38, the one the launcher serves); another name builds
#               the chain beside an existing one without touching it
#   DOCKER="sudo docker"   the docker command (default `docker`); buildx must be installed for the user it runs as
#   VLLM_BASE / VLLM_WHEEL_URL   override the pinned base and wheel (produces a different image; see above)
#   LOG_DIR     per-layer build logs (default ./build-logs, gitignored)
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=${DRY_RUN:-0}; FORCE=${FORCE:-0}; CHECK=${CHECK:-0}; NO_CACHE=${NO_CACHE:-0}
read -r -a DOCKER_CMD <<< "${DOCKER:-docker}"
LOG_DIR=${LOG_DIR:-build-logs}
CTX=patches-v0290
LAUNCHER=scripts/serve-r231-nvidia-daily.sh
VLLM_BASE=${VLLM_BASE:-vllm/vllm-openai@sha256:383e409fc7695d6e40cd40d452f3ec277a3d1c462d7b1510034768d26f2cd397}
VLLM_WHEEL_URL=${VLLM_WHEEL_URL:-https://wheels.vllm.ai/586f1d6d2da011744e1bae26c8686dc206bf648c/vllm-0.29.0rc2-cp38-abi3-manylinux_2_28_x86_64.whl}
FI_VER=0.6.16.post3
IMAGE_REPO=${IMAGE_REPO:-vllm-qwen38}
T=$IMAGE_REPO:v0290rc2-nvfp4kv
export DOCKER_BUILDKIT=1

log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
die(){ log "FAILED: $*"; exit 1; }

EXPECT=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$LAUNCHER" | head -1)
[ -n "$EXPECT" ] || die "no DAILY_IMG= line in $LAUNCHER"
EXPECT=$IMAGE_REPO:${EXPECT#*:}

# Every file a Dockerfile COPYs must exist in the context (checked in DRY_RUN too; no docker needed).
check_copy_sources(){ local df=$1 line src missing=0
  [ -f "$CTX/$df" ] || die "$CTX/$df missing"
  while IFS= read -r line; do
    set -- $line; shift                     # drop COPY
    while [ $# -gt 1 ]; do src=$1; shift; case $src in --*) continue;; esac
      [ -e "$CTX/$src" ] || { log "missing COPY source in $df: $CTX/$src"; missing=1; }; done
  done < <(grep -E '^COPY ' "$CTX/$df")
  [ $missing = 0 ] || die "$df has missing COPY sources"; }

build(){ # $1 tag, $2 Dockerfile, rest = extra docker build args
  local tag=$1 df=$2 attempt rc start created lf; shift 2
  check_copy_sources "$df"
  local cmd=("${DOCKER_CMD[@]}" build --progress=plain -f "$CTX/$df" "$@" -t "$tag" "$CTX")
  [ "$NO_CACHE" = 1 ] && cmd=("${cmd[@]:0:2}" --no-cache "${cmd[@]:2}")
  if [ "$DRY_RUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  if [ "$FORCE" != 1 ] && "${DOCKER_CMD[@]}" image inspect "$tag" >/dev/null 2>&1; then log "--- $tag exists, skipped (FORCE=1 rebuilds)"; return 0; fi
  lf="$LOG_DIR/${tag##*:}.log"
  for attempt in 1 2 3; do
    log "--- build $tag ($df) attempt $attempt, log $lf"
    start=$(date +%s)
    rc=0; "${cmd[@]}" > "$lf" 2>&1 || rc=$?
    grep -aE "APPLIED|IMAGE OK|IDENTITY|FATAL|ERROR|error:|Hunk .* FAILED" "$lf" | tail -6 | cut -c1-200 | sed "s/^/    [${tag##*:}] /" || true
    [ $rc = 0 ] && { log "build $tag OK"; return 0; }
    # Containerd content-store race while another build or pull runs (seen 2026-09-03): retry.
    if grep -aq "failed to export layer\|failed to commit: rename" "$lf"; then log "containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
    # Containerd image store: BuildKit names the image, then fails the unpack with insufficient_scope while fetching a base
    # blob from docker.io/library/vllm-qwen38 (seen on both BuildKit layers of the served chain, 2026-09-04 and 09-05; those
    # images are the ones served). Accepted only if the tag was named in this run and carries a creation time from this run.
    if grep -aq "insufficient_scope" "$lf" && grep -aq "naming to docker.io/library/$tag done" "$lf"; then
      created=$("${DOCKER_CMD[@]}" image inspect -f '{{.Created}}' "$tag" 2>/dev/null || true)
      if [ -n "$created" ] && [ "$(date -d "$created" +%s 2>/dev/null || echo 0)" -ge "$start" ]; then
        log "build $tag: tagged at $created, then the unpack failed with insufficient_scope; keeping the tag"; return 0; fi
    fi
    break
  done
  die "build $tag rc=$rc (log $lf)"; }

log "=== build of $EXPECT from $CTX/ (DRY_RUN=$DRY_RUN FORCE=$FORCE CHECK=$CHECK NO_CACHE=$NO_CACHE) ==="
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$LOG_DIR"
  "${DOCKER_CMD[@]}" buildx version >/dev/null 2>&1 || die "'${DOCKER_CMD[*]} buildx' is not available. Dockerfile.pcieipc and Dockerfile.bss-not-a-compile-factor use RUN --network=none, which needs BuildKit; install the buildx plugin for the user that runs '${DOCKER_CMD[*]}'"
  drv=$("${DOCKER_CMD[@]}" buildx inspect 2>/dev/null | sed -nE 's/^Driver:[[:space:]]+//p' | head -1 || true)
  [ "$drv" = docker ] || die "the active buildx builder uses driver '$drv'; the chain needs the 'docker' driver to see its own FROM vllm-qwen38:* tags (docker buildx use default)"
  if command -v curl >/dev/null; then curl -sfI -m 30 "$VLLM_WHEEL_URL" >/dev/null || die "vLLM wheel not reachable: $VLLM_WHEEL_URL (see the header: build it from tag v0.29.0rc2 and pass VLLM_WHEEL_URL)"; fi
  if ! "${DOCKER_CMD[@]}" image inspect "$VLLM_BASE" >/dev/null 2>&1; then
    log "--- pull $VLLM_BASE"
    "${DOCKER_CMD[@]}" pull "$VLLM_BASE" || die "base image not pullable: $VLLM_BASE (see the header: an exact rebuild needs this digest)"
  fi
else
  printf '%q ' "${DOCKER_CMD[@]}" pull "$VLLM_BASE"; echo
fi

build "$T"                                           Dockerfile --build-arg VLLM_BASE="$VLLM_BASE" --build-arg VLLM_REF=v0.29.0rc2 \
  --build-arg VLLM_WHEEL_URL="$VLLM_WHEEL_URL" --build-arg VLLM_EXPECT_VER=0.29.0rc2 --build-arg OVERLAY_JOBS=3
build "$T-revival"                                   Dockerfile.revival                  --build-arg BASE="$T"
build "$T-revival-prs"                               Dockerfile.prs                      --build-arg BASE="$T-revival"
build "$T-revival-prs-fi0616"                        Dockerfile.fiswap                   --build-arg BASE="$T-revival-prs" --build-arg FI_VER="$FI_VER"
build "$T-revival-prs-fi0616-pcieipc"                Dockerfile.pcieipc                  --build-arg BASE="$T-revival-prs-fi0616" --build-arg PCIE_IPC_BUILD_NATIVE=1
S=$T-revival-prs-fi0616-pcieipc
build "$S-bsshash"                                   Dockerfile.bss-not-a-compile-factor --build-arg BASE="$S"
build "$S-bsshash-mtppcie"                           Dockerfile.pcie-mtp                 --build-arg BASE="$S-bsshash" --network=none
build "$S-bsshash-mtppcie-mtpcache"                  Dockerfile.mtp-cache                --build-arg BASE="$S-bsshash-mtppcie"
FINAL=$S-bsshash-mtppcie-mtpcache-eagleshift
build "$FINAL"                                       Dockerfile.mtp-eagle-shift          --build-arg BASE="$S-bsshash-mtppcie-mtpcache"

[ "$FINAL" = "$EXPECT" ] || die "built $FINAL but $LAUNCHER serves $EXPECT"
if [ "$DRY_RUN" = 1 ]; then log "=== DRY_RUN: final tag $FINAL matches DAILY_IMG in $LAUNCHER ==="; exit 0; fi
"${DOCKER_CMD[@]}" image inspect "$FINAL" >/dev/null 2>&1 || die "$FINAL not present after the build"

if [ "$CHECK" = 1 ]; then
  log "--- identity check of $FINAL (loads libcuda and the compiled ops; nothing runs on the GPU)"
  "${DOCKER_CMD[@]}" run --rm --runtime nvidia --gpus all -e CUDA_VISIBLE_DEVICES= --entrypoint python3 "$FINAL" -c '
import os, torch, vllm, flashinfer
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch
assert vllm.__version__ == "0.29.0rc2", vllm.__version__
assert torch.__version__ == "2.13.0+cu130", torch.__version__
assert flashinfer.__version__ == "0.6.16.post3", flashinfer.__version__
assert hasattr(torch.ops._C, "rotary_embedding") and hasattr(torch.ops._moe_C, "topk_softmax"), "compiled ops not registered"
torch.ops.load_library("/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so"); assert hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4")
root = os.path.dirname(vllm.__file__); rd = lambda p: open(os.path.join(root, p)).read()
fi = rd("v1/attention/backends/flashinfer.py")
for m in ("use_fa2_nvfp4_kv", "_shrink_pooled_int_workspace", "VLLM_SM12X_DFLASH_GRAPHS", "VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM", "_maybe_enable_sm12x_nvfp4_prefill_split_kv"): assert m in fi, m
assert "use_eagle_preserves_target_kv_cache" in rd("v1/core/sched/scheduler.py"), "0134"
assert "maybe_offload_embeddings" in rd("model_executor/model_loader/utils.py"), "0135"
assert "max_capacity_gb" in rd("v1/kv_offload/tiering/fs/manager.py"), "0137"
assert os.path.isfile(os.path.join(os.path.dirname(root), "pcie_ipc_ar21", "_C.so")), "0138 native extension"
for m in ("0147", "0148", "0152", "0154", "0155", "0156", "0158"): assert os.path.isfile("/opt/prs-markers/" + m), "marker " + m
print("IDENTITY OK vllm", vllm.__version__, "torch", torch.__version__, "flashinfer", flashinfer.__version__)' 2>&1 | tail -1 | tee "$LOG_DIR/identity.log" || true
  grep -q "^IDENTITY OK" "$LOG_DIR/identity.log" || die "identity check (log $LOG_DIR/identity.log)"
fi
"${DOCKER_CMD[@]}" image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -F "$T" | tee "$LOG_DIR/images.txt" | sed 's/^/[image] /' || true
log "=== DONE: $FINAL = DAILY_IMG of $LAUNCHER ==="
