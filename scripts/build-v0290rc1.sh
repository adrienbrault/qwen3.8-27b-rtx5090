#!/usr/bin/env bash
# R165 (2026-09-03, user "get started on preparing a new image from that RC"): build the vLLM
# v0.29.0rc1 generation of the Qwen3.8 image chain, same patch stack as v0280 (every diff dry-runs
# clean against the rc1 tree: 0101-0113, 0116-0119, 0129, 0131; the nvfp4 writer .cu is byte-identical
# to v0.28.0, so 0102 + the overlay stay required).
#   base   vllm/vllm-openai:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb (2026-09-01; 27 commits past
#          rc1's merge-base f5c3cc240 on main, no requirements/ change vs rc1; torch 2.13.0+cu130,
#          flashinfer 0.6.18 (+cubin, +jit-cache cu130), instanttensor 0.1.9, CUDA 13.0.2). rc1 itself is
#          main@f5c3cc240 + 5 release-branch cherry-picks (#44834 #54052 #54373 #54745 #54747) and has no
#          official image.
#   wheel  https://wheels.vllm.ai/33898f832c53c3e98999e0ec2c689f61ee92a9bc/ (the rc1 tag commit; cu130 root
#          index), installed --no-deps over the base so python + compiled .so are exactly rc1.
#          sha256 23ba38f22cc21fa99ee60265f5500b46f1953ef7c999672b4cb297bd4b993508 (copy in builds/v0290rc1/).
#   tags   vllm-qwen38:v0290rc1-nvfp4kv                 = 0101-0113 (daily-equivalent of v0280-nvfp4kv)
#          vllm-qwen38:v0290rc1-nvfp4kv-revival         = + 0116-0119
#          vllm-qwen38:v0290rc1-nvfp4kv-revival-graphs  = + 0129
#          vllm-qwen38:v0290rc1-nvfp4kv-revival-graphs-ws = + 0131 (candidate-equivalent of ...-graphs-ws)
# No GPU needed. Waits for the TB 2.1 ladder unit (it times cmd-bound work on the shared cores) and then
# takes the GPU-exclusive flock so it never overlaps a measurement. Nothing here touches the engines.
# Host layout as in the README: patches-v0280/ installed at /srv/qwen5090/patches-v0280 (revival Dockerfiles + diffs alongside).
# Unit: sudo systemd-run --unit=r165-build --collect -p User=adrienbrault -p RuntimeMaxSec=7200 bash /srv/qwen5090/build-v0290rc1.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-03-r165-v0290rc1-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
BASE=vllm/vllm-openai:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb
SHA=33898f832c53c3e98999e0ec2c689f61ee92a9bc
WHEEL="https://wheels.vllm.ai/$SHA/vllm-0.29.0rc1-cp38-abi3-manylinux_2_28_x86_64.whl"
TAG=vllm-qwen38:v0290rc1-nvfp4kv

while systemctl is-active --quiet tb21-ladder; do sleep 30; done
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R165 build start (lock held): $TAG chain from $BASE + rc1 wheel ==="
sudo docker pull "$BASE" > "$R/pull.log" 2>&1 || { log "FAILED: pull $BASE"; exit 1; }
log "base digest: $(sudo docker inspect --format '{{index .RepoDigests 0}}' "$BASE")"

build(){ # $1 tag, $2 dir, $3 dockerfile, rest = --build-arg ...
  local tag=$1 dir=$2 df=$3; shift 3
  log "--- build $tag ($dir/$df $*)"
  ( cd "$dir" && sudo docker build -f "$df" "$@" -t "$tag" . ) > "$R/build-${tag##*:}.log" 2>&1
  local rc=$?
  grep -aE "IMAGE OK|APPLIED|^vllm |compiled extensions|FATAL|error:|Error|FAILED" "$R/build-${tag##*:}.log" | tail -8 | cut -c1-200 | sed "s/^/[${tag##*:}] /" | tee -a "$R/audit.log"
  [ $rc -eq 0 ] || { log "FAILED: build $tag rc=$rc (log $R/build-${tag##*:}.log)"; exit 1; }
}
build "$TAG" /srv/qwen5090/patches-v0280 Dockerfile.v0280-nvfp4kv \
  --build-arg VLLM_BASE="$BASE" --build-arg VLLM_REF=v0.29.0rc1 --build-arg VLLM_WHEEL_URL="$WHEEL" \
  --build-arg VLLM_EXPECT_VER=0.29.0rc1 --build-arg EXPECT_FLASHINFER=0.6.18 --build-arg OVERLAY_JOBS=4
build "$TAG-revival" /srv/qwen5090/patches-v0280 Dockerfile.v0280-nvfp4kv-revival --build-arg BASE="$TAG"
build "$TAG-revival-graphs" /srv/qwen5090/patches-v0280 Dockerfile.v0280-nvfp4kv-revival-graphs --build-arg BASE="$TAG-revival"
build "$TAG-revival-graphs-ws" /srv/qwen5090/patches-v0280 Dockerfile.v0280-nvfp4kv-revival-graphs-ws --build-arg BASE="$TAG-revival-graphs"

# GPU-free identity check of the final layer: version, overlay op registered, patch markers present
sudo docker run --rm --entrypoint python3 "$TAG-revival-graphs-ws" -c '
import os, torch, vllm, flashinfer
torch.ops.load_library("/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so"); assert hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4")
root = os.path.dirname(vllm.__file__); fi = open(os.path.join(root, "v1/attention/backends/flashinfer.py")).read()
for m in ("use_fa2_nvfp4_kv", "_shrink_pooled_int_workspace", "VLLM_SM12X_DFLASH_GRAPHS"): assert m in fi, m
print("IDENTITY OK", vllm.__version__, torch.__version__, flashinfer.__version__)' 2>&1 | tail -1 | tee -a "$R/audit.log"
sudo docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep v0290rc1 | tee "$R/images.txt" | sed 's/^/[image] /' | tee -a "$R/audit.log"
log "=== R165 build DONE — next: audition on :8029 (fp8 daily shape + nvfp4 SEQS 16/32 with and without 0131; fidelity ruler) ==="
