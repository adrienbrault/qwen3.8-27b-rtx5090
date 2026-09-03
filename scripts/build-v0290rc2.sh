#!/usr/bin/env bash
# R168 (2026-09-03, user "focus on .29 and all related improvements, including nvfp4"): build the vLLM v0.29.0rc2
# generation of the Qwen3.8 image chain plus the two FlashInfer-swap diagnosis images. CPU-only; does NOT take the
# GPU-exclusive lock (the SWE-bench campaign holds it for hours and is a score, not a latency read) — runs niced instead.
#   rc2    = rc1 + 3 commits (#52285 platform-plugin logging, #54962 PP tensor sends in gpu_worker.py, #54994 multimodal
#            SHM cache), no requirements/ change, so the rc1 base nightly (torch 2.13.0+cu130, flashinfer 0.6.18) still
#            carries rc2's exact pins. Wheel https://wheels.vllm.ai/586f1d6d2da011744e1bae26c8686dc206bf648c/ (rc2 tag
#            commit, cu130), --no-deps over the base. The whole 0101-0135 chain was applied --fuzz=0 to a local rc2 tree
#            (pristine rc1 tree + the 4 rc2-changed vllm files) with the rc1 tree as control: 0 rejects on both (0101 touches
#            gpu_worker.py but not #54962's hunks).
#   tags   vllm-qwen38:v0290rc2-nvfp4kv              = 0101-0113 (daily-equivalent of v0280-nvfp4kv)
#          vllm-qwen38:v0290rc2-nvfp4kv-revival      = + 0116-0119 + 0129 + 0131 (candidate chain)
#          vllm-qwen38:v0290rc2-nvfp4kv-revival-prs  = + 0132 + 0133 + 0134 + 0135 (embed offload folded in, R167)
#          vllm-qwen38:v0280-nvfp4kv-revival-graphs-ws-fi0618 = the v0.28 candidate image + FlashInfer 0.6.18 (patches-v0290/Dockerfile.fiswap)
#          vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed-fi0616 = the rc1 R167 image + FlashInfer 0.6.16.post3 (mirror)
#   The swap images are built FIRST (minutes; r168-deep-decode.sh needs them), the rc2 chain after (~1 h).
#   FISWAP_ONLY=1 builds just the two swap images (re-run after the cubin-index fix while the rc2 chain build continued).
# Unit: sudo systemd-run --unit=r168-build --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p Nice=19 -p IOSchedulingClass=idle bash /srv/qwen5090/build-v0290rc2.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-03-r168-v0290rc2-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
BASE=vllm/vllm-openai:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb
SHA=586f1d6d2da011744e1bae26c8686dc206bf648c
WHEEL="https://wheels.vllm.ai/$SHA/vllm-0.29.0rc2-cp38-abi3-manylinux_2_28_x86_64.whl"
TAG=vllm-qwen38:v0290rc2-nvfp4kv
V28=vllm-qwen38:v0280-nvfp4kv-revival-graphs-ws
RC1E=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed
D=/srv/qwen5090/patches-v0290
log "=== R168 build start (no GPU lock; niced): fiswap images, then $TAG chain from $BASE + rc2 wheel ==="
for i in "$BASE" "$V28" "$RC1E"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "FAILED: image $i missing"; exit 1; }; done
curl -sfI -m 30 "$WHEEL" >/dev/null || { log "FAILED: rc2 wheel not reachable: $WHEEL"; exit 1; }

build(){ # $1 tag, $2 dockerfile, rest = --build-arg ...
  local tag=$1 df=$2; shift 2
  log "--- build $tag ($df $*)"
  ( cd "$D" && sudo nice -n 19 ionice -c3 docker build -f "$df" "$@" -t "$tag" . ) > "$R/build-${tag##*:}.log" 2>&1
  local rc=$?
  grep -aE "IMAGE OK|APPLIED|^vllm |compiled extensions|FATAL|error:|Error|FAILED" "$R/build-${tag##*:}.log" | tail -8 | cut -c1-200 | sed "s/^/[${tag##*:}] /" | tee -a "$R/audit.log"
  [ $rc -eq 0 ] || { log "FAILED: build $tag rc=$rc (log $R/build-${tag##*:}.log)"; return 1; }
}
# 1) FlashInfer-swap diagnosis images (independent of rc2)
build "$V28-fi0618" Dockerfile.fiswap --build-arg BASE="$V28" --build-arg FI_VER=0.6.18 || log "fi0618 swap image FAILED — r168 cell V28F unavailable"
build "$RC1E-fi0616" Dockerfile.fiswap --build-arg BASE="$RC1E" --build-arg FI_VER=0.6.16.post3 || log "fi0616 swap image FAILED — r168 cell RC1F unavailable"
touch "$R/FISWAP-DONE"
[ "${FISWAP_ONLY:-0}" != 1 ] || { log "=== fiswap-only run DONE ==="; exit 0; }
# 2) rc2 chain
build "$TAG" Dockerfile --build-arg VLLM_REF=v0.29.0rc2 --build-arg VLLM_WHEEL_URL="$WHEEL" --build-arg VLLM_EXPECT_VER=0.29.0rc2 --build-arg OVERLAY_JOBS=3 || exit 1
build "$TAG-revival" Dockerfile.revival --build-arg BASE="$TAG" || exit 1
build "$TAG-revival-prs" Dockerfile.prs --build-arg BASE="$TAG-revival" || exit 1

# Identity check of the final layer (loads libcuda only; nothing runs on the GPU)
sudo docker run --rm --runtime nvidia --gpus all --entrypoint python3 "$TAG-revival-prs" -c '
import os, torch, vllm, flashinfer
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch
assert vllm.__version__ == "0.29.0rc2", vllm.__version__
assert hasattr(torch.ops._C, "rotary_embedding") and hasattr(torch.ops._moe_C, "topk_softmax"), "compiled ops not registered"
torch.ops.load_library("/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so"); assert hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4")
root = os.path.dirname(vllm.__file__); fi = open(os.path.join(root, "v1/attention/backends/flashinfer.py")).read()
for m in ("use_fa2_nvfp4_kv", "_shrink_pooled_int_workspace", "VLLM_SM12X_DFLASH_GRAPHS", "VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM"): assert m in fi, m
assert "use_eagle_preserves_target_kv_cache" in open(os.path.join(root, "v1/core/sched/scheduler.py")).read(), "0134 marker"
assert "maybe_offload_embeddings" in open(os.path.join(root, "model_executor/model_loader/utils.py")).read(), "0135 marker"
print("IDENTITY OK", vllm.__version__, torch.__version__, flashinfer.__version__)' 2>&1 | tail -1 | tee -a "$R/audit.log"
sudo docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E "v0290rc2|fi06" | tee "$R/images.txt" | sed 's/^/[image] /' | tee -a "$R/audit.log"
touch "$R/RC2-DONE"
log "=== R168 build DONE — next: r168-deep-decode.sh (regression isolation), then the rc2 audition ==="
