# Qwen3.8 NVFP4-KV image on vLLM v0.29.0rc1 for the RTX 5090 (sm_120) — 2026-09-03 (R165).
# Same recipe as patches-v0280/Dockerfile.v0280-nvfp4kv (kept in step by hand), defaults pointed at rc1:
#   base  = the 2026-09-01 nightly (same dependency pins as rc1: torch 2.13.0+cu130, flashinfer 0.6.18)
#   wheel = the rc1 per-commit wheel (wheels.vllm.ai/<sha>/, cu130), installed --no-deps => exact rc1
#   diffs = the v0280 chain rebased by codex (NOTES16.md: 10 rebased, 7 verbatim; verify.sh re-runs the
#           --fuzz=0 dry-run chain + py_compile against a pristine rc1 tree, no GPU)
# Build (on flan): cd /srv/qwen5090/patches-v0290 && sudo docker build -f Dockerfile -t vllm-qwen38:v0290rc1-nvfp4kv .
# Then Dockerfile.revival (0116-0119 + 0129 + 0131) and Dockerfile.prs (0132 + 0133) — see ../build-v0290rc1.sh.
ARG VLLM_BASE=vllm/vllm-openai:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb
FROM ${VLLM_BASE}

ARG VLLM_REF=v0.29.0rc1
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/33898f832c53c3e98999e0ec2c689f61ee92a9bc/vllm-0.29.0rc1-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_EXPECT_VER=0.29.0rc1
ARG EXPECT_FLASHINFER=0.6.18
ARG OVERLAY_JOBS=4
ARG WITH_DFLASH_NC=0
ARG SM_ARCH=120a

# 0) optional exact-version wheel over the base (rc tags). --no-deps: the base's pins are the
#    ref's pins by construction (checked in build-v0290rc1.sh); a dependency re-resolve here
#    would silently drift torch/flashinfer. Asserts are GPU-free (ARGs are env vars in RUN).
RUN set -eux; \
    if [ -n "${VLLM_WHEEL_URL}" ]; then \
      W="/tmp/${VLLM_WHEEL_URL##*/}"; \
      python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "${VLLM_WHEEL_URL}" "$W"; \
      python3 -m pip install --no-cache-dir --no-deps "$W"; rm -f "$W"; \
    fi; \
    python3 -c 'import os, torch, vllm, flashinfer; \
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "flashinfer", flashinfer.__version__); \
exp = os.environ["VLLM_EXPECT_VER"]; assert vllm.__version__.startswith(exp), ("vllm", vllm.__version__, "expected", exp); \
efi = os.environ.get("EXPECT_FLASHINFER", ""); assert not efi or flashinfer.__version__ == efi, ("flashinfer", flashinfer.__version__, "expected", efi); \
assert torch.version.cuda.startswith("13."), ("wheel/base CUDA mismatch", torch.version.cuda); \
import importlib.util as iu; assert iu.find_spec("vllm._C_stable_libtorch") and iu.find_spec("vllm._moe_C_stable_libtorch"), "compiled extensions missing from the wheel"; print("compiled extensions present (load test needs the driver: build script)")'

# 1) vLLM csrc at the release tag (headers + the writer .cu), patched (0102)
COPY 0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff /tmp/0102.diff
RUN set -eux; \
    apt-get update -qq && apt-get install -y -qq git patch >/dev/null && rm -rf /var/lib/apt/lists/*; \
    IMG_VER="$(python3 -c 'import vllm; print(vllm.__version__)')"; echo "image vllm $IMG_VER"; \
    case "$IMG_VER" in ${VLLM_EXPECT_VER}*) ;; *) echo "FATAL: image vllm $IMG_VER != ${VLLM_EXPECT_VER}"; exit 1;; esac; \
    rm -rf /opt/vllm-src; \
    git clone -q --filter=blob:none --no-checkout https://github.com/vllm-project/vllm /opt/vllm-src; \
    git -C /opt/vllm-src checkout -q ${VLLM_REF} -- csrc cmake/patches; \
    test -f /opt/vllm-src/csrc/core/registration.h; \
    TINC="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')"; \
    if grep -q "auto strptr = new std::string" "$TINC/torch/csrc/stable/stableivalue_conversions.h"; then \
      mkdir -p /opt/vllm-sm12x/shadow-include/torch/csrc/stable; \
      cp "$TINC/torch/csrc/stable/stableivalue_conversions.h" /opt/vllm-sm12x/shadow-include/torch/csrc/stable/; \
      patch --batch --forward -p1 -d /opt/vllm-sm12x/shadow-include < /opt/vllm-src/cmake/patches/pytorch_stable_string.patch; \
      echo "torch stable-string leak hotfix: shadow header installed"; \
    fi; \
    test -f /opt/vllm-src/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu; \
    patch --batch --forward -p1 -d /opt/vllm-src < /tmp/0102.diff; \
    grep -q swizzle_v_sf /opt/vllm-src/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu; \
    rm -rf /opt/vllm-src/.git

# 2) AOT-build the overlay op for sm_120a (no GPU needed at build)
COPY overlay/ /opt/vllm-sm12x/
ENV VLLM_SM12X_SHADOW_INC=/opt/vllm-sm12x/shadow-include \
    VLLM_SM12X_CSRC=/opt/vllm-src/csrc \
    VLLM_SM12X_BUILD=/opt/vllm-sm12x/build \
    VLLM_SM12X_ARCH=${SM_ARCH} \
    PYTHONPATH=/opt/vllm-sm12x:${PYTHONPATH:-}
RUN set -eux; \
    export TORCH_CUDA_ARCH_LIST="12.0a" MAX_JOBS=${OVERLAY_JOBS}; \
    which nvcc || export PATH="/usr/local/cuda/bin:$PATH"; nvcc --version | tail -1; \
    cd /opt/vllm-sm12x && python3 build_overlay.py; \
    ls -la /opt/vllm-sm12x/build/*.so

# 2b) suffix decoding runtime (R113 S-arm: vLLM's suffix method requires Arctic Inference)
RUN pip install --no-cache-dir arctic-inference==0.1.1

# 3) vLLM python patches (package-root-relative). R107: 0103 XQA-NVFP4 decode (runtime knob
# VLLM_SM12X_NVFP4_XQA=0 reverts to the R106 FA2 decode route) + 0104 drafter FULL-cudagraph.
COPY 0101-sm120-nvfp4kv-fa2-routing-v0280.diff /tmp/0101.diff
COPY 0101a-ALTERNATIVE-dflash-noncausal-nvfp4-fa2-v0280.diff /tmp/0101a.diff
COPY 0103-sm120-nvfp4-xqa-decode-v0280.diff /tmp/0103.diff
COPY 0104-mtp-drafter-full-cudagraph-v0280.diff /tmp/0104.diff
COPY 0105-dflash-noncausal-nvfp4-fa2-v0280.diff /tmp/0105.diff
COPY 0106-dflash2-selector-sampling-v0280.diff /tmp/0106.diff
COPY 0107-dflash-quantized-draft-loader.diff /tmp/0107.diff
COPY 0108-gdn-kernels-v0280.diff /tmp/0108.diff
COPY 0111-replayssm-spec-v0280.diff /tmp/0111.diff
COPY 0112-xqa-verify-v0280.diff /tmp/0112.diff
COPY 0113-dflash-speculator-graphs-v0280.diff /tmp/0113.diff
COPY 0109-dflash-noncausal-complete-v0280.diff /tmp/0109.diff
RUN set -eux; \
    VROOT="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0101.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0103.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0104.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0105.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0106.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0107.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0108.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0109.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0111.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0112.diff; \
    patch --batch --forward -p1 -d "$VROOT" < /tmp/0113.diff; \
    if [ "${WITH_DFLASH_NC}" = "1" ]; then patch --batch --forward -p1 -d "$VROOT" < /tmp/0101a.diff; echo "0101a APPLIED (DFlash non-causal nvfp4 A/B)"; else echo "0101a skipped (primary: skip-layers)"; fi; \
    for f in v1/attention/backends/flashinfer.py v1/worker/gpu_worker.py v1/spec_decode/llm_base_proposer.py v1/worker/gpu_model_runner.py v1/worker/gpu/spec_decode/dflash2/speculator.py model_executor/models/qwen3_dflash.py v1/attention/backends/gdn_attn.py model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py; do python3 -m py_compile "$VROOT/$f"; done; \
    grep -q "use_fa2_nvfp4_kv" "$VROOT/v1/attention/backends/flashinfer.py"; \
    python3 -c "import torch; torch.ops.load_library('/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so'); assert hasattr(torch.ops.vllm_sm12x, 'reshape_and_cache_nvfp4'); print('overlay op registered')"; \
    echo "NVFP4KV IMAGE OK (vllm $(python3 -c 'import vllm; print(vllm.__version__)'), WITH_DFLASH_NC=${WITH_DFLASH_NC})"
