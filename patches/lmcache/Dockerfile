# Tiered-KV image: the base daily image + LMCache main + the six local patches that make
# DRAM/NVMe KV offload correct on this fp8 hybrid. Build ../Dockerfile first.
#
#   docker build -t vllm-qwen36:tiers -f Dockerfile .        (from this directory)
#
# Every patch below carries a CPU regression test that runs as a BUILD GATE — an image that
# fails a regression cannot exist. See README.md here for what each one fixes and why.
FROM vllm-qwen36:patched

# --- LMCache: main @ e38ee415 + 0001, 0002, 0007, 0008 ------------------------
# 0001/0002 are the correctness pair (fp8 fused kernel-page -> logical hybrid page regroup).
# Without them, releases through 0.5.1 store *nothing* on this layout and main stores
# one wrongly-addressed 16-token page per 1616-token block: coherent output, vanished facts.
# 0007 is the VRAM diet (staging-batch knob), 0008 is the fs_native capacity cap that
# actually evicts — unpatched, max_capacity_gb is telemetry only and L2 grows unbounded
# until the host disk is full (measured: 876 GB against a 60 GB cap).
COPY 0001-fix-fused-hybrid-subpage-view.diff /tmp/0001.diff
COPY 0002-strided-fp8-regroup.diff           /tmp/0002.diff
COPY 0007-sidecar-vram-staging-batch.diff    /tmp/0007.diff
COPY 0008-fs-native-cap-enforcement.diff     /tmp/0008.diff
RUN set -e; apt-get update -qq && apt-get install -y -qq git >/dev/null \
 && git clone https://github.com/LMCache/LMCache /opt/lmc-src \
 && cd /opt/lmc-src && git checkout e38ee4157a11703b07845f45fd98e714b25c13cd \
 && for p in 0001 0002 0007 0008; do git apply --check /tmp/$p.diff && git apply /tmp/$p.diff; done \
 && export CPATH="$(python3 -c "import glob;print(':'.join(sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/*/include'))))")" \
 && export LIBRARY_PATH="$(python3 -c "import glob;print(':'.join(sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/*/lib'))))")" \
 && TORCH_CUDA_ARCH_LIST="12.0+PTX" MAX_JOBS=4 \
    pip install --no-cache-dir --no-deps --no-build-isolation . \
 && pip install --no-cache-dir -q pytest \
 && python3 -m pytest -x -q tests/v1/test_vllm_kv_cache_group_edits.py \
                            tests/v1/platform/test_gpu_staging_config.py \
                            tests/v1/distributed/test_fs_native_capacity.py \
                            tests/v1/distributed/test_l2_adapter_factory.py \
 && python3 -c "import lmcache; print('lmcache', lmcache.__version__, '+0001+0002+0007+0008 OK')"

# --- vLLM: 0003, 0005 --------------------------------------------------------
# Both are connector-path scheduler bugs that only fire with a KV connector configured
# AND MTP on AND a hybrid model — i.e. exactly this profile, and nowhere else, which is
# why they survived upstream. Grafted onto the installed tree (paths per FIX_0003_NOTES).
COPY 0003-vllm-connector-eagle-hybrid-hit.diff              /tmp/0003.diff
COPY 0005-vllm-residual-mamba-connector-prefill-boundary.diff /tmp/0005.diff
RUN set -e; VROOT=/usr/local/lib/python3.12/dist-packages/vllm; cd "$VROOT" \
 && for p in 0003 0005; do git apply --unsafe-paths --check /tmp/$p.diff \
                        && git apply --unsafe-paths /tmp/$p.diff; done \
 && python3 -m py_compile v1/core/kv_cache_coordinator.py v1/core/sched/scheduler.py \
 && python3 tests/v1/core/test_connector_eagle_hybrid_hit.py \
 && python3 tests/v1/core/test_connector_mamba_prefill_boundary.py \
 && echo "TIER IMAGE OK (0001+0002+0003+0005+0007+0008)"
