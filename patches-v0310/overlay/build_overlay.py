"""AOT-build the SM12x NVFP4 KV linear-V-scale store overlay.

Compiles the PATCHED nvfp4_kv_cache_kernels.cu (vLLM csrc + 0002 diff) and
overlay_binding.cpp into ONE shared object that registers
torch.ops.vllm_sm12x.reshape_and_cache_nvfp4. Runs at docker-build time
(no GPU needed: nvcc + libtorch only; TORCH_CUDA_ARCH_LIST is pinned so torch
never queries a device). Runtime loader: vllm_sm12x_nvfp4kv.py.

Env:
  VLLM_SM12X_CSRC   csrc root of the vLLM checkout matching the image commit
                    (needs libtorch_stable/*, core/registration.h)  [/opt/vllm-src/csrc]
  VLLM_SM12X_BUILD  output dir for the .so                            [/opt/vllm-sm12x/build]
  VLLM_SM12X_ARCH   gencode arch (5090 = 120a; GB10 = 121a)           [120a]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = Path(os.environ.get("VLLM_SM12X_CSRC", "/opt/vllm-src/csrc"))
BUILD = Path(os.environ.get("VLLM_SM12X_BUILD", "/opt/vllm-sm12x/build"))
ARCH = os.environ.get("VLLM_SM12X_ARCH", "120a")
NAME = "vllm_sm12x_nvfp4kv"

# The .cu is NOT vendored: it is upstream's csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
# with 0002-nvfp4-writer-linear-vscale-sm12x.diff applied (Dockerfile does the patch).
KERNEL_CU = CSRC / "libtorch_stable" / "nvfp4_kv_cache_kernels.cu"


def build() -> Path:
    import torch  # noqa: F401  (needed for cpp_extension)
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a" if ARCH == "120a" else "12.1a")
    assert KERNEL_CU.exists(), f"missing {KERNEL_CU} (clone vLLM csrc + apply 0002)"
    assert (CSRC / "core" / "registration.h").exists(), "csrc/core/registration.h missing"
    if "swizzle_v_sf" not in KERNEL_CU.read_text():
        raise SystemExit(f"{KERNEL_CU} is UNPATCHED (no swizzle_v_sf): apply 0002 first")
    BUILD.mkdir(parents=True, exist_ok=True)
    load(
        name=NAME,
        sources=[str(KERNEL_CU), str(HERE / "overlay_binding.cpp")],
        # shadow dir first (-I beats torch's -isystem): vLLM's torch-2.13
        # stableivalue_conversions.h leak hotfix, if the Dockerfile installed it
        extra_include_paths=[
            d for d in [os.environ.get("VLLM_SM12X_SHADOW_INC", ""), str(CSRC)] if d and Path(d).exists()
        ],
        extra_cuda_cflags=[
            "-O3",
            # same defines vLLM's CMake gives _C_stable_libtorch: USE_CUDA unlocks the
            # cuda C-shim decls (aoti_torch_get_current_cuda_stream etc.), TORCH_TARGET_VERSION
            # pins the stable-ABI surface to torch 2.11
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
            f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
            "-DNVFP4_ENABLE_ELTS16=1",
            "-DENABLE_NVFP4_SM120=1",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-O3", "-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL"],
        build_directory=str(BUILD),
        is_python_module=False,  # registers torch.ops.vllm_sm12x.* on load
        verbose=bool(int(os.environ.get("VLLM_SM12X_VERBOSE", "1"))),
    )
    so = BUILD / f"{NAME}.so"
    assert so.exists(), f"build produced no {so}"
    return so


if __name__ == "__main__":
    so = build()
    import torch

    assert hasattr(torch.ops, "vllm_sm12x") and hasattr(
        torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4"
    ), "op not registered after build"
    print(f"OVERLAY-BUILT {so} arch=sm_{ARCH}")
    sys.exit(0)
