"""Build on the image's existing torch/CUDA toolchain, without a GPU or pip.

Image build: python3 -m pcie_ipc_ar21.build --output /path/to/pcie_ipc_ar21/_C.so
Serving: load the baked library, or compile lazily if it is absent.
"""

import argparse
import functools
import hashlib
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CUDA_FLAGS = [
    "-O3",
    "-std=c++20",
    "-use_fast_math",
    "-DNDEBUG",
    "-Xfatbin=-compress-all",
    "--compress-mode=size",
    "-gencode=arch=compute_120a,code=sm_120a",
]


def build_library(build_root: Path) -> Path:
    import torch
    from torch.utils.cpp_extension import load

    # Explicit -gencode prevents cpp_extension from enumerating visible GPUs.
    # Separate caches by sources, torch ABI version, CUDA and compile flags.
    digest = hashlib.sha256()
    for source in (ROOT / "csrc/bindings.cu",
                   ROOT / "include/flashinfer/comm/pcie_ipc_all_reduce.cuh"):
        digest.update(source.read_bytes())
    digest.update(repr((torch.__version__, torch.version.cuda, CUDA_FLAGS)).encode())
    directory = build_root / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    library = load(
        name="pcie_ipc_ar21_native",
        sources=[str(ROOT / "csrc/bindings.cu")],
        extra_include_paths=[str(ROOT / "include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=CUDA_FLAGS,
        build_directory=str(directory),
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    # is_python_module=False registers TORCH_LIBRARY through load_library;
    # loading has no CUDA allocation, device query or collective side effects.
    return Path(library)


@functools.cache
def load_native():
    import torch

    baked = ROOT / "_C.so"
    if baked.is_file():
        torch.ops.load_library(str(baked))
    else:
        cache = Path(os.environ.get("PCIE_IPC_AR21_BUILD_DIR", "/tmp/pcie-ipc-ar21-build"))
        build_library(cache)

    @torch.library.register_fake("pcie_ipc_ar21_native::all_reduce")
    def fake_all_reduce(handle, inp, blocks, threads, variant):
        return torch.empty_like(inp)

    return torch.ops.pcie_ipc_ar21_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=ROOT / "_build")
    args = parser.parse_args()
    library = build_library(args.build_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, args.output)


if __name__ == "__main__":
    main()
