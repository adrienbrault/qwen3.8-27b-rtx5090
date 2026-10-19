"""Shared fingerprint and byte-identity checks; used outside timed paths."""

import hashlib
from pathlib import Path


def code_sha256():
    h = hashlib.sha256()
    for name in ("__init__.py", "policy.py", "common.py", "kernels.py", "runtime.py",
                 "measurement.py", "evidence.py", "benchmark_source.py", "compilation.py"):
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def benchmark_sha256():
    return hashlib.sha256((Path(__file__).parent / "benchmark_source.py").read_bytes()).hexdigest()


def tensor_sha256(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(
        torch.uint8).numpy().tobytes()).hexdigest()


def fingerprint(device):
    import os
    import torch
    import triton
    props = torch.cuda.get_device_properties(device)
    return {"device_name": torch.cuda.get_device_name(device),
            "device_uuid": str(getattr(props, "uuid", "unavailable")),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "triton": str(triton.__version__),
            "blas_preference": str(torch.backends.cuda.preferred_blas_library()),
            "bf16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG", "")}
