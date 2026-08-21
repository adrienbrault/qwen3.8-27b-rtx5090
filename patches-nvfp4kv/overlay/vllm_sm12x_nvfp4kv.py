"""Runtime loader for the SM12x NVFP4 KV linear-V-scale store overlay.

Imported lazily by the patched vLLM FlashInfer backend (0002b) when
use_fa2_nvfp4_kv is set. Loads the AOT-built .so (build_overlay.py) that
registers torch.ops.vllm_sm12x.reshape_and_cache_nvfp4; falls back to a JIT
build if the .so is missing (first-boot cost ~1-2 min, cached in
TORCH_EXTENSIONS_DIR / VLLM_SM12X_BUILD -- mount it).

Exposes reshape_and_cache_nvfp4_linear(...) with the exact argument order of
torch.ops._C_cache_ops.reshape_and_cache_flash.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_SO = Path(os.environ.get("VLLM_SM12X_NVFP4KV_SO", "/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so"))


def _ensure_loaded() -> None:
    if hasattr(torch.ops, "vllm_sm12x") and hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4"):
        return
    if _SO.exists():
        torch.ops.load_library(str(_SO))
    else:
        # JIT fallback (same sources; needs nvcc + csrc in the image)
        from build_overlay import build  # sibling module on PYTHONPATH

        build()
    assert hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4"), "overlay op missing after load"


_ensure_loaded()

_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else -1
if _major != 12:
    raise RuntimeError(
        f"vllm_sm12x_nvfp4kv overlay loaded on device major {_major}; it is only "
        "meaningful on SM12x (the writer emits linear V scales only for major >= 12)."
    )


def reshape_and_cache_nvfp4_linear(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops.vllm_sm12x.reshape_and_cache_nvfp4(
        key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )
