"""Per-layer scratch/static outputs are prepared before compile/graph capture.

Symbolic M dispatch returns fresh outputs owned by the caller/CUDA graph.

The serving runner must order uses of a layer on its compute stream, as for
its other persistent model buffers. The BA result is consumed inside the
same forward. Independently concurrent forwards on one model are unsupported.
"""

import functools
import json
import logging
import os
import sys
from dataclasses import dataclass

import torch

from .common import benchmark_sha256, code_sha256, fingerprint, tensor_sha256
from .kernels import run_out
from .policy import SUFFIX, admission_reason, input_reason, valid_x_stride, weight_reason

# init_logger preserves its argument; the external package name vllm_ba4 is
# not a child of vllm. Use vLLM's configured worker handler/formatter rather
# than Python's process-local root/lastResort stderr handlers. WorkerProc
# supplies the process tag; never synthesize one from our rank/PID fields.
logger = logging.getLogger("vllm." + __name__)
if "vllm.logger" in sys.modules:
    from vllm.logger import init_logger

    logger = init_logger("vllm." + __name__)
_used = set()


@torch.library.custom_op("vllm::sm12x_ba4_offset_metadata", mutates_args=())
def _offset_metadata(tensor: torch.Tensor) -> torch.Tensor:
    """Carry the live offset in a fresh zero-element tensor's stride.

    Dynamo evaluates this helper on FakeTensor metadata, without evaluating
    tensor producers. Its only FX result is a Tensor: unlike nonstrict_trace,
    no shared TreeSpec can cross vLLM's attention partition boundaries.
    An OpOverload is safe for Torch's AOT key and GraphPickler; an arbitrary
    allow_in_graph Python function is not. The fresh empty result owns zero
    storage bytes, reads/copies no data, and obeys custom_op's no-alias rule.
    Piecewise compilation removes only these dead probes after Dynamo reads
    their strides. The +1 represents every nonnegative offset positively.
    """
    return torch.empty_strided((0,), (tensor.storage_offset() + 1,),
                               dtype=tensor.dtype, device=tensor.device)


@_offset_metadata.register_fake
def _offset_metadata_fake(tensor):
    return torch.empty_strided((0,), (tensor.storage_offset() + 1,),
                               dtype=tensor.dtype, device=tensor.device)


def _storage_offset(tensor):
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sm12x_ba4_offset_metadata.default(tensor).stride(0) - 1
    return tensor.storage_offset()


@torch.library.custom_op("vllm::sm12x_ba4_out",
                         mutates_args=("out", "partial"), device_types="cuda")
def ba4_out(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
            partial: torch.Tensor, name: str, prefix: str,
            x_stride: list[int] | None = None) -> None:
    # Python runs at warmup/capture, never during CUDA graph replay. Fail
    # closed if guard-disabled compiled code receives newly invalid weights.
    # The operand is the current Parameter itself, never a saved .data alias.
    if (weight.requires_grad or tuple(weight.shape) != (48, 5120)
            or tuple(weight.stride()) != (5120, 1)
            or weight.storage_offset() != 0 or weight.dtype != torch.bfloat16
            or weight.device != out.device):
        raise RuntimeError("BA4 live weight metadata refused; re-trace before capture")
    # Empty prefix is the retained microbenchmark's direct bare-kernel API,
    # including informational non-M4 controls, never a try_apply fallback.
    # Serving always supplies its live suffix-validated layer prefix.
    if prefix and (not prefix.removeprefix("PROFILE_ONLY:").endswith(SUFFIX)
            or tuple(x.shape) != (4, 5120) or x.requires_grad
            or not valid_x_stride(x.stride())
            or (x_stride is not None and tuple(x.stride()) != tuple(x_stride))
            or x.storage_offset() % 8 or x.dtype != torch.bfloat16
            or x.device != out.device or tuple(out.shape) != (4, 48)):
        raise RuntimeError("BA4 live input refused; re-trace the original apply path")
    run_out(name, x, weight, out, partial)
    if prefix and prefix not in _used:
        _used.add(prefix)
        profile_only = prefix.startswith("PROFILE_ONLY:")
        display_prefix = prefix.removeprefix("PROFILE_ONLY:")
        logger.warning("SM12X BA4 %s: prefix=%s pid=%d kernel=%s "
                       "phase=warmup_or_capture; verify replay with PROFILE",
                       "PROFILE_USED" if profile_only else "USED", display_prefix, os.getpid(), name)


@ba4_out.register_fake
def _fake(x, weight, out, partial, name, prefix, x_stride=None):
    # Mutation-only op, no returned aliases and no fake/meta allocation.
    return None


@torch.library.custom_op("vllm::sm12x_ba4_dispatch",
                         mutates_args=("partial",), device_types="cuda")
def ba4_dispatch(x: torch.Tensor, weight: torch.Tensor, partial: torch.Tensor,
                 name: str, prefix: str, x_stride: list[int]) -> torch.Tensor:
    """Runtime M dispatch for the single symbolic backbone graph.

    This output never aliases an operand or a persistent BA4 output. M != 4
    uses exactly default_unquantized_gemm's F.linear(x, weight, None), with
    its own allocation/lifetime and no BA4 capacity or out= GEMM variant.
    M4 allocates at warmup/capture; CUDA graph replay reuses the graph-owned
    allocation without running Python/the allocator. Scratch stays per layer.
    """
    if x.ndim != 2 or x.shape[0] != 4:
        return torch.nn.functional.linear(x, weight, None)
    # partial is the persistent prepared-device anchor. A fresh output on
    # x.device cannot enforce the old prepared-output device refusals.
    # Check before allocation, selected work, or USED logging.
    if weight.device != partial.device:
        raise RuntimeError("BA4 live weight metadata refused; re-trace before capture")
    if x.device != partial.device:
        raise RuntimeError("BA4 live input refused; re-trace the original apply path")
    out = torch.empty((4, 48), dtype=x.dtype, device=x.device)
    # Keep all j41m live operand checks and the actual USED emission in one
    # place. A guard-disabled replay with invalid M4 operands fails closed.
    ba4_out(x, weight, out, partial, name, prefix, x_stride)
    return out


@ba4_dispatch.register_fake
def _dispatch_fake(x, weight, partial, name, prefix, x_stride):
    return x.new_empty((x.shape[0], 48))


@dataclass
class State:
    name: str
    x_stride: tuple
    weight: torch.Tensor
    out: torch.Tensor
    partial: torch.Tensor
    profile_only: bool = False
    prefix: str = ""
    buffer: torch.Tensor | None = None

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = self.out

    def matches_weight(self, weight):
        """Validate the live Parameter at eager calls and at every trace."""
        return (weight is self.weight and not weight.requires_grad
                and tuple(weight.shape) == (48, 5120)
                and tuple(weight.stride()) == (5120, 1)
                and _storage_offset(weight) == 0
                and weight.dtype == torch.bfloat16
                and weight.device == self.out.device)


@functools.lru_cache(maxsize=4)
def _read_report(path):
    with open(path) as f:
        return json.load(f)


def prepare_layer(layer, *, output_capacity=4):
    """Called only with the flag on. Never called from apply/capture."""
    prefix = getattr(layer, "prefix", "")
    if not prefix.endswith(SUFFIX):
        return
    layer._sm12x_ba4 = None

    def inactive(reason):
        logger.warning("SM12X BA4 INACTIVE: prefix=%s pid=%d reason=%s",
                       prefix, os.getpid(), reason)

    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        inactive("batch_invariant_precedence")
        return
    w = getattr(layer, "weight", None)
    if w is None:
        inactive("weight_unavailable")
        return
    if w.requires_grad:
        inactive("weight_requires_grad")
        return
    if getattr(layer, "bias", None) is not None:
        inactive("layer_bias")
        return
    capability = torch.cuda.get_device_capability(w.device) if w.is_cuda else ()
    reason = weight_reason(prefix, w.shape, w.stride(), w.dtype, w.device.type,
                           capability, getattr(layer, "tp_size", None),
                           getattr(layer, "output_sizes", []), w.storage_offset())
    if reason:
        inactive(reason)
        return
    if getattr(layer.quant_method._gemm_impl, "__name__", "") != "default_unquantized_gemm":
        inactive("nondefault_original_dispatch")
        return
    rank = layer.tp_rank
    path = os.environ.get("VLLM_SM12X_BA_M4_RESULT", "").replace("{rank}", str(rank))
    try:
        report = _read_report(path)
        profile_only = os.environ.get("VLLM_SM12X_BA_M4_PROFILE_ONLY", "0") == "1"
        reason = admission_reason(report, fingerprint(w.device), code_sha256(), rank,
                                  benchmark_sha256(), profile_only=profile_only)
        if reason:
            inactive(reason)
            return
        if not profile_only:
            binding = _read_report(os.environ.get("VLLM_SM12X_BA_M4_BINDING", ""))
            if (binding.get("binding") != report["serving_evidence"]["binding"]
                    or binding.get("fingerprints", [])[rank] != report["fingerprint"]):
                inactive("serving_launch_binding_changed")
                return
        entry = next((e for e in report["layers"] if e["prefix"] == prefix), None)
        if entry is None or tensor_sha256(w) != entry["weight_sha256"]:
            inactive("prefix_or_weight_identity_changed")
            return
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        inactive("result_unavailable_or_invalid:%s" % type(exc).__name__)
        return
    # Per-layer ownership, not a shared global scratch pool. No packing.
    capacity = output_capacity
    buffer = torch.empty((capacity, 48), dtype=w.dtype, device=w.device)
    out = buffer if capacity == 4 else buffer[:4]
    partial = torch.empty((40, 4, 48), dtype=torch.float32, device=w.device)
    state = State(report["selected"], tuple(entry["x_stride"]), w, out, partial,
                  profile_only, ("PROFILE_ONLY:" if profile_only else "") + prefix,
                  buffer)
    # Materialize the selected library/JIT path before any capture. Do not mark
    # USED until an actual matching model input passes try_apply.
    sample = torch.empty_strided((4, 5120), state.x_stride,
                                 dtype=w.dtype, device=w.device).zero_()
    run_out(state.name, sample, w, out, partial)
    layer._sm12x_ba4 = state
    logger.warning("SM12X BA4 %s: prefix=%s rank=%d pid=%d kernel=%s "
                   "M=4 N=48 K=5120 bf16 w_stride=(5120,1) x_stride=%s "
                   "result=%s stable_buffers=1", "PROFILE_ONLY" if profile_only else "ACTIVE",
                   prefix, rank, os.getpid(), state.name,
                   state.x_stride, path)


def try_apply(layer, x, bias, *, weight_offset=None, input_offset=None):
    state = getattr(layer, "_sm12x_ba4", None)
    if state is None:
        return None
    w = layer.weight
    if (not getattr(layer, "prefix", "").endswith(SUFFIX)
            or not state.matches_weight(w)):
        return None
    if weight_offset is not None and weight_offset != _storage_offset(w):
        return None
    symbolic_m = False
    if torch.compiler.is_compiling():
        from torch.fx.experimental.symbolic_shapes import statically_known_true
        if x.ndim != 2:
            return None
        if statically_known_true(x.shape[0] != 4):
            return None
        symbolic_m = not statically_known_true(x.shape[0] == 4)
    # Offsets supplied by the fixed-M replay harness are additional checks,
    # never substitutes for observing this call's actual tensor metadata.
    actual_input_offset = _storage_offset(x)
    if input_offset is not None and input_offset != actual_input_offset:
        return None
    shape = (4, x.shape[1]) if symbolic_m else x.shape
    if input_reason(shape, x.stride(), x.dtype, x.device, state.out.device,
                    state.x_stride, bias, x.requires_grad,
                    actual_input_offset):
        return None
    prefix = ("PROFILE_ONLY:" if state.profile_only else "") + layer.prefix
    if symbolic_m:
        return ba4_dispatch(x, w, state.partial, state.name, prefix, list(state.x_stride))
    ba4_out(x, w, state.out, state.partial, state.name, prefix, list(state.x_stride))
    return state.out
