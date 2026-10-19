"""Complete out-buffer paths. No packing, atomics, hot-path allocation or sync.

Triton paths multiply BF16 inputs as FP32, reduce K in FP32, and cast once.
The split variants include BOTH launches and their FP32 partial-sum traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _direct(X, W, Y, XS: tl.constexpr, M: tl.constexpr,
            BK: tl.constexpr):
    n, m = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, BK)
    x = tl.load(X + m * XS + k, k < 5120, 0).to(tl.float32)
    w = tl.load(W + n * 5120 + k, k < 5120, 0).to(tl.float32)
    tl.store(Y + m * 48 + n, tl.sum(x * w, 0))


@triton.jit
def _partial(X, W, P, XS: tl.constexpr, M: tl.constexpr,
             BK: tl.constexpr, BN: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    s, m = tl.program_id(1), tl.program_id(2)
    k = s * BK + tl.arange(0, BK)
    x = tl.load(X + m * XS + k, k < 5120, 0).to(tl.float32)
    w = tl.load(W + ns[:, None] * 5120 + k[None, :],
                (ns[:, None] < 48) & (k[None, :] < 5120), 0).to(tl.float32)
    sums = tl.sum(w * x[None, :], 1)
    tl.store(P + (s * M + m) * 48 + ns, sums, ns < 48)


@triton.jit
def _finish(P, Y, M: tl.constexpr, SPLITS: tl.constexpr,
            BS: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1)
    s = tl.arange(0, BS)
    p = tl.load(P + (s[:, None] * M + m) * 48 + n[None, :],
                (s[:, None] < SPLITS) & (n[None, :] < 48), 0)
    tl.store(Y + m * 48 + n, tl.sum(p, 0), n < 48)


@triton.jit
def _stream_weights(W, S, SIZE: tl.constexpr, B: tl.constexpr):
    """Read every byte of a large representative target weight stream."""
    b = tl.program_id(0)
    i = b * B + tl.arange(0, B)
    v = tl.load(W + i, i < SIZE, 0, cache_modifier=".cg").to(tl.int32)
    tl.store(S + b, tl.sum(v, 0))


def stream_weights(weight_bytes, sink):
    _stream_weights[(triton.cdiv(weight_bytes.numel(), 4096),)](
        weight_bytes, sink, weight_bytes.numel(), 4096, num_warps=4)


def run_out(name, x, weight, out, partial):
    """All temporaries/views below are host metadata; device storage is fixed."""
    if name == "torch_mm":
        torch.mm(x, weight.t(), out=out)
    elif name == "torch_rows_mm":
        # M=1 cuBLAS route; include all M calls, not a single-row extrapolation.
        for m in range(x.shape[0]):
            torch.mm(x[m:m + 1], weight.t(), out=out[m:m + 1])
    elif name == "torch_rows_mv":
        # Some cuBLAS versions reject BF16 GEMV. Benchmark records rejection.
        for m in range(x.shape[0]):
            torch.mv(weight, x[m], out=out[m])
    elif name == "triton_direct":
        _direct[(48, x.shape[0])](x, weight, out, x.stride(0), x.shape[0],
                                    8192, num_warps=4, enable_fp_fusion=False)
    elif name.startswith("triton_split"):
        bk = int(name.removeprefix("triton_split"))
        if bk not in (128, 256, 512):
            raise ValueError(name)
        splits = triton.cdiv(5120, bk)
        _partial[(12, splits, x.shape[0])](
            x, weight, partial, x.stride(0), x.shape[0], bk, 4,
            num_warps=4, enable_fp_fusion=False)
        _finish[(3, x.shape[0])](partial, out, x.shape[0], splits,
                                triton.next_power_of_2(splits), 16, num_warps=4)
    else:
        raise ValueError(name)
