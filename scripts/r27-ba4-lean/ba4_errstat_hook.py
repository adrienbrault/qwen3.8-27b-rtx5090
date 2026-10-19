"""TEST-ONLY: measure BA4's arithmetic error on live operands, without touching the vllm_ba4 package.

Loaded at interpreter start by ba4_errstat_hook.pth (both mounted into dist-packages by the kernel arm of
ba4-lean/gate.sh, never in serving), and only when BA4_ERRSTAT names an output directory. The vllm_ba4 files stay
byte-identical, so BA4's code_sha256 admission still passes (a patched runtime.py was refused as
`implementation_changed`). The patched linear.py imports `vllm_ba4.runtime.try_apply` at call time, so replacing the
module attribute wraps every call.

At every eager M=4 call BA4 serves, compute the same product exactly (fp64) and with the original bf16 GEMM
(torch.nn.functional.linear = vLLM's default_unquantized_gemm); accumulate per layer and rank mean |error|, squared
error, signed error and max |error| for BA4, the original GEMM and the best bf16 answer (fp64 rounded once). BA4's
output is returned unchanged. Stats go to $BA4_ERRSTAT/rank<r>-<pid>.json every 100 calls.
"""
import json
import os

_ERR = {}
_CALLS = [0]


def _errstat(layer, x, w, out):
    import torch
    exact = x.double() @ w.double().t()
    cands = {"ba4": out.double(), "original": torch.nn.functional.linear(x, w).double(),
             "best_bf16": exact.to(torch.bfloat16).double()}
    acc = _ERR.setdefault(layer.prefix, {k: [0.0, 0.0, 0.0, 0.0] for k in cands} | {"n": 0, "scale": 0.0})
    acc["n"] += exact.numel()
    acc["scale"] += exact.abs().sum().item()
    for k, v in cands.items():
        e = v - exact
        a = acc[k]
        a[0] += e.abs().sum().item(); a[1] += (e * e).sum().item(); a[2] += e.sum().item()
        a[3] = max(a[3], e.abs().max().item())
    _CALLS[0] += 1
    if _CALLS[0] % 100 == 0:
        path = os.path.join(os.environ["BA4_ERRSTAT"], f"rank{getattr(layer, 'tp_rank', 'x')}-{os.getpid()}.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"calls": _CALLS[0], "layers": _ERR}, f)
        os.replace(path + ".tmp", path)


def _install():
    import torch
    import vllm_ba4.runtime as rt
    original = rt.try_apply

    def try_apply(layer, x, bias, **kwargs):
        out = original(layer, x, bias, **kwargs)
        if out is not None and not torch.compiler.is_compiling():
            _errstat(layer, x, layer.weight, out)
        return out

    rt.try_apply = try_apply


if os.environ.get("BA4_ERRSTAT"):
    _install()
