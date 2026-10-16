"""Numeric diagnostic for the SM12x NVFP4-KV V-scale layout question (FINDINGS R77).

Behavioural probes (depth needles to 100K, cold/warm, under load) PASS with BOTH writers, so
they cannot tell whether the in-tree swizzled-V-scale writer actually mismatches the FlashInfer
FA2 linear scale reader on sm120. This measures it directly:

  same random K/V  --in-tree writer-->  cache A (V scales swizzled, SM100 trtllm-gen pattern)
                   --overlay writer-->  cache B (V scales linear)
  FA2 paged prefill (kv_data_type=uint8 + block scales) on A and on B
  vs. a bf16/fp32 causal GQA reference.

Expected if the overlay is REQUIRED:  err(B) ~ fp4 quantisation noise,  err(A) >> err(B).
Expected if the swizzle is a no-op / reader handles it:  err(A) == err(B) (and/or A == B bytewise).

Run inside the nvfp4kv image on the 5090 (needs ~0.5 GB):
  docker run --rm --runtime nvidia --gpus all --entrypoint python3 \
    -v /srv/qwen5090/patches-nvfp4kv-overlay:/diag:ro vllm-qwen38:nvfp4kv /diag/diag_vsf_layout.py
"""

from __future__ import annotations

import math
import sys

import torch

torch.manual_seed(0)
dev = torch.device("cuda")
assert torch.cuda.get_device_properties(0).major == 12, "sm12x only"

import vllm._custom_ops  # noqa: F401,E402  (loads the compiled ops incl. _C_cache_ops)
from flashinfer import BatchPrefillWithPagedKVCacheWrapper  # noqa: E402
from vllm.utils.torch_utils import nvfp4_split_data_scale  # noqa: E402
from vllm_sm12x_nvfp4kv import reshape_and_cache_nvfp4_linear  # noqa: E402

try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import nvfp4_kv_cache_full_dim
except ImportError:  # name moved? fall back to the layout formula: data D/2 bytes + D/16 scale bytes
    def nvfp4_kv_cache_full_dim(head_size: int) -> int:
        return head_size // 2 + head_size // 16

H_KV, H_Q, D, BS = 4, 8, 256, int(sys.argv[1]) if len(sys.argv) > 1 else 16
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2  # per-16-group V magnitude spread (lognormal sigma); 0 = flat
T = 512  # one sequence, T tokens, causal
NB = T // BS
FULL = nvfp4_kv_cache_full_dim(D)
print(f"config: H_kv={H_KV} H_q={H_Q} D={D} block_size={BS} T={T} full_dim={FULL} group_sigma={SIGMA}")

# K/V with per-token, per-16-group magnitude structure so that a scale permutation is visible.
g = torch.Generator(device=dev).manual_seed(0)
key = torch.randn(T, H_KV, D, device=dev, generator=g, dtype=torch.float32)
value = torch.randn(T, H_KV, D, device=dev, generator=g, dtype=torch.float32)
mag = torch.exp(torch.randn(T, H_KV, D // 16, device=dev, generator=g) * SIGMA)  # lognormal group gains
value = (value.view(T, H_KV, D // 16, 16) * mag[..., None]).view(T, H_KV, D)
key = key.to(torch.bfloat16)
value = value.to(torch.bfloat16)
query = torch.randn(T, H_Q, D, device=dev, generator=g).to(torch.bfloat16)
slot_mapping = torch.arange(T, device=dev, dtype=torch.int64)
one = torch.ones((), device=dev, dtype=torch.float32)


def fresh_cache() -> torch.Tensor:
    return torch.zeros(NB, 2 * H_KV, BS, FULL, device=dev, dtype=torch.uint8)


def store(cache: torch.Tensor, linear: bool) -> None:
    # exactly what FlashInferImpl.forward does before the store op
    k_cache, v_cache = cache.transpose(1, 2).split(H_KV, dim=-2)
    if linear:
        reshape_and_cache_nvfp4_linear(key, value, k_cache, v_cache, slot_mapping, "nvfp4", one, one)
    else:
        torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, "nvfp4", one, one)
    torch.cuda.synchronize()


def fa2_prefill(cache: torch.Tensor) -> torch.Tensor:
    k_side, v_side = cache.split(H_KV, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)
    ws = torch.empty(128 * 1024 * 1024, device=dev, dtype=torch.uint8)
    w = BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
    w.plan(
        qo_indptr=torch.tensor([0, T], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, NB], dtype=torch.int32),
        paged_kv_indices=torch.arange(NB, dtype=torch.int32, device=dev),
        paged_kv_last_page_len=torch.tensor([BS], dtype=torch.int32),
        num_qo_heads=H_Q,
        num_kv_heads=H_KV,
        head_dim_qk=D,
        page_size=BS,
        causal=True,
        sm_scale=1.0 / math.sqrt(D),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = w.run(query, (k_data, v_data), kv_cache_sf=(k_sf, v_sf))
    torch.cuda.synchronize()
    return out.float()


def reference() -> torch.Tensor:
    rep = H_Q // H_KV
    q = query.float().transpose(0, 1)  # (Hq, T, D)
    k = key.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    v = value.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    s = (q @ k.transpose(1, 2)) / math.sqrt(D)
    mask = torch.ones(T, T, device=dev, dtype=torch.bool).tril()
    s = s.masked_fill(~mask, float("-inf")).softmax(-1)
    return (s @ v).transpose(0, 1)  # (T, Hq, D)


ref = reference()
A = fresh_cache(); store(A, linear=False)
B = fresh_cache(); store(B, linear=True)

kA, vA = A.split(H_KV, dim=1); kB, vB = B.split(H_KV, dim=1)
_, kA_sf = nvfp4_split_data_scale(kA); _, kB_sf = nvfp4_split_data_scale(kB)
vA_d, vA_sf = nvfp4_split_data_scale(vA); vB_d, vB_sf = nvfp4_split_data_scale(vB)
print(f"bytes equal: K-scales={torch.equal(kA_sf, kB_sf)}  V-data={torch.equal(vA_d, vB_d)}  V-scales={torch.equal(vA_sf, vB_sf)}"
      f"  (V-scale bytes differing: {(vA_sf != vB_sf).sum().item()} / {vA_sf.numel()})")

def err(o: torch.Tensor) -> tuple[float, float]:
    e = (o - ref)
    return (e.norm() / ref.norm()).item(), (e.abs().max() / ref.abs().max()).item()

outA = fa2_prefill(A); outB = fa2_prefill(B)
rA, mA = err(outA); rB, mB = err(outB)
print(f"FA2 on in-tree(swizzled V-scales) cache A: rel_l2={rA:.4f} rel_max={mA:.4f}")
print(f"FA2 on overlay (linear  V-scales) cache B: rel_l2={rB:.4f} rel_max={mB:.4f}")
ratio = rA / max(rB, 1e-9)
verdict = ("OVERLAY REQUIRED (swizzled writer mismatches the FA2 reader)" if ratio > 2
           else "NO MEASURABLE DIFFERENCE (overlay not needed on this stack)" if 0.5 < ratio < 2
           else "UNEXPECTED: overlay WORSE than in-tree — re-derive")
print(f"VERDICT: {verdict}  (errA/errB = {ratio:.2f})")
