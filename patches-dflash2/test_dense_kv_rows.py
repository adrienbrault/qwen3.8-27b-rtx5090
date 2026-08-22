import torch, types
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
torch.manual_seed(0)
for bits in (4, 8):
    out_f, in_f, group, q_size = 4096, 5120, 128, 2048
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = torch.randint(lo, hi + 1, (out_f, in_f), dtype=torch.int8)
    scale = (torch.rand(out_f, in_f // group) * 0.02 + 0.001).to(torch.bfloat16)
    packed = pack_to_int32(q, bits, packed_dim=1)
    qkv = types.SimpleNamespace(weight_packed=torch.nn.Parameter(packed, requires_grad=False), weight_scale=scale, input_size=in_f)
    attn = types.SimpleNamespace(qkv_proj=qkv, q_size=q_size)
    got = DFlashQwen3Model._dense_kv_rows(attn)
    ref = (q.float().reshape(out_f, in_f // group, group) * scale.float()[..., None]).reshape(out_f, in_f).to(torch.bfloat16)[q_size:]
    assert got.shape == ref.shape and got.dtype == torch.bfloat16, (got.shape, got.dtype)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"W{bits}A16 dequant round-trip: shape {tuple(got.shape)} max|err|={err} -> {'OK' if err == 0 else 'FAIL'}")
    assert err == 0
# dense path unchanged
attn = types.SimpleNamespace(qkv_proj=types.SimpleNamespace(weight=torch.randn(64, 32)), q_size=16)
assert torch.equal(DFlashQwen3Model._dense_kv_rows(attn), attn.qkv_proj.weight[16:]); print("dense path OK")
