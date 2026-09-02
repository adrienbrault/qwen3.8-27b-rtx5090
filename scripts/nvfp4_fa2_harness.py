#!/usr/bin/env python3
"""R158 — standalone differential harness for the SM12x NVFP4-KV FlashInfer FA2 paged reader.

Bug A (R155): DFlash2 draft at draft_tp=2 (rank-local H=4, head_dim 128, 32 tokens/page) reads
corrupt KV through FA2-over-nvfp4, while the target (H=2, hd 256, page 32) and the TP1 draft
(H=8, hd 128, page 16) are clean. Three axes co-vary in the deployed shapes, so this sweeps them
INDEPENDENTLY, outside vLLM, against a float32 reference computed on the dequantized cache:

  write:  torch.ops.vllm_sm12x.reshape_and_cache_nvfp4  (the deployed overlay writer, linear V-SF)
  read:   flashinfer BatchPrefillWithPagedKVCacheWrapper(backend="fa2", "HND"), kv_data_type=uint8,
          kv_cache_sf=(k_sf, v_sf)  — exactly the vLLM FA2-nvfp4 route (flashinfer.py plan/run kwargs)
  ref:    dequant = e2m1(nibble) * float(sf_e4m3) * global_scale  (nvfp4_kv_cache_kernels.cu:
          outputScale = (1/k_scale)/sf, so q = x/(k_scale*sf))

Runs in the revival image on ONE free GPU (needs the flashinfer JIT cache mount or the fp4 FA2
module costs minutes). Prints one row per cell; a cell is RED when its error is >10x the noise
floor of the clean cells. Exit code 1 if any cell is red.

  sudo docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v /srv/qwen5090/cache/flashinfer:/root/.cache/flashinfer -v /srv/qwen5090/probes:/probes:ro \
    vllm-qwen38:v0280-nvfp4kv-revival /probes/nvfp4_fa2_harness.py --out /tmp/harness.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def dequant_side(data: torch.Tensor, sf: torch.Tensor, global_scale: float) -> torch.Tensor:
    """data: (..., D/2) uint8 packed fp4 (low nibble = even element); sf: (..., D/16) e4m3.
    Returns (..., D) float32."""
    lo = (data & 0x0F).to(torch.int64)
    hi = (data >> 4).to(torch.int64)
    table = E2M1.to(data.device)

    def dec(n):
        mag = table[n & 0x7]
        sign = torch.where((n & 0x8) != 0, -1.0, 1.0)
        return mag * sign

    vals = torch.stack([dec(lo), dec(hi)], dim=-1).flatten(-2)  # (..., D) interleaved even/odd
    sfv = sf.to(torch.float32).repeat_interleave(16, dim=-1)
    return vals * sfv * global_scale


def reference_attention(q, k, v, qo_lens, kv_lens, gqa, causal, sm_scale):
    """q: (sum_q, Hq, D) bf16; k, v: (sum_kv, Hkv, D) float32 (dequantized, in request order).
    Queries are the LAST qo_len positions of each request (vLLM verify/append semantics)."""
    outs = []
    q0 = k0 = 0
    for ql, kl in zip(qo_lens, kv_lens):
        qi = q[q0:q0 + ql].float()  # (ql, Hq, D)
        ki = k[k0:k0 + kl]  # (kl, Hkv, D)
        vi = v[k0:k0 + kl]
        ki = ki.repeat_interleave(gqa, dim=1)  # (kl, Hq, D)
        vi = vi.repeat_interleave(gqa, dim=1)
        s = torch.einsum("qhd,khd->hqk", qi, ki) * sm_scale
        if causal:
            # query j (0..ql-1) may see keys <= kl - ql + j
            qpos = torch.arange(ql, device=q.device)[:, None] + (kl - ql)
            kpos = torch.arange(kl, device=q.device)[None, :]
            s = s.masked_fill((kpos > qpos)[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vi))
        q0 += ql
        k0 += kl
    return torch.cat(outs, dim=0)


def run_cell(hd, H, page, gqa, causal, q_len, kv_len, batch, k_scale, v_scale, seed, wrapper_cache, dev):
    torch.manual_seed(seed)
    Hq = H * gqa
    full_dim = hd // 2 + hd // 16
    kv_lens = [kv_len + 7 * i for i in range(batch)]  # ragged, not page-aligned
    qo_lens = [q_len] * batch
    total_kv = sum(kv_lens)
    pages_per_req = [math.ceil(l / page) for l in kv_lens]
    num_pages = sum(pages_per_req) + 3  # slack pages (never referenced)
    # random page assignment so block-table indirection is exercised
    perm = torch.randperm(num_pages - 1, device="cpu") + 1  # page 0 left unused (poison check)
    kv_indices, slot_mapping, kv_indptr, last_page_len = [], [], [0], []
    pi = 0
    for l, npg in zip(kv_lens, pages_per_req):
        pages = perm[pi:pi + npg].tolist()
        pi += npg
        kv_indices += pages
        kv_indptr.append(len(kv_indices))
        last_page_len.append(l - (npg - 1) * page)
        for t in range(l):
            slot_mapping.append(pages[t // page] * page + (t % page))
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=dev)
    last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=dev)
    qo_indptr = torch.tensor([0] + list(itertools.accumulate(qo_lens)), dtype=torch.int32, device=dev)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device=dev)

    key = torch.randn(total_kv, H, hd, device=dev, dtype=torch.bfloat16)
    value = torch.randn(total_kv, H, hd, device=dev, dtype=torch.bfloat16)
    q = torch.randn(sum(qo_lens), Hq, hd, device=dev, dtype=torch.bfloat16)

    # vLLM-shaped cache: (B, 2H, N, full_dim) uint8, HND physical order. Poison so unwritten bytes are loud.
    kv_cache = torch.full((num_pages, 2 * H, page, full_dim), 0x77, dtype=torch.uint8, device=dev)
    k_view, v_view = kv_cache.transpose(1, 2).split(H, dim=-2)  # (B, N, H, full_dim) logical
    ks = torch.tensor([k_scale], dtype=torch.float32, device=dev)
    vs = torch.tensor([v_scale], dtype=torch.float32, device=dev)
    torch.ops.vllm_sm12x.reshape_and_cache_nvfp4(key, value, k_view, v_view, slot_mapping, "nvfp4", ks, vs)

    # Reader views, exactly as vllm/v1/attention/backends/flashinfer.py builds them.
    from vllm.utils.torch_utils import nvfp4_split_data_scale
    k_side, v_side = kv_cache.split(H, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)

    # Dequantized reference, gathered back into request order via slot_mapping.
    k_deq_pages = dequant_side(k_data, k_sf, k_scale)  # (B, H, N, D)
    v_deq_pages = dequant_side(v_data, v_sf, v_scale)
    flat_k = k_deq_pages.permute(0, 2, 1, 3).reshape(num_pages * page, H, hd)
    flat_v = v_deq_pages.permute(0, 2, 1, 3).reshape(num_pages * page, H, hd)
    k_ref = flat_k[slot_mapping]
    v_ref = flat_v[slot_mapping]
    sm_scale = 1.0 / math.sqrt(hd)
    ref = reference_attention(q, k_ref, v_ref, qo_lens, kv_lens, gqa, causal, sm_scale)

    # FA2 route (the plan/run kwargs vLLM uses for use_fa2_nvfp4_kv).
    import flashinfer
    key_w = (hd, causal)
    if key_w not in wrapper_cache:
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        wrapper_cache[key_w] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
    w = wrapper_cache[key_w]
    w.plan(
        qo_indptr, kv_indptr, kv_indices, last_page_len,
        num_qo_heads=Hq, num_kv_heads=H, head_dim_qk=hd, page_size=page,
        causal=causal, sm_scale=sm_scale, window_left=-1, logits_soft_cap=None,
        q_data_type=torch.bfloat16, kv_data_type=torch.uint8, o_data_type=torch.bfloat16,
    )
    out = torch.empty(sum(qo_lens), Hq, hd, dtype=torch.bfloat16, device=dev)
    w.run(q, (k_data, v_data), k_scale=k_scale, v_scale=v_scale, out=out, kv_cache_sf=(k_sf, v_sf))
    torch.cuda.synchronize()
    err = (out.float() - ref).abs()
    denom = ref.abs().mean().item() + 1e-6
    # per-head worst so a single bad head does not average away
    per_head = err.mean(dim=(0, 2)) / denom
    return {
        "max_abs": err.max().item(), "mean_rel": (err.mean().item() / denom),
        "worst_head_rel": per_head.max().item(), "worst_head": int(per_head.argmax().item()),
        "nan": bool(torch.isnan(out).any().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="only the three deployed shapes")
    ap.add_argument("--kv", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    import vllm_sm12x_nvfp4kv  # noqa: F401  (registers torch.ops.vllm_sm12x on SM12x)

    deployed = [  # (hd, H, page, gqa, tag)
        (256, 2, 32, 6, "target-tp2"),
        (128, 4, 32, 4, "draft-tp2 (Bug A)"),
        (128, 8, 16, 4, "draft-tp1"),
    ]
    cells = []
    if args.quick:
        for hd, H, page, gqa, tag in deployed:
            for causal in (False, True):
                cells.append(dict(hd=hd, H=H, page=page, gqa=gqa, causal=causal, q_len=8, batch=2, tag=tag))
    else:
        for hd, H, page, gqa, causal, q_len, batch in itertools.product(
            (128, 256), (2, 4, 8), (16, 32, 64), (4, 6), (False, True), (8, 16), (1, 2)
        ):
            tag = next((t for h2, H2, p2, g2, t in deployed if (h2, H2, p2, g2) == (hd, H, page, gqa)), "")
            cells.append(dict(hd=hd, H=H, page=page, gqa=gqa, causal=causal, q_len=q_len, batch=batch, tag=tag))
    wrapper_cache = {}
    rows = []
    fout = open(args.out, "w") if args.out else None
    t0 = time.time()
    for i, c in enumerate(cells):
        for k_scale, v_scale in ((1.0, 1.0), (0.7, 1.3)):
            try:
                r = run_cell(c["hd"], c["H"], c["page"], c["gqa"], c["causal"], c["q_len"], args.kv, c["batch"],
                             k_scale, v_scale, seed=1234 + i, wrapper_cache=wrapper_cache, dev=dev)
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:200]}
            row = dict(c, k_scale=k_scale, v_scale=v_scale, **r)
            rows.append(row)
            if fout:
                fout.write(json.dumps(row) + "\n"); fout.flush()
    # noise floor = median mean_rel over cells; red = > 10x floor or nan/error
    ok = [r["mean_rel"] for r in rows if "mean_rel" in r and not r["nan"]]
    floor = sorted(ok)[len(ok) // 2] if ok else float("nan")
    red = 0
    print(f"cells={len(rows)} floor(median mean_rel)={floor:.4f} elapsed={time.time()-t0:.0f}s")
    print(f"{'hd':>4} {'H':>2} {'pg':>3} {'gqa':>3} {'caus':>5} {'q':>3} {'b':>2} {'ks':>4} {'mean_rel':>9} {'worst_head':>11} {'max_abs':>8}  tag")
    for r in rows:
        if "error" in r:
            red += 1
            print(f"{r['hd']:>4} {r['H']:>2} {r['page']:>3} {r['gqa']:>3} {str(r['causal']):>5} {r['q_len']:>3} {r['batch']:>2} {r['k_scale']:>4}  ERROR {r['error']}  {r['tag']}")
            continue
        bad = r["nan"] or r["mean_rel"] > 10 * floor or r["worst_head_rel"] > 10 * floor
        red += bad
        print(f"{r['hd']:>4} {r['H']:>2} {r['page']:>3} {r['gqa']:>3} {str(r['causal']):>5} {r['q_len']:>3} {r['batch']:>2} {r['k_scale']:>4} "
              f"{r['mean_rel']:>9.4f} {r['worst_head_rel']:>7.4f}(h{r['worst_head']:<2}) {r['max_abs']:>8.3f}  {'RED ' if bad else 'ok  '}{r['tag']}")
    print(f"SUMMARY red={red} of {len(rows)} floor={floor:.4f}")
    sys.exit(1 if red else 0)


if __name__ == "__main__":
    main()
