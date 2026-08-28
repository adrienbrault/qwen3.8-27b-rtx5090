#!/usr/bin/env python3
"""R104g: patch qwen3_dflash.py so DFlash2's fused context-KV precompute works with quantized
drafts (compressed-tensors pack-quantized has no dense .weight). Reads .orig, writes patched."""
src = open("/srv/qwen5090/patches-v0280/qwen3_dflash.py.orig").read()
old = "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]"
assert old in src, "anchor not found"
new = '''        def _dense_w(lin):
            if hasattr(lin, "weight"):
                return lin.weight
            # Quantized draft (e.g. compressed-tensors pack-quantized): no dense .weight
            # param exists, so materialize it exactly through the layer's own quant
            # kernel -- eye(in) @ W^T = W^T -- once at load time.
            eye = torch.eye(
                lin.input_size,
                dtype=self.hidden_norm.weight.dtype,
                device=self.hidden_norm.weight.device,
            )
            return lin.quant_method.apply(lin, eye, bias=None).t().contiguous()

        kv_weights = [_dense_w(a.qkv_proj)[a.q_size :] for a in layers_attn]'''
src = src.replace(old, new)

old2 = """        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()"""
assert old2 in src, "anchor2 not found"
new2 = """        loader.load_weights(model_weights.items())
        try:
            self.model._build_fused_kv_buffers()
        except AttributeError:
            # Quantized draft: the quant kernel can't run until
            # process_weights_after_loading repacks the layers (e.g. Marlin's
            # g_idx_sort_indices). The lazy rebuild in
            # precompute_and_store_context_kv fires after that and succeeds.
            pass"""
src = src.replace(old2, new2)

open("/srv/qwen5090/patches-v0280/qwen3_dflash.py", "w").write(src)
print("patched ok")
