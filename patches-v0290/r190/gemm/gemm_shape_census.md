# Served TP2 linear census

Source-audited static constructor expansion: qwen3_5.py → qwen3_next.py, qwen_gdn_linear_attn.py, qwen2_moe.py; vision qwen3_vl.py → qwen2_5_vl.py. Quantization targets evaluated in config group order, fused names resolved through constituent checkpoint names. Both ranks have these shapes.

**Corrections:** ungated QKV is 4096, not 3584; the enabled output gate doubles Q and makes serving N=7168. GDN serving fuses QKV+Z to N=8192 and B+A to N=48. Component in_proj_qkv/z timings must not be summed as the serving GEMM.

Bytes = packed weights + group/channel scales; excludes scalar scales, bias, padding/workspaces, embeddings, norms, GDN recurrent state, activation traffic and graph pools. conv1d is constructed as a Linear but executed as a convolution. Vision rows assume default weights TP mode (not data mode), run only on image prefill, and are excluded from the decode sum. No MTP module is constructed by this serving model; drafter is a separate model.

| module prefix | N | K | scheme | weight bytes/rank | benchmark shape |
|---|---:|---:|---|---:|---|
| language_model.model.layers.0.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.0.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.0.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.0.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.0.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.0.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.1.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.1.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.1.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.1.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.1.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.1.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.2.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.2.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.2.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.2.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.2.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.2.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.3.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.3.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.3.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.3.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.4.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.4.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.4.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.4.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.4.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.4.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.5.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.5.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.5.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.5.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.5.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.5.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.6.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.6.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.6.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.6.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.6.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.6.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.7.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.7.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.7.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.7.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.8.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.8.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.8.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.8.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.8.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.8.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.9.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.9.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.9.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.9.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.9.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.9.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.10.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.10.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.10.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.10.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.10.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.10.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.11.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.11.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.11.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.11.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.12.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.12.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.12.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.12.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.12.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.12.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.13.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.13.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.13.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.13.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.13.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.13.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.14.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.14.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.14.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.14.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.14.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.14.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.15.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.15.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.15.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.15.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.16.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.16.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.16.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.16.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.16.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.16.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.17.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.17.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.17.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.17.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.17.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.17.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.18.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.18.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.18.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.18.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.18.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.18.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.19.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.19.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.19.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.19.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.20.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.20.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.20.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.20.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.20.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.20.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.21.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.21.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.21.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.21.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.21.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.21.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.22.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.22.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.22.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.22.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.22.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.22.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.23.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.23.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.23.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.23.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.24.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.24.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.24.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.24.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.24.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.24.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.25.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.25.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.25.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.25.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.25.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.25.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.26.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.26.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.26.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.26.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.26.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.26.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.27.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.27.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.27.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.27.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.28.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.28.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.28.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.28.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.28.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.28.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.29.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.29.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.29.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.29.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.29.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.29.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.30.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.30.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.30.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.30.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.30.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.30.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.31.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.31.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.31.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.31.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.32.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.32.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.32.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.32.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.32.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.32.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.33.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.33.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.33.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.33.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.33.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.33.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.34.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.34.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.34.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.34.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.34.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.34.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.35.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.35.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.35.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.35.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.36.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.36.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.36.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.36.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.36.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.36.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.37.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.37.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.37.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.37.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.37.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.37.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.38.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.38.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.38.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.38.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.38.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.38.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.39.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.39.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.39.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.39.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.40.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.40.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.40.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.40.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.40.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.40.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.41.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.41.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.41.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.41.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.41.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.41.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.42.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.42.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.42.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.42.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.42.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.42.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.43.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.43.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.43.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.43.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.44.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.44.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.44.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.44.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.44.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.44.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.45.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.45.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.45.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.45.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.45.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.45.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.46.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.46.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.46.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.46.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.46.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.46.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.47.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.47.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.47.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.47.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.48.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.48.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.48.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.48.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.48.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.48.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.49.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.49.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.49.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.49.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.49.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.49.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.50.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.50.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.50.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.50.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.50.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.50.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.51.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.51.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.51.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.51.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.52.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.52.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.52.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.52.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.52.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.52.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.53.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.53.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.53.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.53.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.53.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.53.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.54.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.54.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.54.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.54.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.54.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.54.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.55.mlp.gate_up_proj | 17408 | 5120 | NVFP4 W4A4 | 50135040 | gate_up |
| language_model.model.layers.55.mlp.down_proj | 5120 | 8704 | NVFP4 W4A4 | 25067520 | down |
| language_model.model.layers.55.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.55.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.56.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.56.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.56.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.56.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.56.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.56.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.57.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.57.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.57.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.57.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.57.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.57.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.58.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.58.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.58.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.58.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.58.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.58.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.59.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.59.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.59.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.59.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.model.layers.60.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.60.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.60.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.60.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.60.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.60.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.61.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.61.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.61.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.61.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.61.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.61.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.62.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.62.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.62.linear_attn.in_proj_qkvz | 8192 | 5120 | FP8 W8A8 | 41975808 | in_proj_qkvz |
| language_model.model.layers.62.linear_attn.in_proj_ba | 48 | 5120 | BF16 | 491520 | — |
| language_model.model.layers.62.linear_attn.out_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | out_proj |
| language_model.model.layers.62.linear_attn.conv1d | 5120 | 4 | BF16 convolution | 40960 | — |
| language_model.model.layers.63.mlp.gate_up_proj | 17408 | 5120 | FP8 W8A8 | 89198592 | gate_up_fp8 |
| language_model.model.layers.63.mlp.down_proj | 5120 | 8704 | FP8 W8A8 | 44584960 | down_fp8 |
| language_model.model.layers.63.self_attn.qkv_proj | 7168 | 5120 | FP8 W8A8 | 36728832 | qkv |
| language_model.model.layers.63.self_attn.o_proj | 5120 | 3072 | FP8 W8A8 | 15749120 | o_proj |
| language_model.lm_head | 124160 | 5120 | FP8 W8A8 | 636195840 | lm_head |
| visual.blocks.0.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.0.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.0.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.0.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.1.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.1.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.1.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.1.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.2.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.2.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.2.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.2.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.3.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.3.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.3.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.3.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.4.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.4.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.4.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.4.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.5.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.5.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.5.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.5.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.6.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.6.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.6.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.6.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.7.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.7.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.7.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.7.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.8.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.8.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.8.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.8.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.9.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.9.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.9.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.9.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.10.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.10.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.10.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.10.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.11.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.11.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.11.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.11.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.12.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.12.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.12.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.12.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.13.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.13.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.13.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.13.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.14.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.14.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.14.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.14.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.15.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.15.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.15.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.15.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.16.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.16.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.16.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.16.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.17.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.17.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.17.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.17.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.18.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.18.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.18.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.18.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.19.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.19.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.19.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.19.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.20.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.20.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.20.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.20.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.21.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.21.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.21.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.21.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.22.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.22.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.22.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.22.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.23.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.23.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.23.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.23.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.24.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.24.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.24.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.24.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.25.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.25.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.25.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.25.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.blocks.26.attn.qkv | 1728 | 1152 | BF16 vision | 3981312 | — |
| visual.blocks.26.attn.proj | 1152 | 576 | BF16 vision | 1327104 | — |
| visual.blocks.26.mlp.linear_fc1 | 2152 | 1152 | BF16 vision | 4958208 | — |
| visual.blocks.26.mlp.linear_fc2 | 1152 | 2152 | BF16 vision | 4958208 | — |
| visual.merger.linear_fc1 | 2304 | 4608 | BF16 vision | 21233664 | — |
| visual.merger.linear_fc2 | 5120 | 2304 | BF16 vision | 23592960 | — |

## Decode GEMM multiplicities

| benchmark shape | modules |
|---|---:|
| down | 56 |
| down_fp8 | 8 |
| gate_up | 56 |
| gate_up_fp8 | 8 |
| in_proj_qkvz | 48 |
| lm_head | 1 |
| o_proj | 16 |
| out_proj | 48 |
| qkv | 16 |

BF16 in_proj_ba: 48 GEMMs/step, not covered by the NVFP4/FP8 timing suite. Embedding lookup is untied BF16, TP-sharded and offloaded by the launcher; it is not a linear GEMM. lm_head may run multiple times at different M during verification/sampling: multiply using profiled invocation counts, not blindly once per target step.

Decoder/head payload (including conv/B+A; excluding vision): 9,553,810,432 bytes/rank.
