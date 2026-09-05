#!/usr/bin/env python3
# Usage (offline, repo root): python3 deliver/build_shape_census.py > deliver/gemm_shape_census.md
"""Static expansion of source-audited module constructors; stdlib only."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def census(config):
    c = config['text_config']
    h, inter, tp = c['hidden_size'], c['intermediate_size'], 2
    groups = config['quantization_config']['config_groups']
    def scheme(name):
        for group in groups.values():
            if any(re.search(t.removeprefix('re:'), name) for t in group['targets']):
                return 'NVFP4 W4A4' if group['weights']['num_bits'] == 4 else 'FP8 W8A8'
        return 'BF16'
    rows = []
    def add(name, n, k, kind=None, measured='—'):
        kind = kind or scheme(name)
        size = n*k//2 + n*k//16 if kind == 'NVFP4 W4A4' else n*k+4*n if kind == 'FP8 W8A8' else 2*n*k
        rows.append((name, n, k, kind, size, measured))
    for i, typ in enumerate(c['layer_types']):
        root = f'language_model.model.layers.{i}'
        s = scheme(root+'.mlp.gate_proj')
        assert s == scheme(root+'.mlp.up_proj') == scheme(root+'.mlp.down_proj')
        add(root+'.mlp.gate_up_proj', inter, h, s, 'gate_up' if i < 56 else 'gate_up_fp8')
        add(root+'.mlp.down_proj', h, inter//tp, s, 'down' if i < 56 else 'down_fp8')
        if typ == 'full_attention':
            hd = c['head_dim']; q = c['num_attention_heads']*hd
            kv = c['num_key_value_heads']*hd
            add(root+'.self_attn.qkv_proj', (q*(1+c.get('attn_output_gate', True))+2*kv)//tp, h,
                scheme(root+'.self_attn.q_proj'), 'qkv')
            add(root+'.self_attn.o_proj', h, q//tp, measured='o_proj')
        else:
            kd = c['linear_num_key_heads']*c['linear_key_head_dim']
            vd = c['linear_num_value_heads']*c['linear_value_head_dim']
            add(root+'.linear_attn.in_proj_qkvz', (2*kd+2*vd)//tp, h,
                scheme(root+'.linear_attn.in_proj_qkv'), 'in_proj_qkvz')
            add(root+'.linear_attn.in_proj_ba', 2*c['linear_num_value_heads']//tp, h, 'BF16')
            add(root+'.linear_attn.out_proj', h, vd//tp, measured='out_proj')
            add(root+'.linear_attn.conv1d', (2*kd+vd)//tp, c['linear_conv_kernel_dim'], 'BF16 convolution')
    add('language_model.lm_head', c['vocab_size']//tp, h, measured='lm_head')
    # Vision default mm_encoder_tp_mode=weights, TP2. Image-prefill only.
    v = config['vision_config']; d = v['hidden_size']; f = v['intermediate_size']
    for i in range(v['depth']):
        root = f'visual.blocks.{i}'
        for suffix, n, k in [('attn.qkv', 3*d//tp, d), ('attn.proj', d, d//tp),
                             ('mlp.linear_fc1', f//tp, d), ('mlp.linear_fc2', d, f//tp)]:
            assert 'model.'+root+'.'+suffix in config['quantization_config']['ignore']
            add(root+'.'+suffix, n, k, 'BF16 vision')
    merged = d*v['spatial_merge_size']**2
    add('visual.merger.linear_fc1', merged//tp, merged, 'BF16 vision')
    add('visual.merger.linear_fc2', h, merged//tp, 'BF16 vision')
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', default='evidence/target-config.json'); args=p.parse_args()
    rows=census(json.loads(Path(args.config).read_text()))
    print('# Served TP2 linear census\n')
    print('Source-audited static constructor expansion: qwen3_5.py → qwen3_next.py, qwen_gdn_linear_attn.py, qwen2_moe.py; vision qwen3_vl.py → qwen2_5_vl.py. Quantization targets evaluated in config group order, fused names resolved through constituent checkpoint names. Both ranks have these shapes.\n')
    print('**Corrections:** ungated QKV is 4096, not 3584; the enabled output gate doubles Q and makes serving N=7168. GDN serving fuses QKV+Z to N=8192 and B+A to N=48. Component in_proj_qkv/z timings must not be summed as the serving GEMM.\n')
    print('Bytes = packed weights + group/channel scales; excludes scalar scales, bias, padding/workspaces, embeddings, norms, GDN recurrent state, activation traffic and graph pools. conv1d is constructed as a Linear but executed as a convolution. Vision rows assume default weights TP mode (not data mode), run only on image prefill, and are excluded from the decode sum. No MTP module is constructed by this serving model; drafter is a separate model.\n')
    print('| module prefix | N | K | scheme | weight bytes/rank | benchmark shape |\n|---|---:|---:|---|---:|---|')
    for row in rows:
        print('| '+' | '.join(map(str,row))+' |')
    print('\n## Decode GEMM multiplicities\n\n| benchmark shape | modules |\n|---|---:|')
    for shape, count in sorted(Counter(r[5] for r in rows if r[5]!='—').items()):
        print(f'| {shape} | {count} |')
    print('\nBF16 in_proj_ba: 48 GEMMs/step, not covered by the NVFP4/FP8 timing suite. Embedding lookup is untied BF16, TP-sharded and offloaded by the launcher; it is not a linear GEMM. lm_head may run multiple times at different M during verification/sampling: multiply using profiled invocation counts, not blindly once per target step.')
    print(f'\nDecoder/head payload (including conv/B+A; excluding vision): {sum(r[4] for r in rows if not r[0].startswith("visual.")):,} bytes/rank.')


if __name__=='__main__': main()
