#!/usr/bin/env python3
"""Usage: python3 deliver/test-0157.py APPLIED_VLLM_ROOT ORIGINAL_VLLM_ROOT."""
import ast
from pathlib import Path
import sys

root, original = map(Path, sys.argv[1:])
gdn = 'v1/attention/backends/gdn_attn.py'
mamba = 'v1/worker/gpu/model_states/mamba_hybrid.py'
utils = 'v1/worker/gpu/attn_utils.py'
def parse(base, file):
    return ast.parse((base / file).read_text())
def method(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
def calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
            ((isinstance(n.func, ast.Name) and n.func.id == name) or
             (isinstance(n.func, ast.Attribute) and n.func.attr == name))]
def same(a, b):
    assert ast.dump(ast.Module(body=a, type_ignores=[])) == ast.dump(ast.Module(body=b, type_ignores=[]))

build = method(parse(root, gdn), 'build')
old = method(parse(original, gdn), 'build')
# Prove every remaining sort/repeat/cumsum is guarded by ReplaySSM=True.
def visit(node, replay=False):
    if isinstance(node, ast.If):
        guarded = ast.unparse(node.test) == 'self.use_cache_spec_kernel'
        visit(node.test, replay)
        for child in node.body:
            visit(child, replay or guarded)
        for child in node.orelse:
            visit(child, replay)
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {'argsort', 'repeat_interleave', 'cumsum'}:
            assert replay, ast.unparse(node)
    for child in ast.iter_child_nodes(node):
        visit(child, replay)
visit(build)
assert len(calls(build, 'compute_common_gdn_attn_metadata')) == 1
fallback = next(n for n in ast.walk(build) if isinstance(n, ast.If) and
                ast.unparse(n.test) == 'common_gdn_metadata is None')
assert len(calls(fallback, 'compute_common_gdn_attn_metadata')) == 1
print('PASS: non-ReplaySSM build has no sort/repeat/cumsum; omitted tuple computes locally')

prepare = method(parse(root, mamba), 'prepare_attn')
assert len(calls(prepare, 'compute_common_gdn_attn_metadata')) == 1
assert len(calls(prepare, 'info_once')) == 1
capture = method(parse(root, gdn), 'build_for_cudagraph_capture')
assert len(calls(capture, 'compute_common_gdn_attn_metadata')) == 1
assert any(k.arg == 'common_gdn_metadata' for k in calls(capture, 'build')[0].keywords)
extra = method(parse(root, mamba), 'get_extra_attn_kwargs')
assert 'GDNAttentionMetadataBuilder' in ast.unparse(extra)
assert 'extra_kwargs[\'common_gdn_metadata\']' in ast.unparse(extra)
print('PASS: prepare_attn has exactly one helper call; capture and GDN kwargs wired')

old_args = {a.arg for a in old.args.args + old.args.kwonlyargs}
new_pos = build.args.args
with_defaults = {a.arg for a in new_pos[len(new_pos)-len(build.args.defaults):]}
with_defaults |= {a.arg for a, d in zip(build.args.kwonlyargs, build.args.kw_defaults) if d is not None}
added = {a.arg for a in new_pos + build.args.kwonlyargs} - old_args
assert added == {'common_gdn_metadata'} and added <= with_defaults
print('PASS: every added build kwarg has a default')

# Compare entire original ReplaySSM mask and shared split bodies, plus all later code.
old_mask = next(n for n in old.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.use_cache_spec_kernel')
new_mask = next(n for n in build.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.use_cache_spec_kernel')
same(old_mask.body, new_mask.body)
old_split = next(n for n in old.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'spec_sequence_masks is None')
new_split = next(n for n in build.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'spec_sequence_masks is None')
same(old_split.body, new_split.body)
same(old_split.orelse, new_split.orelse[0].body)
same(old.body[old.body.index(old_split)+1:], build.body[build.body.index(new_split)+1:])
helper = method(parse(root, utils), 'compute_common_gdn_attn_metadata')
assert not calls(helper, 'arange')
assert len(calls(helper, 'cumsum')) == 3
assert 'self.spec_token_arange[:spec_token_indx.numel()]' in ast.unparse(build)
print('PASS: ReplaySSM bodies and post-split code unchanged; cached arange retained')
print('PASS: 0157 dependency-free structural checks')
