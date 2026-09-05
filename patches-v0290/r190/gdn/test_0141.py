#!/usr/bin/env python3
"""Dependency-free parser, address coverage, and extracted-launcher tests."""
import ast
import functools
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

root = Path(sys.argv[1])
rel = 'vllm/third_party/flash_linear_attention/ops/'
spec = importlib.util.spec_from_file_location('cfg141', root / rel / 'gdn_decode_config.py')
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
parse = cfg.parse_decode_config
for variant in ('tiled', 'split'):
    for bv in (8, 16, 32, 64):
        for split in (1, 2, 4):
            if bv // split < 8 or (variant == 'tiled' and split != 1):
                continue
            for w in (1, 2, 4):
                for s in (1, 2, 3):
                    text = f'bv={bv},bk=128,warps={w},stages={s},split={split},variant={variant}'
                    assert parse(text) == (bv, 128, w, s, split, variant)
            effective = bv // split
            for batch in (1, 8, 16, 10, 80, 160):
                nv = 128 // effective
                # Every request/head/tile occurs exactly once, no split-K races.
                coords = {(pid // (24*nv), pid % 24, (pid // 24) % nv)
                          for pid in range(batch*24*nv)}
                assert len(coords) == batch * 24 * nv
                assert coords == {(n, h, t) for n in range(batch)
                                  for h in range(24) for t in range(nv)}
good = 'bv=32,bk=128,warps=1,stages=3,split=1'
assert parse(good)[-1] == 'tiled'
for bad in ('', good+',', good+'\n', good+',extra=1', good+',variant=unroll2',
            good.replace('bk=128', 'bk=64'), good.replace('warps=1', 'warps=8'),
            good.replace('stages=3', 'stages=0'), good.replace('split=1', 'split=2'),
            good.replace('bv=32', 'bv=-32'), good.replace('bv=32', 'bv=0'),
            good.replace('bv=32', 'bv=12'), good.replace('bv=32', 'bv= 32'),
            good+',bv=16', good.replace('split=1', 'split=0')):
    try:
        parse(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)

original = ast.parse(Path('src', rel, 'fused_recurrent.py').read_text())
patched = ast.parse((root / rel / 'fused_recurrent.py').read_text())
def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
kname = 'fused_recurrent_gated_delta_rule_packed_decode_kernel'
assert ast.dump(function(original, kname)) == ast.dump(function(patched, kname))
# The split kernel changes only CTA addressing, not recurrence statements.
a, b = function(original, kname).body, function(patched, kname.replace('_kernel', '_split_kernel')).body
def math_tail(nodes):
    pos = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'i_h')
    return [ast.dump(n) for n in nodes[pos:]]
assert math_tail(a) == math_tail(b)

class Tensor:
    def __init__(self, shape, dtype='bf16', contiguous=True):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.device = dtype, NS(index=0)
        self.contiguous = contiguous
    def stride(self, i):
        import math
        return math.prod(self.shape[(i % self.ndim)+1:])
    def numel(self):
        import math
        return math.prod(self.shape)
    def is_contiguous(self):
        return self.contiguous

class Kernel:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls
    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((self.name, grid, kwargs))
        return launch

def launcher(tree, raw=None, old=None, sm=True):
    calls = []
    ns = dict(functools=functools, next_power_of_2=lambda n: 1 << (n-1).bit_length(),
              current_platform=NS(is_device_capability_family=lambda *a, **kw: sm),
              envs=NS(VLLM_SM12X_GDN_DECODE_CFG=raw, VLLM_SM12X_GDN_PACKED_BV=old),
              triton=NS(next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=lambda a,b:(a+b-1)//b),
              torch=NS(bfloat16='bf16'), parse_decode_config=parse,
              logger=NS(info_once=lambda *a, **kw: None))
    for name in (kname, kname.replace('_kernel', '_split_kernel')):
        ns[name] = Kernel(name, calls)
    names = ('_get_packed_decode_launch_config', 'fused_recurrent_gated_delta_rule_packed_decode')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                        + [function(tree, name) for name in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<extracted launcher>', 'exec'), ns)
    return ns[names[1]], calls

inputs = [Tensor((10,5120)), Tensor((10,24)), Tensor((10,24)), Tensor((24,)),
          Tensor((24,)), 128**-0.5, Tensor((11,24,128,128)), Tensor((10,1,24,128)), Tensor((10,), 'i32'), True]
for old in (None,16,32):
    f, calls = launcher(original, old=old)
    f(*inputs)
    g, tuned_calls = launcher(patched, old=old)
    g(*inputs)
    assert calls == tuned_calls, 'unset must dispatch identical launch args'
f, calls = launcher(patched, good+',variant=split')
f(*inputs)
assert calls[0][1] == (960,)
for raw, sm, dtype in ((good,False,'bf16'),(good,True,'fp32'),('bad',True,'bf16')):
    f, calls = launcher(patched, raw, sm=sm)
    inputs[6].dtype = dtype
    try:
        f(*inputs)
    except ValueError:
        assert not calls
    else:
        raise AssertionError('unsupported configuration did not raise')
inputs[6].dtype = 'bf16'
print('Dependency-free tests: PASS (parser, CTA coverage, unchanged math, dispatch, rejection)')
