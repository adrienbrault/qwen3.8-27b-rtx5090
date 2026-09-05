#!/usr/bin/env python3
# Usage: python3 deliver/test_0145.py .work/0145
"""Dependency-free parser, kernel-body, launch and CTA-coverage checks."""
import ast
import pathlib
import runpy
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRATCH = pathlib.Path(sys.argv.pop(1))
REL = pathlib.Path('vllm/third_party/flash_linear_attention/ops')
parse = runpy.run_path(str(SCRATCH / REL / 'gdn_spec_config.py'))['parse_spec_config']
old = ast.parse((ROOT / 'src' / REL / 'fused_sigmoid_gating.py').read_text())
new = ast.parse((SCRATCH / REL / 'fused_sigmoid_gating.py').read_text())


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


class Tensor:
    dtype = 'bf16'
    device = types.SimpleNamespace(index=0)

    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)

    def __len__(self):
        return self.shape[0]

    def numel(self):
        import math
        return math.prod(self.shape)

    def stride(self, i=None):
        result = []
        product = 1
        for size in reversed(self.shape):
            result.insert(0, product)
            product *= size
        return result[i] if i is not None else tuple(result)

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def new_empty(self, *shape, **kwargs):
        return Tensor(*shape)

    def squeeze(self, i):
        return Tensor(*(self.shape[:i] + self.shape[i+1:]))


class Kernel:
    def __getitem__(self, grid):
        def call(**kw):
            self.grid, self.kw = grid, kw
        return call


def wrapper(tree, cfg=None, capable=True):
    kernels = [Kernel(), Kernel()]
    logs = []
    scope = dict(envs=types.SimpleNamespace(VLLM_SM12X_GDN_SPEC_CFG=cfg),
                 torch=types.SimpleNamespace(bfloat16='bf16'),
                 current_platform=types.SimpleNamespace(
                     is_device_capability_family=lambda *a, **k: capable),
                 triton=types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(),
                                              cdiv=lambda a, b: (a+b-1)//b),
                 parse_spec_config=parse,
                 logger=types.SimpleNamespace(info_once=lambda *a, **k: logs.append(a)))
    scope['fused_sigmoid_gating_delta_rule_update_kernel'] = kernels[0]
    scope['fused_sigmoid_gating_delta_rule_update_split_kernel'] = kernels[1]
    function = fn(tree, 'fused_sigmoid_gating_delta_rule_update')
    code = 'from __future__ import annotations\n' + ast.unparse(function)
    exec(compile(code, '<wrapper>', 'exec'), scope)
    return scope[function.name], kernels, logs


def inputs():
    return dict(A_log=Tensor(24), a=Tensor(10, 24), b=Tensor(10, 24), dt_bias=Tensor(24),
                q=Tensor(1, 10, 8, 128), k=Tensor(1, 10, 8, 128), v=Tensor(1, 10, 24, 128),
                initial_state=Tensor(11, 24, 128, 128), cu_seqlens=Tensor(2),
                ssm_state_indices=Tensor(1, 10), num_accepted_tokens=Tensor(1),
                use_qk_l2norm_in_kernel=True)


class Tests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse('bv=16,bk=128,warps=2,stages=3,split=1'),
                         (16, 128, 2, 3, 1, 'tiled'))
        self.assertEqual(parse('bv=32,bk=128,warps=1,stages=1,split=4,variant=split')[4], 4)
        for raw in ('', ' bv=16,bk=128,warps=2,stages=3,split=1',
                    'bv=16,bk=64,warps=2,stages=3,split=1',
                    'bv=16,bk=128,warps=2,stages=3,split=2',
                    'bv=8,bk=128,warps=2,stages=3,split=4,variant=split',
                    'bv=16,bk=128,warps=8,stages=3,split=1',
                    'bv=16,bk=128,warps=2,stages=4,split=1',
                    'bv=16,bk=128,warps=2,stages=3,split=1,variant=pipeline'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw)

    def test_kernel_unchanged_and_copy_body(self):
        name = 'fused_sigmoid_gating_delta_rule_update_kernel'
        self.assertEqual(ast.dump(fn(old, name)), ast.dump(fn(new, name)))
        original = fn(new, name)
        copied = fn(new, name.replace('_kernel', '_split_kernel'))
        self.assertEqual(ast.dump(original.args), ast.dump(copied.args))
        self.assertEqual([ast.dump(x) for x in original.body[1:]],
                         [ast.dump(x) for x in copied.body[4:]])
        self.assertEqual([ast.dump(x) for x in original.decorator_list],
                         [ast.dump(x) for x in copied.decorator_list])

    def test_coverage(self):
        for n in (1, 8, 16):
            for bv in (8, 16, 32, 64):
                nv = 128 // bv
                mapped = {(pid // (24*nv), pid % 24, (pid//24) % nv)
                          for pid in range(n*24*nv)}
                self.assertEqual(mapped, {(r, h, v) for r in range(n)
                                         for h in range(24) for v in range(nv)})

    def test_unset_launch(self):
        for spec in (True, False):
            data = inputs()
            if not spec:
                data['num_accepted_tokens'] = None
            f, k, logs = wrapper(old)
            g, l, newlogs = wrapper(new)
            f(**data)
            g(**data)
            self.assertEqual(k[0].grid, l[0].grid)
            def normalize(kw):
                return {key: (val.shape, val.dtype) if isinstance(val, Tensor) else val
                        for key, val in kw.items()}
            self.assertEqual(normalize(k[0].kw), normalize(l[0].kw))
            self.assertEqual(newlogs, [])
            self.assertFalse(hasattr(l[1], 'grid'))

    def test_optin_and_rejections(self):
        cfg = 'bv=32,bk=128,warps=2,stages=3,split=2,variant=split'
        f, k, logs = wrapper(new, cfg)
        f(**inputs())
        self.assertEqual(k[1].grid, (192,))
        self.assertEqual(k[1].kw['BV'], 16)
        self.assertEqual(len(logs), 1)
        self.assertEqual(k[1].kw['BK'], 128)
        for change in ({'is_kda': True}, {'use_qk_l2norm_in_kernel': False},
                       {'ssm_state_indices': Tensor(1, 9)}, {'q': Tensor(1, 10, 4, 128)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                f(**(inputs() | change))
        f, _, _ = wrapper(new, cfg, capable=False)
        with self.assertRaises(ValueError):
            f(**inputs())
        f, _, _ = wrapper(new, '')
        with self.assertRaises(ValueError):
            f(**inputs())
        f, k, logs = wrapper(new, cfg)
        f(**(inputs() | {'num_accepted_tokens': None}))
        self.assertEqual(k[0].grid, (1, 4, 24))
        self.assertEqual(logs, [])


if __name__ == '__main__':
    unittest.main()
