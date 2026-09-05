#!/usr/bin/env python3
# Usage: python3 deliver/test-0143.py .work/0143 (standard library only).
import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS
import os
import sys
import unittest

root = Path(sys.argv.pop(1))

def extract(path, names, namespace):
    tree = ast.parse((root / path).read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)

class Tags(unittest.TestCase):
    def test_grammar(self):
        ns = dict(os=os, Callable=__import__('collections.abc', fromlist=['Callable']).Callable)
        extract('vllm/envs.py', ['env_with_choices'], ns)
        key = 'VLLM_SM12X_COLLECTIVE_TAGS'
        original = os.environ.pop(key, None)
        try:
            parse = ns['env_with_choices'](key, '0', ['0', '1'])
            self.assertEqual(parse(), '0')
            for value in ['0', '1']:
                os.environ[key] = value
                self.assertEqual(parse(), value)
            for value in ['', 'true', '2', '-1', ' 1', '01']:
                os.environ[key] = value
                with self.assertRaises(ValueError): parse()
        finally:
            os.environ.pop(key, None)
            if original is not None: os.environ[key] = original

    def test_host_annotations(self):
        names = []
        config = NS(use_v2_model_runner=True, parallel_config=NS(
            tensor_parallel_size=2, pipeline_parallel_size=1))
        ns = dict(contextmanager=contextmanager, nullcontext=nullcontext,
                  _phase=ContextVar('test', default=None), ENABLED=False,
                  torch=NS(profiler=NS(record_function=lambda _: nullcontext()), Tensor=object, cuda=NS(nvtx=NS(range=lambda x: names.append(x) or nullcontext()))),
                  logger=NS(info_once=lambda *a, **k: self.assertEqual(k['scope'], 'process')),
                  get_current_vllm_config=lambda: config,
                  current_platform=NS(is_cuda=lambda: True))
        extract('vllm/utils/vocab_collective_tags.py',
                ['validate_collective_tags', 'vocab_phase', 'vocab_range', '_tagged_range'], ns)
        # Disabled must not inspect the tensor or call NVTX.
        with ns['vocab_range'](object(), 'off'): pass
        self.assertEqual(names, [])
        ns['ENABLED'] = True
        ns['validate_collective_tags']()
        config.parallel_config.tensor_parallel_size = 1
        with self.assertRaises(ValueError): ns['validate_collective_tags']()
        tensor = NS(is_cuda=True, ndim=2, shape=(80, 124160), numel=lambda: 80*124160, element_size=lambda: 2, dtype='torch.bfloat16')
        with ns['vocab_range'](tensor, 'target_logits_allgather'): pass
        with self.assertRaises(RuntimeError):
            with ns['vocab_phase']('prompt_logprobs_allgather'):
                with ns['vocab_range'](tensor, 'target_logits_allgather'): pass
                raise RuntimeError()
        self.assertIsNone(ns['_phase'].get())
        self.assertEqual(names, ['vocab_ag:target_logits_allgather:80x124160',
                                'vocab_ag:prompt_logprobs_allgather:80x124160'])
        with self.assertRaises(ValueError): ns['vocab_range'](NS(is_cuda=False), 'cpu')

    def test_bytes(self):
        self.assertEqual(80 * 124160 * 2, 19865600)
        self.assertEqual(160 * 124160 * 2, 39731200)
        self.assertEqual(72 * 16 * (2 + 8), 11520)
        self.assertEqual(80 * 20 * (4 + 8), 19200)
        self.assertAlmostEqual((19865600 - 19200) / 25.56e6, 0.7764632237871675)

    def test_trace_correlation(self):
        import runpy
        summarize = runpy.run_path(str(Path(__file__).with_name('prof-vocab-0143.py')))['summarize']
        events = [
            dict(ph='X', name='vocab_ag:target_logits_allgather:80x124160:bytes=19865600:dtype=torch.bfloat16', ts=10, dur=10, pid=1, tid=2),
            dict(ph='X', cat='cuda_runtime', ts=12, dur=1, pid=1, tid=2, args={'correlation': 7}),
            dict(ph='X', cat='kernel', name='ncclAllGather', ts=40, dur=100, args={'correlation': 7}),
            dict(ph='X', cat='kernel', name='ncclAllGather', ts=50, dur=20, args={'correlation': 9}),
        ]
        totals, unknown = summarize(events)
        self.assertEqual(totals['target_logits_allgather']['gpu_us'], 100)
        self.assertEqual(totals['target_logits_allgather']['input_bytes'], 19865600)
        self.assertEqual(unknown, 1)

    def test_every_collective_is_scoped(self):
        for file in ['model_executor/layers/logits_processor.py',
                     'v1/worker/gpu/sample/batch_shard.py']:
            tree = ast.parse((root / 'vllm' / file).read_text())
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call): continue
                name = ast.unparse(node.func)
                if name not in ('tensor_model_parallel_all_gather', 'tensor_model_parallel_gather',
                                'torch.distributed.all_to_all_single'): continue
                p = parents[node]
                while p in parents and not isinstance(p, ast.With): p = parents[p]
                self.assertIsInstance(p, ast.With)
                self.assertIn('vocab_range', ast.unparse(p.items[0].context_expr))

unittest.main(verbosity=2)
