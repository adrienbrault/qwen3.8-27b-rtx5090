#!/usr/bin/env python3
# Usage: python3 deliver/test-0146.py .work/0146 (stdlib-only extracted-code tests).
import ast
import os
from pathlib import Path
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

ROOT = Path(sys.argv.pop(1))

def extract(file, names, ns):
    tree = ast.parse((ROOT / file).read_text())
    tree.body = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef))
                 and x.name in names]
    exec(compile(ast.unparse(tree), file, 'exec', flags=__import__('__future__').annotations.compiler_flag), ns)
    return ns

class Vec:
    def __init__(self, values, device='cpu', dtype='int32', base=None, start=0):
        self.values = values if base is None else base
        self.start = start
        self.length = len(values)
        self.device, self.dtype, self.ndim = device, dtype, 1
    def data(self):
        return self.values[self.start:self.start+self.length]
    def __getitem__(self, key):
        a, b, step = key.indices(self.length)
        assert step == 1
        return Vec([0]*(b-a), self.device, self.dtype, self.values, self.start+a)
    def __sub__(self, other):
        return Vec([a-b for a,b in zip(self.data(), other.data())])
    def numel(self):
        return self.length
    def is_contiguous(self):
        return True
    def copy_(self, source, non_blocking=False):
        assert non_blocking
        # Model delayed DMA: destination unreadable until event wait.
        pending.append((self, source.data()))
        actions.append('copy')
    def clone(self):
        actions.append('clone')
        return Vec(self.data().copy())

class Event:
    def record(self, stream):
        assert stream == 'main'
        actions.append('record')
    def synchronize(self):
        actions.append('wait')
        for dest, values in pending:
            dest.values[dest.start:dest.start+dest.length] = values
        pending.clear()

class Logger:
    def info_once(self, *args):
        actions.append('log')

pending, actions = [], []
fake = SimpleNamespace(int32='int32', empty=lambda n, **kw: Vec([0]*n),
    cuda=SimpleNamespace(is_current_stream_capturing=lambda: False,
                         current_stream=lambda device: 'main', Event=Event))
ns = extract('vllm/v1/worker/gpu/seqlens_copy.py', {'SeqLensCopy', 'SeqLensSnapshot'},
             {'torch': fake, 'logger': Logger(), 'gpu_sync_allowed': nullcontext})
parser = extract('vllm/envs.py', {'_sm12x_seqlens_one_copy'}, {'os': os})['_sm12x_seqlens_one_copy']

class Tests(unittest.TestCase):
    def setUp(self):
        pending.clear(); actions.clear()
    def test_grammar(self):
        name = 'VLLM_SM12X_SEQLENS_ONE_COPY'
        saved = os.environ.pop(name, None)
        try:
            self.assertFalse(parser())
            for value in ('0', '1'):
                os.environ[name] = value
                self.assertEqual(parser(), value == '1')
            for value in ('', '01', 'true', '-1', '2', ' 1', '1\n', '１'):
                os.environ[name] = value
                with self.assertRaises(ValueError): parser()
        finally:
            os.environ.pop(name, None)
            if saved is not None: os.environ[name] = saved
    def test_copy_order_cache_and_reuse(self):
        copier = ns['SeqLensCopy'](4, 'cuda:0')
        first = copier.begin(Vec([100, 130, 0], 'cuda:0'))
        self.assertEqual(actions, ['copy', 'record'])
        old = first.get(5)
        self.assertEqual(old.data(), [100, 130, 0])
        self.assertIs(first.get(5), old)
        self.assertEqual(actions.count('wait'), 1)
        self.assertLess(actions.index('wait'), actions.index('clone'))
        second = copier.begin(Vec([3], 'cuda:0'))
        self.assertEqual(second.get(5).data(), [3])
        self.assertEqual(old.data(), [100, 130, 0])
        self.assertIsNone(first.source)
    def test_reject_unconsumed_capture_bad_tensor(self):
        copier = ns['SeqLensCopy'](2, 'cuda:0')
        for source in (Vec([1], 'cpu'), Vec([1]*3, 'cuda:0'),
                       Vec([1], 'cuda:0', dtype='float')):
            with self.assertRaises(ValueError): copier.begin(source)
        fake.cuda.is_current_stream_capturing = lambda: True
        try:
            with self.assertRaises(RuntimeError): copier.begin(Vec([1], 'cuda:0'))
        finally: fake.cuda.is_current_stream_capturing = lambda: False
        copier.begin(Vec([1], 'cuda:0'))
        with self.assertRaises(RuntimeError): copier.begin(Vec([2], 'cuda:0'))
    def test_metadata_shares_exact_values_and_disabled_path(self):
        observed = []
        class FlashInferMetadataBuilder:
            def build(self, common_prefix_len, common_attn_metadata):
                observed.append(common_attn_metadata)
                return common_attn_metadata
        groups = [[SimpleNamespace(get_metadata_builder=lambda _: FlashInferMetadataBuilder(),
                                   layer_names=[f'layer{i}'])] for i in range(5)]
        build = extract('vllm/v1/worker/gpu/attn_utils.py', {'build_attn_metadata'},
            {'CommonAttentionMetadata': lambda **kw: kw, 'torch': SimpleNamespace(Tensor=Vec)})['build_attn_metadata']
        args = dict(attn_groups=groups, num_reqs=3, num_tokens=20,
                    query_start_loc_gpu=None, query_start_loc_cpu=Vec([0,10,20,20]),
                    max_query_len=10, seq_lens=Vec([100,130,0], 'cuda:0'), max_seq_len=140,
                    block_tables=[None]*5, slot_mappings=[None]*5,
                    kv_cache_config=SimpleNamespace(kv_cache_groups=[None]*5))
        build(**args)
        self.assertTrue(all('_seq_lens_cpu' not in m for m in observed))
        self.assertEqual(actions, [])
        observed.clear()
        snapshot = ns['SeqLensCopy'](3, 'cuda:0').begin(args['seq_lens'])
        build(**args, sm12x_seq_lens_snapshot=snapshot)
        for m in observed:
            self.assertIs(m['_seq_lens_cpu'], observed[0]['_seq_lens_cpu'])
            self.assertIs(m['_num_computed_tokens_cpu'], observed[0]['_num_computed_tokens_cpu'])
            self.assertEqual(m['_num_computed_tokens_cpu'].data(), [90,120,0])
        self.assertEqual(actions.count('wait'), 1)
        with self.assertRaises(ValueError):
            build(**args, sm12x_seq_lens_snapshot=snapshot, for_cudagraph_capture=True)
        FlashInferMetadataBuilder.use_dcp = True
        with self.assertRaises(ValueError): build(**args, sm12x_seq_lens_snapshot=snapshot)
    def test_draft_counterexample_and_hybrid_groups(self):
        # Direct formulas from _prepare_dflash_inputs_kernel, not an upper bound.
        target, rejected, query, limit = 128, 1, 10, 262144
        draft = min(target-rejected+query, limit)
        self.assertEqual(draft, 137)
        self.assertNotEqual((target+63)//64, (draft+63)//64)
        self.assertEqual(min(limit-1-0+query, limit), limit)
        # Distinct target full/GDN and draft sliding buckets; min bucket = 5.
        sizes = [16,48,5]
        group_size = min(sizes)
        self.assertEqual([(n+group_size-1)//group_size for n in sizes], [4,10,1])

unittest.main(verbosity=2)
