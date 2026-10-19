"""Run-5: real attention partitioning must accept live offset metadata."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE/'tests'), str(HERE)]


class SplitterCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_profile import ProfileCompileTests
        ProfileCompileTests.setUpClass()
        cls.profile = ProfileCompileTests
        cls.h = ProfileCompileTests.h
        cls.torch = cls.h.torch
        # Registers the actual target attention schema and CPU mutation body.
        from gdn_wrapper import attention
        cls.attention = staticmethod(attention)

    def tearDown(self):
        self.torch._dynamo.reset()

    def test_first8192_real_attention_splitter_piecewise_and_aot(self):
        from compile_wrapper import wrap
        from splitter_wrapper import backend
        torch, h, profile = self.torch, self.h, self.profile
        cases = [(False, c) for c in (None, 'aot_eager', 'inductor')]
        if hasattr(torch.compile(lambda x: x), 'aot_compile'):
            cases += [(True, c) for c in (None, 'aot_eager', 'inductor')]
        for aot, compiler in cases:
            with self.subTest(aot=aot, compiler=compiler), torch.no_grad():
                torch._dynamo.reset()
                layer = h.layer(profile=True)
                profile.phase.update(compiled=False, calls=0)
                def apply(method, layer, x):
                    x = profile.producer(x)
                    # Two BA projections separated by the real attention op
                    # force shared nonstrict input/output TreeSpecs across a
                    # boundary, as in the backbone's repeated GDN layers.
                    ba = h.apply(method, layer, x)
                    z = torch.zeros_like(ba)
                    core = torch.empty_like(ba.unsqueeze(-1))
                    self.attention(x, ba, z, core, layer.prefix)
                    return h.apply(method, layer, x + core.squeeze(-1).sum(-1, keepdim=True))
                records = []
                split_backend = backend(compiler, records)
                def compile(gm, inputs):
                    self.assertEqual(profile.phase['calls'], 0)
                    result = split_backend(gm, inputs)
                    profile.phase['compiled'] = True
                    return result
                call, graphs = wrap(types.SimpleNamespace(apply=apply), layer, compile, aot=aot)
                retained = []
                for m in (8192, 4, 8, 12, 65, 4):
                    x = torch.randn(m, 5120, dtype=torch.bfloat16)
                    previous = profile.phase['calls']
                    h.runtime._used.clear()
                    with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                        result = call(x)
                    produced = x + 1
                    ba = torch.nn.functional.linear(produced, layer.weight)
                    expected = torch.nn.functional.linear(produced + ba.sum(-1, keepdim=True), layer.weight)
                    self.assertTrue(torch.equal(result, expected))
                    self.assertEqual(profile.phase['calls'], previous+1)
                    self.assertEqual(selected.call_count, 2 * int(m == 4))
                    self.assertEqual(bool(h.runtime._used), m == 4)
                    retained.append((result, result.clone()))
                    for value, saved in retained: self.assertTrue(torch.equal(value, saved))
                self.assertEqual(len(graphs), 1)
                record, = records
                self.assertEqual(sum(i.is_splitting_graph for i in record['items']), 1)
                self.assertGreaterEqual(len(record['pieces']), 2)
                for range, graph in record['graphs']:
                    if any(n.op == 'call_function' and n.target == torch.ops.vllm.sm12x_ba4_dispatch.default for n in graph.graph.nodes):
                        self.assertTrue(range.is_single_size() and range.start == 4)
                print('J41V_SPLITTER ' + json.dumps(dict(torch=torch.__version__, first=8192,
                    compiler=compiler, wrapper_aot=aot, partitions=len(record['items']),
                    real_attention_split=True, real_piecewise=True,
                    aot_autograd=compiler in ('aot_eager','inductor'),
                    traced_producer_calls=0)), flush=True)


    def test_produced_offsets_and_guardless_weight_offset_through_split_aot(self):
        from compile_wrapper import wrap
        from splitter_wrapper import backend
        torch, h, profile = self.torch, self.h, self.profile
        for offset in (0, 1, 8):
            with self.subTest(offset=offset), torch.no_grad():
                torch._dynamo.reset()
                layer = h.layer(profile=True)
                layer._sm12x_ba4.x_stride = (5136, 1)
                profile.phase.update(compiled=False, calls=0)
                def apply(method, layer, x):
                    value = profile.producer(x)[:, offset:offset+5120]
                    ba = h.apply(method, layer, value)
                    z = torch.zeros_like(ba)
                    core = torch.empty_like(ba.unsqueeze(-1))
                    self.attention(value, ba, z, core, layer.prefix)
                    return h.apply(method, layer, value), core.squeeze(-1)
                records = []
                real_backend = backend('inductor', records)
                def compile(gm, args):
                    self.assertEqual(profile.phase['calls'], 0)
                    result = real_backend(gm, args)
                    profile.phase['compiled'] = True
                    return result
                aot = hasattr(torch.compile(lambda x: x), 'aot_compile')
                call, graphs = wrap(types.SimpleNamespace(apply=apply), layer, compile, aot=aot)
                for m in (8192, 4, 8, 4):
                    x = torch.randn(m, 5136, dtype=torch.bfloat16)
                    previous = profile.phase['calls']
                    h.runtime._used.clear()
                    with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                        result = call(x)
                    value = (x+1)[:, offset:offset+5120]
                    ba = torch.nn.functional.linear(value, layer.weight)
                    self.assertTrue(all(torch.equal(out, ba) for out in result))
                    self.assertEqual(selected.call_count, 2*int(m == 4 and offset % 8 == 0))
                    self.assertEqual(bool(h.runtime._used), m == 4 and offset % 8 == 0)
                    self.assertEqual(profile.phase['calls'], previous+1)
                self.assertEqual(len(graphs), 1)
                self.assertEqual(len(records), 1)
                self.assertGreaterEqual(len(records[0]['items']), 3)
                if offset % 8 == 0:
                    h.mutate(layer, 'offset')
                    h.runtime._used.clear()
                    with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                        with self.assertRaisesRegex(RuntimeError, 'BA4 live weight'):
                            call(torch.randn(4, 5136, dtype=torch.bfloat16))
                    selected.assert_not_called()
                    self.assertFalse(h.runtime._used)
                print('J41V_SPLITTER_OFFSETS ' + json.dumps(dict(torch=torch.__version__,
                    offset=offset, first=8192, wrapper_aot=aot, compiler='inductor',
                    traced_producer_calls=0, misaligned_falls_back=True,
                    aligned_live_weight_offset_fails_closed=offset % 8 == 0)), flush=True)

if __name__ == '__main__': unittest.main(verbosity=2)
