"""Profile's first M8192 call must inspect fake metadata, never run its producers."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE/'tests'), str(HERE)]


class ProfileCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_compile import ServingCompileTests
        ServingCompileTests.setUpClass()
        cls.h = ServingCompileTests()
        cls.torch = cls.h.torch
        torch = cls.torch
        cls.phase = dict(compiled=False, calls=0)
        @torch.library.custom_op('ba4_test::profile_producer', mutates_args=())
        def producer(x: torch.Tensor) -> torch.Tensor:
            if not cls.phase['compiled']:
                raise RuntimeError('J41T producer executed for real during Dynamo tracing')
            cls.phase['calls'] += 1
            return x + 1
        @producer.register_fake
        def fake(x):
            return x.new_empty(x.shape)
        cls.producer = staticmethod(producer)

    def tearDown(self):
        self.torch._dynamo.reset()

    def test_profile_first8192_producer_never_runs_during_trace(self):
        from compile_wrapper import wrap
        torch, h = self.torch, self.h
        backends = (None, 'aot_eager', 'inductor')
        for aot in (False, True):
            if aot and not hasattr(torch.compile(lambda x:x), 'aot_compile'):
                continue  # Torch 2.8 has no wrapper.aot_compile API.
            for backend in (backends[1:] if aot else backends):
                with self.subTest(aot=aot, backend=backend), torch.no_grad():
                    torch._dynamo.reset()
                    layer = h.layer(profile=True)
                    self.phase.update(compiled=False, calls=0)
                    def apply(method, layer, x):
                        return h.apply(method, layer, self.producer(x))
                    def compile_graph(gm, inputs):
                        self.assertEqual(self.phase['calls'], 0)
                        self.phase['compiled'] = True
                        return gm.forward if backend is None else torch._dynamo.lookup_backend(backend)(gm, inputs)
                    call, graphs = wrap(types.SimpleNamespace(apply=apply), layer, compile_graph, aot=aot)
                    for m in (8192, 4, 8, 12, 32, 65, 4):
                        x = torch.randn(m, 5120, dtype=torch.bfloat16)
                        h.runtime._used.clear()
                        previous = self.phase['calls']
                        with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                            out = call(x)
                        self.assertEqual(self.phase['calls'], previous+1)
                        self.assertTrue(torch.equal(out, torch.nn.functional.linear(x+1, layer.weight)))
                        self.assertEqual(selected.call_count, int(m == 4))
                        self.assertEqual(bool(h.runtime._used), m == 4)
                    self.assertEqual(len(graphs), 1)
                    self.assertIn('sm12x_ba4_dispatch', str(graphs[0].graph))
                    print('J41T_PROFILE ' + json.dumps(dict(torch=torch.__version__, first=8192,
                        aot=aot, backend=backend, real_producer_calls=self.phase['calls'],
                        traced_producer_calls=0, graphs=len(graphs))), flush=True)

    def test_produced_view_live_offsets_are_observed_without_real_execution(self):
        from compile_wrapper import wrap
        torch, h = self.torch, self.h
        for offset in (0, 1, 8):
            with self.subTest(offset=offset), torch.no_grad():
                torch._dynamo.reset()
                layer = h.layer(profile=True)
                self.phase.update(compiled=False, calls=0)
                def apply(method, layer, x):
                    produced = self.producer(x).flatten()[offset:-5120+offset].view(-1,5120)
                    return h.apply(method, layer, produced)
                def backend(gm, inputs):
                    self.assertEqual(self.phase['calls'], 0)
                    self.phase['compiled'] = True
                    return torch._dynamo.lookup_backend('inductor')(gm, inputs)
                call, graphs = wrap(types.SimpleNamespace(apply=apply), layer, backend)
                for m in (8192, 4, 8, 4):
                    x = torch.randn(m+1,5120,dtype=torch.bfloat16)
                    h.runtime._used.clear()
                    with mock.patch.object(h.runtime,'run_out',wraps=h.runtime.run_out) as selected:
                        out = call(x)
                    value = (x+1).flatten()[offset:-5120+offset].view(-1,5120)
                    self.assertTrue(torch.equal(out,torch.nn.functional.linear(value,layer.weight)))
                    self.assertEqual(selected.call_count,int(m==4 and offset%8==0))
                    self.assertEqual(bool(h.runtime._used),m==4 and offset%8==0)
                self.assertEqual(len(graphs),1)
                self.assertEqual('sm12x_ba4_dispatch' in str(graphs[0].graph),offset%8==0)
                print('J41T_PRODUCED_OFFSET '+json.dumps(dict(offset=offset,first=8192,
                    torch=torch.__version__,traced_producer_calls=0)),flush=True)


if __name__ == '__main__': unittest.main(verbosity=2)
