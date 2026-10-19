"""Operator-side CUDA checks. These are NOT reported as run on the CPU host."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class GPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import triton
        except ImportError as exc:
            raise unittest.SkipTest("GPU dependencies unavailable: %s" % exc)
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("requires one SM120 GPU")
        cls.torch = torch
        torch.set_grad_enabled(False)
        from vllm_ba4.runtime import ba4_out, State, try_apply
        from vllm_ba4.policy import CANDIDATES
        cls.op, cls.State, cls.route = staticmethod(ba4_out), State, staticmethod(try_apply)
        cls.names = CANDIDATES

    def fixtures(self, m=4):
        t = self.torch
        x = t.empty_strided((m, 5120), (5136, 1), device="cuda", dtype=t.bfloat16).normal_()
        w = t.randn((48, 5120), device="cuda", dtype=t.bfloat16)
        y = t.empty((m, 48), device="cuda", dtype=t.bfloat16)
        p = t.empty((40, m, 48), device="cuda", dtype=t.float32)
        return x, w, y, p

    def test_external_replay_timer_scales_and_trace_accounts_for_helpers(self):
        from vllm_ba4.measurement import validate_replay_timer, replay_us, trace_layers
        from vllm_ba4.kernels import stream_weights
        t = self.torch
        validate_replay_timer(t)  # Raises unless finite/positive and 16x work >2x time.
        x, w, out, partial = self.fixtures()
        stream = t.ones((1024*1024,), dtype=t.uint8, device='cuda')
        sink = t.empty((256,), dtype=t.int32, device='cuda')
        stream_weights(stream, sink)
        self.op(x, w, out, partial, 'triton_split256', '')
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            for _ in range(48):
                stream_weights(stream, sink)
                self.op(x, w, out, partial, 'triton_split256', '')
                stream_weights(stream, sink)
        graph.replay()
        t.cuda.synchronize()
        with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                                            t.profiler.ProfilerActivity.CUDA]) as prof:
            elapsed = replay_us(t.cuda, graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timing.trace.json'
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
        result = trace_layers(events)
        self.assertEqual(result['kernel_counts'], [2]*48)
        self.assertGreater(result['b1_trace_us_per_step'], 0)
        independent_b1_sum = sum(e['dur'] for e in events if e.get('ph') == 'X'
                                 and e.get('cat') == 'kernel'
                                 and '_stream_weights' not in e['name'])
        self.assertAlmostEqual(sum(result['per_layer_us']), independent_b1_sum)
        self.assertLessEqual(result['all_kernel_us'], elapsed*1.1)

    def assert_close(self, actual, x, w):
        reference = (x.double() @ w.double().t()).bfloat16()
        self.torch.testing.assert_close(actual, reference, rtol=1 / 64, atol=1 / 256)

    def test_patterns_and_all_shapes(self):
        t = self.torch
        for m in (4, 8, 12, 32):
            x, w, y, partial = self.fixtures(m)
            for pattern in ("random", "zero", "shard_boundary", "cancellation"):
                if pattern == "zero":
                    x.zero_()
                elif pattern == "shard_boundary":
                    x.zero_()
                    for row in range(m):
                        x[row, row * 17] = 1
                    w[23].fill_(2)
                    w[24].fill_(-3)
                    w[47].fill_(5)
                elif pattern == "cancellation":
                    x.fill_(1)
                    x[:, 1::2] = -1
                    w.fill_(8)
                    w[:, -1] = 7.5
                for name in self.names:
                    with self.subTest(m=m, pattern=pattern, kernel=name):
                        y.fill_(float("nan"))
                        partial.fill_(float("nan"))
                        try:
                            self.op(x, w, y, partial, name, "")
                        except RuntimeError:
                            if name != "torch_rows_mv":
                                raise
                            t.cuda.synchronize()  # Rethrow device failures.
                            continue  # BF16 GEMV may be unavailable.
                        self.assert_close(y, x, w)

    def test_schema_fake_and_functionalization(self):
        x, w, out, partial = self.fixtures()
        self.torch.library.opcheck(self.op, (x, w, out, partial, "triton_split256", ""),
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"))

    def test_compiled_real_route_stable_buffers_and_changed_graph_inputs(self):
        t = self.torch
        x, w, out, partial = self.fixtures()
        layer = SimpleNamespace(prefix="model.layers.0.linear_attn.in_proj_ba", weight=w,
            _sm12x_ba4=self.State("triton_split256", (5136, 1), w, out, partial))
        fn = t.compile(lambda x: self.route(layer, x, None), fullgraph=True)
        fn(x)
        t.cuda.synchronize()
        allocated = t.cuda.memory_allocated()
        t.cuda.reset_peak_memory_stats()
        result = fn(x)
        t.cuda.synchronize()
        self.assertEqual(result.data_ptr(), out.data_ptr())
        self.assertEqual(t.cuda.max_memory_allocated(), allocated)
        self.assert_close(result, x, w)
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            result = fn(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            t.cuda.synchronize()
            self.assert_close(result, x, w)
        self.assertIsNone(self.route(layer, x[:3], None))


if __name__ == "__main__":
    unittest.main()
