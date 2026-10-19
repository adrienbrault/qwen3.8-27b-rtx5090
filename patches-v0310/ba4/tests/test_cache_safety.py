"""Real Torch 2.13 AOT key, GraphPickler, and shared disk-cache regressions.

CUDA inputs are FakeTensors. The disk test compiles a reduced CPU numerical
payload retaining the production graph's BA4 metadata probes (if any), with
rank-specific constants standing in for device-bound generated programs.
"""
import copy
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

import test_compile as serving


def capture(selected):
    import torch
    from compile_wrapper import wrap
    from piecewise_wrapper import load_backend
    from vllm_ba4.compilation import graph_for_range, prune_inputs
    h = serving.ServingCompileTests()
    with torch.no_grad():
        layer = h.layer(profile=True)
        call, graphs = wrap(h, layer)
        call(torch.ones(4, 5120, dtype=torch.bfloat16))
    gm, = graphs
    _, Range = load_backend()
    result = graph_for_range(gm, Range(*selected))
    if result is not gm:
        # Real piecewise caller's input-pruning path.
        result, _, _ = prune_inputs(result, [None for n in result.graph.nodes if n.op == 'placeholder'])
    return gm, result


def probes(gm):
    return [n for n in gm.graph.nodes if n.op == 'call_function'
            and 'offset_metadata' in str(n.target)]


def _legacy_offset_metadata(x):
    return x.as_strided((0,), (x.storage_offset()+1,))


def disk_worker(selected, rank, root, poison=False):
    import torch
    import torch._functorch.config as fc
    import torch._inductor.config as ic
    from torch._inductor.standalone_compile import standalone_compile
    from torch._dynamo.utils import counters
    serving.ServingCompileTests.setUpClass()
    _, actual = capture(selected)
    g = torch.fx.Graph()
    x = g.placeholder('x')
    example = torch.ones(4)
    x.meta['example_value'] = example
    if poison:
        g.call_function(_legacy_offset_metadata, (x,))
    else:
        for node in probes(actual):
            g.call_function(node.target, (x,))
    y = g.call_function(torch.ops.aten.add.Tensor, (x, rank))
    y.meta['example_value'] = example
    g.output((y,))
    reduced = torch.fx.GraphModule({}, g)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = root
    mode = 'from_example_inputs' if selected == (4, 4) else 'from_graph'
    with fc.patch(bypass_autograd_cache_key=True, enable_autograd_cache=True), ic.patch(fx_graph_cache=True):
        random.seed(0)
        fn = standalone_compile(reduced, [example], dynamic_shapes=mode, aot=True, options={})
        value = fn(example)[0].tolist()
    print('J41X_REAL_DISK ' + json.dumps(dict(rank=rank, actual=value, expected=[1.0+rank]*4,
        correct=value == [1.0+rank]*4,
        hit=counters['aot_autograd']['autograd_cache_hit'] > 0,
        nonce_dirs=[p.name for p in Path(root, 'aotautograd').iterdir()
                    if p.is_dir() and re.fullmatch(r'0\.\d+', p.name)],
        production_probe_targets=[str(n.target) for n in probes(actual)])), flush=True)


class CacheSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serving.ServingCompileTests.setUpClass()

    def tearDown(self):
        import torch
        torch._dynamo.reset()

    def test_real_keys_and_graph_pickler_for_both_production_ranges(self):
        import torch
        if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 13):
            return  # The pinned 2.13 image gate exercises the APIs below.
        import torch._functorch.config as fc
        from torch._functorch._aot_autograd.autograd_cache import check_cacheable
        from torch._inductor.standalone_compile import autograd_cache_key
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from piecewise_wrapper import load_backend
        module, _ = load_backend()
        for selected in ((4, 4), (1, 8192)):
            original, gm = capture(selected)
            for label, graph in (('dynamo', original), ('range', gm)):
                check_cacheable(graph)
                data = GraphPickler.dumps(graph, Options(ops_filter=None))
                self.assertGreater(len(data), 0)
                for mode in ('from_example_inputs', 'from_graph'):
                    keys = []
                    for rank in (0, 1):
                        device_graph = copy.deepcopy(graph)
                        fm = FakeTensorMode(shape_env=ShapeEnv())
                        for node in device_graph.graph.nodes:
                            value = node.meta.get('example_value')
                            if isinstance(value, torch.Tensor):
                                # Keep graph/strides/dtypes/offsets identical between
                                # ranks, changing only input FakeTensor devices.
                                sizes = tuple(int(s) for s in value.shape)
                                strides = tuple(int(s) for s in value.stride())
                                meta = torch.empty_strided(sizes, strides, dtype=value.dtype, device='meta')
                                node.meta['example_value'] = FakeTensor(fm, meta, torch.device('cuda', rank))
                        args = module.get_fake_args_from_graph(device_graph)
                        with fc.patch(bypass_autograd_cache_key=True, enable_autograd_cache=True):
                            random.seed(0)
                            key = autograd_cache_key(device_graph, args, mode, aot=True,
                                **({'fake_mode': fm} if mode == 'from_example_inputs' else {}))[0]
                        self.assertRegex(key, r'^a[a-z2-7]+$')
                        keys.append(key)
                    self.assertNotEqual(*keys)
                    print('J41X_REAL_KEYS ' + json.dumps(dict(range=selected, graph=label,
                        mode=mode, keys=keys, graph_pickler_bytes=len(data))), flush=True)
            self.assertFalse(probes(gm), 'compiled ranges must shed only dead offset probes')

    def test_real_disk_cache_two_processes_both_orders_cold_and_warm(self):
        import torch
        if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 13):
            return
        for selected in ((4, 4), (1, 8192)):
            rows = []
            for order in ((0, 1), (1, 0)):
                with tempfile.TemporaryDirectory(prefix='ba4-real-aot-') as root:
                    # Leave a real legacy nonce entry containing a wrong program.
                    poison = subprocess.run([sys.executable, '-B', __file__, '--poison-worker',
                        str(selected[0]), str(selected[1]), '99', root],
                        text=True, capture_output=True, timeout=180)
                    self.assertEqual(poison.returncode, 0, poison.stdout + poison.stderr)
                    artifacts = list(Path(root, 'aotautograd', '0.8444218515250481').iterdir())
                    self.assertTrue(artifacts, poison.stdout + poison.stderr)
                    old_bytes = {str(p): p.read_bytes() for p in artifacts if p.is_file()}
                    for phase in ('cold', 'warm'):
                        for rank in order:
                            env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
                            result = subprocess.run([sys.executable, '-B', __file__, '--disk-worker',
                                str(selected[0]), str(selected[1]), str(rank), root],
                                text=True, capture_output=True, env=env, timeout=180)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            row, = [json.loads(s.split(' ', 1)[1]) for s in result.stdout.splitlines()
                                    if s.startswith('J41X_REAL_DISK ')]
                            row.update(range=selected, order=order, phase=phase)
                            print('J41X_CACHE_BOUNDARY ' + json.dumps(row), flush=True)
                            rows.append(row)
                            self.assertEqual({str(p): p.read_bytes() for p in artifacts if p.is_file()}, old_bytes)
                            self.assertEqual(row['nonce_dirs'], ['0.8444218515250481'])
                            if phase == 'warm':
                                self.assertTrue(row['hit'], row)
            self.assertEqual(sum(r['correct'] for r in rows), 8, rows)
            print('J41X_CACHE_MATRIX ' + json.dumps(dict(range=selected, correct=8, total=8)), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--disk-worker', '--poison-worker'):
        disk_worker(tuple(map(int, sys.argv[2:4])), int(sys.argv[4]), sys.argv[5],
                    poison=sys.argv[1] == '--poison-worker')
    else:
        unittest.main()
