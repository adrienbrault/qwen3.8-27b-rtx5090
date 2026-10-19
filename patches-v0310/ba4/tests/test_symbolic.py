"""Symbolic BA4 regressions through the complete real vLLM compile wrapper."""
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE / 'tests'), str(HERE)]


class SymbolicCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_compile import ServingCompileTests
        ServingCompileTests.setUpClass()
        cls.h = ServingCompileTests()
        cls.torch = cls.h.torch

    def tearDown(self):
        self.torch._dynamo.reset()

    def test_real_wrapper_symbolic_m4_and_exact_fresh_fallbacks(self):
        from compile_wrapper import wrap
        torch, h = self.torch, self.h
        for backend in (None, 'aot_eager', 'inductor'):
            for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
                for first in (4, 32):
                    with self.subTest(backend=backend, kind=kind, first=first), torch.no_grad():
                        torch._dynamo.reset()
                        layer = h.layer(kind, profile=True)
                        compiled, graphs = wrap(h, layer, backend)
                        retained = []
                        for m in (first, 4, 8, 12, 65, 4):
                            x = torch.randn(m, 5120, dtype=torch.bfloat16)
                            h.runtime._used.clear()
                            with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as kernel:
                                out = compiled(x)
                            self.assertTrue(torch.equal(out, torch.nn.functional.linear(x, layer.weight)))
                            self.assertEqual(kernel.call_count, int(m == 4))
                            self.assertEqual(bool(h.runtime._used), m == 4)
                            self.assertNotEqual(out.data_ptr(), layer._sm12x_ba4.out.data_ptr())
                            retained.append((out, out.clone()))
                            for value, saved in retained:
                                self.assertTrue(torch.equal(value, saved))
                        self.assertEqual(len(graphs), 1)
                        self.assertIn('vllm.sm12x_ba4_dispatch.default', str(graphs[0].graph))
                        print('J41Q_SYMBOLIC_GRAPH ' + json.dumps(dict(backend=backend, kind=kind,
                            first=first, graph=str(graphs[0].graph))), flush=True)

    def test_real_wrapper_live_weight_refusals_and_storage_rebinding(self):
        from compile_wrapper import wrap
        torch, h = self.torch, self.h
        for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
            for when in ('before', 'after'):
                for mutation in ('prefix', 'grad', 'stride', 'offset', 'rebind'):
                    with self.subTest(kind=kind, when=when, mutation=mutation), torch.no_grad():
                        torch._dynamo.reset()
                        layer = h.layer(kind, profile=True)
                        compiled, graphs = wrap(h, layer)
                        x = torch.randn(4, 5120, dtype=torch.bfloat16)
                        if when == 'after':
                            compiled(torch.randn(32, 5120, dtype=torch.bfloat16))
                        h.mutate(layer, mutation)
                        h.runtime._used.clear()
                        with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as kernel:
                            if when == 'after' and mutation in ('grad', 'stride', 'offset'):
                                with self.assertRaisesRegex(RuntimeError, 'BA4 live weight'):
                                    compiled(x)
                            else:
                                out = compiled(x)
                                self.assertTrue(torch.equal(out, torch.nn.functional.linear(x, layer.weight)))
                        selected = mutation == 'rebind' or (when == 'after' and mutation == 'prefix')
                        self.assertEqual(kernel.call_count, int(selected))
                        self.assertEqual(bool(h.runtime._used), selected)
                        self.assertEqual(len(graphs), 1)
                        # Python prefix remains trace-specialized exactly as
                        # j41m; fresh trace/eager calls observe the live prefix.
                        if mutation != 'rebind':
                            self.assertIsNone(h.runtime.try_apply(layer, x, None))

    def test_real_wrapper_live_input_refusals(self):
        from compile_wrapper import wrap
        torch, h = self.torch, self.h
        for mutation in ('offset', 'stride', 'grad'):
            with self.subTest(mutation=mutation), torch.no_grad():
                torch._dynamo.reset()
                layer = h.layer(profile=True)
                compiled, graphs = wrap(h, layer)
                compiled(torch.randn(32, 5120, dtype=torch.bfloat16))
                if mutation == 'offset':
                    x = torch.randn(4*5120+1, dtype=torch.bfloat16).as_strided((4, 5120), (5120, 1), 1)
                elif mutation == 'stride':
                    x = torch.randn(5120, 4, dtype=torch.bfloat16).t()
                else:
                    x = torch.randn(4, 5120, dtype=torch.bfloat16, requires_grad=True)
                h.runtime._used.clear()
                with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as kernel:
                    with self.assertRaisesRegex(RuntimeError, 'BA4 live input'):
                        compiled(x)
                kernel.assert_not_called()
                self.assertFalse(h.runtime._used)
                self.assertEqual(len(graphs), 1)

    def test_guard_disabled_symbolic_prepared_device_refusal_before_allocation(self):
        from compile_wrapper import wrap
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        torch, h = self.torch, self.h
        with torch.no_grad():
            layer = h.layer(profile=True)
            compiled, graphs = wrap(h, layer)
            compiled(torch.randn(32, 5120, dtype=torch.bfloat16))
            self.assertEqual(len(graphs), 1)
            mode = FakeTensorMode()
            def tensor(shape, dtype, device):
                return FakeTensor(mode, torch.empty(shape, dtype=dtype, device='meta'),
                                  torch.device(device))
            x = tensor((4, 5120), torch.bfloat16, 'cuda:1')
            weight = tensor((48, 5120), torch.bfloat16, 'cuda:1')
            anchored_weight = tensor((48, 5120), torch.bfloat16, 'cuda:0')
            partial = tensor((40, 4, 48), torch.float32, 'cuda:0')
            # Execute the graph produced by the actual guard-disabled wrapper,
            # replacing only fake dispatch with the production validation body.
            class Live(torch.fx.Interpreter):
                def call_function(self, target, args, kwargs):
                    if target == torch.ops.vllm.sm12x_ba4_dispatch.default:
                        return h.runtime.ba4_dispatch._init_fn(*args, **kwargs)
                    return super().call_function(target, args, kwargs)
            args = []
            for n in graphs[0].graph.nodes:
                if n.op != 'placeholder':
                    continue
                args.append(4 if n.name.startswith('s') else weight if 'weight' in n.name
                            else partial if 'partial' in n.name else x)
            h.runtime._used.clear()
            with mode, mock.patch.object(h.runtime, 'ba4_out', h.runtime.ba4_out._init_fn), \
                    mock.patch.object(h.runtime, 'run_out') as selected, \
                    mock.patch.object(torch, 'empty', wraps=torch.empty) as allocate, \
                    mock.patch.object(h.runtime.logger, 'warning') as log:
                with self.assertRaisesRegex(RuntimeError, 'BA4 live weight metadata refused'):
                    Live(graphs[0]).run(*args)
                allocate.assert_not_called()
                selected.assert_not_called()
                log.assert_not_called()
            self.assertFalse(h.runtime._used)
            input_args = [anchored_weight if v is weight else v for v in args]
            with mode, mock.patch.object(h.runtime, 'ba4_out', h.runtime.ba4_out._init_fn), \
                    mock.patch.object(h.runtime, 'run_out') as selected, \
                    mock.patch.object(torch, 'empty', wraps=torch.empty) as allocate, \
                    mock.patch.object(h.runtime.logger, 'warning') as log:
                with self.assertRaisesRegex(RuntimeError, 'BA4 live input refused'):
                    Live(graphs[0]).run(*input_args)
                allocate.assert_not_called()
                selected.assert_not_called()
                log.assert_not_called()
            self.assertFalse(h.runtime._used)

    def test_real_worker_tagged_symbolic_profile_used(self):
        for rank in (0, 1):
            result = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker', str(rank)],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [s for s in result.stdout.splitlines() if 'SM12X BA4 PROFILE_USED:' in s]
            self.assertEqual(len(rows), 1, result.stdout)
            row = rows[0]
            self.assertRegex(row, rf'\(Worker_TP{rank} pid=\d+\) WARNING .*\[runtime.py:\d+\] SM12X BA4 PROFILE_USED:')
            self.assertNotIn('SM12X BA4', result.stderr)
            print('J41Q_TAGGED_SYMBOLIC ' + row, flush=True)


def worker(rank):
    # Use the same complete, pinned vLLM logger/prefix source as j41o.
    import logging
    import types
    from test_logging import load, function, FIXTURE
    vllm = types.ModuleType('vllm'); vllm.__path__ = []
    envs = types.ModuleType('vllm.envs')
    for k, v in dict(VLLM_LOGGING_PREFIX='', VLLM_LOGGING_LEVEL='INFO',
        VLLM_LOGGING_STREAM='ext://sys.stdout', VLLM_LOGGING_COLOR='0', NO_COLOR=True,
        FORCE_COLOR=False, VLLM_CONFIGURE_LOGGING=True, VLLM_LOGGING_CONFIG_PATH=None).items():
        setattr(envs, k, v)
    vllm.envs = envs
    sys.modules.update({'vllm': vllm, 'vllm.envs': envs})
    vllm.logging_utils = load('vllm.logging_utils', FIXTURE / 'logging_utils/formatter.py')
    logger = load('vllm.logger', FIXTURE / 'logger.py'); vllm.logger = logger
    logger._configure_vllm_root_logger()
    decorate = function(FIXTURE / 'utils/system_utils.py', 'decorate_logs',
                        {'set_vllm_process_name': logger.set_vllm_process_name})
    group = lambda size, r: types.SimpleNamespace(world_size=size, rank_in_group=r)
    ns = dict(model_parallel_is_initialized=lambda: True,
        get_dp_group=lambda: group(1, 0), get_pp_group=lambda: group(1, 0),
        get_pcp_group=lambda: group(1, 0), get_tp_group=lambda: group(2, rank),
        get_dcp_group=lambda: group(1, 0), set_process_title=lambda **kw: None, decorate_logs=decorate)
    function(FIXTURE / 'v1/executor/multiproc_executor.py', 'setup_proc_title_and_log_prefix', ns, 'WorkerProc')(False)
    from test_compile import ServingCompileTests
    from compile_wrapper import wrap
    ServingCompileTests.setUpClass(); h = ServingCompileTests(); torch = h.torch
    with torch.no_grad():
        layer = h.layer(profile=True, rank=rank)
        compiled, graphs = wrap(h, layer)
        for m in (32, 8, 12):
            h.runtime._used.clear()
            compiled(torch.randn(m, 5120, dtype=torch.bfloat16))
            assert not h.runtime._used
        compiled(torch.randn(4, 5120, dtype=torch.bfloat16))
        assert h.runtime._used and len(graphs) == 1


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
