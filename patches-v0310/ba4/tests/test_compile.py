"""Real vLLM apply and Parameter classes under CPU fullgraph Dynamo.

Only distributed/platform initialization and the CUDA kernel are substituted.
The parameter implementation is the complete, unmodified fork source.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


class ServingCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        parameter_path = HERE / 'fixtures/compile/parameter.py'
        assert hashlib.sha256(parameter_path.read_bytes()).hexdigest() == (
            '1a4bbd7400fd1ba79e8e1e8d19666cd22881ec83c35e6e01b3fdc78faee23171')
        spec = importlib.util.spec_from_file_location('ba4_real_parameter', parameter_path)
        cls.parameter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.parameter
        distributed = types.ModuleType('vllm.distributed')
        distributed.get_tensor_model_parallel_rank = lambda: 0
        distributed.get_tensor_model_parallel_world_size = lambda: 2
        logger = types.ModuleType('vllm.logger')
        logger.init_logger = logging.getLogger
        platform = types.ModuleType('vllm.platforms')
        platform.current_platform = types.SimpleNamespace(use_sync_weight_loader=lambda: False)
        cls.platform = platform
        with mock.patch.dict(sys.modules, {'vllm.distributed': distributed, 'vllm.logger': logger}):
            spec.loader.exec_module(cls.parameter)
        kernels = types.ModuleType('vllm_ba4.kernels')
        kernels.run_out = lambda name, x, weight, out, partial: torch.mm(x, weight.t(), out=out)
        # The test_cpu fake runtime deliberately restores this module.
        sys.modules['vllm_ba4.kernels'] = kernels
        cls.runtime = importlib.import_module('vllm_ba4.runtime')
        if hasattr(cls.runtime, 'ba4_dispatch'):
            @cls.runtime.ba4_dispatch.register_kernel('cpu')
            def cpu_dispatch(x, weight, partial, name, prefix, x_stride):
                return cls.runtime.ba4_dispatch._init_fn(x, weight, partial, name, prefix, x_stride)

        @cls.runtime.ba4_out.register_kernel('cpu')
        def cpu_out(x, weight, out, partial, name, prefix, x_stride=None):
            # Run the production opaque-op body, including live validation
            # and logging, replacing only the CUDA kernel launch.
            cls.runtime.ba4_out._init_fn(x, weight, out, partial, name, prefix, x_stride)

        spec = importlib.util.spec_from_file_location('ba4_make_compile', HERE / 'make_layer.py')
        make = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(make)
        source = make.changes()['vllm/model_executor/layers/linear.py'][1]
        tree = ast.parse(source)
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'UnquantizedLinearMethod')
        cls.config = types.SimpleNamespace(kernel_config=types.SimpleNamespace(linear_backend='auto'),
            scheduler_config=types.SimpleNamespace(max_num_batched_tokens=64),
            compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=32))
        def default_unquantized_gemm(layer, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)
        ns = {'torch': torch, 'envs': types.SimpleNamespace(
            VLLM_BATCH_INVARIANT=False, VLLM_SM12X_BA_M4=True),
            'get_current_vllm_config_or_none': lambda: cls.config,
            'dispatch_unquantized_gemm': lambda backend: default_unquantized_gemm,
            'current_platform': types.SimpleNamespace(is_cuda=lambda: True,
                is_cpu=lambda: False, is_xpu=lambda: False, is_cuda_alike=lambda: True)}
        for name in ('__init__', 'apply', 'process_weights_after_loading'):
            node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'real_vllm_linear.py', 'exec'), ns)
        cls.apply = staticmethod(ns['apply'])
        cls.initialize = staticmethod(ns['__init__'])
        cls.postload = staticmethod(ns['process_weights_after_loading'])

    def tearDown(self):
        self.torch._dynamo.reset()

    def layer(self, kind='ModelWeightParameter', profile=False, rank=0):
        torch = self.torch
        kwargs = {'data': torch.randn(48, 5120, dtype=torch.bfloat16),
                  'weight_loader': lambda *args: None}
        if kind == 'ModelWeightParameter':
            kwargs.update(input_dim=1, output_dim=0)
        with mock.patch.dict(sys.modules, {'vllm.platforms': self.platform}):
            w = getattr(self.parameter, kind)(**kwargs)
        layer = torch.nn.Module()
        layer.register_parameter('weight', w)
        layer.prefix = 'model.layers.0.linear_attn.in_proj_ba'
        layer.bias, layer.tp_size, layer.tp_rank, layer.output_sizes = None, 2, rank, [48, 48]
        layer.quant_method = types.SimpleNamespace()
        self.initialize(layer.quant_method)
        # The same final regression can be replayed against the old complete
        # make_layer/runtime, reaching the original storage_offset failure.
        if hasattr(layer.quant_method, '_sm12x_ba4_capacity'):
            self.assertEqual(layer.quant_method._sm12x_ba4_capacity, 64)
        from test_cpu import valid_report
        report = valid_report()
        report['rank'] = rank
        report['code_sha256'] = self.runtime.code_sha256()
        report['fingerprint'] = {'cpu_fixture': True}
        if profile:
            report['verdict'] = 'PROVISIONAL'
        # Exercise the actual postload hook. Bypass GPU eligibility/admission
        # evidence only; all tensor metadata/identity and State setup are real.
        with torch.no_grad(), mock.patch.dict('os.environ', {
                'VLLM_BATCH_INVARIANT': '0', 'VLLM_SM12X_BA_M4_RESULT': 'fixture',
                'VLLM_SM12X_BA_M4_PROFILE_ONLY': str(int(profile))}), \
             mock.patch.object(self.runtime, 'weight_reason', return_value=None), \
             mock.patch.object(self.runtime, 'admission_reason', return_value=None), \
             mock.patch.object(self.runtime, 'fingerprint', return_value=report['fingerprint']), \
             mock.patch.object(self.runtime, 'tensor_sha256', return_value='a'*64), \
             mock.patch.object(self.runtime, '_read_report', side_effect=lambda path: report
                 if path == 'fixture' else {'binding': report['serving_evidence']['binding'],
                                           'fingerprints': [report['fingerprint']]*2}):
            self.postload(layer.quant_method, layer)
        self.assertIsNotNone(layer._sm12x_ba4)
        if hasattr(layer._sm12x_ba4, 'buffer'):
            self.assertEqual(tuple(layer._sm12x_ba4.buffer.shape), (64, 48))
        return layer

    def compile_apply(self, layer, *, guardless=False, dynamic=False):
        """Use the fork wrapper's exact guard_filter_fn when guards are off."""
        graphs = []
        def backend(gm, inputs, **options):
            graphs.append(gm)
            return gm.forward
        options = {'guard_filter_fn': lambda entries: [False for _ in entries]} if guardless else {}
        compiled = self.torch.compile(lambda x: self.apply(layer.quant_method, layer, x),
            backend=backend, fullgraph=True, dynamic=dynamic, options=options)
        return compiled, graphs

    def test_real_parameter_apply_fullgraph_without_offset_escape_hatch(self):
        torch = self.torch
        for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
            for profile in (False, True):
                with self.subTest(parameter=kind, profile=profile), torch.no_grad():
                    torch._dynamo.reset()
                    graphs = []
                    def backend(gm, inputs):
                        graphs.append(gm)
                        return gm.forward
                    layer = self.layer(kind, profile)
                    def default_unquantized_gemm(layer, x, weight, bias):
                        return torch.nn.functional.linear(x, weight, bias)
                    gemm = types.SimpleNamespace(_gemm_impl=default_unquantized_gemm)
                    def forward(x):
                        return self.apply(gemm, layer, x)
                    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
                    x = torch.randn(4, 5120, dtype=torch.bfloat16)
                    for _ in range(2):
                        x.neg_()
                        result = compiled(x)
                        self.assertEqual(result.data_ptr(), layer._sm12x_ba4.out.data_ptr())
                        self.assertTrue(torch.equal(result, x @ layer.weight.data.t()))
                    self.assertEqual(len(graphs), 1)
                    targets = [str(n.target) for n in graphs[0].graph.nodes if n.op == 'call_function']
                    self.assertIn('vllm.sm12x_ba4_out.default', targets)

    def test_vllm_guardless_backed_dynamic_batch_selects_m4_and_keeps_fallback_outputs(self):
        torch = self.torch
        for first in (4, 32):
            torch._dynamo.reset()
            with self.subTest(first_batch=first), torch.no_grad():
                layer = self.layer(profile=True)
                self.assertEqual(layer.quant_method._sm12x_ba4_capacity, 64)
                self.assertEqual(tuple(layer._sm12x_ba4.buffer.shape), (64, 48))
                # This is vLLM wrapper.py/decorators.py's exact combination:
                # fullgraph, dynamic=False, backed marks, evaluate_guards=False.
                compiled, graphs = self.compile_apply(layer, guardless=True)
                retained = []
                for m in (first, 4, 8, 32, 65, 4):
                    x = torch.randn(m, 5120, dtype=torch.bfloat16)
                    torch._dynamo.mark_dynamic(x, 0)
                    self.runtime._used.clear()
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                        result = compiled(x)
                    self.assertEqual(tuple(result.shape), (m, 48))
                    self.assertNotEqual(result.data_ptr(), layer._sm12x_ba4.out.data_ptr())
                    self.assertTrue(torch.equal(result, x @ layer.weight.data.t()))
                    self.assertEqual(kernel.call_count, int(m == 4))
                    self.assertEqual(bool(self.runtime._used), m == 4)
                    retained.append((result, result.clone()))
                    for value, saved in retained:
                        self.assertTrue(torch.equal(value, saved))
                self.assertEqual(len(graphs), 1)
                self.assertIn('vllm.sm12x_ba4_dispatch.default', str(graphs[0].graph))

    def test_static_fullgraph_and_dynamic_input_alignment_run_real_op_body(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer(profile=True)
            method = layer.quant_method
            def forward(x):
                return self.apply(method, layer, x)
            for dynamic in (False, True):
                torch._dynamo.reset()
                compiled = torch.compile(forward, backend='eager', fullgraph=True, dynamic=dynamic)
                for offset in (1, 0, 8):
                    torch._dynamo.reset()
                    self.runtime._used.clear()
                    x = torch.randn(4*5120+offset, dtype=torch.bfloat16).as_strided(
                        (4, 5120), (5120, 1), offset)
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                        result = compiled(x)
                    self.assertTrue(torch.equal(result, torch.nn.functional.linear(x, layer.weight.data)))
                    selected = offset % 8 == 0
                    self.assertEqual(result.data_ptr() == layer._sm12x_ba4.out.data_ptr(), selected and not dynamic)
                    self.assertEqual(kernel.call_count, int(selected))
                    self.assertEqual(bool(self.runtime._used), selected)

    def test_aot_fullgraph_preserves_mutation_alias_and_changed_inputs(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer(profile=True)
            method = layer.quant_method
            def forward(x):
                return self.apply(method, layer, x)
            compiled = torch.compile(forward, backend='aot_eager', fullgraph=True, dynamic=False)
            x = torch.randn(4, 5120, dtype=torch.bfloat16)
            for _ in range(2):
                x.neg_()
                result = compiled(x)
                self.assertEqual(result.data_ptr(), layer._sm12x_ba4.out.data_ptr())
                self.assertTrue(torch.equal(result, x @ layer.weight.data.t()))
            layer.weight.data = torch.zeros_like(layer.weight.data)
            self.assertTrue(torch.equal(compiled(x), torch.zeros(4, 48, dtype=torch.bfloat16)))

    def test_compiled_fallbacks_match_default_baseline(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer()
            method = layer.quant_method
            def forward(x, bias=None):
                return self.apply(method, layer, x, bias)
            compiled = torch.compile(forward, backend='eager', fullgraph=True, dynamic=True)
            inputs = [torch.randn(m, 5120, dtype=torch.bfloat16) for m in (1, 8, 12, 16, 32)]
            inputs += [torch.randn(4, 5120),
                       torch.randn(5120, 4, dtype=torch.bfloat16).t(),
                       torch.randn(4, 5120, dtype=torch.bfloat16, requires_grad=True)]
            for x in inputs:
                # Each fallback remains a complete full graph, including the
                # real Parameter subclass in default_unquantized_gemm.
                torch._dynamo.reset()
                with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                    # FP32 input needs a matching baseline weight; dtype guard
                    # itself is tested below through try_apply to avoid an
                    # unrelated F.linear mixed-dtype error.
                    if x.dtype != layer.weight.dtype:
                        route = torch.compile(lambda x: self.runtime.try_apply(layer, x, None),
                                              backend='eager', fullgraph=True)
                        self.assertIsNone(route(x))
                        continue
                    result = compiled(x)
                kernel.assert_not_called()
                self.assertTrue(torch.equal(result, torch.nn.functional.linear(x, layer.weight, None)))
            x = torch.randn(4, 5120, dtype=torch.bfloat16)
            bias = torch.randn(48, dtype=torch.bfloat16)
            self.assertTrue(torch.equal(compiled(x, bias), torch.nn.functional.linear(x, layer.weight, bias)))
            layer._sm12x_ba4 = None
            self.assertTrue(torch.equal(compiled(x), torch.nn.functional.linear(x, layer.weight)))

    def test_compiled_weight_identity_guard(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer()
            method = layer.quant_method
            def forward(x):
                return self.apply(method, layer, x)
            compiled = torch.compile(forward, backend='eager', fullgraph=True)
            x = torch.randn(4, 5120, dtype=torch.bfloat16)
            self.assertTrue(torch.equal(compiled(x), x @ layer.weight.data.t()))
            layer.weight = torch.nn.Parameter(torch.zeros_like(layer.weight.data), requires_grad=False)
            with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                result = compiled(x)
            kernel.assert_not_called()
            self.assertTrue(torch.equal(result, torch.zeros(4, 48, dtype=torch.bfloat16)))

    def mutate(self, layer, mutation):
        torch = self.torch
        if mutation == 'prefix':
            layer.prefix = 'unrelated.layer'
        elif mutation == 'grad':
            layer.weight.requires_grad_(True)
        elif mutation == 'stride':
            layer.weight.data = layer.weight.data.t().contiguous().t()
        elif mutation == 'offset':
            layer.weight.data = torch.zeros(48*5120+1, dtype=torch.bfloat16).as_strided(
                (48, 5120), (5120, 1), 1)
        elif mutation == 'rebind':
            layer.weight.data = torch.zeros_like(layer.weight.data)
        else:
            raise AssertionError(mutation)

    def test_live_mutations_between_postload_and_first_fullgraph_compile(self):
        torch = self.torch
        for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
            for mutation in ('prefix', 'grad', 'stride', 'offset'):
                with self.subTest(kind=kind, mutation=mutation), torch.no_grad():
                    torch._dynamo.reset()
                    layer = self.layer(kind, profile=True)
                    self.mutate(layer, mutation)
                    x = torch.randn(4, 5120, dtype=torch.bfloat16)
                    self.assertIsNone(self.runtime.try_apply(layer, x, None))
                    compiled, _ = self.compile_apply(layer, guardless=True)
                    route = torch.compile(lambda x: self.runtime.try_apply(layer, x, None),
                                          backend='eager', fullgraph=True)
                    self.assertIsNone(route(x))
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                        first = compiled(x)
                        saved = first.clone()
                        compiled(-x)
                    kernel.assert_not_called()
                    self.assertTrue(torch.equal(first, saved))
                    self.assertTrue(torch.equal(first, torch.nn.functional.linear(x, layer.weight)))

    def test_same_object_storage_rebinding_reads_live_parameter(self):
        torch = self.torch
        for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
            for when in ('before_compile', 'after_compile'):
                with self.subTest(kind=kind, when=when), torch.no_grad():
                    torch._dynamo.reset()
                    layer = self.layer(kind, profile=True)
                    weight = layer.weight
                    x = torch.randn(4, 5120, dtype=torch.bfloat16)
                    compiled, graphs = self.compile_apply(layer, guardless=True)
                    if when == 'after_compile':
                        compiled(x)
                    self.mutate(layer, 'rebind')
                    self.assertIs(layer.weight, weight)
                    for apply in (lambda x: self.apply(layer.quant_method, layer, x), compiled):
                        self.assertTrue(torch.equal(apply(x), torch.zeros(4, 48, dtype=torch.bfloat16)))
                    self.assertEqual(len(graphs), 1)

    def test_guardless_capture_op_fails_closed_on_live_weight_metadata_changes(self):
        torch = self.torch
        for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
            for mutation in ('grad', 'stride', 'offset'):
                with self.subTest(kind=kind, mutation=mutation), torch.no_grad():
                    torch._dynamo.reset()
                    layer = self.layer(kind, profile=True)
                    compiled, graphs = self.compile_apply(layer, guardless=True)
                    x = torch.randn(4, 5120, dtype=torch.bfloat16)
                    compiled(x)
                    self.mutate(layer, mutation)
                    self.assertIsNone(self.runtime.try_apply(layer, x, None))
                    self.runtime._used.clear()
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                        with self.assertRaisesRegex(RuntimeError, 'BA4 live weight'):
                            compiled(x)
                    kernel.assert_not_called()
                    self.assertFalse(self.runtime._used)
                    self.assertEqual(len(graphs), 1)

    def test_guardless_prefix_mutation_is_trace_specialization_like_original(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer(profile=True)
            compiled, graphs = self.compile_apply(layer, guardless=True)
            x = torch.randn(4, 5120, dtype=torch.bfloat16)
            compiled(x)
            self.mutate(layer, 'prefix')
            self.assertIsNone(self.runtime.try_apply(layer, x, None))
            # The original Python prefix predicate also does not run again
            # after guard-disabled tracing, or during a CUDA graph replay.
            self.assertTrue(torch.equal(compiled(x), torch.nn.functional.linear(x, layer.weight)))
            self.assertEqual(len(graphs), 1)
            torch._dynamo.reset()
            fresh, _ = self.compile_apply(layer, guardless=True)
            with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                fresh(x)
            kernel.assert_not_called()

    def test_intermediate_inputs_observe_current_offsets_during_trace(self):
        torch = self.torch
        for offset in (0, 1, 8):
            with self.subTest(offset=offset), torch.no_grad():
                torch._dynamo.reset()
                layer = self.layer(profile=True)
                # Slice columns to produce both aligned and unaligned
                # intermediate views with the validated padded row stride.
                layer._sm12x_ba4.x_stride = (5136, 1)
                def forward(x):
                    return self.apply(layer.quant_method, layer, x.neg()[:, offset:offset+5120])
                compiled = torch.compile(forward, backend='eager', fullgraph=True)
                x = torch.randn(4, 5136, dtype=torch.bfloat16)
                with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                    result = compiled(x)
                self.assertTrue(torch.equal(result, torch.nn.functional.linear(
                    x.neg()[:, offset:offset+5120], layer.weight)))
                self.assertEqual(kernel.call_count, int(offset % 8 == 0))

    def test_offset_metadata_real_fake_and_operator_contract(self):
        """Review probe: correct tensor metadata does not imply a live integer."""
        torch = self.torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        op = torch.ops.vllm.sm12x_ba4_offset_metadata.default
        self.assertFalse(op._schema.is_mutable)
        self.assertIsNone(op._schema.returns[0].alias_info)
        self.assertIsNone(op._schema.arguments[0].alias_info)
        def metadata(value):
            return (tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                    value.dtype, value.device)
        for offset in (0, 1, 8, 2**40+7):
            with self.subTest(offset=offset):
                # Empty views exercise huge offsets without huge allocations.
                x = torch.empty(0).as_strided((0,), (1,), offset)
                version = x._version
                actual, other = op(x), op(x)
                self.assertIsNot(actual, other)
                self.assertIsNone(actual._base)
                self.assertFalse(torch._C._is_alias_of(actual, x))
                self.assertFalse(torch._C._is_alias_of(actual, other))
                self.assertEqual(x._version, version)
                self.assertEqual(actual.numel(), 0)
                self.assertEqual(actual.untyped_storage().nbytes(), 0)
                self.assertEqual(actual.stride(), (offset+1,))
                self.assertEqual(actual.storage_offset(), 0)
                mode = FakeTensorMode(shape_env=ShapeEnv())
                fake = mode.from_tensor(x)
                with mode:
                    expected = op(fake)
                self.assertEqual(metadata(actual), metadata(expected))
                functional = torch.func.functionalize(lambda value: op(value))(x)
                self.assertEqual(metadata(actual), metadata(functional))
                print('J41Y_METADATA_CONTRACT ' + json.dumps(dict(offset=offset,
                    stride=actual.stride(), fresh=True, no_alias=True, unmutated=True,
                    fake_equal=True, functionalization=True)), flush=True)
        for offset in (0, 1, 8, 100001):
            x = torch.arange(offset+16, dtype=torch.float32)[offset:offset+16]
            checks = torch.library.opcheck(op, (x,))
            self.assertTrue(all(value == 'SUCCESS' for value in checks.values()), checks)
            print('J41Y_OPCHECK ' + json.dumps(dict(offset=offset, checks=checks)), flush=True)

    def test_offset_metadata_integer_is_known_trace_specialization(self):
        """Keep the tensor live and assert the review's cached integer behavior."""
        torch = self.torch
        op = torch.ops.vllm.sm12x_ba4_offset_metadata.default
        for compiler in ('aot_eager', 'inductor'):
            with self.subTest(compiler=compiler):
                torch._dynamo.reset()
                call = torch.compile(lambda x: (op(x), op(x).stride(0)),
                                     backend=compiler, fullgraph=True)
                for offset in (0, 1, 8, 100001, 1):
                    x = torch.arange(offset+16, dtype=torch.float32)[offset:offset+16]
                    result, integer_stride = call(x)
                    self.assertEqual(result.stride(), (offset+1,))
                    # This integer is specialized at the first offset (zero),
                    # even with ordinary guards enabled. It is not an accessor.
                    self.assertEqual(integer_stride, 1)
                    print('J41Y_TRACE_SPECIALIZATION ' + json.dumps(dict(
                        compiler=compiler, offset=offset, tensor_stride=result.stride(),
                        returned_integer_stride=integer_stride,
                        known_specialization=True)), flush=True)

    def test_compiled_reuse_validates_live_offsets_before_kernel_and_logging(self):
        """Aligned reuse works; stale trace admission still fails closed at 1."""
        torch = self.torch
        from compile_wrapper import wrap
        from piecewise_wrapper import backend
        for compiler in (None, 'aot_eager', 'inductor'):
            with self.subTest(compiler=compiler), torch.no_grad():
                torch._dynamo.reset()
                layer = self.layer(profile=True)
                call, captured = wrap(self, layer, backend(compiler, []))
                for offset in (0, 8, 1, 0):
                    x = torch.randn(4*5120+offset, dtype=torch.bfloat16).as_strided(
                        (4, 5120), (5120, 1), offset)
                    self.runtime._used.clear()
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel, \
                         mock.patch.object(self.runtime.logger, 'warning') as logged:
                        if offset == 1:
                            with self.assertRaisesRegex(RuntimeError, 'BA4 live input refused'):
                                call(x)
                            kernel.assert_not_called()
                            logged.assert_not_called()
                            self.assertFalse(self.runtime._used)
                        else:
                            actual = call(x)
                            self.assertTrue(torch.equal(actual,
                                torch.nn.functional.linear(x, layer.weight)))
                            kernel.assert_called_once()
                            logged.assert_called_once()
                            self.assertEqual(len(self.runtime._used), 1)
                    print('J41Y_PRODUCTION_REBOUND_INPUT ' + json.dumps(dict(
                        compiler=compiler, offset=offset, selected_calls=kernel.call_count,
                        status='fail_closed_before_kernel_and_logging' if offset == 1
                               else 'numerically_correct')), flush=True)
                self.assertEqual(len(captured), 1)

    def test_guardless_capture_rechecks_input_offset_and_stride(self):
        torch = self.torch
        for mutation in ('offset', 'stride', 'grad'):
            with self.subTest(mutation=mutation), torch.no_grad():
                torch._dynamo.reset()
                layer = self.layer(profile=True)
                x = torch.randn(4, 5120, dtype=torch.bfloat16)
                compiled, graphs = self.compile_apply(layer, guardless=True)
                compiled(x)
                if mutation == 'offset':
                    x = torch.randn(4*5120+1, dtype=torch.bfloat16).as_strided(
                        (4, 5120), (5120, 1), 1)
                elif mutation == 'stride':
                    x = x.t().contiguous().t()
                else:
                    x.requires_grad_(True)
                self.assertIsNone(self.runtime.try_apply(layer, x, None))
                self.runtime._used.clear()
                with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                    with self.assertRaisesRegex(RuntimeError, 'BA4 live input'):
                        compiled(x)
                kernel.assert_not_called()
                self.assertFalse(self.runtime._used)
                self.assertEqual(len(graphs), 1)

    def test_bare_microbenchmark_controls_are_separate_from_serving_refusals(self):
        torch = self.torch
        with torch.no_grad():
            layer = self.layer(profile=True)
            for m in (8, 12, 16, 32):
                x = torch.randn(m, 5120, dtype=torch.bfloat16)
                self.assertIsNone(self.runtime.try_apply(layer, x, None))
                out = torch.empty(m, 48, dtype=torch.bfloat16)
                partial = torch.empty(40, m, 48)
                self.runtime._used.clear()
                with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                    self.runtime.ba4_out(x, layer.weight, out, partial, 'torch_mm', '')
                kernel.assert_called_once()
                self.assertTrue(torch.equal(out, torch.nn.functional.linear(x, layer.weight)))
                self.assertFalse(self.runtime._used)

    def test_rejected_static_shapes_and_alignment_keep_original_output_lifetime(self):
        torch = self.torch
        for m, offset in ((8, 0), (4, 1), (65, 0)):
            for guardless in (False, True):
                with self.subTest(m=m, offset=offset, guardless=guardless), torch.no_grad():
                    torch._dynamo.reset()
                    layer = self.layer(profile=True)
                    x = torch.randn(m*5120+offset, dtype=torch.bfloat16).as_strided(
                        (m, 5120), (5120, 1), offset)
                    y = torch.randn_like(x)
                    if offset:
                        y = torch.randn(m*5120+offset, dtype=torch.bfloat16).as_strided(
                            (m, 5120), (5120, 1), offset)
                    route = torch.compile(lambda x: self.runtime.try_apply(layer, x, None),
                                          backend='eager', fullgraph=True)
                    self.assertIsNone(route(x))
                    compiled, _ = self.compile_apply(layer, guardless=guardless)
                    self.runtime._used.clear()
                    with mock.patch.object(self.runtime, 'run_out', wraps=self.runtime.run_out) as kernel:
                        first = compiled(x)
                        saved = first.clone()
                        second = compiled(y)
                    kernel.assert_not_called()
                    self.assertFalse(self.runtime._used)
                    self.assertNotEqual(first.data_ptr(), layer._sm12x_ba4.out.data_ptr())
                    self.assertTrue(torch.equal(first, saved))
                    self.assertTrue(torch.equal(first, torch.nn.functional.linear(x, layer.weight)))
                    self.assertTrue(torch.equal(second, torch.nn.functional.linear(y, layer.weight)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
