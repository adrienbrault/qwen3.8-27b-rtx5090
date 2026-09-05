#!/usr/bin/env python3
"""Usage: python3 deliver/test-0148.py .work/0148

Dependency-free tests of functions extracted from the applied tree. No vLLM,
torch, CUDA, or PCIe package is imported. Only the lazy fixed_config MAX_ROWS
import is removed; the constant is injected from its source via ast.literal_eval.
Function control flow otherwise executes unchanged against metadata/context spies.
These tests cannot establish GPU correctness or numerical equivalence.
"""
import ast
import __future__
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from itertools import groupby, product
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(sys.argv.pop(1)).resolve()
BASE = Path(__file__).resolve().parents[1]
ENV = 'vllm/envs.py'
COMM = 'vllm/distributed/device_communicators/pcie_ipc_all_reduce.py'
SPEC = 'vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py'
CG = 'vllm/v1/worker/gpu/cudagraph_utils.py'
KNOB = 'VLLM_SM12X_PCIE_IPC_MTP'
AR = 'VLLM_SM12X_PCIE_IPC_AR'
OLD_REFUSAL = 'only the sequential DFlash drafter is supported'


def function_node(relative, name, cls=None, root=ROOT):
    body = ast.parse((root / relative).read_text()).body
    if cls:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


fixed_tree = ast.parse((ROOT / 'pcie_ipc_ar21/fixed_config.py').read_text())
MAX_ROWS = next(ast.literal_eval(n.value) for n in fixed_tree.body
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'MAX_ROWS' for t in n.targets))


class InjectFixedConstant(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if (node.module == 'pcie_ipc_ar21.fixed_config'
                and [(n.name, n.asname) for n in node.names] == [('MAX_ROWS', None)]):
            return None
        raise AssertionError(f'Unexpected import in isolated function: {ast.unparse(node)}')


def isolated(relative, name, ns, cls=None, root=ROOT):
    node = function_node(relative, name, cls, root)
    assert not node.decorator_list, 'Do not silently strip executable decorators'
    node = InjectFixedConstant().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    # Postpone annotations without loading application types.
    exec(compile(module, str(root / relative), 'exec',
                 flags=__future__.annotations.compiler_flag), ns)
    return ns[name]


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        ns = {'os': os}
        isolated(ENV, 'env_with_choices', ns)
        self.enabled = isolated(ENV, '_sm12x_pcie_ipc_mtp_enabled', ns)

    def test_strict_grammar(self):
        for value, expected in ((None, False), ('0', False), ('1', True)):
            with self.subTest(value=value), patch.dict(os.environ, {AR: '1'}, clear=True):
                if value is not None:
                    os.environ[KNOB] = value
                self.assertIs(self.enabled(), expected)
        for value in ('', 'true', 'True', '01', '2', '-1', ' 1', '1 ', 'yes'):
            with self.subTest(value=value), patch.dict(os.environ, {KNOB: value, AR: '1'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'Invalid value'):
                    self.enabled()

    def test_ar_dependency_requires_exactly_one(self):
        for value in (None, '0', '01', 'true', '', '2'):
            with self.subTest(value=value), patch.dict(os.environ, {KNOB: '1'}, clear=True):
                if value is not None:
                    os.environ[AR] = value
                with self.assertRaisesRegex(ValueError, 'requires VLLM_SM12X_PCIE_IPC_AR=1'):
                    self.enabled()

    def test_runtime_knob_excluded_from_compile_factors(self):
        node = function_node(ENV, 'compile_factors')
        ignored = next(ast.literal_eval(n.value) for n in node.body
                       if isinstance(n, ast.AnnAssign) and n.target.id == 'ignored_factors')
        self.assertIn(KNOB, ignored)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.envs = NS(VLLM_SM12X_PCIE_IPC_MTP=False)
        self.reason = isolated(COMM, '_disabled_reason', {
            'envs': self.envs,
            'torch': NS(bfloat16='bf16', version=NS(cuda='13.0'), cuda=NS(
                get_device_capability=lambda _: (12, 0),
                get_device_name=lambda _: 'NVIDIA GeForce RTX 5090')),
        }, 'PcieIpcAllReduce')
        self.draft = NS(dtype='bf16', get_hidden_size=lambda: 5120)
        self.spec = NS(method='mtp', draft_model_config=self.draft,
                       draft_parallel_config=NS(tensor_parallel_size=2))
        self.passes = NS(enable_sp=False, fuse_gemm_comms=False, fuse_allreduce_rms=False)
        self.owner = NS(config=NS(
            speculative_config=self.spec, parallel_config=NS(enable_dbo=False), lora_config=None,
            compilation_config=NS(pass_config=self.passes),
            model_config=NS(dtype='bf16', get_hidden_size=lambda: 5120,
                            is_moe=False, enable_sleep_mode=False)))
        self.custom = NS(disabled=False, mnnvl_only=False)

    def result(self):
        return self.reason(self.owner, True, self.custom, 'fake-device')

    def test_legacy_off_and_supported_mtp_on(self):
        self.assertEqual(self.result(), OLD_REFUSAL)
        self.envs.VLLM_SM12X_PCIE_IPC_MTP = True
        self.assertIsNone(self.result())
        self.envs.VLLM_SM12X_PCIE_IPC_MTP = False
        self.spec.method = 'dflash'
        self.assertIsNone(self.result())
        self.owner.config.speculative_config = None
        self.assertIsNone(self.result())

    def test_other_methods_keep_exact_refusal(self):
        for knob in (False, True):
            self.envs.VLLM_SM12X_PCIE_IPC_MTP = knob
            for method in ('eagle', 'eagle3', 'ngram', 'qwen3_5_mtp'):
                with self.subTest(knob=knob, method=method):
                    self.spec.method = method
                    self.assertEqual(self.result(), OLD_REFUSAL)

    def test_opt_in_requires_mtp(self):
        self.envs.VLLM_SM12X_PCIE_IPC_MTP = True
        for spec in (None, NS(method='dflash')):
            with self.subTest(spec=spec):
                self.owner.config.speculative_config = spec
                self.assertEqual(self.result(), 'VLLM_SM12X_PCIE_IPC_MTP=1 requires speculative method mtp')

    def test_unsupported_draft_metadata(self):
        self.envs.VLLM_SM12X_PCIE_IPC_MTP = True
        for dtype, hidden, tp in (('fp16', 5120, 2), ('bf16', 4096, 2), ('bf16', 5120, 1)):
            with self.subTest(dtype=dtype, hidden=hidden, tp=tp):
                self.draft.dtype = dtype
                self.draft.get_hidden_size = lambda: hidden
                self.spec.draft_parallel_config.tensor_parallel_size = tp
                self.assertEqual(self.result(), 'PCIe IPC MTP requires a bf16 hidden=5120 TP2 drafter')

    def test_fusion_and_sequence_parallel_are_refused(self):
        self.envs.VLLM_SM12X_PCIE_IPC_MTP = True
        for name in ('enable_sp', 'fuse_gemm_comms', 'fuse_allreduce_rms'):
            with self.subTest(pass_name=name):
                setattr(self.passes, name, True)
                self.assertEqual(self.result(), 'PCIe IPC MTP requires unfused tensor-parallel all-reduce dispatch')
                setattr(self.passes, name, False)


def manager_descriptors(query_len, prefill):
    """Execute the real manager descriptor selection using CPU metadata only."""
    modes = NS(FULL='FULL', PIECEWISE='PIECEWISE')
    mode = NS(decode_mode=lambda: modes.FULL,
              mixed_mode=lambda: modes.PIECEWISE if prefill else None,
              separate_routine=lambda: True)
    manager = NS(
        compilation_config=NS(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 320],
                              max_cudagraph_capture_size=320),
        cudagraph_mode=mode, max_num_reqs=16, decode_query_len=query_len,
        vllm_config=NS(speculative_config=NS(uses_dynamic_speculative_decoding=lambda: False)),
        varlen_decode=False, lora_capture_cases=[0], _capture_descs={}, _candidates={})
    init = isolated(CG, '_init_candidates', {
        'CUDAGraphMode': modes, 'defaultdict': defaultdict, 'product': product,
        'groupby': groupby, 'BatchExecutionDescriptor': NS,
        'round_up': lambda value, quantum: (value + quantum - 1) // quantum * quantum,
    }, 'CudaGraphManager')
    init(manager)
    return manager._capture_descs


class CaptureTests(unittest.TestCase):
    def fixture(self, enabled=True, steps=3, fused=False, old=False):
        events = []
        state = NS(depth=0, prepared=set(), contexts=0, disabled=False)

        def prepare(shapes, dtype):
            self.assertEqual(state.contexts, 0, 'must prepare before FIRST capture')
            events.append(('prepare', dtype, len(shapes)))
            state.prepared.update(shapes)

        prepare_draft = isolated(COMM, 'prepare_draft', {'MAX_ROWS': MAX_ROWS}, 'PcieIpcAllReduce')
        state._prepare = prepare
        state.prepare_draft = lambda hidden, dtype: prepare_draft(state, hidden, dtype)

        @contextmanager
        def capture():
            state.contexts += 1
            number = state.contexts
            self.assertEqual(state.depth, 0)
            events.append(('enter', number))
            state.depth += 1
            try:
                yield
            finally:
                state.depth -= 1
                events.append(('exit', number))

        state.capture = capture

        def manager(name, descs):
            def run(fn, *args, **kwargs):
                if enabled:
                    self.assertEqual(state.depth, 1, f'{name} capture not independently wrapped')
                    for descriptors in result._capture_descs.values():
                        for desc in descriptors:
                            rows = desc.num_tokens if name == 'prefill' else desc.num_reqs
                            self.assertIn((rows, 5120), state.prepared)
                events.append(('manager', name, fn))
            result = NS(_capture_descs=descs, use_breakable_cg=True, capture=run,
                        init_breakable_cg_runner=lambda _: events.append(('breakable', name)))
            return result

        owner = NS(
            method='mtp', num_speculative_steps=steps, use_fused_multi_step_decode=fused,
            max_num_reqs=16, dtype='bf16', draft_model_config=NS(get_hidden_size=lambda: 5120),
            prefill_cudagraph_manager=manager('prefill', manager_descriptors(steps + 1, True)),
            decode_cudagraph_manager=manager('decode', manager_descriptors(1, False)),
            last_token_indices=NS(zero_=lambda: events.append(('zero', 'last'))),
            idx_mapping=NS(zero_=lambda: events.append(('zero', 'mapping'))),
            on_prefill_begin=lambda n: events.append(('prefill_begin', n)),
            on_prefill_end=lambda n: events.append(('prefill_end', n)),
            on_multi_step_decode_begin=lambda n: events.append(('decode_begin', n)),
            on_multi_step_decode_end=lambda n: events.append(('decode_end', n)),
            _prefill='prefill_fn', _generate_draft='decode_fn', _generate_fused_drafts='fused_fn')
        for attribute in ('model', 'model_state', 'target_input_buffers', 'block_tables',
                          'target_attn_groups', 'kv_cache_config', 'input_buffers', 'attn_groups'):
            setattr(owner, attribute, attribute)

        def get_group():
            self.assertTrue(enabled, 'knob off must not consult the communicator')
            return NS(device_communicator=NS(pcie_ipc_comm=state))

        fn = isolated(SPEC, 'capture', {
            'envs': NS(VLLM_SM12X_PCIE_IPC_MTP=enabled), 'MAX_ROWS': MAX_ROWS,
            'get_tp_group': get_group, 'nullcontext': nullcontext,
            'logger': NS(info=lambda *args: events.append(('log', args))),
        }, 'AutoRegressiveSpeculator', root=BASE / 'src' if old else ROOT)
        return fn, owner, state, events

    def test_ns3_ns4_real_descriptors_and_both_capture_families(self):
        for steps in (3, 4):
            for fused in (False, True):
                with self.subTest(steps=steps, fused=fused):
                    fn, owner, state, events = self.fixture(steps=steps, fused=fused)
                    prefill = owner.prefill_cudagraph_manager._capture_descs
                    self.assertTrue(any(d.num_reqs is None and d.num_tokens == 320 for d in prefill['PIECEWISE']))
                    self.assertTrue(all(d.num_tokens == d.num_reqs * (steps + 1) for d in prefill['FULL']))
                    fn(owner)
                    self.assertEqual(state.prepared, {(r, 5120) for r in range(1, 321)})
                    self.assertEqual(events[0], ('prepare', 'bf16', 320))
                    self.assertEqual([e for e in events if e[0] in ('enter', 'exit')],
                                     [('enter', 1), ('exit', 1), ('enter', 2), ('exit', 2)])
                    self.assertIn(('manager', 'decode', 'fused_fn' if fused else 'decode_fn'), events)

    def test_prefill_token_bounds_before_prepare_or_capture(self):
        for tokens in (0, 321):
            fn, owner, state, events = self.fixture()
            owner.prefill_cudagraph_manager._capture_descs = {'PIECEWISE': [NS(num_tokens=tokens, num_reqs=None)]}
            with self.assertRaisesRegex(ValueError, 'capture rows must be in 1..320'):
                fn(owner)
            self.assertEqual(events, [])

    def test_malformed_decode_descriptors_before_first_capture(self):
        for tokens, reqs in ((8, 2), (8, None), (0, 0), (321, 321)):
            fn, owner, state, events = self.fixture()
            owner.decode_cudagraph_manager._capture_descs = {'FULL': [NS(num_tokens=tokens, num_reqs=reqs)]}
            with self.assertRaises(ValueError):
                fn(owner)
            self.assertEqual(events, [])

    def test_ns1_only_prefill(self):
        fn, owner, state, events = self.fixture(steps=1)
        owner.decode_cudagraph_manager = None
        fn(owner)
        self.assertEqual(state.contexts, 1)
        self.assertEqual([e[1] for e in events if e[0] == 'manager'], ['prefill'])

    def test_inactive_communicator_and_non_mtp_fail(self):
        for inactive in (True, False):
            fn, owner, state, events = self.fixture()
            if inactive:
                state.disabled = True
            else:
                owner.method = 'eagle'
            with self.assertRaisesRegex(RuntimeError, 'requires an active MTP communicator'):
                fn(owner)
            self.assertEqual(events, [])

    def test_knob_off_matches_original_capture_event_for_event(self):
        for steps in (1, 3, 4):
            for fused in (False, True):
                with self.subTest(steps=steps, fused=fused):
                    fn, owner, state, events = self.fixture(False, steps, fused)
                    original, original_owner, _, original_events = self.fixture(False, steps, fused, True)
                    fn(owner)
                    original(original_owner)
                    self.assertEqual(events, original_events)
                    self.assertEqual(state.prepared, set())
                    self.assertEqual(state.contexts, 0)


if __name__ == '__main__':
    assert MAX_ROWS == 320
    assert 'torch' not in sys.modules and 'vllm' not in sys.modules
    unittest.main(verbosity=2)
