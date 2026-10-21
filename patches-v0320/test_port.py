#!/usr/bin/env python3
"""CPU behavior checks and fresh pinned-source reconstruction; no GPU access."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TREE = None


def method(path, cls, name, namespace):
    tree = ast.parse((TREE / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class PortTests(unittest.TestCase):
    def test_pcie_admission_uses_v032_pass_config_and_rejects_other_shapes(self):
        torch = NS(bfloat16='bf16', version=NS(cuda='13.0'),
                   cuda=NS(get_device_capability=lambda d:(12, 0),
                           get_device_name=lambda d:'NVIDIA GeForce RTX 5090'))
        fn = method('vllm/distributed/device_communicators/pcie_ipc_all_reduce.py',
                    'PcieIpcAllReduce', '_disabled_reason',
                    dict(torch=torch, envs=NS(VLLM_SM12X_PCIE_IPC_MTP=True)))
        model = NS(dtype='bf16', get_hidden_size=lambda:5120, is_moe=False, enable_sleep_mode=False)
        config = NS(parallel_config=NS(enable_dbo=False), model_config=model, lora_config=None,
                    compilation_config=NS(pass_config=NS(fuse_allreduce_rms=False)),
                    speculative_config=NS(method='mtp', draft_model_config=model,
                                          draft_parallel_config=NS(tensor_parallel_size=2)))
        obj = NS(config=config)
        comm = NS(disabled=False, mnnvl_only=False)
        self.assertIsNone(fn(obj, True, comm, None))
        config.compilation_config.pass_config.fuse_allreduce_rms = True
        self.assertIn('unfused', fn(obj, True, comm, None))
        config.compilation_config.pass_config.fuse_allreduce_rms = False
        config.speculative_config.draft_parallel_config.tensor_parallel_size = 1
        self.assertIn('TP2 drafter', fn(obj, True, comm, None))
        self.assertIn('custom all-reduce disabled', fn(obj, False, comm, None))

    def test_splitkv_decorator_import_is_available_at_module_load(self):
        tree = ast.parse((TREE / 'vllm/v1/attention/backends/flashinfer.py').read_text())
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'functools']
        shim = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_enable_sm12x_nvfp4_prefill_split_kv')
        ns = {}
        exec(compile(ast.Module(body=imports+[shim], type_ignores=[]), 'splitkv-module-load', 'exec'), ns)
        self.assertTrue(callable(ns[shim.name].cache_info))

    def test_mtp_shared_vocab_is_not_loaded_on_pp1(self):
        fn = method('vllm/model_executor/models/qwen3_5_mtp.py', 'Qwen3_5MTP',
                    'is_unused_checkpoint_weight', {})
        for pp1 in (False, True):
            obj = NS(_share_target_vocab_weights=pp1)
            for name in ('language_model.embed_tokens.weight', 'lm_head.weight'):
                self.assertEqual(fn(obj, name), pp1)
            self.assertFalse(fn(obj, 'mtp.layers.0.self_attn.q_proj.weight'))
            self.assertTrue(fn(obj, 'language_model.layers.0.mlp.weight'))

    def test_zero_and_variable_drafts_keep_native_recurrent_routing(self):
        import torch
        tree = ast.parse((TREE / 'vllm/v1/attention/backends/gdn_attn.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GDNAttentionMetadataBuilder')
        build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
        branch = next(n for n in build.body if isinstance(n, ast.If)
                      and ast.unparse(n.test).startswith('not self.use_spec_decode'))
        code = compile(ast.Module(body=[branch], type_ignores=[]), 'native-gdn-routing', 'exec')
        for drafts, count in (([0, 0], 2), ([-1, 2, 3], 2), ([-1, -1], 0)):
            ns = dict(self=NS(use_spec_decode=True, num_spec=3),
                      num_decode_draft_tokens_cpu=torch.tensor(drafts),
                      query_start_loc=torch.tensor([0, 1, 2]),
                      async_tensor_h2d=lambda x, device: x)
            exec(code, ns)
            self.assertEqual(ns['num_spec_decodes'], count)
            if count:
                self.assertIsNotNone(ns['spec_sequence_masks'])

    def test_convolution_width_matches_state_layout_after_zero_draft_routing(self):
        import torch
        tree = ast.parse((TREE / 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py').read_text())
        values = [kw.value for call in ast.walk(tree)
                  if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                  and call.func.id == 'causal_conv1d_update'
                  for kw in call.keywords if kw.arg == 'max_query_len']
        self.assertEqual(len(values), 2)
        # Zero scheduled drafts do not prove the packed query/state layout shrank.
        # In the regressing port, spec_query_len=1 overrode this width of four.
        ns = dict(spec_state_indices_tensor=torch.zeros((2, 4)),
                  state_indices=torch.zeros((2, 4)), attn_metadata=NS(spec_query_len=1))
        for value in values:
            self.assertEqual(eval(compile(ast.Expression(value), 'conv-width', 'eval'), ns), 4)

    def test_autotune_reload_uses_each_rank_path_after_dirty_or_warm_boot(self):
        tree = ast.parse((TREE / 'vllm/model_executor/warmup/kernel_warmup.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'flashinfer_autotune')
        code = compile(ast.Module(body=fn.body[-4:], type_ignores=[]), 'autotune-tail', 'exec')
        for dirty in (False, True):
            for rank in (0, 1):
                calls = []
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / ('cache_rank' + str(rank))
                    path.touch()
                    tuner = NS(_dirty=dirty, save_configs=lambda p: calls.append(('save', p)),
                               load_configs=lambda p: calls.append(('load', p)))
                    exec(code, dict(world=NS(world_size=2, barrier=lambda: calls.append(('barrier',))),
                                    tuner=tuner, cache_path=path))
                    self.assertEqual(calls[-1], ('load', str(path)))
                    self.assertEqual(any(x[0] == 'save' for x in calls), dirty)

    def test_mtp_graph_capture_preserves_native_dynamic_specialization(self):
        import torch
        calls = []
        @contextmanager
        def capture():
            calls.append('enter')
            yield
            calls.append('exit')
        comm = NS(disabled=False, prepare_draft=lambda *a: calls.append('prepare'), capture=capture)
        ns = dict(torch=torch, nullcontext=nullcontext, envs=NS(VLLM_SM12X_PCIE_IPC_MTP=True),
                  get_tp_group=lambda: NS(device_communicator=NS(pcie_ipc_comm=comm)),
                  logger=NS(info=lambda *a: None))
        fn = method('vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py',
                    'TargetDependentARSpeculator', 'capture', ns)
        def manager(tokens, reqs):
            return NS(_capture_descs={0: [NS(num_tokens=tokens, num_reqs=reqs)]},
                      use_breakable_cg=False,
                      capture=lambda *a, **kw: calls.append(kw))
        obj = NS(method='mtp', num_speculative_steps=3, dtype=torch.bfloat16,
                 prefill_cudagraph_manager=manager(4, 1), decode_cudagraph_manager=manager(1, 1),
                 draft_model_config=NS(get_hidden_size=lambda:5120),
                 speculative_config=NS(uses_dynamic_speculative_decoding=lambda:True),
                 last_token_indices=torch.ones(1), idx_mapping=torch.ones(1), max_num_reqs=16,
                 supports_decode_metadata_update=True, _prefill=None, _generate_fused_drafts=None,
                 _generate_draft=None, model_state=None, target_input_buffers=None, block_tables=None,
                 target_attn_groups=None, kv_cache_config=None, input_buffers=None, attn_groups=None)
        for hook in ('on_prefill_begin', 'on_prefill_end', 'on_multi_step_decode_begin', 'on_multi_step_decode_end'):
            setattr(obj, hook, lambda *a: None)
        sys.path.insert(0, str(TREE))
        try:
            fn(obj)
            self.assertEqual(calls[0], 'prepare')
            self.assertEqual(calls.count('enter'), 2)
            self.assertTrue([c for c in calls if isinstance(c, dict)][-1]['specialize_spec_tokens'])
            obj.decode_cudagraph_manager = manager(2, 1)
            with self.assertRaises(ValueError): fn(obj)
            obj.decode_cudagraph_manager = manager(1, 1)
            obj.prefill_cudagraph_manager = manager(321, 1)
            with self.assertRaises(ValueError): fn(obj)
        finally:
            sys.path.pop(0)

    def test_ba4_old_engine_reports_are_rejected(self):
        sys.path.insert(0, str(TREE))
        try:
            from vllm_ba4.evidence import serving_reason, SERVING_PROTOCOL, BASE_IMAGE
            self.assertEqual(SERVING_PROTOCOL, 'V032_BA4-serving-profile-v1')
            self.assertEqual(BASE_IMAGE, 'sha256:' + 'a' * 64)
            self.assertEqual(serving_reason({'serving_evidence': {'protocol': 'L1a_BA4-serving-profile-v2'}}),
                             'missing_serving_PROFILE')
        finally:
            sys.path.pop(0)

    def test_plan_is_isolated_and_ba4_stays_disabled(self):
        spec = importlib.util.spec_from_file_location('candidate_plan', HERE / 'plan.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cmd = module.command(Path('/models/checkpoint'), Path('/candidate/v032-candidate'), True)
        self.assertIn('127.0.0.1:18029:8000', cmd)
        self.assertIn('VLLM_SM12X_BA_M4=0', cmd)
        self.assertNotIn('unless-stopped', cmd)
        self.assertNotIn('8020:8000', cmd)
        self.assertNotIn('rm', cmd)
        other = module.command(Path('/models/other-checkpoint'), Path('/candidate/v032-candidate'))
        def mounts(command):
            return {command[i+1] for i, value in enumerate(command) if value == '-v'
                    and not command[i+1].endswith(':/model:ro')}
        self.assertFalse(mounts(cmd) & mounts(other), 'different generations must never reuse tiers')
        with self.assertRaises(ValueError): module.command(Path('/model'), Path('/daily'))


def main():
    global TREE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-archive', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if hashlib.sha256(args.source_archive.read_bytes()).hexdigest() != manifest['upstream']['archive_sha256']:
        raise SystemExit('upstream source archive checksum mismatch')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with tarfile.open(args.source_archive) as tar:
            tar.extractall(root, filter='data')
        TREE = next(root.iterdir())
        spec = importlib.util.spec_from_file_location('candidate_apply', HERE / 'apply.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.install(TREE)
        with unittest.TestCase().assertRaises(ValueError): module.install(TREE)
        with unittest.TestCase().assertRaises(ValueError): module.install(TREE, ba4=True, control_image='old')
        module.install(TREE, ba4=True, control_image='sha256:' + 'a' * 64)
        outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortTests))
        if not outcome.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
