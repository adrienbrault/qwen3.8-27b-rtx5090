"""BA4 evidence through the real vLLM multiproc logging mechanism, CPU only.

Complete archived vLLM logger/formatter sources execute unmodified. Only the
distributed group lookups and GPU work are substituted. Each rank runs in a
fresh interpreter, with the two Python-root handler states observed in run 2.
"""
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
FIXTURE = HERE / 'fixtures/logging'
sys.path.insert(0, str(HERE / 'tests'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def worker(rank, profile, import_before_prefix):
    # Minimal import environment; the logger/formatters themselves are real.
    vllm = types.ModuleType('vllm')
    vllm.__path__ = []
    envs = types.ModuleType('vllm.envs')
    for name, value in dict(VLLM_LOGGING_PREFIX='', VLLM_LOGGING_LEVEL='INFO',
            VLLM_LOGGING_STREAM='ext://sys.stdout', VLLM_LOGGING_COLOR='0',
            NO_COLOR=True, FORCE_COLOR=False, VLLM_CONFIGURE_LOGGING=True,
            VLLM_LOGGING_CONFIG_PATH=None).items():
        setattr(envs, name, value)
    vllm.envs = envs
    sys.modules.update({'vllm': vllm, 'vllm.envs': envs})
    formatter = load('vllm.logging_utils', FIXTURE / 'logging_utils/formatter.py')
    vllm.logging_utils = formatter
    logger = load('vllm.logger', FIXTURE / 'logger.py')
    vllm.logger = logger
    # WorkerProc.worker_main configures logging before WorkerProc init.
    logger._configure_vllm_root_logger()
    handler = logging.getLogger('vllm').handlers[0]
    assert handler.stream is sys.stdout
    from test_cpu import fake_runtime, Tensor
    runtime = fake_runtime()[0] if import_before_prefix else None
    namespace = {'set_vllm_process_name': logger.set_vllm_process_name}
    decorate = function(FIXTURE / 'utils/system_utils.py', 'decorate_logs', namespace)
    group = lambda size, r: types.SimpleNamespace(world_size=size, rank_in_group=r)
    worker_ns = dict(model_parallel_is_initialized=lambda: True,
        get_dp_group=lambda: group(1, 0), get_pp_group=lambda: group(1, 0),
        get_pcp_group=lambda: group(1, 0), get_tp_group=lambda: group(2, rank),
        get_dcp_group=lambda: group(1, 0), set_process_title=lambda **kw: None,
        decorate_logs=decorate)
    setup = function(FIXTURE / 'v1/executor/multiproc_executor.py',
        'setup_proc_title_and_log_prefix', worker_ns, 'WorkerProc')
    setup(False)
    if runtime is None:
        runtime = fake_runtime()[0]
    # Prefix setup does not replace/wrap either stream or the handler.
    assert handler is logging.getLogger('vllm').handlers[0]
    assert handler.stream is sys.stdout and not hasattr(sys.stdout, '_original_write')
    if rank == 0:
        logging.basicConfig()  # BASIC_FORMAT, default stderr StreamHandler.
    else:
        assert not logging.root.handlers  # default dynamic-stderr lastResort.
    logging.getLogger('vllm.worker.control').warning('ordinary worker log')
    # Replay each actual run-2 payload, changing only rank's OS PID to this
    # test worker's real PID. The original captured file is never repaired.
    selected = []
    for raw in (FIXTURE / 'run2-ba4.log').read_text().splitlines():
        payload = raw[raw.index('SM12X BA4 '):]
        if f' rank={rank} ' not in payload:
            continue
        payload = re.sub(r'pid=\d+', f'pid={os.getpid()}', payload)
        if not profile:
            payload = payload.replace('PROFILE_ONLY:', 'ACTIVE:')
        runtime.logger.warning('%s', payload)
        selected.append(re.search(r'prefix=(\S+)', payload)[1])
    assert len(selected) == 48
    # Exercise all production emission sites, including postload and INACTIVE.
    layer = types.SimpleNamespace(prefix=selected[0], weight=Tensor((48, 5120)),
        bias=None, tp_size=2, tp_rank=rank, output_sizes=[48, 48],
        quant_method=types.SimpleNamespace(_gemm_impl=types.SimpleNamespace(
            __name__='default_unquantized_gemm')))
    entry = dict(prefix=layer.prefix, weight_sha256='fixture', x_stride=[5120, 1])
    report = dict(selected='triton_direct', layers=[entry], fingerprint={},
                  serving_evidence=dict(binding={}))
    def report_for(path):
        return report if path == 'report' else dict(binding={}, fingerprints=[{}, {}])
    with mock.patch.dict(os.environ, dict(VLLM_BATCH_INVARIANT='0',
            VLLM_SM12X_BA_M4_PROFILE_ONLY=str(int(profile)),
            VLLM_SM12X_BA_M4_RESULT='report', VLLM_SM12X_BA_M4_BINDING='binding')), \
         mock.patch.object(runtime, '_read_report', side_effect=report_for), \
         mock.patch.object(runtime, 'admission_reason', return_value=None), \
         mock.patch.object(runtime, 'fingerprint', return_value={}), \
         mock.patch.object(runtime, 'tensor_sha256', return_value='fixture'):
        runtime.prepare_layer(layer)
    assert layer._sm12x_ba4 is not None
    # Actual opaque-op body and its once-per-prefix suppression; no GPU I/O.
    for prefix in selected:
        operand = ('PROFILE_ONLY:' if profile else '') + prefix
        for _ in range(2):
            runtime.ba4_out(Tensor((4, 5120)), layer.weight, Tensor((4, 48)),
                Tensor((40, 4, 48)), 'triton_direct', operand, [5120, 1])
    with mock.patch.dict(os.environ, {'VLLM_BATCH_INVARIANT': '1'}):
        runtime.prepare_layer(layer)
    print('J41O_WORKER ' + json.dumps(dict(rank=rank, pid=os.getpid(),
        logger=runtime.logger.name, root_handler='basicConfig' if rank == 0 else 'lastResort',
        handler_bound_before_prefix=True, runtime_import_before_prefix=import_before_prefix,
        profile=profile)), flush=True)


class WorkerLoggingTests(unittest.TestCase):
    def test_real_sources_and_captured_payloads_are_pinned(self):
        record = json.loads((FIXTURE / 'source.json').read_text())
        for name, expected in record['files'].items():
            self.assertEqual(hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest(), expected)
        lines = (FIXTURE / 'run2-ba4.log').read_text().splitlines()
        self.assertEqual(len(lines), 96)
        self.assertEqual(sum('WARNING:vllm_ba4.runtime:' in l for l in lines), 48)
        self.assertTrue(all('(Worker_TP' not in l for l in lines))

    def test_real_multiproc_emission_for_both_root_states_and_import_orders(self):
        for profile in (True, False):
            for before in (True, False):
                for rank in (0, 1):
                    with self.subTest(profile=profile, import_before=before, rank=rank):
                        p = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                            '--worker', str(rank), str(int(profile)), str(int(before))],
                            capture_output=True, text=True, timeout=30)
                        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                        self.assertEqual(p.stderr, '')
                        rows = [s for s in p.stdout.splitlines() if 'SM12X BA4 ' in s]
                        self.assertEqual(len(rows), 98)  # 48 replay + postload + 48 used + inactive.
                        pid = json.loads(p.stdout.split('J41O_WORKER ')[1])['pid']
                        for row in rows:
                            self.assertEqual(row.count(f'(Worker_TP{rank} pid={pid})'), 1)
                            self.assertIn(f'pid={pid} ', row)
                        self.assertIn('ordinary worker log', p.stdout)
                        self.assertEqual(sum('PROFILE_USED:' in s if profile else ' USED:' in s
                                             for s in rows), 48)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), bool(int(sys.argv[3])), bool(int(sys.argv[4])))
    else:
        unittest.main()
