#!/usr/bin/env python3
"""Usage: python3 deliver/test-0142.py SCRATCH_ROOT (stdlib only; no torch/vllm imports)."""
import ast
import contextlib
import functools
import os
from pathlib import Path
import sys
import threading
import unittest
import warnings
from collections import Counter
from types import SimpleNamespace as NS

root = Path(sys.argv.pop(1))

def extract(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

class Device:
    def __init__(self, name):
        self.type = name.split(':')[0]

class Tensor:
    def __init__(self, device='cuda'):
        self.device = Device(device)
        self.is_cuda = device == 'cuda'
    def item(self):
        if fake.mode and self.is_cuda:
            warnings.warn('called a synchronizing operation', UserWarning)
        return 7
    tolist = item
    nonzero = item
    def copy_(self, src, non_blocking=False):
        return self
    def to(self, *args, **kwargs):
        return self
    cpu = to
    cuda = to

class Event:
    def synchronize(self):
        if fake.mode:
            warnings.warn('called a synchronizing operation', UserWarning)

class Graph:
    def replay(self):
        pass

class Log:
    def __init__(self):
        self.tables = []
    def info_once(self, *args):
        pass
    def info(self, message):
        self.tables.append(message)

log = Log()
fake = NS(Tensor=Tensor, device=Device, mode=0)
def set_mode(mode):
    fake.mode = mode
fake.cuda = NS(get_sync_debug_mode=lambda: fake.mode, set_sync_debug_mode=set_mode,
               synchronize=lambda: None, Event=Event, Stream=Event, CUDAGraph=Graph)
fake.accelerator = NS(synchronize=lambda: None)
fake.nonzero = lambda input: input.nonzero()
fake.where = lambda input: input.nonzero()
scope = dict(contextlib=contextlib, functools=functools, sys=sys, threading=threading,
             warnings=warnings, Counter=Counter, torch=fake, logger=log,
             _LOCK=threading.Lock(), __file__='census-test')
extract(root/'vllm/v1/worker/gpu/sync_census.py',
        {'_site', '_transfer', '_Hooks', 'Census', 'install'}, scope)
scope['_HOOKS'] = scope['_Hooks']()
Census = scope['Census']

class Runner:
    execute_model_state = None
    def execute_model(self, prefill=False, **kwargs):
        self.execute_model_state = NS(input_batch=NS(has_prefill=prefill))
    def sample_tokens(self):
        Tensor().item()
        Graph().replay()
        return NS(get_output=lambda: Event().synchronize())

class Tests(unittest.TestCase):
    def setUp(self):
        log.tables.clear()
        fake.mode = 0
    def test_grammar(self):
        for tag, name in [('SYNC_CENSUS', '_sm12x_sync_census'),
                          ('DFLASH_DEVICE_ACCEPT', '_sm12x_dflash_device_accept')]:
            path = root/'vllm/envs.py'
            env_scope = {'os': NS(environ={})}
            extract(path, {name}, env_scope)
            if name not in env_scope:  # standalone 0142 tree only
                continue
            parse = env_scope[name]
            self.assertEqual(parse(), 0)
            for value in ('', 'true', '-1', '2', '01', ' 1', '١'):
                env_scope['os'].environ['VLLM_SM12X_'+tag] = value
                with self.assertRaises(ValueError):
                    parse()
            env_scope['os'].environ['VLLM_SM12X_'+tag] = '1'
            self.assertEqual(parse(), 1)
            self.assertEqual(parse('steps', '50', 1), 50)
            with self.assertRaises(ValueError):
                parse('steps', '0', 1)
    def test_transfer_overloads(self):
        transfer = scope['_transfer']
        cpu, gpu = Tensor('cpu'), Tensor()
        self.assertTrue(transfer('copy_', (cpu, gpu), {}))
        self.assertFalse(transfer('copy_', (cpu, gpu, True), {}))
        self.assertFalse(transfer('copy_', (gpu, gpu), {}))
        self.assertTrue(transfer('to', (gpu, 'cpu'), {}))
        self.assertFalse(transfer('to', (gpu, 'cpu', None, True), {}))
        self.assertFalse(transfer('to', (gpu, cpu, True), {}))
        self.assertFalse(transfer('cuda', (cpu, None, True), {}))
        self.assertFalse(transfer('to', (gpu,), {'dtype': object()}))
    def test_debug_dedup_and_restore_on_error(self):
        c = Census(Runner(), 1, 0)
        row = Counter()
        original, hook = Tensor.item, warnings.showwarning
        fake.mode = 2
        with self.assertRaisesRegex(ValueError, 'test'):
            with c.scope(row, 'target'):
                Tensor().item()
                Tensor('cpu').item()
                Event().synchronize()
                Graph().replay()
                warnings.warn('called a synchronizing operation', UserWarning)
                raise ValueError('test')
        self.assertEqual(sum(row.values()), 4)
        self.assertIs(Tensor.item, original)
        self.assertIs(warnings.showwarning, hook)
        self.assertEqual(fake.mode, 2)
        self.assertFalse(scope['_LOCK'].locked())
    def test_fallback(self):
        old = fake.cuda.set_sync_debug_mode
        fake.cuda.set_sync_debug_mode = lambda mode: (_ for _ in ()).throw(NotImplementedError())
        # Missing getter selects fallback without attempting restoration.
        getter = fake.cuda.get_sync_debug_mode
        fake.cuda.get_sync_debug_mode = lambda: (_ for _ in ()).throw(AttributeError())
        try:
            c = Census(Runner(), 1, 0)
            row = Counter()
            with c.scope(row, 'target'):
                fake.nonzero(Tensor())
                Tensor('cpu').copy_(Tensor())
                Tensor('cpu').copy_(Tensor(), True)
            self.assertEqual(sum(row.values()), 2)
        finally:
            fake.cuda.set_sync_debug_mode = old
            fake.cuda.get_sync_debug_mode = getter
    def test_debug_setter_unsupported(self):
        old = fake.cuda.set_sync_debug_mode
        fake.cuda.set_sync_debug_mode = lambda mode: (_ for _ in ()).throw(NotImplementedError())
        try:
            c = Census(Runner(), 1, 0)
            row = Counter()
            with c.scope(row, 'target'):
                Tensor().item()
            self.assertEqual(sum(row.values()), 1)
            self.assertFalse(scope['_LOCK'].locked())
        finally:
            fake.cuda.set_sync_debug_mode = old

    def test_window_and_late_output(self):
        runner = Runner()
        original = runner.execute_model
        c = Census(runner, 2, 1)
        runner.execute_model(prefill=True)
        runner.sample_tokens().get_output()
        runner.execute_model()
        runner.sample_tokens().get_output()  # warmup
        outputs = []
        for _ in range(2):
            runner.execute_model()
            outputs.append(runner.sample_tokens())
        self.assertEqual(runner.execute_model, original)
        self.assertFalse(log.tables)
        for out in reversed(outputs):
            out.get_output()
            out.get_output()  # consumption hook is one-shot
        self.assertEqual(len(log.tables), 1)
        self.assertEqual(c.finished, 2)
        self.assertTrue(all(sum(row.values()) == 3 for row in c.rows.values()))
    def test_async_overlap_attribution_and_last_output_report(self):
        barrier = threading.Barrier(2, timeout=5)
        release = threading.Event()
        errors, outputs = [], []
        original, hook = Tensor.item, warnings.showwarning
        class AsyncRunner(Runner):
            step = 0
            def execute_model(self):
                self.step += 1
                super().execute_model()
                if self.step == 2:
                    barrier.wait()  # Output 1 already owns a scope.
                    Tensor().item()
                    warnings.warn('target synchronizing operation', UserWarning)
                    barrier.wait()
            def sample_tokens(self):
                result = super().sample_tokens()
                if self.step == 1:
                    def output():
                        barrier.wait()
                        Event().synchronize()
                        warnings.warn('output synchronizing operation', UserWarning)
                        barrier.wait()
                        barrier.wait()  # Still active during sample 2.
                        barrier.wait()
                        if not release.wait(5):
                            raise TimeoutError('main thread did not release output')
                    result.get_output = output
                else:
                    barrier.wait()
                    barrier.wait()
                return result
        runner = AsyncRunner()
        execute, sample = runner.execute_model, runner.sample_tokens
        c = Census(runner, 2, 0)
        runner.execute_model()
        first = runner.sample_tokens()
        def drain():
            try:
                first.get_output()
                outputs[0].get_output()  # Final report must run on this thread.
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=drain)
        worker.start()
        try:
            runner.execute_model()
            outputs.append(runner.sample_tokens())
            self.assertIsNot(Tensor.item, original)
            self.assertEqual(fake.mode, 'warn')
            self.assertEqual(len(scope['_HOOKS'].active), 1)
            self.assertEqual(runner.execute_model, execute)
            self.assertEqual(runner.sample_tokens, sample)
            self.assertFalse(log.tables)
        finally:
            release.set()
            worker.join(6)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(c.finished, 2)
        self.assertEqual(len(log.tables), 1)
        c.report()
        self.assertEqual(len(log.tables), 1)
        def counts(step):
            return Counter({(phase, op): n for (phase, op, site), n in c.rows[step].items()})
        self.assertEqual(counts(1), Counter({
            ('sample+draft', 'api:item'): 1, ('sample+draft', 'api:replay'): 1,
            ('output', 'api:synchronize'): 1,
            ('output', 'debug:output synchronizing operation'): 1}))
        self.assertEqual(counts(2), Counter({
            ('target', 'api:item'): 1,
            ('target', 'debug:target synchronizing operation'): 1,
            ('sample+draft', 'api:item'): 1, ('sample+draft', 'api:replay'): 1,
            ('output', 'api:synchronize'): 1}))
        self.assertIs(Tensor.item, original)
        self.assertIs(warnings.showwarning, hook)
        self.assertEqual(fake.mode, 0)
        self.assertFalse(scope['_HOOKS'].active)

    def test_unsupported_config_and_missing_output(self):
        for method, pp in [('other', 1), ('dflash', 2)]:
            runner = Runner()
            runner.vllm_config = NS(speculative_config=NS(method=method),
                                    parallel_config=NS(pipeline_parallel_size=pp))
            with self.assertRaisesRegex(ValueError, 'DFlash and PP=1'):
                scope['install'](runner, 1, 0)
        runner = Runner()
        runner.sample_tokens = lambda: None
        Census(runner, 1, 0)
        runner.execute_model()
        with self.assertRaisesRegex(RuntimeError, 'AsyncOutput'):
            runner.sample_tokens()

    def test_nested_scope_rejected(self):
        c = Census(Runner(), 1, 0)
        with c.scope(Counter(), 'target'):
            with self.assertRaises(RuntimeError):
                with c.scope(Counter(), 'output'):
                    pass

unittest.main()
