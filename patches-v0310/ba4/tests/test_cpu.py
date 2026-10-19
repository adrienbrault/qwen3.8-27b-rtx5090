"""Dependency-free CPU verification: real patch and real route, fake GPU I/O."""

import ast
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from vllm_ba4 import policy


def patched_tree():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    shutil.copytree(HERE / "fixtures/base", root, dirs_exist_ok=True)
    with (HERE / "L1a_BA4.diff").open("rb") as f:
        subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch",
                        "-p1", "-d", str(root)], stdin=f, check=True,
                       capture_output=True)
    return temp, root


def method(source, name):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "UnquantizedLinearMethod")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def exec_method(source, name, ns):
    node = method(source, name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), ns)
    return ns[name]


class Device:
    def __init__(self, kind="cuda", index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    def __init__(self, shape, stride=None, dtype="torch.bfloat16", device=None, offset=0):
        self.shape = tuple(shape)
        self._stride = tuple(stride or (shape[-1], 1))
        self.dtype = dtype
        self.device = device or Device()
        self._offset = offset
        self.requires_grad = False

    @property
    def data(self):
        return self

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._offset

    def zero_(self):
        return self


def fake_runtime():
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.nn = types.SimpleNamespace(Module=object)
    torch.bfloat16, torch.float32 = "torch.bfloat16", "torch.float32"
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda d: (12, 0))
    torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
    torch._dynamo = types.SimpleNamespace(assume_constant_result=lambda fn: fn, allow_in_graph=lambda fn: fn)
    torch.empty = mock.Mock(side_effect=lambda shape, **kw: Tensor(shape, **kw))
    torch.empty_strided = mock.Mock(side_effect=lambda shape, stride, **kw: Tensor(shape, stride, **kw))

    def decorate(*args, **kwargs):
        def wrap(f):
            f.register_fake = lambda fake: fake
            return f
        return wrap

    torch.library = types.SimpleNamespace(custom_op=decorate)
    kernels = types.ModuleType("vllm_ba4.kernels")
    kernels.run_out = mock.Mock()
    with mock.patch.dict(sys.modules, {"torch": torch, "vllm_ba4.kernels": kernels}):
        # Avoid caching a fake implementation into a later real-GPU test run.
        old = sys.modules.pop("vllm_ba4.runtime", None)
        try:
            runtime = importlib.import_module("vllm_ba4.runtime")
        finally:
            sys.modules.pop("vllm_ba4.runtime", None)
            if old is not None:
                sys.modules["vllm_ba4.runtime"] = old
    return runtime, torch, kernels


def valid_report():
    # Entirely synthetic unit fixture. Never emitted as measured GPU evidence.
    from vllm_ba4.common import benchmark_sha256
    from vllm_ba4.evidence import BASE_IMAGE, SERVING_PROTOCOL, SHARD_ORDER, digest, diagnostic_digest
    from vllm_ba4.measurement import TIMING
    def row(name, cost):
        result = {"name": name, "us_per_step": cost, "timing": TIMING,
                  "correct": True, "graph_ok": True, "allocation_free": True,
                  "per_layer_us": [6.25] * 48, "b1_trace_us_per_step": 300,
                  "kernel_counts": [1] * 48, "cuda_symbols": ["unit_fixture_kernel"],
                  "trace_path": "fixture.trace", "trace_sha256": "a" * 64,
                  "us_per_call": 6.25, "effective_weight_GBps": 48*48*5120*2/300/1000,
                  "all_kernel_us": 1000, "profiled_replay_us": 1100,
                  "max_abs": 0, "max_rel": 0,
                  "max_abs_vs_fp64_bf16": 0, "max_rel_vs_fp64_bf16": 0}
        if name == "current":
            result.update(replay_samples_us=[cost]*3, retained_graph_outputs_correct=True,
                          real_M_capture=True)
        else:
            result.update(paired_trials=[{"control_before_us": 4000,
                                         "candidate_us": cost, "control_after_us": 4000}
                                        for _ in range(3)],
                          trial_us_per_step=[cost]*3, paired_savings_us=[4000-cost]*3,
                          compiled_selected=True, transient_device_bytes=0, packing_us=0,
                          synchronization_inside_timing=False, scratch_bytes=40*4*48*4*48,
                          scratch_used=False)
        return result
    report = {"protocol": policy.PROTOCOL, "verdict": "WIN", "fingerprint": {"gpu": 1},
            "code_sha256": "code", "benchmark_sha256": benchmark_sha256(),
            "rank": 0, "real_capture": True, "probe_mode": "eager_diagnostic",
            "capture_path": "fixture.pt", "capture_sha256": "b" * 64, "timing": TIMING,
            "timing_validation": {"single_work_us": [10]*3, "sixteen_work_us": [100]*3},
            "replay": {"distinct_B1_matrices": 48, "B1_weight_bytes": 48*48*5120*2,
                       "before_bytes_per_layer": 41_943_040, "after_bytes_per_layer": 90_931_200,
                       "stream_bytes_per_step": 6_377_963_520, "timing": TIMING},
            "baseline_impl": "default_unquantized_gemm", "selected": "torch_mm",
            "wmma128x1_seen": True, "cublaslt_requires_per_call_pin": False,
            "compiled_check": {k: True for k in policy.COMPILED_CHECKS},
            "layers": [{"prefix": "model.layers.%d.linear_attn.in_proj_ba" % i,
                        "x_stride": [5120, 1], "weight_stride": [5120, 1],
                        "shard_order": SHARD_ORDER, "capture_x_offsets": {"4": 0},
                        "weight_sha256": "a" * 64} for i in range(48)],
            "measurements": {m: [row("current", 4000), row("torch_mm", 3000)] +
                             [{"name": n, "error": "unavailable unit fixture"}
                              for n in (*policy.CANDIDATES, "cublaslt_mm_probe") if n != "torch_mm"]
                             for m in ("4", "8", "12", "16", "32")}}
    binding = {"base_image_id": BASE_IMAGE, "candidate_image_id": "sha256:" + "c"*64,
               "code_sha256": "code", "benchmark_sha256": benchmark_sha256(),
               "compiler_config_sha256": "d"*64, "launch_config_sha256": "e"*64,
               "source_archive_sha256": "f"*64}
    trials = []
    for trial in range(3):
        phases = {}
        for phase in ("control_before", "candidate", "control_after"):
            candidate = phase == "candidate"
            layers = [dict(e, x_shape=[4,5120], logical_M=4, padded_M=4, x_offset=0,
                           weight_shape=[48,5120], tp_rank=0, tp_size=2, output_sizes=[48,48],
                           calls_per_step=1, route="torch_mm" if candidate else "served_compiled_baseline",
                           kernels=[{"symbol": "unit_fixture_kernel", "count_per_step": 1,
                                     "us_per_step": 5 if candidate else 20}])
                      for e in report["layers"]]
            phases[phase] = {"arm": "L1a_BA4" if candidate else "A_S_L1a", "mode": "PROFILE",
                             "concurrency": 1, "image_id": binding["candidate_image_id" if candidate else "base_image_id"],
                             "binding_sha256": digest(binding), "vllm_compile_capture": True,
                             "outputs_correct": True, "changed_input_correct": True,
                             "graph_replays_per_step": 4, "pcie_pushes_per_step": 138,
                             "mtp_bf16_calls_per_step": 15, "layers": layers,
                             "b1_us_per_step": 240 if candidate else 960,
                             **{kind: {"path": "%s-%s-%s" % (trial, phase, kind),
                                       "sha256": digest([trial, phase, kind])}
                                for kind in ("trace", "probe", "harness")}}
        trials.append(phases)
    report["serving_evidence"] = {"protocol": SERVING_PROTOCOL, "rank": 0,
                                  "fingerprint": report["fingerprint"], "binding": binding,
                                  "diagnostic_sha256": diagnostic_digest(report), "trials": trials}
    for m, rows in report['measurements'].items():
        rows[1]['scratch_bytes'] = 40*int(m)*48*4*48
    report['serving_evidence']['diagnostic_sha256'] = diagnostic_digest(report)
    return report


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp, cls.root = patched_tree()
        cls.base = (HERE / "fixtures/base/vllm/model_executor/layers/linear.py").read_text()
        cls.patched = (cls.root / "vllm/model_executor/layers/linear.py").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pinned_inputs_and_all_outputs_compile(self):
        manifest = json.loads((HERE / "manifest.json").read_text())
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(sha(HERE / "L1a_BA4.diff"), manifest["patch_sha256"])
        for rel, expected in manifest["before"].items():
            self.assertEqual(sha(HERE / "fixtures/base" / rel), expected)
        for rel, expected in manifest["unchanged"].items():
            self.assertEqual(sha(HERE / "fixtures/base" / rel), expected)
            self.assertEqual(sha(self.root / rel), expected)
        for rel, expected in manifest["after"].items():
            self.assertEqual(sha(self.root / rel), expected)
            compile((self.root / rel).read_text(), rel, "exec")

    def test_every_original_node_survives_except_added_route_blocks(self):
        before, after = ast.parse(self.base), ast.parse(self.patched)
        cls = next(n for n in after.body if isinstance(n, ast.ClassDef)
                   and n.name == "UnquantizedLinearMethod")
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in ("__init__", "apply", "process_weights_after_loading"):
                fn.body = [n for n in fn.body if not (
                    isinstance(n, ast.If) and ast.unparse(n.test) == "envs.VLLM_SM12X_BA_M4 and current_platform.is_cuda()")]
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_real_installer_and_base_drift_rejection(self):
        spec = importlib.util.spec_from_file_location("ba4_install", HERE / "install.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(HERE / "fixtures/base", root, dirs_exist_ok=True)
            installer.install(root)
            with self.assertRaises(AssertionError):
                installer.install(root)  # Cannot silently patch an already patched base.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(HERE / "fixtures/base", root, dirs_exist_ok=True)
            (root / "vllm/envs.py").write_text("drift")
            with self.assertRaises(AssertionError):
                installer.install(root)

    def test_flags_registered_and_exact_one_only(self):
        src = (self.root / "vllm/envs.py").read_text()
        tree = ast.parse(src)
        env = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == "environment_variables")
        ns = {"os": os}
        getters = {}
        for k, v in zip(env.keys, env.values):
            if isinstance(k, ast.Constant) and k.value in ("VLLM_SM12X_BA_M4", "VLLM_SM12X_BA_M4_RESULT"):
                getters[k.value] = eval(compile(ast.Expression(v), "flag", "eval"), ns)
        self.assertEqual(len(getters), 2)
        for value in (None, "0", "true", "yes", " 1", "2", "1"):
            with mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["VLLM_SM12X_BA_M4"] = value
                self.assertEqual(getters["VLLM_SM12X_BA_M4"](), value == "1")

    def run_apply(self, source, flag, batch, routed=None, cuda=True):
        runtime = types.ModuleType("vllm_ba4.runtime")
        runtime.try_apply = mock.Mock(return_value=routed)
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object), Tensor=Tensor)
        batch_result, original = object(), object()
        dispatch = mock.Mock(return_value=original)
        fn = exec_method(source, "apply", {"torch": torch,
            "envs": types.SimpleNamespace(VLLM_BATCH_INVARIANT=batch, VLLM_SM12X_BA_M4=flag),
            "current_platform": types.SimpleNamespace(is_cuda=lambda: cuda, is_cuda_alike=lambda: cuda, is_xpu=lambda: False),
            "linear_batch_invariant": mock.Mock(return_value=batch_result)})
        layer, x, bias = types.SimpleNamespace(weight=object()), object(), object()
        with mock.patch.dict(sys.modules, {"vllm_ba4.runtime": runtime}):
            value = fn(types.SimpleNamespace(_gemm_impl=dispatch), layer, x, bias)
        return value, original, batch_result, dispatch, runtime.try_apply, (layer, x, layer.weight, bias)

    def test_flag_off_identity_and_exact_dispatch_arguments(self):
        for batch in (False, True):
            for src in (self.base, self.patched):
                value, original, batch_result, dispatch, route, expected = self.run_apply(src, False, batch)
                self.assertIs(value, batch_result if batch else original)
                route.assert_not_called()
                if not batch:
                    dispatch.assert_called_once_with(*expected)

    def test_batch_invariant_precedes_new_route(self):
        value, original, batch_result, dispatch, route, _ = self.run_apply(self.patched, True, True, object())
        self.assertIs(value, batch_result)
        route.assert_not_called()
        dispatch.assert_not_called()

    def test_route_success_and_fallback(self):
        for result in (None, object()):
            value, original, _, dispatch, route, expected = self.run_apply(self.patched, True, False, result)
            self.assertIs(value, original if result is None else result)
            route.assert_called_once_with(expected[0], expected[1], expected[3])
            self.assertEqual(dispatch.call_count, int(result is None))

    def test_flag_on_noncuda_keeps_original_without_helper_import(self):
        value, original, _, dispatch, route, expected = self.run_apply(self.patched, True, False, object(), cuda=False)
        self.assertIs(value, original)
        route.assert_not_called()
        dispatch.assert_called_once_with(*expected)

    def test_postload_flag_off_imports_nothing(self):
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object), Tensor=Tensor)
        fn = exec_method(self.patched, "process_weights_after_loading", {
            "torch": torch, "envs": types.SimpleNamespace(VLLM_SM12X_BA_M4=False),
            "current_platform": types.SimpleNamespace(is_cpu=lambda: False, is_xpu=lambda: False)})
        with mock.patch.dict(sys.modules, {"vllm_ba4.runtime": None}):
            fn(object(), object())  # An accidental import would raise.

    def test_postload_flag_on_noncuda_imports_nothing(self):
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object), Tensor=Tensor)
        fn = exec_method(self.patched, "process_weights_after_loading", {
            "torch": torch, "envs": types.SimpleNamespace(VLLM_SM12X_BA_M4=True),
            "current_platform": types.SimpleNamespace(is_cpu=lambda: False,
                is_xpu=lambda: False, is_cuda=lambda: False)})
        with mock.patch.dict(sys.modules, {"vllm_ba4.runtime": None}):
            fn(object(), object())


class AdmissionTests(unittest.TestCase):
    def test_valid_and_boundary(self):
        self.assertIsNone(policy.admission_reason(valid_report(), {"gpu": 1}, "code", 0))
        for savings, expected in (([500, 500, 500], True), ([500, 499.999, 500], False),
                                   ([1000, 1000], False), ([math.nan] * 3, False),
                                   ([math.inf] * 3, False)):
            row = {"us_per_step": 1000, "trial_us_per_step": [1000]*len(savings),
                   "paired_savings_us": savings,
                   "paired_trials": [{"control_before_us": 1000+s,
                                      "candidate_us": 1000, "control_after_us": 1000+s}
                                     for s in savings]}
            self.assertEqual(policy.winning(row), expected)

    def test_fastest_correct_complete_path(self):
        fast = valid_report()["measurements"]["4"][1]
        slower = dict(fast, name="triton_split128", us_per_step=4000)
        for bad in (dict(fast, name="torch_rows_mv", us_per_step=10, correct=False),
                    dict(fast, name="triton_direct", us_per_step=10, graph_ok=False),
                    dict(fast, name="torch_rows_mm", us_per_step=10, allocation_free=False)):
            self.assertEqual(policy.select([bad, slower, fast])["name"], "torch_mm")

    def test_stale_incomplete_or_unproven_report_rejected(self):
        mutations = [("protocol", "old"), ("verdict", "RUNNING"), ("fingerprint", {}),
                     ("code_sha256", "old"), ("rank", 1), ("real_capture", False),
                     ("baseline_impl", "other"), ("layers", []), ("selected", "triton_direct"),
                     ("wmma128x1_seen", False), ("compiled_check", {}),
                     ("cublaslt_requires_per_call_pin", True)]
        for key, value in mutations:
            report = valid_report()
            report[key] = value
            self.assertIsNotNone(policy.admission_reason(report, {"gpu": 1}, "code", 0), key)
        report = valid_report()
        report["layers"][1] = report["layers"][0]
        self.assertIsNotNone(policy.admission_reason(report, {"gpu": 1}, "code", 0))
        for report in (None, [], {}, {"protocol": policy.PROTOCOL, "verdict": "WIN", "layers": [None]}):
            self.assertIsNotNone(policy.admission_reason(report, {}, "code", 0))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.torch, self.kernels = fake_runtime()
        self.w = Tensor((48, 5120))
        self.out, self.partial = Tensor((4, 48)), Tensor((40, 4, 48))
        self.prefix = "model.layers.0.linear_attn.in_proj_ba"
        self.layer = types.SimpleNamespace(prefix=self.prefix, weight=self.w, bias=None,
            tp_size=2, tp_rank=0, output_sizes=[48, 48],
            quant_method=types.SimpleNamespace(_gemm_impl=types.SimpleNamespace(__name__="default_unquantized_gemm")))
        self.state = self.runtime.State("torch_mm", (5120, 1), self.w, self.out, self.partial)
        self.layer._sm12x_ba4 = self.state

    def test_matching_route_returns_same_preallocated_buffer(self):
        for _ in range(3):
            self.assertIs(self.runtime.try_apply(self.layer, Tensor((4, 5120)), None), self.out)
        self.assertEqual(self.kernels.run_out.call_count, 3)
        self.torch.empty.assert_not_called()
        self.torch.empty_strided.assert_not_called()

    def test_all_input_fallbacks_do_no_work(self):
        cases = [(Tensor((m, 5120)), None) for m in (1, 3, 8, 12, 16, 32)] + [
            (Tensor((4, 5119)), None), (Tensor((4, 5120), (5136, 1)), None),
            (Tensor((4, 5120), (1, 4)), None),
            (Tensor((4, 5120), dtype="torch.float16"), None),
            (Tensor((4, 5120), device=Device("cpu")), None),
            (Tensor((4, 5120), device=Device("cuda", 1)), None),
            (Tensor((4, 5120), offset=1), None), (Tensor((4, 5120)), object())]
        grad = Tensor((4, 5120))
        grad.requires_grad = True
        cases.append((grad, None))
        for x, bias in cases:
            self.assertIsNone(self.runtime.try_apply(self.layer, x, bias))
        self.kernels.run_out.assert_not_called()

    def test_weight_prefix_and_unprepared_fallbacks(self):
        for field, value in (("weight", Tensor((48, 5120))), ("prefix", "mtp.in_proj_ba"),
                             ("_sm12x_ba4", None)):
            with mock.patch.object(self.layer, field, value):
                self.assertIsNone(self.runtime.try_apply(self.layer, Tensor((4, 5120)), None))
        for field, value in (("shape", (48, 5119)), ("_stride", (1, 48)),
                             ("_offset", 8), ("dtype", "torch.float32"),
                             ("requires_grad", True),
                             ("device", Device("cpu"))):
            with mock.patch.object(self.w, field, value):
                self.assertIsNone(self.runtime.try_apply(self.layer, Tensor((4, 5120)), None))
        self.kernels.run_out.assert_not_called()

    def test_validated_padded_input_stride(self):
        self.state.x_stride = (5136, 1)
        self.assertIs(self.runtime.try_apply(self.layer, Tensor((4, 5120), (5136, 1), offset=8), None), self.out)

    def prepare(self, report):
        with mock.patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0", "VLLM_SM12X_BA_M4_RESULT": "rank{rank}.json",
                                         "VLLM_SM12X_BA_M4_BINDING": "binding.json"}), \
             mock.patch.object(self.runtime, "_read_report", side_effect=lambda path:
                 {"binding": report["serving_evidence"]["binding"],
                  "fingerprints": [report["fingerprint"], report["fingerprint"]]}
                 if path == "binding.json" else report), \
             mock.patch.object(self.runtime, "fingerprint", return_value={"gpu": 1}), \
             mock.patch.object(self.runtime, "code_sha256", return_value="code"), \
             mock.patch.object(self.runtime, "tensor_sha256", return_value="a" * 64):
            self.runtime.prepare_layer(self.layer)

    def test_boot_preparation_only_after_matching_win(self):
        self.prepare(valid_report())
        self.assertIsNotNone(self.layer._sm12x_ba4)
        self.assertEqual(self.torch.empty.call_count, 2)
        self.assertEqual(self.torch.empty_strided.call_count, 1)
        self.kernels.run_out.assert_called_once()

    def test_no_buffers_for_rejected_result(self):
        for report in ({}, [], None, dict(valid_report(), verdict="NO_WIN")):
            self.prepare(report)
            self.assertIsNone(self.layer._sm12x_ba4)
        self.torch.empty.assert_not_called()
        self.kernels.run_out.assert_not_called()

    def test_shard_and_device_allowlist(self):
        args = [self.prefix, (48, 5120), (5120, 1), "torch.bfloat16", "cuda", (12, 0), 2, [48, 48]]
        self.assertIsNone(policy.weight_reason(*args))
        for index, value in ((0, "mtp.in_proj_ba"), (1, (96, 5120)), (2, (1, 48)),
                             (3, "torch.float16"), (4, "cpu"), (5, (12, 1)),
                             (6, 1), (7, [96])):
            case = list(args)
            case[index] = value
            self.assertIsNotNone(policy.weight_reason(*case))


class KernelBodyTests(unittest.TestCase):
    def test_actual_kernel_bodies_strided_rows_shard_boundary_and_poisoned_scratch(self):
        from kernel_interpreter import Pointer, bf16, load_kernel_bodies
        source = HERE / "vllm_ba4/kernels.py"
        if not source.exists():  # Installed package during Docker build.
            import vllm_ba4
            source = Path(vllm_ba4.__file__).parent / "kernels.py"
        tl, kernels = load_kernel_bodies(source)
        stride, offset, m = 5136, 8, 4
        # Distinct beta/alpha rows and asymmetric x values catch reordering,
        # transposes, wrong M stride and wrong scratch leading dimensions.
        x = [999.0] * (offset + m * stride)
        for row in range(m):
            for k in range(5120):
                x[offset + row * stride + k] = float((k * 3 + row) % 7 - 3)
        w = [float((n * 5 + k * 2) % 11 - 5) / 4 for n in range(48) for k in range(5120)]
        cols, rows = (23, 24, 47), (0, 3)
        expected = {(row, n): bf16(math.fsum(x[offset + row * stride + k] * w[n * 5120 + k]
                                             for k in range(5120))) for row in rows for n in cols}
        xp, wp = Pointer(x, index=offset), Pointer(w)
        for row in rows:
            for n in cols:
                out = [math.nan] * (m * 48)
                tl.ids = (n, row, 0)
                kernels._direct(xp, wp, Pointer(out, bf16), stride, m, 8192)
                self.assertEqual(out[row * 48 + n], expected[row, n])
                self.assertEqual(sum(not math.isnan(v) for v in out), 1)
        for bk in (128, 256, 512):
            splits = (5120 + bk - 1) // bk
            partial = [math.nan] * (40 * m * 48)
            for row in rows:
                for group in {n // 4 for n in cols}:
                    for s in range(splits):
                        tl.ids = (group, s, row)
                        kernels._partial(xp, wp, Pointer(partial), stride, m, bk, 4)
            out = [math.nan] * (m * 48)
            for row in rows:
                for group in {n // 16 for n in cols}:
                    tl.ids = (group, row, 0)
                    kernels._finish(Pointer(partial), Pointer(out, bf16), m,
                                    splits, 1 << (splits - 1).bit_length(), 16)
            for row in rows:
                for n in cols:
                    self.assertEqual(out[row * 48 + n], expected[row, n])

    def test_stream_reads_last_partial_tile(self):
        from kernel_interpreter import Pointer, load_kernel_bodies
        source = HERE / "vllm_ba4/kernels.py"
        if not source.exists():
            import vllm_ba4
            source = Path(vllm_ba4.__file__).parent / "kernels.py"
        tl, kernels = load_kernel_bodies(source)
        data = [i % 256 for i in range(5007)]
        sink = [-1, -1]
        for block in (0, 1):
            tl.ids = (block, 0, 0)
            kernels._stream_weights(Pointer(data), Pointer(sink, int), len(data), 4096)
        self.assertEqual(sink, [sum(data[:4096]), sum(data[4096:])])


from test_compile import ServingCompileTests
from test_logging import WorkerLoggingTests
from test_symbolic import SymbolicCompileTests
from test_profile import ProfileCompileTests
from test_splitter import SplitterCompileTests
from test_piecewise import PiecewiseCompileTests
from test_cache_safety import CacheSafetyTests
from test_v3_admission import V3AdmissionTests


if __name__ == "__main__":
    unittest.main()
