"""Execute the complete archived vLLM wrapper, substituting config/platform only."""
import ast
from contextlib import nullcontext
from enum import Enum, IntEnum
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types
from unittest import mock

FIXTURE = Path(__file__).resolve().parent.parent / 'fixtures/compile'


def wrap(harness, layer, backend=None, *, aot=False):
    import torch
    import torch._inductor.codecache  # preload before isolated vLLM imports
    for name, sha in json.loads((FIXTURE / 'source.json').read_text())['files'].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == sha
    class Mode(IntEnum):
        STOCK_TORCH_COMPILE = 2
        VLLM_COMPILE = 3
    class GraphMode(IntEnum):
        NONE = 0
    class Shapes(Enum):
        BACKED = 'backed'
        UNBACKED = 'unbacked'
    graphs = []
    def record(gm, inputs, **options):
        graphs.append(gm)
        return gm.forward if backend is None else (backend if callable(backend) else torch._dynamo.lookup_backend(backend))(gm, inputs)
    config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(
        mode=Mode.VLLM_COMPILE, init_backend=lambda *a, **kw: record,
        dynamic_shapes_config=types.SimpleNamespace(type=Shapes.BACKED, evaluate_guards=False)),
        observability_config=types.SimpleNamespace(enable_layerwise_nvtx_tracing=False))
    envs = types.ModuleType('vllm.envs')
    envs.VLLM_USE_BYTECODE_HOOK = False
    envs.VLLM_USE_AOT_COMPILE = aot
    pkg = types.ModuleType('vllm'); pkg.__path__ = []; pkg.envs = envs
    cfg = types.ModuleType('vllm.config'); cfg.CompilationMode = Mode; cfg.CUDAGraphMode = GraphMode
    cfg.get_current_vllm_config = lambda: config
    shape = types.ModuleType('vllm.config.compilation'); shape.DynamicShapesType = Shapes
    hooks = types.ModuleType('vllm.utils.nvtx_pytorch_hooks')
    hooks.layerwise_nvtx_marker_context = nullcontext
    override = types.ModuleType('vllm.env_override')
    override._apply_constrain_to_fx_strides_patch = lambda: None
    import logging
    logger = types.ModuleType('vllm.logger'); logger.init_logger = logging.getLogger
    modules = {'vllm': pkg, 'vllm.envs': envs, 'vllm.config': cfg,
        'vllm.config.compilation': shape, 'vllm.utils.nvtx_pytorch_hooks': hooks,
        'vllm.env_override': override, 'vllm.logger': logger}
    spec = importlib.util.spec_from_file_location('ba4_real_compile_wrapper', FIXTURE / 'wrapper.py')
    wrapper = importlib.util.module_from_spec(spec)
    previous = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        spec.loader.exec_module(wrapper)
        class Model(wrapper.TorchCompileWithNoGuardsWrapper):
            def forward(self, x):
                return harness.apply(layer.quant_method, layer, x)
        model = Model()
    finally:
        for k, value in previous.items():
            if value is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = value
    # Execute the real decorator's complete dynamic-input marker, including
    # argument binding. Only its enclosing decorator configuration is supplied.
    tree = ast.parse((FIXTURE / 'decorators.py').read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_support_torch_compile')
    node = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == '_mark_dynamic_inputs')
    ns = dict(torch=torch, inspect=inspect, DynamicShapesType=Shapes,
        dynamic_arg_dims={'x': 0}, mark_unbacked_dims=None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(FIXTURE / 'decorators.py'), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), ns)
    aot_fn = None
    def call(x):
        nonlocal aot_fn
        ns['_mark_dynamic_inputs'](model, Shapes.BACKED, x)
        if aot:
            if aot_fn is None:
                aot_fn = model.aot_compile(x)
            return aot_fn(model, x)
        return model(x)
    return call, graphs
