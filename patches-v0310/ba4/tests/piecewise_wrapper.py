"""Run the complete patched A-image PiecewiseBackend, including warm loading."""
import ast
import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_backend():
    import torch
    import torch._inductor.codecache
    import torch._functorch.aot_autograd
    import make_layer
    fixture = HERE / 'fixtures/compile/range_utils.py'
    tree = ast.parse(fixture.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Range')
    ns = dict(dataclass=dataclasses.dataclass)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(fixture), 'exec'), ns)
    Range = ns['Range']
    pkg = types.ModuleType('vllm'); pkg.__path__ = []
    envs = types.ModuleType('vllm.envs'); envs.VLLM_SM12X_BA_M4 = True; pkg.envs = envs
    backends = types.ModuleType('vllm.compilation.backends'); backends.VllmBackend = object
    config = types.ModuleType('vllm.config'); config.VllmConfig = object
    utils = types.ModuleType('vllm.config.utils'); utils.Range = Range
    logger = types.ModuleType('vllm.logger')
    logger.init_logger = lambda name: types.SimpleNamespace(debug_once=lambda *a: None, debug=lambda *a: None)
    modules = {'vllm': pkg, 'vllm.envs': envs, 'vllm.compilation.backends': backends,
               'vllm.config': config, 'vllm.config.utils': utils, 'vllm.logger': logger}
    module = types.ModuleType('ba4_real_piecewise_backend')
    sys.modules[module.__name__] = module
    previous = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        source = make_layer.changes()['vllm/compilation/piecewise_backend.py'][1]
        exec(compile(source, 'real_vllm_piecewise_backend.py', 'exec'), module.__dict__)
    finally:
        for k, v in previous.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    return module, Range


def backend(compiler='inductor', records=None, sizes=(), ranges=((1,8192),)):
    import torch
    module, Range = load_backend()
    def compile(gm, inputs):
        cfg = types.SimpleNamespace(compile_sizes=list(sizes),
            get_compile_ranges=lambda: [Range(*r) for r in ranges])
        config = types.SimpleNamespace(compilation_config=cfg)
        compiled = []
        def compile_range(graph, args, *a, **kw):
            compiled.append((kw['compile_range'], graph))
            # Match the real InductorAdaptor's tracing-context isolation for
            # create_concrete_args' distinct FakeTensorMode.
            with torch._guards.tracing(None):
                return graph.forward if compiler is None else torch._dynamo.lookup_backend(compiler)(graph, args)
        vbackend = types.SimpleNamespace(is_encoder=False, inductor_config={},
            compiler_manager=types.SimpleNamespace(compile=compile_range))
        indices = [i for i, x in enumerate(inputs) if isinstance(x, torch.SymInt)]
        piece = module.PiecewiseBackend(gm, config, 0, 1, indices, vbackend, True)
        if records is not None:
            records.append(dict(piece=piece, graphs=compiled, module=module, config=config,
                                vbackend=vbackend, indices=indices))
        return piece
    return compile
