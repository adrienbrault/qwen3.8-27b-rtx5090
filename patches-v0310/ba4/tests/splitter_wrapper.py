"""Execute pinned vLLM attention splitting, interpreter and piecewise backend."""
import __future__
import ast
from collections import defaultdict
import copy
import dataclasses
import operator
from pathlib import Path
import sys
import types
from unittest import mock
import torch
from torch.fx._lazy_graph_module import _use_lazy_graph_module

FIXTURE = Path(__file__).resolve().parent.parent / 'fixtures/compile'


def load_splitter():
    try:
        from torch._dynamo.aot_compile import SerializableCallable
    except ImportError:
        SerializableCallable = object  # Same compatibility branch as vLLM.
    # Complete definitions, with real Torch implementations. No rewritten node
    # metadata, TreeSpec filter, DCE or alternate partition algorithm.
    loaded = types.ModuleType('ba4_real_splitter')
    sys.modules[loaded.__name__] = loaded
    ns = loaded.__dict__
    ns.update(torch=torch, fx=torch.fx, dataclasses=dataclasses,
              operator=operator, defaultdict=defaultdict,
              _use_lazy_graph_module=_use_lazy_graph_module,
              is_torch_equal_or_newer=lambda version: tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 12),
              instrument=lambda **kw: lambda f: f,
              compilation_counter=types.SimpleNamespace(num_piecewise_capturable_graphs_seen=0),
              SerializableCallable=SerializableCallable,
              __name__='ba4_real_splitter', __package__='vllm.compilation')
    definitions = {'partition_rules.py': {'should_split'}, 'backends.py': {
        'SplitItem', '_is_empty_allocation_node', '_merge_empty_only_subgraphs',
        '_decompose_size_nodes', 'split_graph', 'PiecewiseCompileInterpreter',
        'wrap_with_cudagraph_if_needed'}, 'caching.py': {'VllmSerializableFunction'}}
    for name, names in definitions.items():
        tree = ast.parse((FIXTURE/name).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FIXTURE/name), 'exec',
                     flags=__future__.annotations.compiler_flag), ns)
    tree = ast.parse((FIXTURE/'compilation_config.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CompilationConfig')
    assignment = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == '_attention_ops')
    return ns, ast.literal_eval(assignment.value)


def backend(compiler, records):
    from piecewise_wrapper import load_backend
    module, Range = load_backend()
    ns, splitting_ops = load_splitter()
    def compile(gm, inputs):
        split, items = ns['split_graph'](gm, splitting_ops)
        record = dict(split=split, items=items, graphs=[], splitting_ops=splitting_ops)
        records.append(record)
        cfg = types.SimpleNamespace(compile_sizes=[],
            get_compile_ranges=lambda: [Range(1, 8192)],
            cudagraph_mode=types.SimpleNamespace(has_piecewise_cudagraphs=lambda: False),
            use_inductor_graph_partition=False)
        config = types.SimpleNamespace(compilation_config=cfg)
        def compile_range(graph, args, *a, **kw):
            record['graphs'].append((kw['compile_range'], graph))
            if compiler is None:
                return graph.forward
            from torch._inductor.compile_fx import graph_returns_tuple
            returns_tuple = graph_returns_tuple(graph)
            compile_graph = graph
            if not returns_tuple:
                # Torch 2.8 split_module has no tuple_return=True option.
                # Normalize the compiler ABI and unpack its result, matching
                # vLLM's compiler-interface convention. The real splitter and
                # piecewise graph stay untouched. 2.13 uses native tuples.
                compile_graph = copy.deepcopy(graph)
                output = compile_graph.graph.output_node()
                output.args = ((output.args[0],),)
                compile_graph.recompile()
            with torch._guards.tracing(None):
                compiled = torch._dynamo.lookup_backend(compiler)(compile_graph, args)
            return compiled if returns_tuple else lambda *args: compiled(*args)[0]
        vbackend = types.SimpleNamespace(is_encoder=False, inductor_config={},
            compiler_manager=types.SimpleNamespace(compile=compile_range))
        names = [item.submod_name for item in items if not item.is_splitting_graph]
        fake_args = [n.meta['example_value'] for n in gm.graph.nodes if n.op == 'placeholder']
        with mock.patch.dict(sys.modules, {'vllm.compilation.piecewise_backend': module}):
            ns['PiecewiseCompileInterpreter'](split, names, config, vbackend).run(*fake_args)
        record['pieces'] = [getattr(split, name) for name in names]
        # Use vLLM's actual SerializableCallable wrapper so the real Torch
        # wrapper.aot_compile path validates its backend result type too.
        return ns['VllmSerializableFunction'](gm, inputs, 'splitter-test', split.forward)
    return compile
