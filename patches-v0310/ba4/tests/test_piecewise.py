"""Size-four specialization, cache reconstruction and real GDN code equality."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE/'tests'), str(HERE)]


class PiecewiseCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_compile import ServingCompileTests
        ServingCompileTests.setUpClass()
        cls.h = ServingCompileTests()
        cls.torch = cls.h.torch
        cls.torch.set_num_threads(2)

    def tearDown(self):
        self.torch._dynamo.reset()

    def test_real_piecewise_m4_specialization_general_and_warm_ranges(self):
        from compile_wrapper import wrap
        from piecewise_wrapper import backend
        from vllm_ba4.compilation import artifact_key
        torch, h = self.torch, self.h
        for compiler in (None, 'aot_eager', 'inductor'):
            for kind in ('BasevLLMParameter', 'ModelWeightParameter'):
                for first in (4, 32):
                    with self.subTest(compiler=compiler, kind=kind, first=first), torch.no_grad():
                        torch._dynamo.reset()
                        layer = h.layer(kind, profile=True)
                        records = []
                        call, graphs = wrap(h, layer, backend(compiler, records))
                        retained = []
                        for m in (first, 4, 8, 12, 65, 4):
                            x = torch.randn(m, 5120, dtype=torch.bfloat16)
                            h.runtime._used.clear()
                            with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                                value = call(x)
                            self.assertTrue(torch.equal(value, torch.nn.functional.linear(x, layer.weight)))
                            self.assertEqual(selected.call_count, int(m == 4))
                            self.assertEqual(bool(h.runtime._used), m == 4)
                            retained.append((value, value.clone()))
                            for v, saved in retained: self.assertTrue(torch.equal(v, saved))
                        self.assertEqual(len(graphs), 1)
                        record, = records
                        piece = record['piece']
                        self.assertEqual(piece.compile_sizes, [4])
                        self.assertEqual(record['config'].compilation_config.compile_sizes, [])
                        for range, graph in record['graphs']:
                            selected = range.is_single_size() and range.start == 4
                            self.assertEqual('sm12x_ba4_dispatch' in str(graph.graph), selected)
                            if not selected:
                                self.assertNotIn('partial', str(graph.graph))
                        # Reconstruct the real graph=None AOT warm-load path.
                        runnables = {artifact_key(str(r), e.ba4_input_indices): e.runnable
                                     for r, e in piece.range_entries.items()}
                        warm = record['module'].PiecewiseBackend(None, record['config'], 0, 1,
                            record['indices'], record['vbackend'], True, compiled_runnables=runnables)
                        self.assertEqual(warm.compile_sizes, [4])
                        for m in (4, 8, 12, 65):
                            x = torch.randn(m, 5120, dtype=torch.bfloat16)
                            args = []
                            for n in graphs[0].graph.nodes:
                                if n.op == 'placeholder':
                                    args.append(m if n.name.startswith('s') else layer.weight if 'weight' in n.name
                                                else layer._sm12x_ba4.partial if 'partial' in n.name else x)
                            h.runtime._used.clear()
                            with mock.patch.object(h.runtime, 'run_out', wraps=h.runtime.run_out) as selected:
                                value, = warm(*args)
                            self.assertTrue(torch.equal(value, torch.nn.functional.linear(x, layer.weight)))
                            self.assertEqual(selected.call_count, int(m == 4))
                        print('J41R_PIECEWISE ' + json.dumps(dict(compiler=compiler, kind=kind,
                            first=first, artifacts=list(runnables), warm_load=True)), flush=True)

    def test_real_to_bytes_preserves_pruned_aot_signature(self):
        import pickle
        from compile_wrapper import wrap
        from piecewise_wrapper import backend
        torch, h = self.torch, self.h
        with torch.no_grad():
            layer=h.layer(profile=True); records=[]
            call, _ = wrap(h, layer, backend(None, records))
            call(torch.randn(32,5120,dtype=torch.bfloat16))
            piece=records[0]['piece']
            class Serializable:
                def __init__(self, indices): self.indices=indices
                def serialize(self): return dict(test_payload=True, indices=self.indices)
            for entry in piece.range_entries.values():
                entry.runnable=Serializable(entry.ba4_input_indices)
            artifacts=piece.to_bytes()
            self.assertEqual(set(artifacts), {'(4, 4)', '(1, 8192);ba4-inputs=0,1,2'})
            for key,value in artifacts.items():
                self.assertEqual(('submod_0_'+key).rsplit('_',1), ['submod_0',key])
                self.assertEqual(pickle.loads(value),dict(test_payload=True,
                    indices=None if key=='(4, 4)' else (0,1,2)))

    def test_configured_non_four_sizes_have_no_opaque_node(self):
        from compile_wrapper import wrap
        from piecewise_wrapper import backend
        torch, h = self.torch, self.h
        with torch.no_grad():
            layer=h.layer(profile=True); records=[]
            call, _ = wrap(h, layer, backend(None, records, sizes=(8,12)))
            for m in (32,4,8,12): call(torch.randn(m,5120,dtype=torch.bfloat16))
            for range, graph in records[0]['graphs']:
                self.assertEqual('sm12x_ba4_dispatch' in str(graph.graph),
                                 range.is_single_size() and range.start == 4)

    def assert_general_gdn_structure(self, code):
        """Follow the mm -> fused gate copies/zero -> attention buffer edges.

        Descriptive kernel names include different source ancestry across Torch
        versions and naming modes. Resolve the compiled kernel binding and its
        actual operands instead; the complete AST equality below still includes
        every identifier and the unmodified embedded C++ strings.
        """
        tree = ast.parse(code)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        mm, = [n for n in calls if ast.unparse(n.func) == 'extern_kernels.mm']
        attention, = [n for n in calls if ast.unparse(n.func) ==
                      'torch.ops.vllm.qwen_gdn_attention_core.default']
        bmm, = [n for n in calls if ast.unparse(n.func) ==
                'torch.ops.ba4_test.bmm_fp8.default']
        self.assertLess(bmm.lineno, mm.lineno)
        mm_output, = [k.value for k in mm.keywords if k.arg == 'out']
        bindings = {n.targets[0].id: n.value for n in tree.body
                    if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == 'async_compile.cpp_pybinding'}
        operands = [ast.dump(n) for n in [mm_output, *attention.args[1:4]]]
        fused, = [n for n in calls if isinstance(n.func, ast.Name)
                  and n.func.id in bindings
                  and [ast.dump(a) for a in n.args[:4]] == operands]
        self.assertLess(mm.lineno, fused.lineno)
        self.assertLess(fused.lineno, attention.lineno)
        self.assertIsInstance(bindings[fused.func.id].args[1], ast.Constant)
        self.assertIsInstance(bindings[fused.func.id].args[1].value, str)

    def test_general_real_gdn_inductor_program_identical_to_installed(self):
        from compile_wrapper import wrap
        from piecewise_wrapper import backend
        from gdn_wrapper import model, observations
        from torch._inductor.utils import run_and_get_code
        torch, h = self.torch, self.h
        # Execute the installed function itself, not a substitute refusal.
        path = HERE/'fixtures/compile/installed_runtime.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name == 'try_apply')
        ns = dict(h.runtime.__dict__)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        installed = ns['try_apply']
        def program(code):
            # Removes Python comments/source attribution only. Every generated
            # executable statement and C++ kernel string must be identical.
            return ast.dump(ast.parse(code), include_attributes=False)
        dump = os.environ.get('BA4_GDN_DUMP_DIR')
        for first in (4,32):
            with self.subTest(first=first), torch.no_grad():
                real, layer = model(h)
                proxy=types.SimpleNamespace(apply=lambda method,layer,x: real.forward_cuda(x))
                inputs={m:torch.randn(m,5120,dtype=torch.bfloat16) for m in (4,8,12,32,65)}
                outputs={}; codes={}; gates={}
                for variant in ('installed','opaque','final'):
                    torch._dynamo.reset();records=[]
                    with mock.patch.object(h.runtime,'try_apply',installed if variant=='installed' else h.runtime.try_apply):
                        call, graphs = wrap(proxy,layer,'inductor' if variant=='opaque' else backend('inductor',records))
                        _, generated = run_and_get_code(lambda:call(inputs[first]))
                        general=[c for c in generated if variant=='opaque' or 'sm12x_ba4_dispatch.default' not in c]
                        self.assertEqual(len(general),1)
                        if variant == 'final':
                            self.assertEqual(sum('sm12x_ba4_dispatch.default' in c for c in generated), 1)
                            for range, graph in records[0]['graphs']:
                                self.assertEqual('sm12x_ba4_dispatch' in str(graph.graph),
                                                 range.is_single_size() and range.start == 4)
                        codes[variant]=general[0]
                        outputs[variant]={}; gates[variant]={}
                        for m,x in inputs.items():
                            h.runtime._used.clear()
                            with mock.patch.object(h.runtime,'run_out',wraps=h.runtime.run_out) as selected:
                                observations.clear()
                                outputs[variant][m]=call(x).clone()
                                gates[variant][m]=observations[-1]
                            self.assertEqual(selected.call_count,int(variant!='installed' and m==4))
                        if dump:
                            dest=Path(dump);dest.mkdir(parents=True,exist_ok=True)
                            (dest/f'{variant}-first{first}-fx.txt').write_text(str(graphs[0].graph))
                            for i,c in enumerate(generated): (dest/f'{variant}-first{first}-{i}.py').write_text(c)
                self.assertNotEqual(program(codes['installed']),program(codes['opaque']))
                self.assertEqual(program(codes['installed']),program(codes['final']))
                for m in inputs:
                    self.assertTrue(torch.equal(outputs['installed'][m],outputs['final'][m]))
                    for before,after in zip(gates['installed'][m],gates['final'][m]):
                        self.assertTrue(torch.equal(before,after))
                digest=hashlib.sha256(program(codes['final']).encode()).hexdigest()
                self.assertNotIn('sm12x_ba4',codes['final'])
                self.assert_general_gdn_structure(codes['final'])
                print('J41R_GDN_IDENTICAL '+json.dumps(dict(first=first,program_ast_sha256=digest,
                    torch_version=torch.__version__,cpp_descriptive_names=torch._inductor.config.cpp.descriptive_names,
                    structural_buffer_edges=True,
                    compared_M=[8,12,32,65],m4_selected=True,qkvz_static_fp8=True,gate_inputs_identical=True,opaque_program_differs=True,complete_generated_program_identical_except_comments=True,
                    real_module_sha256=hashlib.sha256((HERE/'fixtures/base/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py').read_bytes()).hexdigest())),flush=True)

    def test_general_real_gdn_inductor_program_with_node_names(self):
        # Run the complete equality/negative-control/size-four gate with a
        # different real Inductor naming policy in the image build itself.
        dump = os.environ.get('BA4_GDN_DUMP_DIR')
        env = {'BA4_GDN_DUMP_DIR': str(Path(dump)/'inductor_node')} if dump else {}
        with self.torch._inductor.config.patch({'cpp.descriptive_names': 'inductor_node'}), \
             mock.patch.dict(os.environ, env):
            self.test_general_real_gdn_inductor_program_identical_to_installed()


if __name__ == '__main__': unittest.main(verbosity=2)
