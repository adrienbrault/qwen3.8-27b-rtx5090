#!/usr/bin/env python3
# Usage (offline): python3 deliver/test_0140.py .work/0140
import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from build_shape_census import census
from make_dispatch_table import make_table
from nvfp4_gemm_census import SHAPES, weight_bytes

ROOT = Path(sys.argv.pop(1))
spec=importlib.util.spec_from_file_location('parser', ROOT/'vllm/nvfp4_dispatch_table.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
A,B=p.SHARED[:2]


class Tests(unittest.TestCase):
    def parse(self, value):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            f.write(value); f.flush(); p.read_table.cache_clear()
            return p.read_table(f.name)

    def test_grammar(self):
        for value in ['', '[]', '{}', '{"rules":{}}', '{"rules":[{}]}']:
            with self.assertRaises(ValueError): self.parse(value)
        for limit in [True, 0, -1, 1.5, '10']:
            with self.assertRaises(ValueError):
                self.parse(json.dumps({'rules':[dict(layer='x',m_max=limit,kernel=A)]}))
        for pattern,kernel in [('[',A),('',A),('x','MarlinNvFp4LinearKernel'),('x','bogus')]:
            with self.assertRaises(ValueError):
                self.parse(json.dumps({'rules':[dict(layer=pattern,m_max=None,kernel=kernel)]}))
        with self.assertRaises(ValueError): p.read_table('')
        with self.assertRaises(ValueError): p.read_table('/does/not/exist/0140')
        self.assertEqual(self.parse('{"rules":[]}'), ())

    def test_first_match(self):
        rng=random.Random(23)
        for _ in range(100):
            rules=tuple((rng.choice([None,1,10,80,160]),rng.choice([A,B])) for _ in range(8))
            table=p.buckets(rules,A)
            for m in [1,2,9,10,11,79,80,81,159,160,161,8192]:
                expected=next((n for bound,n in rules if bound is None or m<=bound),A)
                actual=next(n for bound,n in table if bound is None or m<=bound)
                self.assertEqual(expected,actual)

    def test_runtime_wrapper(self):
        # Execute the exact wrapper class without its GPU imports; fake tensor shape only.
        tree=ast.parse((ROOT/'vllm/model_executor/kernels/linear/nvfp4/dispatch.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        ns={}; exec(compile(ast.Module(body=[cls],type_ignores=[]),'<wrapper>','exec'),ns)
        class Kernel:
            def __init__(self): self.prepares=0
            def process_weights_after_loading(self,layer): self.prepares+=1
            def apply_weights(self,layer,x,bias): return self
        class X:
            shape=(2,5,5120)
            def numel(self): return 2*5*5120
        a,b=Kernel(),Kernel(); w=ns['SharedLayoutDispatch'](a,((10,b),(None,a)))
        w.process_weights_after_loading(None)
        self.assertEqual((a.prepares,b.prepares),(1,0))
        self.assertIs(w.apply_weights(None,X()),b)
        self.assertIsNone(w.input_quant_key())

    def test_census(self):
        rows=census(json.loads(Path('evidence/target-config.json').read_text()))
        self.assertEqual(sum(r[3]=='NVFP4 W4A4' for r in rows),112)
        self.assertEqual(sum(r[5]=='in_proj_qkvz' for r in rows),48)
        self.assertEqual(sum(r[5]=='qkv' for r in rows),16)
        for name,n,k,scheme,size,shape in rows:
            if shape!='—':
                self.assertEqual((n,k),SHAPES[shape][:2])
                self.assertEqual(size,weight_bytes(n,k,SHAPES[shape][2]))

    def test_generator_isolates_m(self):
        data={'baseline_nvfp4':A,'records':[
            dict(shape='gate_up',M=m,kernel=k,us=t) for m in [10,80]
            for k,t in [(A,100),(B,90)]]}
        generated=make_table(data)
        rules=self.parse(json.dumps(generated))
        match=p.module_rules(rules,'language_model.model.layers.55.mlp.gate_up_proj')
        table=p.buckets(match,A)
        for m in [1,9,10,11,79,80,81,8192]:
            self.assertEqual(next(k for bound,k in table if bound is None or m<=bound),B if m in [10,80] else A)
        self.assertFalse(p.module_rules(rules,'language_model.model.layers.56.mlp.gate_up_proj'))
        self.assertFalse(p.module_rules(rules,'draft.layers.0.self_attn.qkv_proj'))


unittest.main()
