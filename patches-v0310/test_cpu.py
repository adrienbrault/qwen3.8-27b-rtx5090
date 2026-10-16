"""CPU regression entrypoint; full original constructors, stubbed GPU layers.

No vLLM installation required. Linux --torch uses real CPU nn.Module/ModuleList.
This checks Python construction/name resolution, not CUDA loading or numerics.
"""
import argparse
import ast
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    import torch
    Module = torch.nn.Module
    ModuleList = torch.nn.ModuleList
except ImportError:
    torch = None
    class Module:
        def __init__(self, *args, **kwargs):
            pass
    ModuleList = list


def config(mode="off"):
    text = NS(model_type="qwen3_5_text", hidden_size=8, intermediate_size=16,
              hidden_act="silu", rms_norm_eps=1e-6, vocab_size=32,
              num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
              tie_word_embeddings=False, mtp_num_hidden_layers=1)
    mm = NS(mm_encoder_tp_mode="weights", video_pruning_rate=0,
            is_multimodal_pruning_enabled=lambda: False)
    return NS(model_config=NS(hf_text_config=text,
                             hf_config=NS(vision_config=NS(out_hidden_size=8)),
                             get_multimodal_config=lambda: mm),
              cache_config=NS(mamba_cache_mode="align", use_replayssm=mode=="standard",
                              use_replayssm_spec=mode=="spec", replayssm_buffer_len=16),
              scheduler_config=NS(), quant_config=None,
              parallel_config=NS(use_sequence_parallel_moe=False,
                                 pipeline_parallel_size=1,
                                 eplb_config=NS(num_redundant_experts=0)))


def constructors(tree):
    """Retain every executable constructor statement and actual target nesting."""
    calls = []
    class Layer(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            calls.append(kwargs)
        def tie_weights(self, value):
            return self
    class NextLayer(Module):
        pass
    class NextModel(Module):
        pass
    class VisionParent(Module):
        def _mark_tower_model(self, *args):
            return nullcontext()
        def _mark_language_model(self, *args):
            return nullcontext()
    def make_layers(n, factory, prefix):
        return 0, n, ModuleList(factory(f"{prefix}.{i}") for i in range(n))
    scope = dict(nn=NS(Module=Module),
                 torch=torch or NS(nn=NS(ModuleList=ModuleList)),
                 Qwen3NextDecoderLayer=NextLayer, Qwen3NextModel=NextModel,
                 VisionParent=VisionParent, Module=Module,
                 extract_layer_index=lambda p: int(re.findall(r"\d+", p)[-1]),
                 get_pp_group=lambda: NS(world_size=1, is_last_rank=True),
                 maybe_prefix=lambda p, n: f"{p}.{n}" if p else n,
                 make_layers=make_layers,
                 make_empty_intermediate_tensors_factory=lambda *a: lambda *b: None,
                 is_model_fused_shared_expert_compatible=lambda *a: False,
                 cached_tokenizer_from_config=lambda *a: None)
    for name in ("Qwen3NextAttention", "QwenGatedDeltaNetAttention",
                 "Qwen3NextMLP", "Qwen3NextSparseMoeBlock", "Qwen3_5RMSNorm",
                 "ParallelLMHead", "PPMissingLayer", "LogitsProcessor",
                 "VocabParallelEmbedding", "ColumnParallelLinear",
                 "Qwen3_VisionTransformer"):
        scope[name] = Layer
    targets = {
        "qwen3_5.py": {
            "Qwen3_5DecoderLayer": "Qwen3NextDecoderLayer",
            "Qwen3_5Model": "Qwen3NextModel",
            "Qwen3_5ForCausalLMBase": "Module",
            "Qwen3_5ForCausalLM": "Qwen3_5ForCausalLMBase",
            "Qwen3_5ForConditionalGeneration": "VisionParent",
        },
        "qwen3_5_mtp.py": {
            "Qwen3_5MultiTokenPredictor": "Module",
            "Qwen3_5MTP": "Module",
        },
    }
    for filename, classes in targets.items():
        path = Path(tree) / "vllm/model_executor/models" / filename
        nodes = {n.name: n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.ClassDef)}
        for name, base in classes.items():
            cls = copy.deepcopy(nodes[name])
            cls.bases = [ast.Name(id=base, ctx=ast.Load())]
            cls.keywords = []
            cls.decorator_list = []
            cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == "__init__"] or [ast.Pass()]
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
            exec(compile(module, str(path), "exec"), scope)
    return scope, calls


class ConstructorChecks(unittest.TestCase):
    def test_served_dense_target_and_mtp_full_constructors_off_and_on(self):
        # Served Qwen3.8 uses the Qwen3_5 architectures, not a separate model file.
        for phase in ("A", "B"):
            for mode in ("off", "standard", "spec"):
                for model in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
                              "Qwen3_5MTP"):
                    with self.subTest(phase=phase, mode=mode, model=model):
                        scope, calls = constructors(OUT / f"work-{phase}")
                        cfg = config(mode)
                        obj = scope[model](vllm_config=cfg)
                        attention = [c for c in calls if "cache_config" in c]
                        self.assertTrue(attention, "full attention constructor not reached")
                        self.assertTrue(all(c["cache_config"] is cfg.cache_config
                                            for c in attention))
                        self.assertTrue(hasattr(obj, "make_empty_intermediate_tensors"))

    def test_all_mode_still_rejected_before_layer_allocation(self):
        for phase in ("A", "B"):
            for model in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MTP"):
                scope, calls = constructors(OUT / f"work-{phase}")
                cfg = config()
                cfg.cache_config.mamba_cache_mode = "all"
                with self.assertRaises(NotImplementedError):
                    scope[model](vllm_config=cfg)

    def test_no_duplicate_stacked_fields_or_imports(self):
        cache = ast.parse((OUT / "work-B/vllm/config/cache.py").read_text())
        cls = next(n for n in cache.body if isinstance(n, ast.ClassDef) and n.name == "CacheConfig")
        names = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
        self.assertEqual(names.count("use_replayssm_spec"), 1)
        source = (OUT / "work-B/vllm/model_executor/models/qwen3_5.py").read_text()
        imports = [a.name for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.ImportFrom) for a in n.names]
        self.assertEqual(imports.count("SupportsReplaySSM"), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true", help="require real CPU torch")
    parser.add_argument("--tree", type=Path, help="build smoke: installed vLLM parent directory")
    args = parser.parse_args()
    if args.torch and torch is None:
        parser.error("--torch requires CPU torch; use linux-check.sh --torch")
    print(f"Python {sys.version.split()[0]}; torch {getattr(torch, '__version__', 'absent (macOS layer stubs)')}", flush=True)
    if args.tree:
        # This branch is used only in the offline candidate image, which has
        # real vLLM dependencies. Prefer the requested source tree explicitly.
        sys.path.insert(0, str(args.tree))
        from vllm.config import CacheConfig
        cache = CacheConfig()
        assert cache.use_replayssm is False
        assert getattr(cache, "use_replayssm_spec", False) is False
        for mode in ("off", "standard", "spec"):
            for name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MTP"):
                scope, _ = constructors(args.tree)
                scope[name](vllm_config=config(mode))
        print("Build constructor smoke: 9 full constructors PASS")
        return
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConstructorChecks)
    static = load_file("r27c_static", OUT / "tests/static_contracts.py")
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(static.Contracts))
    extra = OUT / "tests/r27c_contracts.py"
    if extra.exists():
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(load_file("r27c_contracts", extra)))
    if torch is not None:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(
            load_file("r27e_capture", OUT / "tests/r27e_capture.py")))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
