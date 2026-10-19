#!/usr/bin/env python3
"""Reproduce the exact diff/hash manifest from pinned fixtures and helpers."""

import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "vllm-qwen38:v0310-r27c-A_S-L1a-ubplan"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("patch anchor count !=1: " + before[:80])
    return text.replace(before, after, 1)


def changes():
    result = {}
    rel = "vllm/model_executor/layers/linear.py"
    before = (HERE / "fixtures/base" / rel).read_text()
    after = replace_once(before,
        "        self._gemm_impl = dispatch_unquantized_gemm(linear_backend)\n",
        "        self._gemm_impl = dispatch_unquantized_gemm(linear_backend)\n"
        "        if envs.VLLM_SM12X_BA_M4 and current_platform.is_cuda():\n"
        "            # Construction has the current config; postload need not.\n"
        "            if config is None:\n"
        "                raise RuntimeError(\"BA4 requires construction-time vLLM config\")\n"
        "            self._sm12x_ba4_capacity = max(\n"
        "                4, config.scheduler_config.max_num_batched_tokens,\n"
        "                config.compilation_config.max_cudagraph_capture_size or 0)\n")
    after = replace_once(after,
        "\n    def apply(\n        self,\n        layer: torch.nn.Module,\n        x: torch.Tensor,\n        bias: torch.Tensor | None = None,\n    ) -> torch.Tensor:\n        if envs.VLLM_BATCH_INVARIANT",
        "\n        # N1: boot-time setup only; disabled means no import/allocation/logging.\n"
        "        if envs.VLLM_SM12X_BA_M4 and current_platform.is_cuda():\n"
        "            from vllm_ba4.runtime import prepare_layer\n\n"
        "            prepare_layer(layer, output_capacity=self._sm12x_ba4_capacity)\n\n"
        "    def apply(\n        self,\n        layer: torch.nn.Module,\n"
        "        x: torch.Tensor,\n        bias: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor:\n        if envs.VLLM_BATCH_INVARIANT")
    after = replace_once(after,
        "            return linear_batch_invariant(x, layer.weight, bias)\n"
        "        return self._gemm_impl(layer, x, layer.weight, bias)",
        "            return linear_batch_invariant(x, layer.weight, bias)\n"
        "        if envs.VLLM_SM12X_BA_M4 and current_platform.is_cuda():\n"
        "            from vllm_ba4.runtime import try_apply\n\n"
        "            ba4_result = try_apply(layer, x, bias)\n"
        "            if ba4_result is not None:\n"
        "                return ba4_result\n"
        "        return self._gemm_impl(layer, x, layer.weight, bias)")
    result[rel] = before, after
    rel = "vllm/envs.py"
    before = (HERE / "fixtures/base" / rel).read_text()
    after = replace_once(before, "    VLLM_BATCH_INVARIANT: bool = False\n",
                         "    VLLM_BATCH_INVARIANT: bool = False\n"
                         "    VLLM_SM12X_BA_M4: bool = False\n"
                         "    VLLM_SM12X_BA_M4_RESULT: str = \"\"\n")
    anchor = '    "VLLM_BATCH_INVARIANT": lambda: bool(int(os.getenv("VLLM_BATCH_INVARIANT", "0"))),\n'
    after = replace_once(after, anchor, anchor +
        '    # N1: enable only this layer/shape after serving PROFILE qualification (or isolated PROFILE_ONLY).\n'
        '    "VLLM_SM12X_BA_M4": lambda: os.getenv("VLLM_SM12X_BA_M4", "0") == "1",\n'
        '    "VLLM_SM12X_BA_M4_RESULT": lambda: os.getenv("VLLM_SM12X_BA_M4_RESULT", ""),\n')
    after = replace_once(after, '    VLLM_SM12X_BA_M4: bool = False\n',
                         '    VLLM_SM12X_BA_M4: bool = False\n'
                         '    VLLM_SM12X_BA_M4_PROFILE_ONLY: bool = False\n'
                         '    VLLM_SM12X_BA_M4_BINDING: str = ""\n')
    marker = '    "VLLM_SM12X_BA_M4_RESULT": lambda: os.getenv("VLLM_SM12X_BA_M4_RESULT", ""),\n'
    after = replace_once(after, marker, marker +
        '    "VLLM_SM12X_BA_M4_PROFILE_ONLY": lambda: os.getenv("VLLM_SM12X_BA_M4_PROFILE_ONLY", "0") == "1",\n'
        '    "VLLM_SM12X_BA_M4_BINDING": lambda: os.getenv("VLLM_SM12X_BA_M4_BINDING", ""),\n')
    result[rel] = before, after
    rel = "vllm/compilation/piecewise_backend.py"
    before = (HERE / "fixtures/compile/piecewise_backend.py").read_text()
    after = replace_once(before, "from vllm.config import VllmConfig\n",
                         "from vllm import envs\nfrom vllm.config import VllmConfig\n")
    after = replace_once(after,
        "        self.compile_sizes = self.compilation_config.compile_sizes\n",
        "        self.compile_sizes = self.compilation_config.compile_sizes\n"
        "        if envs.VLLM_SM12X_BA_M4 and sym_shape_indices:\n"
        "            from vllm_ba4.compilation import compile_sizes\n\n"
        "            self.compile_sizes = compile_sizes(\n"
        "                self.compile_sizes, graph, compiled_runnables)\n")
    after = replace_once(after,
        "            range_entry.runnable = self.vllm_backend.compiler_manager.compile(\n"
        "                self.graph,\n",
        "            compile_graph = self.graph\n"
        "            if envs.VLLM_SM12X_BA_M4:\n"
        "                from vllm_ba4.compilation import graph_for_range\n\n"
        "                compile_graph = graph_for_range(\n"
        "                    self.graph, range_entry.compile_range)\n"
        "            range_entry.runnable = self.vllm_backend.compiler_manager.compile(\n"
        "                compile_graph,\n")
    after = replace_once(after,
        "    compiled: bool = False\n", "    compiled: bool = False\n"
        "    ba4_input_indices: tuple[int, ...] | None = None\n")
    after = replace_once(after,
        "                    self.graph, range_entry.compile_range)\n",
        "                    self.graph, range_entry.compile_range)\n"
        "                if compile_graph is not self.graph:\n"
        "                    from vllm_ba4.compilation import prune_inputs\n\n"
        "                    compile_graph, args_list, range_entry.ba4_input_indices = (\n"
        "                        prune_inputs(compile_graph, args_list))\n")
    after = replace_once(after,
        "                out[str(range_key)] = serialize(entry.runnable)\n",
        "                key = str(range_key)\n"
        "                if entry.ba4_input_indices is not None:\n"
        "                    from vllm_ba4.compilation import artifact_key\n\n"
        "                    key = artifact_key(key, entry.ba4_input_indices)\n"
        "                out[key] = serialize(entry.runnable)\n")
    after = replace_once(after,
        "            key = str(range_entry.compile_range)\n",
        "            key = str(range_entry.compile_range)\n"
        "            if envs.VLLM_SM12X_BA_M4:\n"
        "                from vllm_ba4.compilation import loaded_key\n\n"
        "                key, range_entry.ba4_input_indices = loaded_key(\n"
        "                    key, self.compiled_runnables)\n")
    after = replace_once(after,
        "        return range_entry.runnable(*args)\n",
        "        if range_entry.ba4_input_indices is not None:\n"
        "            args = tuple(args[i] for i in range_entry.ba4_input_indices)\n"
        "        return range_entry.runnable(*args)\n")
    result[rel] = before, after
    for path in sorted((HERE / "vllm_ba4").glob("*.py")):
        result["vllm_ba4/" + path.name] = "", path.read_text()
    return result


def main():
    (HERE / "vllm_ba4/benchmark_source.py").write_bytes(
        (HERE.parent / "ba4_microbench.py").read_bytes())
    files = changes()
    patch = "".join("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/" + rel if before else "/dev/null", tofile="b/" + rel))
        for rel, (before, after) in files.items())
    (HERE / "L1a_BA4.diff").write_text(patch)
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    manifest = {"base": BASE, "patch_sha256": sha(patch),
                "base_image_id": "sha256:631a6f24ccca2f5470a3d0772d833670a16613c69bb4191663f718e473086f5e",
                "before": {r: sha(b) for r, (b, a) in files.items() if b},
                "after": {r: sha(a) for r, (b, a) in files.items()},
                "unchanged": {str(p.relative_to(HERE / "fixtures/base")): sha(p.read_text())
                              for p in sorted((HERE / "fixtures/base").rglob("*.py"))
                              if str(p.relative_to(HERE / "fixtures/base")) not in files}}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
