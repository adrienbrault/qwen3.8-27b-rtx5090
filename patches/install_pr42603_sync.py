#!/usr/bin/env python3
"""Graft vLLM PR #42603 — fix MTP draft-loop buffer race (cudaErrorIllegalAddress
under concurrency + fp8 KV on Blackwell). Adds a stream sync after the draft
input-buffer writes and before the draft model forward, in
vllm/v1/spec_decode/llm_base_proposer.py. Idempotent, ast.parse-verified."""

import ast
import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)
source = open(path).read()

marker = "# PR42603: commit draft input-buffer writes before draft forward"
if marker in source:
    ast.parse(source)
    print("already patched")
    sys.exit(0)

# Anchor: the draft-loop "Run the model." block that consumes the just-written
# self.input_ids / self.hidden_states / self.inputs_embeds cudagraph buffers.
anchor = '''            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
'''

replacement = '''            # PR42603: commit draft input-buffer writes before draft forward
            # The loop writes shared GPU buffers (input_ids / hidden_states /
            # inputs_embeds) then immediately launches the draft forward that
            # reads them. CUDA ops are async, so under concurrency the draft
            # (FlashInfer) kernels can observe stale/partial buffers -> illegal
            # memory access. Stream-scoped sync makes the writes visible first.
            torch.accelerator.current_stream().synchronize()

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
'''

count = source.count(anchor)
assert count == 1, f"Run-the-model anchor found {count}x (expected 1)"
patched = source.replace(anchor, replacement, 1)

# Ensure torch is importable in this module.
assert "import torch" in patched, "torch not imported in llm_base_proposer.py"

ast.parse(patched)
open(path, "w").write(patched)
print("PATCHED_PR42603_MTP_DRAFT_SYNC")
