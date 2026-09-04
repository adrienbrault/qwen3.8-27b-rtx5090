# SPDX-License-Identifier: Apache-2.0
"""Read-only R184 tactics, explicitly transplanted to the smaller R185 slab.

This is deliberately not a FlashInfer AutoTuner cache loader. The file retains
its original metadata/capacity; only the validated launch triples are reused.
A smaller slab is legal with these triples, but its performance is unmeasured.
"""

import ast
import hashlib
import json
from pathlib import Path

MAX_ROWS = 320
MAX_NUMEL = MAX_ROWS * 5120
MAX_BLOCKS = 128
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
CAPTURE_ROWS = (
    1, 2, 4, 8, 10, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88,
    96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
    200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320,
)
CACHE_PATH = Path(__file__).with_name("pcie-tune-free.json")


class FixedConfigs:
    def __init__(self):
        raw = CACHE_PATH.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw)
        meta = document["_metadata"]
        if (meta["gpu"], meta["cuda_version"]) != (
            "NVIDIA GeForce RTX 5090", "13.0"
        ):
            raise ValueError("unexpected R184 tuning hardware/toolchain")
        self.table = {}
        for text, value in document.items():
            if text in ("_metadata", "_generation"):
                continue
            op, runner, shapes, extras = ast.literal_eval(text)
            if (op, runner, extras) != (
                "flashinfer::pcie_ipc_all_reduce",
                "PcieIpcAllReduceRunner",
                (1, 2, "rootcplx-noswitch", 128, 41943040, "torch.bfloat16"),
            ):
                raise ValueError("unexpected R184 tuning key")
            if len(shapes) != 1 or shapes[0][1] != 5120:
                raise ValueError("unexpected R184 hidden size")
            batch = shapes[0][0]
            if batch not in BATCHES or batch in self.table:
                raise ValueError("unexpected/duplicate R184 bucket")
            if value[0] != runner or len(value[1]) != 3:
                raise ValueError("invalid R184 tactic encoding")
            variant, blocks, threads = value[1]
            if not (
                all(type(x) is int for x in value[1])
                and variant in (0, 1) and 0 < blocks <= MAX_BLOCKS
                and 32 <= threads <= 1024 and threads % 32 == 0
            ):
                raise ValueError("unlaunchable R184 tactic")
            # Native binding argument order: blocks, threads, variant.
            self.table[batch] = (blocks, threads, variant)
        if set(self.table) != set(BATCHES):
            raise ValueError("R184 tuning file must contain all eight buckets")

    def resolve(self, rows, hidden):
        if not (0 < rows <= MAX_ROWS and hidden > 0):
            return None
        numel = rows * hidden
        if not (16 <= numel <= MAX_NUMEL and numel % 8 == 0):
            return None
        if hidden != 5120:
            # Upstream TP2 seed, never labelled a measured DFlash config.
            return (16, 128, 0)
        bucket = max(batch for batch in BATCHES if batch <= rows)
        return self.table[bucket]
