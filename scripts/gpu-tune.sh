#!/usr/bin/env bash
# GPU tuning for the dual-5090 box: persistence + per-GPU power caps + mem OC on BOTH cards.
# History: the original (never in repo — fixed 2026-08-31) was GPU0-only and used the legacy
# nvmlDeviceSetMemClkVfOffset, which driver 610 accepts WITHOUT EFFECT (offset reads back 0;
# unit reported success while every R130-R136 number ran at stock memclk). This version:
#   - applies to every GPU
#   - power: GPU0 600W, GPU1 575W (its vendor max)
#   - tries the legacy VfOffset API first, VERIFIES by readback, falls back to the modern
#     nvmlDeviceSetClockOffsets struct API, and verifies again; exits nonzero if neither sticks.
# Deploy: /usr/local/bin/gpu-tune.sh, invoked by gpu-tune.service at boot.
set -uo pipefail
nvidia-smi -pm 1 >/dev/null
nvidia-smi -i 0 -pl 600 >/dev/null 2>&1
nvidia-smi -i 1 -pl 575 >/dev/null 2>&1
nvidia-smi -rgc >/dev/null 2>&1 || true   # no leftover core-clock lock (factory boost)

python3 - <<'PY'
import sys
import pynvml as N
N.nvmlInit()
TARGET = 4500
ok = True
for i in range(N.nvmlDeviceGetCount()):
    h = N.nvmlDeviceGetHandleByIndex(i)
    applied = None
    # attempt 1: legacy VfOffset API
    try:
        N.nvmlDeviceSetMemClkVfOffset(h, TARGET)
        if N.nvmlDeviceGetMemClkVfOffset(h) == TARGET:
            applied = "legacy VfOffset"
    except Exception:
        pass
    # attempt 2: modern ClockOffsets struct API (driver 610 path)
    if applied is None:
        try:
            info = N.c_nvmlClockOffset_v1_t()
            info.version = N.nvmlClockOffset_v1
            info.type = N.NVML_CLOCK_MEM
            info.pstate = N.NVML_PSTATE_0
            info.clockOffsetMHz = TARGET
            N.nvmlDeviceSetClockOffsets(h, info)
            chk = N.c_nvmlClockOffset_v1_t()
            chk.version = N.nvmlClockOffset_v1
            chk.type = N.NVML_CLOCK_MEM
            chk.pstate = N.NVML_PSTATE_0
            N.nvmlDeviceGetClockOffsets(h, chk)
            if chk.clockOffsetMHz == TARGET:
                applied = "ClockOffsets struct"
        except Exception as e:
            print(f"GPU{i}: ClockOffsets API failed: {e}")
    if applied:
        print(f"GPU{i}: mem offset +{TARGET} applied via {applied}")
    else:
        print(f"GPU{i}: MEM OC FAILED TO STICK (both APIs)")
        ok = False
sys.exit(0 if ok else 1)
PY
