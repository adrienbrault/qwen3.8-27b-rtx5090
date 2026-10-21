#!/usr/bin/env python3
"""CPU reproduction of the real DeviceInfo introspection/copy/compile seam.

Only the CUDA metadata extension is replaced; Python DeviceInfo and Dynamo are
the installed implementations. The serving boot remains the GPU integration test.
"""
import copy
from types import SimpleNamespace

import torch
import humming.device as device


def main():
    class Metadata:
        def __init__(self, index):
            self.index = index
            self.sm_count = 170

    device._get_device_info_extension = lambda: SimpleNamespace(_DeviceInfo=Metadata)
    if hasattr(device.DeviceInfo, "_instances"):
        device.DeviceInfo._instances.clear()
    value = device.DeviceInfo(0)
    # Dynamo's speculative object handling can inspect a not-yet-initialized
    # slotted object; the old __getattr__ recurses on its missing _value slot.
    uninitialized = object.__new__(device.DeviceInfo)
    assert not hasattr(uninitialized, "__dict__")
    assert copy.copy(value) is value
    assert copy.deepcopy(value) is value

    def forward(x):
        return x + device.DeviceInfo(0).sm_count

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    assert torch.equal(compiled(torch.ones(3)), torch.full((3,), 171.0))
    print("DEVICEINFO PASS: missing-slot introspection, copy, deepcopy, fullgraph Dynamo")


if __name__ == "__main__":
    main()
