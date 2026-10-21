# SPDX-License-Identifier: Apache-2.0
"""TP2 ownership/preparation adapter for FlashInfer df8b5c1's unchanged kernel.

Control traffic uses the existing TP Gloo group, never a new WORLD group.
One CUDA stream at a time; the communicator scopes capture stream transfers.
No autotuner, topology subprocess, FlashInfer import, or file writes at boot.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from .build import load_native
from .fixed_config import FixedConfigs, MAX_NUMEL, MAX_ROWS


class JointFailure(RuntimeError):
    """All live ranks have observed the same local-stage failure."""


def exchange(group, value):
    # vLLM custom_all_reduce.py:389 documents Gloo object all-gather's
    # inference-tensor restriction. Allocate its internal CPU buffers with
    # inference mode explicitly disabled, including capture setup.
    with torch.inference_mode(False):
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, value, group=group)
    return gathered


def stage(group, label, action):
    value = None
    error = None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = exchange(group, error)
    if any(item is not None for item in errors):
        raise JointFailure(f"{label}: {errors}")
    return value


class PcieIpcAllReduceWorkspace:
    def __init__(self, group, device):
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        assert self.world_size == 2
        assert dist.get_backend(group) == "gloo"
        self.max_numel = MAX_NUMEL
        self._local = self._peer = self._handle = 0
        self._resolved = {}
        self._stream = None
        self._capture_depth = 0
        self._closed = False
        self.configs = stage(group, "loading fixed configs", FixedConfigs)
        digests = exchange(group, self.configs.digest)
        if len(set(digests)) != 1:
            raise JointFailure("ranks loaded different R184 tuning files")
        self.native = stage(group, "loading native module", load_native)
        try:
            # Each stage contains only local operations before its verdict.
            # No rank opens IPC after a peer failed allocation.
            def allocate():
                self._local, encoded = self.native.allocate(MAX_NUMEL, self.device.index)
                return encoded

            encoded = stage(group, "allocating slab", allocate)
            handles = exchange(group, encoded)

            def open_peer():
                self._peer = self.native.open_ipc(handles[1 - self.rank], self.device.index)

            stage(group, "opening peer slab", open_peer)

            def bind():
                ptrs = [self._local, self._peer]
                if self.rank == 1:
                    ptrs.reverse()
                self._handle = self.native.init(ptrs, self.rank, MAX_NUMEL, self.device.index)

            stage(group, "binding workspace", bind)
        except JointFailure:
            # Everyone enters cleanup, including ranks with no allocation.
            self.destroy()
            raise

    def prepare(self, shapes, dtype=torch.bfloat16):
        """Collective; resolve metadata only, before any real CUDA capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare must run before CUDA capture")

        def resolve():
            ordered = tuple(sorted(set((int(r), int(h)) for r, h in shapes)))
            return tuple(
                (shape, self.configs.resolve(*shape) if dtype == torch.bfloat16 else None)
                for shape in ordered
            )

        configs = stage(self.group, "resolving shape metadata", resolve)
        proposals = exchange(self.group, (str(dtype), configs))
        if any(proposal != proposals[0] for proposal in proposals):
            raise JointFailure("ranks requested different prepare shapes/configs")
        added = {}
        for shape, config in configs:
            key = (*shape, dtype)
            if config is not None and key not in self._resolved:
                self._resolved[key] = config
                added[shape] = config
        return added

    def supports(self, inp):
        # Align unusual offset views inside the native binding. A local
        # pointer address must not select a different collective on one rank.
        return (
            not self._closed and bool(self._handle)
            and inp.device == self.device and inp.dtype == torch.bfloat16
            and inp.dim() == 2 and inp.is_contiguous()
            and 0 < inp.shape[0] <= MAX_ROWS
            and 16 <= inp.numel() <= self.max_numel and inp.numel() % 8 == 0
        )

    def is_resolved(self, inp):
        return (*inp.shape, inp.dtype) in self._resolved

    def rebind_stream(self):
        """Caller must synchronize first; equivalent to main's rebind_stream."""
        self._stream = None

    def _check_stream(self):
        if torch.cuda.is_current_stream_capturing():
            return
        current = torch.cuda.current_stream(self.device)
        if self._stream is None:
            self._stream = current
        elif current != self._stream:
            torch.cuda.synchronize(self.device)
            self.rebind_stream()
            self._stream = current

    @contextmanager
    def capture(self):
        """Transfer the workspace for capture warmups and restore replay ordering.

        Entered inside torch.cuda.stream(capture_stream), before cuda.graph.
        Synchronize BOTH boundaries: replay bypasses Python all_reduce.
        """
        if self._capture_depth:
            self._capture_depth += 1
            try:
                yield
            finally:
                self._capture_depth -= 1
            return
        previous = self._stream
        torch.cuda.synchronize(self.device)
        self.rebind_stream()
        self._stream = torch.cuda.current_stream(self.device)
        self._capture_depth = 1
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            self.rebind_stream()
            self._stream = previous
            self._capture_depth = 0

    def all_reduce(self, inp):
        key = (*inp.shape, inp.dtype)
        if key not in self._resolved:
            self.prepare([tuple(inp.shape)], inp.dtype)
        self._check_stream()
        blocks, threads, variant = self._resolved[key]
        return self.native.all_reduce(self._handle, inp, blocks, threads, variant)

    def destroy(self):
        """Collective, before process-group destruction; no atexit barriers."""
        if self._closed:
            return
        stage(self.group, "draining workspace", lambda: torch.cuda.synchronize(self.device))

        def unmap():
            if self._handle:
                self.native.dispose(self._handle)
                self._handle = 0
            if self._peer:
                self.native.close_ipc(self._peer, self.device.index)
                self._peer = 0

        # Also a barrier between peer unmap and local free.
        stage(self.group, "unmapping peer slab", unmap)

        def free():
            if self._local:
                self.native.free_local(self._local, self.device.index)
                self._local = 0

        stage(self.group, "freeing local slab", free)
        self._closed = True
        self._resolved.clear()
