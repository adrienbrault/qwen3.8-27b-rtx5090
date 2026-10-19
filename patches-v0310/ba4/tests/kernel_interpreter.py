"""Small stdlib interpreter for executing the actual Triton kernel bodies.

Checks indexing, masks, scratch coverage and shard boundaries. It is not a
Triton compiler or a model of GPU reduction ordering/performance.
"""

import ast
import itertools
import math
from pathlib import Path
import struct
import types


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


class Vec:
    def __init__(self, shape, values):
        self.shape, self.values = tuple(shape), list(values)
        assert math.prod(self.shape) == len(self.values)

    def at(self, coords):
        coords = coords[len(coords) - len(self.shape):] if self.shape else ()
        index = 0
        for size, coordinate in zip(self.shape, coords):
            index = index * size + (coordinate if size > 1 else 0)
        return self.values[index]

    def op(self, other, fn):
        other = vector(other)
        shape = broadcast(self.shape, other.shape)
        return Vec(shape, [fn(self.at(c), other.at(c)) for c in coordinates(shape)])

    def __add__(self, other):
        return self.op(other, lambda a, b: a + b)

    __radd__ = __add__

    def __mul__(self, other):
        return self.op(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __lt__(self, other):
        return self.op(other, lambda a, b: a < b)

    def __and__(self, other):
        return self.op(other, lambda a, b: a and b)

    def __getitem__(self, key):
        assert len(self.shape) == 1
        if key == (slice(None), None):
            return Vec((self.shape[0], 1), self.values)
        if key == (None, slice(None)):
            return Vec((1, self.shape[0]), self.values)
        raise AssertionError(key)

    def to(self, dtype):
        return Vec(self.shape, [dtype(v) for v in self.values])


def vector(value):
    return value if isinstance(value, Vec) else Vec((), [value])


def broadcast(a, b):
    rank = max(len(a), len(b))
    a, b = (1,) * (rank - len(a)) + a, (1,) * (rank - len(b)) + b
    assert all(x == y or x == 1 or y == 1 for x, y in zip(a, b))
    return tuple(max(x, y) for x, y in zip(a, b))


def coordinates(shape):
    return itertools.product(*(range(s) for s in shape))


class Pointer:
    def __init__(self, values, cast=float, index=0):
        self.values, self.cast, self.index = values, cast, vector(index)

    def __add__(self, index):
        return Pointer(self.values, self.cast, self.index + index)


class TL:
    constexpr = object
    float32 = float
    int32 = int

    def __init__(self):
        self.ids = (0, 0, 0)

    def program_id(self, axis):
        return self.ids[axis]

    def arange(self, start, stop):
        return Vec((stop - start,), range(start, stop))

    def load(self, pointer, mask=True, other=0, **kwargs):
        mask = vector(mask)
        shape = broadcast(pointer.index.shape, mask.shape)
        result = []
        for c in coordinates(shape):
            if mask.at(c):
                i = pointer.index.at(c)
                assert 0 <= i < len(pointer.values), (i, len(pointer.values))
                result.append(pointer.values[i])
            else:
                result.append(other)
        return Vec(shape, result)

    def store(self, pointer, values, mask=True):
        values, mask = vector(values), vector(mask)
        shape = broadcast(broadcast(pointer.index.shape, values.shape), mask.shape)
        for c in coordinates(shape):
            if mask.at(c):
                i = pointer.index.at(c)
                assert 0 <= i < len(pointer.values), (i, len(pointer.values))
                pointer.values[i] = pointer.cast(values.at(c))

    def sum(self, value, axis):
        shape = value.shape[:axis] + value.shape[axis + 1:]
        return Vec(shape, [math.fsum(value.at(c[:axis] + (k,) + c[axis:])
                          for k in range(value.shape[axis])) for c in coordinates(shape)])


def load_kernel_bodies(path):
    tl = TL()
    ns = {"tl": tl}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_direct", "_partial", "_finish", "_stream_weights"):
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), node.name, "exec"), ns)
    return tl, types.SimpleNamespace(**{k: v for k, v in ns.items() if k.startswith("_")})
