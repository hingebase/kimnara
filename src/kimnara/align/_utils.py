# Copyright 2026 hingebase

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

__all__ = [
    "isaligned",
]

import itertools
from typing import TypeAlias

import numpy as np
import optype.numpy as onp

import kimnara as kn
from kimnara import _spec

_ScalarType: TypeAlias = """
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.intp
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.uintp
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128"""


def isaligned(
    value: np.ndarray[onp.AtLeast1D, np.dtype[_ScalarType]],
    align: "str | kn.Alignment",
) -> bool:
    match kn.Alignment(align):
        case kn.A:
            return True
        case kn.C:
            return value.flags.c_contiguous
        case kn.F:
            return value.flags.f_contiguous
        case align:
            return _isaligned_simd(value, align.value)


def _isaligned_simd(
    value: np.ndarray[onp.AtLeast1D, np.dtype[_ScalarType]],
    spec: _spec.Alignment,
) -> bool:
    if not value.size:
        return True
    multiple_of = spec.multiple_of
    if value.ctypes.data % multiple_of:
        return False
    strides = value.strides
    if strides[-1] != value.itemsize:
        return False
    try:
        leading_dimension = strides[-2]
    except IndexError:  # 1-D array
        # We don't know if the underlying memory was padded properly
        # Assuming false here
        return value.nbytes % multiple_of == 0
    for shape0, (stride0, stride1) in zip(
        reversed(value.shape),
        itertools.pairwise(reversed(strides)),
        strict=False,
    ):
        if stride1 < shape0 * stride0 or stride1 % multiple_of:
            return False
    return bool(leading_dimension % spec.not_multiple_of)
