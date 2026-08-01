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
    "SCT",
    "ArrayLike",
    "Inexact",
    "Number",
    "NumberT",
    "NumericT",
    "Scalar",
]

import sys
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import Any, TypeVar

if sys.version_info >= (3, 14):
    _Number = np.number
else:
    _Number = np.number[Any]

Inexact = np.float32 | np.float64 | np.complex64 | np.complex128
Number: TypeAlias = """
    np.int8 | np.int16 | np.int32 | np.int64 | np.intp
    | np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.uintp
    | Inexact"""
NumberT = TypeVar(
    "NumberT",
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
)
NumericT = TypeVar("NumericT", bound=_Number | npt.NDArray[_Number])
Scalar: TypeAlias = "np.bool_ | Number"
SCT = TypeVar(
    "SCT",
    np.bool_,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
)

_SCT = TypeVar("_SCT", bound=np.generic, default=_Number | np.bool_)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])
ArrayLike = _SCT | np.ndarray[_ShapeT, np.dtype[_SCT]]
