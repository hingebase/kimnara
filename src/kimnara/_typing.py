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
    "AtLeast1DT",
    "CustomInliningRule",
    "FastMathOptions",
    "Input",
    "Number",
    "NumberT",
    "NumericT",
    "Output",
    "Outputs",
    "Scalar",
    "ShapeT",
    "UFuncKwargs",
]

import sys
from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    cpu_options,
    ir,
)
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import Any, TypeAliasType, TypedDict, TypeVar

if sys.version_info >= (3, 14):
    _Number = np.number
else:
    _Number = np.number[Any]

# np.ndarray.shape was invariant in numpy<2.1,
# where we must use `AtLeast1DT` instead of `onp.AtLeast1D`
AtLeast1DT = TypeVar("AtLeast1DT", bound=onp.AtLeast1D)

Number = (
    np.int8 | np.int16 | np.int32 | np.int64 | np.intp
    | np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.uintp
    | np.float32 | np.float64 | np.complex64 | np.complex128
)
NumberT = TypeVar(
    "NumberT",
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
)
NumericT = TypeVar("NumericT", bound=_Number | npt.NDArray[_Number])
Scalar = np.bool_ | Number
SCT = TypeVar(
    "SCT",
    np.bool_,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
)
ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])

_SCT = TypeVar("_SCT", bound=np.generic)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])
ArrayLike = TypeAliasType(
    "ArrayLike",
    _SCT | np.ndarray[_ShapeT, np.dtype[_SCT]],
    type_params=(_SCT, _ShapeT),
)
Input = (
    complex | ArrayLike[Scalar] | PlainQuantity[NumberT]
        # The magnitude was invariant before
        # https://github.com/hgrecco/pint/pull/2303
        | PlainQuantity[np.ndarray[ShapeT, np.dtype[NumberT]]]
)
Output = ArrayLike[Scalar] | NumpyQuantity[ArrayLike[Number]]
Outputs = Output | tuple[Output, ...]

CustomInliningRule = Callable[[ir.Expr, Any, Any], bool]
_FastMathFlags = Literal[
    "fast",
    "nnan", "ninf", "nsz", "arcp",
    "contract", "afn", "reassoc",
]
FastMathOptions: TypeAlias = """
    bool
    | set[_FastMathFlags]
    | dict[_FastMathFlags, bool]
    | cpu_options.FastMathOptions"""


class UFuncKwargs(TypedDict, total=False):
    where: onp.ToJustBool | onp.ToJustBoolND | None
    order: Literal["A", "C", "F", "K"]
    subok: bool
