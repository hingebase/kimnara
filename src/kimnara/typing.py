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

"""Types useful in annotations."""

__all__ = ["Intrinsic", "Mut"]

from collections.abc import Callable
from typing import Annotated

import numpy as np
from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core import cpu, typing  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypeVar, TypeVarTuple, Unpack

from . import _spec

_T = TypeVar(
    "_T",
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
_Ts = TypeVarTuple("_Ts")


Intrinsic = tuple[
    typing.Signature,
    Callable[
        [cpu.CPUContext, ir.IRBuilder, typing.Signature, tuple[Unpack[_Ts]]],
        ir.Value | None,
    ],
]

Mut = Annotated[_T, _spec.Mutable]
