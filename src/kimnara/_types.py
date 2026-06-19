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

__all__ = ["AVX", "AVX512", "SSE", "A", "Alignment", "C", "F", "Mut", "Pad"]

import enum
from typing import Annotated

import numpy as np
from optype.typing import AnyComplex
from typing_extensions import TypeVar, final

from . import _spec

_T = TypeVar(
    "_T",
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
Mut = Annotated[_T, _spec.Mutable]


@final
class Alignment(enum.Enum):
    A = "A"
    C = "C"
    F = "F"
    SSE = "SSE"
    AVX = "AVX"
    AVX512 = "AVX512"
    MKL = "MKL"


# We don't expose kn.Alignment.MKL to outer namespace
# since the name `kn.MKL` would be confusing.
A = Alignment.A
C = Alignment.C
F = Alignment.F
SSE = Alignment.SSE
AVX = Alignment.AVX
AVX512 = Alignment.AVX512


class Pad:
    __slots__ = ("value",)

    def __init__(self, value: AnyComplex) -> None:
        self.value = _spec.scalar(value)
