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

__all__ = ["CPointer", "Intrinsic", "Mut", "Pointer"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

import optype.numpy as onp
from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core.cpu import (  # pyright: ignore[reportMissingTypeStubs]
    CPUContext,
)
from numba.core.typing.templates import (  # pyright: ignore[reportMissingTypeStubs]
    Signature,
)
from typing_extensions import Any, Protocol, TypeVar, TypeVarTuple, Unpack

from . import _spec
from ._typing import SCT

if TYPE_CHECKING:
    import numpy as np

_T_co = TypeVar(
    "_T_co",
    bound="np.number[Any] | np.bool_ | CPointer[Any]",
    covariant=True,
)
_Ts = TypeVarTuple("_Ts")


class ArrayCTypes(Protocol):
    @property
    def data(self) -> int: ...


class CPointer(Protocol[_T_co]):
    """Reflects the runtime behavior of `numba.types.CPointer`.

    Unlike real C pointers, this type doesn't support pointer
    arithmetic. Please cast it to integer, and then cast back later when
    you need item access.
    """

    def __getitem__(self, idx: onp.ToJustInt, /) -> _T_co:
        """Load an element from the pointer at specified offset.

        To dereference the pointer, use `pointer[0]`.
        """
        ...

    def __setitem__(self, idx: onp.ToJustInt, val: object, /) -> None:
        """Store an element to the pointer at specified offset."""


Intrinsic = tuple[
    Signature,
    Callable[
        [CPUContext, ir.IRBuilder, Signature, tuple[Unpack[_Ts]]],
        ir.Value | None,
    ],
]

Mut = Annotated[SCT, _spec.Mutable]
Pointer = CPointer[SCT] | ArrayCTypes
