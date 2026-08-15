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
    "EqMixIn",
    "at_least_1d",
    "base_repr",
    "function_base",
    "is_editable",
    "isclass",
    "num",
    "unreachable",
]

import copyreg
import functools
import importlib.metadata
import json
import operator
import sys
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NoReturn, TypeGuard, cast

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy_typing_compat import NUMPY_GE_2_0
from typing_extensions import Any, Protocol, TypeVar

from kimnara._typing import ArrayLike, Scalar

if TYPE_CHECKING:
    from kimnara import _spec

if sys.version_info >= (3, 11):
    from inspect import isclass
else:
    from typing_extensions import TypeIs

    # Keep the return type same as `inspect.isclass`
    def isclass(x: object, /) -> TypeIs[type[object]]:
        # https://github.com/python/cpython/issues/89828
        return not isinstance(x, types.GenericAlias) and isinstance(x, type)

_T = TypeVar("_T", bound=np.generic)


class EqMixIn:
    __slots__ = ()
    _slotnames = classmethod(
        cast(
            "Callable[[type[EqMixIn]], list[str]]",
            copyreg._slotnames,  # pyright: ignore[reportAttributeAccessIssue]  # ruff: ignore[private-member-access]
        ),
    )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        for name in self._slotnames():
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, name) for name in self._slotnames()))

    def __init_subclass__(cls) -> None:
        for base in cls.__mro__:
            if "__dict__" in vars(base):
                unreachable()
        super().__init_subclass__()


class _AtLeast1D(Protocol):
    def __call__(
        self,
        obj: ArrayLike[_T],
        /,
    ) -> TypeGuard[np.ndarray[onp.AtLeast1D, np.dtype[_T]]]: ...


at_least_1d: _AtLeast1D = operator.attrgetter("ndim")
num = {code: np.dtype(code).num for code in (
    "fdFD"  # Floating and complex types
    "bBhHiIqQ"  # Integer types
)}.__getitem__


def base_repr(x: object, /) -> str:
    return type.__repr__(x) if isclass(x) else object.__repr__(x)  # noqa: PLC2801


def calculate_padding(
    shape: Sequence[int],
    dtype: type[Scalar],
    spec: "_spec.Alignment",
) -> int:
    itemsize = np.dtype(dtype).itemsize
    nbytes = shape[-1] * itemsize
    mask = spec.multiple_of - 1
    aligned = (nbytes + mask) & ~mask
    if len(shape) > 1 and aligned % spec.not_multiple_of == 0:
        aligned += spec.multiple_of
    quot, rem = divmod(aligned - nbytes, itemsize)
    if rem:
        message = f"{np.dtype(dtype)!r} is unsupported in Kimnara"
        raise TypeError(message)
    return quot


@functools.lru_cache(maxsize=1)
def is_editable() -> bool:
    try:
        dist = importlib.metadata.distribution("kimnara")
    except ModuleNotFoundError:
        pass
    else:
        if sys.version_info >= (3, 13):
            if origin := dist.origin:
                try:
                    return origin.dir_info.editable
                except AttributeError:
                    return False
        elif text := dist.read_text("direct_url.json"):
            match json.loads(text):
                case {"dir_info": {"editable": True}}:
                    return True
                case _:
                    return False
    return NotImplemented


def unreachable() -> NoReturn:
    message = "Unreachable"
    raise AssertionError(message)


class _FunctionBase(Protocol):
    def _calculate_shapes(
        self,
        broadcast_shape: tuple[int, ...],
        dim_sizes: dict[str, int],
        list_of_core_dims: list[tuple[str, ...]],
    ) -> list[tuple[int, ...]]: ...

    def _create_arrays(
        self,
        broadcast_shape: tuple[int, ...],
        dim_sizes: dict[str, int],
        list_of_core_dims: list[tuple[str, ...]],
        dtypes: Sequence[npt.DTypeLike] | None,
        results: tuple[ArrayLike[np.generic], ...] | None = ...,
    ) -> tuple[npt.NDArray[Any], ...]: ...

    def _parse_input_dimensions(
        self,
        args: tuple[ArrayLike[np.generic], ...],
        input_core_dims: list[tuple[str, ...]],
    ) -> tuple[tuple[int, ...], dict[str, int]]: ...

    def _update_dim_sizes(
        self,
        dim_sizes: dict[str, int],
        arg: ArrayLike[np.generic],
        core_dims: tuple[str, ...],
    ) -> None: ...


function_base: _FunctionBase = getattr(
    np.lib,
    "_function_base_impl" if NUMPY_GE_2_0 else "function_base",
)
