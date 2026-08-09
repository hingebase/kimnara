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
    "is_editable",
    "isclass",
    "num",
    "unreachable",
]

import functools
import importlib.metadata
import json
import operator
import sys
import types
from collections.abc import Sequence
from typing import NoReturn, TypeGuard

import numpy as np
import optype.numpy as onp
from typing_extensions import Protocol, TypeVar

from kimnara import _spec
from kimnara._typing import ArrayLike, Scalar

if sys.version_info >= (3, 11):
    from inspect import isclass
else:
    from typing_extensions import TypeIs

    # Keep the return type same as `inspect.isclass`
    def isclass(x: object, /) -> TypeIs[type[object]]:
        # https://github.com/python/cpython/issues/89828
        return not isinstance(x, types.GenericAlias) and isinstance(x, type)

_T = TypeVar("_T", bound=np.generic)


class EqMixIn:  # ruff: ignore[eq-without-hash]
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True


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
    spec: _spec.Alignment,
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
