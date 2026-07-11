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

__all__ = ["Alignment", "Mutable", "Type", "scalar", "ureg"]

import abc
import contextlib
import math
import operator
import sys
from typing import TYPE_CHECKING, cast

import metpy.units  # pyright: ignore[reportMissingTypeStubs]
import numba.core.types  # pyright: ignore[reportMissingTypeStubs]
from optype.typing import AnyComplex
from typing_extensions import (
    Any,
    NamedTuple,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    TypeForm,
    final,
)
from typing_inspection import introspection

from kimnara import _spec, _utils

if TYPE_CHECKING:
    import pint
    from pint.facets.plain import PlainQuantity

if sys.version_info >= (3, 11):
    _Complex = SupportsComplex
else:
    _Complex = SupportsComplex | complex


@final
class Alignment(NamedTuple):
    name: str
    multiple_of: int = 1
    not_multiple_of: float = math.nan


@final
class Mutable:
    pass


class Type(abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: "_spec.TypingContext",
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def to_numba(self) -> numba.core.types.Type:
        raise NotImplementedError

    @abc.abstractmethod
    def to_python(self) -> TypeForm[Any]:
        raise NotImplementedError


def scalar(value: AnyComplex) -> complex:
    if isinstance(value, SupportsIndex):
        with _suppress:
            return operator.index(value)
    if isinstance(value, SupportsFloat):
        with _suppress:
            return float(value)
    if isinstance(value, _Complex):
        with _suppress:
            return complex(value)
    message = f"Cannot convert {_utils.base_repr(value)} to a numeric scalar"
    raise TypeError(message)


ureg = cast(
    "pint.registry.GenericUnitRegistry[PlainQuantity[Any], pint.Unit]",
    cast("pint.registry.ApplicationRegistry", metpy.units.units).get(),
)
_suppress = contextlib.suppress(Exception)
