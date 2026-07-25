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

__all__ = ["Alignment", "Mutable", "Type", "dimensionless", "scalar", "ureg"]

import abc
import contextlib
import dataclasses
import math
import operator
import sys
import types
import warnings
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
    override,
)
from typing_inspection import introspection

from kimnara import _spec, _utils

if TYPE_CHECKING:
    import pint
    from pint.facets.plain import PlainQuantity

if sys.version_info >= (3, 11):
    _Complex = SupportsComplex
else:
    from exceptiongroup import ExceptionGroup

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
    catch = _Catch([])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if isinstance(value, SupportsIndex):
            with catch:
                return operator.index(value)
        if isinstance(value, SupportsFloat):
            with catch:
                return float(value)
        if isinstance(value, _Complex):
            with catch:
                return complex(value)
    message = f"Cannot convert {_utils.base_repr(value)} to a numeric scalar"
    if excs := catch.caught:
        raise ExceptionGroup(message, excs)
    raise TypeError(message)


@dataclasses.dataclass
class _Catch(contextlib.AbstractContextManager["_Catch", bool]):
    caught: list[Exception]

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
        /,
    ) -> bool:
        if suppress := isinstance(exc_value, Exception):
            self.caught.append(exc_value)
        return suppress


ureg = cast(
    "pint.registry.GenericUnitRegistry[PlainQuantity[Any], pint.Unit]",
    cast("pint.registry.ApplicationRegistry", metpy.units.units).get(),
)
dimensionless = ureg.dimensionless
