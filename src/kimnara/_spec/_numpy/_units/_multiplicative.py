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

__all__ = ["MultiplicativeUnit"]

import dataclasses
import math
from typing import cast

import pint
from typing_extensions import override

import kimnara as kn
from kimnara import _spec
from kimnara._spec._numpy import _units
from kimnara._typing import ArrayLike, NumberT


class _BaseMultiplicativeDequantifier(_units.BaseDequantifier[NumberT]):
    __slots__ = ("_inner",)

    def __init__(
        self,
        inner: "_units.NonMultiplicativeDequantifier[NumberT]",
    ) -> None:
        self._inner = inner

    @override
    def dequantify(self, value: object) -> ArrayLike[NumberT]:
        raise NotImplementedError


@dataclasses.dataclass
class MultiplicativeUnit(_units.BaseUnit[NumberT]):
    obj: pint.Unit

    @override
    def dequantifier(
        self,
        align: "kn.Alignment",
        dtype: type[NumberT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = True,
    ) -> _BaseMultiplicativeDequantifier[NumberT]:
        unit: _units.NonMultiplicativeUnit[NumberT] = (
            _units.NonMultiplicativeUnit(self.obj)
        )
        dequantifier = unit.dequantifier(
            align,
            dtype,
            pad_value,
            prefer_scalar=prefer_scalar,
            readonly=readonly,
        )
        if _dimensionless(self.obj):
            return _MultiplicativeDequantifier(dequantifier)
        return _BaseMultiplicativeDequantifier(dequantifier)

    @override
    def has_unit(self) -> bool:
        return True


class _MultiplicativeDequantifier(_BaseMultiplicativeDequantifier[NumberT]):
    __slots__ = ()

    @override
    def dequantify(self, value: object) -> ArrayLike[NumberT]:
        raise NotImplementedError


def _dimensionless(unit: pint.Unit) -> bool:
    try:
        scale = cast("float", unit.m_from(_spec.dimensionless))  # pyright: ignore[reportUnknownMemberType]
    except pint.DimensionalityError:
        return False
    return math.isclose(scale, 1.)
