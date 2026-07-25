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

__all__ = ["NonMultiplicativeDequantifier", "NonMultiplicativeUnit"]

import dataclasses
from typing import TYPE_CHECKING

import pint
from typing_extensions import override

from kimnara._spec._numpy import _units
from kimnara._typing import ArrayLike, NumberT

if TYPE_CHECKING:
    import kimnara as kn


@dataclasses.dataclass(slots=True)
class NonMultiplicativeDequantifier(_units.BaseDequantifier[NumberT]):
    inner: _units.NaiveDequantifier[NumberT]
    unit: pint.Unit

    @override
    def dequantify(self, value: object) -> ArrayLike[NumberT]:
        raise NotImplementedError


@dataclasses.dataclass
class NonMultiplicativeUnit(_units.BaseUnit[NumberT]):
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
    ) -> NonMultiplicativeDequantifier[NumberT]:
        unit: _units.UnitNaive[NumberT] = _units.UnitNaive()
        dequantifier = unit.dequantifier(
            align,
            dtype,
            pad_value,
            prefer_scalar=prefer_scalar,
            readonly=readonly,
            safe=False,
        )
        return NonMultiplicativeDequantifier(dequantifier, self.obj)

    @override
    def has_unit(self) -> bool:
        return True
