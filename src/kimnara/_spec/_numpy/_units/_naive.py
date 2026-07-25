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

__all__ = ["NaiveDequantifier", "UnitNaive"]

from typing import TYPE_CHECKING

from typing_extensions import override

from kimnara._spec._numpy import _units
from kimnara._typing import SCT, ArrayLike

if TYPE_CHECKING:
    import kimnara as kn


class NaiveDequantifier(_units.BaseDequantifier[SCT]):
    __slots__ = ()

    @override
    def dequantify(self, value: object) -> ArrayLike[SCT]:
        raise NotImplementedError


class UnitNaive(_units.BaseUnit[SCT]):
    @override
    def dequantifier(
        self,
        align: "kn.Alignment",
        dtype: type[SCT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = True,
        safe: bool = True,
    ) -> NaiveDequantifier[SCT]:
        raise NotImplementedError

    @override
    def has_unit(self) -> bool:
        return False
