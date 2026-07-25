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

import numpy as np
import numpy.typing as npt
from typing_extensions import Any, TypeVar, override

from kimnara._spec._numpy import _units

if TYPE_CHECKING:
    import kimnara as kn

_SCT = TypeVar("_SCT", bound=np.number[Any] | np.bool_)


class NaiveDequantifier(_units.BaseDequantifier[_SCT]):
    __slots__ = ()

    @override
    def dequantify(self, value: object) -> _SCT | npt.NDArray[_SCT]:
        raise NotImplementedError


class UnitNaive(_units.BaseUnit[_SCT]):
    @override
    def dequantifier(
        self,
        align: "kn.Alignment",
        dtype: type[_SCT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = True,
        safe: bool = True,
    ) -> NaiveDequantifier[_SCT]:
        raise NotImplementedError

    @override
    def has_unit(self) -> bool:
        return False
