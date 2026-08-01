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

import ctypes
import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pint
import pydantic_core
from typing_extensions import Any, override

import kimnara as kn
from kimnara import _utils
from kimnara._spec._numpy import _units
from kimnara._typing import ArrayLike, NumberT

from . import _accel
from ._converters import PyUFunc_FromFuncAndData

if TYPE_CHECKING:
    import numpy.typing as npt

_DOC = b"Round doubles to integers."
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


@dataclasses.dataclass(slots=True)
class NonMultiplicativeDequantifier(_units.BaseDequantifier[NumberT]):
    inner: _units.NaiveDequantifier[NumberT]
    unit: pint.Unit

    @override
    def dequantify(self, value: object) -> ArrayLike[NumberT]:
        if not _units.is_quantity(value):
            error_type = "is_instance_of"
            message = "Expect quantities rather than plain numbers"
            raise pydantic_core.PydanticCustomError(error_type, message)
        magnitude = value.magnitude
        dtype = np.result_type(magnitude, None).type
        dequantifier = self.inner
        if isinstance(magnitude, np.ndarray):
            fast_path = dequantifier.dtype is dtype
            if fast_path and _utils.at_least_1d(magnitude):
                magnitude = kn.array(
                    magnitude,
                    dtype,
                    align=dequantifier.align,
                    pad_value=dequantifier.pad_value,
                )
            else:
                magnitude = magnitude.astype(dtype, order="C", subok=False)
            quantity = type(value)(magnitude, value.units)
            quantity.ito(self.unit)  # pyright: ignore[reportUnknownMemberType]
        else:
            fast_path = False
            quantity = type(value)(magnitude.astype(dtype), value.units)
            magnitude = quantity.m_as(self.unit)  # pyright: ignore[reportUnknownMemberType]
        func = _func[dtype]
        if fast_path:
            out: npt.NDArray[NumberT] = func(magnitude, magnitude)
        else:
            kwargs = dequantifier.control_output(magnitude)
            out = func(magnitude, **kwargs)
        return dequantifier.postprocess(out)

    @override
    def get_unit(self) -> pint.Unit:
        return self.unit


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
        )
        return NonMultiplicativeDequantifier(dequantifier, self.obj)

    @override
    def has_unit(self) -> bool:
        return True


def _init() -> None:
    for i, code in enumerate("bhlqBHLQfd"):
        dtype = np.dtype(code)
        name = b"round_to_" + dtype.name.encode("ascii")
        types = bytes(map(_utils.num, f"d{code}"))
        _func[dtype.type] = PyUFunc_FromFuncAndData(
            _accel.round_funcs + i * _POINTER_SIZE,
            None,
            types,
            1,
            1,
            1,
            -1,  # None
            name,
            _DOC,
            0,
        )
        _name.append(name)
        _types.append(types)
    for i, code in enumerate("FD", 10):
        dtype = np.dtype(code)
        name = b"round_to_" + dtype.name.encode("ascii")
        types = bytes(map(_utils.num, f"D{code}"))
        _func[dtype.type] = PyUFunc_FromFuncAndData(
            _accel.round_funcs + i * _POINTER_SIZE,
            None,
            types,
            1,
            1,
            1,
            -1,  # None
            name,
            _DOC,
            0,
        )
        _name.append(name)
        _types.append(types)


_func: dict[type[np.number[Any]], np.ufunc] = {}
_name: list[bytes] = []
_types: list[bytes] = []
_init()
