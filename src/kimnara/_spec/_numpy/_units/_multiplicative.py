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

import ctypes
import dataclasses
import math
from typing import cast

import numpy as np
import optype as op
import pint
from typing_extensions import Any, override

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._spec._numpy import _units
from kimnara._typing import ArrayLike, NumberT

from . import _accel
from ._converters import PyUFunc_FromFuncAndData

_DOC = b"Scale in double precision."
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


class _MultiplicativeDequantifier(_units.BaseDequantifier[NumberT]):
    __slots__ = ("_inner", "_types")

    def __init__(
        self,
        inner: "_units.NonMultiplicativeDequantifier[NumberT]",
        *args: type[Any],
    ) -> None:
        self._inner = inner
        self._types = args

    @override
    def dequantify(self, value: object) -> ArrayLike[NumberT]:
        if isinstance(value, self._types):
            return self._inner.inner.dequantify(value)
        if not _units.is_quantity(value):
            message = "Expect quantities rather than plain numbers"
            raise TypeError(message)
        if not value._is_multiplicative:  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
            return self._inner.dequantify(value)
        dequantifier = self._inner.inner
        base = value.magnitude
        scale = cast("float", self._inner.unit.m_from(value.units))  # pyright: ignore[reportUnknownMemberType]
        if not math.isclose(scale, 1.) or not dequantifier.dtype_match(base):
            func = _func[dequantifier.dtype]
            kwargs = dequantifier.control_output(base)
            out = func(base, scale, **kwargs)
        elif _utils.at_least_1d(base):
            out = kn.asarray(
                base,
                align=dequantifier.align,
                pad_value=dequantifier.pad_value,
            )
        elif isinstance(base, np.ndarray):
            out = np.require(base, requirements=("A", "E"))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        else:
            out = base
        return dequantifier.postprocess(out, base)  # pyright: ignore[reportUnknownArgumentType]


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
    ) -> _MultiplicativeDequantifier[NumberT]:
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
            return _MultiplicativeDequantifier(
                dequantifier,
                op.JustInt,
                op.JustFloat,
                op.JustComplex,
                np.number,
                np.ndarray,
            )
        return _MultiplicativeDequantifier(dequantifier)

    @override
    def has_unit(self) -> bool:
        return True


def _dimensionless(unit: pint.Unit) -> bool:
    try:
        scale = cast("float", unit.m_from(_spec.dimensionless))  # pyright: ignore[reportUnknownMemberType]
    except pint.DimensionalityError:
        return False
    return math.isclose(scale, 1.)


def _init() -> None:
    ntypes = 10
    for i, code in enumerate("bhlqBHLQfd"):
        dtype = np.dtype(code)
        name = b"scale_to_" + dtype.name.encode("ascii")
        types = bytes(
            map(_utils.num, f"{f'd{code}'.join('bhlqBHLQfd')}d{code}"),
        )
        _func[dtype.type] = PyUFunc_FromFuncAndData(
            _accel.scale_funcs + ntypes * i * _POINTER_SIZE,
            None,
            types,
            ntypes,
            2,
            1,
            -1,  # None
            name,
            _DOC,
            0,
        )
        _name.append(name)
        _types.append(types)
    ntypes = 12
    for i, code in enumerate("FD"):
        dtype = np.dtype(code)
        name = b"scale_to_" + dtype.name.encode("ascii")
        types = bytes(
            map(_utils.num, f"{f'd{code}'.join('bhlqBHLQfdFD')}d{code}"),
        )
        _func[dtype.type] = PyUFunc_FromFuncAndData(
            _accel.scale_funcs + (100 + ntypes * i) * _POINTER_SIZE,
            None,
            types,
            ntypes,
            2,
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
