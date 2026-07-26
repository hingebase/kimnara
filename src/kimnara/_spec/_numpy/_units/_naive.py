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

import ctypes

import numpy as np
from typing_extensions import Any, override

import kimnara as kn
from kimnara import _utils
from kimnara._spec._numpy import _units
from kimnara._typing import SCT, ArrayLike

from . import _accel
from ._converters import PyUFunc_FromFuncAndData

_DOC = b"Round doubles to integers."
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


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
