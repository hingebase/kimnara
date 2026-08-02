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

__all__ = ["PyUFunc_FromFuncAndData"]

import collections
import ctypes
import dataclasses
import decimal
import fractions
import math
import operator
import threading
from collections.abc import Callable
from typing import TypeGuard

import llvmlite.binding  # pyright: ignore[reportMissingTypeStubs]
import numba.core.ccallback  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
from numpy_typing_compat import NUMPY_GE_2_0
from pint.facets.nonmultiplicative import definitions
from pint.facets.plain import UnitDefinition
from typing_extensions import Any, Never, override

import kimnara as kn
from kimnara import _spec, _utils

_DecimalLike = decimal.Decimal | fractions.Fraction

_DOC = b"Simplified FMA ufunc where `x2` and `x3` must be scalars."
_NAME = b"fma"
_NANF = np.uint32(0x7ff80000)
_NANF_2 = np.uint64(0x7ff800007ff80000)
_TYPES = bytes(map(_utils.num, "ffffddddFffFDddD"))


class _LogarithmicConverter(definitions.LogarithmicConverter):
    @override
    def from_reference(self, value: Any, inplace: bool = False) -> Any:
        scale = self.logfactor / math.log(self.logbase)
        offset = scale * -math.log(self.scale)
        if inplace:
            np.log(value, value)
            _fma(value, scale, offset, value)
        else:
            value = np.log(value)
            if _is_ndarray(value):
                _fma(value, scale, offset, value)
            else:
                value = _fma(value, scale, offset)
        return value

    @override
    def to_reference(self, value: Any, inplace: bool = False) -> Any:
        scale = math.log(self.logbase) / self.logfactor
        offset = math.log(self.scale)
        if inplace:
            _fma(value, scale, offset, value)
            np.exp(value, value)
        else:
            value = _fma(value, scale, offset)
            if _is_ndarray(value):
                np.exp(value, value)
            else:
                value = np.exp(value)
        return value


class _OffsetConverter(definitions.OffsetConverter):
    @override
    def from_reference(self, value: object, inplace: bool = False) -> Any:
        if isinstance(value, _DecimalLike):
            # https://github.com/hgrecco/pint/pull/2318
            return super().from_reference(value, inplace)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        scale = 1. / self.scale
        offset = -self.offset / self.scale
        if inplace:
            _fma(value, scale, offset, value)
        else:
            value = _fma(value, scale, offset)
        return value

    @override
    def to_reference(self, value: object, inplace: bool = False) -> Any:
        if isinstance(value, _DecimalLike):
            return super().to_reference(value, inplace)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if inplace:
            _fma(value, self.scale, self.offset, value)
        else:
            value = _fma(value, self.scale, self.offset)
        return value


def _fma(x1: object, x2: object, x3: object, /, out: object = None) -> object:
    global _fma  # ruff: ignore[global-statement]
    with _lock:
        if not _cfuncs:
            compiler = numba.cfunc(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                f"void({ 'CPointer(intp), ' * 3 }voidptr)",
                cache=_utils.is_editable() is False,
            )
            try:
                for i, impl in enumerate([_sfma, _dfma, _csfma, _zdfma]):
                    cfunc = compiler(impl)
                    _cfuncs.append(cfunc)
                    _func[i] = cfunc.address  # pyright: ignore[reportUnknownMemberType]
                _fma = PyUFunc_FromFuncAndData(
                    ctypes.byref(_func),
                    None,
                    _TYPES,
                    4,
                    3,
                    1,
                    -1,  # None
                    _NAME,
                    _DOC,
                    0,
                )
            except:
                _cfuncs.clear()
                raise
    return _fma(x1, x2, x3, out)


def _import_umath() -> Callable[
    [object, None, bytes, int, int, int, int, bytes, bytes, int],
    np.ufunc,
]:
    ufunc_api = operator.attrgetter(
        "_core._multiarray_umath._UFUNC_API" if NUMPY_GE_2_0 else "_UFUNC_API",
    )
    # ruff: disable[non-lowercase-variable-in-function]
    PyCapsule_GetPointer = ctypes.PYFUNCTYPE(
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.py_object,
        ctypes.c_char_p,
    )(("PyCapsule_GetPointer", ctypes.pythonapi))
    PyUFunc_API = PyCapsule_GetPointer(ufunc_api(np), None)
    # ruff: enable[non-lowercase-variable-in-function]
    return ctypes.PYFUNCTYPE(
        ctypes.py_object,
        ctypes.c_void_p,  # func
        ctypes.c_void_p,  # data
        ctypes.c_char_p,  # types
        ctypes.c_int,  # ntypes
        ctypes.c_int,  # nin
        ctypes.c_int,  # nout
        ctypes.c_int,  # identity
        ctypes.c_char_p,  # name
        ctypes.c_char_p,  # doc
        ctypes.c_int,  # unused
    )(PyUFunc_API[1])


def _is_ndarray(value: object) -> TypeGuard[npt.NDArray[np.inexact[Any]]]:
    return isinstance(value, np.ndarray)


def _update(definition: UnitDefinition) -> UnitDefinition:
    match cvt := definition.converter:
        case definitions.LogarithmicConverter():
            return dataclasses.replace(
                definition,
                converter=_LogarithmicConverter(
                    scale=cvt.scale,
                    logbase=cvt.logbase,
                    logfactor=cvt.logfactor,
                ),
            )
        case definitions.OffsetConverter():
            return dataclasses.replace(
                definition,
                converter=_OffsetConverter(scale=cvt.scale, offset=cvt.offset),
            )
        case _:
            return definition


def _sfma(
    args: "kn.typing.CPointer[np.intp]",
    dimensions: "kn.typing.CPointer[np.intp]",
    steps: "kn.typing.CPointer[np.intp]",
    _data: "kn.typing.CPointer[Never]",
) -> None:
    out = args[3]
    out_step = steps[3]
    if steps[1] | steps[2]:
        kn.ops.invalid()
        for _ in range(dimensions[0]):
            kn.ops.cast("kn.typing.CPointer[np.uint32]", out)[0] = _NANF
            out += out_step
        return
    in_step = steps[0]

    base = args[0]
    scale = kn.ops.cast("kn.typing.CPointer[np.float32]", args[1])[0]
    offset = kn.ops.cast("kn.typing.CPointer[np.float32]", args[2])[0]
    for _ in range(dimensions[0]):
        kn.ops.cast("kn.typing.CPointer[np.float32]", out)[0] = kn.ops.fma(
            kn.ops.cast("kn.typing.CPointer[np.float32]", base)[0],
            scale,
            offset,
        )
        base += in_step
        out += out_step


def _dfma(
    args: "kn.typing.CPointer[np.intp]",
    dimensions: "kn.typing.CPointer[np.intp]",
    steps: "kn.typing.CPointer[np.intp]",
    _data: "kn.typing.CPointer[Never]",
) -> None:
    out = args[3]
    out_step = steps[3]
    if steps[1] | steps[2]:
        kn.ops.invalid()
        for _ in range(dimensions[0]):
            kn.ops.cast("kn.typing.CPointer[np.float64]", out)[0] = math.nan
            out += out_step
        return
    in_step = steps[0]

    base = args[0]
    scale = kn.ops.cast("kn.typing.CPointer[np.float64]", args[1])[0]
    offset = kn.ops.cast("kn.typing.CPointer[np.float64]", args[2])[0]
    for _ in range(dimensions[0]):
        kn.ops.cast("kn.typing.CPointer[np.float64]", out)[0] = kn.ops.fma(
            kn.ops.cast("kn.typing.CPointer[np.float64]", base)[0],
            scale,
            offset,
        )
        base += in_step
        out += out_step


def _csfma(
    args: "kn.typing.CPointer[np.intp]",
    dimensions: "kn.typing.CPointer[np.intp]",
    steps: "kn.typing.CPointer[np.intp]",
    _data: "kn.typing.CPointer[Never]",
) -> None:
    out = args[3]
    out_step = steps[3]
    if steps[1] | steps[2]:
        kn.ops.invalid()
        for _ in range(dimensions[0]):
            kn.ops.cast("kn.typing.CPointer[np.uint64]", out)[0] = _NANF_2
            out += out_step
        return
    in_step = steps[0]

    base = args[0]
    scale = kn.ops.cast("kn.typing.CPointer[np.float32]", args[1])[0]
    offset = kn.ops.cast("kn.typing.CPointer[np.float32]", args[2])[0]
    for _ in range(dimensions[0]):
        kn.ops.cast("kn.typing.CPointer[np.complex64]", out)[0] = kn.ops.fma(
            kn.ops.cast("kn.typing.CPointer[np.complex64]", base)[0],
            scale,
            offset,
        )
        base += in_step
        out += out_step


def _zdfma(
    args: "kn.typing.CPointer[np.intp]",
    dimensions: "kn.typing.CPointer[np.intp]",
    steps: "kn.typing.CPointer[np.intp]",
    _data: "kn.typing.CPointer[Never]",
) -> None:
    out = args[3]
    out_step = steps[3]
    if steps[1] | steps[2]:
        kn.ops.invalid()
        for _ in range(dimensions[0]):
            ptr = kn.ops.cast("kn.typing.CPointer[np.float64]", out)
            ptr[0] = ptr[1] = math.nan
            out += out_step
        return
    in_step = steps[0]

    base = args[0]
    scale = kn.ops.cast("kn.typing.CPointer[np.float64]", args[1])[0]
    offset = kn.ops.cast("kn.typing.CPointer[np.float64]", args[2])[0]
    for _ in range(dimensions[0]):
        kn.ops.cast("kn.typing.CPointer[np.complex128]", out)[0] = kn.ops.fma(
            kn.ops.cast("kn.typing.CPointer[np.complex128]", base)[0],
            scale,
            offset,
        )
        base += in_step
        out += out_step


PyUFunc_FromFuncAndData = _import_umath()
_cfuncs: list[numba.core.ccallback.CFunc] = []
_func = (ctypes.c_void_p * 4)()
_lock = threading.Lock()
if llvmlite.binding.get_host_cpu_features().get("fma"):  # pyright: ignore[reportUnknownMemberType]
    # ruff: disable[private-member-access]
    _spec.ureg._units = collections.ChainMap({  # pyright: ignore[reportPrivateUsage]
        k: _update(v) for k, v in _spec.ureg._units.items()  # pyright: ignore[reportPrivateUsage]
    })
    # ruff: enable[private-member-access]
