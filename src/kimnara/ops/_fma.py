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

__all__ = ["fma"]

import sys
from collections.abc import Callable
from ctypes import CFUNCTYPE, c_double, c_float
from typing import TYPE_CHECKING, cast, no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
import optype.numpy as onp
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    cgutils,
    cpu,
    types,
    typing,
)
from numpy import complex64, float32
from typing_extensions import TypeVar, overload

from . import _lib

if TYPE_CHECKING:
    from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]

    import kimnara as kn

if sys.version_info >= (3, 13):
    from math import fma as _fma
else:
    _fma = CFUNCTYPE(c_double, c_double, c_double, c_double)(("fma", _lib.c))

_Op = complex | float32 | complex64
_T = TypeVar("_T", float, float32)


@overload
def fma(a: _T, b: _T, c: _T, /) -> _T: ...
@overload
def fma(a: float32, b: float32, c: complex64, /) -> complex64: ...
@overload
def fma(a: complex64, b: float32, c: complex64 | float32, /) -> complex64: ...
@overload
def fma(a: float32, b: complex64, c: complex64 | float32, /) -> complex64: ...
@overload
def fma(a: float, b: float, c: onp.ToJustComplex128, /) -> complex: ...
@overload
def fma(a: onp.ToJustComplex128, b: float, c: complex, /) -> complex: ...
@overload
def fma(a: float, b: onp.ToJustComplex128, c: complex, /) -> complex: ...
@no_type_check
def fma(  # ruff: ignore[too-many-return-statements]
    a: object,
    b: object,
    c: object,
    /,
) -> object:
    match a, b, c:
        case float32(), float32(), float32():
            return _fmaf(a, b, c)
        case float(), float(), float():
            return _fma(a, b, c)
        case float32(), float32(), complex64():
            return complex64(_fmaf(a, b, c.real), c.imag)
        case complex64(), float32(), complex64() | float32():
            return complex64(
                _fmaf(a.real, b, c.real),
                _fmaf(a.imag, b, c.imag),
            )
        case float32(), complex64(), complex64() | float32():
            return fma(b, a, c)
        case float(), float(), complex():
            return complex(_fma(a, b, c.real), c.imag)
        case complex(), float(), complex() | float():
            return complex(_fma(a.real, b, c.real), _fma(a.imag, b, c.imag))
        case float(), complex(), complex() | float():
            return fma(b, a, c)
        case _:
            raise NotImplementedError


def _codegen(
    context: cpu.CPUContext,
    builder: "ir.IRBuilder",
    signature: typing.Signature,
    args: "tuple[ir.Value, ir.Value]",
) -> "ir.Instruction":
    c = cast(
        "cgutils.ValueStructProxy",
        context.make_complex(builder, signature.return_type),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    )
    c.real, c.imag = args
    return c._getvalue()  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # ruff: ignore[private-member-access]


@numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
@no_type_check
def _complex64(
    _: typing.Context,
    real: types.Float,
    imag: types.Float,
) -> "kn.typing.Intrinsic[ir.Value, ir.Value]":
    return types.complex64(real, imag), _codegen


@numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
@no_type_check
def _intrinsic(
    _: typing.Context,
    a: types.Float,
    b: types.Float,
    c: types.Float,
) -> "kn.typing.Intrinsic[ir.Value, ir.Value, ir.Value]":
    return (
        a(a, b, c),
        lambda _context, builder, _signature, args: builder.fma(*args),
    )


@numba.extending.overload(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    fma,
    strict=False,
    inline="always",
)
@no_type_check
def _(  # ruff: ignore[too-many-return-statements]
    a: types.Complex | types.Float,
    b: types.Complex | types.Float,
    c: types.Complex | types.Float,
    /,
) -> Callable[[_Op, _Op, _Op], object] | None:
    match a, b, c:
        case (
            (types.float32, types.float32, types.float32)
            | (types.float64, types.float64, types.float64)
        ):
            return lambda x, y, z: _intrinsic(x, y, z)
        case types.float32, types.float32, types.complex64:
            return lambda x, y, z: _complex64(_intrinsic(x, y, z.real), z.imag)
        case types.complex64, types.float32, types.float32:
            return lambda x, y, z: _complex64(
                _intrinsic(x.real, y, z),
                x.imag * y,
            )
        case types.complex64, types.float32, types.complex64:
            return lambda x, y, z: _complex64(
                _intrinsic(x.real, y, z.real),
                _intrinsic(x.imag, y, z.imag),
            )
        case types.float64, types.float64, types.complex128:
            return lambda x, y, z: complex(_intrinsic(x, y, z.real), z.imag)
        case types.complex128, types.float64, types.float64:
            return lambda x, y, z: complex(
                _intrinsic(x.real, y, z),
                x.imag * y,
            )
        case types.complex128, types.float64, types.complex128:
            return lambda x, y, z: complex(
                _intrinsic(x.real, y, z.real),
                _intrinsic(x.imag, y, z.imag),
            )
        case (
            (types.float64, types.complex128, types.complex128 | types.float64)
            | (types.float32, types.complex64, types.complex64 | types.float32)
        ):
            return lambda x, y, z: fma(y, x, z)
        case _:
            return None


_fmaf = CFUNCTYPE(c_float, c_float, c_float, c_float)(("fmaf", _lib.c))
