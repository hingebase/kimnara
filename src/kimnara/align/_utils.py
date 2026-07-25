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

__all__ = [
    "array",
    "asarray",
    "empty",
    "isaligned",
]

import itertools
import warnings
from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from optype.typing import AnyComplex
from typing_extensions import Any, TypeVar, overload

import kimnara as kn
from kimnara import _spec, _utils

_ScalarType: TypeAlias = """
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.intp
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.uintp
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128"""
_ShapeT = TypeVar("_ShapeT", bound=onp.AtLeast1D)
_T = TypeVar(
    "_T",
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)

_DTYPES = frozenset(map(np.dtype, (
    "fdFD"  # Floating and complex types
    "bBhHlLqQ"  # Integer types
    "?"  # Boolean type
)))


@overload
def array(
    a: np.ndarray[_ShapeT, np.dtype[_ScalarType]],
    dtype: type[_T],
    *,
    copy: bool | None = ...,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[_ShapeT, np.dtype[_T]]: ...
@overload
def array(
    a: np.ndarray[_ShapeT, np.dtype[_T]],
    dtype: None = ...,
    *,
    copy: bool | None = ...,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[_ShapeT, np.dtype[_T]]: ...
def array(
    a: np.ndarray[onp.AtLeast1D, np.dtype[Any]],
    dtype: type[_ScalarType] | None = None,
    *,
    copy: bool | None = True,
    align: "str | kn.Alignment" = "host",
    pad_value: AnyComplex | None = None,
) -> npt.NDArray[_ScalarType]:
    _check_array_dtype_and_ndim(a)
    dtype = dtype or a.dtype.type
    match align := kn.Alignment(align):
        case kn.A | kn.C | kn.F:
            if not copy:
                if a.dtype == np.dtype(dtype) and isaligned(a, align):
                    return np.asarray(a)
                if copy is False:
                    raise ValueError(_messages["copy"])
            return a.astype(dtype, order=align._name_, subok=False)
        case _:
            pass
    if not copy:
        if (
            a.dtype == np.dtype(dtype)
            and isaligned(a, align)
            and (
                pad_value is None
                # Skip checking the content of padding bytes
                # as it can be quite slow
                or _calculate_padding(a.shape, dtype, align.value) == 0
            )
        ):
            return np.asarray(a)
        if copy is False:
            raise ValueError(_messages["copy"])
    shape = a.shape
    if padding := _calculate_padding(shape, dtype, align.value):
        arr = _empty(shape, padding, dtype, align, pad_value)
        if np.iscomplexobj(a) and not issubclass(dtype, np.complexfloating):
            warnings.warn(
                "Casting complex values to real discards the imaginary part",
                category=np.exceptions.ComplexWarning,
                stacklevel=2,
            )
        arr[:] = a
        return arr
    with align.allocator:
        return a.astype(dtype, order="C", subok=False)


@overload
def asarray(
    a: np.ndarray[_ShapeT, np.dtype[_ScalarType]],
    dtype: type[_T],
    *,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[_ShapeT, np.dtype[_T]]: ...
@overload
def asarray(
    a: np.ndarray[_ShapeT, np.dtype[_T]],
    dtype: None = ...,
    *,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[_ShapeT, np.dtype[_T]]: ...
def asarray(
    a: np.ndarray[_ShapeT, np.dtype[Any]],
    dtype: type[_ScalarType] | None = None,
    *,
    align: "str | kn.Alignment" = "host",
    pad_value: AnyComplex | None = None,
) -> np.ndarray[_ShapeT, np.dtype[_ScalarType]]:
    return array(a, dtype, align=align, copy=None, pad_value=pad_value)


@overload
def empty(
    shape: _ShapeT,
    dtype: type[_T] = np.float64,
    *,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[_ShapeT, np.dtype[_T]]: ...
@overload
def empty(
    shape: int,
    dtype: type[_T] = np.float64,
    *,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> onp.Array1D[_T]: ...
@overload
def empty(
    shape: Sequence[int],
    dtype: type[_T] = np.float64,
    *,
    align: "str | kn.Alignment" = ...,
    pad_value: AnyComplex | None = ...,
) -> np.ndarray[onp.AtLeast1D, np.dtype[_T]]: ...
def empty(
    shape: int | Sequence[int],
    dtype: type[_ScalarType] = np.float64,
    *,
    align: "str | kn.Alignment" = "host",
    pad_value: AnyComplex | None = None,
) -> npt.NDArray[_ScalarType]:
    if isinstance(shape, int):
        shape = (shape,)
    elif not shape:
        raise ValueError(_messages["ndim"])
    match kn.Alignment(align):
        case kn.A | kn.C:
            return np.empty(shape, dtype)
        case kn.F:
            return np.empty(shape, dtype, order="F")
        case align:
            if padding := _calculate_padding(shape, dtype, align.value):
                return _empty(shape, padding, dtype, align, pad_value)
            with align.allocator:
                return np.empty(shape, dtype)


def isaligned(
    value: np.ndarray[onp.AtLeast1D, np.dtype[_ScalarType]],
    align: "str | kn.Alignment",
) -> bool:
    _check_array_dtype_and_ndim(value)
    if not value.size:
        return True
    match kn.Alignment(align):
        case kn.A:
            return value.flags.aligned
        case kn.C:
            flags = value.flags
            return flags.aligned and flags.c_contiguous
        case kn.F:
            flags = value.flags
            return flags.aligned and flags.f_contiguous
        case align:
            return _isaligned_simd(value, align.value)


def _calculate_padding(
    shape: Sequence[int],
    dtype: type[_ScalarType],
    spec: _spec.Alignment,
) -> int:
    itemsize = np.dtype(dtype).itemsize
    nbytes = shape[-1] * itemsize
    mask = spec.multiple_of - 1
    aligned = (nbytes + mask) & ~mask
    if len(shape) > 1 and aligned % spec.not_multiple_of == 0:
        aligned += spec.multiple_of
    quot, rem = divmod(aligned - nbytes, itemsize)
    if rem:
        message = f"{np.dtype(dtype)!r} is unsupported in Kimnara"
        raise TypeError(message)
    return quot


def _check_array_dtype_and_ndim(x: object) -> None:
    if not isinstance(x, np.ndarray):
        message = f"Expected np.ndarray, got {_utils.base_repr(x)}"
        raise TypeError(message)
    dtype = cast("np.dtype[Any]", x.dtype)
    if dtype not in _DTYPES:  # Also compares byteorder
        message = f"{dtype!r} is unsupported in Kimnara"
        raise ValueError(message)
    if x.ndim == 0:
        raise ValueError(_messages["ndim"])


def _empty(
    shape: Sequence[int],
    padding: int,
    dtype: type[_ScalarType],
    align: "kn.Alignment",
    pad_value: AnyComplex | None,
) -> npt.NDArray[_ScalarType]:
    shape = list(shape)
    n = shape[-1]
    shape[-1] = n + padding
    with align.allocator:
        arr = np.empty(shape, dtype)
    if pad_value is not None:
        arr[..., n:] = _spec.scalar(pad_value)
    return arr[..., :n]


def _isaligned_simd(
    value: np.ndarray[onp.AtLeast1D, np.dtype[_ScalarType]],
    spec: _spec.Alignment,
) -> bool:
    multiple_of = spec.multiple_of
    if value.ctypes.data % multiple_of:
        return False
    strides = value.strides
    if strides[-1] != value.itemsize:
        return False
    try:
        leading_dimension = strides[-2]
    except IndexError:  # 1-D array
        # We don't know if the underlying memory was padded properly
        # Assuming false here
        return value.nbytes % multiple_of == 0
    for shape0, (stride0, stride1) in zip(
        reversed(value.shape),
        itertools.pairwise(reversed(strides)),
        strict=False,
    ):
        if stride1 < shape0 * stride0 or stride1 % multiple_of:
            return False
    return bool(leading_dimension % spec.not_multiple_of)


_messages = {
    "copy": "Unable to avoid copy while creating an array as requested",
    "ndim": "0-dimensional array is discouraged in Kimnara",
}
