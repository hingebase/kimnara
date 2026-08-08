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

"""Test kimnara.align API."""

import secrets
from typing import cast, no_type_check

import hypothesis
import hypothesis.strategies as st
import numba  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import pytest
from numpy_typing_compat import NUMPY_GE_2_0
from typing_extensions import Any, overload

import kimnara as kn

_SIMD = (kn.SSE, kn.AVX, kn.AVX512, kn.Alignment.MKL)
_TYPE_CODES = (
    "fdFD"  # Floating and complex types
    "bBhHiIqQ"  # Integer types
    "?"  # Boolean type
)


def test_allocator(subtests: pytest.Subtests) -> None:
    """Test direct use of allocator objects."""
    with subtests.test("32-byte alignment"), kn.align.AVXAllocator:
        assert _exactly_aligned(32)

    with subtests.test("64-byte alignment"), kn.align.AVX512Allocator:
        assert _exactly_aligned(64)

    with subtests.test("Default 16-byte alignment"):
        assert _exactly_aligned(16)


def test_constructors() -> None:
    """Test `kn.array`, `kn.asarray` and `kn.empty`."""
    with pytest.raises(ValueError, match="0-dimensional array"):
        kn.empty([])

    a = kn.empty((2, 511), align="mkl", pad_value=1)
    assert not a.flags.c_contiguous
    base = a.base  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert base is not None
    assert base.flags.carray
    assert base.shape == (2, 520)
    np.testing.assert_equal(base[:, 511:], 1)  # pyright: ignore[reportUnknownArgumentType]
    a[0] = range(1000, 1511)
    a[1] = range(2000, 2511)
    for align in _SIMD:
        assert kn.asarray(a, np.float64, align=align) is a
        b = kn.asarray(a, align=align, pad_value=1)
        assert not np.shares_memory(a, b)
        np.testing.assert_array_equal(a, b, strict=True)
        base = b.base  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert base is not None
        assert base.flags.carray
        assert base.shape == (2, 520 if align == kn.Alignment.MKL else 512)

    a = kn.empty(512, align=kn.Alignment.MKL)
    assert a.base is None  # pyright: ignore[reportUnknownMemberType]
    assert a.shape == (512,)
    for align in kn.Alignment:
        assert kn.asarray(a, np.float64, align=align, pad_value=2) is a
    a[:] = range(512)
    b = kn.array(a, align="A", pad_value=1)
    assert not np.shares_memory(a, b)
    np.testing.assert_array_equal(a, b, strict=True)


@hypothesis.given(
    st.integers(min_value=3, max_value=1048576),
    st.sampled_from(_TYPE_CODES),
)
def test_isaligned(n: int, dtype: str) -> None:
    """Test `kn.isaligned` for non-empty arrays."""
    arr = np.empty((n,), dtype)
    for align in "ACF":
        assert kn.isaligned(arr, align)
    ptr = arr.ctypes.data
    nbytes = arr.nbytes
    for align in _SIMD:
        multiple_of = align.value.multiple_of
        if ptr % multiple_of or nbytes % multiple_of:
            assert not kn.isaligned(arr, align)
        else:
            assert kn.isaligned(arr, align)

    sub = cast("onp.Array1D[Any]", arr[::2])
    for align in kn.Alignment:
        assert (align is not kn.A) ^ kn.isaligned(sub, align)

    arr = np.empty((2, n), dtype)
    transposed = arr.T
    for align in "AC":
        assert kn.isaligned(arr, align)
    for align in "AF":
        assert kn.isaligned(transposed, align)
    assert not kn.isaligned(arr, kn.F)
    assert not kn.isaligned(transposed, kn.C)
    ptr = arr.ctypes.data
    for align in _SIMD:
        assert not kn.isaligned(transposed, align)
        multiple_of = align.value.multiple_of
        if (
            ptr % multiple_of
            or nbytes % multiple_of
            or arr.strides[0] % align.value.not_multiple_of == 0
        ):
            assert not kn.isaligned(arr, align)
        else:
            assert kn.isaligned(arr, align)
            sub = cast("onp.Array2D[Any]", arr[:, :-1])
            assert kn.isaligned(sub, align)


@hypothesis.given(
    st.integers(min_value=1, max_value=64 if NUMPY_GE_2_0 else 32),
    st.sampled_from(_TYPE_CODES),
)
def test_isaligned_empty(n: int, dtype: str) -> None:
    """Test `kn.isaligned` for zero-sized arrays."""
    shape = [1] * n
    shape[secrets.randbelow(n)] = 0
    arr = cast(
        "np.ndarray[onp.AtLeast1D, np.dtype[Any]]",
        np.empty(shape, dtype),
    )
    for align in kn.Alignment:
        assert kn.isaligned(arr, align)


def test_numba_guvectorize() -> None:
    """`numba.guvectorize` should preserve aligned input arrays."""
    with kn.align.AVXAllocator:
        avx = np.empty((3, 3, 100))[:, :, :-3]
    with kn.align.AVX512Allocator:
        avx512 = np.empty((3, 3, 256), np.float32)[:, :, :-7]
    assert _numba_guvectorize_1d(avx, 32).all()
    assert not _numba_guvectorize_1d(avx, 64).all()
    assert _numba_guvectorize_1d(avx512, 64).all()
    assert _numba_guvectorize_2d(avx, 32).all()
    assert not _numba_guvectorize_2d(avx, 64).all()
    assert _numba_guvectorize_2d(avx512, 64).all()


def test_misaligned() -> None:
    """`kn.isaligned` should respect `np.ndarray.flags.aligned`.

    This is unnecessary for SIMD alignment which is more strict.
    """
    buf = np.empty(25, np.int8)
    arr = cast("onp.Array1D[np.int32]", buf[1:].view(np.int32))
    for align in "ACF":
        assert not kn.isaligned(arr, align)
    arr = cast("onp.Array1D[np.int32]", buf[:-1].view(np.int32))
    for align in "ACF":
        assert kn.isaligned(arr, align)
    arr = cast("onp.Array1D[np.int32]", np.ndarray(5, "i", buf, strides=5))
    for align in "ACF":
        assert not kn.isaligned(arr, align)


def _exactly_aligned(nbytes: int) -> bool:
    arrs = (
        np.empty((2**i + j,), np.uint8) for i in range(22) for j in range(32)
    )
    twice = 2 * nbytes
    return {arr.ctypes.data % twice for arr in arrs} == {0, nbytes}


@overload
def _numba_guvectorize_1d(
    x: npt.NDArray[np.float32],
    align: int,
) -> npt.NDArray[np.bool_]: ...

@overload
def _numba_guvectorize_1d(
    x: npt.NDArray[np.float64],
    align: int,
) -> npt.NDArray[np.bool_]: ...

@numba.guvectorize(  # pyright: ignore[reportUnknownMemberType]
    ["(float32[:], intp, bool_[:])", "(float64[:], intp, bool_[:])"],
    "(n),()->()",
)
@no_type_check
def _numba_guvectorize_1d(
    x: onp.Array1D[np.float32 | np.float64],
    align: np.intp,
    out: onp.Array1D[np.bool_],
) -> None:
    out[0] = x.ctypes.data % align == 0


@overload
def _numba_guvectorize_2d(
    x: npt.NDArray[np.float32],
    align: int,
) -> npt.NDArray[np.bool_]: ...

@overload
def _numba_guvectorize_2d(
    x: npt.NDArray[np.float64],
    align: int,
) -> npt.NDArray[np.bool_]: ...

@numba.guvectorize(  # pyright: ignore[reportUnknownMemberType]
    ["(float32[:, :], intp, bool_[:])", "(float64[:, :], intp, bool_[:])"],
    "(m,n),()->()",
)
@no_type_check
def _numba_guvectorize_2d(
    x: onp.Array2D[np.float32 | np.float64],
    align: np.intp,
    out: onp.Array1D[np.bool_],
) -> None:
    for i in range(len(x)):
        if x[i].ctypes.data % align:
            out[0] = False
            return
    out[0] = True
