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
from typing import TYPE_CHECKING, cast

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from numpy_typing_compat import NUMPY_GE_2_0
from typing_extensions import Any

import kimnara as kn

if TYPE_CHECKING:
    import optype.numpy as onp

_SIMD = (kn.SSE, kn.AVX, kn.AVX512, kn.Alignment.MKL)
_TYPE_CODES = (
    "fdFD"  # Floating and complex types
    "bBhHlLqQ"  # Integer types
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


def _exactly_aligned(nbytes: int) -> bool:
    arrs = (
        np.empty((2**i + j,), np.uint8) for i in range(22) for j in range(32)
    )
    twice = 2 * nbytes
    return {arr.ctypes.data % twice for arr in arrs} == {0, nbytes}
