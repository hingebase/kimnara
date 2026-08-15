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

"""Test `kn.gufunc`.

Features shared with `kn.func`, `kn.ufunc` or `kn.cfunc` are not tested
here.
"""

from typing import Annotated

import annotated_types as at
import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import pytest
from pint.facets.numpy.quantity import NumpyQuantity

import kimnara as kn


def test_multi_output_nb() -> None:
    """Numba gufunc should support multiple outputs."""
    desired = np.arange(8, dtype=np.int16).reshape(2, 2, 2)  # pyright: ignore[reportUnknownMemberType]
    actual = _multi_output_nb(desired)
    assert isinstance(actual, tuple)
    out1, out2 = actual

    assert onp.is_array_3d(out1, np.int16)
    np.testing.assert_array_equal(out1, desired, strict=True)
    assert not kn.isaligned(out1, kn.SSE)
    assert out1.base is None  # pyright: ignore[reportUnknownMemberType]

    assert isinstance(out2, NumpyQuantity)
    assert onp.is_array_2d(out2.magnitude, np.int8)
    np.testing.assert_equal(out2.magnitude, 1)
    assert kn.isaligned(out2.magnitude, kn.SSE)
    assert out2.magnitude.base is not None  # pyright: ignore[reportUnknownMemberType]


def test_multi_output_py() -> None:
    """Python gufunc should support multiple outputs."""
    desired = np.arange(4, dtype=np.uint16).reshape(2, 2)  # pyright: ignore[reportUnknownMemberType]
    out1, out2 = _multi_output_py(desired)

    assert isinstance(out1, NumpyQuantity)
    assert onp.is_array_1d(out1.magnitude, np.uint8)
    np.testing.assert_equal(out1.magnitude, 1)

    assert onp.is_array_2d(out2, np.uint16)
    np.testing.assert_array_equal(out2, desired, strict=True)


def test_order_nb(subtests: pytest.Subtests) -> None:
    """Numba gufunc should support the `order` keyword argument."""
    with subtests.test("Fast path: output arrays allocated by NumPy"):
        desired = np.arange(4, dtype=np.float64).reshape(2, 2, order="F")  # pyright: ignore[reportUnknownMemberType]
        actual = _order_nb_1(desired)  # Pass order="C" internally
        assert onp.is_array_2d(actual, np.complex128)
        for align in kn.C, kn.SSE:
            assert kn.isaligned(actual, align)
        np.testing.assert_array_equal(actual.real, desired, strict=True)
        im = actual.imag.view(np.int64)
        assert im[0, 0] == 1  # desired.flags.f_contiguous
        assert im[0, 1] == 1  # actual.flags.c_contiguous
        assert im[1, 0] == desired.ctypes.data  # `desired` hasn't been copied
        assert im[1, 1] == actual.ctypes.data  # `actual` hasn't been copied

    with subtests.test("Slow path: output arrays allocated by Kimnara"):
        x = np.empty((2, 2), np.int64, order="F")
        with pytest.raises(
            ValueError,
            match="order='C' conflicts with SIMD alignment",
        ):
            _order_nb_2(x, order="C")
        out = _order_nb_2(x)
        assert onp.is_array_2d(out, np.int64)
        assert kn.isaligned(out, kn.AVX)
        assert not kn.isaligned(out, kn.C)
        assert out[0, 0] == 1  # x.flags.f_contiguous
        assert out[0, 1] == 1  # out[1:].flags.c_contiguous
        assert out[1, 0] == x.ctypes.data  # `x` hasn't been copied
        assert out[1, 1] == out.ctypes.data  # `out` hasn't been copied

        desired = np.arange(16, dtype=np.int64).reshape(2, 2, 4)  # pyright: ignore[reportUnknownMemberType]
        actual = _order_nb_2(desired, order="C")
        assert onp.is_array_3d(actual, np.int64)
        for align in kn.C, kn.AVX:
            assert kn.isaligned(actual, align)
        np.testing.assert_array_equal(actual, desired, strict=True)

    with subtests.test("Dynamic path: postpone the decision until called"):
        x = np.empty((2, 3), np.int64, order="F")
        with pytest.raises(
            ValueError,
            match="order='C' conflicts with SSE alignment",
        ):
            _order_nb_3(x, order="C")
        out = _order_nb_3(x)
        assert onp.is_array_2d(out, np.int64)
        assert kn.isaligned(out, kn.SSE)
        assert not kn.isaligned(out, kn.C)
        assert out[0, 0] == 1  # x.flags.f_contiguous
        assert out[0, 1] == 1  # out[1:].flags.c_contiguous
        assert out[1, 0] == x.ctypes.data  # `x` hasn't been copied
        assert out[1, 1] == out.ctypes.data  # `out` hasn't been copied

        desired = np.arange(8, dtype=np.int64).reshape(2, 2, 2)  # pyright: ignore[reportUnknownMemberType]
        actual = _order_nb_3(desired, order="C")
        assert onp.is_array_3d(actual, np.int64)
        for align in kn.C, kn.SSE:
            assert kn.isaligned(actual, align)
        np.testing.assert_array_equal(actual, desired, strict=True)


def test_order_py() -> None:
    """Python gufunc should support the `order` keyword argument."""
    desired = np.arange(4, dtype=np.int64).reshape(2, 2, order="F")  # pyright: ignore[reportUnknownMemberType]

    actual = _order_py(desired)
    np.testing.assert_array_equal(actual, desired, strict=True)
    assert actual.flags.f_contiguous

    actual = _order_py(desired, order="A")  # Pass order="C" internally
    np.testing.assert_array_equal(actual, desired, strict=True)
    assert actual.flags.c_contiguous


@kn.gufunc("(n)->(n),()", cache=False, nopython=True, parallel="workqueue")
def _multi_output_nb(
    x: onp.Array1D[np.int16],
    out1: np.ndarray[tuple[int, ...], np.dtype[np.int16]],
    out2: Annotated[onp.Array1D[np.int8], at.Unit("dimensionless"), kn.SSE],
) -> None:
    out1[:] = x
    out2[0] = 1


@kn.gufunc("(n)->(),(n)")
def _multi_output_py(x: onp.Array1D[np.uint16]) -> tuple[
    Annotated[np.uint8, at.Unit("dimensionless")],
    npt.NDArray[np.uint16],
]:
    return np.uint8(1), x


@kn.gufunc("(m,n)->(m,n)", cache=False, nopython=True)
def _order_nb_1(
    x: onp.Array2D[np.float64],
    out: Annotated[
        np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
        kn.SSE,
    ],
) -> None:
    out.real[:] = x
    im = out.imag.view(np.int64)
    im[0, 0] = x.flags.f_contiguous
    im[0, 1] = out.flags.c_contiguous
    im[1, 0] = x.ctypes.data
    im[1, 1] = out.ctypes.data


@kn.gufunc("(m,n)->(m,n)", cache=False, nopython=True, parallel="workqueue")
def _order_nb_2(
    x: npt.NDArray[np.int64],
    out: Annotated[onp.Array2D[np.int64], kn.AVX],
) -> None:
    if out.shape == (2, 2):
        out[0, 0] = x.flags.f_contiguous
        out[0, 1] = out[1:].flags.c_contiguous
        out[1, 0] = x.ctypes.data
        out[1, 1] = out.ctypes.data
    else:
        out[:] = x


@kn.gufunc("(m,n)->(m,n)", cache=False, nopython=True)
def _order_nb_3(
    x: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    out: Annotated[npt.NDArray[np.int64], kn.SSE],
) -> None:
    if out.shape == (2, 3):
        out[0, 0] = x.flags.f_contiguous
        out[0, 1] = out[1:].flags.c_contiguous
        out[1, 0] = x.ctypes.data
        out[1, 1] = out.ctypes.data
    else:
        out[:] = x


@kn.gufunc("(m,n)->(m,n)")
def _order_py(x: onp.Array2D[np.int64]) -> onp.Array2D[np.int64]:
    return x
