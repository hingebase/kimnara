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

"""Test `kn.ufunc`.

Features shared with `kn.func`, `kn.gufunc` or `kn.cfunc` are not tested
here.
"""

from typing import Annotated

import annotated_types as at
import numpy as np
import optype.numpy as onp
import pytest
from pint.facets.numpy.quantity import NumpyQuantity

import kimnara as kn


def test_methods() -> None:
    """Numba/Python ufunc should support all methods except for `at`."""
    x = kn.quantity(
        np.linspace(0, .008, 9, dtype=np.float64).reshape(3, 3),
        "km",
    )
    desired0 = np.array([[0, 1, 2], [3, 5, 7], [9, 12, 15]], np.int64)
    desired1 = np.array([[0, 1, 3], [3, 7, 12], [6, 13, 21]], np.int64)
    desired2 = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]], np.int64)

    for ufunc in _reorderable_nb, _reorderable_py:
        accumulated = ufunc.accumulate(x)
        assert isinstance(accumulated, NumpyQuantity)
        np.testing.assert_array_equal(
            accumulated.magnitude, desired0, strict=True)
        accumulated = ufunc.accumulate(x, axis=1)
        assert isinstance(accumulated, NumpyQuantity)
        np.testing.assert_array_equal(
            accumulated.magnitude, desired1, strict=True)

        reduced = ufunc.reduce(x)
        assert isinstance(reduced, NumpyQuantity)
        np.testing.assert_array_equal(
            reduced.magnitude, desired0[2], strict=True)
        reduced = ufunc.reduce(x, axis=1)
        assert isinstance(reduced, NumpyQuantity)
        np.testing.assert_array_equal(
            reduced.magnitude, desired1[:, 2], strict=True)
        for axis in (0, 1), None:
            reduced = ufunc.reduce(x, axis)
            assert isinstance(reduced, NumpyQuantity)
            np.testing.assert_equal(reduced.magnitude, 36)

        reduced = ufunc.reduceat(x, [0, 0])
        assert isinstance(reduced, NumpyQuantity)
        np.testing.assert_array_equal(
            reduced.magnitude, desired0[::2], strict=True)
        reduced = ufunc.reduceat(x, [0, 0], axis=1)
        assert isinstance(reduced, NumpyQuantity)
        np.testing.assert_array_equal(
            reduced.magnitude, desired1[:, ::2], strict=True)

        out = ufunc(x[0], x[:, 0])
        assert isinstance(out, NumpyQuantity)
        assert onp.is_array_1d(out.magnitude, np.int64)
        np.testing.assert_array_equal(
            out.magnitude, np.diag(desired2), strict=True)

        outer = ufunc.outer(x[0], x[:, 0], order="F")
        assert isinstance(outer, NumpyQuantity)
        assert onp.is_array_2d(outer.magnitude, np.int64)
        assert outer.magnitude.flags.f_contiguous
        np.testing.assert_array_equal(outer.magnitude, desired2, strict=True)

    for ufunc in _not_reorderable_nb, _not_reorderable_py:
        for axis in (0, 1), None:
            with pytest.raises(ValueError, match="not reorderable"):
                ufunc.reduce(x.magnitude, axis)

    for ufunc in _not_swappable_nb, _not_swappable_py:
        with pytest.raises(ValueError, match="same kind"):
            ufunc.accumulate(x.magnitude)
        with pytest.raises(ValueError, match="same kind"):
            ufunc.reduce(x.magnitude)
        with pytest.raises(ValueError, match="same kind"):
            ufunc.reduceat(x.magnitude, [0, 0])

    for ufunc in _unary_nb, _unary_py:
        with pytest.raises(ValueError, match="binary functions"):
            ufunc.accumulate(x.magnitude)
        with pytest.raises(ValueError, match="binary functions"):
            ufunc.reduce(x.magnitude)
        with pytest.raises(ValueError, match="binary functions"):
            ufunc.reduceat(x.magnitude, [0, 0])


def test_multi_output() -> None:
    """Python ufunc should support multiple outputs."""
    x1 = np.array([7, 11, 13], np.int64)
    x2 = np.array([2, 3, 5], np.int64)
    # Construct the desired results manually since the type stub for
    # np.divmod.outer is missing
    desired_quot = np.array([[3, 2, 1], [5, 3, 2], [6, 4, 2]], np.int64)
    desired_rem = np.array([[1, 1, 2], [1, 2, 1], [1, 1, 3]], np.int64)

    actual_quot, actual_rem = _multi_output(x1, x2)
    assert onp.is_array_1d(actual_quot, np.int64)
    assert isinstance(actual_rem, NumpyQuantity)
    assert onp.is_array_1d(actual_rem.magnitude, np.int64)
    np.testing.assert_array_equal(
        actual_quot, np.diag(desired_quot), strict=True)
    np.testing.assert_array_equal(
        actual_rem.magnitude, np.diag(desired_rem), strict=True)

    actual_quot, actual_rem = _multi_output.outer(x1, x2, order="F")
    assert onp.is_array_2d(actual_quot, np.int64)
    assert actual_quot.flags.f_contiguous
    assert isinstance(actual_rem, NumpyQuantity)
    assert onp.is_array_2d(actual_rem.magnitude, np.int64)
    assert actual_rem.magnitude.flags.f_contiguous
    np.testing.assert_array_equal(actual_quot, desired_quot, strict=True)
    np.testing.assert_array_equal(
        actual_rem.magnitude, desired_rem, strict=True)


@kn.ufunc
def _multi_output(x1: int, x2: int) -> tuple[
    int,
    Annotated[int, at.Unit("dimensionless")],
]:
    return divmod(x1, x2)


@kn.ufunc(cache=False, nopython=True)
def _not_reorderable_nb(x1: np.float64, x2: np.float64) -> np.float64:
    return x1 + x2


@kn.ufunc
def _not_reorderable_py(x1: float, x2: float) -> float:
    return x1 + x2


@kn.ufunc(cache=False, identity=0, nopython=True, parallel="workqueue")
def _not_swappable_nb(
    x1: np.float64,
    x2: Annotated[np.float64, at.Unit("dimensionless")],
) -> np.float64:
    return x1 + x2


@kn.ufunc
def _not_swappable_py(
    x1: Annotated[float, at.Unit("dimensionless")],
    x2: float,
) -> float:
    return x1 + x2


@kn.ufunc(cache=False, identity=0, nopython=True)
def _reorderable_nb(
    x1: Annotated[np.int64, at.Unit("m")],
    x2: Annotated[np.int64, at.Unit("m")],
) -> Annotated[np.int64, at.Unit("m")]:
    return x1 + x2


@kn.ufunc(identity="reorderable")
def _reorderable_py(
    x1: Annotated[int, at.Unit("m")],
    x2: Annotated[int, at.Unit("m")],
) -> Annotated[int, at.Unit("m")]:
    return x1 + x2


@kn.ufunc(cache=False, nopython=True, parallel="workqueue")
def _unary_nb(x: np.float64) -> np.float64:
    return x


@kn.ufunc
def _unary_py(x: float) -> float:
    return x
