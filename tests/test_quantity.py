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

"""Test `kn.quantity`."""

import contextlib
import datetime as dt
import math
from contextlib import AbstractContextManager
from typing import TypeGuard, cast

import hypothesis
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import numpy_typing_compat as nptc
import optype.numpy as onp
import pandas as pd
import pint
import pytest
import xarray as xr
from pint.facets.numpy.quantity import NumpyQuantity

import kimnara as kn

_IINFO = np.iinfo(np.int64)
_TYPE_CODES = (
    "fdFD"  # Floating and complex types
    "bBhHiIqQ"  # Integer types
)


@hypothesis.given(st.floats(), st.floats())
def test_builtin_complex(re: float, im: float) -> None:
    """Test `kn.quantity(builtins.complex)`."""
    magnitude = kn.quantity(complex(re, im)).magnitude
    np.testing.assert_equal(magnitude.real, re, strict=True)
    np.testing.assert_equal(magnitude.imag, im, strict=True)


@hypothesis.given(st.floats())
def test_builtin_float(f: float) -> None:
    """Test `kn.quantity(builtins.float)`."""
    magnitude = kn.quantity(f).magnitude
    np.testing.assert_equal(magnitude, f, strict=True)


@hypothesis.given(st.integers(min_value=_IINFO.min, max_value=_IINFO.max))
def test_builtin_int(i: int) -> None:
    """Test `kn.quantity(builtins.int)`."""
    magnitude = kn.quantity(i).magnitude
    assert isinstance(magnitude, np.int64)  # pyright: ignore[reportArgumentType]
    assert magnitude == i


def test_builtin_int_overflow() -> None:
    """Test `kn.quantity(builtins.int)` beyond int64."""
    for i in _IINFO.min - 1, _IINFO.max + 1:
        with pytest.raises(OverflowError):
            kn.quantity(i)


def test_datetime_timedelta() -> None:
    """Test `kn.quantity(datetime.timedelta)`."""
    for key in (
        "days",
        "seconds",
        "microseconds",
        "milliseconds",
        "minutes",
        "hours",
        "weeks",
    ):
        value = dt.timedelta(**{key: 1})
        assert math.isclose(kn.quantity(value, key[:-1]).magnitude, 1)


def test_pandas_timedelta() -> None:
    """Test `kn.quantity(pd.Timedelta)`."""
    # No need to test the `datetime.timedelta` syntax again
    for unit in (
        "day",
        "second",
        "microsecond",
        "millisecond",
        "minute",
        "hour",
        "nanosecond",
    ):
        value = pd.Timedelta(1, unit)
        assert math.isclose(kn.quantity(value, unit).magnitude, 1)

    # https://github.com/pandas-dev/pandas/pull/46171
    value = pd.Timedelta("NaT")
    with pytest.raises(TypeError):
        kn.quantity(value)


def test_numpy_timedelta64() -> None:
    """Test `kn.quantity(np.timedelta64)` and its (x)array variants."""
    with _np25_deprecated_generic_timedelta():
        for quantity in [
            kn.quantity(np.timedelta64(1, "m")),
            kn.quantity(np.timedelta64(1), "minute"),  # pyright: ignore[reportDeprecated]
        ]:
            assert math.isclose(quantity.magnitude, 1)
            assert str(quantity.units) == "minute"

    with _np25_deprecated_generic_timedelta():
        quantity = kn.quantity(np.timedelta64("NaT"))  # pyright: ignore[reportDeprecated]
    assert math.isnan(quantity.magnitude)
    assert str(quantity.units) == "second"

    with _np25_deprecated_generic_timedelta():
        quantity = kn.quantity(np.timedelta64(0))  # pyright: ignore[reportDeprecated]
    assert quantity.magnitude == 0
    assert str(quantity.units) == "second"

    with (
        _np25_deprecated_generic_timedelta(),
        pytest.raises(pint.UndefinedUnitError),
    ):
        kn.quantity(np.timedelta64(1))  # pyright: ignore[reportDeprecated]

    sctype = np.dtype("m8[s]")
    for unit in "Y", "M", "W", "D", "h", "m", "s":
        scalar = np.timedelta64(1, unit)
        seconds = scalar.astype(sctype).view(np.int64)
        assert math.isclose(kn.quantity(scalar, "s").magnitude, seconds)
        array = np.full(2, scalar)
        magnitude = kn.quantity(array, "s").magnitude
        np.testing.assert_allclose(magnitude, seconds, strict=False)
        # Year/month will be converted to 365/31 days by pd.Series at
        # https://github.com/pydata/xarray/blob/v2026.07.0/xarray/core/variable.py#L223
        if unit not in "YM":
            quantity = kn.quantity(xr.DataArray(array), "s").data
            assert _is_quantity(quantity)
            np.testing.assert_allclose(quantity.m, seconds, strict=False)

    sctype = np.dtype("m8[as]")
    for unit in "ms", "us", "ns", "ps", "fs", "as":
        scalar = np.timedelta64(1, unit)
        attoseconds = scalar.astype(sctype).view(np.int64)
        assert math.isclose(kn.quantity(scalar, "as").magnitude, attoseconds)
        array = np.full(2, scalar)
        magnitude = kn.quantity(array, "as").magnitude
        np.testing.assert_allclose(magnitude, attoseconds, strict=False)
        # Precision loss during the same conversion as above
        if unit not in {"ps", "fs", "as"}:
            quantity = kn.quantity(xr.DataArray(array), "as").data
            assert _is_quantity(quantity)
            np.testing.assert_allclose(quantity.m, attoseconds, strict=False)


@hypothesis.given(
    st.integers(min_value=0, max_value=127),
    st.sampled_from(_TYPE_CODES),
)
def test_numpy_number(i: int, dtype: str) -> None:
    """Test `kn.quantity(np.number)` and its pint/xarray variants."""
    number = cast("type[np.number]", np.dtype(dtype).type)
    assert issubclass(number, np.number)
    scalar = number(i)

    quantity = kn.quantity(scalar)
    magnitude = quantity.magnitude
    assert isinstance(magnitude, number)
    assert magnitude == i

    quantity = kn.quantity(quantity)
    magnitude = quantity.magnitude
    assert isinstance(magnitude, number)
    assert magnitude == i

    da = xr.DataArray(quantity)
    quantity = kn.quantity(da).data
    assert _is_quantity(quantity)
    magnitude = quantity.magnitude
    assert isinstance(magnitude, number)
    assert magnitude == i

    da = xr.DataArray(scalar)  # Converted to 0-D array here
    quantity = kn.quantity(da).data
    assert _is_quantity(quantity)
    magnitude = quantity.magnitude
    assert isinstance(magnitude, np.ndarray)  # Different from the case above
    np.testing.assert_equal(magnitude, number(i), strict=True)


@hypothesis.given(st.sampled_from(_TYPE_CODES))
def test_numpy_ndarray(dtype: str) -> None:
    """Test `kn.quantity(np.ndarray)` and its pint/xarray variants."""
    number = cast("type[np.number]", np.dtype(dtype).type)
    assert issubclass(number, np.number)
    array = np.arange(100, dtype=number)

    quantity = kn.quantity(array)
    np.testing.assert_array_equal(quantity.magnitude, array, strict=True)

    quantity = kn.quantity(quantity)
    np.testing.assert_array_equal(quantity.magnitude, array, strict=True)

    for da in xr.DataArray(quantity), xr.DataArray(array):
        quantity = kn.quantity(da).data
        assert _is_quantity(quantity)
        np.testing.assert_array_equal(quantity.magnitude, array, strict=True)


def test_xarray_unsupported_dtype() -> None:
    """Test `kn.quantity(xr.DataArray)` with unsupported dtypes.

    This test case is not covered by a static type checker.
    """
    for value in True, False, None, "str", b"bytes", np.datetime64(0, "ns"):
        with pytest.raises(TypeError):
            kn.quantity(xr.DataArray(value))


def test_unit_conversions() -> None:
    """Test `kn.quantity(..., str)`."""
    quantity = kn.quantity(1000, "m")
    assert math.isclose(kn.quantity(quantity, "km").magnitude, 1)
    assert math.isclose(kn.quantity(quantity, "mm").magnitude, 1e6)
    for units in "", "m2":
        with pytest.raises(pint.DimensionalityError):
            kn.quantity(quantity, units)
    with pytest.raises(pint.DimensionalityError):
        kn.quantity(kn.quantity(1000), "m")

    da = xr.DataArray(quantity)
    for units, desired in [("km", 1), ("mm", 1e6)]:
        quantity = kn.quantity(da, units).data
        assert _is_quantity(quantity)
        assert math.isclose(quantity.magnitude, desired)
    for units in "", "m2":
        with pytest.raises(pint.DimensionalityError):
            kn.quantity(quantity, units)
    with pytest.raises(pint.DimensionalityError):
        kn.quantity(kn.quantity(xr.DataArray(1000)), "m")

    da = xr.Dataset({"dim": np.arange(3)}).coords["dim"]
    da.attrs["units"] = "m"
    assert isinstance(da.variable, xr.IndexVariable)

    da = kn.quantity(da, "mm")
    assert isinstance(da.variable, xr.Variable)
    quantity = da.data
    assert _is_quantity(quantity)
    np.testing.assert_allclose(quantity.m, [.0, 1000., 2000.], strict=True)

    da = cast("xr.DataArray", da.coords["dim"])
    assert isinstance(da.variable, xr.IndexVariable)
    array = da.data
    assert onp.is_array_1d(array, np.int64)
    np.testing.assert_array_equal(array, [0, 1, 2], strict=True)
    assert da.attrs.get("units") == "m"


def _is_quantity(x: object) -> TypeGuard[
    NumpyQuantity[np.number | npt.NDArray[np.number]]
]:
    return isinstance(x, pint.Quantity)


if getattr(nptc, "NUMPY_GE_2_5", False):
    def _np25_deprecated_generic_timedelta() -> AbstractContextManager[object]:
        return pytest.warns(
            DeprecationWarning,
            match="The 'generic' unit for NumPy timedelta is deprecated",
        )
else:
    def _np25_deprecated_generic_timedelta() -> AbstractContextManager[object]:
        return contextlib.nullcontext()
