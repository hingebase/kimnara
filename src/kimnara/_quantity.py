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

__all__ = ["quantity"]

import datetime
import functools
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import xarray as xr
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import Any, TypeVar, overload

from kimnara import _spec, _utils

if TYPE_CHECKING:
    from typing import type_check_only

    import metpy  # pyright: ignore[reportMissingTypeStubs]

    @type_check_only
    class _DataArray(xr.DataArray):
        @property
        def metpy(self) -> metpy.MetPyDataArrayAccessor: ...

_Quantity = _spec.ureg.Quantity
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_T = TypeVar("_T", bound=np.number | npt.NDArray[np.number])


@overload
def quantity(
    # Using PlainQuantity for input annotations and NumpyQuantity for
    # output annotations
    # Use pint.Quantity directly once we require pint >=0.25.1
    value: _T | PlainQuantity[_T],
    units: str | None = ...,
) -> NumpyQuantity[_T]: ...
@overload
def quantity(
    # np.timedelta64 is a subclass of np.signedinteger at runtime,
    # but not in the type stubs
    value: np.ndarray[_ShapeT, np.dtype[np.timedelta64]],
    units: str | None = ...,
) -> NumpyQuantity[np.ndarray[_ShapeT, np.dtype[np.float64]]]: ...
@overload
def quantity(
    value: op.JustInt,
    units: str | None = ...,
) -> NumpyQuantity[np.int64]: ...
@overload
def quantity(
    value: op.JustFloat | datetime.timedelta | np.timedelta64,
    units: str | None = ...,
) -> NumpyQuantity[np.float64]: ...
@overload
def quantity(
    value: op.JustComplex,
    units: str | None = ...,
) -> NumpyQuantity[np.complex128]: ...
@overload
def quantity(value: xr.DataArray, units: str | None = ...) -> xr.DataArray: ...
def quantity(value: object, units: str | None = None) -> object:
    return _quantity(value, units)


@functools.singledispatch
def _numpy_quantity(value: object, units: str | None) -> PlainQuantity[Any]:
    del units
    message = f"Cannot convert {_utils.base_repr(value)} to quantity"
    raise TypeError(message)


@_numpy_quantity.register
def _(value: np.timedelta64, units: str | None) -> PlainQuantity[Any]:
    return _convert_timedelta(value, value, units)


@_numpy_quantity.register
def _(value: np.number, units: str | None) -> PlainQuantity[Any]:
    return _Quantity(value, units)


@_numpy_quantity.register(np.ndarray)
def _(value: npt.NDArray[Any], units: str | None) -> PlainQuantity[Any]:
    dtype = value.dtype
    sctype = dtype.type
    if issubclass(sctype, np.timedelta64):
        return _convert_timedelta(value, dtype, units)
    if issubclass(sctype, np.number):
        return _Quantity(value, units)
    message = f"{dtype} is unsupported"
    raise TypeError(message)


@functools.singledispatch
def _quantity(value: object, units: str | None) -> object:
    return _numpy_quantity(value, units)


@_quantity.register(op.JustInt)
def _(value: int, units: str | None) -> PlainQuantity[Any]:
    return _Quantity(np.int64(value), units)


@_quantity.register(op.JustFloat)
def _(value: float, units: str | None) -> PlainQuantity[Any]:
    return _Quantity(np.float64(value), units)


@_quantity.register(op.JustComplex)
def _(value: complex, units: str | None) -> PlainQuantity[Any]:
    return _Quantity(np.complex128(value), units)


@_quantity.register
def _(value: datetime.timedelta, units: str | None) -> PlainQuantity[Any]:
    # https://github.com/pandas-dev/pandas/issues/46819
    total_seconds = value / datetime.timedelta(seconds=1)
    quantity = _Quantity(total_seconds, "s")
    if units is not None:
        quantity.ito(units)  # pyright: ignore[reportUnknownMemberType]
    return quantity


@_quantity.register(PlainQuantity)
def _(value: PlainQuantity[Any], units: str | None) -> PlainQuantity[Any]:
    # https://github.com/hgrecco/pint/issues/2207
    if value._REGISTRY is not _spec.ureg:  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        message = "Cannot operate quantities of different registries"
        raise ValueError(message)
    value = _numpy_quantity(value.magnitude, value.units)
    if units is None:
        return value
    return value.to(units)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@_quantity.register(xr.DataArray)
def _(value: "_DataArray", units: str | None) -> xr.DataArray:
    if isinstance(value.variable, xr.IndexVariable):
        value = value._replace(value.variable.to_base_variable())  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]
    if "units" in value.attrs:
        value = cast("_DataArray", value.metpy.quantify())
    return value.copy(deep=False, data=_quantity(value.data, units))


def _convert_timedelta(
    value: np.timedelta64 | npt.NDArray[np.timedelta64],
    dtype: onp.AnyTimeDelta64DType,
    units: str | None,
) -> PlainQuantity[Any]:
    unit, count = np.datetime_data(dtype)
    raw_value = value / np.timedelta64(1, unit)  # Cast NaTs to NaNs
    match unit:
        case "generic":
            if units is None:
                # If a timedelta array is full of NaNs and zeros, the
                # unit becomes irrelevant
                # The fact cannot be generalized to other units like dB
                # and degC
                mask = np.isnan(raw_value)
                mask |= np.logical_not(raw_value)  # i.e. raw_value == 0
                if mask.all():
                    unit = "s"
            else:
                unit = units
                count = 1
                units = None
        case "M":
            # There is no such unit called `gregorian_month` in Pint
            unit = "day"
            count *= 30.436875  # Same as NumPy
        case _:
            unit = _time_unit_map.get(unit, unit)
    raw_value *= count
    quantity = _Quantity(raw_value, unit)
    if units is not None:
        quantity.ito(units)  # pyright: ignore[reportUnknownMemberType]
    return quantity


_time_unit_map = {"Y": "gregorian_year", "W": "week", "D": "d", "m": "min"}
