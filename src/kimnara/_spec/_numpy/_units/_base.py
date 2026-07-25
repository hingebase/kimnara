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

__all__ = ["BaseDequantifier", "BaseUnit"]

import abc
import datetime
import functools
from typing import Generic, TypeGuard

import numpy as np
import numpy.typing as npt
import xarray as xr
from pint.facets.numpy.quantity import NumpyQuantity
from typing_extensions import Any

import kimnara as kn
from kimnara import _utils
from kimnara._typing import SCT, ArrayLike, NumericT


class BaseDequantifier(abc.ABC, Generic[SCT]):
    __slots__ = ()

    def __call__(self, value: object) -> ArrayLike[SCT]:
        return self.dequantify(_quantify(value))

    @abc.abstractmethod
    def dequantify(self, value: object) -> ArrayLike[SCT]:
        raise NotImplementedError


class BaseUnit(abc.ABC, Generic[SCT]):
    @abc.abstractmethod
    def dequantifier(
        self,
        align: "kn.Alignment",
        dtype: type[SCT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = True,
    ) -> BaseDequantifier[SCT]:
        raise NotImplementedError

    def dtype(self, annotation: object) -> type[np.number[Any] | np.bool_]:
        match annotation:
            case np.bool_:
                if self.has_unit():
                    message = "Boolean values with units are prohibited"
                    raise kn.TypeInferenceError(message)
                return annotation
            case (
                np.int8 | np.int16 | np.int32 | np.int64
                | np.uint8 | np.uint16 | np.uint32 | np.uint64
                | np.float32 | np.float64 | np.complex64 | np.complex128
            ):
                return annotation
            case np.intc | np.uintc:
                return np.iinfo(annotation).dtype.type
            case _:
                if (
                    _utils.isclass(annotation)
                    and issubclass(annotation, np.generic)
                ):
                    message = f"Dtype {annotation} is unsupported"
                else:
                    message = "A numpy scalar type is required"
                raise kn.TypeInferenceError(message)

    @abc.abstractmethod
    def has_unit(self) -> bool:
        raise NotImplementedError


def _is_timedelta_array(
    value: npt.NDArray[Any],
) -> TypeGuard[npt.NDArray[np.timedelta64]]:
    return np.issubdtype(value.dtype, np.timedelta64)


@functools.singledispatch
def _quantify(value: object) -> object:
    return value


@_quantify.register(datetime.timedelta)
@_quantify.register(np.timedelta64)
def _(value: datetime.timedelta | np.timedelta64) -> NumpyQuantity[np.float64]:
    return kn.quantity(value)


@_quantify.register(np.ndarray)
def _(value: npt.NDArray[Any]) -> object:
    return kn.quantity(value) if _is_timedelta_array(value) else value


@_quantify.register(NumpyQuantity)
def _(value: NumpyQuantity[NumericT]) -> NumpyQuantity[NumericT]:
    return kn.quantity(value)


@_quantify.register
def _(value: xr.DataArray) -> object:
    del value
    message = (
        "Don't pass `xarray.DataArray` directly to Kimnara functions. "
        "Instead, use `xarray.apply_ufunc` to unwrap it."
    )
    raise TypeError(message)
