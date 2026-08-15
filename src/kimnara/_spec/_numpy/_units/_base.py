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

__all__ = ["BaseDequantifier", "BaseUnit", "is_quantity"]

import abc
import datetime
import functools
from typing import Generic, TypeGuard

import numpy as np
import optype.numpy as onp
import pint
import pydantic_core
import xarray as xr
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.numpy.quantity import NumpyQuantity

import kimnara as kn
from kimnara import _utils
from kimnara._typing import SCT, ArrayLike, Number, NumericT, Scalar


class BaseDequantifier(_utils.EqMixIn, abc.ABC, Generic[SCT]):
    __slots__ = ()

    def __call__(self, value: object) -> ArrayLike[SCT]:
        try:
            return self.dequantify(_quantify(value))
        except FloatingPointError as e:
            match e.args:
                case [str(msg)] if msg.startswith((
                    "invalid value encountered in round_to_",
                    "invalid value encountered in scale_to_",
                )):
                    pass
                case _:
                    raise
            match np.dtype(msg[38:]).kind:
                case "i":
                    error_type = "int_type"
                case "f":
                    error_type = "float_type"
                case "c":
                    error_type = "complex_type"
                case _:
                    _utils.unreachable()
            raise pydantic_core.PydanticCustomError(
                error_type,
                "{message}",
                {"message": msg},
            ) from None
        except ValueError as e:
            if isinstance(e, pydantic_core.PydanticCustomError):
                raise
            # Cleaner error message than raising ValueError directly
            error_type = "value_error"
            raise pydantic_core.PydanticCustomError(
                error_type,
                "{message}",
                {"message": e},
            ) from None
        except pint.DimensionalityError as e:
            error_type = "value_error"
            raise pydantic_core.PydanticCustomError(
                error_type,
                "Cannot convert from '{units1}'{dim1} to "
                    "'{units2}'{dim2}{extra_msg}",
                {
                    "units1": e.units1,
                    "units2": e.units2,
                    "dim1": f" ({e.dim1})" if e.dim1 else None,
                    "dim2": f" ({e.dim2})" if e.dim2 else None,
                    "extra_msg": e.extra_msg,
                },
            ) from None

    @abc.abstractmethod
    def dequantify(self, value: object) -> ArrayLike[SCT]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def unit(self) -> pint.Unit | None:
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

    def dtype(self, annotation: object) -> type[Scalar]:
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


def is_quantity(
    x: object,
    /,
) -> TypeGuard[NonMultiplicativeQuantity[ArrayLike[Number]]]:
    # The type of the magnitude has been forced by `_quantify`
    return isinstance(x, pint.Quantity)  # pyright: ignore[reportUnknownMemberType]


@functools.singledispatch
def _quantify(value: object) -> object:
    if onp.is_array_nd(value, np.timedelta64):
        return kn.quantity(value)
    return value


@_quantify.register(datetime.timedelta)
@_quantify.register(np.timedelta64)
def _(value: datetime.timedelta | np.timedelta64) -> NumpyQuantity[np.float64]:
    return kn.quantity(value)


@_quantify.register(NumpyQuantity)
def _(value: NumpyQuantity[NumericT]) -> NumpyQuantity[NumericT]:
    return kn.quantity(value)


@_quantify.register
def _(value: xr.DataArray) -> object:
    del value
    error_type = "is_instance_of"
    message = (
        "Don't pass `xarray.DataArray` directly to Kimnara functions. "
        "Instead, use `xarray.apply_ufunc` to unwrap it."
    )
    raise pydantic_core.PydanticCustomError(error_type, message)
