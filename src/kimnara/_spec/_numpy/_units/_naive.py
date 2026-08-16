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

__all__ = ["NaiveDequantifier", "UnitNaive"]

import functools
import math
from typing import Literal, NoReturn, TypeGuard, cast

import numpy as np
import numpy.typing as npt
import optype as op
import pint
import pydantic_core
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from typing_extensions import Any, override

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._spec._numpy import _units
from kimnara._typing import SCT, ArrayLike, ShapeT


class NaiveDequantifier(_units.BaseDequantifier[SCT]):
    __slots__ = ("_prefer_scalar", "_readonly", "align", "dtype", "pad_value")

    def __init__(
        self,
        align: "kn.Alignment",
        dtype: type[SCT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = False,
    ) -> None:
        self.align = align
        self.dtype = dtype
        self.pad_value = pad_value
        self._prefer_scalar = prefer_scalar
        self._readonly = readonly

    @override
    def dequantify(self, value: object) -> ArrayLike[SCT]:
        if _units.is_quantity(value):
            if _dimensional(value):
                error_type = "is_instance_of"
                message = "Expect plain numbers rather than quantities"
                raise pydantic_core.PydanticCustomError(error_type, message)
            value = value.magnitude
        return self.postprocess(self._dequantify(value), value)

    @property
    @override
    def unit(self) -> None:
        pass

    def control_output(
        self,
        value: ArrayLike[np.number[Any]],
    ) -> dict[str, object]:
        match align := self.align:
            case kn.A | kn.C | kn.F:
                return {"order": align._name_}
            case _:
                if not _utils.at_least_1d(value):
                    return {}
        out = kn.empty(
            value.shape,
            self.dtype,
            align=align,
            pad_value=self.pad_value,
        )
        return {"out": out}

    def dtype_match(
        self,
        value: ArrayLike[np.number[Any] | np.bool_, ShapeT],
    ) -> TypeGuard[ArrayLike[SCT, ShapeT]]:
        return value.dtype.type is self.dtype

    def postprocess(
        self,
        value: ArrayLike[SCT],
        original: object = None,
    ) -> ArrayLike[SCT]:
        if isinstance(value, np.ndarray):
            if self._prefer_scalar and value.ndim == 0:
                return value[()]
            if self._readonly:
                if value is original and value.flags.writeable:
                    value = value.view()
                value.flags.writeable = False
        return value

    @functools.singledispatchmethod
    def _dequantify(self, value: object) -> ArrayLike[SCT]:
        del self, value
        error_type = "is_instance_of"
        message = "Unsupported"
        raise pydantic_core.PydanticCustomError(error_type, message)

    @_dequantify.register(bool)
    @_dequantify.register(np.bool_)
    def _(self, value: bool | np.bool_) -> SCT:  # ruff: ignore[boolean-type-hint-positional-argument]
        return self.dtype(value)

    @_dequantify.register(op.JustInt)
    def _(self, value: int) -> SCT:
        dtype = self.dtype
        if dtype is np.bool_ and 0 != value != 1:
            _validation_error("scalar", int, dtype)
        return dtype(value)

    @_dequantify.register(op.JustFloat)
    def _(self, value: float) -> SCT:
        dtype = self.dtype
        if np.float64 is not dtype is not np.complex128:
            _validation_error("scalar", float, dtype)
        return dtype(value)

    @_dequantify.register(op.JustComplex)
    def _(self, value: complex) -> np.complex128:
        if self.dtype is not np.complex128:
            _validation_error("scalar", complex, self.dtype)
        return np.complex128(value)

    @_dequantify.register(np.number)
    def _(self, value: np.number[Any]) -> SCT:
        try:
            return value.astype(self.dtype, casting="safe")
        except TypeError:
            _validation_error("scalar", value.dtype, self.dtype)

    @_dequantify.register(np.ndarray)
    def _(self, value: npt.NDArray[Any]) -> npt.NDArray[SCT]:
        dtype = self.dtype
        if _utils.at_least_1d(value):
            if not np.can_cast(value, dtype, casting="safe"):
                _validation_error("array data", value.dtype, dtype)
            return kn.asarray(
                value,
                dtype,
                align=self.align,
                pad_value=self.pad_value,
            )
        try:
            return value.astype(
                dtype,
                order=self.align.order,
                casting="safe",
                subok=False,
                copy=not value.flags.aligned,
            )
        except TypeError:
            _validation_error("array data", value.dtype, dtype)


class UnitNaive(_units.BaseUnit[SCT]):
    @override
    def dequantifier(
        self,
        align: "kn.Alignment",
        dtype: type[SCT],
        pad_value: complex | None = None,
        *,
        prefer_scalar: bool = False,
        readonly: bool = True,
    ) -> NaiveDequantifier[SCT]:
        return NaiveDequantifier(
            align,
            dtype,
            pad_value,
            prefer_scalar=prefer_scalar,
            readonly=readonly,
        )

    @override
    def has_unit(self) -> bool:
        return False


def _dimensional(value: NonMultiplicativeQuantity[Any]) -> bool:
    if not value._is_multiplicative:  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
        return True
    try:
        scale = cast("float", value.units.m_from(_spec.dimensionless))  # pyright: ignore[reportUnknownMemberType]
    except pint.DimensionalityError:
        return True
    return not math.isclose(scale, 1.)


def _validation_error(
    obj: Literal["scalar", "array data"],
    from_: npt.DTypeLike,
    to: npt.DTypeLike,
) -> NoReturn:
    to = np.dtype(to)
    match to.kind:
        case "b":
            error_type = "bool_type"
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
        "Cannot cast {obj} from dtype('{from}') to dtype('{to}') according to "
            "the rule 'safe'",
        {"obj": obj, "from": np.dtype(from_), "to": to},
    )
