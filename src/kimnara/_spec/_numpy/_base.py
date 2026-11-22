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

__all__ = ["ScalarType", "Type", "parse_units"]

import ctypes
import sys
from collections.abc import Sequence
from typing import Annotated, Any, no_type_check, override

import annotated_types as at
import numba.core.types  # pyright: ignore[reportMissingTypeStubs]
import numba.np.numpy_support  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pint
import pydantic
from typing_extensions import TypeForm
from typing_inspection import introspection

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._spec import _generic
from kimnara._typing import Scalar

from . import _units


class Type(_spec.Type):
    __slots__ = ("dequantifier", "dtype")
    dequantifier: _units.BaseDequantifier[Any]
    dtype: type[Scalar]

    def empty(self, shape: Sequence[int]) -> npt.NDArray[Scalar]:
        return np.empty(shape, self.dtype)

    @override
    def to_python(self) -> TypeForm[Any]:
        return Annotated[Any, pydantic.PlainValidator(self.dequantifier)]

    @override
    def to_units(self) -> pint.Unit | None:
        return self.dequantifier.unit


class ScalarType(Type):
    __slots__ = ()

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: _generic.TypingContext,
    ) -> None:
        if not ctx.allow_scalar:
            message = "NumPy scalar is unsupported in this context"
            raise kn.TypeInferenceError(message)
        match ctx.next_ndim():
            case int(ndim) if ndim:
                message = f"ndim conflict: {ndim}, 0"
                raise kn.TypeInferenceError(message)
            case _:
                pass
        unit = parse_units(annotation, ctx)
        self.dtype = dtype = unit.dtype(annotation.type)
        self.dequantifier = unit.dequantifier(kn.A, dtype, prefer_scalar=True)

    @override
    def to_ctypes(self) -> type["ctypes._SimpleCData[Any]"]:  # pyright: ignore[reportPrivateUsage]
        dtype = self.dtype
        if issubclass(dtype, np.complexfloating):
            if sys.version_info < (3, 14):
                message = "Complex numbers require Python 3.14 or later"
                raise kn.TypeInferenceError(message)
            match dtype:
                case np.complex64:
                    return ctypes.c_float_complex
                case np.complex128:
                    return ctypes.c_double_complex
                case _:
                    _utils.unreachable()
        return np.ctypeslib.as_ctypes_type(dtype)

    @override
    @no_type_check
    def to_numba(self) -> numba.core.types.Boolean | numba.core.types.Number:
        return numba.np.numpy_support.from_dtype(self.dtype)


def parse_units(
    annotation: introspection.InspectedAnnotation,
    ctx: _generic.TypingContext,
) -> _units.BaseUnit[Any]:
    if found := {
        _parse_units(x.unit)
        for x in annotation.metadata
        if isinstance(x, at.Unit)
    }:
        if not ctx.allow_units:
            message = "Units are unsupported in this context"
            raise kn.TypeInferenceError(message)
        if len(found) > 1:
            flat = ", ".join(map(str, found))
            message = f"Units conflict: {flat}"
            raise kn.TypeInferenceError(message)
        unit = found.pop()

        # Fast path:
        # https://github.com/hgrecco/pint/blob/0.25.3/pint/facets/plain/unit.py#L152-L153
        quantity = unit * 1

        if quantity._is_multiplicative:  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
            return _units.MultiplicativeUnit(unit)
        return _units.NonMultiplicativeUnit(unit)
    return _units.UnitNaive()


_parse_units = _spec.ureg.parse_units
