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

__all__ = ["ScalarType", "Type", "dtype", "scalar_type"]

import abc
import functools
from typing import Annotated, cast, no_type_check

import annotated_types as at
import numba.np.numpy_support  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pint
import pydantic
from typing_extensions import Any, TypedDict, TypeForm, override
from typing_inspection import introspection

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._spec import _generic

_Unit = _spec.ureg.Unit


class Type(_spec.Type):
    __slots__ = ("dtype", "unit")
    unit: pint.Unit | None

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: _generic.TypingContext,
    ) -> None:
        if found := {
            _Unit(x.unit)
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
            self.unit = found.pop()
        else:
            self.unit = None

    @override
    @no_type_check
    def to_python(self) -> TypeForm[Any]:
        return Annotated[Any, pydantic.PlainValidator(self.validator)]

    @abc.abstractmethod
    def validator(self, value: object) -> object:
        raise NotImplementedError


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
        super().__init__(annotation, ctx)
        self.dtype = scalar_type(annotation.type, bool(self.unit))

    @override
    @no_type_check
    def to_numba(self) -> numba.types.Boolean | numba.types.Number:
        return numba.np.numpy_support.from_dtype(self.dtype)

    @override
    def validator(self, value: object) -> np.number | np.bool_:
        raise NotImplementedError


def dtype(value: object) -> np.dtype[Any] | None:
    # https://numpy.org/doc/stable/user/basics.interoperability.html#dtype-interoperability
    # https://github.com/numpy/numpy/issues/14331
    for attr in "__numpy_dtype__", "dtype", "__array_interface__":
        try:
            obj = getattr(value, attr)
        except AttributeError:  # noqa: PERF203
            pass
        else:
            if attr == "__array_interface__":
                obj = _adapter.validate_python(obj)
                return np.dtype(obj["typestr"])
            if isinstance(obj, np.dtype):
                return cast("np.dtype[Any]", obj)
            if attr == "__numpy_dtype__":
                # https://github.com/numpy/numpy/blob/v2.4.0/numpy/_core/tests/test_dtype.py#L1645-L1653
                message = (
                    "`.__numpy_dtype__` must return a NumPy dtype instance, "
                    f"got {_utils.base_repr(value)}"
                )
                raise TypeError(message)
    return None


@functools.lru_cache(maxsize=30)
def scalar_type(
    annotation: object,
    has_unit: bool,  # noqa: FBT001
) -> type[np.number | np.bool_]:
    match annotation:
        case np.bool_:
            if has_unit:
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


class _ArrayInterface(TypedDict, closed=False):
    typestr: str


_adapter = pydantic.TypeAdapter(_ArrayInterface, config={"strict": True})
