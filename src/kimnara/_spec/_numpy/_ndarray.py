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

__all__ = ["TypingContext"]

import builtins
import sys
import typing
from typing import get_args, get_origin

import numba.np.numpy_support  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pint
from typing_extensions import Any, override
from typing_inspection import introspection, typing_objects

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._spec import _generic

from . import _base


class TypingContext(_generic.TypingContext):
    __slots__ = ()

    @override
    def infer(self, annotation: object) -> _spec.Type:  # noqa: C901
        inspected = self.parse(annotation)
        match inspected.type:
            case None:
                return _generic.NoneType(inspected, self)
            case builtins.bool:
                inspected = inspected._replace(type=np.bool_)
            case builtins.complex:
                inspected = inspected._replace(type=np.complex128)
            case builtins.float:
                inspected = inspected._replace(type=np.float64)
            case builtins.int:
                inspected = inspected._replace(type=np.int64)
            case typing.Tuple:  # noqa: UP006
                message = "Incomplete type: typing.Tuple without args"
                raise kn.TypeInferenceError(message)
            case _:
                pass
        tp = inspected.type
        if _utils.isclass(tp) and issubclass(tp, np.generic):
            return _base.ScalarType(inspected, self)
        match get_origin(tp):
            case builtins.tuple:
                return _generic.TupleType(inspected, self)
            case np.ndarray:
                return _ArrayType(inspected, self)
            case origin:
                if introspection.is_union_origin(origin):
                    return _generic.UnionType(inspected, self)
        message = f"Cannot infer the runtime type from {_utils.base_repr(tp)}"
        raise kn.TypeInferenceError(message)

    def get_align(self, meta: list[Any]) -> kn.Alignment:
        if found := {x for x in meta if isinstance(x, kn.Alignment)}:
            if len(found) > 1:
                flat = ", ".join(x._name_ for x in found)
                message = f"Alignments conflict: {flat}"
                raise kn.TypeInferenceError(message)
            align = found.pop()
            if align not in self.allow_align:
                message = (
                    f"Alignment {align._name_} is unsupported in this context"
                )
                raise kn.TypeInferenceError(message)
            return align
        return self.align

    def get_dtype(
        self,
        annotation: object,
        unit: pint.Unit | None,
    ) -> tuple[type[np.number | np.bool_], bool]:
        if get_origin(annotation) is not np.dtype:
            message = "The second argument of np.ndarray[] must be np.dtype[]"
            raise kn.TypeInferenceError(message)
        match self.parse_args(annotation):
            case []:
                message = "Dtype argument is missing"
                raise kn.TypeInferenceError(message)
            case [arg]:
                pass
            case _:
                message = "Too many dtype arguments"
                raise kn.TypeInferenceError(message)
        readonly = self.readonly
        for x in arg.metadata:
            if x is _spec.Mutable:
                readonly = False
        return _base.scalar_type(arg.type, bool(unit)), readonly

    def get_ndim(self, shape: object) -> int:
        if (
            typing_objects.is_any(shape)
            or shape is tuple
            or shape is typing.Tuple  # noqa: UP006
        ):
            return self._default_ndim()
        if get_origin(shape) is not tuple:
            message = "The first argument of np.ndarray[] must be tuple[]"
            raise kn.TypeInferenceError(message)
        match get_args(shape):
            case [arg, builtins.Ellipsis]:
                self._check_dim(0, self.parse(arg))
                return self._default_ndim()
            case [()] if sys.version_info < (3, 11):
                # https://github.com/python/cpython/issues/91137
                ndim = 0
            case args:
                for i, arg in enumerate(args):
                    self._check_dim(i, self.parse(arg))
                ndim = len(args)
        default = self.ndim
        if default is not None:
            if default != ndim:
                message = f"ndim conflict: {default}, {ndim}"
                raise kn.TypeInferenceError(message)
            return default
        return ndim

    def get_pad_value(self, meta: list[Any]) -> complex | None:
        if found := {x.value for x in meta if isinstance(x, kn.Pad)}:
            if len(found) > 1:
                flat = ", ".join(map(str, found))
                message = f"Padding values conflict: {flat}"
                raise kn.TypeInferenceError(message)
            return found.pop()
        return self.pad_value

    def _check_dim(
        self,
        i: int,
        annotation: introspection.InspectedAnnotation,
    ) -> None:
        arg = annotation.type
        if introspection.is_union_origin(get_origin(arg)):
            for x in self.parse_args(arg):
                self._check_dim(i, x)
            return
        if typing_objects.is_any(arg) or arg is int:
            return
        if not typing_objects.is_literal(get_origin(arg)):
            message = f"shape[{i}] is not integral"
            raise kn.TypeInferenceError(message)
        for x in get_args(arg):
            if type(x) is not int:
                message = f"shape[{i}] is not integral"
                raise kn.TypeInferenceError(message)
            if x < 0:
                message = f"shape[{i}] can be negative"
                raise kn.TypeInferenceError(message)

    def _default_ndim(self) -> int:
        if self.ndim is None:
            message = "ndim is unknown"
            raise kn.TypeInferenceError(message)
        return self.ndim


class _ArrayType(_base.Type):
    __slots__ = ("_align", "_ndim", "_pad_value", "_readonly")

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_array:
            message = "NumPy array is unsupported in this context"
            raise kn.TypeInferenceError(message)
        match ctx.parse_args(annotation.type):
            case (shape, dtype):
                pass
            case args:
                message = (
                    "np.ndarray[] should have exactly 2 arguments, got "
                    f"{len(args)}"
                )
                raise kn.TypeInferenceError(message)
        super().__init__(annotation, ctx)
        meta = annotation.metadata
        self._align = ctx.get_align(meta)
        self.dtype, self._readonly = ctx.get_dtype(dtype.type, self.unit)
        self._ndim = ctx.get_ndim(shape.type)
        self._pad_value = ctx.get_pad_value(meta)

    @override
    def to_numba(self) -> numba.types.Array:
        match self._align:
            case (kn.C | kn.F) as align:
                layout = align._name_
            case _:
                layout = "A"
        return numba.types.Array(
            numba.np.numpy_support.from_dtype(self.dtype),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            self._ndim,
            layout,
            self._readonly,
        )

    @override
    def validator(self, value: object) -> npt.NDArray[np.number | np.bool_]:
        raise NotImplementedError
