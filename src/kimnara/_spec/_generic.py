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

__all__ = ["NoneType", "TupleType", "TypingContext", "UnionType"]

import abc
import builtins
import sys
import types
from typing import Literal, get_args, no_type_check

import numba  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import Any, NamedTuple, TypeForm, override
from typing_inspection import introspection

import kimnara as kn
from kimnara import _spec


class TypingContext(NamedTuple):
    align: "kn.Alignment"
    allow_align: frozenset["kn.Alignment"] = frozenset()
    allow_array: bool = False
    allow_none: bool = False
    allow_optional: bool = False
    allow_scalar: bool = False
    allow_tuple: bool | Literal["nested"] = False
    allow_units: bool = False
    ndim: int | None = None
    pad_value: complex | None = None
    readonly: bool = False

    @abc.abstractmethod
    def infer(self, annotation: object) -> _spec.Type:
        raise NotImplementedError

    @staticmethod
    def parse(annotation: object) -> introspection.InspectedAnnotation:
        inspected = introspection.inspect_annotation(
            annotation,
            annotation_source=introspection.AnnotationSource.BARE,
            unpack_type_aliases="eager",
        )
        if expanded := _expand_generic_alias(inspected.type):
            return inspected._replace(type=expanded)
        return inspected

    def parse_args(
        self,
        annotation: object,
    ) -> tuple[introspection.InspectedAnnotation, ...]:
        return tuple(map(self.parse, get_args(annotation)))


class NoneType(_spec.Type):
    __slots__ = ()

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_none:
            message = "None is unsupported in this context"
            raise kn.TypeInferenceError(message)

    @override
    def to_numba(self) -> numba.types.NoneType:
        return numba.types.none

    @override
    @no_type_check
    def to_python(self) -> TypeForm[None]:
        pass


class TupleType(_spec.Type):
    __slots__ = ("_args",)

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        match ctx.allow_tuple:
            case True:
                ctx = ctx._replace(allow_tuple=False, allow_none=False)
            case False:
                message = "tuple is unsupported in this context"
                raise kn.TypeInferenceError(message)
            case _:
                ctx = ctx._replace(allow_none=False)
        match get_args(annotation.type):
            case [_, builtins.Ellipsis]:
                message = "tuple[T, ...] is unsupported in this context"
                raise kn.TypeInferenceError(message)
            case [()] if sys.version_info < (3, 11):
                # https://github.com/python/cpython/issues/91137
                self._args = ()
            case args:
                self._args = tuple(map(ctx.infer, args))

    @override
    def to_numba(self) -> numba.types.Tuple | numba.types.UniTuple:
        return numba.types.Tuple(self._args)

    @override
    def to_python(self) -> TypeForm[tuple[Any, ...]]:
        return tuple[self._args]


class UnionType(_spec.Type):
    __slots__ = ("_type",)

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_optional:
            message = "Optional[T] is unsupported in this context"
            raise kn.TypeInferenceError(message)
        t = None
        none = False
        for arg in map(
            ctx._replace(allow_optional=False, allow_none=True).infer,
            get_args(annotation.type),
        ):
            if isinstance(arg, NoneType):
                none = True
            elif t:
                message = "Cannot type Union that is not an Optional"
                raise kn.TypeInferenceError(message)
            else:
                t = arg
        if not (t and none):
            message = "Cannot type Union that is not an Optional"
            raise kn.TypeInferenceError(message)
        self._type = t

    @override
    def to_numba(self) -> numba.types.Optional:
        return numba.types.Optional(self._type.to_numba())

    @override
    @no_type_check
    def to_python(self) -> TypeForm[Any]:
        return None | self._type.to_python()


def _expand_generic_alias(annotation: object) -> types.GenericAlias | None:
    if isinstance(annotation, types.GenericAlias):
        try:
            value = annotation.__value__
        except AttributeError:
            pass
        else:
            return value[get_args(annotation)]
    return None
