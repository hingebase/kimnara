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
from collections.abc import Iterable, Iterator
from typing import (
    TYPE_CHECKING,
    ForwardRef,
    Literal,
    cast,
    get_args,
    get_origin,
    no_type_check,
)

import numba.core.types  # pyright: ignore[reportMissingTypeStubs]
import pint
from typing_extensions import (
    Any,
    NamedTuple,
    Self,
    Sentinel,
    TypeAliasType,
    TypeForm,
    evaluate_forward_ref,
    override,
)
from typing_inspection import introspection, typing_objects

import kimnara as kn
from kimnara import _spec

if TYPE_CHECKING:
    from _typeshed import SupportsGetItem


class TypingContext(NamedTuple):
    align: "kn.Alignment"
    allow_align: frozenset["kn.Alignment"] = frozenset()
    allow_array: bool = False
    allow_none: bool = False
    allow_optional: bool | Literal["units"] = False
    allow_scalar: bool = False
    allow_tuple: bool | Literal["nested"] = False
    allow_units: bool = False

    # https://github.com/microsoft/pyright/issues/11115
    ndim: object = Sentinel("MISSING")

    pad_value: complex | None = None
    readonly: bool = False

    @abc.abstractmethod
    def infer(self, annotation: object) -> _spec.Type:
        raise NotImplementedError

    @staticmethod
    def make_tuple(members: Iterable[_spec.Type]) -> "TupleType":
        return TupleType.from_members(members)

    def next_ndim(self) -> object:
        ndim = self.ndim
        if isinstance(ndim, Iterator):
            return next(cast("Iterator[int]", ndim))
        return ndim

    @classmethod
    def parse(cls, annotation: object) -> introspection.InspectedAnnotation:
        # Can annotation be a ForwardRef?
        inspected = introspection.inspect_annotation(
            annotation,
            annotation_source=introspection.AnnotationSource.BARE,
            unpack_type_aliases="eager",
        )
        if annotation := _expand_type_alias(inspected.type):
            merged = cls.parse(annotation)
            merged.metadata.extend(inspected.metadata)
            return merged
        return inspected

    @classmethod
    def parse_args(
        cls,
        annotation: object,
    ) -> tuple[introspection.InspectedAnnotation, ...]:
        return tuple(map(cls.parse, get_args(annotation)))


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
    def to_ctypes(self) -> None:
        pass

    @override
    def to_numba(self) -> numba.core.types.NoneType:
        return numba.core.types.none

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

    @classmethod
    def from_members(cls, members: Iterable[_spec.Type]) -> Self:
        self = cls.__new__(cls)
        self._args = tuple(members)
        return self

    @override
    def __iter__(self) -> Iterator[_spec.Type]:
        return iter(self._args)

    @override
    def __len__(self) -> int:
        return len(self._args)

    @override
    def to_numba(self) -> numba.core.types.Tuple | numba.core.types.UniTuple:
        return numba.core.types.Tuple([arg.to_numba() for arg in self._args])

    @override
    def to_python(self) -> TypeForm[tuple[Any, ...]]:
        return tuple[tuple(arg.to_python() for arg in self._args)]

    @override
    def to_units(self) -> tuple[pint.Unit | None, ...] | None:
        units: list[pint.Unit | None] = []
        for arg in self._args:
            unit = arg.to_units()
            if isinstance(unit, tuple):
                message = "Nested tuples cannot have units"
                raise kn.TypeInferenceError(message)
            units.append(unit)
        return tuple(units) if any(units) else None


class UnionType(_spec.Type):
    __slots__ = ("_type",)

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        match ctx.allow_optional:
            case True:
                ctx = ctx._replace(
                    allow_optional=False,
                    allow_none=True,
                    allow_units=False,

                    # Allowing `tuple[...] | None` would make a mess in
                    # Python typing, although the runtime behavior is
                    # well-defined
                    # The workaround is `tuple[T1 | None, T2 | None]`
                    allow_tuple=False,
                )
            case False:
                message = "Optional[T] is unsupported in this context"
                raise kn.TypeInferenceError(message)
            case _:
                ctx = ctx._replace(
                    allow_optional=False,
                    allow_none=True,
                    allow_tuple=False,
                )
        t = None
        none = False
        for arg in map(ctx.infer, get_args(annotation.type)):
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
    def to_numba(self) -> numba.core.types.Optional:
        return numba.core.types.Optional(self._type.to_numba())

    @override
    @no_type_check
    def to_python(self) -> TypeForm[Any]:
        return None | self._type.to_python()

    # `.to_units()` is only called for return types, where
    # optional types cannot have units


def _expand_type_alias(annotation: object) -> object:
    if typing_objects.is_typealiastype(annotation):
        return _get_value(annotation)
    origin = get_origin(annotation)
    if not typing_objects.is_typealiastype(origin):
        return None
    value = _get_value(origin)
    try:
        return value[get_args(annotation)]
    except TypeError:
        return value


def _get_value(tat: TypeAliasType) -> "SupportsGetItem[tuple[Any, ...], Any]":
    value = tat.__value__
    if isinstance(value, str):
        value = evaluate_forward_ref(
            ForwardRef(value, module=tat.__module__),
            type_params=tat.__type_params__,  # pyright: ignore[reportArgumentType]
        )
    return value
