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

__all__ = [
    "FunctionType",
    "NeverType",
    "NoneType",
    "OptionalType",
    "PointerType",
    "TupleType",
    "TypingContext",
]

import abc
import builtins
import ctypes
from collections.abc import Iterable, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Literal,
    NamedTuple,
    NoReturn,
    Self,
    cast,
    get_args,
    get_origin,
    no_type_check,
    override,
)

import numba.core.types  # pyright: ignore[reportMissingTypeStubs]
import pint
from typing_extensions import (  # ruff: ignore[deprecated-import]
    Sentinel,
    TypeAliasType,
    TypeForm,
    evaluate_forward_ref,
)
from typing_inspection import introspection, typing_objects

import kimnara as kn
from kimnara import _spec, _utils

if TYPE_CHECKING:
    from _typeshed import SupportsGetItem


class TypingContext(NamedTuple):
    align: "kn.Alignment"
    allow_align: frozenset["kn.Alignment"] = frozenset()
    allow_array: bool = False
    allow_never: bool = False
    allow_none: bool = False
    allow_optional: bool | Literal["units"] = False
    allow_pointer: bool = False
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

    def infer_with_diagnostics(
        self,
        annotation: object,
        template: str = "Failed to infer return type: {0}",
        *args: object,
    ) -> _spec.Type:
        try:
            return self.infer(annotation)
        except kn.TypeInferenceError as e:
            e.args = (template.format(*args, e),)
            raise

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


class FunctionType(_spec.Type):
    __slots__ = ("_argtypes", "_restype")

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_pointer:
            message = "Pointer is unsupported in this context"
            raise kn.TypeInferenceError(message)
        argtypes, restype = get_args(annotation.type)
        ctx = _spec.TypingContext(
            align=kn.A,
            allow_pointer=True,
            allow_scalar=True,
        )
        self._argtypes = tuple(map(ctx.infer, argtypes))
        self._restype = ctx._replace(allow_none=True).infer(restype)

    @override
    def to_ctypes(self) -> type["ctypes._CFunctionType"]:  # pyright: ignore[reportPrivateUsage]
        return ctypes.CFUNCTYPE(
            self._restype.to_ctypes(),
            *[t.to_ctypes() or _utils.unreachable() for t in self._argtypes],
        )

    @override
    def to_numba(self) -> numba.core.types.ExternalFunctionPointer:
        sig = self._restype.to_numba()(*[t.to_numba() for t in self._argtypes])  # pyright: ignore[reportUnknownVariableType]
        return numba.core.types.ExternalFunctionPointer(sig, None)  # pyright: ignore[reportUnknownArgumentType]

    @override
    def to_python(self) -> NoReturn:
        raise NotImplementedError


class NeverType(_spec.Type):
    __slots__ = ()

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_never:
            message = "Never and NoReturn are unsupported in this context"
            raise kn.TypeInferenceError(message)

    @override
    def to_ctypes(self) -> None:
        pass

    @override
    def to_numba(self) -> NoReturn:
        raise NotImplementedError

    @override
    def to_python(self) -> NoReturn:
        raise NotImplementedError


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
    @no_type_check
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


class OptionalType(_spec.Type):
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


class PointerType(_spec.Type):
    __slots__ = ("pointee", "readonly")

    @override
    def __init__(
        self,
        annotation: introspection.InspectedAnnotation,
        ctx: TypingContext,
    ) -> None:
        if not ctx.allow_pointer:
            message = "Pointer is unsupported in this context"
            raise kn.TypeInferenceError(message)
        tp = annotation.type
        if get_origin(tp) is kn.typing.CPointer:
            self._from_cpointer(tp)
        else:
            self._from_union(annotation)

    @override
    def to_ctypes(self) -> type["ctypes._PointerLike"]:  # pyright: ignore[reportPrivateUsage]
        return ctypes.POINTER(self.pointee.to_ctypes())

    @override
    def to_numba(
        self,
    ) -> numba.core.types.CPointer | numba.core.types.RawPointer:
        dtype = self.pointee
        if isinstance(dtype, NeverType):
            return numba.core.types.voidptr
        ptr = numba.core.types.CPointer(dtype.to_numba())
        if self.readonly:
            ptr.mutable = False
        return ptr

    @override
    def to_python(self) -> NoReturn:
        raise NotImplementedError

    def _from_cpointer(self, tp: object) -> None:
        ctx = _spec.TypingContext(
            align=kn.A,
            allow_never=True,
            allow_pointer=True,
            allow_scalar=True,
        )
        [arg] = ctx.parse_args(tp)
        for x in arg.metadata:
            if x is _spec.MUTABLE:
                self.readonly = False
                break
        else:
            self.readonly = True
        self.pointee = ctx.infer(arg.type)

    def _from_union(
        self,
        annotation: introspection.InspectedAnnotation,
    ) -> None:
        ctx = _spec.TypingContext(align=kn.A, allow_pointer=True)
        dtype = None
        self.readonly = True
        for arg in map(ctx.infer, get_args(annotation.type)):
            match arg:
                case FunctionType():
                    message = "Cannot type Union that is not an Optional"
                    raise kn.TypeInferenceError(message)
                case PointerType():
                    if not arg.readonly:
                        self.readonly = False
                    pointee = arg.pointee
                    if not isinstance(pointee, NeverType):
                        if dtype is None:
                            dtype = pointee
                        elif dtype != pointee:
                            message = (
                                f"Types conflict: CPointer[{dtype.to_numba()}]"
                                f", CPointer[{pointee.to_numba()}]"
                            )
                            raise kn.TypeInferenceError(message)
                case _:
                    _utils.unreachable()
        # `kn.typing.CPointer` is covariant, so
        # `CPointer[T] | CPointer[Never]` is equivalent to `CPointer[T]`
        self.pointee = dtype or NeverType(
            annotation,
            ctx._replace(allow_never=True),
        )


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
