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

__all__ = ["NumbaParallelUFunc", "NumbaUFunc", "Output", "Outputs", "PyUFunc"]

import abc
import inspect
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    SupportsIndex,
    Unpack,
    cast,
    overload,
    override,
)

import numba  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numba.np.ufunc import dufunc, ufuncbuilder  # pyright: ignore[reportMissingTypeStubs]
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import Sentinel, TypeVar

import kimnara as kn
from kimnara import _spec
from kimnara._typing import (
    ArrayLike,
    FastMathOptions,
    Input,
    Number,
    Numeric,
    Scalar,
    UFuncKwargs,
)
from kimnara._typing import Output as _Output
from kimnara._typing import Outputs as _Outputs

from . import _common

if TYPE_CHECKING:
    from kimnara._spec._numpy._base import ScalarType

type Output = op.JustInt | op.JustFloat | op.JustComplex | Number
type Outputs = tuple[complex | Scalar, ...]
type _InputAtLeast1D = (
    np.ndarray[onp.AtLeast1D, np.dtype[Scalar]]
    | PlainQuantity[np.ndarray[onp.AtLeast1D, np.dtype[Number]]]
)
type _MaybeQuantity[T: Numeric] = T | NumpyQuantity[T]
_ShapeT = TypeVar(
    "_ShapeT",
    tuple[int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, *tuple[int, ...]],
    tuple[int, int, int, *tuple[int, ...]],
    tuple[int, int, *tuple[int, ...]],
    tuple[int, *tuple[int, ...]],
)
_BooleanOutputT = TypeVar("_BooleanOutputT", bound=onp.ToJustBool)
_OutputT = TypeVar("_OutputT", bound=Output)

MISSING = Sentinel("MISSING")


class _UFunc[T](_common.BaseUFuncWrapper[T]):
    @property
    @override
    def signature(self) -> None:
        pass

    @overload
    def __call__[O: onp.ToJustBool](
        self: "_UFunc[O]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def __call__[O: Output](
        self: "_UFunc[O]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> _MaybeQuantity[ArrayLike[Number]]: ...

    @overload
    def __call__[O: Outputs](
        self: "_UFunc[O]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> tuple[_Output, ...]: ...

    @override
    def __call__(
        self,
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> _Outputs:
        wrapper = self.wrap_call(**kwargs)
        return self.validate_call(wrapper)(*args)

    @overload
    def accumulate[O: onp.ToJustBool, ShapeT: onp.AtLeast1D](
        self: "_UFunc[O]",
        array: np.ndarray[ShapeT, np.dtype[np.bool_]],
        /,
        axis: SupportsIndex = ...,
    ) -> np.ndarray[ShapeT, np.dtype[np.bool_]]: ...

    @overload
    def accumulate[O: Output, ShapeT: onp.AtLeast1D](
        self: "_UFunc[O]",
        array: np.ndarray[ShapeT, np.dtype[Number]]
            | PlainQuantity[np.ndarray[ShapeT, np.dtype[Number]]],
        /,
        axis: SupportsIndex = ...,
    ) -> _MaybeQuantity[np.ndarray[ShapeT, np.dtype[Number]]]: ...

    def accumulate(
        self: "_UFunc[Any]",
        array: _InputAtLeast1D,
        /,
        axis: SupportsIndex = 0,
    ) -> object:
        wrapper = self.wrap_accumulate(axis)
        self._calling_method = "accumulate"
        return self.validate_call(wrapper, alternative=True)(array)

    @overload
    def reduce[O: onp.ToJustBool](
        self: "_UFunc[O]",
        array: onp.ToJustBool | npt.NDArray[np.bool_],
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        *,
        keepdims: bool = ...,
        initial: object = ...,
        where: onp.ToJustBool | onp.ToJustBoolND | None = ...,
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def reduce[O: Output](
        self: "_UFunc[O]",
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        *,
        keepdims: bool = ...,
        initial: object = ...,
        where: onp.ToJustBool | onp.ToJustBoolND | None = ...,
    ) -> _MaybeQuantity[ArrayLike[Number]]: ...

    def reduce(
        self,
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = 0,
        *,
        keepdims: bool = False,
        initial: object = MISSING,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
    ) -> object:
        wrapper = self.wrap_reduce(axis, initial, where, keepdims=keepdims)
        self._calling_method = "reduce"
        return self.validate_call(wrapper, alternative=True)(array)

    @overload
    def reduceat(
        self: "_UFunc[_BooleanOutputT]",
        array: np.ndarray[_ShapeT, np.dtype[np.bool_]],
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = ...,
    ) -> np.ndarray[_ShapeT, np.dtype[np.bool_]]: ...

    @overload
    def reduceat(
        self: "_UFunc[_OutputT]",
        array: np.ndarray[_ShapeT, np.dtype[Number]]
            | PlainQuantity[np.ndarray[_ShapeT, np.dtype[Number]]],
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = ...,
    ) -> _MaybeQuantity[np.ndarray[_ShapeT, np.dtype[Number]]]: ...

    def reduceat(
        self,
        array: _InputAtLeast1D,
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = 0,
    ) -> object:
        wrapper = self.wrap_reduceat(indices, axis)
        self._calling_method = "reduceat"
        return self.validate_call(wrapper, alternative=True)(array)

    @overload
    def outer[O: onp.ToJustBool](
        self: "_UFunc[O]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def outer[O: Output](
        self: "_UFunc[O]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> _MaybeQuantity[ArrayLike[Number]]: ...

    @overload
    def outer[O: Outputs](
        self: "_UFunc[O]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> tuple[_Output, ...]: ...

    def outer(
        self,
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> object:
        if self._ufunc.nin != 2:  # ruff: ignore[magic-value-comparison]
            message = "outer product only supported for binary functions"
            raise ValueError(message) from None
        wrapper = self.wrap_outer(**kwargs)
        return self.validate_call(wrapper)(a, b)

    @override
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_scalar=True,
            allow_units=True,
        )

    @abc.abstractmethod
    def wrap_accumulate(self, axis: SupportsIndex) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_call(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_reduce(
        self,
        axis: SupportsIndex | Sequence[SupportsIndex] | None,
        initial: object,
        where: onp.ToJustBool | onp.ToJustBoolND | None,
        *,
        keepdims: bool = False,
    ) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_reduceat(
        self,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex,
    ) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_outer(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[..., Any]:
        raise NotImplementedError

    @override
    def alternative_signature(self) -> inspect.Signature:
        match self.argtypes:
            case [x1, x2] if x1 == x2 == self.restype:
                pass
            case _:
                message = (
                    f"{self._calling_method} only supported for binary "
                    "functions returning a single value of the same kind"
                )
                raise ValueError(message)
        return_annotation = self.restype.to_python()
        arg = inspect.Parameter(
            "array",
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=return_annotation,
        )
        return inspect.Signature([arg], return_annotation=return_annotation)


class _NumbaUFuncBase[T, N](_UFunc[T], _common.UFuncWrapper[T]):
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        wrapped: Callable[..., T],
        *,
        boundscheck: bool | None = None,
        cache: bool | None = None,
        fastmath: FastMathOptions = False,
        identity: Literal[0, 1, "reorderable"] | None = None,
        parallel: N = False,
    ) -> None:
        self.infer(wrapped)
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = self.restype.to_numba()
        decorator = numba.vectorize(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [restype(*argtypes)],
            boundscheck=boundscheck,
            cache=_common.can_cache(wrapped, cache=cache),
            fastmath=fastmath,
            identity=identity,
            target="parallel" if parallel else "cpu",
        )
        self.build(decorator, parallel)  # pyright: ignore[reportUnknownArgumentType]

    @property
    @override
    def nout(self) -> Literal[1]:
        return 1

    @override
    def wrap_accumulate(
        self,
        axis: SupportsIndex,
    ) -> Callable[[npt.NDArray[Scalar]], npt.NDArray[Scalar]]:
        return lambda array: self._ufunc.accumulate(array, axis)

    @override
    def wrap_call(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[
        [Unpack[tuple[ArrayLike[Scalar], ...]]],
        ArrayLike[Scalar] | tuple[ArrayLike[Scalar], ...],
    ]:
        return lambda *args: self._ufunc(*args, **kwargs)

    @override
    def wrap_reduce(
        self,
        axis: SupportsIndex | Sequence[SupportsIndex] | None,
        initial: object,
        where: onp.ToJustBool | onp.ToJustBoolND | None,
        *,
        keepdims: bool = False,
    ) -> Callable[[ArrayLike[Scalar]], ArrayLike[Scalar]]:
        if initial is MISSING:
            return lambda array: self._ufunc.reduce(
                array,
                axis=axis,
                keepdims=keepdims,
                where=where,
            )
        return lambda array: self._ufunc.reduce(
            array,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @override
    def wrap_reduceat(
        self,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex,
    ) -> Callable[[npt.NDArray[Scalar]], npt.NDArray[Scalar]]:
        return lambda array: self._ufunc.reduceat(array, indices, axis)  # pyright: ignore[reportArgumentType]

    @override
    def wrap_outer(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[
        [ArrayLike[Scalar], ArrayLike[Scalar]],
        ArrayLike[Scalar] | tuple[ArrayLike[Scalar], ...],
    ]:
        return lambda a, b: self._ufunc.outer(a, b, **kwargs)

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_scalar=True,
            allow_units=True,
        )

    @abc.abstractmethod
    def build(
        self,
        decorator: Callable[[Callable[..., T]], Any],
        name: N,
    ) -> None:
        raise NotImplementedError


class NumbaUFunc[T](_NumbaUFuncBase[T, Literal[False]], _common.Dispatchable):
    @override
    def build(
        self,
        decorator: Callable[[Callable[..., T]], dufunc.DUFunc],
        name: Literal[False],
    ) -> None:
        ufunc = decorator(self.__wrapped__)
        self.dispatcher = cast(
            "ufuncbuilder.UFuncDispatcher",
            ufunc._dispatcher,  # pyright: ignore[reportAttributeAccessIssue]  # ruff: ignore[private-member-access]
        )
        self._ufunc = cast("np.ufunc", ufunc.ufunc)


class NumbaParallelUFunc[T](_NumbaUFuncBase[T, str | Literal[True]]):
    @override
    def build(
        self,
        decorator: Callable[[Callable[..., T]], np.ufunc],
        name: str | Literal[True],
    ) -> None:
        with kn.threading.using_backend(name):
            self._ufunc = decorator(self.__wrapped__)


class PyUFunc[T](_UFunc[T]):
    def __init__(
        self,
        wrapped: Callable[..., T],
        *,
        identity: Literal["reorderable"] | None = None,
    ) -> None:
        self.infer(wrapped)
        nin = len(self.argtypes)
        nout = len(self.restype)
        if identity:
            self._ufunc = np.frompyfunc(wrapped, nin, nout, identity=None)
        else:
            self._ufunc = np.frompyfunc(wrapped, nin, nout)

    @override
    def wrap_accumulate(
        self,
        axis: SupportsIndex,
    ) -> Callable[[npt.NDArray[Scalar]], npt.NDArray[Scalar]]:
        return lambda array: self._ufunc.accumulate(
            array,
            axis,
            out=np.empty_like(array),
        )

    @override
    def wrap_call(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[..., Any]:
        order = "F" if kwargs.pop("order", "C") == "F" else "C"

        def wrapper(
            *args: ArrayLike[Scalar],
        ) -> npt.NDArray[Scalar] | tuple[npt.NDArray[Scalar], ...]:
            shape = np.broadcast_shapes(*[x.shape for x in args])
            out = [
                np.empty(shape, cast("ScalarType", arg).dtype, order)
                for arg in self.restype
            ]
            return self._ufunc(*args, *out, casting="unsafe", **kwargs)
        return wrapper

    @override
    def wrap_reduce(
        self,
        axis: SupportsIndex | Sequence[SupportsIndex] | None,
        initial: object,
        where: onp.ToJustBool | onp.ToJustBoolND | None,
        *,
        keepdims: bool = False,
    ) -> Callable[..., Any]:
        index = slice(1) if keepdims else 0

        def wrapper(array: ArrayLike[Scalar]) -> npt.NDArray[Scalar]:
            if not all(array.shape):
                message = (
                    "zero-size array to reduction operation "
                    f"{self._ufunc.__name__} which has no identity"
                )
                raise ValueError(message)
            if axis is None:
                indices = (index,) * array.ndim
            else:
                indices = cast("list[int | slice]", [slice(None)])
                indices *= array.ndim
                match axis:
                    case [*_]:
                        for i in axis:
                            indices[i] = index
                    case _:
                        indices[axis] = index
                indices = tuple(indices)
            out = np.empty(
                # `np.generic.__getitem__` exists at runtime but
                # missing in type stubs before numpy 2.5.0
                cast("tuple[int, ...]", array[indices].shape),  # pyright: ignore[reportArgumentType, reportCallIssue, reportIndexIssue]
                cast("ScalarType", self.restype).dtype,
            )
            if initial is MISSING:
                return self._ufunc.reduce(
                    array,
                    axis,
                    out=out,
                    keepdims=keepdims,
                    where=where,
                )
            return self._ufunc.reduce(
                array,
                axis,
                out=out,
                keepdims=keepdims,
                initial=initial,
                where=where,
            )
        return wrapper

    @override
    def wrap_reduceat(
        self,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex,
    ) -> Callable[[npt.NDArray[Scalar]], npt.NDArray[Scalar]]:
        return lambda array: self._ufunc.reduceat(array, indices, axis).astype(  # pyright: ignore[reportArgumentType]
            cast("ScalarType", self.restype).dtype,
            order="C",
        )

    @override
    def wrap_outer(self, **kwargs: Unpack[UFuncKwargs]) -> Callable[..., Any]:
        order = "F" if kwargs.pop("order", "C") == "F" else "C"

        def wrapper(
            a: ArrayLike[Scalar],
            b: ArrayLike[Scalar],
        ) -> npt.NDArray[Scalar] | tuple[npt.NDArray[Scalar], ...]:
            shape = a.shape + b.shape
            out = tuple(
                np.empty(shape, cast("ScalarType", arg).dtype, order)
                for arg in self.restype
            )
            return self._ufunc.outer(a, b, out=out, casting="unsafe", **kwargs)
        return wrapper

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_scalar=True,
            allow_tuple=True,
            allow_units=True,
        )
