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
    "BooleanOutputT",
    "NumbaParallelUFunc",
    "NumbaUFunc",
    "OutputT",
    "OutputsT",
    "PyUFunc",
]

import abc
import inspect
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    TypeAlias,
    cast,
    no_type_check,
)

import numba  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numba.np.ufunc import dufunc  # pyright: ignore[reportMissingTypeStubs]
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import (
    Any,
    Sentinel,
    SupportsIndex,
    TypeVar,
    Unpack,
    overload,
    override,
)

import kimnara as kn
from kimnara import _spec
from kimnara._typing import (
    ArrayLike,
    FastMathOptions,
    Input,
    Number,
    NumericT,
    Output,
    Outputs,
    Scalar,
    UFuncKwargs,
)

from . import _common

if TYPE_CHECKING:
    from kimnara._spec._numpy._base import ScalarType

_AccumulateShapeT = TypeVar("_AccumulateShapeT", bound=onp.AtLeast1D)
_Input = NumericT | PlainQuantity[NumericT]
_InputAtLeast1D: TypeAlias = """
    np.ndarray[onp.AtLeast1D, np.dtype[Scalar]]
    | PlainQuantity[np.ndarray[onp.AtLeast1D, np.dtype[Number]]]"""
_Output = NumericT | NumpyQuantity[NumericT]
_ReduceAtShapeT = TypeVar(
    "_ReduceAtShapeT",
    tuple[int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, Unpack[tuple[int, ...]]],
    tuple[int, Unpack[tuple[int, ...]]],
)
_T = TypeVar("_T")
_U = TypeVar("_U", np.ufunc, dufunc.DUFunc)

BooleanOutputT = TypeVar("BooleanOutputT", bound=onp.ToJustBool)
OutputT = TypeVar(
    "OutputT",
    bound=op.JustInt | op.JustFloat | op.JustComplex | Number,
)
OutputsT = TypeVar("OutputsT", bound=tuple[complex | Scalar, ...])

MISSING = Sentinel("MISSING")


class _UFunc(_common.BaseUFuncWrapper[_T]):
    @property
    @override
    def signature(self) -> None:
        pass

    @overload
    def __call__(
        self: "_UFunc[BooleanOutputT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def __call__(
        self: "_UFunc[OutputT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> _Output[ArrayLike[Number]]: ...

    @overload
    def __call__(
        self: "_UFunc[OutputsT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> tuple[Output, ...]: ...

    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        wrapper = self.wrap_call(**kwargs)
        return self.validate_call(wrapper)(*args)

    @overload
    def accumulate(
        self: "_UFunc[BooleanOutputT]",
        array: np.ndarray[_AccumulateShapeT, np.dtype[np.bool_]],
        /,
        axis: SupportsIndex = ...,
    ) -> np.ndarray[_AccumulateShapeT, np.dtype[np.bool_]]: ...

    @overload
    def accumulate(
        self: "_UFunc[OutputT]",
        array: _Input[np.ndarray[_AccumulateShapeT, np.dtype[Number]]],
        /,
        axis: SupportsIndex = ...,
    ) -> _Output[np.ndarray[_AccumulateShapeT, np.dtype[Number]]]: ...

    def accumulate(
        self: "_UFunc[BooleanOutputT] | _UFunc[OutputT]",
        array: _InputAtLeast1D,
        /,
        axis: SupportsIndex = 0,
    ) -> object:
        sig = self._ufunc_method_signature("accumulate")
        wrapper = self.wrap_accumulate(axis)
        return self.validate_call(wrapper, sig)(array)

    @overload
    def reduce(
        self: "_UFunc[BooleanOutputT]",
        array: onp.ToJustBool | npt.NDArray[np.bool_],
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        *,
        keepdims: bool = ...,
        initial: object = ...,
        where: onp.ToJustBool | onp.ToJustBoolND | None = ...,
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def reduce(
        self: "_UFunc[OutputT]",
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        *,
        keepdims: bool = ...,
        initial: object = ...,
        where: onp.ToJustBool | onp.ToJustBoolND | None = ...,
    ) -> _Output[ArrayLike[Number]]: ...

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
        sig = self._ufunc_method_signature("reduce")
        wrapper = self.wrap_reduce(
            axis,
            initial,
            keepdims=keepdims,
            where=where,
        )
        return self.validate_call(wrapper, sig)(array)

    @overload
    def reduceat(
        self: "_UFunc[BooleanOutputT]",
        array: np.ndarray[_ReduceAtShapeT, np.dtype[np.bool_]],
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = ...,
    ) -> np.ndarray[_ReduceAtShapeT, np.dtype[np.bool_]]: ...

    @overload
    def reduceat(
        self: "_UFunc[OutputT]",
        array: _Input[np.ndarray[_ReduceAtShapeT, np.dtype[Number]]],
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = ...,
    ) -> _Output[np.ndarray[_ReduceAtShapeT, np.dtype[Number]]]: ...

    def reduceat(
        self,
        array: _InputAtLeast1D,
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = 0,
    ) -> object:
        sig = self._ufunc_method_signature("reduceat")
        wrapper = self.wrap_reduceat(indices, axis)
        return self.validate_call(wrapper, sig)(array)

    @overload
    def outer(
        self: "_UFunc[BooleanOutputT]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def outer(
        self: "_UFunc[OutputT]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> _Output[ArrayLike[Number]]: ...

    @overload
    def outer(
        self: "_UFunc[OutputsT]",
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> tuple[Output, ...]: ...

    def outer(
        self,
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> object:
        try:
            type1, type2 = self.argtypes
        except ValueError:
            message = "outer product only supported for binary functions"
            raise ValueError(message) from None
        arg1, arg2 = self.sig.parameters.values()
        arg1 = inspect.Parameter(
            arg1.name,
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=type1,
        )
        arg2 = inspect.Parameter(
            arg2.name,
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=type2,
        )
        sig = inspect.Signature(
            [arg1, arg2],
            return_annotation=self.sig.return_annotation,
        )
        wrapper = self.wrap_outer(**kwargs)
        return self.validate_call(wrapper, sig)(a, b)

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
            readonly=True,
        )

    @abc.abstractmethod
    def wrap_accumulate(self, axis: SupportsIndex) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_call(
        self,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Callable[..., Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_reduce(
        self,
        axis: SupportsIndex | Sequence[SupportsIndex] | None,
        initial: object,
        *,
        keepdims: bool = False,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
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
    def wrap_outer(
        self,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Callable[..., Any]:
        raise NotImplementedError

    def _ufunc_method_signature(self, method: str) -> inspect.Signature:
        match self.argtypes:
            case [x1, x2] if x1 == x2 == self.restype:
                pass
            case _:
                message = (
                    f"{method} only supported for binary functions returning a"
                    " single value of the same kind"
                )
                raise TypeError(message)
        arg = inspect.Parameter(
            "array",
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=x1,
        )
        return inspect.Signature([arg], return_annotation=x1)


class _NumbaUFuncBase(_UFunc[_T], _common.UFuncWrapper[_T], Generic[_T, _U]):
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        wrapped: Callable[..., _T],
        *,
        boundscheck: bool | None = None,
        cache: bool | None = None,
        fastmath: FastMathOptions = False,
        identity: Literal[0, 1, "reorderable"] | None = None,
        parallel: bool | str = False,
    ) -> None:
        if isinstance(parallel, str):
            raise NotImplementedError
        self.infer(wrapped)
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = self.restype.to_numba()
        ufunc = cast(
            "_U",
            numba.vectorize(  # pyright: ignore[reportUnknownMemberType]
                [restype(*argtypes)],
                boundscheck=boundscheck,
                cache=_common.can_cache(wrapped, cache=cache),
                fastmath=fastmath,
                identity=identity,
                target="parallel" if parallel else "cpu",
            )(wrapped),
        )
        self.postprocess(ufunc)

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
        *,
        keepdims: bool = False,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
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
    def postprocess(self, ufunc: _U) -> None:
        raise NotImplementedError


class NumbaUFunc(_NumbaUFuncBase[_T, dufunc.DUFunc], _common.Dispatchable):
    @no_type_check
    @override
    def postprocess(self, ufunc: dufunc.DUFunc) -> None:
        self.dispatcher = ufunc._dispatcher  # ruff: ignore[private-member-access]
        self._ufunc = ufunc.ufunc


class NumbaParallelUFunc(_NumbaUFuncBase[_T, np.ufunc]):
    @override
    def postprocess(self, ufunc: np.ufunc) -> None:
        self._ufunc = ufunc


class PyUFunc(_UFunc[_T]):
    def __init__(
        self,
        wrapped: Callable[..., _T],
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
    def wrap_call(
        self,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Callable[..., Any]:
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
        *,
        keepdims: bool = False,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
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
                cast("tuple[int, ...]", array[indices].shape),  # pyright: ignore[reportIndexIssue]
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
    def wrap_outer(
        self,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Callable[..., Any]:
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
