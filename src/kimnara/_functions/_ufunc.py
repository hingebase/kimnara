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

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import (
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
    Input,
    Number,
    NumericT,
    Output,
    Outputs,
    Scalar,
    UFuncKwargs,
)

from . import _common

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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


class _NumbaUFuncBase(_UFunc[_T], _common.UFuncWrapper[_T]):
    @property
    @override
    def nout(self) -> Literal[1]:
        return 1

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


class NumbaUFunc(_NumbaUFuncBase[_T], _common.Dispatchable):
    pass


class NumbaParallelUFunc(_NumbaUFuncBase[_T]):
    pass


class PyUFunc(_UFunc[_T]):
    @property
    @override
    def types(self) -> list[str]:
        raise NotImplementedError

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
