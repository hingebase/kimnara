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
    "NumbaParallelUFunc",
    "NumbaType",
    "NumbaUFunc",
    "PyUFunc",
    "PythonType",
]

import abc
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity
from typing_extensions import (
    Sentinel,
    SupportsIndex,
    TypeVar,
    Unpack,
    override,
)

import kimnara as kn
from kimnara import _spec
from kimnara._typing import (
    Input,
    Number,
    Output,
    Outputs,
    Scalar,
    UFuncKwargs,
)

from . import _common

NumbaType = np.generic | complex
PythonType = np.generic | complex | tuple[np.generic | complex, ...]
_InputArray = npt.NDArray[Scalar] | PlainQuantity[npt.NDArray[Number]]
_OutputArray = npt.NDArray[Scalar] | NumpyQuantity[npt.NDArray[Number]]
_T = TypeVar("_T", NumbaType, PythonType)
MISSING = Sentinel("MISSING")


class _UFunc(_common.BaseUFuncWrapper[_T]):
    @property
    @override
    def signature(self) -> None:
        pass

    @abc.abstractmethod
    def accumulate(
        self,
        array: _InputArray,
        /,
        axis: SupportsIndex = ...,
    ) -> _OutputArray:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce(
        self,
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        *,
        keepdims: bool = ...,
        initial: object = ...,
        where: onp.ToJustBool | onp.ToJustBoolND | None = ...,
    ) -> Output:
        raise NotImplementedError

    @abc.abstractmethod
    def reduceat(
        self,
        array: _InputArray,
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = ...,
    ) -> _OutputArray:
        raise NotImplementedError

    @abc.abstractmethod
    def outer(
        self,
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Output:
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


class _NumbaUFuncBase(
    _UFunc[NumbaType],
    _common.UFuncWrapper[NumbaType],
):
    @property
    @override
    def nout(self) -> Literal[1]:
        return 1

    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        raise NotImplementedError

    @override
    def accumulate(
        self,
        array: _InputArray,
        /,
        axis: SupportsIndex = 0,
    ) -> _OutputArray:
        raise NotImplementedError

    @override
    def reduce(
        self,
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = 0,
        *,
        keepdims: bool = False,
        initial: object = MISSING,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
    ) -> Output:
        raise NotImplementedError

    @override
    def reduceat(
        self,
        array: _InputArray,
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = 0,
    ) -> _OutputArray:
        raise NotImplementedError

    @override
    def outer(
        self,
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Output:
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
            allow_units=True,
        )


class NumbaUFunc(_NumbaUFuncBase, _common.Dispatchable):
    pass


class NumbaParallelUFunc(_NumbaUFuncBase):
    pass


class PyUFunc(_UFunc[PythonType]):
    @property
    @override
    def types(self) -> list[str]:
        raise NotImplementedError

    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        raise NotImplementedError

    @override
    def accumulate(
        self,
        array: _InputArray,
        /,
        axis: SupportsIndex = 0,
    ) -> _OutputArray:
        raise NotImplementedError

    @override
    def reduce(
        self,
        array: Input,
        /,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = 0,
        *,
        keepdims: bool = False,
        initial: object = MISSING,
        where: onp.ToJustBool | onp.ToJustBoolND | None = True,
    ) -> Output:
        raise NotImplementedError

    @override
    def reduceat(
        self,
        array: _InputArray,
        /,
        indices: onp.ToJustIntStrict1D,
        axis: SupportsIndex = 0,
    ) -> _OutputArray:
        raise NotImplementedError

    @override
    def outer(
        self,
        a: Input,
        b: Input,
        /,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Output:
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
