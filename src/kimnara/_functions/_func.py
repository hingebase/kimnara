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

__all__ = ["NumbaFunc", "OutputT", "OutputsT", "PyFunc"]

from collections.abc import Callable

from optype.typing import AnyComplex
from typing_extensions import TypeVar, overload, override

import kimnara as kn
from kimnara import _spec
from kimnara._typing import ArrayLike, Input, Output, Scalar

from . import _common

_RawOutputs = complex | ArrayLike[Scalar] | None | tuple["_RawOutputs", ...]
_T = TypeVar("_T")
_WrappedOutputs = Output | None | tuple["_WrappedOutputs", ...]

OutputT = TypeVar("OutputT", bound=complex | ArrayLike[Scalar] | None)
OutputsT = TypeVar("OutputsT", bound=tuple[_RawOutputs, ...])


class _Func(_common.Validator[_T]):
    def __init__(
        self,
        wrapped: Callable[..., _T],
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: AnyComplex | None = None,
    ) -> None:
        self.infer(wrapped, align=align, pad_value=pad_value)

    @overload
    def __call__(self: "_Func[OutputT]", *args: Input) -> Output | None: ...

    @overload
    def __call__(
        self: "_Func[OutputsT]",
        *args: Input,
    ) -> tuple[_WrappedOutputs, ...]: ...

    def __call__(self, *args: Input) -> object:
        raise NotImplementedError


class NumbaFunc(_Func[_T], _common.Dispatchable):
    @override
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=align,
            allow_align=frozenset(kn.Alignment),
            allow_array=True,
            allow_optional=True,
            allow_scalar=True,
            allow_units=True,
            pad_value=pad_value,
            readonly=True,
        )

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        allow_align = frozenset([kn.A, kn.C, kn.F])
        return _spec.TypingContext(
            align=align if align in allow_align else kn.A,
            allow_align=allow_align,
            allow_array=True,
            allow_none=True,
            allow_optional=True,
            allow_scalar=True,
            allow_tuple="nested",
            allow_units=True,
            pad_value=pad_value,
        )


class PyFunc(_Func[_T]):
    @override
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=align,
            allow_align=frozenset(kn.Alignment),
            allow_array=True,
            allow_optional=True,
            allow_scalar=True,
            allow_units=True,
            ndim=None,
            pad_value=pad_value,
            readonly=True,
        )

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_align=frozenset([kn.A]),
            allow_array=True,
            allow_none=True,
            allow_optional=True,
            allow_scalar=True,
            allow_tuple="nested",
            allow_units=True,
            ndim=None,
            pad_value=pad_value,
        )
