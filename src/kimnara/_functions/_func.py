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

__all__ = ["NumbaFunc", "Output", "Outputs", "PyFunc"]

from collections.abc import Callable
from typing import Literal, cast, overload, override

import numba  # pyright: ignore[reportMissingTypeStubs]
from numba.core import compiler, registry  # pyright: ignore[reportMissingTypeStubs]
from optype.typing import AnyComplex

import kimnara as kn
from kimnara import _spec
from kimnara._typing import (
    ArrayLike,
    CustomInliningRule,
    FastMathOptions,
    Input,
    Scalar,
)
from kimnara._typing import Output as _Output

from . import _common

type _RawOutputs = complex | ArrayLike[Scalar] | tuple[_RawOutputs, ...] | None

# Numba will convert NumPy scalars to Python scalars on output
# Python functions will forward the return value as-is
type _WrappedOutputs = complex | _Output | tuple[_WrappedOutputs, ...] | None

type Output = complex | ArrayLike[Scalar] | None
type Outputs = tuple[_RawOutputs, ...]


class _Func[T](_common.Validator[T]):
    _func: Callable[..., object]

    def __init__(
        self,
        wrapped: Callable[..., T],
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: AnyComplex | None = None,
    ) -> None:
        self.infer(wrapped, align=align, pad_value=pad_value)

    @overload
    def __call__[O: Output](
        self: "_Func[O]",
        *args: Input | None,
    ) -> complex | _Output | None: ...

    @overload
    def __call__[O: Outputs](
        self: "_Func[O]",
        *args: Input | None,
    ) -> tuple[_WrappedOutputs, ...]: ...

    def __call__(self, *args: Input | None) -> object:
        return self._func(*args)


class NumbaFunc[T](_Func[T], _common.Dispatchable):
    @override
    def __init__(
        self,
        wrapped: Callable[..., T],
        *,
        align: "str | kn.Alignment" = "A",
        boundscheck: bool | None = None,
        cache: bool | None = None,
        error_model: Literal["python", "numpy"] = "numpy",
        fastmath: FastMathOptions = False,
        forceinline: bool = False,
        inline: Literal["always", "never"] | CustomInliningRule = "never",
        nogil: bool = False,
        pad_value: AnyComplex | None = None,
        parallel: bool | str = False,
        pipeline_class: type[compiler.CompilerBase] | None = None,
    ) -> None:
        super().__init__(wrapped, align=align, pad_value=pad_value)
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = self.restype.to_numba()
        decorator = numba.njit(  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
            [restype(*argtypes)],
            boundscheck=boundscheck,
            cache=_common.can_cache(wrapped, cache=cache),
            error_model=error_model,
            fastmath=fastmath,  # pyright: ignore[reportArgumentType]
            forceinline=forceinline,
            inline=inline,
            no_cfunc_wrapper=True,
            nogil=nogil,
            parallel=bool(parallel),
            pipeline_class=pipeline_class,
        )
        if parallel:
            with kn.threading.using_backend(parallel):
                func = cast("registry.CPUDispatcher", decorator(wrapped))
        else:
            func = cast("registry.CPUDispatcher", decorator(wrapped))
        self.dispatcher = func
        self._func = self.validate_call(func)

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
            allow_optional="units",
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


class PyFunc[T](_Func[T]):
    @override
    def __init__(
        self,
        wrapped: Callable[..., T],
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: AnyComplex | None = None,
    ) -> None:
        super().__init__(wrapped, align=align, pad_value=pad_value)
        self._func = self.validate_call(wrapped)

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
            allow_optional="units",
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
