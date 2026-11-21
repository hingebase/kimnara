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

__all__ = ["cfunc", "func", "gufunc", "ufunc"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import optype.numpy as onp
from numba.core import compiler  # pyright: ignore[reportMissingTypeStubs]
from optype.typing import AnyComplex

from kimnara._typing import CustomInliningRule, FastMathOptions

from . import _cfunc, _func, _gufunc, _ufunc

if TYPE_CHECKING:
    import kimnara as kn

type _GUFuncAlignment = str | Literal[
    kn.Alignment.A,
    kn.Alignment.C,
    kn.Alignment.SSE,
    kn.Alignment.AVX,
    kn.Alignment.AVX512,
    kn.Alignment.MKL,
]


class _PyFuncDecorator(Protocol):
    # Split into two overloads so that `NonTuple | tuple[Any, ...]`
    # won't be accepted
    @overload
    def __call__[T: _func.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _func.PyFunc[T]: ...

    @overload
    def __call__[T: _func.Outputs](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _func.PyFunc[T]: ...


class _NumbaFuncDecorator(Protocol):
    @overload
    def __call__[T: _func.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _func.NumbaFunc[T]: ...

    @overload
    def __call__[T: _func.Outputs](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _func.NumbaFunc[T]: ...


class _PyGUFuncDecorator(Protocol):
    @overload
    def __call__[T: _gufunc.BooleanOutput](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _gufunc.PyGUFunc[T]: ...

    @overload
    def __call__[T: _gufunc.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _gufunc.PyGUFunc[T]: ...

    @overload
    def __call__[T: _gufunc.Outputs](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _gufunc.PyGUFunc[T]: ...


class _PyUFuncDecorator(Protocol):
    @overload
    def __call__[T: onp.ToJustBool](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.PyUFunc[T]: ...

    @overload
    def __call__[T: _ufunc.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.PyUFunc[T]: ...

    @overload
    def __call__[T: _ufunc.Outputs](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.PyUFunc[T]: ...


class _NumbaUFuncDecorator(Protocol):
    @overload
    def __call__[T: onp.ToJustBool](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.NumbaUFunc[T]: ...

    @overload
    def __call__[T: _ufunc.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.NumbaUFunc[T]: ...


class _NumbaParallelUFuncDecorator(Protocol):
    @overload
    def __call__[T: onp.ToJustBool](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.NumbaParallelUFunc[T]: ...

    @overload
    def __call__[T: _ufunc.Output](
        self,
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.NumbaParallelUFunc[T]: ...


@overload
def cfunc(wrapped: Callable[..., object], /) -> _cfunc.PyCFunc: ...

@overload
def cfunc(
    *,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    forceinline: Literal[False] = ...,
    inline: Literal["never"] = ...,
    nopython: Literal[False] = ...,
    parallel: Literal[False] = ...,
    pipeline_class: None = ...,
) -> Callable[[Callable[..., object]], _cfunc.PyCFunc]: ...

@overload
def cfunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    error_model: Literal["python", "numpy"] = ...,
    fastmath: FastMathOptions = ...,
    forceinline: bool = ...,
    inline: Literal["always", "never"] | CustomInliningRule = ...,
    nopython: Literal[True],
    parallel: bool | str = ...,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> Callable[[Callable[..., object]], _cfunc.NumbaCFunc]: ...


def cfunc(  # ruff: ignore[too-many-arguments]
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    boundscheck: bool | None = None,
    cache: bool | None = None,
    error_model: Literal["python", "numpy"] = "numpy",
    fastmath: FastMathOptions = False,
    forceinline: bool = False,
    inline: Literal["always", "never"] | CustomInliningRule = "never",
    nopython: bool = False,
    parallel: bool | str = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> Callable[..., Any]:
    if wrapped:
        return _cfunc.PyCFunc(wrapped)
    if not nopython:
        return _cfunc.PyCFunc

    def wrapper(wrapped: Callable[..., object], /) -> _cfunc.NumbaCFunc:
        return _cfunc.NumbaCFunc(
            wrapped,
            boundscheck=boundscheck,
            cache=cache,
            error_model=error_model,
            fastmath=fastmath,
            forceinline=forceinline,
            inline=inline,
            parallel=parallel,
            pipeline_class=pipeline_class,
        )
    return wrapper


@overload
def func[T: _func.Output](
    wrapped: Callable[..., T],
    /,
) -> _func.PyFunc[T]: ...

@overload
def func[T: _func.Outputs](
    wrapped: Callable[..., T],
    /,
) -> _func.PyFunc[T]: ...

@overload
def func(
    *,
    align: "str | kn.Alignment" = ...,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    forceinline: Literal[False] = ...,
    inline: Literal["never"] = ...,
    nogil: Literal[False] = ...,
    nopython: Literal[False] = ...,
    pad_value: AnyComplex | None = ...,
    parallel: Literal[False] = ...,
    pipeline_class: None = ...,
) -> _PyFuncDecorator: ...

@overload
def func(
    *,
    align: "str | kn.Alignment" = ...,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    error_model: Literal["python", "numpy"] = ...,
    fastmath: FastMathOptions = ...,
    forceinline: bool = ...,
    inline: Literal["always", "never"] | CustomInliningRule = ...,
    nogil: bool = ...,
    nopython: Literal[True],
    pad_value: AnyComplex | None = ...,
    parallel: bool | str = ...,
    pipeline_class: type[compiler.CompilerBase] | None = ...,
) -> _NumbaFuncDecorator: ...


def func(  # ruff: ignore[too-many-arguments]
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    align: "str | kn.Alignment" = "A",
    boundscheck: bool | None = None,
    cache: bool | None = None,
    error_model: Literal["python", "numpy"] = "numpy",
    fastmath: FastMathOptions = False,
    forceinline: bool = False,
    inline: Literal["always", "never"] | CustomInliningRule = "never",
    nogil: bool = False,
    nopython: bool = False,
    pad_value: AnyComplex | None = None,
    parallel: bool | str = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> Callable[..., Any]:
    if wrapped:
        return _func.PyFunc(wrapped)
    if not nopython:
        def pyfunc[T](wrapped: Callable[..., T], /) -> _func.PyFunc[T]:
            return _func.PyFunc(wrapped, align=align, pad_value=pad_value)
        return pyfunc

    def wrapper[T](wrapped: Callable[..., T], /) -> _func.NumbaFunc[T]:
        return _func.NumbaFunc(
            wrapped,
            align=align,
            boundscheck=boundscheck,
            cache=cache,
            error_model=error_model,
            fastmath=fastmath,
            forceinline=forceinline,
            inline=inline,
            nogil=nogil,
            pad_value=pad_value,
            parallel=parallel,
            pipeline_class=pipeline_class,
        )
    return wrapper


@overload
def gufunc(
    signature: str,
    *,
    align: _GUFuncAlignment = ...,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    nopython: Literal[False] = ...,
    pad_value: AnyComplex | None = ...,
    parallel: bool | str = ...,
) -> _PyGUFuncDecorator: ...

@overload
def gufunc(
    signature: str,
    *,
    align: _GUFuncAlignment = ...,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    nopython: Literal[True],
    pad_value: AnyComplex | None = ...,
    parallel: Literal[False] = ...,
) -> Callable[[Callable[..., None]], _gufunc.NumbaGUFunc]: ...

@overload
def gufunc(
    signature: str,
    *,
    align: _GUFuncAlignment = ...,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    nopython: Literal[True],
    pad_value: AnyComplex | None = ...,
    parallel: Literal[True] | str,
) -> Callable[[Callable[..., None]], _gufunc.NumbaParallelGUFunc]: ...


def gufunc(  # ruff: ignore[too-many-arguments]
    signature: str,
    *,
    align: _GUFuncAlignment = "A",
    boundscheck: bool | None = None,
    cache: bool | None = None,
    fastmath: FastMathOptions = False,
    nopython: bool = False,
    pad_value: AnyComplex | None = None,
    parallel: bool | str = False,
) -> Callable[..., Any]:
    if not nopython:
        def pygufunc[T](wrapped: Callable[..., T], /) -> _gufunc.PyGUFunc[T]:
            return _gufunc.PyGUFunc(
                wrapped,
                signature,
                align=align,
                pad_value=pad_value,
                parallel=parallel,
            )
        return pygufunc

    if not parallel:
        def numba_gufunc(
            wrapped: Callable[..., None],
            /,
        ) -> _gufunc.NumbaGUFunc:
            return _gufunc.NumbaGUFunc(
                wrapped,
                signature,
                align=align,
                boundscheck=boundscheck,
                cache=cache,
                fastmath=fastmath,
                pad_value=pad_value,
            )
        return numba_gufunc

    def wrapper(
        wrapped: Callable[..., None],
        /,
    ) -> _gufunc.NumbaParallelGUFunc:
        return _gufunc.NumbaParallelGUFunc(
            wrapped,
            signature,
            align=align,
            boundscheck=boundscheck,
            cache=cache,
            fastmath=fastmath,
            pad_value=pad_value,
            parallel=parallel,
        )
    return wrapper


@overload
def ufunc[T: onp.ToJustBool](
    wrapped: Callable[..., T],
    /,
) -> _ufunc.PyUFunc[T]: ...

@overload
def ufunc[T: _ufunc.Output](
    wrapped: Callable[..., T],
    /,
) -> _ufunc.PyUFunc[T]: ...

@overload
def ufunc[T: _ufunc.Outputs](
    wrapped: Callable[..., T],
    /,
) -> _ufunc.PyUFunc[T]: ...

@overload
def ufunc(
    *,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    identity: Literal["reorderable"] | None = ...,
    nopython: Literal[False] = ...,
    parallel: bool | str = ...,
) -> _PyUFuncDecorator: ...

@overload
def ufunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    identity: Literal[0, 1, "reorderable"] | None = ...,
    nopython: Literal[True],
    parallel: Literal[False] = ...,
) -> _NumbaUFuncDecorator: ...

@overload
def ufunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    identity: Literal[0, 1, "reorderable"] | None = ...,
    nopython: Literal[True],
    parallel: Literal[True] | str,
) -> _NumbaParallelUFuncDecorator: ...


def ufunc(  # ruff: ignore[too-many-arguments]
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    boundscheck: bool | None = None,
    cache: bool | None = None,
    fastmath: FastMathOptions = False,
    identity: Literal[0, 1, "reorderable"] | None = None,
    nopython: bool = False,
    parallel: bool | str = False,
) -> Callable[..., Any]:
    if wrapped:
        return _ufunc.PyUFunc(wrapped)
    if not nopython:
        match identity:
            case "reorderable" | None:
                def pyufunc[T](
                    wrapped: Callable[..., T],
                    /,
                ) -> _ufunc.PyUFunc[T]:
                    return _ufunc.PyUFunc(wrapped, identity=identity)
                return pyufunc
            case _:
                message = "identity must be None or 'reorderable'"
                raise ValueError(message)

    if not parallel:
        def numba_ufunc[T](
            wrapped: Callable[..., T],
            /,
        ) -> _ufunc.NumbaUFunc[T]:
            return _ufunc.NumbaUFunc(
                wrapped,
                boundscheck=boundscheck,
                cache=cache,
                fastmath=fastmath,
                identity=identity,
            )
        return numba_ufunc

    def wrapper[T](
        wrapped: Callable[..., T],
        /,
    ) -> _ufunc.NumbaParallelUFunc[T]:
        return _ufunc.NumbaParallelUFunc(
            wrapped,
            boundscheck=boundscheck,
            cache=cache,
            fastmath=fastmath,
            identity=identity,
            parallel=parallel,
        )
    return wrapper
