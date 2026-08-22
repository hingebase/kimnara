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
from typing import TYPE_CHECKING, Literal, TypeAlias

from numba.core import compiler  # pyright: ignore[reportMissingTypeStubs]
from optype.typing import AnyComplex
from typing_extensions import Any, Protocol, TypeVar, overload

from kimnara._typing import CustomInliningRule, FastMathOptions

from . import _cfunc, _func, _gufunc, _ufunc

if TYPE_CHECKING:
    import kimnara as kn

_GUFuncAlignment: TypeAlias = """str | Literal[
    kn.Alignment.A,
    kn.Alignment.C,
    kn.Alignment.SSE,
    kn.Alignment.AVX,
    kn.Alignment.AVX512,
    kn.Alignment.MKL,
]"""
_T = TypeVar("_T")


class _PyFuncDecorator(Protocol):
    # Split into two overloads so that `NonTuple | tuple[Any, ...]`
    # won't be accepted
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _func.OutputT],
        /,
    ) -> _func.PyFunc[_func.OutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _func.OutputsT],
        /,
    ) -> _func.PyFunc[_func.OutputsT]: ...


class _NumbaFuncDecorator(Protocol):
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _func.OutputT],
        /,
    ) -> _func.NumbaFunc[_func.OutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _func.OutputsT],
        /,
    ) -> _func.NumbaFunc[_func.OutputsT]: ...


class _PyGUFuncDecorator(Protocol):
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _gufunc.BooleanOutputT],
        /,
    ) -> _gufunc.PyGUFunc[_gufunc.BooleanOutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _gufunc.OutputT],
        /,
    ) -> _gufunc.PyGUFunc[_gufunc.OutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _gufunc.OutputsT],
        /,
    ) -> _gufunc.PyGUFunc[_gufunc.OutputsT]: ...


class _PyUFuncDecorator(Protocol):
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.BooleanOutputT],
        /,
    ) -> _ufunc.PyUFunc[_ufunc.BooleanOutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.OutputT],
        /,
    ) -> _ufunc.PyUFunc[_ufunc.OutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.OutputsT],
        /,
    ) -> _ufunc.PyUFunc[_ufunc.OutputsT]: ...


class _NumbaUFuncDecorator(Protocol):
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.BooleanOutputT],
        /,
    ) -> _ufunc.NumbaUFunc[_ufunc.BooleanOutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.OutputT],
        /,
    ) -> _ufunc.NumbaUFunc[_ufunc.OutputT]: ...


class _NumbaParallelUFuncDecorator(Protocol):
    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.BooleanOutputT],
        /,
    ) -> _ufunc.NumbaParallelUFunc[_ufunc.BooleanOutputT]: ...

    @overload
    def __call__(
        self,
        wrapped: Callable[..., _ufunc.OutputT],
        /,
    ) -> _ufunc.NumbaParallelUFunc[_ufunc.OutputT]: ...


@overload
def cfunc(
    wrapped: Callable[..., _cfunc.PythonT],
    /,
) -> _cfunc.PyCFunc[_cfunc.PythonT]: ...

@overload
def cfunc(
    *,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    forceinline: Literal[False] = ...,
    inline: Literal["never"] = ...,
    nogil: Literal[False] = ...,
    nopython: Literal[False] = ...,
    parallel: Literal[False] = ...,
    pipeline_class: None = ...,
) -> Callable[
    [Callable[..., _cfunc.PythonT]],
    _cfunc.PyCFunc[_cfunc.PythonT],
]: ...

@overload
def cfunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    error_model: Literal["python", "numpy"] = ...,
    fastmath: FastMathOptions = ...,
    forceinline: bool = ...,
    inline: Literal["always", "never"] | CustomInliningRule = ...,
    nogil: bool = ...,
    nopython: Literal[True],
    parallel: bool | str = ...,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> Callable[
    [Callable[..., _cfunc.NumbaT]],
    _cfunc.NumbaCFunc[_cfunc.NumbaT],
]: ...


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
    nogil: bool = False,
    nopython: bool = False,
    parallel: bool | str = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> Callable[..., Any]:
    if wrapped:
        return _cfunc.PyCFunc(wrapped)
    if not nopython:
        return _cfunc.PyCFunc

    def wrapper(
        wrapped: Callable[..., _cfunc.NumbaT],
        /,
    ) -> _cfunc.NumbaCFunc[_cfunc.NumbaT]:
        return _cfunc.NumbaCFunc(
            wrapped,
            boundscheck=boundscheck,
            cache=cache,
            error_model=error_model,
            fastmath=fastmath,
            forceinline=forceinline,
            inline=inline,
            nogil=nogil,
            parallel=parallel,
            pipeline_class=pipeline_class,
        )
    return wrapper


@overload
def func(
    wrapped: Callable[..., _func.OutputT],
    /,
) -> _func.PyFunc[_func.OutputT]: ...

@overload
def func(
    wrapped: Callable[..., _func.OutputsT],
    /,
) -> _func.PyFunc[_func.OutputsT]: ...

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
        def pyfunc(
            wrapped: Callable[..., _T],
            /,
        ) -> _func.PyFunc[_T]:
            return _func.PyFunc(wrapped, align=align, pad_value=pad_value)
        return pyfunc

    def wrapper(
        wrapped: Callable[..., _T],
        /,
    ) -> _func.NumbaFunc[_T]:
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
        def pygufunc(
            wrapped: Callable[..., _T],
            /,
        ) -> _gufunc.PyGUFunc[_T]:
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
def ufunc(
    wrapped: Callable[..., _ufunc.BooleanOutputT],
    /,
) -> _ufunc.PyUFunc[_ufunc.BooleanOutputT]: ...

@overload
def ufunc(
    wrapped: Callable[..., _ufunc.OutputT],
    /,
) -> _ufunc.PyUFunc[_ufunc.OutputT]: ...

@overload
def ufunc(
    wrapped: Callable[..., _ufunc.OutputsT],
    /,
) -> _ufunc.PyUFunc[_ufunc.OutputsT]: ...

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
                def pyufunc(
                    wrapped: Callable[..., _T],
                    /,
                ) -> _ufunc.PyUFunc[_T]:
                    return _ufunc.PyUFunc(wrapped, identity=identity)
                return pyufunc
            case _:
                message = "identity must be None or 'reorderable'"
                raise ValueError(message)

    if not parallel:
        def numba_ufunc(
            wrapped: Callable[..., _T],
            /,
        ) -> _ufunc.NumbaUFunc[_T]:
            return _ufunc.NumbaUFunc(
                wrapped,
                boundscheck=boundscheck,
                cache=cache,
                fastmath=fastmath,
                identity=identity,
            )
        return numba_ufunc

    def wrapper(
        wrapped: Callable[..., _T],
        /,
    ) -> _ufunc.NumbaParallelUFunc[_T]:
        return _ufunc.NumbaParallelUFunc(
            wrapped,
            boundscheck=boundscheck,
            cache=cache,
            fastmath=fastmath,
            identity=identity,
            parallel=parallel,
        )
    return wrapper
