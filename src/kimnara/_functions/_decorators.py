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
from typing_extensions import Any, overload

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


def cfunc(  # noqa: PLR0913
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
    wrapped: Callable[..., _func.PythonType],
    /,
) -> _func.PyFunc: ...

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
) -> Callable[[Callable[..., _func.PythonType]], _func.PyFunc]: ...

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
) -> Callable[[Callable[..., _func.NumbaType]], _func.NumbaFunc]: ...


def func(  # noqa: PLR0913
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
    raise NotImplementedError


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
) -> Callable[[Callable[..., _gufunc.PythonType]], _gufunc.PyGUFunc]: ...

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
) -> Callable[
    [Callable[..., _gufunc.NumbaType]],
    _gufunc.NumbaGUFunc,
]: ...

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
) -> Callable[
    [Callable[..., _gufunc.NumbaType]],
    _gufunc.NumbaParallelGUFunc,
]: ...


def gufunc(  # noqa: PLR0913
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
    raise NotImplementedError


@overload
def ufunc(
    wrapped: Callable[..., _ufunc.PythonType],
    /,
) -> _ufunc.PyUFunc: ...

@overload
def ufunc(
    *,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    identity: Literal["reorderable"] | None = ...,
    nopython: Literal[False] = ...,
    parallel: bool | str = ...,
) -> Callable[[Callable[..., _ufunc.PythonType]], _ufunc.PyUFunc]: ...

@overload
def ufunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    identity: Literal[0, 1, "reorderable"] | None = ...,
    nopython: Literal[True],
    parallel: Literal[False] = ...,
) -> Callable[
    [Callable[..., _ufunc.NumbaType]],
    _ufunc.NumbaUFunc,
]: ...

@overload
def ufunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: FastMathOptions = ...,
    identity: Literal[0, 1, "reorderable"] | None = ...,
    nopython: Literal[True],
    parallel: Literal[True] | str,
) -> Callable[
    [Callable[..., _ufunc.NumbaType]],
    _ufunc.NumbaParallelUFunc,
]: ...


def ufunc(  # noqa: PLR0913
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
    raise NotImplementedError
