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

from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    compiler,
    cpu_options,
    ir,
)
from optype.typing import AnyComplex
from typing_extensions import Any, overload

if TYPE_CHECKING:
    import kimnara as kn

_CustomInliningRule = Callable[[ir.Expr, Any, Any], bool]
_FastMathFlags = Literal[
    "fast",
    "nnan", "ninf", "nsz", "arcp",
    "contract", "afn", "reassoc",
]
_FastMathOptions: TypeAlias = """
    bool
    | set[_FastMathFlags]
    | dict[_FastMathFlags, bool]
    | cpu_options.FastMathOptions"""
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
    wrapped: Callable[..., Any],
    /,
) -> ...: ...

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
) -> ...: ...

@overload
def cfunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    error_model: Literal["python", "numpy"] = ...,
    fastmath: _FastMathOptions = ...,
    forceinline: bool = ...,
    inline: Literal["always", "never"] | _CustomInliningRule = ...,
    nogil: bool = ...,
    nopython: bool,
    parallel: bool | str = ...,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> ...: ...


def cfunc(  # noqa: PLR0913
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    boundscheck: bool | None = None,
    cache: bool | None = None,
    error_model: Literal["python", "numpy"] = "python",
    fastmath: _FastMathOptions = False,
    forceinline: bool = False,
    inline: Literal["always", "never"] | _CustomInliningRule = "never",
    nogil: bool = False,
    nopython: bool = False,
    parallel: bool | str = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> ...:
    raise NotImplementedError


@overload
def func(
    wrapped: Callable[..., Any],
    /,
) -> ...: ...

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
) -> ...: ...

@overload
def func(
    *,
    align: "str | kn.Alignment" = ...,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    error_model: Literal["python", "numpy"] = ...,
    fastmath: _FastMathOptions = ...,
    forceinline: bool = ...,
    inline: Literal["always", "never"] | _CustomInliningRule = ...,
    nogil: bool = ...,
    nopython: bool,
    pad_value: AnyComplex | None = ...,
    parallel: bool | str = ...,
    pipeline_class: type[compiler.CompilerBase] | None = ...,
) -> ...: ...


def func(  # noqa: PLR0913
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    align: "str | kn.Alignment" = "A",
    boundscheck: bool | None = None,
    cache: bool | None = None,
    error_model: Literal["python", "numpy"] = "python",
    fastmath: _FastMathOptions = False,
    forceinline: bool = False,
    inline: Literal["always", "never"] | _CustomInliningRule = "never",
    nogil: bool = False,
    nopython: bool = False,
    pad_value: AnyComplex | None = None,
    parallel: bool | str = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
) -> ...:
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
) -> ...: ...

@overload
def gufunc(
    signature: str,
    *,
    align: _GUFuncAlignment = ...,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: _FastMathOptions = ...,
    nopython: bool,
    pad_value: AnyComplex | None = ...,
    parallel: bool | str = ...,
) -> ...: ...


def gufunc(  # noqa: PLR0913
    signature: str,
    *,
    align: _GUFuncAlignment = "A",
    boundscheck: bool | None = None,
    cache: bool | None = None,
    fastmath: _FastMathOptions = False,
    nopython: bool = False,
    pad_value: AnyComplex | None = None,
    parallel: bool | str = False,
) -> ...:
    raise NotImplementedError


@overload
def ufunc(
    wrapped: Callable[..., Any],
    /,
) -> ...: ...

@overload
def ufunc(
    *,
    boundscheck: Literal[True] | None = ...,
    cache: Literal[False] | None = ...,
    fastmath: Literal[False] = ...,
    # np.vectorize calls np.frompyfunc without the `identity` argument
    identity: None = ...,
    nopython: Literal[False] = ...,
    parallel: bool | str = ...,
) -> ...: ...

@overload
def ufunc(
    *,
    boundscheck: bool | None = ...,
    cache: bool | None = ...,
    fastmath: _FastMathOptions = ...,
    identity: Literal[0, 1, "reorderable"] | None = ...,
    nopython: bool,
    parallel: bool | str = ...,
) -> ...: ...


def ufunc(  # noqa: PLR0913
    wrapped: Callable[..., Any] | None = None,
    /,
    *,
    boundscheck: bool | None = None,
    cache: bool | None = None,
    fastmath: _FastMathOptions = False,
    identity: Literal[0, 1, "reorderable"] | None = None,
    nopython: bool = False,
    parallel: bool | str = False,
) -> ...:
    raise NotImplementedError
