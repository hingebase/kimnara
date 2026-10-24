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

__all__ = ["ccall", "cdecl", "cycall", "cydecl"]

import ctypes
import types
from collections.abc import Callable
from typing import cast

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import Any, overload

from kimnara import _utils

from . import _call, _decl


@overload
def cdecl(wrapped: Callable[..., Any], /) -> _decl.Decl: ...

@overload
def cdecl(
    lib: str = ...,
    *,
    function_name: str | None = ...,
) -> Callable[[Callable[..., Any]], _decl.Decl]: ...


def cdecl(
    lib: Callable[..., Any] | str = "c",
    *,
    function_name: str | None = None,
) -> _decl.Decl | Callable[[Callable[..., Any]], _decl.Decl]:
    if callable(lib):
        return _decl.cdecl(lib)

    def decorator(wrapped: Callable[..., Any], /) -> _decl.Decl:
        return _decl.cdecl(wrapped, lib, function_name)
    return decorator


def cydecl(
    module: types.ModuleType | str,
    *,
    function_name: str | None = None,
) -> Callable[[Callable[..., Any]], _decl.Decl]:
    def decorator(wrapped: Callable[..., Any], /) -> _decl.Decl:
        return _decl.cydecl(wrapped, module, function_name)
    return decorator


def ccall(
    *args: object,
    lib: str = "c",
    function_name: str | None = None,
) -> Any:  # ruff: ignore[any-type]
    op = _call.locate()
    dll = ctypes.cdll[lib] if "c" != lib != "m" else _utils.libc
    if not function_name:
        function_name = op.frame.f_code.co_name
    return _call.call(op, args, (function_name, dll))


def cycall(
    *args: object,
    module: types.ModuleType | str,
    function_name: str | None = None,
) -> Any:  # ruff: ignore[any-type]
    op = _call.locate()
    module_name = cast("str", getattr(module, "__name__", module))
    if not function_name:
        function_name = op.frame.f_code.co_name
    address = numba.extending.get_cython_function_address(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        module_name,
        function_name,
    )
    return _call.call(op, args, address)  # pyright: ignore[reportUnknownArgumentType]
