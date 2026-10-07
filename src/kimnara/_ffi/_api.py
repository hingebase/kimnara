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

import types
from collections.abc import Callable

from typing_extensions import Any, overload


@overload
def cdecl(wrapped: Callable[..., Any], /) -> ...: ...

@overload
def cdecl(
    lib: str = ...,
    *,
    function_name: str | None = ...,
) -> Callable[[Callable[..., Any]], Any]: ...


def cdecl(
    lib: Callable[..., Any] | str = "c",
    *,
    function_name: str | None = None,
) -> ...:
    raise NotImplementedError


def cydecl(
    module: types.ModuleType | str,
    *,
    function_name: str | None = None,
) -> ...:
    raise NotImplementedError


def ccall(
    *args: object,
    lib: str = "c",
    function_name: str | None = None,
) -> Any:  # ruff: ignore[any-type]
    raise NotImplementedError


def cycall(
    *args: object,
    module: types.ModuleType | str,
    function_name: str | None = None,
) -> Any:  # ruff: ignore[any-type]
    raise NotImplementedError
