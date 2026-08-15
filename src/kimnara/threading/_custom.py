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

__all__ = ["register_custom_backend", "using_backend"]

import dataclasses
import types
import warnings
from contextlib import AbstractContextManager
from typing import Literal

import numpy as np
from llvmlite import binding, ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core import cgutils  # pyright: ignore[reportMissingTypeStubs]
from numba.core.cgutils import (  # pyright: ignore[reportMissingTypeStubs]
    get_or_insert_function,  # pyright: ignore[reportUnknownVariableType]
)
from typing_extensions import override

import kimnara as kn

_FUNCTIONS_TO_BE_MONKEY_PATCHED = (
    "get_num_threads",
    "get_thread_id",
    "numba_parallel_for",
)


def register_custom_backend(
    name: str,
    get_num_threads: int,
    get_thread_id: int,
    parallel_for: int,
    vectorize: type[np.vectorize],
) -> None:
    if not name.isidentifier():
        message = f"Invalid backend name: {name!r}"
        raise ValueError(message)
    if name in _registry:
        message = f"Occupied backend name: {name!r}"
        raise ValueError(message)
    binding.add_symbol(f"{name}_get_num_threads", get_num_threads)  # pyright: ignore[reportUnknownMemberType]
    binding.add_symbol(f"{name}_get_thread_id", get_thread_id)  # pyright: ignore[reportUnknownMemberType]
    binding.add_symbol(f"{name}_numba_parallel_for", parallel_for)  # pyright: ignore[reportUnknownMemberType]
    _registry[name] = vectorize


@dataclasses.dataclass
class _UsingBackend(AbstractContextManager[type[np.vectorize], None]):
    name: str
    vectorize: type[np.vectorize]

    @override
    def __enter__(self) -> type[np.vectorize]:
        if _active_backends:
            warnings.warn(
                f"Switching threading backend from {_active_backends[-1]!r} to"
                f" {self.name!r}. This is experimental and may be disallowed "
                "in the future.",
                category=FutureWarning,
                stacklevel=2,
            )
        _active_backends.append(self.name)
        return self.vectorize

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
        /,
    ) -> None:
        _active_backends.pop()


def using_backend(name: str | Literal[True], /) -> _UsingBackend:
    match name:
        case True:
            message = (
                "parallel=True will be supported in "
                "https://github.com/hingebase/kimnara/issues/14 . "
                "Please specify a backend name (e.g. parallel='openmp') "
                "at this moment."
            )
            raise NotImplementedError(message)
        case "prefer_openmp":
            name = "tbb" if (
                kn.threading.tbb_available()
                and not kn.threading.openmp_available()
            ) else "openmp"
        case "prefer_tbb":
            name = "openmp" if (
                kn.threading.openmp_available()
                and not kn.threading.tbb_available()
            ) else "tbb"
        case _:
            pass
    try:
        vectorize = _registry[name]
    except KeyError as e:
        match name:
            case "openmp":
                kn.threading.OpenMPNumPyVectorize.backend_unavailable(
                    "LLVM OpenMP runtime is not installed",
                    cause=e,
                )
            case "tbb":
                kn.threading.TBBNumPyVectorize.backend_unavailable(
                    "oneTBB runtime is not installed",
                    cause=e,
                )
            case _:
                message = f"Threading backend {name!r} is unavailable"
                raise kn.threading.BackendUnavailableError(message) from e
    return _UsingBackend(name, vectorize)


def _get_or_insert_function(
    module: ir.Module,
    fnty: ir.Type,
    name: str,
) -> ir.Function:
    if name in _FUNCTIONS_TO_BE_MONKEY_PATCHED and _active_backends:
        name = f"{_active_backends[-1]}_{name}"
    return get_or_insert_function(module, fnty, name)  # pyright: ignore[reportUnknownVariableType]


cgutils.get_or_insert_function = _get_or_insert_function
_active_backends: list[str] = []
_registry = {"prefer_openmp": np.vectorize, "prefer_tbb": np.vectorize}
