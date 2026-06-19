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

__all__ = ["NumPyVectorize", "add_dll_directory", "check_num_threads"]

import abc
import contextlib
import ctypes
import importlib.metadata
import math
import os
import pathlib
import platform
import sys
import sysconfig
import threading
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol

import numpy as np
import numpy.typing as npt
from numpy_typing_compat import NUMPY_GE_2_0
from typing_extensions import Unpack

import kimnara as kn

_ArrayOrScalar = npt.NDArray[Any] | np.generic


class NumPyVectorize(np.vectorize, abc.ABC):
    _in_and_out_core_dims: tuple[list[tuple[str, ...]], list[tuple[str, ...]]]

    # https://github.com/numpy/numpy/pull/30293/changes
    if TYPE_CHECKING:
        __module__: Literal["numpy"] = "numpy"

    @classmethod
    def backend_site_package_available(cls) -> bool:
        name = cls.backend_site_package_name()
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            return False
        return True

    @classmethod
    @abc.abstractmethod
    def backend_site_package_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def backend_unavailable(
        cls,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> NoReturn:
        if not (
            "AMD64" != platform.machine() != "x86_64"
            or pathlib.Path(sys.prefix, "conda-meta").is_dir()
            or cls.backend_site_package_available()
        ):
            message = "Install kimnara[nonfree] to enable threading"
        raise kn.threading.BackendUnavailableError(message) from cause

    @abc.abstractmethod
    def np_vectorize_impl(self, func: Callable[[int], None], n: int) -> None:
        raise NotImplementedError

    def _vectorize_call_with_signature(
        self,
        func: Callable[
            [Unpack[tuple[Any, ...]]],
            _ArrayOrScalar | tuple[_ArrayOrScalar, ...],
        ],
        args: Sequence[npt.ArrayLike],
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], ...]:
        input_core_dims, output_core_dims = self._in_and_out_core_dims

        if len(args) != len(input_core_dims):
            msg = "wrong number of positional arguments: expected %r, got %r"
            raise TypeError(msg % (len(input_core_dims), len(args)))
        args = tuple(np.asanyarray(arg) for arg in args)

        broadcast_shape, dim_sizes = _function_base._parse_input_dimensions(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            args, input_core_dims)
        input_shapes = _function_base._calculate_shapes(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            broadcast_shape, dim_sizes, input_core_dims)
        args = [np.broadcast_to(arg, shape, subok=True)
                for arg, shape in zip(args, input_shapes, strict=True)]

        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)

        lock = threading.Lock()
        null = contextlib.nullcontext()

        def wrapper(i: int) -> None:
            index = np.unravel_index(i, broadcast_shape)  # pyright: ignore[reportUnknownMemberType]
            results = func(*(arg[index] for arg in args))

            if not isinstance(results, tuple):
                results = (results,)
            n_results = len(results)

            if nout != n_results:
                m = "wrong number of outputs from pyfunc: expected %r, got %r"
                raise ValueError(m % (nout, n_results))

            nonlocal lock, outputs
            with lock:
                if outputs is None:
                    for result, core_dims in zip(
                        results,
                        output_core_dims,
                        strict=True,
                    ):
                        _function_base._update_dim_sizes(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            dim_sizes, result, core_dims)

                    outputs = _function_base._create_arrays(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        broadcast_shape, dim_sizes, output_core_dims, otypes,
                        results)
                    lock = null

            for output, result in zip(outputs, results, strict=True):
                output[index] = result

        self.np_vectorize_impl(wrapper, math.prod(broadcast_shape))
        if outputs is None:
            outputs = _last_resort(
                broadcast_shape, dim_sizes, output_core_dims, otypes)
        return outputs[0] if nout == 1 else outputs


def add_dll_directory() -> contextlib.AbstractContextManager[object]:
    if sys.platform == "win32":
        data = pathlib.Path(sysconfig.get_path("data"))
        prefix = os.getenv("CONDA_PREFIX")
        if not prefix or not data.samefile(prefix):
            return os.add_dll_directory(str(data / "Library/bin"))
    return contextlib.nullcontext()


class _Module(Protocol):
    @property
    def get_num_threads(self) -> int: ...


def check_num_threads(mod: _Module) -> None:
    if _get_num_threads(mod) <= 1:
        message = (
            "You only have access to one CPU core "
            "and probably wish not to enable threading"
        )
        warnings.warn(message, kn.PerformanceWarning, stacklevel=7)


def _get_num_threads(mod: _Module) -> int:
    return ctypes.CFUNCTYPE(ctypes.c_int)(mod.get_num_threads)()


def _last_resort(
    broadcast_shape: tuple[int, ...],
    dim_sizes: dict[str, int],
    output_core_dims: list[tuple[str, ...]],
    otypes: Sequence[npt.DTypeLike] | None,
) -> tuple[npt.NDArray[Any], ...]:
    if otypes is None:
        msg = "cannot call `vectorize` on size 0 inputs unless `otypes` is set"
        raise ValueError(msg)
    if any(dim not in dim_sizes for dims in output_core_dims for dim in dims):
        msg = (
            "cannot call `vectorize` with a signature including new output "
            "dimensions on size 0 inputs"
        )
        raise ValueError(msg)
    return _function_base._create_arrays(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        broadcast_shape, dim_sizes, output_core_dims, otypes)


class _FunctionBase(Protocol):
    def _calculate_shapes(
        self,
        broadcast_shape: tuple[int, ...],
        dim_sizes: dict[str, int],
        list_of_core_dims: list[tuple[str, ...]],
    ) -> list[tuple[int, ...]]: ...

    def _create_arrays(
        self,
        broadcast_shape: tuple[int, ...],
        dim_sizes: dict[str, int],
        list_of_core_dims: list[tuple[str, ...]],
        dtypes: Sequence[npt.DTypeLike] | None,
        results: tuple[_ArrayOrScalar, ...] | None = ...,
    ) -> tuple[npt.NDArray[Any], ...]: ...

    def _parse_input_dimensions(
        self,
        args: tuple[npt.NDArray[Any], ...],
        input_core_dims: list[tuple[str, ...]],
    ) -> tuple[tuple[int, ...], dict[str, int]]: ...

    def _update_dim_sizes(
        self,
        dim_sizes: dict[str, int],
        arg: _ArrayOrScalar,
        core_dims: tuple[str, ...],
    ) -> None: ...


_function_base: _FunctionBase = getattr(
    np.lib,
    "_function_base_impl" if NUMPY_GE_2_0 else "function_base",
)
