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

"""Test `kn.func`.

Features shared with `kn.ufunc`, `kn.gufunc` and `kn.cfunc` like input
validation, threading and pickling are also tested here.
"""

import pickle  # ruff: ignore[suspicious-pickle-import]
import sys
from typing import TYPE_CHECKING, Annotated, cast

import annotated_types as at
import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import pint
import pydantic
import pytest
from numba.core import types  # pyright: ignore[reportMissingTypeStubs]
from pint.facets.numpy.quantity import NumpyQuantity
from typing_extensions import Any

import kimnara as kn

if TYPE_CHECKING:
    from numba.core.typing.templates import (  # pyright: ignore[reportMissingTypeStubs]
        Signature,
    )


def test_input_validation() -> None:
    """Demonstrate input validation in Kimnara functions."""
    quantity = kn.quantity(1e42, "km")
    with np.errstate(invalid="raise"), pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\n.+type=int_type",
    ):
        _optional_input_nb(quantity)

    quantity = pint.UnitRegistry().Quantity(np.int64(1), "m")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    with pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\n.+different registries \[type=value_error",
    ):
        _optional_input_nb(quantity)  # pyright: ignore[reportUnknownArgumentType]

    quantity = kn.quantity(1, "s")
    with pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\s+Cannot convert from.+type=value_error",
    ):
        _optional_input_nb(quantity)

    with pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\s+Expect quantities.+type=is_instance_of",
    ):
        _optional_input_nb(1)

    with pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\s+Expect plain numbers.+type=is_instance_of",
    ):
        _parallel(quantity)

    with pytest.raises(
        pydantic.ValidationError,
        match=r"\nx\n.+safe.+type=float_type",
    ):
        _parallel(1e42)


def test_nested_tuple_nb() -> None:
    """Numba func should be able to return nested tuples."""
    [cres] = _nested_tuple_nb.dispatcher.overloads.values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return_types = cast("Signature", cres.signature).return_type  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert isinstance(return_types, types.Tuple)
    # ruff: disable[magic-value-comparison]
    assert len(return_types) == 5
    assert return_types[0] == types.Optional(types.int8)
    assert return_types[1] == types.int16
    assert return_types[2] == types.Optional(types.int32[:])
    assert return_types[3] == types.int32[::1]
    tup = return_types[4]  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(tup, types.Tuple)
    assert len(tup) == 2
    # ruff: enable[magic-value-comparison]
    assert tup[0] == types.int64
    assert tup[1] == types.Optional(types.uint8[::1])

    return_values = _nested_tuple_nb()
    assert return_values[0] is None
    assert isinstance(return_values[1], NumpyQuantity)
    assert onp.is_array_1d(return_values[2], np.int32)
    assert isinstance(return_values[3], NumpyQuantity)
    tup = return_values[4]
    assert isinstance(tup, tuple)
    assert isinstance(tup[0], int)  # Converted by Numba
    assert tup[1] is None


def test_nested_tuple_py() -> None:
    """Python func should be able to return nested tuples."""
    return_values = _nested_tuple_py()
    assert isinstance(return_values[0], np.uint16)  # pyright: ignore[reportArgumentType]
    assert isinstance(return_values[1], NumpyQuantity)
    assert return_values[2] is None
    assert isinstance(return_values[3], NumpyQuantity)
    tup = return_values[4]
    assert isinstance(tup, tuple)
    assert onp.is_array_1d(tup[0], np.complex64)
    assert isinstance(tup[1], np.complex128)  # pyright: ignore[reportArgumentType]


def test_option_precedence() -> None:
    """Argument-level options should override function-level ones."""
    _option_precedence(
        kn.empty((2, 1), np.bool_, align="avx512", pad_value=True),
        kn.empty((2, 1), np.int64, align="avx", pad_value=1),
        kn.empty((2, 1), np.float64, align="avx2", pad_value=1),
        kn.empty((2, 255), np.complex128, align="mkl", pad_value=0),
    )


def test_optional_input_nb() -> None:
    """Numba func should allow `Optional[T]` inputs."""
    [[arg]] = _optional_input_nb.dispatcher.overloads  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert arg == types.Optional(types.int64)
    assert _optional_input_nb(None)
    quantity = kn.quantity(0, "m")
    assert not _optional_input_nb(quantity)


def test_optional_input_py() -> None:
    """Python func should allow `Optional[T]` inputs."""
    assert _optional_input_py(None)
    quantity = kn.quantity(np.empty(1), "m")
    assert not _optional_input_py(quantity)


def test_parallel() -> None:
    """Numba func should be affected by the threading monkey patch."""
    llvm = _parallel.inspect_llvm()
    assert "@workqueue_get_num_threads" in llvm
    assert "@workqueue_get_thread_id" in llvm
    assert "@workqueue_numba_parallel_for" in llvm
    assert "@get_num_threads" not in llvm
    assert "@get_thread_id" not in llvm
    assert "@numba_parallel_for" not in llvm
    asm = _parallel.inspect_asm()
    prefix = "$_" if sys.platform == "darwin" else "$"
    assert f"{prefix}workqueue_get_num_threads" in asm
    assert f"{prefix}workqueue_get_thread_id" in asm
    assert f"{prefix}workqueue_numba_parallel_for" in asm
    assert f"{prefix}get_num_threads" not in asm
    assert f"{prefix}get_thread_id" not in asm
    assert f"{prefix}numba_parallel_for" not in asm


def test_pickle() -> None:
    """Functions should support pickling."""
    assert pickle.loads(pickle.dumps(_parallel)) is _parallel  # ruff: ignore[suspicious-pickle-usage]


def _base(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    base = x.base  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert base is not None
    return base  # pyright: ignore[reportUnknownVariableType]


@kn.func(cache=False, nopython=True)
def _nested_tuple_nb() -> tuple[
    np.int8 | None,
    Annotated[np.int16, at.Unit("dimensionless")],
    onp.Array1D[np.int32] | None,
    Annotated[onp.Array1D[np.int32], at.Unit("dimensionless"), kn.C],
    tuple[np.int64, Annotated[onp.Array1D[np.uint8], kn.C] | None],
]:
    arr = np.ones(2, np.int32)
    return None, np.int16(2), arr, arr, (np.int64(3), None)


@kn.func
def _nested_tuple_py() -> tuple[
    np.uint16 | None,
    Annotated[np.uint32, at.Unit("dimensionless")],
    np.ndarray[tuple[int, ...], np.dtype[np.uint64]] | None,
    Annotated[onp.Array1D[np.complex64], at.Unit("dimensionless")],
    tuple[npt.NDArray[np.complex64], np.complex128 | None],
]:
    arr = np.ones(2, np.complex64)
    return np.uint16(2), np.uint32(3), None, arr, (arr, np.complex128(4))


@kn.func(align=kn.AVX, pad_value=0)
def _option_precedence(
    default: onp.Array2D[np.bool_],
    override_align: Annotated[onp.Array2D[np.int64], kn.SSE],
    override_pad_value: Annotated[onp.Array2D[np.float64], kn.Pad(None)],
    override_both: Annotated[onp.Array2D[np.complex128], kn.AVX512, kn.Pad(1)],
) -> None:
    assert kn.isaligned(default, kn.AVX)
    assert not kn.isaligned(default, kn.AVX512)
    np.testing.assert_equal(_base(default)[:, 1:], 0)

    assert kn.isaligned(override_align, kn.SSE)
    assert not kn.isaligned(override_align, kn.AVX)
    np.testing.assert_equal(_base(override_align)[:, 1:], 0)

    assert kn.isaligned(override_pad_value, kn.AVX)
    assert not kn.isaligned(override_pad_value, kn.AVX512)
    np.testing.assert_equal(_base(override_pad_value)[:, 1:], 1)

    assert kn.isaligned(override_both, kn.AVX512)
    assert not kn.isaligned(override_both, kn.Alignment.MKL)
    np.testing.assert_equal(_base(override_both)[:, 255:], 1)


@kn.func(cache=False, nopython=True)
def _optional_input_nb(x: Annotated[np.int64, at.Unit("m")] | None) -> bool:
    return x is None


@kn.func
def _optional_input_py(
    x: Annotated[onp.Array1D[np.float64], at.Unit("m")] | None,
) -> bool:
    return x is None


@kn.func(cache=False, nopython=True, parallel="workqueue")
def _parallel(x: onp.Array1D[np.float32]) -> np.float32:
    return x.sum()  # pyright: ignore[reportUnknownMemberType]
