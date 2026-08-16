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

"""Test unit conversion."""

import math
from typing import cast

import llvmlite.binding  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import optype.numpy as onp
import pint.facets
import pytest
from typing_extensions import Any, Sentinel

_UnitRegistry = pint.registry.GenericUnitRegistry[
    pint.facets.plain.PlainQuantity[Any],
    pint.Unit,
]
MISSING = Sentinel("MISSING")


@pytest.mark.skipif(
    not llvmlite.binding.get_host_cpu_features().get("fma"),  # pyright: ignore[reportUnknownMemberType]
    reason="The monkey patch is not applicable",
)
def test_converters(subtests: pytest.Subtests) -> None:
    """Our monkey patch should be in compliance with Pint.

    In `kn._spec._numpy._units._converters` we replace the default
    logarithmic and offset converters with FMA-enabled implementations.
    They are expected to be faster for large arrays on modern CPUs.
    """
    ureg = cast("_UnitRegistry", pint.UnitRegistry())
    default = cast("_UnitRegistry", pint.get_application_registry().get())
    for unit in "degF", "dBm":
        # ruff: disable[private-member-access]
        old = ureg._units[unit].converter  # pyright: ignore[reportPrivateUsage]
        new = default._units[unit].converter  # pyright: ignore[reportPrivateUsage]
        # ruff: enable[private-member-access]
        assert type(new) is not type(old)
        for attr in "scale", "offset", "logbase", "logfactor":
            assert getattr(new, attr, MISSING) == getattr(old, attr, MISSING)

    rng = np.random.default_rng(np.random.Philox())
    with subtests.test("Real non-multiplicative quantities"):
        _real_non_multiplicative_quantities(ureg, default, rng)
    with subtests.test("Complex non-multiplicative quantities"):
        _complex_non_multiplicative_quantities(ureg, default, rng)


def _real_non_multiplicative_quantities(
    ureg: _UnitRegistry,
    default: _UnitRegistry,
    rng: np.random.Generator,
) -> None:
    x = rng.lognormal(mean=10., sigma=10., size=1000)
    x[0] = math.nan
    # It's just a bad idea to express a near-zero temperature in degF
    for from_, to, atol in [("K", "degF", 2e-14), ("W", "dBm", .0)]:
        actual = default.Quantity(x, from_).m_as(to)  # pyright: ignore[reportUnknownMemberType]
        desired = ureg.Quantity(x, from_).m_as(to)  # pyright: ignore[reportUnknownMemberType]
        assert onp.is_array_1d(actual, np.float64)
        assert onp.is_array_1d(desired, np.float64)
        np.testing.assert_allclose(actual, desired)
        actual = default.Quantity(desired, to).m_as(from_)  # pyright: ignore[reportUnknownMemberType]
        desired = ureg.Quantity(desired, to).m_as(from_)  # pyright: ignore[reportUnknownMemberType]
        assert onp.is_array_1d(actual, np.float64)
        assert onp.is_array_1d(desired, np.float64)
        np.testing.assert_allclose(actual, desired, atol=atol)

    x = x.astype(np.float32)
    # Much more tolerance is required for float32
    # Again, it's developers' responsibility to avoid storing tiny
    # numbers as float32
    for from_, to, atol in [("K", "degF", 3e-5), ("W", "dBm", 3e-6)]:
        actual = default.Quantity(x, from_).m_as(to)  # pyright: ignore[reportUnknownMemberType]
        quantity = ureg.Quantity(x, from_)
        if to == "dBm":
            # LogarithmicConverter calls `np.log(builtins.float)`
            # Hence the float32 -> float64 upcast in `quantity.to()`
            quantity.ito(to)  # pyright: ignore[reportUnknownMemberType]
            desired = quantity.magnitude
        else:
            desired = quantity.m_as(to)  # pyright: ignore[reportUnknownMemberType]
        assert onp.is_array_1d(actual, np.float32)
        assert onp.is_array_1d(desired, np.float32)
        np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=atol)
        actual = default.Quantity(desired, to).m_as(from_)  # pyright: ignore[reportUnknownMemberType]
        quantity = ureg.Quantity(desired, to)
        if to == "dBm":
            quantity.ito(from_)  # pyright: ignore[reportUnknownMemberType]
            desired = quantity.magnitude
        else:
            desired = quantity.m_as(from_)  # pyright: ignore[reportUnknownMemberType]
        assert onp.is_array_1d(actual, np.float32)
        assert onp.is_array_1d(desired, np.float32)
        np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=atol)


def _complex_non_multiplicative_quantities(
    ureg: _UnitRegistry,
    default: _UnitRegistry,
    rng: np.random.Generator,
) -> None:
    x = 1j * rng.uniform(-.5 * math.pi, .5 * math.pi, size=1000)
    np.exp(x, x)
    x *= rng.lognormal(mean=10., sigma=10., size=1000)
    # Unlike the real number case, np.log will report invalid values
    # where either the real part or the imaginary part is NaN

    # We omitted the test for OffsetConverter as there is no known
    # meaningful quantity to use
    actual = default.Quantity(x, "W").m_as("dBm")  # pyright: ignore[reportUnknownMemberType]
    desired = ureg.Quantity(x, "W").m_as("dBm")  # pyright: ignore[reportUnknownMemberType]
    assert onp.is_array_1d(actual, np.complex128)
    assert onp.is_array_1d(desired, np.complex128)
    np.testing.assert_allclose(actual, desired)
    actual = default.Quantity(desired, "dBm").m_as("W")  # pyright: ignore[reportUnknownMemberType]
    desired = ureg.Quantity(desired, "dBm").m_as("W")  # pyright: ignore[reportUnknownMemberType]
    assert onp.is_array_1d(actual, np.complex128)
    assert onp.is_array_1d(desired, np.complex128)
    np.testing.assert_allclose(actual, desired)

    x = x.astype(np.complex64)
    actual = default.Quantity(x, "W").m_as("dBm")  # pyright: ignore[reportUnknownMemberType]
    quantity = ureg.Quantity(x, "W")
    quantity.ito("dBm")  # pyright: ignore[reportUnknownMemberType]
    desired = quantity.magnitude
    assert onp.is_array_1d(actual, np.complex64)
    assert onp.is_array_1d(desired, np.complex64)
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=3e-6)
    actual = default.Quantity(desired, "dBm").m_as("W")  # pyright: ignore[reportUnknownMemberType]
    quantity = ureg.Quantity(desired, "dBm")
    quantity.ito("W")  # pyright: ignore[reportUnknownMemberType]
    desired = quantity.magnitude
    assert onp.is_array_1d(actual, np.complex64)
    assert onp.is_array_1d(desired, np.complex64)
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=3e-6)
