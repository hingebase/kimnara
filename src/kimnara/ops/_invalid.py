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

__all__ = ["invalid"]

import sys
from collections.abc import Callable
from typing import no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from numba.core import types  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing.context import (  # pyright: ignore[reportMissingTypeStubs]
    Context,
)

import kimnara as kn
from kimnara import _numba


def invalid() -> int:
    frame = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    message = f"invalid value encountered in {frame.f_code.co_name}"
    raise kn.ValidationError(message)


@numba.extending.intrinsic  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
@no_type_check
def _intrinsic(_: Context) -> "kn.typing.Intrinsic[()]":
    return (
        types.intc(),
        lambda context, builder, _signature, _args: _numba.call(
            # In MSVC, `feraiseexcept` was implemented as a inline
            # function whose symbol doesn't exist in UCRT libraries
            "fesetexceptflag",
            context,
            builder,
            types.intc(types.CPointer(types.ulong), types.intc),
            [
                context.insert_unique_const(
                    builder.module,
                    ".const.flagp",
                    context.get_constant(types.ulong, 0x10000010),
                ),
                context.get_constant(types.intc, 0x10),
            ],
        ) if sys.platform == "win32" else _numba.call(
            "feraiseexcept",
            context,
            builder,
            types.intc(types.intc),
            [context.get_constant(types.intc, 1)],
        ),
    )


@numba.extending.overload(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    invalid,
    inline="always",
)
@no_type_check
def _() -> Callable[[], object] | None:
    return lambda: _intrinsic()
