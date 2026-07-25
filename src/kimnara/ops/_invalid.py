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
from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core import types, typing  # pyright: ignore[reportMissingTypeStubs]

import kimnara as kn

from . import _lib


def invalid() -> int:
    frame = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    message = f"invalid value encountered in {frame.f_code.co_name}"
    raise kn.ValidationError(message)


@numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
@no_type_check
def _intrinsic(_: typing.Context) -> "kn.typing.Intrinsic[()]":
    return (
        types.intc(),
        lambda context, builder, _signature, _args: _lib.call(
            # In MSVC, `feraiseexcept` was implemented as a inline
            # function whose symbol doesn't exist in UCRT libraries
            "fesetexceptflag",
            context,
            builder,
            types.intc(types.CPointer(types.ulong), types.intc),
            [
                context.insert_unique_const(
                    builder.module,
                    ".flagp",
                    ir.Constant(
                        context.get_argument_type(types.ulong),
                        0x10000010,
                    ),
                ),
                ir.Constant(context.get_argument_type(types.intc), 0x10),
            ],
        ) if sys.platform == "win32" else _lib.call(
            "feraiseexcept",
            context,
            builder,
            types.intc(types.intc),
            [ir.Constant(context.get_argument_type(types.intc), 1)],
        ),
    )


@numba.extending.overload(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    invalid,
    inline="always",
)
@no_type_check
def _() -> Callable[[], object] | None:
    return lambda: _intrinsic()
