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

__all__ = ["c", "call"]

import ctypes
import sys
from collections.abc import Sequence

from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core import cgutils, cpu  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing.templates import (  # pyright: ignore[reportMissingTypeStubs]
    Signature,
)

if sys.platform == "win32":
    c = ctypes.cdll.ucrtbase
else:
    c = ctypes.CDLL(None)


def call(
    name: str,
    context: cpu.CPUContext,
    builder: ir.IRBuilder,
    signature: Signature,
    args: Sequence[ir.Value],
) -> ir.Instruction:
    fnty = ir.FunctionType(
        context.get_argument_type(signature.return_type),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        map(context.get_argument_type, signature.args),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    )
    fn = cgutils.get_or_insert_function(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        builder.module,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        fnty,
        name,
    )
    return builder.call(fn, args)  # pyright: ignore[reportUnknownMemberType]
