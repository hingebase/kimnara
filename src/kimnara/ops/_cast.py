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

__all__ = ["cast_"]

from collections.abc import Callable
from typing import TYPE_CHECKING, ForwardRef, cast, get_origin, no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
import numba.np.numpy_support  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    bytecode,
    types,
)
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    context,
)
from typing_extensions import evaluate_forward_ref

import kimnara as kn
from kimnara import _spec, _utils

if TYPE_CHECKING:
    from types import ModuleType
    from typing import cast as cast_

    from _typeshed import Unused
    from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
else:
    def cast_(typ: str, val: object) -> object:
        raise NotImplementedError

_DTYPES = np.bool_, np.integer, np.floating, np.complexfloating


@numba.extending.overload(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    cast_,
    strict=False,
    inline="always",
)
@no_type_check
def _(
    typ: types.Type,
    _: types.Type,
) -> Callable[["Unused", object], object] | None:
    # The type should always be quoted according to
    # https://docs.astral.sh/ruff/rules/runtime-cast-value/
    if not isinstance(typ, types.StringLiteral):
        return None

    @numba.extending.intrinsic
    def intrinsic(
        typingctx: context.Context,
        x: types.Type,
    ) -> "kn.typing.Intrinsic[ir.Value]":
        if x != types.intp:
            raise NotImplementedError

        # typingctx.callstack[0] is the wrapping lambda
        frame = cast("context.CallFrame", typingctx.callstack[1])

        func_id = cast("bytecode.FunctionIdentity", frame.func_id)
        annotation = evaluate_forward_ref(
            ForwardRef(cast("str", typ.literal_value)),
            owner=cast("ModuleType | None", func_id.module),  # pyright: ignore[reportAttributeAccessIssue]
        )
        inspected = _spec.TypingContext.parse(annotation).type
        if get_origin(inspected) is not kn.typing.CPointer:
            raise NotImplementedError
        [arg] = _spec.TypingContext.parse_args(inspected)
        dtype = arg.type
        if not (_utils.isclass(dtype) and issubclass(dtype, _DTYPES)):
            raise NotImplementedError
        f = types.CPointer(numba.np.numpy_support.from_dtype(dtype))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        return (  # pyright: ignore[reportUnknownVariableType]
            f(x),
            lambda context, builder, signature, args: builder.inttoptr(  # pyright: ignore[reportUnknownMemberType]
                args[0],
                context.get_value_type(signature.return_type),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            ),
        )

    # The literal string will be converted to regular string even if
    # prefer_literal=True, which is useless inside the intrinsic
    return lambda _, x: intrinsic(x)
