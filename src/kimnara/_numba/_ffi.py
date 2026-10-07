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

__all__ = ["call", "get_cython_function_address"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core import cgutils, cpu, types  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing.context import (  # pyright: ignore[reportMissingTypeStubs]
    Context,
)

if TYPE_CHECKING:
    import kimnara as kn


def call(
    name: str,
    builder: ir.IRBuilder,
    fnty: ir.FunctionType,
    args: Sequence[ir.Value],
) -> ir.Instruction:
    fn = cgutils.get_or_insert_function(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        builder.module,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        fnty,
        name,
    )
    return builder.call(fn, args)  # pyright: ignore[reportUnknownMemberType]


def get_cython_function_address(
    module_name: str,
    function_name: str,
    context: cpu.CPUContext,
    builder: ir.IRBuilder,
) -> ir.Instruction:
    c_module_name = context.insert_const_string(builder.module, module_name)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    pyapi = context.get_python_api(builder)  # pyright: ignore[reportUnknownMemberType]
    gil_state = cast("ir.Instruction", pyapi.gil_ensure())
    module = cast("ir.Instruction", pyapi.import_module(c_module_name))  # pyright: ignore[reportUnknownMemberType]
    pyx_capi = cast(
        "ir.Instruction",
        pyapi.object_getattr_string(module, "__pyx_capi__"),  # pyright: ignore[reportUnknownMemberType]
    )
    pyapi.decref(module)  # pyright: ignore[reportUnknownMemberType]
    capsule = cast(
        "ir.Instruction",
        pyapi.dict_getitem_string(pyx_capi, function_name),  # pyright: ignore[reportUnknownMemberType]
    )
    pyapi.decref(pyx_capi)  # pyright: ignore[reportUnknownMemberType]
    capsule_name = call(
        "PyCapsule_GetName",
        builder,
        ir.FunctionType(pyapi.cstring, [pyapi.pyobj]),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        [capsule],
    )
    fnty = ir.FunctionType(
        context.get_argument_type(types.uintp),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        [pyapi.pyobj, pyapi.cstring],  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    )
    address = call(
        "PyCapsule_GetPointer",
        builder,
        fnty,
        [capsule, capsule_name],
    )
    pyapi.gil_release(gil_state)  # pyright: ignore[reportUnknownMemberType]
    return address


@numba.extending.overload(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    numba.extending.get_cython_function_address,  # pyright: ignore[reportUnknownMemberType]
    strict=False,
)
@no_type_check
def _(module_name: types.Type, function_name: types.Type) -> object:
    if not isinstance(module_name, types.StringLiteral):
        return None
    if not isinstance(function_name, types.StringLiteral):
        return None

    @numba.extending.intrinsic
    @no_type_check
    def intrinsic(_: Context) -> "kn.typing.Intrinsic[()]":
        return (
            types.uintp(),
            lambda context, builder, _sig, _args: get_cython_function_address(
                module_name.literal_value,
                function_name.literal_value,
                context,
                builder,
            ),
        )

    return lambda _module_name, _function_name: intrinsic()
