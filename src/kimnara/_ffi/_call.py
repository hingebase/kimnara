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

__all__ = ["c", "call", "cython", "locate"]

import ast
import ctypes
import functools
import inspect
import sys
import textwrap
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import cast, no_type_check

import executing
import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from llvmlite import binding  # pyright: ignore[reportMissingTypeStubs]
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    bytecode,
    ir,
    registry,
    typeinfer,
    types,
)
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    context,
    ctypes_utils,
    templates,
    typeof,
)
from typing_extensions import Any, TypeVar

import kimnara as kn
from kimnara import _numba, _spec, _utils
from kimnara._typing import VarArgsIntrinsic

if sys.version_info >= (3, 14):
    try:
        from ctypes import c_double_complex, c_float_complex

        from numba.core.typing.ctypes_utils import (  # pyright: ignore[reportMissingTypeStubs]
            _FROM_CTYPES,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # ruff: ignore[import-private-name]
            _TO_CTYPES,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # ruff: ignore[import-private-name]
        )
    except ImportError:
        pass
    else:
        _FROM_CTYPES[c_float_complex] = types.complex64  # pyright: ignore[reportArgumentType]
        _FROM_CTYPES[c_double_complex] = types.complex128  # pyright: ignore[reportArgumentType]
        _TO_CTYPES[types.complex64] = c_float_complex  # pyright: ignore[reportArgumentType]
        _TO_CTYPES[types.complex128] = c_double_complex  # pyright: ignore[reportArgumentType]

_Inferable = ast.AnnAssign | ast.Expr  # | ast.Return
_T = TypeVar("_T", bound="ctypes._CData")  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
_Target = ast.Attribute | ast.Name


def c(
    *args: types.Type,
    lib: types.Type | None = None,
    function_name: types.Type | None = None,
) -> Callable[..., object] | None:
    if function_name and not isinstance(function_name, types.StringLiteral):
        return None
    if lib:
        if not isinstance(lib, types.StringLiteral):
            return None
        libname = cast("str", lib.literal_value)
        if "c" != libname != "m":
            binding.load_library_permanently(libname)  # pyright: ignore[reportUnknownMemberType]
    names, signature_holder = _unpack(args)

    @numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
    @functools.wraps(signature_holder, assigned=())
    def intrinsic(
        typingctx: context.Context,
        *args: types.Type,
    ) -> VarArgsIntrinsic:
        restype, function = _resolve(typingctx, function_name)
        return (  # pyright: ignore[reportUnknownVariableType]
            restype(*map(_to_numba, args)),
            lambda context, builder, signature, args: _numba.call(
                function, context, builder, signature, args),
        )

    # https://github.com/numba/numba/issues/5291
    positional = [names] if names else []
    if lib:
        positional.append("lib")
    if function_name:
        positional.append("function_name")
    return eval(f"lambda {','.join(positional)}: f({names})", {"f": intrinsic})  # ruff: ignore[suspicious-eval-usage]


def call(
    op: executing.executing.Executing,
    args: tuple[object, ...],
    func_spec: int | tuple[str, ctypes.CDLL],
) -> object:
    restype = _restype_ctypes(op)
    cargs, cargtypes = zip(*map(_to_ctypes, args), strict=True)
    cfunc = ctypes.CFUNCTYPE(restype, *cargtypes)(func_spec)  # pyright: ignore[reportUnknownArgumentType]
    return cfunc(*cargs)


def cython(
    *args: types.Type,
    module: types.Type,
    function_name: types.Type | None = None,
) -> Callable[..., object] | None:
    if function_name and not isinstance(function_name, types.StringLiteral):
        return None
    match module:
        case types.Module():
            module_name = cast("str", module.pymod.__name__)  # pyright: ignore[reportUnknownMemberType]
        case types.StringLiteral():
            module_name = cast("str", module.literal_value)
        case _:
            return None
    names, signature_holder = _unpack(args)

    @numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
    @functools.wraps(signature_holder, assigned=())
    def intrinsic(
        typingctx: context.Context,
        *args: types.Type,
    ) -> VarArgsIntrinsic:
        restype, function = _resolve(typingctx, function_name)
        return (  # pyright: ignore[reportUnknownVariableType]
            restype(*map(_to_numba, args)),
            lambda context, builder, signature, args: _numba.call(
                _numba.get_cython_function_address(
                    module_name,
                    function,
                    context,
                    builder,
                ),
                context,
                builder,
                signature,
                args,
            ),
        )

    positional = [names, "module"] if names else ["module"]
    if function_name:
        positional.append("function_name")
    return eval(f"lambda {','.join(positional)}: f({names})", {"f": intrinsic})  # ruff: ignore[suspicious-eval-usage]


def locate() -> executing.executing.Executing:
    op = executing.Source.executing(sys._getframe(2))  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    node = op.node
    if not isinstance(node, ast.Call):
        message = "Cannot locate the caller"
        if node:
            message = f"{message}:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    if not isinstance(node.parent, _Inferable):
        message = f"Cannot infer the return type:\n{ast.unparse(node.parent)}"
        raise kn.TypeInferenceError(message)
    return op


@functools.singledispatch
def _expr(node: ast.AST) -> str | ast.Expression:
    message = f"Invalid annotation:\n{ast.unparse(node)}"
    raise kn.TypeInferenceError(message)


@_expr.register(ast.Attribute)
@_expr.register(ast.Name)
@_expr.register(ast.Subscript)
def _(node: _Target | ast.Subscript) -> ast.Expression:
    return ast.Expression(node)


@_expr.register
def _(node: ast.Constant) -> str:
    value = node.value
    if not isinstance(value, str):
        message = f"Invalid annotation:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    return value


def _locate(inferrer: typeinfer.TypeInferer) -> typeinfer.CallConstraint:
    candidates = {
        id(constr): constr
        for _, constr in inferrer.calls  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(constr, typeinfer.CallConstraint)
    }
    frame = inspect.currentframe()
    while frame:
        try:
            return candidates[id(frame.f_locals["self"])]
        except KeyError:
            pass
        frame = frame.f_back
    message = "Cannot locate the caller"
    raise kn.TypeInferenceError(message)


def _resolve(
    typingctx: context.Context,
    function_name: types.StringLiteral | None,
) -> tuple[types.Type, str]:
    frame = cast("context.CallFrame", typingctx.callstack[1])
    inferrer = cast("typeinfer.TypeInferer", frame.typeinfer)
    constr = _locate(inferrer)
    loc = cast("ir.Loc", constr.loc)
    line = cast("int", loc.line)
    func_id = cast("bytecode.FunctionIdentity", frame.func_id)
    caller = cast("Callable[..., Any]", func_id.func)  # pyright: ignore[reportAttributeAccessIssue]
    lines, lnum = inspect.getsourcelines(caller)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    node = _visit(tree, line - lnum + 1)
    restype = _restype_numba(node, constr.target, inferrer)
    if function_name:
        function = cast("str", function_name.literal_value)
    else:
        function = caller.__code__.co_name
    return restype, function


def _restype_ctypes(
    op: executing.executing.Executing,
) -> type["ctypes._CData"] | None:  # pyright: ignore[reportPrivateUsage]
    node = op.node.parent
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, _Target):
            message = f"Invalid statement:\n{ast.unparse(node)}"
            raise kn.TypeInferenceError(message)
        expr = _expr(node.annotation)
        code = compile(expr, op.source.filename, "eval")
        ann = eval(code, op.frame.f_globals)  # ruff: ignore[suspicious-eval-usage]
        ctx = _spec.TypingContext(align=kn.A, allow_scalar=True)
        return ctx.infer(ann).to_ctypes()
    return None


@functools.singledispatch
def _restype_numba(
    node: ast.AST,
    target: str,
    inferrer: typeinfer.TypeInferer,
) -> types.Type:
    raise NotImplementedError


@_restype_numba.register
def _(
    node: ast.AnnAssign,
    target: str,
    inferrer: typeinfer.TypeInferer,
) -> types.Type:
    if not isinstance(node.value, ast.Call):
        message = f"Cannot locate the caller:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    match node.target:
        case ast.Attribute(attr=attr):
            for _, constr in inferrer.calls:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if (
                    isinstance(constr, typeinfer.SetAttrConstraint)
                    and constr.attr == attr  # pyright: ignore[reportUnknownMemberType]
                    and cast("ir.Var", constr.value).name == target
                ):
                    break
            else:
                msg = f"Cannot locate the caller:\n{ast.unparse(node.value)}"
                raise kn.TypeInferenceError(msg)
        case ast.Name(id=name):
            if target != name and not target.startswith(f"{name}."):
                msg = f"Cannot locate the caller:\n{ast.unparse(node.value)}"
                raise kn.TypeInferenceError(msg)
        case _:
            message = f"Invalid annotation:\n{ast.unparse(node)}"
            raise kn.TypeInferenceError(message)
    func_id = cast("bytecode.FunctionIdentity", inferrer.func_id)
    expr = _expr(node.annotation)
    filename = cast("str", func_id.filename)  # pyright: ignore[reportAttributeAccessIssue]
    code = compile(expr, filename, "eval")
    module = cast("ModuleType", func_id.module)  # pyright: ignore[reportAttributeAccessIssue]
    ann = eval(code, vars(module))  # ruff: ignore[suspicious-eval-usage]
    ctx = _spec.TypingContext(align=kn.A, allow_scalar=True)
    return ctx.infer(ann).to_numba()


@_restype_numba.register
def _(
    node: ast.Expr,
    target: str,
    inferrer: typeinfer.TypeInferer,
) -> types.NoneType:
    if not isinstance(node.value, ast.Call):
        message = f"Cannot locate the caller:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    func_ir = cast("ir.FunctionIR", inferrer.func_ir)
    expr = cast("ir.Expr", func_ir.get_definition(target))  # pyright: ignore[reportUnknownMemberType]
    refcount = 3  # ir.Assign + sys.getrefcount + ?
    if sys.getrefcount(func_ir.get_assignee(expr)) > refcount:  # pyright: ignore[reportUnknownMemberType]
        message = f"Cannot locate the caller:\n{ast.unparse(node.value)}"
        raise kn.TypeInferenceError(message)
    return types.void


@functools.singledispatch
@no_type_check
def _to_ctypes(arg: object) -> tuple[Any, type[Any]]:
    try:
        # Make the type conversion consistent in ctypes and Numba
        argtype = typeof.typeof(arg)
    except ValueError:
        pass
    else:
        try:
            cargtype = ctypes_utils.to_ctypes(argtype)
        except TypeError:
            pass
        else:
            return arg, cargtype
    try:
        wrapped = arg._as_parameter_
    except AttributeError:
        pass
    else:
        return _to_ctypes(wrapped)
    message = f"Cannot convert {_utils.base_repr(arg)} to ctypes"
    raise kn.TypeInferenceError(message)


@_to_ctypes.register
def _(_: None) -> tuple[None, type[ctypes.c_void_p]]:
    return None, ctypes.c_void_p


@_to_ctypes.register(ctypes.Array.__mro__[-2])
def _(carg: _T) -> tuple[_T, type[_T]]:
    return carg, type(carg)


@functools.singledispatch
def _to_numba(argtype: types.Type) -> types.Type:
    return argtype


@_to_numba.register
def _(argtype: types.ArrayCTypes) -> types.CPointer:
    return types.CPointer(argtype.dtype)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


@_to_numba.register
def _(argtype: types.Dispatcher) -> types.ExternalFunctionPointer:
    dispatcher = cast("registry.CPUDispatcher", argtype.dispatcher)
    [cres] = dispatcher.overloads.values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sig = cast("templates.Signature", cres.signature)
    return types.ExternalFunctionPointer(sig, None)


@_to_numba.register
def _(argtype: types.FunctionType) -> types.ExternalFunctionPointer:
    ftype = argtype.get_precise()
    return types.ExternalFunctionPointer(ftype.signature, None, ftype.cconv)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


@_to_numba.register
def _(_: types.NoneType) -> types.RawPointer:
    return types.voidptr


def _unpack(args: tuple[types.Type, ...]) -> tuple[str, Any]:
    names: list[str] = []
    params = [_typingctx]
    for i in range(len(args)):
        name = f"arg{i}"
        names.append(name)
        params.append(inspect.Parameter(name, _typingctx.POSITIONAL_ONLY))
    sig = inspect.Signature(params)
    return ",".join(names), SimpleNamespace(__signature__=sig)


def _visit(tree: ast.Module, lineno: int) -> _Inferable:
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, _Inferable) and node.lineno == lineno
    ]
    try:
        [node] = found
    except ValueError as e:
        message = "Cannot locate the caller"
        raise kn.TypeInferenceError(message) from e
    return node


_typingctx = inspect.Parameter("typingctx", inspect.Parameter.POSITIONAL_ONLY)
