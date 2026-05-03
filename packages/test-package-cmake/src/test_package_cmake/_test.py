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

__all__ = [
    "test_logging_capi",
]

import ctypes
import importlib.metadata
import sys

import kimnara as kn


def test_logging_capi(logger: object) -> None:
    dll = ctypes.cdll[_locate_dll()]
    func = ctypes.CFUNCTYPE(None, ctypes.c_size_t)(("TestLoggingCAPI", dll))
    if not isinstance(logger, int):
        logger = kn.logging.get_pointer(logger)  # pyright: ignore[reportUnknownMemberType]
    func(logger)


def _locate_dll() -> str:
    match sys.platform:
        case "darwin":
            name = "libtest.dylib"
        case "win32":
            name = "test.dll"
        case p if p.startswith("linux"):
            name = "libtest.so"
        case _:
            raise NotImplementedError
    if paths := importlib.metadata.files("test-package-cmake"):
        for p in paths:
            if p.name == name:
                return str(p.locate())
    raise FileNotFoundError(name)
