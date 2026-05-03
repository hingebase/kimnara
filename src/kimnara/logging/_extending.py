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

__all__ = ["get_library", "get_pointer"]

from typing import no_type_check

import llvmlite.binding  # pyright: ignore[reportMissingTypeStubs]
import spydlog as spd  # pyright: ignore[reportMissingTypeStubs]

from . import _spdlog


def get_library() -> str:
    return _spdlog.__file__


@no_type_check
def get_pointer(logger: spd.logger | None = None) -> int:
    if not logger:
        return _spdlog.get_pointer(spd.default_logger())
    # The C++ function won't perform a type check
    if not isinstance(logger, spd.logger):
        message = "Not a spydlog.logger instance"
        raise TypeError(message)
    return _spdlog.get_pointer(logger)


llvmlite.binding.load_library_permanently(get_library())  # pyright: ignore[reportUnknownMemberType]
