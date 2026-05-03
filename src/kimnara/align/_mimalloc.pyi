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

from typing import Final

from typing_extensions import CapsuleType

calloc_funcs: Final[int]
free_funcs: Final[int]
malloc_funcs: Final[int]
realloc_funcs: Final[int]

def set_destructor(capsule: CapsuleType, /) -> None: ...
