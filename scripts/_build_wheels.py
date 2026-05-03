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

import os
import pathlib
import platform
import runpy
import subprocess  # noqa: S404
import sys
import zlib


def main() -> None:
    if sys.platform == "darwin":
        match platform.machine():
            case "arm64":
                os.environ["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
            case "x86_64":
                # Requires Xcode <=16.4
                # https://developer.apple.com/support/xcode/
                os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"
            case _:
                raise NotImplementedError

    uv = os.getenv("UV", "uv")
    for python in "3.10", "3.11", "3.12":
        subprocess.run(  # noqa: S603
            [uv, "build", "--wheel", "-p", python, "--managed-python"],
            check=True,
        )

    dist = pathlib.Path("dist")
    if sys.platform.startswith("linux"):
        wheels = list(dist.glob("*-linux*.whl"))
        sys.argv[1:] = [
            "repair",
            "-z", str(zlib.Z_BEST_COMPRESSION),
            "-w", "dist",
            "--exclude", "libiomp5.so",
            "--exclude", "libtbb.so.12",
            *map(str, wheels),
        ]
        try:
            runpy.run_module("auditwheel", run_name="__main__")
        except SystemExit as e:
            if e.code:
                raise
        for wheel in wheels:
            wheel.unlink()

    sys.argv[1:] = map(str, dist.glob("*-abi3-*.whl"))
    runpy.run_module("abi3audit", run_name="__main__")


if __name__ == "__main__":
    main()
