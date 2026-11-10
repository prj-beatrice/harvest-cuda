"""Build an editable install or wheel using a system or PyPI CUDA toolkit.

Install build dependencies first. This script never installs CUDA or changes
the system toolkit. It prepares missing linker files inside PyPI CUDA.
"""
import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def prepare_windows_cufft(toolkit):
    # NVIDIA's Windows cuFFT wheel ships the DLL but no import library.
    library = toolkit / "lib/x64/cufft.lib"
    dll = next((p for p in (toolkit / "bin").rglob("cufft64_*.dll")), None)
    if dll is None or (library.exists() and library.stat().st_mtime >= dll.stat().st_mtime):
        return
    exports = subprocess.check_output(["dumpbin", "/nologo", "/exports", str(dll)], text=True)
    names = re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(cufft\w+)\s*$",
                       exports, re.MULTILINE)
    if not names:
        raise RuntimeError(f"No cuFFT exports found in {dll}")
    library.parent.mkdir(parents=True, exist_ok=True)
    definition = library.with_suffix(".def")
    definition.write_text(f"LIBRARY {dll.name}\nEXPORTS\n" + "\n".join(names) + "\n")
    subprocess.run(["lib", "/nologo", "/machine:x64", f"/def:{definition}",
                    f"/out:{library}"], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wheel", action="store_true")
    p.add_argument("--arch", default="75;80;86;89;90;100;120", help="CMake CUDA architectures")
    args = p.parse_args()
    env = os.environ.copy()
    if os.name == "nt":
        # PyPI CUDA does not install Visual Studio's CUDA build integration.
        env.setdefault("CMAKE_GENERATOR", "Ninja")
    nvidia = find_spec("nvidia")
    roots = (nvidia.submodule_search_locations or []) if nvidia else []
    compilers = [Path(root) / "cu13" / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
                 for root in roots]
    nvcc = next((path for path in compilers if path.is_file()), None)
    selected_nvcc = env.get("CUDACXX") or shutil.which("nvcc")
    if not selected_nvcc and nvcc is not None:
        env["CUDACXX"] = str(nvcc)
        selected_nvcc = str(nvcc)
    # The PyPI toolkit needs linker files even when its nvcc was explicitly
    # selected through CUDACXX or PATH. Leave unrelated system toolkits alone.
    if selected_nvcc:
        compiler = Path(shutil.which(selected_nvcc) or selected_nvcc)
        if any(compiler.resolve() == path.resolve() for path in compilers):
            toolkit = compiler.resolve().parent.parent
            if os.name == "nt":
                prepare_windows_cufft(toolkit)
            else:
                for library in ["cudart", "cufft"]:
                    candidates = sorted((toolkit / "lib").glob(f"lib{library}.so.*"))
                    link = toolkit / "lib" / f"lib{library}.so"
                    if candidates and not link.exists():
                        link.symlink_to(candidates[0].name)
    flags = ["--config-settings=cmake.define.HARVEST_CUDA_ENABLE_CUDA=ON",
             f"--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES={args.arch}"]
    if args.wheel:
        command = ["uv", "build", "--wheel", "--no-build-isolation", "--python", sys.executable, *flags]
    else:
        command = ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "-e", ".", *flags]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
