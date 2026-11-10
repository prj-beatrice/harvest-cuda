"""GPU-accelerated WORLD Harvest. Requires an NVIDIA GPU; no CPU fallback."""

import os
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

# Windows does not search PATH for extension dependencies in Python 3.8+.
# Keep handles alive so the DLL search directories remain registered.
_dll_directories = []
if os.name == "nt":
    _nvidia = find_spec("nvidia")
    _roots = (_nvidia.submodule_search_locations or []) if _nvidia else []
    _paths = [Path(root) / "cu13" / directory
              for root in _roots for directory in ("bin", "bin/x86_64", "lib")]
    if os.environ.get("CUDA_PATH"):
        _paths.append(Path(os.environ["CUDA_PATH"]) / "bin")
    for _path in _paths:
        if _path.is_dir():
            _dll_directories.append(os.add_dll_directory(str(_path)))

try:
    from ._core import clear_cache, cuda_available, harvest
except ImportError as _error:
    # The relative ELF RPATH covers a normal installation. Shared system-site
    # packages or --target can put NVIDIA packages in a different directory.
    _libraries = ("libcudart.so.13", "libcufft.so.12")
    if sys.platform != "linux" or not str(_error).startswith(tuple(name + ":" for name in _libraries)):
        raise
    _nvidia = find_spec("nvidia")
    _roots = (_nvidia.submodule_search_locations or []) if _nvidia else []
    _paths = [next((Path(root) / "cu13/lib" / name for root in _roots
                    if (Path(root) / "cu13/lib" / name).is_file()), None)
              for name in _libraries]
    if any(path is None for path in _paths):
        raise
    import ctypes
    _cuda_libraries = [ctypes.CDLL(str(path)) for path in _paths]
    from ._core import clear_cache, cuda_available, harvest

__all__ = ["harvest", "cuda_available", "clear_cache"]
__version__ = version("harvest-cuda")
