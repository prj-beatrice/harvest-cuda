import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_adapters(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the standalone C++ adapter tests")
    source = Path(__file__).parent / "cpp"
    subprocess.run([cmake, "-S", str(source), "-B", str(tmp_path)], check=True)
    subprocess.run([cmake, "--build", str(tmp_path), "--config", "Release"], check=True)
    for name in ["test_cuda_device", "test_contour", "test_workspace_cache"]:
        executable = name + (".exe" if os.name == "nt" else "")
        candidates = [tmp_path / executable, tmp_path / "Release" / executable]
        subprocess.run([str(next(path for path in candidates if path.is_file()))], check=True, timeout=120)
