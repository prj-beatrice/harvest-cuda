"""Standalone installed-wheel check; only NumPy and CUDA runtimes are needed."""
import argparse
import faulthandler
import importlib.metadata
from pathlib import Path
import sys

faulthandler.enable()

import numpy as np
from harvest_cuda import __version__, _core, clear_cache, cuda_available, harvest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    assert __version__ == importlib.metadata.version("harvest-cuda")
    # Catch accidental source-tree imports and build dependencies in the venv.
    assert Path(_core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    for name in ("nanobind", "scikit-build-core", "nvidia-cuda-nvcc", "nvidia-cuda-cccl"):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError(f"Build dependency present in runtime environment: {name}")
    for name in ("nvidia-cuda-runtime", "nvidia-cufft", "nvidia-nvjitlink"):
        importlib.metadata.distribution(name)

    fs = 16000
    t = np.arange(fs // 2) / fs
    x = sum(np.sin(2 * np.pi * 137 * h * t) / h for h in range(1, 6))
    expected = _core._reference(x, fs)
    assert np.count_nonzero(expected[0]) > 0
    clear_cache()
    if cuda_available(device=0):
        actual = harvest(x, fs, device=0)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)
        clear_cache()
        recreated = harvest(x, fs)
        for got, want in zip(recreated, actual):
            np.testing.assert_array_equal(got, want)
        clear_cache()
        print("Installed wheel: GPU execution and reference comparison passed")
    else:
        assert not args.require_cuda, "A working CUDA GPU is required for this check"
        try:
            harvest(x, fs)
        except RuntimeError as error:
            assert "no CUDA backend" not in str(error), "A CUDA-enabled wheel is required"
        else:
            raise AssertionError("GPU-only API unexpectedly succeeded without a GPU")
        print("Installed wheel: runtime loading and CPU reference passed; no GPU available")


if __name__ == "__main__":
    main()
