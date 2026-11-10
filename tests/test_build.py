import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(os.name == "nt", reason="Linux PyPI toolkit linker layout")
@pytest.mark.parametrize("selection", ["auto", "explicit", "path", "system"])
@pytest.mark.parametrize("wheel", [False, True])
def test_build_selects_cuda_and_prepares_selected_pypi_toolkit(tmp_path, monkeypatch, selection, wheel):
    script = Path(__file__).resolve().parents[1] / "tools/build.py"
    spec = importlib.util.spec_from_file_location("harvest_build_helper", script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    toolkit = tmp_path / "nvidia/cu13"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "lib").mkdir()
    nvcc = toolkit / "bin/nvcc"
    nvcc.touch()
    for library in ("cudart", "cufft"):
        (toolkit / "lib" / f"lib{library}.so.13").touch()
    monkeypatch.setattr(helper, "find_spec", lambda name: SimpleNamespace(
        submodule_search_locations=[str(tmp_path / "other-site/nvidia"), str(tmp_path / "nvidia")]))
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "user-cache"))
    selected = str(nvcc) if selection != "system" else "/system/cuda/bin/nvcc"
    if selection in ("explicit", "system"):
        monkeypatch.setenv("CUDACXX", selected)
    monkeypatch.setattr(helper.shutil, "which", lambda name: str(nvcc) if selection == "path" else None)
    commands = []
    monkeypatch.setattr(helper.subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    monkeypatch.setattr(sys, "argv", [str(script), *(["--wheel"] if wheel else [])])
    helper.main()
    command, options = commands[0]
    assert "--config-settings=cmake.define.HARVEST_CUDA_ENABLE_CUDA=ON" in command
    assert options["env"]["UV_CACHE_DIR"] == str(tmp_path / "user-cache")
    if selection != "path":
        assert options["env"]["CUDACXX"] == selected
    for library in ("cudart", "cufft"):
        link = toolkit / "lib" / f"lib{library}.so"
        assert link.is_symlink() == (selection != "system")
        if selection != "system":
            assert link.resolve().name == f"lib{library}.so.13"
