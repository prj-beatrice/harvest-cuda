"""Exercise runtime discovery without executing GPU work."""
import builtins
import importlib.metadata
import json
from pathlib import Path
import sys
import shutil
import subprocess
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("layout", ["user-site", "target", "split", "nested", "system-only", "missing"])
def test_windows_dll_discovery(tmp_path, monkeypatch, layout):
    source = Path(__file__).resolve().parents[1] / "python/harvest_cuda/__init__.py"
    roots = [tmp_path / layout]
    if layout == "split":
        roots.append(tmp_path / "second-site")
    expected = []
    if layout not in ("missing", "system-only"):
        for i, root in enumerate(roots):
            subdir = "bin/x86_64" if layout == "nested" else ("bin" if i == 0 else "lib")
            directory = root / "nvidia/cu13" / subdir
            directory.mkdir(parents=True)
            if layout == "nested":
                expected.append(str(directory.parent))
            expected.append(str(directory))
    environment = {}
    if layout == "system-only":
        directory = tmp_path / "system-cuda/bin"
        directory.mkdir(parents=True)
        environment["CUDA_PATH"] = str(directory.parent)
        expected.append(str(directory))
    registered, handles = [], []

    def add_dll_directory(path):
        registered.append(path)
        handle = object()
        handles.append(handle)
        return handle

    fake_os = SimpleNamespace(name="nt", environ=environment, add_dll_directory=add_dll_directory)
    core = SimpleNamespace(clear_cache=object(), cuda_available=object(), harvest=object())

    def importing(name, *args, **kwargs):
        if name == "os":
            return fake_os
        if name == "_core":
            return core
        return builtins.__import__(name, *args, **kwargs)

    # Search actual namespace-package locations, outside the interpreter's
    # default purelib, with no dependence on installed NVIDIA packages.
    monkeypatch.setattr(sys, "path", [str(root) for root in roots])
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    scope = {"__name__": "harvest_cuda", "__package__": "harvest_cuda",
             "__builtins__": {**vars(builtins), "__import__": importing}}
    exec(compile(source.read_text(), str(source), "exec"), scope)
    assert registered == expected
    assert scope["_dll_directories"] == handles
    assert scope["harvest"] is core.harvest
    assert scope["__version__"] == "test-version"


@pytest.mark.parametrize("message", ["undefined symbol: unrelated_failure",
                                      "libcudart.so.13: cannot open shared object file"])
def test_import_preserves_error_without_runtime_packages(monkeypatch, message):
    source = Path(__file__).resolve().parents[1] / "python/harvest_cuda/__init__.py"
    error = ImportError(message)

    def importing(name, *args, **kwargs):
        if name == "_core":
            raise error
        if name == "os":
            return SimpleNamespace(name="posix")
        if name == "sys":
            return SimpleNamespace(platform="linux")
        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    scope = {"__name__": "harvest_cuda", "__package__": "harvest_cuda",
             "__builtins__": {**vars(builtins), "__import__": importing}}
    with pytest.raises(ImportError) as caught:
        exec(compile(source.read_text(), str(source), "exec"), scope)
    assert caught.value is error


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-library discovery")
def test_linux_runtime_in_another_site_directory(tmp_path):
    from harvest_cuda import _core

    # Copy only Harvest, leaving NVIDIA dependencies in the original site.
    # A fresh process must find those libraries without our relative RPATH.
    package = tmp_path / "harvest_cuda"
    package.mkdir()
    source = Path(__file__).resolve().parents[1] / "python/harvest_cuda/__init__.py"
    shutil.copy2(source, package / "__init__.py")
    shutil.copy2(_core.__file__, package)
    env = {key: value for key, value in os.environ.items()
           if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    env.update(PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES="")
    # Keep dependency paths, but do not run editable-install import hooks.
    subprocess.run([sys.executable, "-S", "-c", """
from pathlib import Path
import json
import sys
sys.path[:0] = [str(Path(sys.argv[1]).parent), *json.loads(sys.argv[2])]
import harvest_cuda
from harvest_cuda import _core
assert Path(_core.__file__).parent == Path(sys.argv[1])
assert not harvest_cuda.cuda_available()
harvest_cuda.clear_cache()
""", str(package), json.dumps(sys.path)], cwd=tmp_path, env=env, check=True)
