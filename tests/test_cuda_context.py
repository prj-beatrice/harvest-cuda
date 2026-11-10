"""Check CUDA context interoperability and shared cache cleanup on a real GPU."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import os

import numpy as np
import pytest
from harvest_cuda import _core
from signals import signal


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("warm_first", [False, True])
@pytest.mark.parametrize("seconds", [0.3, 2.1])
@pytest.mark.parametrize("device", [0, 1])
def test_custom_context_and_shared_cache_cleanup(warm_first, seconds, device):
    if not _core.cuda_available(device=device):
        pytest.skip(f"Supported visible GPU {device} unavailable")
    driver = ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
    driver.cuCtxCreate_v2.argtypes = [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int]
    driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
    driver.cuCtxSetCurrent.argtypes = [ct.c_void_p]
    driver.cuCtxDestroy_v2.argtypes = [ct.c_void_p]

    def check(result):
        assert result == 0, f"CUDA driver error {result}"

    def current():
        result = ct.c_void_p()
        check(driver.cuCtxGetCurrent(ct.byref(result)))
        return result.value

    check(driver.cuInit(0))
    x = signal(seconds=seconds, fs=16000)
    expected = _core.harvest(x, 16000, device=device)
    previous = current()
    custom = ct.c_void_p()
    check(driver.cuCtxCreate_v2(ct.byref(custom), 0, 0))
    try:
        check(driver.cuCtxSetCurrent(previous))

        def worker():
            if warm_first:
                _core.harvest(x, 16000)
            check(driver.cuCtxSetCurrent(custom))
            for context in (custom.value, None, custom.value):
                check(driver.cuCtxSetCurrent(context))
                actual = _core.harvest(x, 16000, device=device)
                assert current() == context
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)
                _core.clear_cache()
                assert current() == context
                # Recreate after cleanup and leave a workspace for a different
                # thread to release after this worker has exited.
                actual = _core.harvest(x, 16000, device=device)
                assert current() == context
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result()
        check(driver.cuCtxSetCurrent(custom))
        _core.clear_cache()
        assert current() == custom.value
        check(driver.cuCtxSynchronize())
    finally:
        check(driver.cuCtxSetCurrent(previous))
        check(driver.cuCtxDestroy_v2(custom))


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_clear_cache_releases_live_workers_gpu_memory():
    driver = ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
    driver.cuDevicePrimaryCtxRetain.argtypes = [ct.POINTER(ct.c_void_p), ct.c_int]
    driver.cuDevicePrimaryCtxRelease_v2.argtypes = [ct.c_int]
    driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
    driver.cuCtxSetCurrent.argtypes = [ct.c_void_p]
    driver.cuMemGetInfo_v2.argtypes = [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]

    def check(result):
        assert result == 0, f"CUDA driver error {result}"

    def free_memory():
        free, total = ct.c_size_t(), ct.c_size_t()
        check(driver.cuMemGetInfo_v2(ct.byref(free), ct.byref(total)))
        return free.value

    check(driver.cuInit(0))
    previous, primary = ct.c_void_p(), ct.c_void_p()
    check(driver.cuCtxGetCurrent(ct.byref(previous)))
    check(driver.cuDevicePrimaryCtxRetain(ct.byref(primary), 0))
    try:
        check(driver.cuCtxSetCurrent(primary))
        _core.clear_cache()
        x = signal(seconds=2.1, fs=16000)
        with ThreadPoolExecutor(max_workers=1) as pool:
            expected = pool.submit(_core.harvest, x, 16000).result()
            before = free_memory()
            # The worker remains alive: central cleanup must reach its cache.
            _core.clear_cache()
            assert free_memory() - before >= 1024 * 1024
            actual = pool.submit(_core.harvest, x, 16000).result()
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(got, want)
    finally:
        _core.clear_cache()
        check(driver.cuCtxSetCurrent(previous))
        check(driver.cuDevicePrimaryCtxRelease_v2(0))
