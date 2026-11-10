import numpy as np
import pytest
import pyworld
from harvest_cuda import _core
from benchmark import compare
from signals import signal


@pytest.mark.parametrize("fs", [8000, 16000, 24000, 44100, 48000])
@pytest.mark.parametrize("period", [1.0, 2.5, 5.0, 10.0])
def test_pinned_reference_and_profile(fs, period):
    x = signal(seconds=0.35, fs=fs)
    expected, times = pyworld.harvest(x, fs, frame_period=period)
    reference = _core._reference(x, fs, frame_period=period)
    profile = _core._profile(x, fs, frame_period=period, gpu=False)
    assert len(reference) == 2
    assert len(profile) == 3
    assert "filter" in profile[2] and "events" in profile[2]
    assert all(np.isfinite(ms) and ms >= 0 for ms in profile[2].values())
    for result in [reference, profile]:
        np.testing.assert_array_equal(result[1], times)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("fs", [8000, 16000, 24000, 44100, 48000])
@pytest.mark.parametrize("kind", ["speechlike", "low", "high", "noise", "silence"])
def test_cuda_matches_pyworld(fs, kind):
    x = signal(kind, seconds=0.7, fs=fs)
    expected, times = pyworld.harvest(x, fs)
    actual, actual_times = _core.harvest(x, fs)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual_times, times)
    accuracy = compare(expected, actual)
    assert accuracy["vuv_disagreement"] <= 0.001
    if accuracy["joint_voiced_frames"]:
        assert accuracy["cents_p99"] <= 1.0
        assert accuracy["cents_max"] < 50.0
    elif kind in ("low", "high", "speechlike"):
        pytest.fail("Test signal unexpectedly has no voiced frames")


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("kind", ["speechlike", "silence"])
def test_profile_preserves_public_output(kind):
    x = signal(kind, seconds=0.35, fs=16000)
    actual = _core.harvest(x, 16000)
    profiled = _core._profile(x, 16000)
    assert len(actual) == 2
    assert len(profiled) == 3
    np.testing.assert_array_equal(profiled[0], actual[0])
    np.testing.assert_array_equal(profiled[1], actual[1])
    assert "filter_events" in profiled[2]
    assert "filter" not in profiled[2] and "events" not in profiled[2]
    assert all(np.isfinite(ms) and ms >= 0 for ms in profiled[2].values())


@pytest.mark.parametrize("kwargs", [
    {"fs": 0}, {"f0_floor": 0}, {"f0_floor": 800},
    {"f0_ceil": float("inf")}, {"f0_ceil": 5000},
    {"frame_period": 0}, {"frame_period": float("nan")},
])
def test_invalid_parameters(kwargs):
    params = {"fs": 24000, **kwargs}
    with pytest.raises(ValueError):
        _core.harvest(np.zeros(2400), **params)


@pytest.mark.parametrize("x", [np.zeros(0), np.array([np.nan, 0.0]), np.array([np.inf, 0.0])])
def test_invalid_samples(x):
    with pytest.raises(ValueError):
        _core.harvest(x, 24000)


@pytest.mark.parametrize("x", [np.zeros(100, dtype=np.float32), np.zeros((2, 100)), np.zeros(100)[::2], [0.0] * 100])
def test_rejects_wrong_array_type(x):
    with pytest.raises(TypeError):
        _core.harvest(x, 24000)


def test_gpu_required():
    if _core.cuda_available():
        pytest.skip("GPU available")
    with pytest.raises(RuntimeError):
        _core.harvest(np.zeros(2400), 24000)


def test_device_argument_validation():
    x = np.zeros(2400)
    assert not _core.cuda_available(device=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        _core.harvest(x, 24000, device=-1)
    with pytest.raises(TypeError):
        _core.harvest(x, 24000, 71.0, 800.0, 5.0, 0)
    with pytest.raises(TypeError):
        _core.harvest(x, 24000, device="cuda:0")


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_explicit_device_and_failed_switch():
    x = signal(seconds=0.35, fs=16000)
    expected = _core.harvest(x, 16000)
    assert _core.cuda_available(device=0)
    assert not _core.cuda_available(device=2**31 - 1)
    with pytest.raises(ValueError, match="out of range"):
        _core.harvest(x, 16000, device=2**31 - 1)
    for actual in [_core.harvest(x, 16000, device=0),
                   _core._profile(x, 16000, device=0)[:2]]:
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_switch_devices_and_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor

    if not _core.cuda_available(device=1):
        pytest.skip("Two supported visible GPUs required")
    x = signal(seconds=0.35, fs=16000)
    expected = pyworld.harvest(x, 16000)

    def worker(order):
        for device in order:
            actual = _core.harvest(x, 16000, device=device)
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)
        _core.clear_cache()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, [(0, 1, 0), (1, 0, 1)]))


def test_clear_empty_cache():
    from harvest_cuda import clear_cache

    # This must also be safe in a CPU development build or without a driver.
    clear_cache()
    clear_cache()


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_clear_cache_preserves_outputs_and_other_threads():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from harvest_cuda import clear_cache

    x = signal(seconds=0.7, fs=24000)
    expected = _core.harvest(x, 24000)
    saved = tuple(a.copy() for a in expected)
    ready = Barrier(2)

    def worker():
        before = _core.harvest(x, 24000)
        ready.wait(timeout=30)
        ready.wait(timeout=30)
        after = _core.harvest(x, 24000)
        clear_cache()
        return before, after

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        ready.wait(timeout=30)
        clear_cache()
        clear_cache()
        ready.wait(timeout=30)
        actual = [*future.result(), _core.harvest(x, 24000)]
    clear_cache()
    for result in [expected, *actual]:
        for got, want in zip(result, saved):
            np.testing.assert_array_equal(got, want)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("period,floor,ceil", [(1.0, 50, 900), (2.5, 100, 400), (7.3, 65, 1000)])
def test_cuda_parameters_and_readonly_input(period, floor, ceil):
    x = signal(seconds=1.1, fs=44100)
    x.flags.writeable = False
    expected = pyworld.harvest(x, 44100, floor, ceil, period)
    actual = _core.harvest(x, 44100, floor, ceil, period)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_concurrent_calls_and_output_ownership():
    from concurrent.futures import ThreadPoolExecutor
    import gc
    inputs = [signal(seconds=0.3, fs=16000), signal("high", seconds=0.5, fs=24000)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda args: _core.harvest(*args), zip(inputs, [16000, 24000])))
    gc.collect()
    for x, fs, actual in zip(inputs, [16000, 24000], results):
        expected = pyworld.harvest(x, fs)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)
        np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_clear_cache_preserves_outputs_and_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor
    from harvest_cuda import clear_cache

    x = signal(seconds=0.7, fs=24000)
    expected = _core.harvest(x, 24000)
    saved = tuple(a.copy() for a in expected)
    clear_cache()
    clear_cache()  # Releasing an empty cache is safe.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_core.harvest, x, 24000) for _ in range(2)]
        pool.submit(clear_cache).result()
        actual = [f.result() for f in futures]
    clear_cache()
    actual.append(_core.harvest(x, 24000))
    for result in [expected, *actual]:
        for a, b in zip(result, saved):
            np.testing.assert_array_equal(a, b)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("length", [6, 24, 240])
def test_short_silent_input(length):
    x = np.zeros(length)
    expected = pyworld.harvest(x, 24000)
    actual = _core.harvest(x, 24000)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_filter_cache_configuration_and_waveform_changes():
    # Several inputs share an FFT length, but change the internal sample rate,
    # band boundaries, or waveform. Returning to a previous key must be safe.
    for fs, seconds, floor, ceil, kind in [
        (8000, 1.2, 71, 800, "speechlike"),
        (11025, 1.0, 71, 800, "speechlike"),
        (8000, 1.2, 65, 900, "high"),
        (8000, 1.2, 71, 800, "silence"),
        (8000, 1.2, 71, 800, "high"),
        (8000, 1.2, 71, 800, "speechlike"),
        # Longer block convolutions, and whole-recording filters whose bank
        # exceeds 256 MiB (filters longer than a block use the whole-FFT path).
        (8000, 20.0, 71, 800, "speechlike"),
        # Same block filter, different whole-recording FFT length: reuse its
        # spectrum, but refresh the frequency-bin corrections in both directions.
        (8000, 6.0, 71, 800, "speechlike"),
        # Grow the phase table within one FFT size, then change FFT size and
        # grow again inside retained capacity whose tail has the old period.
        (8000, 7.0, 71, 800, "speechlike"),
        (8000, 3.0, 71, 800, "speechlike"),
        (8000, 3.5, 71, 800, "high"),
        (8000, 20.0, 71, 800, "high"),
        (8000, 20.0, 65, 900, "speechlike"),
        (8000, 20.0, 4, 90, "low"),
        (8000, 20.0, 4, 90, "silence"),
        (8000, 1.2, 71, 800, "speechlike"),
    ]:
        x = signal(kind, seconds=seconds, fs=fs)
        expected = pyworld.harvest(x, fs, floor, ceil)
        actual = _core.harvest(x, fs, floor, ceil)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
def test_separate_voiced_sections_and_endpoint_smoothing():
    # The voiced sections also cross the bounded refinement chunks.
    fs = 16000
    t = np.arange(fs * 8) / fs
    x = sum(np.sin(2 * np.pi * 137 * h * t) / h for h in range(1, 6))
    x[(t % 0.8 > 0.5) & (t < 7.5)] = 0
    expected = pyworld.harvest(x, fs)
    actual = _core.harvest(x, fs)
    assert np.count_nonzero(expected[0]) > 100
    assert np.count_nonzero(expected[0] == 0) > 100
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("fs,frequency,floor,ceil", [
    (8000, 16, 8, 90),       # Large windows use the cuFFT refinement path.
    (11025, 30, 10, 100),    # Cross the shared-memory/cuFFT window boundary.
    (192000, 75, 40, 250),   # Internal sampling rate exceeds 8 kHz.
    (8000, 2400, 1200, 3900),  # Small FFTs with fewer than six harmonics.
])
def test_refinement_window_sizes(fs, frequency, floor, ceil):
    # Low pitches exercise both FFT paths across a refinement chunk boundary.
    seconds = 4.3 if frequency < 40 else 1.6
    t = np.arange(round(seconds * fs)) / fs
    pitch = frequency * (1 + 0.2 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / fs
    harmonics = min(6, int(fs / (2 * frequency * 1.2)))
    x = sum(0.2 * np.sin(h * phase + 0.1 * h) / h
            for h in range(1, harmonics + 1))
    x += np.random.default_rng(456).normal(0, 0.0001, len(t))
    expected = pyworld.harvest(x, fs, floor, ceil)
    for _ in range(2):
        actual = _core.harvest(x, fs, floor, ceil)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("fs", [16000, 24000, 32000, 40000, 48000, 56000,
                               64000, 72000, 80000, 88000, 96000, 192000, 44100])
def test_parallel_decimation_ratios(fs):
    # Long enough to use GPU preprocessing; cover every supported IIR ratio,
    # both integral and fractional internal sample rates, and voiced gaps.
    # A partial final decimation group must still write the last output sample.
    x = signal(seconds=2.1, fs=fs)[:-1]
    expected = pyworld.harvest(x, fs)
    actual = _core.harvest(x, fs)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("fs", [72000, 80000, 192000])
@pytest.mark.parametrize("constant", [True, False])
def test_decimation_handles_dc_input(fs, constant):
    # Degenerate inputs can amplify IIR rounding artifacts into spurious F0.
    # Require valid output without reproducing WORLD's numerical artifacts.
    x = np.ones(round(2.2 * fs)) if constant else 1000 + 0.01 * signal(seconds=2.2, fs=fs)
    expected = pyworld.harvest(x, fs)
    actual = _core.harvest(x, fs)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0].shape == actual[1].shape
    assert np.all(np.isfinite(actual[0]))
    assert np.all(actual[0] >= 0)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("length", [32767, 32768, 32769])
def test_decimation_dispatch_boundary(length):
    x = signal(seconds=length / 24000, fs=24000)
    expected = pyworld.harvest(x, 24000)
    actual = _core.harvest(x, 24000)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not _core.cuda_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("length", [24576, 24577, 24578, 24800])
def test_block_filter_preserves_world_final_bins(length):
    fs = 8000
    # Cover complete tiles and one/two-sample tails used by the derivative halo.
    x = signal(seconds=length / fs, fs=fs)
    i = np.arange(len(x))
    # The global FFT length is 32768 for this shape. Excite both bins that
    # WORLD overwrites so ordinary overlap-save convolution would differ.
    x += 0.01 * ((1 - 2 * (i % 2)) + np.cos(2 * np.pi * (0.5 - 1 / 32768) * i))
    expected = pyworld.harvest(x, fs)
    actual = _core.harvest(x, fs)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7, atol=1e-5)
