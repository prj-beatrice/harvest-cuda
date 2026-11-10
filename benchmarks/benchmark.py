"""Run from the repository root: python benchmarks/benchmark.py --help."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import platform
import statistics
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pyworld
from harvest_cuda import _core
from signals import signal


def compare(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} versus {actual.shape}")
    for name, f0 in (("reference", reference), ("actual", actual)):
        if not np.all(np.isfinite(f0)) or np.any(f0 < 0):
            raise ValueError(f"{name} F0 must contain only finite, nonnegative values")
    rv, av = reference > 0, actual > 0
    both = rv & av
    cents = np.abs(1200 * np.log2(actual[both] / reference[both]))
    return {
        "frames": len(reference),
        "reference_voiced_frames": int(rv.sum()),
        "joint_voiced_frames": int(both.sum()),
        "vuv_disagreement": float(np.mean(rv != av)),
        "cents_p99": float(np.quantile(cents, 0.99)) if len(cents) else None,
        "cents_max": float(cents.max()) if len(cents) else None,
        "over_600_cents": int((cents > 600).sum()),
    }


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return result, {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, nargs="+", default=[1, 10, 60])
    p.add_argument("--fs", type=int, nargs="+", default=[16000, 24000, 48000])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-seconds", type=float, default=5.0,
                   help="Continuous GPU warmup per case before timing (default: 5 seconds)")
    p.add_argument("--kind", default="speechlike", choices=["speechlike", "low", "high", "noise", "silence"])
    p.add_argument("--cpu-only", action="store_true", help="Profile the reference; never selects CPU for the public API")
    p.add_argument("--wav", type=Path, help="Use a mono PCM16 WAV at its original rate and duration instead of synthetic cases")
    p.add_argument("--output", type=Path, default=Path("benchmark-results/latest.json"))
    args = p.parse_args()
    if args.repeats < 1 or any(s <= 0 for s in args.seconds):
        p.error("repeats and seconds must be positive")
    if not np.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        p.error("warmup-seconds must be finite and nonnegative")
    if not args.cpu_only and not _core.cuda_available():
        p.error("CUDA unavailable; use --cpu-only to profile the reference")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor(),
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pyworld", "harvest-cuda"]},
        "signal": args.kind, "synthetic": True, "cases": [],
        "timing_scope": "NumPy input to completed NumPy output; includes transfers and CPU postprocessing",
    }
    root = Path(__file__).resolve().parents[1]
    sources = [root / "CMakeLists.txt", root / "pyproject.toml", *sorted((root / "src").glob("*")),
               root / "benchmarks/signals.py", root / "benchmarks/benchmark.py"]
    report["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources if path.is_file()}
    if args.wav:
        with wave.open(str(args.wav), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                p.error("--wav requires mono PCM16")
            wav_fs = wav.getframerate()
            waveform = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768
        cases = [(wav_fs, len(waveform) / wav_fs, waveform)]
        report.update(synthetic=False, signal=str(args.wav), sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest())
    else:
        cases = [(fs, seconds, signal(args.kind, seconds, fs)) for fs in args.fs for seconds in args.seconds]
    try:
        report["nvidia_smi_before"] = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as e:
        report["nvidia_smi_before"] = str(e)
    if Path("/proc/cpuinfo").exists():
        report["cpu"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), report["cpu"])
    for fs, seconds, x in cases:
        (reference, times), cpu = timed(lambda: pyworld.harvest(x, fs), args.repeats)
        own, own_times, stages = _core._profile(x, fs, gpu=False)
        np.testing.assert_array_equal(own_times, times)
        np.testing.assert_allclose(own, reference, rtol=1e-10, atol=1e-8)
        case = {"fs": fs, "seconds": seconds, "cpu": cpu, "cpu_stages_ms": stages}
        if not args.cpu_only:
            (_, _), cold = timed(lambda: _core.harvest(x, fs), 1)
            warm_start = time.perf_counter()
            warm_calls = 0
            while time.perf_counter() - warm_start < args.warmup_seconds:
                _core.harvest(x, fs)
                warm_calls += 1
            warmup = {"seconds": time.perf_counter() - warm_start, "calls": warm_calls}
            (actual, actual_times), gpu = timed(lambda: _core.harvest(x, fs), args.repeats)
            _, _, gpu_stages = _core._profile(x, fs)
            np.testing.assert_array_equal(actual_times, times)
            case.update(gpu=gpu, first_call_for_shape=cold, warmup=warmup, gpu_stages_ms=gpu_stages,
                        speedup=cpu["median_ms"] / gpu["median_ms"], accuracy=compare(reference, actual))
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
