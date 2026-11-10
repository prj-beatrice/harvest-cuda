"""Deterministic test signals; these are not a substitute for real speech."""
import numpy as np


def signal(kind="speechlike", seconds=1.0, fs=24000):
    rng = np.random.default_rng(20261110)
    t = np.arange(round(seconds * fs)) / fs
    if kind == "silence":
        return np.zeros(len(t), dtype=np.float64)
    if kind == "noise":
        return rng.normal(0, 0.05, len(t))
    if kind == "low":
        frequency = 80 + 5 * np.sin(2 * np.pi * 2 * t)
    elif kind == "high":
        frequency = 650 + 30 * np.sin(2 * np.pi * 3 * t)
    else:
        frequency = 160 + 45 * np.sin(2 * np.pi * 0.7 * t) + 3 * np.sin(2 * np.pi * 5 * t)
    phase = 2 * np.pi * np.cumsum(frequency) / fs
    x = sum(np.sin(h * phase + h * 0.1) / h**1.3 for h in range(1, 11))
    envelope = np.minimum(1, t / 0.02) * np.minimum(1, (seconds - t) / 0.02)
    if kind == "speechlike":
        # Intermittent voicing, amplitude variation, vibrato, and additive noise.
        envelope *= np.clip(8 * np.sin(2 * np.pi * 1.8 * t), 0, 1)
    return np.ascontiguousarray(0.2 * x * envelope + rng.normal(0, 0.0002, len(t)))
