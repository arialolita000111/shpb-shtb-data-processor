from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shpb_processor.config import material_bar_parameters
from shpb_processor.dispersion import DispersionSettings, correct_wave_segments
from shpb_processor.dispersion.correction import _cached_frequency_response
from shpb_processor.models import PulseWindow, WaveSegments


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dispersion correction cache behavior.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5_000, 50_000, 200_000, 500_000],
        help="Signal lengths to benchmark.",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Warm-cache iterations per size.")
    parser.add_argument("--sampling-frequency", type=float, default=5_000_000.0, help="Sampling frequency in Hz.")
    args = parser.parse_args()

    bar = material_bar_parameters("steel")
    settings = DispersionSettings(enabled=True, poisson_ratio=0.30, amplitude_correction=True)

    print("points,cold_s,warm_avg_s,cache_hits,cache_misses")
    for length in args.sizes:
        segments = _segments(length, args.sampling_frequency)
        _cached_frequency_response.cache_clear()

        cold_start = time.perf_counter()
        correct_wave_segments(segments, bar, settings)
        cold_s = time.perf_counter() - cold_start

        warm_start = time.perf_counter()
        for _ in range(args.iterations):
            correct_wave_segments(segments, bar, settings)
        warm_avg_s = (time.perf_counter() - warm_start) / max(args.iterations, 1)

        cache_info = _cached_frequency_response.cache_info()
        print(
            f"{length},{cold_s:.6f},{warm_avg_s:.6f},"
            f"{cache_info.hits},{cache_info.misses}"
        )
    return 0


def _segments(length: int, sampling_frequency_hz: float) -> WaveSegments:
    time_s = np.arange(length, dtype=float) / sampling_frequency_hz
    center = float(time_s[-1] * 0.45) if length else 0.0
    width = max(float(time_s[-1] / 20.0), 1e-6) if length else 1e-6
    pulse = 800e-6 * np.exp(-0.5 * ((time_s - center) / width) ** 2)
    window = PulseWindow(
        start_s=float(time_s[0]) if length else 0.0,
        end_s=float(time_s[-1]) if length else 0.0,
        label="full",
        confidence=1.0,
    )
    return WaveSegments(
        time_s=time_s,
        incident=pulse,
        reflected=-0.4 * pulse,
        transmitted=0.6 * pulse,
        incident_window=window,
        reflected_window=window,
        transmitted_window=window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
