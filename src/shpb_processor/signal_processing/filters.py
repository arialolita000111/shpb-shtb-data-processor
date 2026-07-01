from __future__ import annotations

import numpy as np


def apply_filter(
    signal: np.ndarray,
    method: str = "none",
    sampling_frequency_hz: float | None = None,
    window_points: int = 11,
    cutoff_hz: float | None = None,
    order: int = 4,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if method == "none":
        return values.copy()
    if method == "moving_average":
        return moving_average(values, window_points)
    if method == "savgol":
        return savgol_smooth(values, window_points, order=min(3, max(1, order)))
    if method == "butterworth":
        if sampling_frequency_hz is None or cutoff_hz is None:
            raise ValueError("Butterworth filtering requires sampling_frequency_hz and cutoff_hz.")
        return butterworth_lowpass(values, sampling_frequency_hz, cutoff_hz, order)
    if method == "median":
        medfilt = _scipy_signal_function("medfilt")
        kernel = _odd_window(window_points, minimum=3)
        return medfilt(values, kernel_size=kernel)
    raise ValueError(f"Unsupported filter method: {method}")


def moving_average(signal: np.ndarray, window_points: int = 11) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    window = max(1, int(window_points))
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def savgol_smooth(signal: np.ndarray, window_points: int = 11, order: int = 3) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    window = _odd_window(window_points, minimum=order + 2)
    if len(values) < window:
        return values.copy()
    savgol_filter = _scipy_signal_function("savgol_filter")
    return savgol_filter(values, window_length=window, polyorder=min(order, window - 1), mode="interp")


def butterworth_lowpass(signal: np.ndarray, sampling_frequency_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    nyquist = 0.5 * sampling_frequency_hz
    if cutoff_hz >= nyquist:
        raise ValueError("cutoff_hz must be below the Nyquist frequency.")
    butter = _scipy_signal_function("butter")
    filtfilt = _scipy_signal_function("filtfilt")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    padlen = min(3 * max(len(a), len(b)), max(0, len(values) - 1))
    if padlen < 1:
        return values.copy()
    return filtfilt(b, a, values, padlen=padlen)


def _odd_window(window_points: int, minimum: int = 3) -> int:
    window = max(minimum, int(window_points))
    if window % 2 == 0:
        window += 1
    return window


def _scipy_signal_function(name: str):
    try:
        from scipy import signal
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Filter method requires scipy.signal.{name}; install project dependencies before using this filter."
        ) from exc
    return getattr(signal, name)
