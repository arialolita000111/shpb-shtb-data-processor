from __future__ import annotations

import numpy as np


def baseline_correct(
    time_s: np.ndarray,
    signal: np.ndarray,
    method: str = "start_mean",
    window_s: tuple[float, float] | None = None,
    polynomial_order: int = 1,
    manual_offset: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    if len(values) == 0:
        return values.copy(), {}

    if method == "none":
        return values.copy(), {"offset": 0.0}
    if method == "manual":
        return values - manual_offset, {"offset": float(manual_offset)}

    mask = _baseline_mask(time, len(values), window_s)
    if not np.any(mask):
        mask = np.arange(len(values)) < max(3, len(values) // 20)

    if method in {"start_mean", "window_mean"}:
        offset = float(np.nanmean(values[mask]))
        return values - offset, {"offset": offset}

    if method == "linear_drift":
        baseline = np.interp(time, [time[0], time[-1]], [np.nanmean(values[mask]), np.nanmean(values[-np.count_nonzero(mask) :])])
        return values - baseline, {"offset": float(np.nanmean(baseline)), "slope": float((baseline[-1] - baseline[0]) / max(time[-1] - time[0], 1e-30))}

    if method == "polynomial":
        degree = max(0, min(polynomial_order, np.count_nonzero(mask) - 1))
        coeffs = np.polyfit(time[mask], values[mask], degree)
        baseline = np.polyval(coeffs, time)
        return values - baseline, {"offset": float(np.nanmean(baseline)), "polynomial_order": float(degree)}

    raise ValueError(f"Unsupported baseline correction method: {method}")


def check_signal_anomalies(time_s: np.ndarray, signals: dict[str, np.ndarray]) -> list[str]:
    warnings: list[str] = []
    time = np.asarray(time_s, dtype=float)
    if len(time) < 2:
        warnings.append("Time vector has fewer than two points.")
    else:
        if not np.all(np.diff(time) > 0):
            warnings.append("Time values are not strictly increasing.")
        diffs = np.diff(time)
        median_dt = np.nanmedian(diffs)
        if median_dt > 0 and np.nanmax(np.abs(diffs - median_dt)) > 0.05 * median_dt:
            warnings.append("Sampling interval varies by more than 5%.")

    for name, values in signals.items():
        values = np.asarray(values, dtype=float)
        if np.isnan(values).any():
            warnings.append(f"{name} contains NaN values.")
        if np.isinf(values).any():
            warnings.append(f"{name} contains Inf values.")
        if len(values) > 5:
            median = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values - median))
            if mad > 0:
                robust_z = np.abs(values - median) / (1.4826 * mad)
                spike_fraction = float(np.mean(robust_z > 12))
                if spike_fraction > 0.001:
                    warnings.append(f"{name} has isolated spike-like outliers.")
    return warnings


def _baseline_mask(time: np.ndarray, length: int, window_s: tuple[float, float] | None) -> np.ndarray:
    if window_s is None:
        mask = np.zeros(length, dtype=bool)
        mask[: max(3, length // 20)] = True
        return mask
    start, end = window_s
    return (time >= start) & (time <= end)
