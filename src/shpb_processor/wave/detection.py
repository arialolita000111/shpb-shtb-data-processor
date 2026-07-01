from __future__ import annotations

import numpy as np
from pydantic import Field

from shpb_processor.models import PulseWindow, WaveSegments
from shpb_processor.models.base import ProcessorModel


class FixedPulseWindows(ProcessorModel):
    incident: PulseWindow
    reflected: PulseWindow
    transmitted: PulseWindow


class PulseDetectionSettings(ProcessorModel):
    threshold_sigma: float = Field(default=6.0, gt=0)
    relative_threshold: float = Field(default=0.08, gt=0, lt=1)
    noise_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    min_duration_s: float = Field(default=0.0, ge=0)
    merge_gap_s: float | None = Field(default=None, ge=0)
    padding_s: float | None = Field(default=None, ge=0)
    auto_padding_fraction: float = Field(default=0.10, ge=0, le=1)
    min_auto_padding_s: float = Field(default=5e-6, ge=0)
    max_auto_padding_s: float = Field(default=10e-6, ge=0)
    fixed_windows: FixedPulseWindows | None = None


def detect_pulses(
    time_s: np.ndarray,
    incident_signal: np.ndarray,
    transmitted_signal: np.ndarray,
    settings: PulseDetectionSettings | None = None,
) -> WaveSegments:
    settings = settings or PulseDetectionSettings()
    time = np.asarray(time_s, dtype=float)
    incident = np.asarray(incident_signal, dtype=float)
    transmitted = np.asarray(transmitted_signal, dtype=float)
    if settings.fixed_windows is not None:
        return _segments_from_fixed_windows(time, incident, transmitted, settings.fixed_windows)

    dt = _median_dt(time)

    incident_windows = _find_windows(time, incident, settings, dt)
    transmitted_windows = _find_windows(time, transmitted, settings, dt)
    warnings: list[str] = []

    if len(incident_windows) >= 2:
        incident_window = incident_windows[0]
        reflected_window = incident_windows[1]
    elif len(incident_windows) == 1:
        incident_window, reflected_window = _split_single_incident_window(incident_windows[0])
        warnings.append("Incident and reflected pulses may overlap; a single incident-bar pulse was split for review.")
    else:
        incident_window = _fallback_window(time, "incident")
        reflected_window = _fallback_window(time, "reflected", offset_fraction=0.45)
        warnings.append("Incident/reflected pulses were not detected; fallback windows were created.")

    if transmitted_windows:
        transmitted_window = transmitted_windows[0]
    else:
        transmitted_window = _fallback_window(time, "transmitted", offset_fraction=0.35)
        warnings.append("Transmitted pulse was not detected; fallback window was created.")

    incident_segment = _mask_signal(time, incident, incident_window)
    reflected_segment = _mask_signal(time, incident, reflected_window)
    transmitted_segment = _mask_signal(time, transmitted, transmitted_window)

    return WaveSegments(
        time_s=time,
        incident=incident_segment,
        reflected=reflected_segment,
        transmitted=transmitted_segment,
        incident_window=incident_window,
        reflected_window=reflected_window,
        transmitted_window=transmitted_window,
        metadata={
            "warnings": warnings,
            "incident_candidate_count": len(incident_windows),
            "transmitted_candidate_count": len(transmitted_windows),
            "window_padding_mode": "auto" if settings.padding_s is None else "manual",
            "window_padding_s": _representative_padding_s(settings, incident_windows + transmitted_windows),
        },
    )


def _segments_from_fixed_windows(
    time: np.ndarray,
    incident: np.ndarray,
    transmitted: np.ndarray,
    windows: FixedPulseWindows,
) -> WaveSegments:
    return WaveSegments(
        time_s=time,
        incident=_mask_signal(time, incident, windows.incident),
        reflected=_mask_signal(time, incident, windows.reflected),
        transmitted=_mask_signal(time, transmitted, windows.transmitted),
        incident_window=windows.incident,
        reflected_window=windows.reflected,
        transmitted_window=windows.transmitted,
        metadata={
            "warnings": [],
            "window_source": "fixed_manual_template",
            "window_padding_mode": "fixed",
            "window_padding_s": 0.0,
        },
    )


def _find_windows(
    time: np.ndarray,
    signal: np.ndarray,
    settings: PulseDetectionSettings,
    dt: float,
) -> list[PulseWindow]:
    if len(signal) == 0:
        return []
    noise_count = max(5, int(len(signal) * settings.noise_fraction))
    baseline = signal[:noise_count]
    noise_level = float(np.nanmean(np.abs(baseline)) + settings.threshold_sigma * np.nanstd(baseline))
    peak = float(np.nanmax(np.abs(signal)))
    threshold = max(noise_level, settings.relative_threshold * peak, 1e-15)
    active = np.abs(signal) >= threshold

    if not np.any(active):
        return []

    merge_gap_points = int(round((settings.merge_gap_s or 8 * dt) / max(dt, 1e-30)))
    min_points = max(2, int(round(settings.min_duration_s / max(dt, 1e-30))))
    ranges = _active_ranges(active, merge_gap_points, min_points)
    windows: list[PulseWindow] = []
    for start_idx, end_idx in ranges:
        raw_start_s = float(time[start_idx])
        raw_end_s = float(time[end_idx])
        padding_s = _window_padding_s(settings, raw_end_s - raw_start_s)
        start_s = max(float(time[0]), raw_start_s - padding_s)
        end_s = min(float(time[-1]), raw_end_s + padding_s)
        local_peak = float(np.nanmax(np.abs(signal[start_idx : end_idx + 1])))
        confidence = min(1.0, max(0.0, local_peak / max(threshold, 1e-30) / 8.0))
        windows.append(PulseWindow(start_s=start_s, end_s=end_s, label="pulse", confidence=confidence))
    return windows


def _active_ranges(active: np.ndarray, merge_gap_points: int, min_points: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - previous <= merge_gap_points + 1:
            previous = idx
            continue
        if previous - start + 1 >= min_points:
            ranges.append((start, previous))
        start = previous = idx
    if previous - start + 1 >= min_points:
        ranges.append((start, previous))
    return ranges


def _split_single_incident_window(window: PulseWindow) -> tuple[PulseWindow, PulseWindow]:
    midpoint = 0.5 * (window.start_s + window.end_s)
    incident = PulseWindow(start_s=window.start_s, end_s=midpoint, label="incident", confidence=0.35)
    reflected = PulseWindow(start_s=midpoint, end_s=window.end_s, label="reflected", confidence=0.25)
    return incident, reflected


def _fallback_window(time: np.ndarray, label: str, offset_fraction: float = 0.2) -> PulseWindow:
    start = float(time[0] + offset_fraction * (time[-1] - time[0]))
    duration = float(0.15 * (time[-1] - time[0]))
    return PulseWindow(start_s=start, end_s=min(float(time[-1]), start + duration), label=label, confidence=0.0)


def _mask_signal(time: np.ndarray, signal: np.ndarray, window: PulseWindow) -> np.ndarray:
    masked = np.full_like(signal, np.nan, dtype=float)
    mask = (time >= window.start_s) & (time <= window.end_s)
    masked[mask] = signal[mask]
    return masked


def _median_dt(time: np.ndarray) -> float:
    if len(time) < 2:
        return 1.0
    dt = float(np.nanmedian(np.diff(time)))
    return dt if dt > 0 else 1.0


def _window_padding_s(settings: PulseDetectionSettings, raw_duration_s: float) -> float:
    if settings.padding_s is not None:
        return settings.padding_s
    if raw_duration_s <= 0:
        return settings.min_auto_padding_s
    padding = settings.auto_padding_fraction * raw_duration_s
    if settings.max_auto_padding_s > 0:
        padding = min(padding, settings.max_auto_padding_s)
    return max(padding, settings.min_auto_padding_s)


def _representative_padding_s(settings: PulseDetectionSettings, windows: list[PulseWindow]) -> float:
    if settings.padding_s is not None:
        return settings.padding_s
    durations = [window.duration_s for window in windows if window.duration_s > 0]
    duration = float(np.nanmedian(durations)) if durations else 0.0
    return _window_padding_s(settings, duration)
