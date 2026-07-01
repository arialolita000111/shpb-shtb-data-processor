from __future__ import annotations

from functools import lru_cache
import math
from typing import Literal

import numpy as np
from pydantic import Field

from shpb_processor.models import BarParameters, WaveSegments
from shpb_processor.models.base import ProcessorModel


class DispersionSettings(ProcessorModel):
    """Frequency-domain stress-wave dispersion correction settings.

    The implementation follows the Tyas-Pope processing flow: isolate each
    pulse, transform it to the frequency domain, apply a distance-dependent
    phase correction, optionally apply a mild radial-amplitude correction, and
    transform back to the time domain. The built-in phase-velocity law is a
    conservative low-frequency approximation. Signed phase/amplitude strengths
    are allowed so validation data can calibrate whether the correction should
    sharpen or attenuate high-frequency content for a given bar/specimen setup.
    """

    enabled: bool = False
    model: Literal["frequency_domain"] = "frequency_domain"
    calibration_source: str | None = None
    poisson_ratio: float | None = Field(default=None, ge=0.0, lt=0.5)
    phase_strength: float | None = Field(default=None, ge=-0.5, le=0.5)
    incident_phase_strength: float | None = Field(default=None, ge=-0.5, le=0.5)
    reflected_phase_strength: float | None = Field(default=None, ge=-0.5, le=0.5)
    transmitted_phase_strength: float | None = Field(default=None, ge=-0.5, le=0.5)
    amplitude_correction: bool = True
    amplitude_strength: float = Field(default=0.04, ge=-0.95, le=0.5)
    incident_amplitude_strength: float | None = Field(default=None, ge=-0.95, le=0.5)
    reflected_amplitude_strength: float | None = Field(default=None, ge=-0.95, le=0.5)
    transmitted_amplitude_strength: float | None = Field(default=None, ge=-0.95, le=0.5)
    taper_fraction: float = Field(default=0.05, ge=0.0, le=0.25)
    incident_taper_fraction: float | None = Field(default=None, ge=0.0, le=0.25)
    reflected_taper_fraction: float | None = Field(default=None, ge=0.0, le=0.25)
    transmitted_taper_fraction: float | None = Field(default=None, ge=0.0, le=0.25)
    max_correction_frequency_hz: float | None = Field(default=None, gt=0.0)
    incident_max_correction_frequency_hz: float | None = Field(default=None, gt=0.0)
    reflected_max_correction_frequency_hz: float | None = Field(default=None, gt=0.0)
    transmitted_max_correction_frequency_hz: float | None = Field(default=None, gt=0.0)


def correct_wave_segments(
    segments: WaveSegments,
    bar: BarParameters,
    settings: DispersionSettings | None = None,
) -> WaveSegments:
    settings = settings or DispersionSettings()
    if not settings.enabled:
        metadata = dict(segments.metadata)
        metadata.setdefault("dispersion_correction_enabled", False)
        metadata.setdefault("dispersion_correction_model", settings.model)
        metadata.setdefault("dispersion_amplitude_correction", settings.amplitude_correction)
        if settings.calibration_source:
            metadata.setdefault("dispersion_calibration_source", settings.calibration_source)
        return WaveSegments(
            time_s=segments.time_s,
            incident=segments.incident,
            reflected=segments.reflected,
            transmitted=segments.transmitted,
            incident_window=segments.incident_window,
            reflected_window=segments.reflected_window,
            transmitted_window=segments.transmitted_window,
            metadata=metadata,
        )

    incident_radius = 0.5 * bar.incident_diameter_m
    transmitted_radius = 0.5 * bar.transmitted_diameter_m
    corrected_incident, incident_delta = _correct_masked_signal(
        segments.time_s,
        segments.incident,
        distance_m=bar.incident_gauge_distance_m,
        bar=bar,
        radius_m=incident_radius,
        settings=settings,
        wave_label="incident",
    )
    corrected_reflected, reflected_delta = _correct_masked_signal(
        segments.time_s,
        segments.reflected,
        distance_m=-bar.incident_gauge_distance_m,
        bar=bar,
        radius_m=incident_radius,
        settings=settings,
        wave_label="reflected",
    )
    corrected_transmitted, transmitted_delta = _correct_masked_signal(
        segments.time_s,
        segments.transmitted,
        distance_m=-bar.transmitted_gauge_distance_m,
        bar=bar,
        radius_m=transmitted_radius,
        settings=settings,
        wave_label="transmitted",
    )

    metadata = dict(segments.metadata)
    metadata.update(
        {
            "dispersion_correction_enabled": True,
            "dispersion_correction_model": settings.model,
            "dispersion_poisson_ratio": _resolved_poisson_ratio(bar, settings),
            "dispersion_phase_strength": _resolved_phase_strength(bar, settings),
            "dispersion_incident_phase_strength": _resolved_wave_phase_strength(bar, settings, "incident"),
            "dispersion_reflected_phase_strength": _resolved_wave_phase_strength(bar, settings, "reflected"),
            "dispersion_transmitted_phase_strength": _resolved_wave_phase_strength(bar, settings, "transmitted"),
            "dispersion_amplitude_correction": settings.amplitude_correction,
            "dispersion_amplitude_strength": float(settings.amplitude_strength),
            "dispersion_incident_amplitude_strength": _resolved_wave_amplitude_strength(settings, "incident"),
            "dispersion_reflected_amplitude_strength": _resolved_wave_amplitude_strength(settings, "reflected"),
            "dispersion_transmitted_amplitude_strength": _resolved_wave_amplitude_strength(settings, "transmitted"),
            "dispersion_taper_fraction": float(settings.taper_fraction),
            "dispersion_incident_taper_fraction": _resolved_wave_taper_fraction(settings, "incident"),
            "dispersion_reflected_taper_fraction": _resolved_wave_taper_fraction(settings, "reflected"),
            "dispersion_transmitted_taper_fraction": _resolved_wave_taper_fraction(settings, "transmitted"),
            "dispersion_max_correction_frequency_hz": settings.max_correction_frequency_hz,
            "dispersion_incident_max_correction_frequency_hz": _resolved_wave_max_frequency(settings, "incident"),
            "dispersion_reflected_max_correction_frequency_hz": _resolved_wave_max_frequency(settings, "reflected"),
            "dispersion_transmitted_max_correction_frequency_hz": _resolved_wave_max_frequency(settings, "transmitted"),
            "dispersion_incident_rms_delta": incident_delta,
            "dispersion_reflected_rms_delta": reflected_delta,
            "dispersion_transmitted_rms_delta": transmitted_delta,
            "dispersion_note": "frequency_domain_low_frequency_approximation",
        }
    )
    if settings.calibration_source:
        metadata["dispersion_calibration_source"] = settings.calibration_source
    warnings = list(metadata.get("warnings", []))
    if bar.poisson_ratio is None and settings.poisson_ratio is None:
        warnings.append("Dispersion correction used default Poisson ratio 0.30; set bar.poisson_ratio for calibrated results.")
    metadata["warnings"] = warnings

    return WaveSegments(
        time_s=segments.time_s,
        incident=corrected_incident,
        reflected=corrected_reflected,
        transmitted=corrected_transmitted,
        incident_window=segments.incident_window,
        reflected_window=segments.reflected_window,
        transmitted_window=segments.transmitted_window,
        metadata=metadata,
    )


def _correct_masked_signal(
    time_s: np.ndarray,
    signal: np.ndarray,
    distance_m: float,
    bar: BarParameters,
    radius_m: float,
    settings: DispersionSettings,
    wave_label: Literal["incident", "reflected", "transmitted"],
) -> tuple[np.ndarray, float]:
    corrected = np.asarray(signal, dtype=float).copy()
    valid = np.isfinite(corrected)
    if np.count_nonzero(valid) < 8 or abs(distance_m) <= 0.0:
        return corrected, 0.0

    time = np.asarray(time_s, dtype=float)[valid]
    values = corrected[valid]
    dt = _median_dt(time)
    if dt <= 0:
        return corrected, 0.0

    mean = float(np.nanmean(values))
    centered = values - mean
    c0 = bar.resolved_wave_speed_m_s
    window, response = _cached_frequency_response(
        len(centered),
        dt,
        float(distance_m),
        float(radius_m),
        float(c0),
        _resolved_poisson_ratio(bar, settings),
        _resolved_wave_phase_strength(bar, settings, wave_label),
        bool(settings.amplitude_correction),
        _resolved_wave_amplitude_strength(settings, wave_label),
        _resolved_wave_taper_fraction(settings, wave_label),
        _resolved_wave_max_frequency(settings, wave_label),
    )
    spectrum = np.fft.rfft(centered * window)
    spectrum *= response

    reconstructed = np.fft.irfft(spectrum, n=len(centered)) + mean
    corrected[valid] = reconstructed
    return corrected, _normalized_rms_delta(values, reconstructed)


@lru_cache(maxsize=128)
def _cached_frequency_response(
    length: int,
    dt: float,
    distance_m: float,
    radius_m: float,
    wave_speed_m_s: float,
    poisson_ratio: float,
    phase_strength: float,
    amplitude_correction: bool,
    amplitude_strength: float,
    taper_fraction: float,
    max_correction_frequency_hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    window = _cosine_taper(length, taper_fraction)
    frequencies = np.fft.rfftfreq(length, dt)
    omega = 2.0 * math.pi * frequencies

    correction_mask = np.ones_like(frequencies, dtype=bool)
    if max_correction_frequency_hz is not None:
        correction_mask = frequencies <= max_correction_frequency_hz

    phase_velocity = _phase_velocity_approximation_from_strength(
        frequencies,
        wave_speed_m_s,
        radius_m,
        phase_strength,
    )
    phase = np.zeros_like(frequencies, dtype=float)
    finite = correction_mask & np.isfinite(phase_velocity) & (phase_velocity > 0)
    phase[finite] = (
        (wave_speed_m_s / phase_velocity[finite] - 1.0)
        * omega[finite]
        * distance_m
        / wave_speed_m_s
    )
    response = np.exp(1j * phase)

    if amplitude_correction:
        ka = omega * radius_m / max(wave_speed_m_s, 1e-30)
        amplitude = 1.0 + amplitude_strength * (ka**2 / (1.0 + ka**2))
        amplitude = np.clip(amplitude, 0.05, 2.0)
        response[correction_mask] *= amplitude[correction_mask]

    # Keep cached arrays immutable so callers cannot accidentally alter future corrections.
    window.setflags(write=False)
    response.setflags(write=False)
    return window, response


def _phase_velocity_approximation(
    frequencies_hz: np.ndarray,
    c0: float,
    radius_m: float,
    bar: BarParameters,
    settings: DispersionSettings,
) -> np.ndarray:
    return _phase_velocity_approximation_from_strength(
        frequencies_hz,
        c0,
        radius_m,
        _resolved_phase_strength(bar, settings),
    )


def _phase_velocity_approximation_from_strength(
    frequencies_hz: np.ndarray,
    c0: float,
    radius_m: float,
    strength: float,
) -> np.ndarray:
    omega = 2.0 * math.pi * frequencies_hz
    ka = omega * radius_m / max(c0, 1e-30)
    ratio = 1.0 - strength * (ka**2 / (1.0 + ka**2))
    ratio = np.clip(ratio, 0.35, 1.05)
    return c0 * ratio


def _resolved_phase_strength(bar: BarParameters, settings: DispersionSettings) -> float:
    if settings.phase_strength is not None:
        return float(settings.phase_strength)
    nu = _resolved_poisson_ratio(bar, settings)
    return float(min(0.25, max(0.03, 0.12 * (1.0 + nu))))


def _resolved_wave_phase_strength(
    bar: BarParameters,
    settings: DispersionSettings,
    wave_label: Literal["incident", "reflected", "transmitted"],
) -> float:
    value = getattr(settings, f"{wave_label}_phase_strength")
    if value is not None:
        return float(value)
    return _resolved_phase_strength(bar, settings)


def _resolved_wave_amplitude_strength(
    settings: DispersionSettings,
    wave_label: Literal["incident", "reflected", "transmitted"],
) -> float:
    value = getattr(settings, f"{wave_label}_amplitude_strength")
    if value is not None:
        return float(value)
    return float(settings.amplitude_strength)


def _resolved_wave_taper_fraction(
    settings: DispersionSettings,
    wave_label: Literal["incident", "reflected", "transmitted"],
) -> float:
    value = getattr(settings, f"{wave_label}_taper_fraction")
    if value is not None:
        return float(value)
    return float(settings.taper_fraction)


def _resolved_wave_max_frequency(
    settings: DispersionSettings,
    wave_label: Literal["incident", "reflected", "transmitted"],
) -> float | None:
    value = getattr(settings, f"{wave_label}_max_correction_frequency_hz")
    if value is not None:
        return float(value)
    if settings.max_correction_frequency_hz is None:
        return None
    return float(settings.max_correction_frequency_hz)


def _resolved_poisson_ratio(bar: BarParameters, settings: DispersionSettings) -> float:
    if settings.poisson_ratio is not None:
        return float(settings.poisson_ratio)
    if bar.poisson_ratio is not None:
        return float(bar.poisson_ratio)
    return 0.30


def _cosine_taper(length: int, fraction: float) -> np.ndarray:
    if length <= 1 or fraction <= 0:
        return np.ones(length, dtype=float)
    edge = int(max(1, min(length // 2, round(length * fraction))))
    window = np.ones(length, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, edge)))
    window[:edge] = ramp
    window[-edge:] = ramp[::-1]
    return window


def _median_dt(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.0
    dt = float(np.nanmedian(np.diff(time_s)))
    return dt if dt > 0 else 0.0


def _normalized_rms_delta(original: np.ndarray, corrected: np.ndarray) -> float:
    delta = np.asarray(corrected, dtype=float) - np.asarray(original, dtype=float)
    scale = float(np.nanmax(np.abs(original))) if len(original) else 0.0
    if scale <= 0 or not np.isfinite(scale):
        return 0.0
    return float(np.sqrt(np.nanmean(delta**2)) / scale)
