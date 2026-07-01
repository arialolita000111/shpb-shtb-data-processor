from __future__ import annotations

import numpy as np
from pydantic import Field

from shpb_processor.models.base import ProcessorModel
from shpb_processor.models import (
    AlignedWaves,
    BarParameters,
    CalculationResult,
    ExperimentType,
    SignConvention,
    SpecimenParameters,
)


class CalculationSettings(ProcessorModel):
    zero_reference_enabled: bool = True
    zero_reference_fraction: float = Field(default=0.05, gt=0, le=0.5)
    zero_reference_min_points: int = Field(default=3, ge=1)
    force_zero_initial_strain: bool = True
    force_zero_initial_stress: bool = True
    wave_head_ratio_threshold: float = Field(default=0.05, gt=0, le=1)


def compute_three_wave(
    waves: AlignedWaves,
    bar: BarParameters,
    specimen: SpecimenParameters,
    sign_convention: SignConvention = SignConvention.COMPRESSION_POSITIVE,
    calculation_settings: CalculationSettings | None = None,
) -> CalculationResult:
    calculation_settings = calculation_settings or CalculationSettings()
    incident, reflected, transmitted, sign_warnings = _resolve_signs(waves, sign_convention)
    c_bar = bar.resolved_wave_speed_m_s
    area_s = specimen.cross_section_area_m2

    force_incident = bar.elastic_modulus_pa * bar.incident_area_m2 * (incident + reflected)
    force_transmitted = bar.elastic_modulus_pa * bar.transmitted_area_m2 * transmitted
    strain_rate = c_bar / specimen.length_m * (incident - reflected - transmitted)
    strain = _integrate_strain(waves.time_s, strain_rate)
    stress = (force_incident + force_transmitted) / (2.0 * area_s)
    strain, stress, correction_summary, correction_warnings = _apply_zero_reference_correction(
        waves,
        strain,
        stress,
        calculation_settings,
    )

    true_strain, true_stress, true_warnings = engineering_to_true(
        strain,
        stress,
        specimen.experiment_type,
    )
    balance_error = calculate_balance_error(force_incident, force_transmitted)

    summary = summarize_result(waves.time_s, strain, strain_rate, stress, balance_error)
    summary.update(correction_summary)
    warnings = [*sign_warnings, *correction_warnings, *true_warnings]
    return CalculationResult(
        method="three_wave",
        time_s=waves.time_s,
        strain=strain,
        strain_rate_s1=strain_rate,
        engineering_stress_pa=stress,
        true_strain=true_strain,
        true_stress_pa=true_stress,
        force_incident_n=force_incident,
        force_transmitted_n=force_transmitted,
        balance_error=balance_error,
        summary=summary,
        warnings=warnings,
    )


def compute_two_wave(
    waves: AlignedWaves,
    bar: BarParameters,
    specimen: SpecimenParameters,
    sign_convention: SignConvention = SignConvention.COMPRESSION_POSITIVE,
    calculation_settings: CalculationSettings | None = None,
) -> CalculationResult:
    calculation_settings = calculation_settings or CalculationSettings()
    incident, reflected, transmitted, sign_warnings = _resolve_signs(waves, sign_convention)
    c_bar = bar.resolved_wave_speed_m_s
    area_s = specimen.cross_section_area_m2

    force_incident = bar.elastic_modulus_pa * bar.incident_area_m2 * (incident + reflected)
    force_transmitted = bar.elastic_modulus_pa * bar.transmitted_area_m2 * transmitted
    strain_rate = -2.0 * c_bar / specimen.length_m * reflected
    strain = _integrate_strain(waves.time_s, strain_rate)
    stress = force_transmitted / area_s
    strain, stress, correction_summary, correction_warnings = _apply_zero_reference_correction(
        waves,
        strain,
        stress,
        calculation_settings,
    )

    true_strain, true_stress, true_warnings = engineering_to_true(
        strain,
        stress,
        specimen.experiment_type,
    )
    balance_error = calculate_balance_error(force_incident, force_transmitted)
    summary = summarize_result(waves.time_s, strain, strain_rate, stress, balance_error)
    summary.update(correction_summary)

    return CalculationResult(
        method="two_wave",
        time_s=waves.time_s,
        strain=strain,
        strain_rate_s1=strain_rate,
        engineering_stress_pa=stress,
        true_strain=true_strain,
        true_stress_pa=true_stress,
        force_incident_n=force_incident,
        force_transmitted_n=force_transmitted,
        balance_error=balance_error,
        summary=summary,
        warnings=[*sign_warnings, *correction_warnings, *true_warnings],
    )


def calculate_balance_error(
    force_incident: np.ndarray,
    force_transmitted: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    denominator = np.maximum.reduce(
        [np.abs(force_incident), np.abs(force_transmitted), np.full_like(force_incident, epsilon)]
    )
    return np.abs(force_incident - force_transmitted) / denominator


def engineering_to_true(
    engineering_strain: np.ndarray,
    engineering_stress_pa: np.ndarray,
    experiment_type: ExperimentType,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    strain = np.asarray(engineering_strain, dtype=float)
    stress = np.asarray(engineering_stress_pa, dtype=float)
    warnings: list[str] = []

    if experiment_type == ExperimentType.COMPRESSION:
        valid = strain < 0.999999
        true_strain = np.full_like(strain, np.nan, dtype=float)
        true_stress = np.full_like(stress, np.nan, dtype=float)
        true_strain[valid] = -np.log1p(-strain[valid])
        true_stress[valid] = stress[valid] * (1.0 - strain[valid])
        if not np.all(valid):
            warnings.append("Compression true strain is undefined for engineering strain >= 1.")
    else:
        valid = strain > -0.999999
        true_strain = np.full_like(strain, np.nan, dtype=float)
        true_stress = np.full_like(stress, np.nan, dtype=float)
        true_strain[valid] = np.log1p(strain[valid])
        true_stress[valid] = stress[valid] * (1.0 + strain[valid])
        if not np.all(valid):
            warnings.append("Tension true strain is undefined for engineering strain <= -1.")

    return true_strain, true_stress, warnings


def summarize_result(
    time_s: np.ndarray,
    strain: np.ndarray,
    strain_rate: np.ndarray,
    stress_pa: np.ndarray,
    balance_error: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(strain) & np.isfinite(strain_rate) & np.isfinite(stress_pa)
    if not np.any(finite):
        return {}

    strain_f = strain[finite]
    rate_f = strain_rate[finite]
    stress_f = stress_pa[finite]
    balance_f = balance_error[finite]
    time_f = time_s[finite]

    peak_idx = int(np.nanargmax(np.abs(stress_f)))
    peak_strain = float(strain_f[peak_idx])
    peak_stress = float(stress_f[peak_idx])
    max_strain = float(np.nanmax(strain_f))
    active = np.abs(rate_f) > 0.05 * max(np.nanmax(np.abs(rate_f)), 1e-12)

    energy_density = float(abs(np.trapezoid(stress_f, strain_f))) if len(strain_f) > 1 else 0.0
    initial_modulus = _estimate_initial_modulus(strain_f, stress_f)
    duration = float(time_f[-1] - time_f[0]) if len(time_f) > 1 else 0.0

    return {
        "peak_stress_pa": peak_stress,
        "peak_stress_mpa": peak_stress / 1e6,
        "peak_strain": peak_strain,
        "max_strain": max_strain,
        "max_abs_strain_rate_s^-1": float(np.nanmax(np.abs(rate_f))),
        "average_abs_strain_rate_s^-1": float(np.nanmean(np.abs(rate_f[active]))) if np.any(active) else 0.0,
        "absorbed_energy_density_j_m3": energy_density,
        "initial_modulus_pa": initial_modulus,
        "mean_balance_error": float(np.nanmean(balance_f)),
        "max_balance_error": float(np.nanmax(balance_f)),
        "duration_s": duration,
    }


def _integrate_strain(time_s: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
    if len(time_s) < 2:
        return np.zeros_like(strain_rate)
    dt = np.diff(np.asarray(time_s, dtype=float))
    rate = np.asarray(strain_rate, dtype=float)
    increments = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate(([0.0], np.cumsum(increments)))


def _apply_zero_reference_correction(
    waves: AlignedWaves,
    strain: np.ndarray,
    stress_pa: np.ndarray,
    settings: CalculationSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[str]]:
    corrected_strain = np.asarray(strain, dtype=float).copy()
    corrected_stress = np.asarray(stress_pa, dtype=float).copy()
    warnings: list[str] = []
    summary: dict[str, float] = {
        "zero_reference_enabled": 1.0 if settings.zero_reference_enabled else 0.0,
        "zero_reference_applied": 0.0,
        "zero_reference_points": 0.0,
        "initial_strain_offset": 0.0,
        "initial_stress_offset_pa": 0.0,
        "initial_stress_offset_mpa": 0.0,
        "wave_head_initial_ratio": _initial_wave_ratio(waves),
        "wave_head_retention_ok": 1.0,
    }

    if summary["wave_head_initial_ratio"] > settings.wave_head_ratio_threshold:
        summary["wave_head_retention_ok"] = 0.0
        warnings.append(
            "Wave head may be insufficiently retained; initial wave amplitude is already above the selected threshold."
        )

    if not settings.zero_reference_enabled or len(corrected_strain) == 0:
        return corrected_strain, corrected_stress, summary, warnings

    reference_points = max(settings.zero_reference_min_points, int(np.ceil(len(corrected_strain) * settings.zero_reference_fraction)))
    reference_points = min(reference_points, len(corrected_strain))
    reference_slice = slice(0, reference_points)

    strain_offset = float(corrected_strain[0]) if settings.force_zero_initial_strain else float(np.nanmean(corrected_strain[reference_slice]))
    stress_offset = float(np.nanmean(corrected_stress[reference_slice])) if settings.force_zero_initial_stress else 0.0

    corrected_strain = corrected_strain - strain_offset
    corrected_stress = corrected_stress - stress_offset
    summary.update(
        {
            "zero_reference_applied": 1.0,
            "zero_reference_points": float(reference_points),
            "initial_strain_offset": strain_offset,
            "initial_stress_offset_pa": stress_offset,
            "initial_stress_offset_mpa": stress_offset / 1e6,
        }
    )
    if summary["wave_head_retention_ok"] == 0.0:
        warnings.append("Zero-reference correction was applied, but missing pre-wave data cannot be recovered.")
    return corrected_strain, corrected_stress, summary, warnings


def _initial_wave_ratio(waves: AlignedWaves) -> float:
    ratios: list[float] = []
    for values in (waves.incident, waves.reflected, waves.transmitted):
        array = np.asarray(values, dtype=float)
        if len(array) == 0:
            continue
        peak = float(np.nanmax(np.abs(array)))
        if peak <= 0 or not np.isfinite(peak):
            continue
        ratios.append(float(abs(array[0]) / peak))
    return max(ratios) if ratios else 0.0


def _resolve_signs(
    waves: AlignedWaves,
    sign_convention: SignConvention,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    incident = np.asarray(waves.incident, dtype=float).copy()
    reflected = np.asarray(waves.reflected, dtype=float).copy()
    transmitted = np.asarray(waves.transmitted, dtype=float).copy()

    if sign_convention == SignConvention.RAW or len(incident) == 0:
        return incident, reflected, transmitted, []

    dominant = incident[int(np.nanargmax(np.abs(incident)))]
    polarity = -1.0 if dominant < 0 else 1.0
    warnings: list[str] = []
    if polarity < 0:
        warnings.append("Signals were polarity-flipped to make the dominant incident pulse positive.")
    return polarity * incident, polarity * reflected, polarity * transmitted, warnings


def _estimate_initial_modulus(strain: np.ndarray, stress: np.ndarray) -> float:
    finite = np.isfinite(strain) & np.isfinite(stress)
    strain = strain[finite]
    stress = stress[finite]
    if len(strain) < 5:
        return float("nan")
    max_strain = np.nanmax(strain)
    if max_strain <= 0:
        return float("nan")
    mask = (strain >= 0.02 * max_strain) & (strain <= 0.2 * max_strain)
    if np.count_nonzero(mask) < 3:
        mask = np.arange(len(strain)) < min(10, len(strain))
    if np.nanmax(strain[mask]) - np.nanmin(strain[mask]) <= 1e-12:
        return float("nan")
    slope, _ = np.polyfit(strain[mask], stress[mask], 1)
    return float(slope)
