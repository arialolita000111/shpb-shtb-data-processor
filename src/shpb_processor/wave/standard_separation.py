from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from shpb_processor.calculation.methods import (
    calculate_balance_error,
    engineering_to_true,
    summarize_result,
)
from shpb_processor.models import BarParameters, ProcessedSignalData, SignConvention, SpecimenParameters
from shpb_processor.models.base import ProcessorModel


class StandardWaveSeparationSettings(ProcessorModel):
    """Settings for standard SHPB virtual-gauge recursive wave separation."""

    enabled: bool = False
    transmitted_gauge_to_free_end_m: float = Field(default=0.0, ge=0)
    incident_gauge_to_specimen_m: float | None = Field(default=None, ge=0)
    transmitted_gauge_to_specimen_m: float | None = Field(default=None, ge=0)
    zero_reference_fraction: float = Field(default=0.0, ge=0, le=0.5)
    zero_reference_min_points: int = Field(default=3, ge=1)


class StandardWaveSeparationResult(ProcessorModel):
    time_s: np.ndarray
    gauge1_signal: np.ndarray
    gauge1_right_going: np.ndarray
    gauge1_left_going: np.ndarray
    incident_end_right_going: np.ndarray
    incident_end_left_going: np.ndarray
    gauge2_signal: np.ndarray
    gauge2_right_going: np.ndarray
    gauge2_left_going: np.ndarray
    transmitted_end_right_going: np.ndarray
    transmitted_end_left_going: np.ndarray
    specimen_force_from_incident_end_n: np.ndarray
    specimen_force_from_transmitted_end_n: np.ndarray
    engineering_strain: np.ndarray
    strain_rate_s1: np.ndarray
    engineering_stress_pa: np.ndarray
    true_strain: np.ndarray
    true_stress_pa: np.ndarray
    balance_error: np.ndarray
    summary: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def wave_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "gauge1_signal": self.gauge1_signal,
                "gauge1_right_going": self.gauge1_right_going,
                "gauge1_left_going": self.gauge1_left_going,
                "incident_end_right_going": self.incident_end_right_going,
                "incident_end_left_going": self.incident_end_left_going,
                "gauge2_signal": self.gauge2_signal,
                "gauge2_right_going": self.gauge2_right_going,
                "gauge2_left_going": self.gauge2_left_going,
                "transmitted_end_right_going": self.transmitted_end_right_going,
                "transmitted_end_left_going": self.transmitted_end_left_going,
                "specimen_force_from_incident_end_n": self.specimen_force_from_incident_end_n,
                "specimen_force_from_transmitted_end_n": self.specimen_force_from_transmitted_end_n,
            }
        )

    def result_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "engineering_strain": self.engineering_strain,
                "strain_rate_s^-1": self.strain_rate_s1,
                "engineering_stress_pa": self.engineering_stress_pa,
                "engineering_stress_mpa": self.engineering_stress_pa / 1e6,
                "true_strain": self.true_strain,
                "true_stress_pa": self.true_stress_pa,
                "true_stress_mpa": self.true_stress_pa / 1e6,
                "force_incident_n": self.specimen_force_from_incident_end_n,
                "force_transmitted_n": self.specimen_force_from_transmitted_end_n,
                "balance_error": self.balance_error,
            }
        )


def separate_standard_shpb(
    processed: ProcessedSignalData,
    bar: BarParameters,
    specimen: SpecimenParameters,
    settings: StandardWaveSeparationSettings,
    sign_convention: SignConvention = SignConvention.COMPRESSION_POSITIVE,
) -> StandardWaveSeparationResult:
    """Separate overlapping waves using the standard SHPB virtual-gauge method."""

    if not settings.enabled:
        raise ValueError("Standard SHPB wave separation is disabled.")
    time_s = np.asarray(processed.time_s, dtype=float)
    gauge1 = _zero_reference(
        np.asarray(processed.incident_strain, dtype=float),
        settings.zero_reference_fraction,
        settings.zero_reference_min_points,
    )
    gauge2 = _zero_reference(
        np.asarray(processed.transmitted_strain, dtype=float),
        settings.zero_reference_fraction,
        settings.zero_reference_min_points,
    )
    _validate_inputs(time_s, gauge1, gauge2, bar, settings)

    c_bar = bar.resolved_wave_speed_m_s
    incident_distance = (
        settings.incident_gauge_to_specimen_m
        if settings.incident_gauge_to_specimen_m is not None
        else bar.incident_gauge_distance_m
    )
    transmitted_distance = (
        settings.transmitted_gauge_to_specimen_m
        if settings.transmitted_gauge_to_specimen_m is not None
        else bar.transmitted_gauge_distance_m
    )
    incident_tau = float(incident_distance) / c_bar
    transmitted_tau = float(transmitted_distance) / c_bar
    free_end_tau = settings.transmitted_gauge_to_free_end_m / c_bar

    gauge2_right, gauge2_left = _separate_transmitted_bar(time_s, gauge2, free_end_tau)
    transmitted_end_right = _interp_zero(time_s, gauge2_right, time_s + transmitted_tau)
    transmitted_end_left = _interp_zero(time_s, gauge2_left, time_s - transmitted_tau)
    transmitted_force = bar.elastic_modulus_pa * bar.transmitted_area_m2 * (
        transmitted_end_right + transmitted_end_left
    )

    incident_ea = bar.elastic_modulus_pa * bar.incident_area_m2
    virtual_incident_end_strain = transmitted_force / incident_ea
    gauge1_right, gauge1_left = _separate_incident_bar(time_s, gauge1, virtual_incident_end_strain, incident_tau)
    incident_end_right = _separate_incident_end_right(time_s, gauge1, virtual_incident_end_strain, incident_tau)
    incident_end_left = virtual_incident_end_strain - incident_end_right
    incident_force = incident_ea * (incident_end_right + incident_end_left)

    strain_rate = c_bar / specimen.length_m * (
        incident_end_right - incident_end_left - transmitted_end_right + transmitted_end_left
    )
    strain = _integrate_strain(time_s, strain_rate)
    stress = 0.5 * (incident_force + transmitted_force) / specimen.cross_section_area_m2
    if sign_convention == SignConvention.TENSION_POSITIVE:
        strain_rate = -strain_rate
        strain = -strain
        stress = -stress

    true_strain, true_stress, true_warnings = engineering_to_true(strain, stress, specimen.experiment_type)
    balance_error = calculate_balance_error(incident_force, transmitted_force)
    summary = summarize_result(time_s, strain, strain_rate, stress, balance_error)
    summary.update(
        {
            "standard_wave_separation_enabled": 1.0,
            "standard_wave_incident_tau_s": incident_tau,
            "standard_wave_transmitted_tau_s": transmitted_tau,
            "standard_wave_free_end_tau_s": free_end_tau,
            "standard_wave_transmitted_gauge_to_free_end_m": settings.transmitted_gauge_to_free_end_m,
        }
    )
    metadata = {
        "model": "standard_shpb_recursive_virtual_gauge",
        "interpolation": "linear_left_right_zero_fill",
        "incident_end_left_policy": "virtual_force_closure",
        "incident_gauge_to_specimen_m": float(incident_distance),
        "transmitted_gauge_to_specimen_m": float(transmitted_distance),
        "transmitted_gauge_to_free_end_m": settings.transmitted_gauge_to_free_end_m,
        "wave_speed_m_s": c_bar,
        "zero_reference_fraction": settings.zero_reference_fraction,
        "force_closure_max_abs_n": float(np.nanmax(np.abs(incident_force - transmitted_force))),
        "gauge1_reconstruction_max_abs_strain": float(np.nanmax(np.abs(gauge1_right + gauge1_left - gauge1))),
        "gauge2_reconstruction_max_abs_strain": float(np.nanmax(np.abs(gauge2_right + gauge2_left - gauge2))),
    }
    return StandardWaveSeparationResult(
        time_s=time_s,
        gauge1_signal=gauge1,
        gauge1_right_going=gauge1_right,
        gauge1_left_going=gauge1_left,
        incident_end_right_going=incident_end_right,
        incident_end_left_going=incident_end_left,
        gauge2_signal=gauge2,
        gauge2_right_going=gauge2_right,
        gauge2_left_going=gauge2_left,
        transmitted_end_right_going=transmitted_end_right,
        transmitted_end_left_going=transmitted_end_left,
        specimen_force_from_incident_end_n=incident_force,
        specimen_force_from_transmitted_end_n=transmitted_force,
        engineering_strain=strain,
        strain_rate_s1=strain_rate,
        engineering_stress_pa=stress,
        true_strain=true_strain,
        true_stress_pa=true_stress,
        balance_error=balance_error,
        summary=summary,
        metadata=metadata,
        warnings=true_warnings,
    )


def _separate_transmitted_bar(time_s: np.ndarray, gauge_signal: np.ndarray, free_end_tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros_like(gauge_signal, dtype=float)
    for idx, t_value in enumerate(time_s):
        right[idx] = gauge_signal[idx] + float(_interp_zero(time_s, right, np.asarray([t_value - 2.0 * free_end_tau_s]))[0])
    return right, gauge_signal - right


def _separate_incident_bar(
    time_s: np.ndarray,
    gauge_signal: np.ndarray,
    virtual_end_strain: np.ndarray,
    gauge_to_specimen_tau_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros_like(gauge_signal, dtype=float)
    for idx, t_value in enumerate(time_s):
        virtual_delayed = float(_interp_zero(time_s, virtual_end_strain, np.asarray([t_value - gauge_to_specimen_tau_s]))[0])
        previous_right = float(_interp_zero(time_s, right, np.asarray([t_value - 2.0 * gauge_to_specimen_tau_s]))[0])
        right[idx] = gauge_signal[idx] - virtual_delayed + previous_right
    return right, gauge_signal - right


def _separate_incident_end_right(
    time_s: np.ndarray,
    gauge_signal: np.ndarray,
    virtual_end_strain: np.ndarray,
    gauge_to_specimen_tau_s: float,
) -> np.ndarray:
    right = np.zeros_like(gauge_signal, dtype=float)
    for idx, t_value in enumerate(time_s):
        gauge_delayed = float(_interp_zero(time_s, gauge_signal, np.asarray([t_value - gauge_to_specimen_tau_s]))[0])
        previous_right = float(_interp_zero(time_s, right, np.asarray([t_value - 2.0 * gauge_to_specimen_tau_s]))[0])
        virtual_previous = float(
            _interp_zero(time_s, virtual_end_strain, np.asarray([t_value - 2.0 * gauge_to_specimen_tau_s]))[0]
        )
        right[idx] = gauge_delayed + previous_right - virtual_previous
    return right


def _interp_zero(time_s: np.ndarray, values: np.ndarray, query_time_s: np.ndarray) -> np.ndarray:
    return np.interp(query_time_s, time_s, values, left=0.0, right=0.0)


def _zero_reference(values: np.ndarray, fraction: float, min_points: int) -> np.ndarray:
    if fraction <= 0:
        return values.astype(float, copy=True)
    count = min(len(values), max(min_points, int(round(len(values) * fraction))))
    if count <= 0:
        return values.astype(float, copy=True)
    return values.astype(float, copy=True) - float(np.nanmedian(values[:count]))


def _integrate_strain(time_s: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
    if len(time_s) < 2:
        return np.zeros_like(strain_rate)
    increments = 0.5 * (strain_rate[1:] + strain_rate[:-1]) * np.diff(time_s)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _validate_inputs(
    time_s: np.ndarray,
    gauge1: np.ndarray,
    gauge2: np.ndarray,
    bar: BarParameters,
    settings: StandardWaveSeparationSettings,
) -> None:
    if time_s.ndim != 1 or len(time_s) < 3:
        raise ValueError("Standard wave separation requires at least three time samples.")
    if gauge1.shape != time_s.shape or gauge2.shape != time_s.shape:
        raise ValueError("Standard wave separation signals must match the time array length.")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(gauge1)) or not np.all(np.isfinite(gauge2)):
        raise ValueError("Standard wave separation inputs must contain only finite values.")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Standard wave separation time values must be strictly increasing.")
    if bar.resolved_wave_speed_m_s <= 0:
        raise ValueError("Standard wave separation requires a positive bar wave speed.")
    if settings.transmitted_gauge_to_free_end_m <= 0:
        raise ValueError("Standard wave separation requires transmitted_gauge_to_free_end_m > 0.")
