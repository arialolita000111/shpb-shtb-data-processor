from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from shpb_processor.models import AlignedWaves, ProcessedSignalData, QualityReport, SpecimenParameters, WaveSegments
from shpb_processor.models.base import ProcessorModel


class QualitySettings(ProcessorModel):
    """Quality gate settings for standard, brittle, and soft-material examples."""

    mean_balance_pass_threshold: float = Field(default=0.15, ge=0.0)
    mean_balance_review_threshold: float = Field(default=0.35, ge=0.0)
    brittle_mean_balance_review_threshold: float = Field(default=0.45, ge=0.0)
    tensile_calibration_mean_balance_review_threshold: float = Field(default=0.55, ge=0.0)
    soft_material_weak_transmission_review_ok: bool = True
    soft_material_min_transmitted_to_incident_ratio: float = Field(default=1.0e-3, ge=0.0)
    material_response_profile: Literal[
        "auto",
        "standard",
        "brittle",
        "soft_weak_transmission",
        "tension_calibration",
        "calibration",
    ] = "auto"


def evaluate_quality(
    processed: ProcessedSignalData,
    segments: WaveSegments | None = None,
    aligned: AlignedWaves | None = None,
    balance_threshold: float | None = None,
    settings: QualitySettings | None = None,
    specimen: SpecimenParameters | None = None,
) -> QualityReport:
    settings = settings or QualitySettings()
    if balance_threshold is not None:
        settings = settings.model_copy(update={"mean_balance_pass_threshold": balance_threshold})
    if settings.mean_balance_review_threshold < settings.mean_balance_pass_threshold:
        settings = settings.model_copy(update={"mean_balance_review_threshold": settings.mean_balance_pass_threshold})
    if settings.brittle_mean_balance_review_threshold < settings.mean_balance_review_threshold:
        settings = settings.model_copy(update={"brittle_mean_balance_review_threshold": settings.mean_balance_review_threshold})
    if settings.tensile_calibration_mean_balance_review_threshold < settings.mean_balance_review_threshold:
        settings = settings.model_copy(
            update={"tensile_calibration_mean_balance_review_threshold": settings.mean_balance_review_threshold}
        )

    warnings: list[str] = []
    metrics: dict[str, float] = {}
    details: dict[str, object] = {}
    material_profile = _resolve_material_profile(specimen, settings)
    details["material_response_profile"] = material_profile
    time = np.asarray(processed.time_s, dtype=float)

    if len(time) < 2:
        warnings.append("Time vector is too short.")
    else:
        diffs = np.diff(time)
        metrics["sampling_frequency_hz"] = float(1.0 / np.nanmedian(diffs)) if np.nanmedian(diffs) > 0 else 0.0
        metrics["sampling_interval_cv"] = _cv(diffs)
        if not np.all(diffs > 0):
            warnings.append("Time column is not strictly increasing.")
        if metrics["sampling_interval_cv"] > 0.05:
            warnings.append("Sampling interval is not stable.")

    for name, signal in {
        "incident": processed.incident_strain,
        "transmitted": processed.transmitted_strain,
    }.items():
        signal = np.asarray(signal, dtype=float)
        if not np.all(np.isfinite(signal)):
            warnings.append(f"{name} signal contains NaN or Inf.")
        metrics[f"{name}_peak_abs_strain"] = float(np.nanmax(np.abs(signal))) if len(signal) else 0.0
        metrics[f"{name}_snr"] = _snr(signal)
        if metrics[f"{name}_snr"] < 5:
            warnings.append(f"{name} signal has low signal-to-noise ratio.")

    if segments:
        if segments.incident_window.confidence < 0.2:
            warnings.append("Incident pulse detection confidence is low.")
        if segments.reflected_window.confidence < 0.2:
            warnings.append("Reflected pulse detection confidence is low.")
        if segments.transmitted_window.confidence < 0.2:
            warnings.append("Transmitted pulse detection confidence is low.")
        overlap_fraction = _incident_reflected_overlap_fraction(segments)
        metrics["incident_reflected_overlap_fraction"] = overlap_fraction
        if overlap_fraction > 0.10:
            warnings.append("Incident and reflected windows overlap; wave separation should be reviewed.")
        incident_peak = _peak_abs(segments.incident)
        transmitted_peak = _peak_abs(segments.transmitted)
        if incident_peak > 0:
            metrics["transmitted_to_incident_peak_ratio"] = transmitted_peak / incident_peak
            if metrics["transmitted_to_incident_peak_ratio"] < 0.02:
                warnings.append("Transmitted pulse is very weak relative to the incident pulse.")
            if transmitted_peak > 0:
                metrics["transmitted_segment_peak_abs_strain"] = transmitted_peak
        for key in [
            "dispersion_correction_enabled",
            "dispersion_incident_rms_delta",
            "dispersion_reflected_rms_delta",
            "dispersion_transmitted_rms_delta",
        ]:
            value = segments.metadata.get(key)
            if isinstance(value, bool):
                details[key] = value
            elif isinstance(value, (int, float)) and np.isfinite(value):
                metrics[key] = float(value)
        warnings.extend(segments.metadata.get("warnings", []))

    if aligned:
        balance = aligned.force_balance_error
        if balance is not None and len(balance):
            metrics["mean_balance_error"] = float(np.nanmean(balance))
            metrics["max_balance_error"] = float(np.nanmax(balance))
            _classify_balance_error(
                metrics["mean_balance_error"],
                metrics,
                warnings,
                details,
                settings,
                material_profile,
            )
        for metadata_key, metric_key in {
            "auto_alignment_final_error": "auto_alignment_final_error",
            "auto_alignment_final_wave_relation_error": "auto_alignment_wave_relation_error",
            "auto_alignment_final_force_balance_error": "auto_alignment_force_balance_error",
            "auto_alignment_force_balance_improvement": "auto_alignment_force_balance_improvement",
            "auto_alignment_final_error": "auto_alignment_final_error",
        }.items():
            value = aligned.metadata.get(metadata_key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics[metric_key] = float(value)
        final_error = aligned.metadata.get("auto_alignment_final_error")
        review_threshold = aligned.metadata.get("auto_alignment_review_threshold")
        if isinstance(final_error, (int, float)) and np.isfinite(final_error):
            threshold = float(review_threshold) if isinstance(review_threshold, (int, float)) and review_threshold > 0 else 0.30
            metrics["auto_alignment_confidence"] = float(max(0.0, min(1.0, 1.0 - final_error / threshold)))
        status = aligned.metadata.get("auto_alignment_status")
        if isinstance(status, str):
            details["auto_alignment_status"] = status
            if status == "review":
                warnings.append("Auto alignment status is review.")
            elif status == "failed":
                warnings.append("Auto alignment status is failed.")
        wave_head_ratio = _initial_wave_ratio(aligned)
        metrics["wave_head_initial_ratio"] = wave_head_ratio
        if wave_head_ratio > 0.05:
            warnings.append("Aligned waves start after appreciable wave amplitude; pre-wave baseline may be truncated.")
        warnings.extend(aligned.metadata.get("warnings", []))

    grade = _grade_from_warnings(warnings, metrics, settings, details, material_profile)
    status = _status_from_grade_and_warnings(grade, warnings, details)
    details["review_status"] = status
    return QualityReport(grade=grade, status=status, metrics=metrics, warnings=_dedupe(warnings), details=details)


def _snr(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return 0.0
    noise = signal[: max(5, len(signal) // 10)]
    noise_std = float(np.nanstd(noise))
    peak = float(np.nanmax(np.abs(signal)))
    return peak / max(noise_std, 1e-15)


def _cv(values: np.ndarray) -> float:
    mean = float(np.nanmean(np.abs(values)))
    if mean <= 1e-30:
        return float("inf")
    return float(np.nanstd(values) / mean)


def _classify_balance_error(
    mean_balance: float,
    metrics: dict[str, float],
    warnings: list[str],
    details: dict[str, object],
    settings: QualitySettings,
    material_profile: str,
) -> None:
    metrics["mean_balance_pass_threshold"] = settings.mean_balance_pass_threshold
    metrics["mean_balance_review_threshold"] = settings.mean_balance_review_threshold
    metrics["brittle_mean_balance_review_threshold"] = settings.brittle_mean_balance_review_threshold
    metrics["tensile_calibration_mean_balance_review_threshold"] = (
        settings.tensile_calibration_mean_balance_review_threshold
    )
    active_threshold, gate_name = _active_balance_review_threshold(settings, material_profile)
    metrics["active_mean_balance_review_threshold"] = active_threshold
    metrics["mean_balance_excess_over_pass_threshold"] = max(
        0.0,
        mean_balance - settings.mean_balance_pass_threshold,
    )
    metrics["mean_balance_excess_over_active_review_threshold"] = max(
        0.0,
        mean_balance - active_threshold,
    )
    details["active_force_balance_gate"] = gate_name
    if mean_balance <= settings.mean_balance_pass_threshold:
        details["force_balance_quality"] = "pass"
        return

    transmitted_ratio = metrics.get("transmitted_to_incident_peak_ratio", 0.0)
    soft_review = (
        material_profile == "soft_weak_transmission"
        and settings.soft_material_weak_transmission_review_ok
        and transmitted_ratio >= settings.soft_material_min_transmitted_to_incident_ratio
    )
    if soft_review:
        details["force_balance_quality"] = "soft_weak_transmission_review"
        details["soft_weak_transmission_policy"] = "review_metric"
        metrics["soft_material_min_transmitted_to_incident_ratio"] = settings.soft_material_min_transmitted_to_incident_ratio
        warnings.append(
            "Soft-material weak transmission makes three-wave force balance a review metric rather than a hard failure."
        )
        return

    if material_profile == "calibration":
        details["force_balance_quality"] = "calibration_metric_only"
        warnings.append("Calibration-only data should use wave-speed metrics rather than specimen force balance as the acceptance gate.")
        return

    if material_profile == "brittle" and mean_balance <= settings.brittle_mean_balance_review_threshold:
        details["force_balance_quality"] = "brittle_material_review_band"
        warnings.append("Brittle-material force balance is in the material-model calibration review band.")
        return

    if material_profile == "tension_calibration" and mean_balance <= settings.tensile_calibration_mean_balance_review_threshold:
        details["force_balance_quality"] = "tension_calibration_review_band"
        warnings.append("Tensile calibration force balance is in the material/contact calibration review band.")
        return

    if mean_balance <= settings.mean_balance_review_threshold:
        details["force_balance_quality"] = "review_band"
        warnings.append("Mean force balance error is in the review band.")
        return

    details["force_balance_quality"] = "fail"
    warnings.append("Mean force balance error exceeds the selected failure threshold.")


def _active_balance_review_threshold(settings: QualitySettings, material_profile: str) -> tuple[float, str]:
    if material_profile == "brittle":
        return settings.brittle_mean_balance_review_threshold, "brittle_material_review_band"
    if material_profile == "tension_calibration":
        return settings.tensile_calibration_mean_balance_review_threshold, "tension_calibration_review_band"
    if material_profile == "soft_weak_transmission":
        return settings.mean_balance_review_threshold, "soft_weak_transmission_review_metric"
    if material_profile == "calibration":
        return settings.mean_balance_review_threshold, "calibration_metric_only"
    return settings.mean_balance_review_threshold, "standard_review_band"


def _grade_from_warnings(
    warnings: list[str],
    metrics: dict[str, float],
    settings: QualitySettings,
    details: dict[str, object],
    material_profile: str,
) -> str:
    severity = _warning_severity_summary(warnings, details, material_profile)
    details["quality_warning_severity"] = severity
    if not warnings and metrics.get("mean_balance_error", 0.0) <= settings.mean_balance_pass_threshold * 0.5:
        return "A"
    if severity["hard_error_count"] > 0:
        details["quality_grade_basis"] = "hard_error"
        return "C"
    if details.get("force_balance_quality") == "fail":
        details["quality_grade_basis"] = "force_balance_failure"
        return "C"
    if (
        material_profile == "soft_weak_transmission"
        and details.get("force_balance_quality") == "soft_weak_transmission_review"
        and severity["unexpected_warning_count"] == 0
    ):
        details["quality_grade_basis"] = "soft_material_expected_review"
        return "B"
    if len(warnings) >= 4:
        details["quality_grade_basis"] = "warning_count"
        return "C"
    if len(warnings) >= 1:
        details["quality_grade_basis"] = "review_warning"
        return "B"
    details["quality_grade_basis"] = "clean"
    return "A"


def _warning_severity_summary(
    warnings: list[str],
    details: dict[str, object],
    material_profile: str,
) -> dict[str, int]:
    hard_tokens = [
        "not strictly increasing",
        "too short",
        "NaN or Inf",
        "exceeds the selected failure threshold",
    ]
    review_tokens = [
        "Soft-material weak transmission makes three-wave force balance",
        "Brittle-material force balance is in the material-model calibration review band",
        "Tensile calibration force balance is in the material/contact calibration review band",
        "Calibration-only data should use wave-speed metrics",
        "Mean force balance error is in the review band",
    ]
    soft_expected_tokens = [
        "Transmitted pulse is very weak relative to the incident pulse",
        "Auto alignment status is failed",
        "Auto alignment is unreliable",
        "selected objective error is high",
        "pre-wave baseline may be truncated",
    ]
    hard_count = 0
    expected_review_count = 0
    unexpected_count = 0
    force_balance_quality = details.get("force_balance_quality")
    for warning in warnings:
        if any(token in warning for token in hard_tokens):
            hard_count += 1
            continue
        if any(token in warning for token in review_tokens):
            expected_review_count += 1
            continue
        if (
            material_profile == "soft_weak_transmission"
            and force_balance_quality == "soft_weak_transmission_review"
            and any(token in warning for token in soft_expected_tokens)
        ):
            expected_review_count += 1
            continue
        unexpected_count += 1
    return {
        "hard_error_count": hard_count,
        "expected_review_count": expected_review_count,
        "unexpected_warning_count": unexpected_count,
        "total_warning_count": len(warnings),
    }


def _status_from_grade_and_warnings(grade: str, warnings: list[str], details: dict[str, object]) -> str:
    fail_tokens = ["not strictly increasing", "too short", "NaN or Inf"]
    if any(any(token in warning for token in fail_tokens) for warning in warnings):
        return "fail"
    if details.get("force_balance_quality") == "fail":
        return "fail"
    if any("Auto alignment status is failed" in warning for warning in warnings):
        if details.get("force_balance_quality") != "soft_weak_transmission_review":
            return "fail"
    if grade == "B" or warnings:
        return "review"
    return "pass"


def _resolve_material_profile(specimen: SpecimenParameters | None, settings: QualitySettings) -> str:
    if settings.material_response_profile != "auto":
        return settings.material_response_profile
    if specimen is None:
        return "standard"
    material_name = (specimen.material_name or "").lower()
    specimen_id = (specimen.specimen_id or "").lower()
    label = f"{material_name} {specimen_id}"
    if "calibration" in label or "none_" in label:
        return "calibration"
    if any(token in label for token in ("foam", "soft", "cellular", "low_impedance")):
        return "soft_weak_transmission"
    if any(token in label for token in ("concrete", "brittle", "ceramic", "rock", "mortar")):
        return "brittle"
    return "standard"


def _incident_reflected_overlap_fraction(segments: WaveSegments) -> float:
    start = max(segments.incident_window.start_s, segments.reflected_window.start_s)
    end = min(segments.incident_window.end_s, segments.reflected_window.end_s)
    overlap = max(0.0, end - start)
    denominator = max(
        min(segments.incident_window.duration_s, segments.reflected_window.duration_s),
        1e-30,
    )
    return float(overlap / denominator)


def _initial_wave_ratio(aligned: AlignedWaves) -> float:
    ratios: list[float] = []
    for signal in (aligned.incident, aligned.reflected, aligned.transmitted):
        values = np.asarray(signal, dtype=float)
        if len(values) == 0:
            continue
        peak = _peak_abs(values)
        if peak > 0:
            ratios.append(float(abs(values[0]) / peak))
    return max(ratios) if ratios else 0.0


def _peak_abs(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.nanmax(np.abs(finite)))


def _dedupe(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        result.append(warning)
    return result
