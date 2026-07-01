from __future__ import annotations

import math
from typing import Any

from shpb_processor.dispersion import DispersionSettings
from shpb_processor.models import AlignedWaves, BarParameters, WaveSegments
from shpb_processor.wave import AlignmentSettings, align_waves


def resolve_effective_gauge_distances(
    segments: WaveSegments,
    bar: BarParameters,
    dispersion: DispersionSettings | None,
    alignment: AlignmentSettings | None,
) -> tuple[WaveSegments, BarParameters]:
    if not (dispersion and dispersion.enabled):
        return segments, bar

    incident_input = float(bar.incident_gauge_distance_m)
    transmitted_input = float(bar.transmitted_gauge_distance_m)
    incident_missing = incident_input <= 0.0
    transmitted_missing = transmitted_input <= 0.0
    effective_incident = incident_input if incident_input > 0.0 else 0.0
    effective_transmitted = transmitted_input if transmitted_input > 0.0 else 0.0
    incident_source = "input" if incident_input > 0.0 else "unresolved"
    transmitted_source = "input" if transmitted_input > 0.0 else "unresolved"
    metadata: dict[str, Any] = {
        "input_incident_gauge_distance_m": incident_input,
        "input_transmitted_gauge_distance_m": transmitted_input,
        "gauge_distance_resolution_triggered": bool(incident_missing or transmitted_missing),
    }
    warnings: list[str] = []

    if incident_missing or transmitted_missing:
        prealigned = _estimate_alignment(segments, bar, alignment, metadata, warnings)
        passed = bool(prealigned and prealigned.metadata.get("auto_alignment_status") == "passed")
        incident_estimate = _incident_distance_estimate(prealigned, bar) if prealigned else None

        if incident_estimate is not None:
            metadata["estimated_incident_gauge_distance_m"] = incident_estimate
        if incident_missing and passed and incident_estimate is not None:
            effective_incident = incident_estimate
            incident_source = "auto_alignment"

        incident_for_transmitted = effective_incident if effective_incident > 0.0 else None
        transmitted_estimate = (
            _transmitted_distance_estimate(prealigned, bar, incident_for_transmitted)
            if prealigned and incident_for_transmitted is not None
            else None
        )
        if transmitted_estimate is not None:
            metadata["estimated_transmitted_gauge_distance_m"] = transmitted_estimate
        if transmitted_missing and passed and transmitted_estimate is not None:
            effective_transmitted = transmitted_estimate
            transmitted_source = "auto_alignment"

        _add_resolution_warnings(
            warnings,
            incident_missing,
            transmitted_missing,
            incident_source,
            transmitted_source,
            passed,
        )

    metadata.update(
        {
            "effective_incident_gauge_distance_m": effective_incident,
            "effective_transmitted_gauge_distance_m": effective_transmitted,
            "incident_gauge_distance_source": incident_source,
            "transmitted_gauge_distance_source": transmitted_source,
        }
    )
    resolved_segments = _segments_with_distance_metadata(segments, metadata, warnings)
    effective_bar = bar.model_copy(
        update={
            "incident_gauge_distance_m": effective_incident,
            "transmitted_gauge_distance_m": effective_transmitted,
        }
    )
    return resolved_segments, effective_bar


def _estimate_alignment(
    segments: WaveSegments,
    bar: BarParameters,
    alignment: AlignmentSettings | None,
    metadata: dict[str, Any],
    warnings: list[str],
) -> AlignedWaves | None:
    try:
        prealigned = align_waves(segments, bar, alignment)
    except Exception as exc:
        metadata["gauge_distance_estimation_status"] = "failed"
        metadata["gauge_distance_estimation_error"] = str(exc)
        warnings.append("Gauge distance estimation failed; zero input distance was kept for unresolved items.")
        return None

    metadata.update(
        {
            "gauge_distance_estimation_status": prealigned.metadata.get("auto_alignment_status", "unknown"),
            "gauge_distance_estimation_error": prealigned.metadata.get("auto_alignment_final_error"),
            "gauge_distance_estimation_incident_shift_s": prealigned.incident_shift_s,
            "gauge_distance_estimation_reflected_shift_s": prealigned.reflected_shift_s,
            "gauge_distance_estimation_transmitted_shift_s": prealigned.transmitted_shift_s,
        }
    )
    return prealigned


def _incident_distance_estimate(prealigned: AlignedWaves | None, bar: BarParameters) -> float | None:
    if prealigned is None:
        return None
    estimate = 0.5 * bar.resolved_wave_speed_m_s * (prealigned.incident_shift_s - prealigned.reflected_shift_s)
    return _positive_finite_or_none(estimate)


def _transmitted_distance_estimate(
    prealigned: AlignedWaves | None,
    bar: BarParameters,
    incident_effective_m: float | None,
) -> float | None:
    if prealigned is None or incident_effective_m is None:
        return None
    estimate = (
        bar.resolved_wave_speed_m_s * (prealigned.incident_shift_s - prealigned.transmitted_shift_s)
        - incident_effective_m
    )
    return _positive_finite_or_none(estimate)


def _positive_finite_or_none(value: float) -> float | None:
    if math.isfinite(value) and value > 0.0:
        return float(value)
    return None


def _add_resolution_warnings(
    warnings: list[str],
    incident_missing: bool,
    transmitted_missing: bool,
    incident_source: str,
    transmitted_source: str,
    passed: bool,
) -> None:
    if not passed:
        warnings.append(
            "Gauge distance estimation did not pass automatic alignment quality checks; zero input distance was kept for unresolved items."
        )
    unresolved = []
    if incident_missing and incident_source == "unresolved":
        unresolved.append("incident/reflected")
    if transmitted_missing and transmitted_source == "unresolved":
        unresolved.append("transmitted")
    if unresolved:
        waves = ", ".join(unresolved)
        warnings.append(f"Gauge distance is unknown for {waves}; dispersion correction used zero distance for affected waves.")


def _segments_with_distance_metadata(
    segments: WaveSegments,
    distance_metadata: dict[str, Any],
    warnings: list[str],
) -> WaveSegments:
    metadata = dict(segments.metadata)
    metadata.update(distance_metadata)
    if warnings:
        metadata["warnings"] = list(metadata.get("warnings", [])) + warnings
    return segments.model_copy(update={"metadata": metadata})
