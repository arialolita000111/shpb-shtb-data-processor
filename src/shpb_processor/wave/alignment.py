from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field

from shpb_processor.calculation.methods import calculate_balance_error
from shpb_processor.models import AlignedWaves, BarParameters, WaveSegments
from shpb_processor.models.base import ProcessorModel


class AlignmentSettings(ProcessorModel):
    auto_micro_adjust: bool = True
    alignment_objective: Literal["wave_relation", "force_balance", "hybrid"] = "force_balance"
    search_range_s: float | None = Field(default=None, gt=0)
    known_distance_search_limit_s: float = Field(default=10e-6, gt=0)
    known_distance_pulse_fraction: float = Field(default=0.25, gt=0, le=1)
    unknown_distance_pulse_fraction: float = Field(default=0.50, gt=0, le=2)
    max_iterations: int = Field(default=8, ge=1)
    pass_error_threshold: float = Field(default=0.15, gt=0)
    review_error_threshold: float = Field(default=0.30, gt=0)
    min_overlap_points: int = Field(default=20, ge=3)
    min_peak_strain: float = Field(default=1e-12, gt=0)
    min_overlap_incident_peak_fraction: float = Field(default=0.50, gt=0, le=1)
    min_transmitted_peak_fraction: float = Field(default=0.02, ge=0, le=1)
    improvement_tolerance: float = Field(default=1e-6, ge=0)
    align_unknown_distances: bool = True
    force_balance_active_fraction: float = Field(default=0.05, gt=0, le=1)
    hybrid_force_balance_weight: float = Field(default=0.70, ge=0, le=1)


def align_waves(
    segments: WaveSegments,
    bar: BarParameters,
    settings: AlignmentSettings | None = None,
) -> AlignedWaves:
    settings = settings or AlignmentSettings()
    time = np.asarray(segments.time_s, dtype=float)
    dt = _median_dt(time)

    t_inc, inc = _extract_segment(time, segments.incident)
    t_ref, ref = _extract_segment(time, segments.reflected)
    t_tra, tra = _extract_segment(time, segments.transmitted)

    warnings: list[str] = []
    if min(len(t_inc), len(t_ref), len(t_tra)) < 2:
        raise ValueError("Not enough points in one or more wave segments for alignment.")

    unknown_distances = bar.incident_gauge_distance_m <= 0 or bar.transmitted_gauge_distance_m <= 0
    incident_shift, reflected_shift, transmitted_shift, initial_method = _initial_shifts(
        t_inc,
        t_ref,
        t_tra,
        bar,
        unknown_distances,
        settings,
    )
    if unknown_distances:
        warnings.append("Gauge distance is zero or unknown; automatic alignment used the selected objective without physical propagation constraints.")

    initial_error, initial_points = _score_alignment(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
        bar,
        settings,
    )
    initial_metrics = _alignment_score_metrics(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
        bar,
        settings,
    )

    final_reflected_delta = 0.0
    final_transmitted_delta = 0.0
    final_error = initial_error
    iterations = 0
    search_range = _resolve_search_range(segments, settings, unknown_distances, dt)

    if settings.auto_micro_adjust:
        optimization = _optimize_reflected_transmitted_shifts(
            t_inc,
            inc,
            t_ref,
            ref,
            t_tra,
            tra,
            incident_shift,
            reflected_shift,
            transmitted_shift,
            dt,
            search_range,
            bar,
            settings,
        )
        final_reflected_delta = optimization["reflected_delta_s"]
        final_transmitted_delta = optimization["transmitted_delta_s"]
        final_error = optimization["final_error"]
        iterations = int(optimization["iterations"])
        if not math.isfinite(final_error):
            warnings.append("Auto micro-alignment could not evaluate enough overlapping points; using the initial alignment.")
            final_reflected_delta = 0.0
            final_transmitted_delta = 0.0
            final_error = initial_error
        elif math.isfinite(initial_error) and final_error > initial_error:
            warnings.append("Auto micro-alignment did not improve the selected alignment objective; using the initial alignment.")
            final_reflected_delta = 0.0
            final_transmitted_delta = 0.0
            final_error = initial_error
    else:
        warnings.append("Auto micro-alignment is disabled.")

    reflected_shift_final = reflected_shift + final_reflected_delta
    transmitted_shift_final = transmitted_shift + final_transmitted_delta

    aligned = _build_aligned_from_shifts(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift_final,
        transmitted_shift_final,
        dt,
        bar,
        settings,
    )
    alignment_method = initial_method
    if aligned is None:
        warnings.append("Shifted wave windows do not overlap; aligned by normalized segment starts for review.")
        incident_shift, reflected_shift_final, transmitted_shift_final = _normalized_start_shifts(t_inc, t_ref, t_tra)
        alignment_method = "normalized_segment_start_fallback"
        aligned = _build_aligned_from_shifts(
            t_inc,
            inc,
            t_ref,
            ref,
            t_tra,
            tra,
            incident_shift,
            reflected_shift_final,
            transmitted_shift_final,
            dt,
            bar,
            settings,
            require_minimum_points=False,
        )
        final_error, _ = _score_alignment(
            t_inc,
            inc,
            t_ref,
            ref,
            t_tra,
            tra,
            incident_shift,
            reflected_shift_final,
            transmitted_shift_final,
            dt,
            bar,
            settings,
        )
    final_metrics = _alignment_score_metrics(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift_final,
        transmitted_shift_final,
        dt,
        bar,
        settings,
    )
    if aligned is None:
        raise ValueError("Aligned wave windows do not overlap after fallback alignment.")

    incident_peak = float(np.nanmax(np.abs(aligned.incident))) if len(aligned.incident) else 0.0
    transmitted_peak = float(np.nanmax(np.abs(aligned.transmitted))) if len(aligned.transmitted) else 0.0
    weak_transmitted = incident_peak > 0 and transmitted_peak < settings.min_transmitted_peak_fraction * incident_peak
    passed, status = _alignment_status(final_error, settings, weak_transmitted)
    if weak_transmitted:
        warnings.append("Auto alignment is unreliable; transmitted wave amplitude is too weak for stable matching.")
    if status == "review":
        warnings.append("Auto alignment result should be reviewed; selected objective error is moderate.")
    elif status == "failed":
        warnings.append("Auto alignment is unreliable; selected objective error is high.")

    metadata = {
        "warnings": warnings,
        "alignment": alignment_method,
        "auto_micro_adjust_enabled": settings.auto_micro_adjust,
        "auto_alignment_objective": settings.alignment_objective,
        "auto_alignment_relationship": _objective_description(settings.alignment_objective),
        "auto_alignment_initial_error": float(initial_error) if math.isfinite(initial_error) else None,
        "auto_alignment_final_error": float(final_error) if math.isfinite(final_error) else None,
        "auto_alignment_initial_wave_relation_error": _metadata_float(initial_metrics["wave_relation_error"]),
        "auto_alignment_final_wave_relation_error": _metadata_float(final_metrics["wave_relation_error"]),
        "auto_alignment_initial_force_balance_error": _metadata_float(initial_metrics["force_balance_error"]),
        "auto_alignment_final_force_balance_error": _metadata_float(final_metrics["force_balance_error"]),
        "auto_alignment_force_balance_improvement": _improvement_ratio(
            initial_metrics["force_balance_error"],
            final_metrics["force_balance_error"],
        ),
        "auto_alignment_initial_overlap_points": int(initial_points),
        "auto_alignment_reflected_delta_s": float(final_reflected_delta),
        "auto_alignment_transmitted_delta_s": float(final_transmitted_delta),
        "auto_alignment_iterations": iterations,
        "auto_alignment_search_range_s": float(search_range),
        "auto_alignment_unknown_distances": bool(unknown_distances),
        "auto_alignment_passed": bool(passed),
        "auto_alignment_status": status,
        "auto_alignment_pass_error_threshold": settings.pass_error_threshold,
        "auto_alignment_review_threshold": settings.review_error_threshold,
        "auto_alignment_incident_peak": incident_peak,
        "auto_alignment_transmitted_peak": transmitted_peak,
        "auto_alignment_transmitted_too_weak": bool(weak_transmitted),
        "initial_incident_shift_s": float(incident_shift),
        "initial_reflected_shift_s": float(reflected_shift),
        "initial_transmitted_shift_s": float(transmitted_shift),
        "final_incident_shift_s": float(incident_shift),
        "final_reflected_shift_s": float(reflected_shift_final),
        "final_transmitted_shift_s": float(transmitted_shift_final),
    }
    aligned.metadata.update(metadata)
    return aligned


def _initial_shifts(
    t_inc: np.ndarray,
    t_ref: np.ndarray,
    t_tra: np.ndarray,
    bar: BarParameters,
    unknown_distances: bool,
    settings: AlignmentSettings,
) -> tuple[float, float, float, str]:
    if unknown_distances and settings.align_unknown_distances:
        incident_shift, reflected_shift, transmitted_shift = _normalized_start_shifts(t_inc, t_ref, t_tra)
        return incident_shift, reflected_shift, transmitted_shift, "normalized_segment_start_unknown_distance"

    c_bar = bar.resolved_wave_speed_m_s
    return (
        bar.incident_gauge_distance_m / c_bar,
        -bar.incident_gauge_distance_m / c_bar,
        -bar.transmitted_gauge_distance_m / c_bar,
        "propagation_time_with_auto_micro_adjust",
    )


def _normalized_start_shifts(
    t_inc: np.ndarray,
    t_ref: np.ndarray,
    t_tra: np.ndarray,
) -> tuple[float, float, float]:
    incident_shift = 0.0
    reflected_shift = float(t_inc[0] - t_ref[0])
    transmitted_shift = float(t_inc[0] - t_tra[0])
    return incident_shift, reflected_shift, transmitted_shift


def _optimize_reflected_transmitted_shifts(
    t_inc: np.ndarray,
    inc: np.ndarray,
    t_ref: np.ndarray,
    ref: np.ndarray,
    t_tra: np.ndarray,
    tra: np.ndarray,
    incident_shift: float,
    reflected_shift: float,
    transmitted_shift: float,
    dt: float,
    search_range: float,
    bar: BarParameters,
    settings: AlignmentSettings,
) -> dict[str, float | int]:
    best_ref_delta = 0.0
    best_tra_delta = 0.0
    best_error, _ = _score_alignment(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
        bar,
        settings,
    )

    coarse_step = max(dt, search_range / 20.0)
    for candidate_ref in _axis_candidates(0.0, search_range, coarse_step, search_range):
        for candidate_tra in _axis_candidates(0.0, search_range, coarse_step, search_range):
            error, _ = _score_alignment(
                t_inc,
                inc,
                t_ref,
                ref,
                t_tra,
                tra,
                incident_shift,
                reflected_shift + candidate_ref,
                transmitted_shift + candidate_tra,
                dt,
                bar,
                settings,
            )
            if error < best_error:
                best_error = error
                best_ref_delta = float(candidate_ref)
                best_tra_delta = float(candidate_tra)

    step = max(dt, coarse_step / 2.0)
    iterations = 1

    for iteration in range(max(0, settings.max_iterations - 1)):
        previous_error = best_error
        span = search_range if iteration == 0 else min(search_range, 4.0 * step)

        for candidate in _axis_candidates(best_ref_delta, span, step, search_range):
            error, _ = _score_alignment(
                t_inc,
                inc,
                t_ref,
                ref,
                t_tra,
                tra,
                incident_shift,
                reflected_shift + candidate,
                transmitted_shift + best_tra_delta,
                dt,
                bar,
                settings,
            )
            if error < best_error:
                best_error = error
                best_ref_delta = candidate

        for candidate in _axis_candidates(best_tra_delta, span, step, search_range):
            error, _ = _score_alignment(
                t_inc,
                inc,
                t_ref,
                ref,
                t_tra,
                tra,
                incident_shift,
                reflected_shift + best_ref_delta,
                transmitted_shift + candidate,
                dt,
                bar,
                settings,
            )
            if error < best_error:
                best_error = error
                best_tra_delta = candidate

        iterations = iteration + 2
        improvement = previous_error - best_error if math.isfinite(previous_error) and math.isfinite(best_error) else float("inf")
        if improvement < settings.improvement_tolerance and step <= dt * 1.01:
            break
        if improvement < settings.improvement_tolerance and iteration > 0:
            break
        step = max(dt, step / 2.0)

    return {
        "reflected_delta_s": float(best_ref_delta),
        "transmitted_delta_s": float(best_tra_delta),
        "final_error": float(best_error),
        "iterations": iterations,
    }


def _score_alignment(
    t_inc: np.ndarray,
    inc: np.ndarray,
    t_ref: np.ndarray,
    ref: np.ndarray,
    t_tra: np.ndarray,
    tra: np.ndarray,
    incident_shift: float,
    reflected_shift: float,
    transmitted_shift: float,
    dt: float,
    bar: BarParameters,
    settings: AlignmentSettings,
) -> tuple[float, int]:
    metrics = _alignment_score_metrics(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
        bar,
        settings,
    )
    return float(metrics["objective_error"]), int(metrics["overlap_points"])


def _alignment_score_metrics(
    t_inc: np.ndarray,
    inc: np.ndarray,
    t_ref: np.ndarray,
    ref: np.ndarray,
    t_tra: np.ndarray,
    tra: np.ndarray,
    incident_shift: float,
    reflected_shift: float,
    transmitted_shift: float,
    dt: float,
    bar: BarParameters,
    settings: AlignmentSettings,
) -> dict[str, float | int]:
    interpolated = _interpolate_common(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
    )
    if interpolated is None:
        return _failed_score_metrics(0)

    _, incident, reflected, transmitted = interpolated
    if len(incident) < settings.min_overlap_points:
        return _failed_score_metrics(int(len(incident)))

    full_incident_peak = float(np.nanmax(np.abs(inc)))
    overlap_incident_peak = float(np.nanmax(np.abs(incident)))
    if full_incident_peak < settings.min_peak_strain:
        return _failed_score_metrics(int(len(incident)))
    if overlap_incident_peak < settings.min_overlap_incident_peak_fraction * full_incident_peak:
        return _failed_score_metrics(int(len(incident)))

    wave_relation_error = _wave_relation_error(
        inc,
        incident,
        reflected,
        transmitted,
        settings,
    )
    force_balance_error = _force_balance_alignment_error(
        incident,
        reflected,
        transmitted,
        bar,
        settings,
    )
    objective_error = _objective_error(wave_relation_error, force_balance_error, settings)
    return {
        "objective_error": float(objective_error),
        "wave_relation_error": float(wave_relation_error),
        "force_balance_error": float(force_balance_error),
        "overlap_points": int(len(incident)),
    }


def _failed_score_metrics(overlap_points: int) -> dict[str, float | int]:
    return {
        "objective_error": float("inf"),
        "wave_relation_error": float("inf"),
        "force_balance_error": float("inf"),
        "overlap_points": int(overlap_points),
    }


def _wave_relation_error(
    full_incident: np.ndarray,
    incident: np.ndarray,
    reflected: np.ndarray,
    transmitted: np.ndarray,
    settings: AlignmentSettings,
) -> float:
    full_incident_peak = float(np.nanmax(np.abs(full_incident)))
    if full_incident_peak < settings.min_peak_strain:
        return float("inf")

    residual = incident - (transmitted - reflected)
    active_level = max(settings.force_balance_active_fraction * full_incident_peak, settings.min_peak_strain)
    active = np.abs(incident) >= active_level
    finite = np.isfinite(residual)
    mask = active & finite
    if np.count_nonzero(mask) < settings.min_overlap_points:
        mask = finite
    if np.count_nonzero(mask) < settings.min_overlap_points:
        return float("inf")
    return float(_rms(residual[mask]) / full_incident_peak)


def _force_balance_alignment_error(
    incident: np.ndarray,
    reflected: np.ndarray,
    transmitted: np.ndarray,
    bar: BarParameters,
    settings: AlignmentSettings,
) -> float:
    force_incident = bar.elastic_modulus_pa * bar.incident_area_m2 * (incident + reflected)
    force_transmitted = bar.elastic_modulus_pa * bar.transmitted_area_m2 * transmitted
    residual = force_incident - force_transmitted
    force_level = 0.5 * (np.abs(force_incident) + np.abs(force_transmitted))
    finite = np.isfinite(residual) & np.isfinite(force_level)
    if np.count_nonzero(finite) < settings.min_overlap_points:
        return float("inf")

    peak_force = float(np.nanmax(force_level[finite]))
    if peak_force <= 0:
        return float("inf")
    active_level = max(settings.force_balance_active_fraction * peak_force, 1e-12)
    mask = finite & (force_level >= active_level)
    if np.count_nonzero(mask) < settings.min_overlap_points:
        mask = finite
    if np.count_nonzero(mask) < settings.min_overlap_points:
        return float("inf")

    scale = max(_rms(force_level[mask]), 1e-12)
    return float(_rms(residual[mask]) / scale)


def _objective_error(
    wave_relation_error: float,
    force_balance_error: float,
    settings: AlignmentSettings,
) -> float:
    if settings.alignment_objective == "wave_relation":
        return wave_relation_error
    if settings.alignment_objective == "force_balance":
        return force_balance_error

    weight = settings.hybrid_force_balance_weight
    if math.isfinite(wave_relation_error) and math.isfinite(force_balance_error):
        return math.sqrt(weight * force_balance_error**2 + (1.0 - weight) * wave_relation_error**2)
    if math.isfinite(force_balance_error):
        return force_balance_error
    if math.isfinite(wave_relation_error):
        return wave_relation_error
    return float("inf")


def _rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("inf")
    return float(np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2)))


def _build_aligned_from_shifts(
    t_inc: np.ndarray,
    inc: np.ndarray,
    t_ref: np.ndarray,
    ref: np.ndarray,
    t_tra: np.ndarray,
    tra: np.ndarray,
    incident_shift: float,
    reflected_shift: float,
    transmitted_shift: float,
    dt: float,
    bar: BarParameters,
    settings: AlignmentSettings,
    require_minimum_points: bool = True,
) -> AlignedWaves | None:
    interpolated = _interpolate_common(
        t_inc,
        inc,
        t_ref,
        ref,
        t_tra,
        tra,
        incident_shift,
        reflected_shift,
        transmitted_shift,
        dt,
    )
    if interpolated is None:
        return None

    common_time_abs, incident, reflected, transmitted = interpolated
    if require_minimum_points and len(common_time_abs) < settings.min_overlap_points:
        return None

    common_time = common_time_abs - common_time_abs[0]
    force_incident = bar.elastic_modulus_pa * bar.incident_area_m2 * (incident + reflected)
    force_transmitted = bar.elastic_modulus_pa * bar.transmitted_area_m2 * transmitted
    balance_error = calculate_balance_error(force_incident, force_transmitted)

    return AlignedWaves(
        time_s=common_time,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_shift_s=incident_shift,
        reflected_shift_s=reflected_shift,
        transmitted_shift_s=transmitted_shift,
        force_balance_error=balance_error,
        metadata={},
    )


def _interpolate_common(
    t_inc: np.ndarray,
    inc: np.ndarray,
    t_ref: np.ndarray,
    ref: np.ndarray,
    t_tra: np.ndarray,
    tra: np.ndarray,
    incident_shift: float,
    reflected_shift: float,
    transmitted_shift: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    t_inc_shifted = t_inc + incident_shift
    t_ref_shifted = t_ref + reflected_shift
    t_tra_shifted = t_tra + transmitted_shift

    start = max(float(t_inc_shifted[0]), float(t_ref_shifted[0]), float(t_tra_shifted[0]))
    end = min(float(t_inc_shifted[-1]), float(t_ref_shifted[-1]), float(t_tra_shifted[-1]))
    if end <= start:
        return None

    common_time_abs = np.arange(start, end + 0.5 * dt, dt)
    if len(common_time_abs) < 2:
        return None
    return (
        common_time_abs,
        np.interp(common_time_abs, t_inc_shifted, inc),
        np.interp(common_time_abs, t_ref_shifted, ref),
        np.interp(common_time_abs, t_tra_shifted, tra),
    )


def _axis_candidates(center: float, span: float, step: float, search_range: float) -> np.ndarray:
    lower = max(-search_range, center - span)
    upper = min(search_range, center + span)
    if upper < lower:
        return np.array([center], dtype=float)
    count = max(1, int(np.floor((upper - lower) / step)))
    candidates = lower + np.arange(count + 1, dtype=float) * step
    candidates = np.concatenate([candidates, np.array([center, 0.0, upper], dtype=float)])
    candidates = np.clip(candidates, -search_range, search_range)
    return np.unique(np.round(candidates / max(step * 1e-6, 1e-15)) * max(step * 1e-6, 1e-15))


def _resolve_search_range(
    segments: WaveSegments,
    settings: AlignmentSettings,
    unknown_distances: bool,
    dt: float,
) -> float:
    if settings.search_range_s is not None:
        return max(settings.search_range_s, dt)

    pulse_width = _representative_pulse_width_s(segments)
    if unknown_distances:
        search_range = settings.unknown_distance_pulse_fraction * pulse_width
    else:
        search_range = min(settings.known_distance_search_limit_s, settings.known_distance_pulse_fraction * pulse_width)
    return max(float(search_range), 4.0 * dt)


def _representative_pulse_width_s(segments: WaveSegments) -> float:
    durations = [
        segments.incident_window.duration_s,
        segments.reflected_window.duration_s,
        segments.transmitted_window.duration_s,
    ]
    valid = [duration for duration in durations if duration > 0]
    if not valid:
        return 40e-6
    return float(np.nanmedian(valid))


def _objective_description(objective: str) -> str:
    if objective == "force_balance":
        return "E*A_incident*(incident + reflected) ~= E*A_transmitted*transmitted"
    if objective == "hybrid":
        return "force balance and incident ~= transmitted - reflected"
    return "incident ~= transmitted - reflected"


def _metadata_float(value: float | int) -> float | None:
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _improvement_ratio(initial_error: float | int, final_error: float | int) -> float | None:
    initial = float(initial_error)
    final = float(final_error)
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0:
        return None
    return float((initial - final) / initial)


def _alignment_status(final_error: float, settings: AlignmentSettings, weak_transmitted: bool = False) -> tuple[bool, str]:
    if weak_transmitted:
        return False, "failed"
    if not math.isfinite(final_error):
        return False, "failed"
    if final_error < settings.pass_error_threshold:
        return True, "passed"
    if final_error < settings.review_error_threshold:
        return False, "review"
    return False, "failed"


def _extract_segment(time: np.ndarray, masked_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(masked_signal)
    t = time[valid]
    values = masked_signal[valid]
    order = np.argsort(t)
    return t[order], values[order]


def _median_dt(time: np.ndarray) -> float:
    if len(time) < 2:
        return 1.0
    dt = float(np.nanmedian(np.diff(time)))
    return dt if dt > 0 else 1.0
