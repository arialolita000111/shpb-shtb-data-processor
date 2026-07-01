from __future__ import annotations

from datetime import datetime
from typing import Any

from shpb_processor import __version__
from shpb_processor.models import (
    AcquisitionParameters,
    AlignedWaves,
    BarParameters,
    CalculationResult,
    ColumnMapping,
    QualityReport,
    SignConvention,
    SpecimenParameters,
    WaveSegments,
)


def build_processing_report(
    source_path: str,
    bar: BarParameters,
    specimen: SpecimenParameters,
    acquisition: AcquisitionParameters,
    mapping: ColumnMapping,
    segments: WaveSegments | None,
    aligned: AlignedWaves | None,
    three_wave: CalculationResult | None,
    two_wave: CalculationResult | None,
    quality: QualityReport | None,
    sign_convention: SignConvention,
    standard_wave_separation: Any | None = None,
) -> list[dict[str, Any]]:
    standard_mode = standard_wave_separation is not None
    rows: list[dict[str, Any]] = [
        {"item": "source_file", "value": source_path},
        {"item": "processed_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"item": "software_version", "value": __version__},
        {"item": "processing_mode", "value": "standard_wave_separation" if standard_mode else "legacy_three_two_wave"},
        {"item": "bar_material", "value": bar.material_name},
        {"item": "bar_elastic_modulus_pa", "value": bar.elastic_modulus_pa},
        {"item": "bar_density_kg_m3", "value": bar.density_kg_m3},
        {"item": "bar_wave_speed_m_s", "value": bar.resolved_wave_speed_m_s},
        {"item": "incident_bar_diameter_m", "value": bar.incident_diameter_m},
        {"item": "transmitted_bar_diameter_m", "value": bar.transmitted_diameter_m},
        {"item": "incident_gauge_distance_m", "value": bar.incident_gauge_distance_m},
        {"item": "transmitted_gauge_distance_m", "value": bar.transmitted_gauge_distance_m},
        {"item": "specimen_id", "value": specimen.specimen_id},
        {"item": "specimen_material", "value": specimen.material_name},
        {"item": "experiment_type", "value": specimen.experiment_type.value},
        {"item": "specimen_length_m", "value": specimen.length_m},
        {"item": "specimen_area_m2", "value": specimen.cross_section_area_m2},
        {"item": "sampling_frequency_hz", "value": acquisition.sampling_frequency_hz},
        {"item": "time_unit", "value": acquisition.time_unit},
        {"item": "strain_unit", "value": acquisition.strain_unit},
        {"item": "voltage_to_microstrain_per_volt", "value": acquisition.voltage_to_microstrain_per_volt},
        {"item": "time_column", "value": mapping.time_column},
        {"item": "incident_column", "value": mapping.incident_column},
        {"item": "transmitted_column", "value": mapping.transmitted_column},
        {"item": "sign_convention", "value": sign_convention.value},
        {"item": "true_stress_assumption", "value": "volume_constancy_optional_for_mvp"},
    ]

    if standard_mode:
        rows.append({"item": "standard_wave_primary_result", "value": True})
        for key, value in standard_wave_separation.summary.items():
            rows.append({"item": f"standard_wave_{key}", "value": value})
        for key, value in standard_wave_separation.metadata.items():
            rows.append({"item": f"standard_wave_metadata_{key}", "value": value})
        for warning in standard_wave_separation.warnings:
            rows.append({"item": "standard_wave_warning", "value": warning})
    if segments and not standard_mode:
        rows.extend(
            [
                {"item": "incident_window_s", "value": f"{segments.incident_window.start_s:g}-{segments.incident_window.end_s:g}"},
                {"item": "reflected_window_s", "value": f"{segments.reflected_window.start_s:g}-{segments.reflected_window.end_s:g}"},
                {"item": "transmitted_window_s", "value": f"{segments.transmitted_window.start_s:g}-{segments.transmitted_window.end_s:g}"},
                {
                    "item": "dispersion_correction_enabled",
                    "value": bool(segments.metadata.get("dispersion_correction_enabled", False)),
                },
            ]
        )
        for item in [
            "window_source",
            "window_padding_mode",
            "window_padding_s",
            "input_incident_gauge_distance_m",
            "input_transmitted_gauge_distance_m",
            "effective_incident_gauge_distance_m",
            "effective_transmitted_gauge_distance_m",
            "incident_gauge_distance_source",
            "transmitted_gauge_distance_source",
            "gauge_distance_resolution_triggered",
            "gauge_distance_estimation_status",
            "gauge_distance_estimation_error",
            "gauge_distance_estimation_incident_shift_s",
            "gauge_distance_estimation_reflected_shift_s",
            "gauge_distance_estimation_transmitted_shift_s",
            "estimated_incident_gauge_distance_m",
            "estimated_transmitted_gauge_distance_m",
            "dispersion_correction_model",
            "dispersion_calibration_source",
            "dispersion_poisson_ratio",
            "dispersion_phase_strength",
            "dispersion_incident_phase_strength",
            "dispersion_reflected_phase_strength",
            "dispersion_transmitted_phase_strength",
            "dispersion_amplitude_correction",
            "dispersion_amplitude_strength",
            "dispersion_incident_amplitude_strength",
            "dispersion_reflected_amplitude_strength",
            "dispersion_transmitted_amplitude_strength",
            "dispersion_taper_fraction",
            "dispersion_incident_taper_fraction",
            "dispersion_reflected_taper_fraction",
            "dispersion_transmitted_taper_fraction",
            "dispersion_max_correction_frequency_hz",
            "dispersion_incident_max_correction_frequency_hz",
            "dispersion_reflected_max_correction_frequency_hz",
            "dispersion_transmitted_max_correction_frequency_hz",
            "dispersion_incident_rms_delta",
            "dispersion_reflected_rms_delta",
            "dispersion_transmitted_rms_delta",
            "dispersion_note",
        ]:
            if item in segments.metadata:
                rows.append({"item": item, "value": segments.metadata[item]})
    elif segments and standard_mode:
        rows.append(
            {
                "item": "legacy_window_alignment_note",
                "value": "not used as primary result in standard wave separation mode",
            }
        )
        rows.append(
            {
                "item": "dispersion_correction_enabled",
                "value": bool(segments.metadata.get("dispersion_correction_enabled", False)),
            }
        )
    if aligned and not standard_mode:
        rows.extend(
            [
                {"item": "incident_shift_s", "value": aligned.incident_shift_s},
                {"item": "reflected_shift_s", "value": aligned.reflected_shift_s},
                {"item": "transmitted_shift_s", "value": aligned.transmitted_shift_s},
            ]
        )
        alignment_items = [
            "alignment",
            "auto_micro_adjust_enabled",
            "auto_alignment_objective",
            "auto_alignment_relationship",
            "auto_alignment_initial_error",
            "auto_alignment_final_error",
            "auto_alignment_initial_wave_relation_error",
            "auto_alignment_final_wave_relation_error",
            "auto_alignment_initial_force_balance_error",
            "auto_alignment_final_force_balance_error",
            "auto_alignment_force_balance_improvement",
            "auto_alignment_reflected_delta_s",
            "auto_alignment_transmitted_delta_s",
            "auto_alignment_iterations",
            "auto_alignment_search_range_s",
            "auto_alignment_unknown_distances",
            "auto_alignment_passed",
            "auto_alignment_status",
        ]
        for item in alignment_items:
            if item in aligned.metadata:
                rows.append({"item": item, "value": aligned.metadata[item]})
    if three_wave and not standard_mode:
        for key, value in three_wave.summary.items():
            rows.append({"item": f"three_wave_{key}", "value": value})
        for warning in three_wave.warnings:
            rows.append({"item": "three_wave_warning", "value": warning})
    if two_wave and not standard_mode:
        for key, value in two_wave.summary.items():
            rows.append({"item": f"two_wave_{key}", "value": value})
        for warning in two_wave.warnings:
            rows.append({"item": "two_wave_warning", "value": warning})
    if quality:
        rows.append({"item": "quality_grade", "value": quality.grade})
        rows.append({"item": "quality_status", "value": quality.status})
        for item in [
            "material_response_profile",
            "force_balance_quality",
            "active_force_balance_gate",
            "soft_weak_transmission_policy",
            "quality_grade_basis",
            "review_status",
            "auto_alignment_status",
        ]:
            if item in quality.details:
                rows.append({"item": item, "value": quality.details[item]})
        severity = quality.details.get("quality_warning_severity")
        if isinstance(severity, dict):
            for key, value in severity.items():
                rows.append({"item": f"quality_warning_{key}", "value": value})
        for item in [
            "mean_balance_pass_threshold",
            "mean_balance_review_threshold",
            "brittle_mean_balance_review_threshold",
            "tensile_calibration_mean_balance_review_threshold",
            "active_mean_balance_review_threshold",
            "mean_balance_excess_over_pass_threshold",
            "mean_balance_excess_over_active_review_threshold",
            "transmitted_to_incident_peak_ratio",
            "soft_material_min_transmitted_to_incident_ratio",
            "wave_head_initial_ratio",
            "auto_alignment_confidence",
        ]:
            if item in quality.metrics:
                rows.append({"item": item, "value": quality.metrics[item]})
        for warning in quality.warnings:
            rows.append({"item": "warning", "value": warning})
    return rows
