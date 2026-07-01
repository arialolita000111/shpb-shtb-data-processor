import numpy as np

from shpb_processor.models import (
    AcquisitionParameters,
    AlignedWaves,
    BarParameters,
    ColumnMapping,
    ExperimentType,
    ProcessedSignalData,
    PulseWindow,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
    WaveSegments,
)
from shpb_processor.quality import QualitySettings, evaluate_quality
from shpb_processor.report import build_processing_report


def test_standard_material_high_balance_error_remains_failure():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.60, balance_error=0.52)
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.008,
        material_name="ductile_tension_surrogate",
        experiment_type=ExperimentType.TENSION,
    )

    report = evaluate_quality(processed, segments, aligned, specimen=specimen)

    assert report.status == "fail"
    assert report.grade == "C"
    assert report.details["force_balance_quality"] == "fail"
    assert report.details["quality_grade_basis"] == "hard_error"
    assert report.details["quality_warning_severity"]["hard_error_count"] == 1


def test_tension_calibration_profile_reviews_surrogate_balance_error():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.60, balance_error=0.50)
    specimen = SpecimenParameters(
        shape=SpecimenShape.DOGBONE,
        diameter_m=0.006,
        length_m=0.016,
        material_name="ductile_tension_surrogate",
        experiment_type=ExperimentType.TENSION,
    )
    settings = QualitySettings(material_response_profile="tension_calibration")

    report = evaluate_quality(processed, segments, aligned, settings=settings, specimen=specimen)

    assert report.status == "review"
    assert report.details["material_response_profile"] == "tension_calibration"
    assert report.details["force_balance_quality"] == "tension_calibration_review_band"


def test_brittle_material_balance_review_band_is_review_not_failure():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.45, balance_error=0.40)
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.008,
        material_name="concrete_brittle_surrogate",
    )

    report = evaluate_quality(processed, segments, aligned, specimen=specimen)

    assert report.status == "review"
    assert report.details["material_response_profile"] == "brittle"
    assert report.details["force_balance_quality"] == "brittle_material_review_band"
    assert report.details["active_force_balance_gate"] == "brittle_material_review_band"
    assert report.metrics["active_mean_balance_review_threshold"] == 0.45
    assert report.metrics["mean_balance_excess_over_active_review_threshold"] == 0.0


def test_soft_material_weak_transmission_reports_review_metric():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.0015, balance_error=0.98)
    aligned.metadata["auto_alignment_status"] = "failed"
    aligned.metadata["warnings"] = [
        "Auto alignment is unreliable; transmitted wave amplitude is too weak for stable matching."
    ]
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.020,
        material_name="polymeric_foam_surrogate",
    )

    report = evaluate_quality(processed, segments, aligned, specimen=specimen)

    assert report.status == "review"
    assert report.grade == "B"
    assert report.details["material_response_profile"] == "soft_weak_transmission"
    assert report.details["force_balance_quality"] == "soft_weak_transmission_review"
    assert report.details["active_force_balance_gate"] == "soft_weak_transmission_review_metric"
    assert report.details["quality_grade_basis"] == "soft_material_expected_review"
    assert report.details["quality_warning_severity"]["unexpected_warning_count"] == 0
    assert report.metrics["mean_balance_excess_over_active_review_threshold"] > 0.0
    assert report.metrics["transmitted_to_incident_peak_ratio"] >= 1.0e-3


def test_soft_material_missing_transmission_still_fails():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.0, balance_error=0.98)
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.020,
        material_name="polymeric_foam_surrogate",
    )
    settings = QualitySettings(soft_material_min_transmitted_to_incident_ratio=1.0e-3)

    report = evaluate_quality(processed, segments, aligned, settings=settings, specimen=specimen)

    assert report.status == "fail"
    assert report.details["force_balance_quality"] == "fail"


def test_processing_report_exposes_material_quality_gate_details():
    processed, segments, aligned = _quality_inputs(transmitted_scale=0.0015, balance_error=0.98)
    aligned.metadata["auto_alignment_status"] = "failed"
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=0.010,
        length_m=0.020,
        material_name="polymeric_foam_surrogate",
    )
    quality = evaluate_quality(processed, segments, aligned, specimen=specimen)
    bar = BarParameters(
        incident_diameter_m=0.0145,
        transmitted_diameter_m=0.0145,
        elastic_modulus_pa=200.0e9,
        density_kg_m3=7800.0,
    )
    acquisition = AcquisitionParameters(sampling_frequency_hz=2_000_000.0, time_unit="us", strain_unit="microstrain")
    mapping = ColumnMapping(time_column="time_us", incident_column="incident", transmitted_column="transmitted")

    rows = build_processing_report(
        source_path="soft.csv",
        bar=bar,
        specimen=specimen,
        acquisition=acquisition,
        mapping=mapping,
        segments=segments,
        aligned=aligned,
        three_wave=None,
        two_wave=None,
        quality=quality,
        sign_convention=SignConvention.COMPRESSION_POSITIVE,
    )
    report = {row["item"]: row["value"] for row in rows}

    assert report["material_response_profile"] == "soft_weak_transmission"
    assert report["force_balance_quality"] == "soft_weak_transmission_review"
    assert report["active_force_balance_gate"] == "soft_weak_transmission_review_metric"
    assert report["soft_weak_transmission_policy"] == "review_metric"
    assert report["quality_grade_basis"] == "soft_material_expected_review"
    assert report["quality_warning_hard_error_count"] == 0
    assert report["quality_warning_unexpected_warning_count"] == 0
    assert report["active_mean_balance_review_threshold"] == 0.35
    assert report["transmitted_to_incident_peak_ratio"] >= 1.0e-3


def _quality_inputs(transmitted_scale: float, balance_error: float):
    time = np.linspace(0.0, 120e-6, 300)
    incident = _gaussian(time, 35e-6, 5e-6, 800e-6)
    reflected = _gaussian(time, 70e-6, 6e-6, -300e-6)
    transmitted = _gaussian(time, 58e-6, 6e-6, transmitted_scale * 800e-6)
    processed = ProcessedSignalData(
        time_s=time,
        incident_strain=incident,
        transmitted_strain=transmitted,
    )
    segments = WaveSegments(
        time_s=time,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_window=PulseWindow(start_s=20e-6, end_s=50e-6, label="incident", confidence=1.0),
        reflected_window=PulseWindow(start_s=58e-6, end_s=88e-6, label="reflected", confidence=1.0),
        transmitted_window=PulseWindow(start_s=45e-6, end_s=75e-6, label="transmitted", confidence=1.0),
    )
    aligned = AlignedWaves(
        time_s=time,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_shift_s=0.0,
        reflected_shift_s=0.0,
        transmitted_shift_s=0.0,
        force_balance_error=np.full_like(time, balance_error, dtype=float),
    )
    return processed, segments, aligned


def _gaussian(time: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)
