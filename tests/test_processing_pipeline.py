import numpy as np

from shpb_processor.io import detect_columns
from shpb_processor.models import (
    AcquisitionParameters,
    BarParameters,
    ColumnMapping,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
)
from shpb_processor.processing import process_dataframe
from shpb_processor.sample_data import generate_synthetic_shpb_case
from shpb_processor.signal_processing import PreprocessingSettings
from shpb_processor.wave import StandardWaveSeparationSettings


def _synthetic_parameters(metadata):
    acquisition = AcquisitionParameters(
        sampling_frequency_hz=float(metadata["sampling_frequency_hz"]),
        time_unit="us",
        strain_unit="microstrain",
    )
    bar = BarParameters(
        incident_diameter_m=float(metadata["incident_diameter_m"]),
        transmitted_diameter_m=float(metadata["transmitted_diameter_m"]),
        elastic_modulus_pa=float(metadata["elastic_modulus_pa"]),
        density_kg_m3=float(metadata["density_kg_m3"]),
        wave_speed_m_s=float(metadata["wave_speed_m_s"]),
        incident_gauge_distance_m=float(metadata["incident_gauge_distance_m"]),
        transmitted_gauge_distance_m=float(metadata["transmitted_gauge_distance_m"]),
    )
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=float(metadata["specimen_diameter_m"]),
        length_m=float(metadata["specimen_length_m"]),
    )
    return acquisition, bar, specimen


def test_column_detection_on_synthetic_data():
    dataframe, _ = generate_synthetic_shpb_case("ideal")
    detected = detect_columns(dataframe)

    assert detected.time_column == str(dataframe.columns[0])
    assert detected.incident_column == str(dataframe.columns[1])
    assert detected.transmitted_column == str(dataframe.columns[2])


def test_full_pipeline_computes_positive_compression_response():
    dataframe, metadata = generate_synthetic_shpb_case("ideal")
    acquisition, bar, specimen = _synthetic_parameters(metadata)
    mapping = ColumnMapping(
        time_column=str(dataframe.columns[0]),
        incident_column=str(dataframe.columns[1]),
        transmitted_column=str(dataframe.columns[2]),
    )

    bundle = process_dataframe(
        dataframe,
        "synthetic",
        acquisition,
        bar,
        specimen,
        mapping=mapping,
        preprocessing=PreprocessingSettings(baseline_method="start_mean", filter_method="none"),
        sign_convention=SignConvention.COMPRESSION_POSITIVE,
    )

    assert len(bundle.aligned.time_s) > 100
    assert np.nanmax(bundle.three_wave.strain) > 0
    assert np.nanmax(bundle.two_wave.strain) > 0
    assert bundle.three_wave.summary["peak_stress_mpa"] > 0
    assert bundle.two_wave.summary["peak_stress_mpa"] > 0
    assert bundle.three_wave.summary["mean_balance_error"] < 0.20
    assert bundle.standard_wave_separation is None


def test_standard_wave_separation_runs_as_optional_processing_branch():
    dataframe, metadata = generate_synthetic_shpb_case("ideal")
    acquisition, bar, specimen = _synthetic_parameters(metadata)
    mapping = ColumnMapping(
        time_column=str(dataframe.columns[0]),
        incident_column=str(dataframe.columns[1]),
        transmitted_column=str(dataframe.columns[2]),
    )

    bundle = process_dataframe(
        dataframe,
        "synthetic",
        acquisition,
        bar,
        specimen,
        mapping=mapping,
        preprocessing=PreprocessingSettings(baseline_method="start_mean", filter_method="none"),
        wave_separation=StandardWaveSeparationSettings(
            enabled=True,
            transmitted_gauge_to_free_end_m=0.75,
        ),
        sign_convention=SignConvention.COMPRESSION_POSITIVE,
    )

    standard = bundle.standard_wave_separation
    assert standard is not None
    assert standard.time_s.shape == bundle.processed.time_s.shape
    assert standard.wave_dataframe().shape[0] == len(bundle.processed.time_s)
    assert standard.result_dataframe().shape[0] == len(bundle.processed.time_s)
    assert standard.summary["standard_wave_separation_enabled"] == 1.0
    assert standard.summary["peak_stress_mpa"] > 0
    assert standard.metadata["transmitted_gauge_to_free_end_m"] == 0.75
    assert standard.metadata["gauge1_reconstruction_max_abs_strain"] < 1e-12
    assert standard.metadata["gauge2_reconstruction_max_abs_strain"] < 1e-12
