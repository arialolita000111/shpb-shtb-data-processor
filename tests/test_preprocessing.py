import numpy as np
import pandas as pd

from shpb_processor.models import AcquisitionParameters, ColumnMapping
from shpb_processor.signal_processing import (
    baseline_correct,
    infer_sampling_frequency_from_column,
    infer_sampling_frequency_hz,
    moving_average,
    PreprocessingSettings,
    process_signals,
    savgol_smooth,
)
from shpb_processor.signal_processing.filters import apply_filter


def test_baseline_correct_start_mean():
    time = np.linspace(0, 1, 100)
    signal = np.ones_like(time) * 5
    signal[30:40] += 2
    corrected, info = baseline_correct(time, signal, method="start_mean")
    assert abs(corrected[:5].mean()) < 1e-12
    assert info["offset"] == 5


def test_filters_preserve_length():
    signal = np.sin(np.linspace(0, 1, 101))
    assert len(moving_average(signal, 5)) == len(signal)
    assert len(savgol_smooth(signal, 11, 3)) == len(signal)


def test_no_filter_path_does_not_require_scipy():
    signal = np.sin(np.linspace(0, 1, 101))
    filtered = apply_filter(signal, method="none")
    assert filtered.tolist() == signal.tolist()
    assert filtered is not signal


def test_process_signals_resolves_bom_column_name():
    dataframe = pd.DataFrame(
        {
            "\ufeffTime/us": [0.0, 1.0, 2.0],
            "Incident_strain(με)": [0.0, 10.0, 20.0],
            "Transmitted_strain(με)": [0.0, 5.0, 10.0],
        }
    )
    processed = process_signals(
        dataframe,
        ColumnMapping(
            time_column="Time/us",
            incident_column="Incident_strain(με)",
            transmitted_column="Transmitted_strain(με)",
        ),
        AcquisitionParameters(sampling_frequency_hz=1_000_000, time_unit="us", strain_unit="με"),
    )
    assert processed.time_s.tolist() == [0.0, 1e-6, 2e-6]


def test_infer_sampling_frequency_hz_from_seconds():
    time_s = np.array([0.0, 2e-7, 4e-7, 6e-7])
    assert infer_sampling_frequency_hz(time_s) == 5_000_000


def test_infer_sampling_frequency_from_column_respects_time_unit():
    dataframe = pd.DataFrame({"Time/us": [0.0, 0.2, 0.4, 0.6]})
    assert infer_sampling_frequency_from_column(dataframe, "Time/us", "us") == 5_000_000


def test_process_signals_converts_voltage_with_calibration_factor():
    dataframe = pd.DataFrame(
        {
            "time_us": [0.0, 1.0, 2.0],
            "incident_v": [0.0, 0.5, 1.0],
            "transmitted_v": [0.0, 0.25, 0.5],
        }
    )

    processed = process_signals(
        dataframe,
        ColumnMapping(
            time_column="time_us",
            incident_column="incident_v",
            transmitted_column="transmitted_v",
        ),
        AcquisitionParameters(
            sampling_frequency_hz=1_000_000,
            time_unit="μs",
            strain_unit="voltage",
            voltage_to_microstrain_per_volt=1000.0,
        ),
        PreprocessingSettings(baseline_method="none"),
    )

    assert processed.incident_strain.tolist() == [0.0, 500e-6, 1000e-6]
    assert processed.transmitted_strain.tolist() == [0.0, 250e-6, 500e-6]
