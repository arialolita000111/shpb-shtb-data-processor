import numpy as np

from shpb_processor.calculation import CalculationSettings, compute_three_wave
from shpb_processor.models import (
    AlignedWaves,
    BarParameters,
    PulseWindow,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
)
from shpb_processor.wave import PulseDetectionSettings, detect_pulses


def test_detect_pulses_adds_default_pre_and_post_padding():
    dt = 1e-6
    time = np.arange(0.0, 200e-6, dt)
    incident = np.zeros_like(time)
    transmitted = np.zeros_like(time)
    incident[(time >= 50e-6) & (time <= 60e-6)] = 1.0
    incident[(time >= 100e-6) & (time <= 110e-6)] = -0.5
    transmitted[(time >= 80e-6) & (time <= 90e-6)] = 0.4

    segments = detect_pulses(
        time,
        incident,
        transmitted,
        PulseDetectionSettings(relative_threshold=0.2, threshold_sigma=1.0),
    )

    assert segments.incident_window.start_s <= 47e-6
    assert segments.incident_window.end_s >= 65e-6
    assert segments.transmitted_window.start_s <= 77e-6
    assert segments.metadata["window_padding_s"] >= 5e-6


def test_zero_reference_correction_removes_initial_stress_offset_and_warns_for_missing_head():
    time = np.linspace(0.0, 50e-6, 100)
    incident = np.linspace(0.8e-3, 1.0e-3, len(time))
    reflected = np.linspace(-0.1e-3, -0.05e-3, len(time))
    transmitted = np.linspace(0.4e-3, 0.5e-3, len(time))
    waves = AlignedWaves(
        time_s=time,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_shift_s=0.0,
        reflected_shift_s=0.0,
        transmitted_shift_s=0.0,
    )
    bar = BarParameters(
        incident_diameter_m=0.0145,
        transmitted_diameter_m=0.0145,
        elastic_modulus_pa=200e9,
        density_kg_m3=7800,
    )
    specimen = SpecimenParameters(shape=SpecimenShape.CYLINDER, diameter_m=0.01, length_m=0.008)

    result = compute_three_wave(
        waves,
        bar,
        specimen,
        SignConvention.COMPRESSION_POSITIVE,
        CalculationSettings(zero_reference_fraction=0.10),
    )

    reference_points = int(result.summary["zero_reference_points"])
    assert abs(np.mean(result.engineering_stress_pa[:reference_points])) < 1e-6
    assert result.summary["initial_stress_offset_pa"] != 0
    assert result.summary["wave_head_retention_ok"] == 0.0
    assert any("Wave head may be insufficiently retained" in warning for warning in result.warnings)
