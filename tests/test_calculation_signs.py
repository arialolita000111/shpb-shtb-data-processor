import numpy as np
import pytest

from shpb_processor.calculation import CalculationSettings, compute_three_wave, compute_two_wave
from shpb_processor.models import (
    AlignedWaves,
    BarParameters,
    ExperimentType,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
)


@pytest.mark.parametrize(
    ("experiment_type", "sign_convention"),
    [
        (ExperimentType.COMPRESSION, SignConvention.COMPRESSION_POSITIVE),
        (ExperimentType.TENSION, SignConvention.TENSION_POSITIVE),
    ],
)
def test_two_wave_strain_sign_matches_three_wave_for_reflected_release_pulse(
    experiment_type: ExperimentType,
    sign_convention: SignConvention,
):
    time_s = np.linspace(0.0, 100e-6, 101)
    pulse = np.sin(np.pi * np.linspace(0.0, 1.0, len(time_s))) ** 2
    incident = 1.0e-3 * pulse
    reflected = -0.4e-3 * pulse
    transmitted = 0.6e-3 * pulse
    waves = AlignedWaves(
        time_s=time_s,
        incident=incident,
        reflected=reflected,
        transmitted=transmitted,
        incident_shift_s=0.0,
        reflected_shift_s=0.0,
        transmitted_shift_s=0.0,
    )
    bar = BarParameters(
        incident_diameter_m=16e-3,
        transmitted_diameter_m=16e-3,
        elastic_modulus_pa=210e9,
        density_kg_m3=7800.0,
        wave_speed_m_s=5188.75,
    )
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=3e-3,
        length_m=10e-3,
        experiment_type=experiment_type,
    )
    settings = CalculationSettings(zero_reference_enabled=False)

    three_wave = compute_three_wave(waves, bar, specimen, sign_convention, settings)
    two_wave = compute_two_wave(waves, bar, specimen, sign_convention, settings)

    assert three_wave.strain[-1] > 0.0
    assert two_wave.strain[-1] > 0.0
    np.testing.assert_allclose(two_wave.strain_rate_s1, three_wave.strain_rate_s1)
    np.testing.assert_allclose(two_wave.strain, three_wave.strain)
