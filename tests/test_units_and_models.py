from math import pi, sqrt

import pytest

from shpb_processor.calculation import area_to_m2, length_to_m, modulus_to_pa, strain_to_unitless, time_to_s
from shpb_processor.models import BarParameters, SpecimenParameters, SpecimenShape


def test_unit_conversions():
    assert length_to_m(10, "mm") == pytest.approx(0.01)
    assert area_to_m2(100, "mm2") == pytest.approx(1e-4)
    assert modulus_to_pa(200, "GPa") == pytest.approx(200e9)
    assert time_to_s(10, "μs") == pytest.approx(10e-6)
    assert strain_to_unitless(800, "με") == pytest.approx(800e-6)


def test_bar_and_specimen_geometry():
    bar = BarParameters(
        incident_diameter_m=0.0145,
        transmitted_diameter_m=0.0145,
        elastic_modulus_pa=200e9,
        density_kg_m3=7800,
    )
    assert bar.incident_area_m2 == pytest.approx(pi * 0.0145**2 / 4)
    assert bar.resolved_wave_speed_m_s == pytest.approx(sqrt(200e9 / 7800))

    specimen = SpecimenParameters(shape=SpecimenShape.CYLINDER, diameter_m=0.01, length_m=0.008)
    assert specimen.cross_section_area_m2 == pytest.approx(pi * 0.01**2 / 4)

    rectangular = SpecimenParameters(
        shape=SpecimenShape.RECTANGLE,
        width_m=0.012,
        thickness_m=0.008,
        length_m=0.008,
    )
    assert rectangular.cross_section_area_m2 == pytest.approx(0.012 * 0.008)

    custom = SpecimenParameters(shape=SpecimenShape.CUSTOM, area_m2=42e-6, length_m=0.008)
    assert custom.cross_section_area_m2 == pytest.approx(42e-6)
