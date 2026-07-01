from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from shpb_processor.models import BarParameters


@dataclass(frozen=True)
class BarMaterial:
    name: str
    elastic_modulus_pa: float
    density_kg_m3: float
    poisson_ratio: float | None
    note: str = ""

    @property
    def wave_speed_m_s(self) -> float:
        return sqrt(self.elastic_modulus_pa / self.density_kg_m3)


BAR_MATERIALS: dict[str, BarMaterial] = {
    "steel": BarMaterial("steel", 200e9, 7800.0, 0.30, "Common steel bar"),
    "aluminum_alloy": BarMaterial("aluminum_alloy", 70e9, 2700.0, 0.33, "Common aluminum alloy"),
    "titanium_alloy": BarMaterial("titanium_alloy", 110e9, 4430.0, 0.34, "Typical titanium alloy"),
    "magnesium_alloy": BarMaterial("magnesium_alloy", 45e9, 1800.0, 0.29, "Typical magnesium alloy"),
    "pmma": BarMaterial("pmma", 3.2e9, 1180.0, 0.35, "Low-impedance PMMA bar"),
}


def material_bar_parameters(
    material_key: str = "steel",
    incident_diameter_m: float = 0.0145,
    transmitted_diameter_m: float = 0.0145,
    incident_gauge_distance_m: float = 0.5,
    transmitted_gauge_distance_m: float = 0.25,
) -> BarParameters:
    material = BAR_MATERIALS[material_key]
    return BarParameters(
        incident_diameter_m=incident_diameter_m,
        transmitted_diameter_m=transmitted_diameter_m,
        elastic_modulus_pa=material.elastic_modulus_pa,
        density_kg_m3=material.density_kg_m3,
        poisson_ratio=material.poisson_ratio,
        wave_speed_m_s=material.wave_speed_m_s,
        material_name=material.name,
        incident_gauge_distance_m=incident_gauge_distance_m,
        transmitted_gauge_distance_m=transmitted_gauge_distance_m,
    )
