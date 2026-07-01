from __future__ import annotations

from enum import Enum
from math import pi, sqrt
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .base import ProcessorModel


class ExperimentType(str, Enum):
    COMPRESSION = "compression"
    TENSION = "tension"


class SignConvention(str, Enum):
    COMPRESSION_POSITIVE = "compression_positive"
    TENSION_POSITIVE = "tension_positive"
    RAW = "raw"


class SpecimenShape(str, Enum):
    CYLINDER = "cylinder"
    RECTANGLE = "rectangle"
    SHEET = "sheet"
    DOGBONE = "dogbone"
    CUSTOM = "custom"


class BarParameters(ProcessorModel):
    incident_diameter_m: float = Field(gt=0, description="Incident bar diameter in m")
    transmitted_diameter_m: float = Field(gt=0, description="Transmitted bar diameter in m")
    elastic_modulus_pa: float = Field(gt=0, description="Bar Young's modulus in Pa")
    density_kg_m3: float = Field(gt=0, description="Bar density in kg/m^3")
    material_name: str = "steel"
    poisson_ratio: float | None = Field(default=None, ge=0, lt=0.5)
    wave_speed_m_s: float | None = Field(default=None, gt=0)
    incident_gauge_distance_m: float = Field(default=0.0, ge=0)
    transmitted_gauge_distance_m: float = Field(default=0.0, ge=0)

    @property
    def incident_area_m2(self) -> float:
        return pi * self.incident_diameter_m**2 / 4.0

    @property
    def transmitted_area_m2(self) -> float:
        return pi * self.transmitted_diameter_m**2 / 4.0

    @property
    def average_area_m2(self) -> float:
        return 0.5 * (self.incident_area_m2 + self.transmitted_area_m2)

    @property
    def resolved_wave_speed_m_s(self) -> float:
        if self.wave_speed_m_s:
            return self.wave_speed_m_s
        return sqrt(self.elastic_modulus_pa / self.density_kg_m3)

    @property
    def incident_impedance_n_s_m(self) -> float:
        return self.density_kg_m3 * self.resolved_wave_speed_m_s * self.incident_area_m2

    @property
    def transmitted_impedance_n_s_m(self) -> float:
        return self.density_kg_m3 * self.resolved_wave_speed_m_s * self.transmitted_area_m2


class SpecimenParameters(ProcessorModel):
    shape: SpecimenShape = SpecimenShape.CYLINDER
    length_m: float = Field(gt=0, description="Initial specimen length or gauge length")
    experiment_type: ExperimentType = ExperimentType.COMPRESSION
    diameter_m: float | None = Field(default=None, gt=0)
    width_m: float | None = Field(default=None, gt=0)
    thickness_m: float | None = Field(default=None, gt=0)
    area_m2: float | None = Field(default=None, gt=0)
    density_kg_m3: float | None = Field(default=None, gt=0)
    specimen_id: str = ""
    material_name: str = ""

    @property
    def cross_section_area_m2(self) -> float:
        if self.area_m2:
            return self.area_m2
        if self.shape == SpecimenShape.CYLINDER and self.diameter_m:
            return pi * self.diameter_m**2 / 4.0
        if self.shape in {SpecimenShape.RECTANGLE, SpecimenShape.SHEET, SpecimenShape.DOGBONE}:
            if self.width_m and self.thickness_m:
                return self.width_m * self.thickness_m
        raise ValueError("Specimen cross-section area is missing or cannot be inferred.")


class AcquisitionParameters(ProcessorModel):
    sampling_frequency_hz: float | None = Field(default=None, gt=0)
    time_unit: Literal["s", "ms", "us", "μs"] = "s"
    strain_unit: Literal["strain", "microstrain", "με", "ue", "voltage"] = "strain"
    voltage_to_microstrain_per_volt: float | None = Field(default=None, gt=0)
    signal_start_time_s: float = 0.0
    trigger_time_s: float | None = None

    @field_validator("time_unit", mode="before")
    @classmethod
    def normalize_time_unit(cls, value: str) -> str:
        if value == "µs":
            return "μs"
        return value


class ColumnMapping(ProcessorModel):
    time_column: str | None = None
    incident_column: str
    transmitted_column: str


class ExportConfig(ProcessorModel):
    output_path: Path
    include_raw_data: bool = True
    include_processed_data: bool = True
    image_format: Literal["png", "jpg", "svg", "pdf"] = "png"
