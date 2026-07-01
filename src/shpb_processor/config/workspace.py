from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from shpb_processor.calculation import CalculationSettings
from shpb_processor.dispersion import DispersionSettings
from shpb_processor.io.loader import TableLoadOptions
from shpb_processor.models import (
    AcquisitionParameters,
    BarParameters,
    ColumnMapping,
    ExperimentType,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
)
from shpb_processor.models.base import ProcessorModel
from shpb_processor.quality import QualitySettings
from shpb_processor.signal_processing import PreprocessingSettings
from shpb_processor.wave import AlignmentSettings, PulseDetectionSettings, StandardWaveSeparationSettings

from .materials import material_bar_parameters


class OutputSettings(ProcessorModel):
    include_excel_workbook: bool = True
    include_html_report: bool = True
    include_pdf_report: bool = True
    include_figures: bool = True
    report_title: str = "SHPB/SHTB processing report"


class CurveStyle(ProcessorModel):
    color: str | None = None
    line_style: str | None = None
    width: float | None = Field(default=None, gt=0)
    show_points: bool | None = None


def default_curve_styles() -> dict[str, CurveStyle]:
    return {}


class PlotStyleSettings(ProcessorModel):
    preset: str = "dark_screen"
    background: str = "#000000"
    foreground: str = "#ffffff"
    grid_enabled: bool = True
    grid_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    default_width: float = Field(default=1.5, gt=0)
    color_sequence: list[str] = Field(
        default_factory=lambda: [
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#ff9900",
            "#7b3294",
            "#008837",
            "#222222",
            "#999999",
        ]
    )
    line_style_sequence: list[str] = Field(default_factory=lambda: ["solid", "dash", "dashdot", "dot"])
    curves: dict[str, CurveStyle] = Field(default_factory=default_curve_styles)


class WorkspaceConfig(ProcessorModel):
    """Serializable processing configuration for GUI, CLI, and Python API use."""

    acquisition: AcquisitionParameters = Field(
        default_factory=lambda: AcquisitionParameters(
            sampling_frequency_hz=5_000_000.0,
            time_unit="us",
            strain_unit="με",
        )
    )
    bar: BarParameters = Field(default_factory=lambda: material_bar_parameters("steel"))
    specimen: SpecimenParameters = Field(
        default_factory=lambda: SpecimenParameters(
            shape=SpecimenShape.CYLINDER,
            diameter_m=0.010,
            length_m=0.008,
            experiment_type=ExperimentType.COMPRESSION,
            specimen_id="sample",
            material_name="unknown",
        )
    )
    mapping: ColumnMapping | None = None
    table: TableLoadOptions = Field(default_factory=TableLoadOptions)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    pulse_detection: PulseDetectionSettings = Field(default_factory=PulseDetectionSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    calculation: CalculationSettings = Field(default_factory=CalculationSettings)
    dispersion: DispersionSettings = Field(default_factory=DispersionSettings)
    wave_separation: StandardWaveSeparationSettings = Field(default_factory=StandardWaveSeparationSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    sign_convention: SignConvention = SignConvention.COMPRESSION_POSITIVE
    plot_style: PlotStyleSettings = Field(default_factory=PlotStyleSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)


def load_workspace_config(path: str | Path) -> WorkspaceConfig:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Workspace config must contain an object: {source}")
    return WorkspaceConfig.model_validate(data)


def save_workspace_config(config: WorkspaceConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    if target.suffix.lower() in {".yaml", ".yml"}:
        target.write_text(_dump_yaml(data), encoding="utf-8")
    else:
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def default_workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig()


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install project dependencies first.") from exc
    return yaml.safe_load(text) or {}


def _dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install project dependencies first.") from exc
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
