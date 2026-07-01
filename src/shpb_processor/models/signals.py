from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from .base import ProcessorModel


class RawSignalData(ProcessorModel):
    dataframe: pd.DataFrame
    source_path: str = ""
    sheet_name: str | int | None = None
    columns: list[str] = Field(default_factory=list)
    stats: dict[str, dict[str, float]] = Field(default_factory=dict)


class ProcessedSignalData(ProcessorModel):
    time_s: np.ndarray
    incident_strain: np.ndarray
    transmitted_strain: np.ndarray
    dataframe: pd.DataFrame | None = None
    preprocessing_log: list[str] = Field(default_factory=list)


class PulseWindow(ProcessorModel):
    start_s: float
    end_s: float
    label: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class WaveSegments(ProcessorModel):
    time_s: np.ndarray
    incident: np.ndarray
    reflected: np.ndarray
    transmitted: np.ndarray
    incident_window: PulseWindow
    reflected_window: PulseWindow
    transmitted_window: PulseWindow
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlignedWaves(ProcessorModel):
    time_s: np.ndarray
    incident: np.ndarray
    reflected: np.ndarray
    transmitted: np.ndarray
    incident_shift_s: float
    reflected_shift_s: float
    transmitted_shift_s: float
    force_balance_error: np.ndarray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
