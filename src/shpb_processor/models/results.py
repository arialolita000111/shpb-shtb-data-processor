from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from .base import ProcessorModel


class CalculationResult(ProcessorModel):
    method: str
    time_s: np.ndarray
    strain: np.ndarray
    strain_rate_s1: np.ndarray
    engineering_stress_pa: np.ndarray
    true_strain: np.ndarray
    true_stress_pa: np.ndarray
    force_incident_n: np.ndarray
    force_transmitted_n: np.ndarray
    balance_error: np.ndarray
    summary: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "engineering_strain": self.strain,
                "strain_rate_s^-1": self.strain_rate_s1,
                "engineering_stress_pa": self.engineering_stress_pa,
                "engineering_stress_mpa": self.engineering_stress_pa / 1e6,
                "true_strain": self.true_strain,
                "true_stress_pa": self.true_stress_pa,
                "true_stress_mpa": self.true_stress_pa / 1e6,
                "force_incident_n": self.force_incident_n,
                "force_transmitted_n": self.force_transmitted_n,
                "balance_error": self.balance_error,
            }
        )


class QualityReport(ProcessorModel):
    grade: str
    status: str = "pass"
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
