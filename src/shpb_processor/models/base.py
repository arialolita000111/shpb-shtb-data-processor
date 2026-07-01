from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ProcessorModel(BaseModel):
    """Base model that allows numpy arrays and pandas objects in payloads."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
