from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from shpb_processor.calculation import strain_to_unitless, time_to_s
from shpb_processor.models import AcquisitionParameters, ColumnMapping, ProcessedSignalData
from shpb_processor.models.base import ProcessorModel

from .filters import apply_filter
from .preprocessing import baseline_correct, check_signal_anomalies


class PreprocessingSettings(ProcessorModel):
    baseline_method: str = "start_mean"
    baseline_window_s: tuple[float, float] | None = None
    filter_method: str = "none"
    filter_window_points: int = Field(default=11, gt=0)
    filter_cutoff_hz: float | None = Field(default=None, gt=0)
    filter_order: int = Field(default=4, gt=0)
    remove_nan_rows: bool = True


def process_signals(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
    acquisition: AcquisitionParameters,
    settings: PreprocessingSettings | None = None,
) -> ProcessedSignalData:
    settings = settings or PreprocessingSettings()
    log: list[str] = []

    incident_raw = _numeric_column(dataframe, mapping.incident_column)
    transmitted_raw = _numeric_column(dataframe, mapping.transmitted_column)

    if mapping.time_column:
        time_raw = _numeric_column(dataframe, mapping.time_column)
        time_s = np.asarray(time_to_s(time_raw, acquisition.time_unit), dtype=float)
    elif acquisition.sampling_frequency_hz:
        time_s = np.arange(len(dataframe), dtype=float) / acquisition.sampling_frequency_hz
        time_s += acquisition.signal_start_time_s
        log.append("Generated time column from sampling frequency.")
    else:
        raise ValueError("A time column or sampling frequency is required.")

    incident = np.asarray(
        strain_to_unitless(
            incident_raw,
            acquisition.strain_unit,
            acquisition.voltage_to_microstrain_per_volt,
        ),
        dtype=float,
    )
    transmitted = np.asarray(
        strain_to_unitless(
            transmitted_raw,
            acquisition.strain_unit,
            acquisition.voltage_to_microstrain_per_volt,
        ),
        dtype=float,
    )

    valid = np.isfinite(time_s) & np.isfinite(incident) & np.isfinite(transmitted)
    if settings.remove_nan_rows and not np.all(valid):
        removed = int(np.count_nonzero(~valid))
        time_s = time_s[valid]
        incident = incident[valid]
        transmitted = transmitted[valid]
        log.append(f"Removed {removed} rows containing invalid numeric values.")

    order = np.argsort(time_s)
    if not np.all(order == np.arange(len(time_s))):
        time_s = time_s[order]
        incident = incident[order]
        transmitted = transmitted[order]
        log.append("Sorted rows by time.")

    incident, incident_baseline = baseline_correct(
        time_s,
        incident,
        method=settings.baseline_method,
        window_s=settings.baseline_window_s,
    )
    transmitted, transmitted_baseline = baseline_correct(
        time_s,
        transmitted,
        method=settings.baseline_method,
        window_s=settings.baseline_window_s,
    )
    log.append(f"Baseline corrected incident signal: {incident_baseline}.")
    log.append(f"Baseline corrected transmitted signal: {transmitted_baseline}.")

    sampling_frequency = acquisition.sampling_frequency_hz or infer_sampling_frequency_hz(time_s)
    incident = apply_filter(
        incident,
        method=settings.filter_method,
        sampling_frequency_hz=sampling_frequency,
        window_points=settings.filter_window_points,
        cutoff_hz=settings.filter_cutoff_hz,
        order=settings.filter_order,
    )
    transmitted = apply_filter(
        transmitted,
        method=settings.filter_method,
        sampling_frequency_hz=sampling_frequency,
        window_points=settings.filter_window_points,
        cutoff_hz=settings.filter_cutoff_hz,
        order=settings.filter_order,
    )
    if settings.filter_method != "none":
        log.append(f"Applied {settings.filter_method} filter.")

    warnings = check_signal_anomalies(
        time_s,
        {"incident": incident, "transmitted": transmitted},
    )
    log.extend(warnings)

    processed_df = pd.DataFrame(
        {
            "time_s": time_s,
            "incident_strain": incident,
            "transmitted_strain": transmitted,
        }
    )
    return ProcessedSignalData(
        time_s=time_s,
        incident_strain=incident,
        transmitted_strain=transmitted,
        dataframe=processed_df,
        preprocessing_log=log,
    )


def _numeric_column(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    resolved = _resolve_column(dataframe, column)
    values = pd.to_numeric(dataframe[resolved], errors="coerce").to_numpy(dtype=float)
    if np.all(~np.isfinite(values)):
        raise ValueError(f"Column contains no numeric values: {resolved}")
    return values


def infer_sampling_frequency_hz(time_s: np.ndarray) -> float | None:
    time = np.asarray(time_s, dtype=float)
    time = time[np.isfinite(time)]
    if len(time) < 2:
        return None
    time = np.sort(time)
    dt = np.nanmedian(np.diff(time))
    if dt <= 0:
        return None
    return float(1.0 / dt)


def infer_sampling_frequency_from_column(
    dataframe: pd.DataFrame,
    time_column: str,
    time_unit: str,
) -> float | None:
    time_raw = _numeric_column(dataframe, time_column)
    time_s = np.asarray(time_to_s(time_raw, time_unit), dtype=float)
    return infer_sampling_frequency_hz(time_s)


def _resolve_column(dataframe: pd.DataFrame, requested: str) -> str:
    if requested in dataframe.columns:
        return requested

    normalized_requested = _normalize_column_key(requested)
    for column in dataframe.columns:
        if _normalize_column_key(str(column)) == normalized_requested:
            return str(column)

    available = ", ".join(str(column) for column in dataframe.columns)
    raise KeyError(f"Column not found: {requested}. Available columns: {available}")


def _normalize_column_key(value: str) -> str:
    return (
        value.replace("\ufeff", "")
        .replace("µ", "μ")
        .strip()
        .lower()
    )
