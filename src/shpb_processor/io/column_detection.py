from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pydantic import Field

from shpb_processor.models.base import ProcessorModel


class ColumnDetectionResult(ProcessorModel):
    time_column: str | None = None
    incident_column: str | None = None
    transmitted_column: str | None = None
    confidence: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


TIME_KEYWORDS = ["time", "时间", "t/s", "t(ms)", "t(us)", "t/μs", "t"]
INCIDENT_KEYWORDS = ["incident", "input", "inc", "入射", "入射杆", "incident_strain"]
TRANSMITTED_KEYWORDS = ["transmitted", "transmission", "trans", "透射", "透射杆", "transmitted_strain"]


def detect_columns(dataframe: pd.DataFrame, sampling_frequency_hz: float | None = None) -> ColumnDetectionResult:
    numeric_columns = [col for col in dataframe.columns if pd.to_numeric(dataframe[col], errors="coerce").notna().any()]
    result = ColumnDetectionResult()
    notes: list[str] = []

    result.time_column, time_score = _detect_time_column(dataframe, numeric_columns)
    if result.time_column is None and sampling_frequency_hz:
        notes.append("No time column found; time will be generated from sampling frequency.")
    elif result.time_column is None:
        notes.append("No reliable time column found.")

    result.incident_column, incident_score = _detect_by_name(numeric_columns, INCIDENT_KEYWORDS)
    result.transmitted_column, transmitted_score = _detect_by_name(numeric_columns, TRANSMITTED_KEYWORDS)

    signal_candidates = [c for c in numeric_columns if c != result.time_column]
    if result.incident_column is None or result.transmitted_column is None:
        fallback_incident, fallback_transmitted = _detect_by_pulse_timing(dataframe, signal_candidates)
        if result.incident_column is None:
            result.incident_column = fallback_incident
            incident_score = 0.55 if fallback_incident else 0.0
        if result.transmitted_column is None:
            result.transmitted_column = fallback_transmitted
            transmitted_score = 0.5 if fallback_transmitted else 0.0

    if result.incident_column and result.transmitted_column == result.incident_column:
        alternatives = [c for c in signal_candidates if c != result.incident_column]
        result.transmitted_column = alternatives[0] if alternatives else None
        transmitted_score = 0.35 if result.transmitted_column else 0.0

    result.confidence = {
        "time": time_score,
        "incident": incident_score,
        "transmitted": transmitted_score,
    }
    result.notes = notes
    return result


def _detect_time_column(dataframe: pd.DataFrame, numeric_columns: list[str]) -> tuple[str | None, float]:
    best_column = None
    best_score = 0.0
    for column in numeric_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            continue
        diffs = np.diff(values)
        monotonic = float(np.mean(diffs > 0))
        regularity = 1.0 / (1.0 + _coefficient_of_variation(diffs))
        name_score = _keyword_score(column, TIME_KEYWORDS)
        range_score = 1.0 if values[-1] > values[0] else 0.0
        score = 0.4 * name_score + 0.35 * monotonic + 0.2 * regularity + 0.05 * range_score
        if score > best_score:
            best_column = column
            best_score = score
    if best_score < 0.55:
        return None, best_score
    return best_column, best_score


def _detect_by_name(columns: list[str], keywords: list[str]) -> tuple[str | None, float]:
    best_column = None
    best_score = 0.0
    for column in columns:
        score = _keyword_score(column, keywords)
        if score > best_score:
            best_column = column
            best_score = score
    if best_score <= 0:
        return None, 0.0
    return best_column, min(1.0, 0.65 + 0.35 * best_score)


def _detect_by_pulse_timing(dataframe: pd.DataFrame, columns: list[str]) -> tuple[str | None, str | None]:
    arrivals: list[tuple[int, float, str]] = []
    for column in columns:
        signal = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
        signal = np.nan_to_num(signal, nan=0.0)
        if len(signal) < 5:
            continue
        baseline = signal[: max(5, len(signal) // 20)]
        threshold = np.nanmean(np.abs(baseline)) + 5.0 * np.nanstd(baseline)
        threshold = max(threshold, 0.08 * np.nanmax(np.abs(signal)), 1e-15)
        hits = np.flatnonzero(np.abs(signal) >= threshold)
        if len(hits):
            arrivals.append((int(hits[0]), float(np.nanmax(np.abs(signal))), column))
    if not arrivals:
        return (columns[0], columns[1]) if len(columns) >= 2 else (columns[0] if columns else None, None)
    arrivals.sort(key=lambda item: item[0])
    incident = arrivals[0][2]
    transmitted = arrivals[1][2] if len(arrivals) > 1 else None
    if transmitted is None and len(columns) >= 2:
        transmitted = next((c for c in columns if c != incident), None)
    return incident, transmitted


def _keyword_score(column: str, keywords: list[str]) -> float:
    name = column.strip().lower()
    for keyword in keywords:
        key = keyword.lower()
        if name == key:
            return 1.0
        if re.search(rf"(^|[^a-z0-9]){re.escape(key)}([^a-z0-9]|$)", name):
            return 0.9
        if key in name:
            return 0.75
    return 0.0


def _coefficient_of_variation(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("inf")
    mean = np.nanmean(np.abs(values))
    if mean <= 1e-30:
        return float("inf")
    return float(np.nanstd(values) / mean)
