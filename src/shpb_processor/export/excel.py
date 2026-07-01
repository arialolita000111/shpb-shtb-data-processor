from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from shpb_processor.models import AlignedWaves, CalculationResult, ProcessedSignalData, QualityReport, RawSignalData, WaveSegments
from shpb_processor.wave import StandardWaveSeparationResult


def export_excel(
    path: str | Path,
    raw: RawSignalData | pd.DataFrame | None = None,
    processed: ProcessedSignalData | None = None,
    segments: WaveSegments | None = None,
    aligned: AlignedWaves | None = None,
    three_wave: CalculationResult | None = None,
    two_wave: CalculationResult | None = None,
    standard_wave_separation: StandardWaveSeparationResult | None = None,
    quality: QualityReport | None = None,
    report_rows: list[dict[str, Any]] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = _excel_engine()
    if engine is None:
        raise RuntimeError("Excel export requires xlsxwriter or openpyxl. Install project dependencies first.")

    with pd.ExcelWriter(output, engine=engine) as writer:
        if raw is not None:
            raw_df = raw.dataframe if hasattr(raw, "dataframe") else raw
            _write_sheet(writer, "raw_data", raw_df)
        if processed is not None:
            processed_df = processed.dataframe if processed.dataframe is not None else _processed_df(processed)
            _write_sheet(writer, "processed_signal", processed_df)
        if segments is not None:
            _write_sheet(writer, "wave_segments", _segments_df(segments))
        if aligned is not None:
            _write_sheet(writer, "aligned_waves", _aligned_df(aligned))
            if aligned.force_balance_error is not None:
                _write_sheet(writer, "force_balance", _force_balance_df(aligned))
        if three_wave is not None:
            _write_sheet(writer, "three_wave_results", three_wave.to_dataframe())
        if two_wave is not None:
            _write_sheet(writer, "two_wave_results", two_wave.to_dataframe())
        if standard_wave_separation is not None:
            _write_sheet(writer, "standard_wave_sep", standard_wave_separation.wave_dataframe())
            _write_sheet(writer, "standard_wave_results", standard_wave_separation.result_dataframe())
        summary_rows = []
        if aligned is not None:
            alignment_keys = [
                "auto_alignment_objective",
                "auto_alignment_relationship",
                "auto_alignment_initial_error",
                "auto_alignment_final_error",
                "auto_alignment_initial_wave_relation_error",
                "auto_alignment_final_wave_relation_error",
                "auto_alignment_initial_force_balance_error",
                "auto_alignment_final_force_balance_error",
                "auto_alignment_force_balance_improvement",
                "auto_alignment_status",
            ]
            summary_rows.extend(
                {"method": "alignment", "parameter": key, "value": aligned.metadata[key]}
                for key in alignment_keys
                if key in aligned.metadata
            )
        if three_wave is not None:
            summary_rows.extend({"method": "three_wave", "parameter": k, "value": v} for k, v in three_wave.summary.items())
        if two_wave is not None:
            summary_rows.extend({"method": "two_wave", "parameter": k, "value": v} for k, v in two_wave.summary.items())
        if standard_wave_separation is not None:
            summary_rows.extend(
                {"method": "standard_wave", "parameter": k, "value": v}
                for k, v in standard_wave_separation.summary.items()
            )
            summary_rows.extend(
                {"method": "standard_wave_metadata", "parameter": k, "value": v}
                for k, v in standard_wave_separation.metadata.items()
            )
        if quality is not None:
            summary_rows.append({"method": "quality", "parameter": "grade", "value": quality.grade})
            summary_rows.append({"method": "quality", "parameter": "status", "value": quality.status})
            summary_rows.extend({"method": "quality_detail", "parameter": k, "value": v} for k, v in quality.details.items())
            summary_rows.extend({"method": "quality", "parameter": k, "value": v} for k, v in quality.metrics.items())
        _write_sheet(writer, "summary_parameters", pd.DataFrame(summary_rows))
        if report_rows is not None:
            _write_sheet(writer, "processing_report", pd.DataFrame(report_rows))
    return output


def _excel_engine() -> str | None:
    if importlib.util.find_spec("xlsxwriter"):
        return "xlsxwriter"
    if importlib.util.find_spec("openpyxl"):
        return "openpyxl"
    return None


def _write_sheet(writer: pd.ExcelWriter, name: str, dataframe: pd.DataFrame) -> None:
    safe_name = name[:31]
    dataframe.to_excel(writer, sheet_name=safe_name, index=False)
    worksheet = writer.sheets[safe_name]
    if hasattr(worksheet, "freeze_panes") and callable(getattr(worksheet, "freeze_panes")):
        worksheet.freeze_panes(1, 0)
    elif hasattr(worksheet, "freeze_panes"):
        worksheet.freeze_panes = "A2"


def _processed_df(processed: ProcessedSignalData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": processed.time_s,
            "incident_strain": processed.incident_strain,
            "transmitted_strain": processed.transmitted_strain,
        }
    )


def _segments_df(segments: WaveSegments) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": segments.time_s,
            "incident_segment": segments.incident,
            "reflected_segment": segments.reflected,
            "transmitted_segment": segments.transmitted,
        }
    )


def _aligned_df(aligned: AlignedWaves) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": aligned.time_s,
            "incident_strain": aligned.incident,
            "reflected_strain": aligned.reflected,
            "transmitted_strain": aligned.transmitted,
            "transmitted_minus_reflected": aligned.transmitted - aligned.reflected,
            "tr_minus_re_minus_incident": (aligned.transmitted - aligned.reflected) - aligned.incident,
        }
    )


def _force_balance_df(aligned: AlignedWaves) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": aligned.time_s,
            "force_balance_error": aligned.force_balance_error,
        }
    )
