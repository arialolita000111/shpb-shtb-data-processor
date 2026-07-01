from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from shpb_processor.config import WorkspaceConfig, load_workspace_config, save_workspace_config
from shpb_processor.export import export_excel
from shpb_processor.io import load_table
from shpb_processor.processing import ProcessingBundle, process_dataframe
from shpb_processor.report import write_html_report, write_pdf_report


SUPPORTED_INPUT_SUFFIXES = {".csv", ".txt", ".dat", ".xlsx", ".xls"}


def run_batch(
    config_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    config = load_workspace_config(config_path)
    return run_batch_from_config(config, input_dir, output_dir)


def run_batch_from_config(
    config: WorkspaceConfig,
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    source_dir = Path(input_dir)
    output = Path(output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_workspace_config(config, output / "resolved_config.json")

    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for input_path in _iter_input_files(source_dir):
        sample_name = safe_output_name(input_path.stem)
        sample_dir = output / sample_name
        try:
            loaded = load_table(input_path, config.table)
            bundle = process_dataframe(
                loaded.dataframe,
                source_path=str(input_path),
                acquisition=config.acquisition,
                bar=config.bar,
                specimen=config.specimen,
                mapping=config.mapping,
                preprocessing=config.preprocessing,
                pulse_detection=config.pulse_detection,
                alignment=config.alignment,
                calculation=config.calculation,
                dispersion=config.dispersion,
                wave_separation=config.wave_separation,
                quality=config.quality,
                sign_convention=config.sign_convention,
            )
            write_sample_outputs(sample_dir, bundle, config)
            summary_rows.append(_summary_row(input_path, sample_dir, bundle, "processed", ""))
        except Exception as exc:
            sample_dir.mkdir(parents=True, exist_ok=True)
            message = str(exc)
            (sample_dir / "error.txt").write_text(message, encoding="utf-8")
            row = {
                "source_file": str(input_path),
                "sample_dir": str(sample_dir),
                "status": "failed",
                "error": message,
            }
            summary_rows.append(row)
            failure_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output / "batch_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failure_rows).to_csv(output / "failures.csv", index=False, encoding="utf-8-sig")
    (output / "failures.json").write_text(json.dumps(failure_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def process_single_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: WorkspaceConfig,
) -> ProcessingBundle:
    loaded = load_table(input_path, config.table)
    bundle = process_dataframe(
        loaded.dataframe,
        source_path=str(input_path),
        acquisition=config.acquisition,
        bar=config.bar,
        specimen=config.specimen,
        mapping=config.mapping,
        preprocessing=config.preprocessing,
        pulse_detection=config.pulse_detection,
        alignment=config.alignment,
        calculation=config.calculation,
        dispersion=config.dispersion,
        wave_separation=config.wave_separation,
        quality=config.quality,
        sign_convention=config.sign_convention,
    )
    write_sample_outputs(Path(output_dir), bundle, config)
    return bundle


def write_sample_outputs(sample_dir: str | Path, bundle: ProcessingBundle, config: WorkspaceConfig) -> Path:
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if bundle.processed.dataframe is not None:
        bundle.processed.dataframe.to_csv(sample_dir / "processed_signals.csv", index=False, encoding="utf-8-sig")
    if bundle.standard_wave_separation is not None:
        bundle.standard_wave_separation.wave_dataframe().to_csv(
            sample_dir / "standard_wave_separation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bundle.standard_wave_separation.result_dataframe().to_csv(
            sample_dir / "standard_wave_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
    _results_dataframe(bundle).to_csv(sample_dir / "results.csv", index=False, encoding="utf-8-sig")
    _summary_dataframe(bundle).to_csv(sample_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bundle.report_rows).to_csv(sample_dir / "processing_report.csv", index=False, encoding="utf-8-sig")
    (sample_dir / "quality_report.json").write_text(
        json.dumps(bundle.quality.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_workspace_config(config, sample_dir / "config.json")
    if config.output.include_excel_workbook:
        export_excel(
            sample_dir / "result.xlsx",
            raw=bundle.raw,
            processed=bundle.processed,
            segments=bundle.segments,
            aligned=bundle.aligned,
            three_wave=bundle.three_wave,
            two_wave=bundle.two_wave,
            standard_wave_separation=bundle.standard_wave_separation,
            quality=bundle.quality,
            report_rows=bundle.report_rows,
        )
    if config.output.include_html_report:
        write_html_report(
            sample_dir / "report.html",
            bundle,
            title=config.output.report_title,
            include_figures=config.output.include_figures,
        )
    if config.output.include_pdf_report:
        write_pdf_report(
            sample_dir / "report.pdf",
            bundle,
            title=config.output.report_title,
        )
    return sample_dir


def _iter_input_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        and path.stem.lower() not in {"readme", "license", "citation"}
    )


def _results_dataframe(bundle: ProcessingBundle) -> pd.DataFrame:
    if bundle.standard_wave_separation is not None:
        standard = bundle.standard_wave_separation.result_dataframe()
        standard.insert(0, "method", "standard_wave")
        return standard
    three = bundle.three_wave.to_dataframe()
    three.insert(0, "method", "three_wave")
    two = bundle.two_wave.to_dataframe()
    two.insert(0, "method", "two_wave")
    return pd.concat([three, two], ignore_index=True)


def _summary_dataframe(bundle: ProcessingBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bundle.standard_wave_separation is not None:
        rows.extend(
            {"method": "standard_wave", "parameter": key, "value": value}
            for key, value in bundle.standard_wave_separation.summary.items()
        )
        rows.extend(
            {"method": "standard_wave_metadata", "parameter": key, "value": value}
            for key, value in bundle.standard_wave_separation.metadata.items()
        )
    else:
        for method, result in [("three_wave", bundle.three_wave), ("two_wave", bundle.two_wave)]:
            rows.extend({"method": method, "parameter": key, "value": value} for key, value in result.summary.items())
    rows.append({"method": "quality", "parameter": "grade", "value": bundle.quality.grade})
    rows.append({"method": "quality", "parameter": "status", "value": bundle.quality.status})
    rows.extend(
        {"method": "quality_detail", "parameter": key, "value": value}
        for key, value in bundle.quality.details.items()
    )
    rows.extend({"method": "quality", "parameter": key, "value": value} for key, value in bundle.quality.metrics.items())
    return pd.DataFrame(rows)


def _summary_row(
    input_path: Path,
    sample_dir: Path,
    bundle: ProcessingBundle,
    status: str,
    error: str,
) -> dict[str, Any]:
    standard = bundle.standard_wave_separation
    primary = standard.summary if standard is not None else bundle.three_wave.summary
    return {
        "source_file": str(input_path),
        "sample_dir": str(sample_dir),
        "status": status,
        "error": error,
        "primary_method": "standard_wave" if standard is not None else "three_wave",
        "quality_grade": bundle.quality.grade,
        "review_status": bundle.quality.status,
        "material_response_profile": bundle.quality.details.get("material_response_profile"),
        "force_balance_quality": bundle.quality.details.get("force_balance_quality"),
        "active_force_balance_gate": bundle.quality.details.get("active_force_balance_gate"),
        "quality_grade_basis": bundle.quality.details.get("quality_grade_basis"),
        **_quality_severity_fields(bundle),
        "active_mean_balance_review_threshold": bundle.quality.metrics.get("active_mean_balance_review_threshold"),
        "mean_balance_excess_over_active_review_threshold": bundle.quality.metrics.get(
            "mean_balance_excess_over_active_review_threshold"
        ),
        "transmitted_to_incident_peak_ratio": bundle.quality.metrics.get("transmitted_to_incident_peak_ratio"),
        "three_wave_peak_stress_mpa": bundle.three_wave.summary.get("peak_stress_mpa"),
        "two_wave_peak_stress_mpa": bundle.two_wave.summary.get("peak_stress_mpa"),
        "standard_wave_peak_stress_mpa": bundle.standard_wave_separation.summary.get("peak_stress_mpa")
        if bundle.standard_wave_separation is not None
        else None,
        "max_strain": primary.get("max_strain"),
        "max_abs_strain_rate_s^-1": primary.get("max_abs_strain_rate_s^-1"),
        "mean_balance_error": primary.get("mean_balance_error"),
        "report_html": str(sample_dir / "report.html"),
        "report_pdf": str(sample_dir / "report.pdf"),
    }


def _quality_severity_fields(bundle: ProcessingBundle) -> dict[str, Any]:
    severity = bundle.quality.details.get("quality_warning_severity")
    if not isinstance(severity, dict):
        return {}
    return {f"quality_warning_{key}": value for key, value in severity.items()}


def safe_output_name(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or "sample"
