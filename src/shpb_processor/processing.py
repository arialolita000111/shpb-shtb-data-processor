from __future__ import annotations

import pandas as pd
from pydantic import Field

from shpb_processor.calculation import CalculationSettings, compute_three_wave, compute_two_wave
from shpb_processor.dispersion import DispersionSettings, correct_wave_segments
from shpb_processor.gauge_distance import resolve_effective_gauge_distances
from shpb_processor.io import detect_columns
from shpb_processor.models import (
    AcquisitionParameters,
    AlignedWaves,
    BarParameters,
    CalculationResult,
    ColumnMapping,
    ProcessedSignalData,
    QualityReport,
    RawSignalData,
    SignConvention,
    SpecimenParameters,
    WaveSegments,
)
from shpb_processor.models.base import ProcessorModel
from shpb_processor.quality import QualitySettings, evaluate_quality
from shpb_processor.report import build_processing_report
from shpb_processor.signal_processing import PreprocessingSettings, process_signals
from shpb_processor.wave import AlignmentSettings, PulseDetectionSettings, align_waves, detect_pulses
from shpb_processor.wave import StandardWaveSeparationResult, StandardWaveSeparationSettings, separate_standard_shpb


class ProcessingBundle(ProcessorModel):
    raw: RawSignalData
    processed: ProcessedSignalData
    uncorrected_segments: WaveSegments | None = None
    segments: WaveSegments
    aligned: AlignedWaves
    three_wave: CalculationResult
    two_wave: CalculationResult
    quality: QualityReport
    standard_wave_separation: StandardWaveSeparationResult | None = None
    report_rows: list[dict[str, object]] = Field(default_factory=list)


def process_dataframe(
    dataframe: pd.DataFrame,
    source_path: str,
    acquisition: AcquisitionParameters,
    bar: BarParameters,
    specimen: SpecimenParameters,
    mapping: ColumnMapping | None = None,
    preprocessing: PreprocessingSettings | None = None,
    pulse_detection: PulseDetectionSettings | None = None,
    alignment: AlignmentSettings | None = None,
    calculation: CalculationSettings | None = None,
    dispersion: DispersionSettings | None = None,
    wave_separation: StandardWaveSeparationSettings | None = None,
    quality: QualitySettings | None = None,
    sign_convention: SignConvention = SignConvention.COMPRESSION_POSITIVE,
) -> ProcessingBundle:
    if mapping is None:
        detected = detect_columns(dataframe, acquisition.sampling_frequency_hz)
        if not detected.incident_column or not detected.transmitted_column:
            raise ValueError("Incident and transmitted strain columns could not be identified.")
        mapping = ColumnMapping(
            time_column=detected.time_column,
            incident_column=detected.incident_column,
            transmitted_column=detected.transmitted_column,
        )

    raw = RawSignalData(
        dataframe=dataframe,
        source_path=source_path,
        columns=list(dataframe.columns),
        stats=_numeric_stats(dataframe),
    )
    processed = process_signals(dataframe, mapping, acquisition, preprocessing)
    segments = detect_pulses(
        processed.time_s,
        processed.incident_strain,
        processed.transmitted_strain,
        pulse_detection,
    )
    dispersion_enabled = bool(dispersion and dispersion.enabled)
    uncorrected_segments = segments.model_copy(deep=True) if dispersion_enabled else None
    segments, effective_bar = resolve_effective_gauge_distances(segments, bar, dispersion, alignment)
    segments = correct_wave_segments(segments, effective_bar, dispersion)
    aligned = align_waves(segments, effective_bar, alignment)
    three_wave = compute_three_wave(aligned, effective_bar, specimen, sign_convention, calculation)
    two_wave = compute_two_wave(aligned, effective_bar, specimen, sign_convention, calculation)
    standard_wave_result = None
    if wave_separation and wave_separation.enabled:
        standard_wave_result = separate_standard_shpb(
            processed,
            effective_bar,
            specimen,
            wave_separation,
            sign_convention,
        )
    quality_report = evaluate_quality(processed, segments, aligned, settings=quality, specimen=specimen)
    report_rows = build_processing_report(
        source_path=source_path,
        bar=bar,
        specimen=specimen,
        acquisition=acquisition,
        mapping=mapping,
        segments=segments,
        aligned=aligned,
        three_wave=three_wave,
        two_wave=two_wave,
        quality=quality_report,
        sign_convention=sign_convention,
        standard_wave_separation=standard_wave_result,
    )
    return ProcessingBundle(
        raw=raw,
        processed=processed,
        uncorrected_segments=uncorrected_segments,
        segments=segments,
        aligned=aligned,
        three_wave=three_wave,
        two_wave=two_wave,
        quality=quality_report,
        standard_wave_separation=standard_wave_result,
        report_rows=report_rows,
    )


def _numeric_stats(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in dataframe.columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        if values.notna().any():
            stats[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
    return stats
