import importlib.util

import pandas as pd
import pytest

from shpb_processor.export import export_excel
from shpb_processor.io import load_table
from shpb_processor.models import AcquisitionParameters, BarParameters, ColumnMapping, SpecimenParameters, SpecimenShape
from shpb_processor.processing import process_dataframe
from shpb_processor.sample_data import generate_synthetic_shpb_case
from shpb_processor.wave import StandardWaveSeparationSettings


def test_load_csv_roundtrip(tmp_path):
    dataframe, _ = generate_synthetic_shpb_case("ideal")
    path = tmp_path / "sample.csv"
    dataframe.to_csv(path, index=False)
    loaded = load_table(path)
    assert len(loaded.dataframe) == len(dataframe)
    assert "Time/us" in loaded.columns


def test_load_utf8_sig_csv_strips_bom_and_preserves_microstrain_header(tmp_path):
    dataframe, _ = generate_synthetic_shpb_case("ideal")
    path = tmp_path / "sample_sig.csv"
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")

    loaded = load_table(path)

    assert loaded.columns[:3] == ["Time/us", "Incident_strain(με)", "Transmitted_strain(με)"]


@pytest.mark.skipif(
    not (importlib.util.find_spec("xlsxwriter") or importlib.util.find_spec("openpyxl")),
    reason="Excel writer dependency not installed",
)
def test_load_xlsx_auto_header_uses_valid_excel_header_argument(tmp_path):
    dataframe, _ = generate_synthetic_shpb_case("ideal")
    path = tmp_path / "sample.xlsx"
    dataframe.to_excel(path, index=False)

    loaded = load_table(path)

    assert len(loaded.dataframe) == len(dataframe)
    assert loaded.columns[:3] == ["Time/us", "Incident_strain(με)", "Transmitted_strain(με)"]


@pytest.mark.skipif(
    not (importlib.util.find_spec("xlsxwriter") or importlib.util.find_spec("openpyxl")),
    reason="Excel writer dependency not installed",
)
def test_excel_export_contains_workbook(tmp_path):
    dataframe, metadata = generate_synthetic_shpb_case("ideal")
    acquisition = AcquisitionParameters(
        sampling_frequency_hz=float(metadata["sampling_frequency_hz"]),
        time_unit="us",
        strain_unit="με",
    )
    bar = BarParameters(
        incident_diameter_m=float(metadata["incident_diameter_m"]),
        transmitted_diameter_m=float(metadata["transmitted_diameter_m"]),
        elastic_modulus_pa=float(metadata["elastic_modulus_pa"]),
        density_kg_m3=float(metadata["density_kg_m3"]),
        wave_speed_m_s=float(metadata["wave_speed_m_s"]),
        incident_gauge_distance_m=float(metadata["incident_gauge_distance_m"]),
        transmitted_gauge_distance_m=float(metadata["transmitted_gauge_distance_m"]),
    )
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=float(metadata["specimen_diameter_m"]),
        length_m=float(metadata["specimen_length_m"]),
    )
    mapping = ColumnMapping(
        time_column="Time/us",
        incident_column="Incident_strain(με)",
        transmitted_column="Transmitted_strain(με)",
    )
    bundle = process_dataframe(dataframe, "synthetic", acquisition, bar, specimen, mapping=mapping)
    output = export_excel(
        tmp_path / "result.xlsx",
        raw=bundle.raw,
        processed=bundle.processed,
        segments=bundle.segments,
        aligned=bundle.aligned,
        three_wave=bundle.three_wave,
        two_wave=bundle.two_wave,
        quality=bundle.quality,
        report_rows=bundle.report_rows,
    )
    assert output.exists()
    workbook = pd.ExcelFile(output)
    assert "three_wave_results" in workbook.sheet_names
    assert "two_wave_results" in workbook.sheet_names


@pytest.mark.skipif(
    not (importlib.util.find_spec("xlsxwriter") or importlib.util.find_spec("openpyxl")),
    reason="Excel writer dependency not installed",
)
def test_excel_export_contains_standard_wave_sheets_when_enabled(tmp_path):
    dataframe, metadata = generate_synthetic_shpb_case("ideal")
    acquisition = AcquisitionParameters(
        sampling_frequency_hz=float(metadata["sampling_frequency_hz"]),
        time_unit="us",
        strain_unit="microstrain",
    )
    bar = BarParameters(
        incident_diameter_m=float(metadata["incident_diameter_m"]),
        transmitted_diameter_m=float(metadata["transmitted_diameter_m"]),
        elastic_modulus_pa=float(metadata["elastic_modulus_pa"]),
        density_kg_m3=float(metadata["density_kg_m3"]),
        wave_speed_m_s=float(metadata["wave_speed_m_s"]),
        incident_gauge_distance_m=float(metadata["incident_gauge_distance_m"]),
        transmitted_gauge_distance_m=float(metadata["transmitted_gauge_distance_m"]),
    )
    specimen = SpecimenParameters(
        shape=SpecimenShape.CYLINDER,
        diameter_m=float(metadata["specimen_diameter_m"]),
        length_m=float(metadata["specimen_length_m"]),
    )
    mapping = ColumnMapping(
        time_column="Time/us",
        incident_column=str(dataframe.columns[1]),
        transmitted_column=str(dataframe.columns[2]),
    )
    bundle = process_dataframe(
        dataframe,
        "synthetic",
        acquisition,
        bar,
        specimen,
        mapping=mapping,
        wave_separation=StandardWaveSeparationSettings(enabled=True, transmitted_gauge_to_free_end_m=0.75),
    )

    output = export_excel(
        tmp_path / "result.xlsx",
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

    workbook = pd.ExcelFile(output)
    assert "standard_wave_sep" in workbook.sheet_names
    assert "standard_wave_results" in workbook.sheet_names
