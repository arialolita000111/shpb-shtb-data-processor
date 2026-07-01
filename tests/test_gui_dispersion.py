import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from shpb_processor.config import WorkspaceConfig
from shpb_processor.models import ExperimentType, SignConvention
from shpb_processor.ui.main_window import MainWindow
from shpb_processor.wave import StandardWaveSeparationSettings


def test_gui_workspace_config_reflects_dispersion_checkbox():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        dispersion_index = window.center_tabs.indexOf(window.dispersion_plot)
        assert not window.center_tabs.isTabVisible(dispersion_index)

        window.dispersion_enabled.setChecked(True)
        config = window._workspace_config_from_ui("test", include_manual_windows=False)
        assert config.dispersion.enabled is True
        assert window.center_tabs.isTabVisible(dispersion_index)

        window.dispersion_enabled.setChecked(False)
        config = window._workspace_config_from_ui("test", include_manual_windows=False)
        assert config.dispersion.enabled is False
        assert not window.center_tabs.isTabVisible(dispersion_index)
    finally:
        window.close()
        app.processEvents()


def test_gui_workspace_config_reflects_standard_wave_separation_controls():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        segment_index = window.center_tabs.indexOf(window.segment_plot)
        aligned_index = window.center_tabs.indexOf(window.aligned_plot)
        standard_index = window.center_tabs.indexOf(window.standard_wave_plot)
        assert window.center_tabs.isTabVisible(segment_index)
        assert window.center_tabs.isTabVisible(aligned_index)
        assert not window.center_tabs.isTabVisible(standard_index)
        assert not window.transmitted_free_end_distance.isEnabled()

        config = window._workspace_config_from_ui("test", include_manual_windows=False)
        assert config.wave_separation.enabled is False

        window.standard_wave_enabled.setChecked(True)
        window.transmitted_free_end_distance.setValue(812.5)
        config = window._workspace_config_from_ui("test", include_manual_windows=False)
        assert config.wave_separation.enabled is True
        assert config.wave_separation.transmitted_gauge_to_free_end_m == pytest.approx(0.8125)
        assert not window.center_tabs.isTabVisible(segment_index)
        assert not window.center_tabs.isTabVisible(aligned_index)
        assert window.center_tabs.isTabVisible(standard_index)
        assert window.transmitted_free_end_distance.isEnabled()

        loaded = WorkspaceConfig(
            wave_separation=StandardWaveSeparationSettings(
                enabled=True,
                transmitted_gauge_to_free_end_m=0.625,
            )
        )
        window._apply_workspace_config(loaded)
        assert window.standard_wave_enabled.isChecked() is True
        assert window.transmitted_free_end_distance.value() == pytest.approx(625.0)
        assert not window.center_tabs.isTabVisible(segment_index)
        assert not window.center_tabs.isTabVisible(aligned_index)
        assert window.center_tabs.isTabVisible(standard_index)
        assert window.transmitted_free_end_distance.isEnabled()

        window.center_tabs.setCurrentWidget(window.standard_wave_plot)
        window.standard_wave_enabled.setChecked(False)
        assert not window.center_tabs.isTabVisible(standard_index)
        assert window.center_tabs.isTabVisible(segment_index)
        assert window.center_tabs.isTabVisible(aligned_index)
        assert not window.transmitted_free_end_distance.isEnabled()
        assert window.center_tabs.currentWidget() is not window.standard_wave_plot
    finally:
        window.close()
        app.processEvents()


def test_gui_standard_wave_mode_hides_legacy_tabs_and_plots_only_standard_result():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.standard_wave_enabled.setChecked(True)
        window.transmitted_free_end_distance.setValue(750.0)
        window.plot_curve_overrides.setText("result.standard_wave=#123456,dashdot,3.5")
        window._update_plot_style_from_ui()
        window.process_auto()

        assert not window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.segment_plot))
        assert not window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.aligned_plot))
        assert window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.standard_wave_plot))
        assert _legend_labels(window.result_plot) == ["standard wave"]
        standard_pen = window.result_plot.getPlotItem().listDataItems()[0].opts["pen"]
        assert standard_pen.color().name().lower() == "#123456"
        assert standard_pen.widthF() == pytest.approx(3.5)
        assert standard_pen.style() == Qt.PenStyle.DashDotLine
        summary_text = window.summary.toPlainText()
        assert "Main workflow: standard SHPB single-station wave separation" in summary_text
        assert "Engineering peak stress" in summary_text
        assert "Three-wave peak stress" not in summary_text
        assert "Two-wave peak stress" not in summary_text

        window.standard_wave_enabled.setChecked(False)
        window._plot_all()

        assert window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.segment_plot))
        assert window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.aligned_plot))
        assert not window.center_tabs.isTabVisible(window.center_tabs.indexOf(window.standard_wave_plot))
        assert _legend_labels(window.result_plot) == ["three-wave", "two-wave"]
    finally:
        window.close()
        app.processEvents()


def test_gui_plot_style_sequence_applies_when_no_curve_override():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.plot_background.setCurrentText("#ffffff")
        window._set_sequence_combo(window.plot_color_sequence, ["#abcdef", "#fedcba"], is_color=True)
        window._set_sequence_combo(window.plot_line_style_sequence, ["dot", "dash"], is_color=False)
        window.plot_curve_overrides.setText("")
        window._update_plot_style_from_ui()
        window.process_auto()

        items = window.result_plot.getPlotItem().listDataItems()
        assert len(items) == 2
        first_pen = items[0].opts["pen"]
        second_pen = items[1].opts["pen"]
        assert first_pen.color().name().lower() == "#abcdef"
        assert first_pen.style() == Qt.PenStyle.DotLine
        assert second_pen.color().name().lower() == "#fedcba"
        assert second_pen.style() == Qt.PenStyle.DashLine
        assert window._plot_style.background == "#ffffff"
    finally:
        window.close()
        app.processEvents()


def test_gui_result_plot_uses_dropdown_color_and_line_sequences():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.plot_curve_overrides.setText("")
        window.process_auto()

        color_index = window.plot_color_sequence.findText("Grayscale publication")
        line_index = window.plot_line_style_sequence.findText("Solid/dashed")
        assert color_index >= 0
        assert line_index >= 0
        window.plot_color_sequence.setCurrentIndex(color_index)
        window.plot_line_style_sequence.setCurrentIndex(line_index)
        app.processEvents()

        items = window.result_plot.getPlotItem().listDataItems()
        assert len(items) == 2
        first_pen = items[0].opts["pen"]
        second_pen = items[1].opts["pen"]
        assert first_pen.color().name().lower() == "#272727"
        assert first_pen.style() == Qt.PenStyle.SolidLine
        assert second_pen.color().name().lower() == "#4d4d4d"
        assert second_pen.style() == Qt.PenStyle.DashLine
    finally:
        window.close()
        app.processEvents()


def test_gui_default_line_width_updates_curve_pens():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.plot_curve_overrides.setText("")
        window.plot_default_width.setValue(3.2)
        window._update_plot_style_from_ui()
        window.process_auto()

        items = window.result_plot.getPlotItem().listDataItems()
        assert len(items) == 2
        assert items[0].opts["pen"].widthF() == pytest.approx(3.2)
        assert items[1].opts["pen"].widthF() == pytest.approx(3.2)
    finally:
        window.close()
        app.processEvents()


def test_gui_plots_raw_waveform_after_import_without_processing():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()

        raw_items = window.raw_plot.getPlotItem().listDataItems()
        segment_items = window.segment_plot.getPlotItem().listDataItems()

        assert len(raw_items) == 4
        assert len(segment_items) == 0
        assert window.bundle is None
    finally:
        window.close()
        app.processEvents()


def test_gui_preprocessing_preview_updates_without_full_processing():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.baseline_method.setCurrentText("none")
        window.filter_method.setCurrentText("none")
        window._plot_imported_raw_waveform()

        raw_only_items = window.raw_plot.getPlotItem().listDataItems()
        assert len(raw_only_items) == 2

        window.filter_method.setCurrentText("moving_average")
        window.filter_window.setValue(21)
        window._plot_imported_raw_waveform()

        preview_items = window.raw_plot.getPlotItem().listDataItems()
        segment_items = window.segment_plot.getPlotItem().listDataItems()

        assert len(preview_items) == 4
        assert len(segment_items) == 0
        assert window.bundle is None
    finally:
        window.close()
        app.processEvents()


def test_gui_preprocessing_change_refreshes_processed_result_without_button_click():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        window.process_auto()
        assert window.bundle is not None
        assert not any("Applied moving_average filter." in entry for entry in window.bundle.processed.preprocessing_log)
        before = window.bundle.processed.incident_strain.copy()

        window.filter_method.setCurrentText("moving_average")
        window.filter_window.setValue(21)
        window._recalculate_windows_live()

        assert any("Applied moving_average filter." in entry for entry in window.bundle.processed.preprocessing_log)
        after = window.bundle.processed.incident_strain
        assert not np.allclose(before, after)
        assert len(window.result_plot.getPlotItem().listDataItems()) == 2
    finally:
        window.close()
        app.processEvents()


def test_gui_specimen_geometry_modes_feed_workspace_config():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.specimen_shape.setCurrentText("rectangle")
        window.specimen_width.setValue(12.0)
        window.specimen_thickness.setValue(8.0)
        rectangle = window._workspace_config_from_ui("test", include_manual_windows=False).specimen

        assert rectangle.shape.value == "rectangle"
        assert rectangle.cross_section_area_m2 == pytest.approx(96e-6)
        assert window.specimen_width.isEnabled()
        assert window.specimen_thickness.isEnabled()
        assert not window.specimen_diameter.isEnabled()

        window.specimen_shape.setCurrentText("custom")
        window.specimen_area.setValue(42.0)
        custom = window._workspace_config_from_ui("test", include_manual_windows=False).specimen

        assert custom.shape.value == "custom"
        assert custom.cross_section_area_m2 == pytest.approx(42e-6)
        assert window.specimen_area.isEnabled()
        assert not window.specimen_width.isEnabled()
    finally:
        window.close()
        app.processEvents()


def test_gui_window_values_update_segment_preview_without_full_processing():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        for spin, value in zip(window.window_spins, [50.0, 130.0, 250.0, 330.0, 200.0, 280.0]):
            spin.setValue(value)
        window._plot_segment_preview_from_current_windows()

        segment_items = window.segment_plot.getPlotItem().listDataItems()
        finite_counts = [
            int(np.count_nonzero(np.isfinite(item.getData()[1])))
            for item in segment_items
        ]

        assert len(segment_items) == 3
        assert all(count > 2 for count in finite_counts)
        assert window.bundle is None
    finally:
        window.close()
        app.processEvents()


def test_gui_recalculate_rejects_empty_manual_windows_without_dialog():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.load_sample()
        for spin in window.window_spins:
            spin.setValue(0.0)

        assert window._rebuild_from_current_windows(show_errors=False, log_success=False) is False
        assert window.bundle is None
    finally:
        window.close()
        app.processEvents()


def test_gui_infers_seconds_and_unitless_strain_for_test6_workbook():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        path = Path("examples/test-6.xlsx")
        if not path.exists():
            pytest.skip("examples/test-6.xlsx is not included in the cleaned release package")
        dataframe = pd.read_excel(path)
        window.dataframe = dataframe
        window.source_path = str(path)
        window._populate_columns(list(dataframe.columns))
        window._apply_detected_columns()
        window._apply_inferred_acquisition_units()

        assert window.time_unit.currentText() == "s"
        assert window.strain_unit.currentText() == "strain"
        assert window.sampling_frequency.value() == pytest.approx(12_499_000.0, rel=1e-3)
    finally:
        window.close()
        app.processEvents()


def test_gui_aligned_plot_uses_separate_balance_error_axis_for_test6_workbook():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        path = Path("examples/test-6.xlsx")
        if not path.exists():
            pytest.skip("examples/test-6.xlsx is not included in the cleaned release package")
        dataframe = pd.read_excel(path)
        window.dataframe = dataframe
        window.source_path = str(path)
        window._populate_columns(list(dataframe.columns))
        window._apply_detected_columns()
        window._apply_inferred_acquisition_units()
        window.incident_diameter.setValue(16.0)
        window.transmitted_diameter.setValue(16.0)
        window.elastic_modulus.setValue(210.0)
        window.density.setValue(7800.0)
        window.incident_distance.setValue(0.0)
        window.transmitted_distance.setValue(0.0)
        window.specimen_diameter.setValue(3.0)
        window.specimen_length.setValue(10.0)
        window.experiment_type.setCurrentText(ExperimentType.COMPRESSION.value)
        window.sign_convention.setCurrentText(SignConvention.COMPRESSION_POSITIVE.value)

        window.process_auto()

        aligned_items = window.aligned_plot.getPlotItem().listDataItems()
        assert len(aligned_items) == 4
        assert window._aligned_balance_view is not None
        assert window._aligned_balance_curve is not None
        balance_x, balance_y = window._aligned_balance_curve.getData()
        assert len(balance_x) == len(window.bundle.aligned.time_s)
        assert np.nanmax(balance_y) > 1.0
        for item in aligned_items:
            _, y_values = item.getData()
            assert np.nanmax(np.abs(y_values)) < 1_000.0
    finally:
        window.close()
        app.processEvents()


def _legend_labels(plot) -> list[str]:
    legend = plot.getPlotItem().legend
    if legend is None:
        return []
    return [label.text for _, label in legend.items]
