from __future__ import annotations

from datetime import datetime
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
)

from shpb_processor.alignment_profiles import (
    delete_alignment_profile,
    load_alignment_profiles,
    save_alignment_profile,
)
from shpb_processor.batch import run_batch_from_config, safe_output_name, write_sample_outputs
from shpb_processor.calculation import CalculationSettings, compute_three_wave, compute_two_wave
from shpb_processor.config.workspace import (
    CurveStyle,
    PlotStyleSettings,
    WorkspaceConfig,
    default_curve_styles,
    load_workspace_config,
    save_workspace_config,
)
from shpb_processor.dispersion import DispersionSettings, correct_wave_segments
from shpb_processor.export import export_excel
from shpb_processor.gauge_distance import resolve_effective_gauge_distances
from shpb_processor.io import detect_columns, load_table
from shpb_processor.i18n import SUPPORTED_LANGUAGES, get_language, set_language, tr, tr_message
from shpb_processor.models import (
    AcquisitionParameters,
    BarParameters,
    ColumnMapping,
    ExperimentType,
    PulseWindow,
    SignConvention,
    SpecimenParameters,
    SpecimenShape,
    WaveSegments,
)
from shpb_processor.processing import ProcessingBundle, process_dataframe
from shpb_processor.quality import evaluate_quality
from shpb_processor.report import build_processing_report
from shpb_processor.sample_data import generate_synthetic_shpb_case
from shpb_processor.signal_processing import PreprocessingSettings, infer_sampling_frequency_from_column, process_signals
from shpb_processor.wave import (
    AlignmentSettings,
    FixedPulseWindows,
    PulseDetectionSettings,
    StandardWaveSeparationSettings,
    align_waves,
    detect_pulses,
)


COLOR_SEQUENCE_OPTIONS: list[tuple[str, str, list[str]]] = [
    (
        "nature_standard",
        "plot.color_sequence.nature_standard",
        ["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E", "#CFCECE"],
    ),
    (
        "nature_pastel",
        "plot.color_sequence.nature_pastel",
        ["#484878", "#7884B4", "#B4C0E4", "#E4E4F0", "#E4CCD8", "#F0C0CC"],
    ),
    (
        "okabe_ito",
        "plot.color_sequence.okabe_ito",
        ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#7F7F7F"],
    ),
    (
        "screen_bright",
        "plot.color_sequence.screen_bright",
        ["#00A7FF", "#FF3B30", "#34C759", "#FFD60A", "#BF5AF2", "#FFFFFF", "#FF9F0A", "#8E8E93"],
    ),
    (
        "grayscale",
        "plot.color_sequence.grayscale",
        ["#272727", "#4D4D4D", "#767676", "#A8A8A8", "#CFCECE", "#E5E5E5"],
    ),
]

LINE_STYLE_SEQUENCE_OPTIONS: list[tuple[str, str, list[str]]] = [
    ("origin_basic", "plot.line_sequence.origin_basic", ["solid", "dash", "dashdot", "dot"]),
    ("publication_solid", "plot.line_sequence.publication_solid", ["solid"]),
    ("solid_dash", "plot.line_sequence.solid_dash", ["solid", "dash"]),
    ("mixed_readable", "plot.line_sequence.mixed_readable", ["solid", "dash", "dot", "dashdot"]),
    ("black_white", "plot.line_sequence.black_white", ["solid", "dash", "dashdot", "dot", "longdash"]),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("app.title"))
        self.resize(1480, 920)
        self.dataframe: pd.DataFrame | None = None
        self.source_path = ""
        self.bundle: ProcessingBundle | None = None
        self.processed = None
        self.mapping: ColumnMapping | None = None
        self._aligned_balance_view: pg.ViewBox | None = None
        self._aligned_balance_curve: pg.PlotDataItem | None = None
        self._aligned_balance_resize_connected = False
        self._updating_sampling_frequency = False
        self._updating_wave_speed = False
        self._updating_window_spins = False
        self._updating_plot_style = False
        self._plot_style = self._plot_style_preset("dark_screen")
        self._live_window_recalc_running = False
        self._alignment_profile_path = Path(__file__).resolve().parents[3] / "alignment_profiles.json"
        self._raw_waveform_preview_timer = QTimer(self)
        self._raw_waveform_preview_timer.setSingleShot(True)
        self._raw_waveform_preview_timer.setInterval(160)
        self._raw_waveform_preview_timer.timeout.connect(self._plot_imported_raw_waveform)
        self._live_window_recalc_timer = QTimer(self)
        self._live_window_recalc_timer.setSingleShot(True)
        self._live_window_recalc_timer.setInterval(180)
        self._live_window_recalc_timer.timeout.connect(self._recalculate_windows_live)

        pg.setConfigOptions(antialias=True)
        self._language_actions: dict[str, QAction] = {}
        self._language_menu = None
        self._build_language_menu()
        self._build_ui()
        self._set_plot_style_controls(self._plot_style)
        self._connect_live_updates()
        self._update_optional_feature_ui()
        self._apply_plot_style_to_all_plots()
        self._update_wave_speed_from_bar_inputs()
        self._update_voltage_conversion_enabled()

    def _build_language_menu(self) -> None:
        language_menu = self.menuBar().addMenu(tr("menu.language"))
        self._language_menu = language_menu
        group = QActionGroup(self)
        group.setExclusive(True)
        for code, display_name in SUPPORTED_LANGUAGES.items():
            action = QAction(display_name, self)
            action.setCheckable(True)
            action.setChecked(code == get_language())
            action.triggered.connect(lambda _checked=False, language=code: self._change_language(language))
            group.addAction(action)
            language_menu.addAction(action)
            self._language_actions[code] = action

    def _change_language(self, language: str) -> None:
        set_language(language)
        self._refresh_language_menu_labels()
        for code, action in self._language_actions.items():
            action.setChecked(code == language)
        if hasattr(self, "log"):
            self._log(tr("log.language_saved", language=SUPPORTED_LANGUAGES.get(language, language)))
        QMessageBox.information(
            self,
            tr("dialog.language_saved.title"),
            tr("dialog.language_saved.message", language=SUPPORTED_LANGUAGES.get(language, language)),
        )

    def _refresh_language_menu_labels(self) -> None:
        if self._language_menu is not None:
            self._language_menu.setTitle(tr("menu.language"))
        for code, action in self._language_actions.items():
            action.setText(SUPPORTED_LANGUAGES.get(code, code))

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([360, 820, 300])
        root_layout.addWidget(splitter)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        root_layout.addWidget(self.log)
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        file_group = QGroupBox(tr("group.import_data"))
        file_layout = QVBoxLayout(file_group)
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path)
        row = QHBoxLayout()
        import_button = QPushButton(tr("button.import_data"))
        import_button.clicked.connect(self.load_file)
        sample_button = QPushButton(tr("button.load_sample"))
        sample_button.clicked.connect(self.load_sample)
        row.addWidget(import_button)
        row.addWidget(sample_button)
        file_layout.addLayout(row)
        layout.addWidget(file_group)

        column_group = QGroupBox(tr("group.columns"))
        column_layout = QFormLayout(column_group)
        self.time_column = QComboBox()
        self.incident_column = QComboBox()
        self.transmitted_column = QComboBox()
        column_layout.addRow(tr("label.time_column"), self.time_column)
        column_layout.addRow(tr("label.incident_column"), self.incident_column)
        column_layout.addRow(tr("label.transmitted_column"), self.transmitted_column)
        layout.addWidget(column_group)

        acquisition_group = QGroupBox(tr("group.acquisition"))
        acquisition_layout = QFormLayout(acquisition_group)
        self.sampling_frequency = _spin(1.0, 100_000_000.0, 5_000_000.0, 0)
        self.time_unit = _combo(["s", "ms", "μs"], "μs")
        self.strain_unit = _combo(["strain", "με", "voltage"], "με")
        self.voltage_to_microstrain = _spin(1e-9, 1_000_000_000.0, 1000.0, 6)
        self.voltage_to_microstrain.setSingleStep(100.0)
        self.voltage_to_microstrain.setSuffix(" με/V")
        self.voltage_to_microstrain.setToolTip(tr("tooltip.voltage_to_microstrain"))
        self.baseline_method = _combo(["start_mean", "window_mean", "linear_drift", "polynomial", "manual", "none"], "start_mean")
        self.filter_method = _combo(["none", "moving_average", "savgol", "butterworth", "median"], "none")
        self.filter_window = _int_spin(1, 999, 11)
        self.filter_cutoff = _spin(1.0, 10_000_000.0, 100_000.0, 0)
        self.dispersion_enabled = QCheckBox(tr("check.dispersion_enabled"))
        self.dispersion_enabled.setChecked(False)
        self.standard_wave_enabled = QCheckBox(tr("check.standard_wave_enabled"))
        self.standard_wave_enabled.setChecked(False)
        self.auto_micro_align = QCheckBox(tr("check.auto_micro_align"))
        self.auto_micro_align.setChecked(True)
        self.alignment_objective = _combo(["force_balance", "hybrid", "wave_relation"], "force_balance")
        acquisition_layout.addRow(tr("label.sampling_frequency_hz"), self.sampling_frequency)
        acquisition_layout.addRow(tr("label.time_unit"), self.time_unit)
        acquisition_layout.addRow(tr("label.strain_unit"), self.strain_unit)
        acquisition_layout.addRow(tr("label.voltage_conversion"), self.voltage_to_microstrain)
        acquisition_layout.addRow(tr("label.baseline_method"), self.baseline_method)
        acquisition_layout.addRow(tr("label.filter_method"), self.filter_method)
        acquisition_layout.addRow(tr("label.filter_window_points"), self.filter_window)
        acquisition_layout.addRow(tr("label.filter_cutoff_hz"), self.filter_cutoff)
        acquisition_layout.addRow(tr("label.dispersion_correction"), self.dispersion_enabled)
        acquisition_layout.addRow(tr("label.wave_separation"), self.standard_wave_enabled)
        acquisition_layout.addRow(tr("label.auto_alignment"), self.auto_micro_align)
        acquisition_layout.addRow(tr("label.alignment_objective"), self.alignment_objective)
        layout.addWidget(acquisition_group)

        bar_group = QGroupBox(tr("group.bar_parameters"))
        bar_layout = QFormLayout(bar_group)
        self.incident_diameter = _spin(0.1, 200.0, 14.5, 3)
        self.transmitted_diameter = _spin(0.1, 200.0, 14.5, 3)
        self.elastic_modulus = _spin(0.1, 1000.0, 200.0, 3)
        self.density = _spin(100.0, 30000.0, 7800.0, 1)
        self.wave_speed = _spin(0.0, 20000.0, 5063.7, 2)
        self.wave_speed.setReadOnly(True)
        self.wave_speed.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.wave_speed.setToolTip(tr("tooltip.wave_speed"))
        self.incident_distance = _spin(0.0, 10000.0, 500.0, 3)
        self.transmitted_distance = _spin(0.0, 10000.0, 250.0, 3)
        self.transmitted_free_end_distance = _spin(0.0, 10000.0, 750.0, 3)
        bar_layout.addRow(tr("label.incident_diameter_mm"), self.incident_diameter)
        bar_layout.addRow(tr("label.transmitted_diameter_mm"), self.transmitted_diameter)
        bar_layout.addRow(tr("label.elastic_modulus_gpa"), self.elastic_modulus)
        bar_layout.addRow(tr("label.density_kg_m3"), self.density)
        bar_layout.addRow(tr("label.wave_speed_m_s"), self.wave_speed)
        bar_layout.addRow(tr("label.incident_gauge_distance_mm"), self.incident_distance)
        bar_layout.addRow(tr("label.transmitted_gauge_distance_mm"), self.transmitted_distance)
        bar_layout.addRow(tr("label.transmitted_free_end_distance_mm"), self.transmitted_free_end_distance)
        layout.addWidget(bar_group)

        specimen_group = QGroupBox(tr("group.specimen_sign"))
        specimen_layout = QFormLayout(specimen_group)
        self.specimen_id = QLineEdit("sample_001")
        self.specimen_shape = _combo(["cylinder", "rectangle", "custom"], "cylinder")
        self.specimen_diameter = _spin(0.01, 200.0, 10.0, 3)
        self.specimen_width = _spin(0.01, 200.0, 10.0, 3)
        self.specimen_thickness = _spin(0.01, 200.0, 10.0, 3)
        self.specimen_area = _spin(0.000001, 1_000_000.0, 78.539816, 6)
        self.specimen_area.setSuffix(" mm2")
        self.specimen_length = _spin(0.01, 200.0, 8.0, 3)
        self.experiment_type = _combo(["compression", "tension"], "compression")
        self.sign_convention = _combo(["compression_positive", "tension_positive", "raw"], "compression_positive")
        specimen_layout.addRow(tr("label.specimen_id"), self.specimen_id)
        specimen_layout.addRow(tr("label.specimen_shape"), self.specimen_shape)
        specimen_layout.addRow(tr("label.specimen_diameter_mm"), self.specimen_diameter)
        specimen_layout.addRow(tr("label.section_length_mm"), self.specimen_width)
        specimen_layout.addRow(tr("label.section_width_mm"), self.specimen_thickness)
        specimen_layout.addRow(tr("label.section_area_mm2"), self.specimen_area)
        specimen_layout.addRow(tr("label.initial_height_gauge_mm"), self.specimen_length)
        specimen_layout.addRow(tr("label.experiment_type"), self.experiment_type)
        specimen_layout.addRow(tr("label.sign_convention"), self.sign_convention)
        self._update_specimen_geometry_enabled()
        layout.addWidget(specimen_group)

        style_group = QGroupBox(tr("group.plot_style"))
        style_layout = QFormLayout(style_group)
        self.plot_style_preset = _combo(["dark_screen", "light_report", "nature_style", "high_contrast"], "dark_screen")
        self.plot_background = _combo(["#000000", "#ffffff", "#f5f5f5"], "#000000")
        self.plot_grid_enabled = QCheckBox(tr("check.show_grid"))
        self.plot_grid_enabled.setChecked(True)
        self.plot_default_width = _spin(0.1, 10.0, 1.5, 2)
        self.plot_default_width.setSingleStep(0.1)
        self.plot_color_sequence = QComboBox()
        self.plot_line_style_sequence = QComboBox()
        self.plot_color_sequence.setMinimumWidth(260)
        self.plot_line_style_sequence.setMinimumWidth(260)
        self._populate_color_sequence_combo()
        self._populate_line_style_sequence_combo()
        self.plot_curve_overrides = QLineEdit()
        self.plot_curve_overrides.setPlaceholderText("result.standard_wave=#222222,solid,2.2; standard.gauge1_right=#444444,dash")
        style_layout.addRow(tr("label.style_preset"), self.plot_style_preset)
        style_layout.addRow(tr("label.background_color"), self.plot_background)
        style_layout.addRow(tr("label.grid"), self.plot_grid_enabled)
        style_layout.addRow(tr("label.default_line_width"), self.plot_default_width)
        style_layout.addRow(tr("label.color_sequence"), self.plot_color_sequence)
        style_layout.addRow(tr("label.line_style_sequence"), self.plot_line_style_sequence)
        style_layout.addRow(tr("label.curve_overrides"), self.plot_curve_overrides)
        layout.addWidget(style_group)

        action_row = QHBoxLayout()
        process_button = QPushButton(tr("button.process_auto"))
        process_button.clicked.connect(self.process_auto)
        recalc_button = QPushButton(tr("button.recalculate_windows"))
        recalc_button.clicked.connect(self.recalculate_with_windows)
        action_row.addWidget(process_button)
        action_row.addWidget(recalc_button)
        layout.addLayout(action_row)

        export_row = QHBoxLayout()
        export_button = QPushButton(tr("button.export_excel"))
        export_button.clicked.connect(self.export_results)
        report_button = QPushButton(tr("button.generate_report"))
        report_button.clicked.connect(self.generate_report_package)
        export_row.addWidget(export_button)
        export_row.addWidget(report_button)
        layout.addLayout(export_row)

        template_row = QHBoxLayout()
        template_export_button = QPushButton(tr("button.export_batch_template"))
        template_export_button.clicked.connect(self.export_batch_template)
        template_load_button = QPushButton(tr("button.load_batch_template"))
        template_load_button.clicked.connect(self.load_batch_template)
        template_row.addWidget(template_export_button)
        template_row.addWidget(template_load_button)
        layout.addLayout(template_row)

        batch_button = QPushButton(tr("button.run_batch_current"))
        batch_button.clicked.connect(self.run_batch_with_current_template)
        layout.addWidget(batch_button)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_center_panel(self) -> QWidget:
        tabs = QTabWidget()
        self.center_tabs = tabs
        self.preview_table = QTableWidget()
        tabs.addTab(self.preview_table, tr("tab.data_preview"))
        self.raw_plot = pg.PlotWidget(title=tr("plot.raw_preprocessed"))
        self.segment_plot = pg.PlotWidget(title=tr("plot.segment"))
        self.dispersion_plot = pg.PlotWidget(title=tr("plot.dispersion_comparison"))
        self.aligned_plot = pg.PlotWidget(title=tr("plot.aligned"))
        self.standard_wave_plot = pg.PlotWidget(title=tr("plot.standard_wave"))
        self.result_plot = pg.PlotWidget(title=tr("plot.result"))
        for title, plot in [
            (tr("tab.raw_waveform"), self.raw_plot),
            (tr("tab.segment"), self.segment_plot),
            (tr("tab.dispersion_comparison"), self.dispersion_plot),
            (tr("tab.aligned"), self.aligned_plot),
            (tr("tab.standard_wave"), self.standard_wave_plot),
            (tr("tab.results"), self.result_plot),
        ]:
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.addLegend(offset=(12, 12), brush=(0, 0, 0, 130), labelTextColor="w")
            tabs.addTab(plot, title)
        return tabs

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        window_group = QGroupBox(tr("group.wave_windows_us"))
        grid = QGridLayout(window_group)
        labels = [
            tr("label.incident_start"),
            tr("label.incident_end"),
            tr("label.reflected_start"),
            tr("label.reflected_end"),
            tr("label.transmitted_start"),
            tr("label.transmitted_end"),
        ]
        self.window_spins: list[QDoubleSpinBox] = []
        for index, label in enumerate(labels):
            spin = _spin(0.0, 1_000_000.0, 0.0, 3)
            self.window_spins.append(spin)
            grid.addWidget(QLabel(label), index, 0)
            grid.addWidget(spin, index, 1)
        layout.addWidget(window_group)

        profile_group = QGroupBox(tr("group.alignment_profiles"))
        profile_layout = QGridLayout(profile_group)
        self.alignment_profile_name = QLineEdit("default")
        self.alignment_profile_combo = QComboBox()
        self.save_alignment_profile_button = QPushButton(tr("button.save"))
        self.load_alignment_profile_button = QPushButton(tr("button.load"))
        self.delete_alignment_profile_button = QPushButton(tr("button.delete"))
        profile_layout.addWidget(QLabel(tr("label.name")), 0, 0)
        profile_layout.addWidget(self.alignment_profile_name, 0, 1, 1, 2)
        profile_layout.addWidget(QLabel(tr("label.existing")), 1, 0)
        profile_layout.addWidget(self.alignment_profile_combo, 1, 1, 1, 2)
        profile_layout.addWidget(self.save_alignment_profile_button, 2, 0)
        profile_layout.addWidget(self.load_alignment_profile_button, 2, 1)
        profile_layout.addWidget(self.delete_alignment_profile_button, 2, 2)
        layout.addWidget(profile_group)
        self._refresh_alignment_profile_list()

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(QLabel(tr("label.status_summary")))
        layout.addWidget(self.summary)
        return panel

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("dialog.import_data.title"), "", "Data files (*.csv *.txt *.dat *.xlsx *.xls)")
        if not path:
            return
        try:
            result = load_table(path)
            self.dataframe = result.dataframe
            self.source_path = path
            self.file_path.setText(path)
            self._reset_processing_state_for_new_data()
            self._populate_columns(result.columns)
            self._apply_detected_columns()
            self._apply_inferred_acquisition_units()
            self._update_sampling_frequency_from_time_column()
            self._preview_dataframe()
            self._plot_imported_raw_waveform()
            self._log(tr("log.file_imported", file=Path(path).name, rows=len(result.dataframe), columns=len(result.columns)))
        except Exception as exc:
            self._error(str(exc))

    def load_sample(self) -> None:
        dataframe, metadata = generate_synthetic_shpb_case("ideal")
        self.dataframe = dataframe
        self.source_path = "synthetic_shpb_ideal"
        self.file_path.setText(self.source_path)
        self._reset_processing_state_for_new_data()
        self._populate_columns(list(dataframe.columns))
        self._apply_detected_columns()
        self._preview_dataframe()
        self.sampling_frequency.setValue(float(metadata["sampling_frequency_hz"]))
        self._set_combo_text(self.time_unit, self._display_time_unit(str(metadata["time_unit"])))
        self._update_sampling_frequency_from_time_column()
        self._set_combo_text(self.strain_unit, self._display_strain_unit(str(metadata["strain_unit"])))
        self._update_voltage_conversion_enabled()
        self.incident_diameter.setValue(float(metadata["incident_diameter_m"]) * 1e3)
        self.transmitted_diameter.setValue(float(metadata["transmitted_diameter_m"]) * 1e3)
        self.elastic_modulus.setValue(float(metadata["elastic_modulus_pa"]) / 1e9)
        self.density.setValue(float(metadata["density_kg_m3"]))
        self._update_wave_speed_from_bar_inputs()
        self.incident_distance.setValue(float(metadata["incident_gauge_distance_m"]) * 1e3)
        self.transmitted_distance.setValue(float(metadata["transmitted_gauge_distance_m"]) * 1e3)
        self._set_combo_text(self.specimen_shape, SpecimenShape.CYLINDER.value)
        self.specimen_diameter.setValue(float(metadata["specimen_diameter_m"]) * 1e3)
        self.specimen_area.setValue(float(metadata["specimen_diameter_m"]) ** 2 * np.pi / 4.0 * 1e6)
        self.specimen_length.setValue(float(metadata["specimen_length_m"]) * 1e3)
        self._update_specimen_geometry_enabled()
        self._plot_imported_raw_waveform()
        self._log(tr("log.sample_loaded"))

    def process_auto(self) -> None:
        if self.dataframe is None:
            self._error(tr("error.import_first"))
            return
        try:
            self.mapping = self._mapping_from_ui()
            bundle = process_dataframe(
                self.dataframe,
                self.source_path,
                self._acquisition_from_ui(),
                self._bar_from_ui(),
                self._specimen_from_ui(),
                mapping=self.mapping,
                preprocessing=self._preprocessing_from_ui(),
                alignment=self._alignment_from_ui(),
                calculation=self._calculation_from_ui(),
                dispersion=self._dispersion_from_ui(),
                wave_separation=self._standard_wave_separation_from_ui(),
                sign_convention=self._sign_from_ui(),
            )
            self.bundle = bundle
            self.processed = bundle.processed
            self._set_window_spins(bundle.segments)
            self._plot_all()
            self._update_summary()
            self._log(tr("log.auto_process_done", grade=bundle.quality.grade))
        except Exception as exc:
            self._error(str(exc))

    def recalculate_with_windows(self) -> None:
        self._rebuild_from_current_windows(show_errors=True, log_success=True)

    def _rebuild_from_current_windows(self, show_errors: bool, log_success: bool) -> bool:
        if self.dataframe is None:
            if show_errors:
                self._error(tr("error.import_first"))
            return False
        try:
            mapping = self._mapping_from_ui()
            self.mapping = mapping
            processed = process_signals(
                self.dataframe,
                mapping,
                self._acquisition_from_ui(),
                self._preprocessing_from_ui(),
            )
            windows = self._windows_from_ui()
            uncorrected_segments = self._segments_from_windows(
                processed.time_s,
                processed.incident_strain,
                processed.transmitted_strain,
                windows,
            )
            self._plot_segments(uncorrected_segments, title=tr("plot.segment_preview"))
            self._validate_segments_for_alignment(uncorrected_segments)
            bar = self._bar_from_ui()
            specimen = self._specimen_from_ui()
            dispersion_settings = self._dispersion_from_ui()
            alignment_settings = self._alignment_from_ui()
            segments_for_dispersion, effective_bar = resolve_effective_gauge_distances(
                uncorrected_segments,
                bar,
                dispersion_settings,
                alignment_settings,
            )
            segments = correct_wave_segments(segments_for_dispersion, effective_bar, dispersion_settings)
            aligned = align_waves(segments, effective_bar, alignment_settings)
            three_wave = compute_three_wave(aligned, effective_bar, specimen, self._sign_from_ui(), self._calculation_from_ui())
            two_wave = compute_two_wave(aligned, effective_bar, specimen, self._sign_from_ui(), self._calculation_from_ui())
            standard_wave = None
            wave_separation_settings = self._standard_wave_separation_from_ui()
            if wave_separation_settings.enabled:
                from shpb_processor.wave import separate_standard_shpb

                standard_wave = separate_standard_shpb(
                    processed,
                    effective_bar,
                    specimen,
                    wave_separation_settings,
                    self._sign_from_ui(),
                )
            quality = evaluate_quality(processed, segments, aligned, specimen=specimen)
            report_rows = build_processing_report(
                self.source_path,
                bar,
                specimen,
                self._acquisition_from_ui(),
                mapping,
                segments,
                aligned,
                three_wave,
                two_wave,
                quality,
                self._sign_from_ui(),
                standard_wave_separation=standard_wave,
            )
            self.bundle = ProcessingBundle(
                raw=self.bundle.raw if self.bundle else process_dataframe(
                    self.dataframe,
                    self.source_path,
                    self._acquisition_from_ui(),
                    bar,
                    specimen,
                    mapping,
                    alignment=self._alignment_from_ui(),
                    calculation=self._calculation_from_ui(),
                    dispersion=self._dispersion_from_ui(),
                    wave_separation=self._standard_wave_separation_from_ui(),
                ).raw,
                processed=processed,
                uncorrected_segments=uncorrected_segments if dispersion_settings.enabled else None,
                segments=segments,
                aligned=aligned,
                three_wave=three_wave,
                two_wave=two_wave,
                quality=quality,
                standard_wave_separation=standard_wave,
                report_rows=report_rows,
            )
            self.processed = processed
            self._plot_all()
            self._update_summary()
            if log_success:
                self._log(tr("log.manual_recalculated", grade=quality.grade))
            return True
        except Exception as exc:
            if show_errors:
                self._error(self._friendly_recalculate_error(exc))
            return False

    def _schedule_window_recalculate(self, *_args) -> None:
        if self._updating_window_spins or self._live_window_recalc_running:
            return
        if self.dataframe is None:
            return
        self._plot_segment_preview_from_current_windows()
        if self.bundle is None:
            return
        self._live_window_recalc_timer.start()

    def _recalculate_windows_live(self) -> None:
        if self._updating_window_spins or self._live_window_recalc_running:
            return
        self._live_window_recalc_running = True
        try:
            if not self._rebuild_from_current_windows(show_errors=False, log_success=False):
                self._plot_segment_preview_from_current_windows()
        finally:
            self._live_window_recalc_running = False

    def export_results(self) -> None:
        if self.bundle is None:
            self._error(tr("error.process_first"))
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("dialog.export_excel.title"), "SHPB_SHTB_Result.xlsx", "Excel workbook (*.xlsx)")
        if not path:
            return
        try:
            output = export_excel(
                path,
                raw=self.bundle.raw,
                processed=self.bundle.processed,
                segments=self.bundle.segments,
                aligned=self.bundle.aligned,
                three_wave=self.bundle.three_wave,
                two_wave=self.bundle.two_wave,
                standard_wave_separation=self.bundle.standard_wave_separation,
                quality=self.bundle.quality,
                report_rows=self.bundle.report_rows,
            )
            self._log(tr("log.exported", output=output))
        except Exception as exc:
            self._error(str(exc))

    def generate_report_package(self) -> None:
        if self.bundle is None:
            self._error(tr("error.process_first"))
            return

        base_dir = QFileDialog.getExistingDirectory(self, tr("dialog.select_report_dir.title"))
        if not base_dir:
            return

        default_name = self._default_report_folder_name()
        folder_name, ok = QInputDialog.getText(
            self,
            tr("dialog.report_folder_name.title"),
            tr("dialog.report_folder_name.message"),
            text=default_name,
        )
        if not ok:
            return
        folder_name = safe_output_name(folder_name)
        if not folder_name:
            self._error(tr("error.report_folder_empty"))
            return

        output_dir = Path(base_dir) / folder_name
        if output_dir.exists() and any(output_dir.iterdir()):
            reply = QMessageBox.question(
                self,
                tr("dialog.overwrite.title"),
                tr("dialog.overwrite_report.message", output_dir=output_dir),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            config = self._workspace_config_from_ui(
                report_title=tr("report.title.single", name=folder_name),
            )
            output = write_sample_outputs(output_dir, self.bundle, config)
            self._log(tr("log.report_generated", output=output))
            QMessageBox.information(
                self,
                tr("dialog.report_done.title"),
                tr("dialog.report_done.message", output=output),
            )
        except Exception as exc:
            self._error(str(exc))

    def export_batch_template(self) -> None:
        if self.dataframe is None:
            self._error(tr("error.need_representative_data"))
            return
        if self._fixed_windows_from_ui(require_valid=True) is None:
            return

        default_name = f"{safe_output_name(Path(self.source_path).stem or self.specimen_id.text())}_batch_config.json"
        path, _ = QFileDialog.getSaveFileName(self, tr("dialog.export_batch_template.title"), default_name, "Config files (*.json *.yaml *.yml)")
        if not path:
            return
        try:
            config = self._workspace_config_from_ui(
                report_title=tr("report.title.batch"),
                include_manual_windows=True,
            )
            output = save_workspace_config(config, path)
            self._log(tr("log.batch_template_exported", output=output))
            QMessageBox.information(
                self,
                tr("dialog.batch_template_exported.title"),
                tr("dialog.batch_template_exported.message", output=output),
            )
        except Exception as exc:
            self._error(str(exc))

    def load_batch_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("dialog.load_batch_template.title"), "", "Config files (*.json *.yaml *.yml)")
        if not path:
            return
        try:
            config = load_workspace_config(path)
            self._apply_workspace_config(config)
            self._log(tr("log.batch_template_loaded", path=path))
        except Exception as exc:
            self._error(str(exc))

    def run_batch_with_current_template(self) -> None:
        if self.dataframe is None:
            self._error(tr("error.need_representative_data"))
            return
        if self._fixed_windows_from_ui(require_valid=True) is None:
            return

        input_dir = QFileDialog.getExistingDirectory(self, tr("dialog.select_batch_input_dir.title"))
        if not input_dir:
            return
        base_dir = QFileDialog.getExistingDirectory(self, tr("dialog.select_batch_output_dir.title"))
        if not base_dir:
            return
        default_name = f"{safe_output_name(Path(input_dir).name)}_batch_results"
        folder_name, ok = QInputDialog.getText(
            self,
            tr("dialog.batch_results_folder_name.title"),
            tr("dialog.batch_results_folder_name.message"),
            text=default_name,
        )
        if not ok:
            return
        folder_name = safe_output_name(folder_name)
        if not folder_name:
            self._error(tr("error.batch_folder_empty"))
            return

        output_dir = Path(base_dir) / folder_name
        if output_dir.exists() and any(output_dir.iterdir()):
            reply = QMessageBox.question(
                self,
                tr("dialog.overwrite.title"),
                tr("dialog.overwrite_batch.message", output_dir=output_dir),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            config = self._workspace_config_from_ui(
                report_title=tr("report.title.batch_named", name=folder_name),
                include_manual_windows=True,
            )
            output = run_batch_from_config(config, input_dir, output_dir)
            self._log(tr("log.batch_done", output=output))
            QMessageBox.information(
                self,
                tr("dialog.batch_done.title"),
                tr("dialog.batch_done.message", output=output),
            )
        except Exception as exc:
            self._error(str(exc))

    def _save_alignment_profile_from_ui(self) -> None:
        name = self.alignment_profile_name.text().strip() or self.alignment_profile_combo.currentText().strip()
        try:
            save_alignment_profile(self._alignment_profile_path, name, self._alignment_profile_payload())
            self._refresh_alignment_profile_list(selected_name=name)
            self._log(tr("log.profile_saved", name=name))
        except Exception as exc:
            self._error(str(exc))

    def _load_selected_alignment_profile(self) -> None:
        name = self.alignment_profile_combo.currentText().strip()
        if not name:
            self._error(tr("error.select_profile_to_load"))
            return
        try:
            profile = load_alignment_profiles(self._alignment_profile_path).get(name)
            if profile is None:
                self._error(tr("error.profile_not_found", name=name))
                return
            self.alignment_profile_name.setText(name)
            self._apply_alignment_profile(profile)
            self._log(tr("log.profile_loaded", name=name))
        except Exception as exc:
            self._error(str(exc))

    def _delete_selected_alignment_profile(self) -> None:
        name = self.alignment_profile_combo.currentText().strip()
        if not name:
            return
        reply = QMessageBox.question(
            self,
            tr("dialog.delete_profile.title"),
            tr("dialog.delete_profile.message", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_alignment_profile(self._alignment_profile_path, name)
            self._refresh_alignment_profile_list()
            self._log(tr("log.profile_deleted", name=name))
        except Exception as exc:
            self._error(str(exc))

    def _refresh_alignment_profile_list(self, selected_name: str | None = None) -> None:
        current = selected_name or self.alignment_profile_combo.currentText()
        self.alignment_profile_combo.clear()
        try:
            names = sorted(load_alignment_profiles(self._alignment_profile_path))
        except Exception as exc:
            if hasattr(self, "log"):
                self._log(tr("log.profile_read_failed", error=exc))
            names = []
        self.alignment_profile_combo.addItems(names)
        if current in names:
            self.alignment_profile_combo.setCurrentText(current)
        has_profiles = bool(names)
        self.load_alignment_profile_button.setEnabled(has_profiles)
        self.delete_alignment_profile_button.setEnabled(has_profiles)

    def _alignment_profile_payload(self) -> dict[str, object]:
        window_keys = [
            "incident_start_us",
            "incident_end_us",
            "reflected_start_us",
            "reflected_end_us",
            "transmitted_start_us",
            "transmitted_end_us",
        ]
        payload: dict[str, object] = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "windows_us": {key: spin.value() for key, spin in zip(window_keys, self.window_spins)},
            "alignment": {
                "auto_micro_adjust": self.auto_micro_align.isChecked(),
                "alignment_objective": self.alignment_objective.currentText(),
            },
            "dispersion": {
                "enabled": self.dispersion_enabled.isChecked(),
            },
            "acquisition": {
                "time_unit": self.time_unit.currentText(),
                "strain_unit": self.strain_unit.currentText(),
                "voltage_to_microstrain_per_volt": self.voltage_to_microstrain.value()
                if self.strain_unit.currentText() == "voltage"
                else None,
            },
            "bar": {
                "incident_diameter_mm": self.incident_diameter.value(),
                "transmitted_diameter_mm": self.transmitted_diameter.value(),
                "elastic_modulus_gpa": self.elastic_modulus.value(),
                "density_kg_m3": self.density.value(),
                "incident_gauge_distance_mm": self.incident_distance.value(),
                "transmitted_gauge_distance_mm": self.transmitted_distance.value(),
                "wave_speed_m_s": self.wave_speed.value(),
            },
            "specimen": {
                "shape": self.specimen_shape.currentText(),
                "diameter_mm": self.specimen_diameter.value(),
                "cross_section_length_mm": self.specimen_width.value(),
                "cross_section_width_mm": self.specimen_thickness.value(),
                "area_mm2": self.specimen_area.value(),
                "length_mm": self.specimen_length.value(),
                "experiment_type": self.experiment_type.currentText(),
            },
        }
        if self.bundle is not None:
            payload["last_alignment_result"] = {
                "incident_shift_us": self.bundle.aligned.incident_shift_s * 1e6,
                "reflected_shift_us": self.bundle.aligned.reflected_shift_s * 1e6,
                "transmitted_shift_us": self.bundle.aligned.transmitted_shift_s * 1e6,
                "force_balance_error": self.bundle.aligned.metadata.get("auto_alignment_final_force_balance_error"),
                "status": self.bundle.aligned.metadata.get("auto_alignment_status"),
            }
        return payload

    def _apply_alignment_profile(self, profile: dict[str, object]) -> None:
        window_keys = [
            "incident_start_us",
            "incident_end_us",
            "reflected_start_us",
            "reflected_end_us",
            "transmitted_start_us",
            "transmitted_end_us",
        ]
        self._updating_window_spins = True
        try:
            windows = profile.get("windows_us", {})
            if isinstance(windows, dict):
                for key, spin in zip(window_keys, self.window_spins):
                    if key in windows:
                        spin.setValue(float(windows[key]))

            alignment = profile.get("alignment", {})
            if isinstance(alignment, dict):
                if "auto_micro_adjust" in alignment:
                    self.auto_micro_align.setChecked(bool(alignment["auto_micro_adjust"]))
                objective = alignment.get("alignment_objective")
                if isinstance(objective, str) and self.alignment_objective.findText(objective) >= 0:
                    self.alignment_objective.setCurrentText(objective)

            dispersion = profile.get("dispersion", {})
            if isinstance(dispersion, dict) and "enabled" in dispersion:
                self.dispersion_enabled.setChecked(bool(dispersion["enabled"]))

            acquisition = profile.get("acquisition", {})
            if isinstance(acquisition, dict):
                time_unit = acquisition.get("time_unit")
                if isinstance(time_unit, str):
                    self._set_combo_text(self.time_unit, self._display_time_unit(time_unit))
                strain_unit = acquisition.get("strain_unit")
                if isinstance(strain_unit, str):
                    self._set_combo_text(self.strain_unit, self._display_strain_unit(strain_unit))
                coefficient = acquisition.get("voltage_to_microstrain_per_volt")
                if coefficient is not None:
                    self.voltage_to_microstrain.setValue(float(coefficient))

            bar = profile.get("bar", {})
            if isinstance(bar, dict):
                self._set_spin_if_present(self.incident_diameter, bar, "incident_diameter_mm")
                self._set_spin_if_present(self.transmitted_diameter, bar, "transmitted_diameter_mm")
                self._set_spin_if_present(self.elastic_modulus, bar, "elastic_modulus_gpa")
                self._set_spin_if_present(self.density, bar, "density_kg_m3")
                self._set_spin_if_present(self.incident_distance, bar, "incident_gauge_distance_mm")
                self._set_spin_if_present(self.transmitted_distance, bar, "transmitted_gauge_distance_mm")
                self._update_wave_speed_from_bar_inputs()

            specimen = profile.get("specimen", {})
            if isinstance(specimen, dict):
                shape = specimen.get("shape")
                if isinstance(shape, str):
                    self._set_combo_text(self.specimen_shape, shape)
                self._set_spin_if_present(self.specimen_diameter, specimen, "diameter_mm")
                self._set_spin_if_present(self.specimen_width, specimen, "cross_section_length_mm")
                self._set_spin_if_present(self.specimen_thickness, specimen, "cross_section_width_mm")
                self._set_spin_if_present(self.specimen_area, specimen, "area_mm2")
                self._set_spin_if_present(self.specimen_length, specimen, "length_mm")
                experiment_type = specimen.get("experiment_type")
                if isinstance(experiment_type, str) and self.experiment_type.findText(experiment_type) >= 0:
                    self.experiment_type.setCurrentText(experiment_type)
                self._update_specimen_geometry_enabled()
        finally:
            self._updating_window_spins = False

        self._update_voltage_conversion_enabled()
        if self.dataframe is not None and self.processed is not None:
            self._rebuild_from_current_windows(show_errors=False, log_success=False)

    def _populate_columns(self, columns: list[str]) -> None:
        for combo in [self.time_column, self.incident_column, self.transmitted_column]:
            combo.clear()
        self.time_column.addItem(self._auto_time_column_text())
        self.time_column.addItems(columns)
        self.incident_column.addItems(columns)
        self.transmitted_column.addItems(columns)

    def _apply_detected_columns(self) -> None:
        if self.dataframe is None:
            return
        detected = detect_columns(self.dataframe, self.sampling_frequency.value())
        if detected.time_column:
            self.time_column.setCurrentText(detected.time_column)
        if detected.incident_column:
            self.incident_column.setCurrentText(detected.incident_column)
        if detected.transmitted_column:
            self.transmitted_column.setCurrentText(detected.transmitted_column)
        self._log(
            tr(
                "log.columns_detected",
                time=detected.time_column,
                incident=detected.incident_column,
                transmitted=detected.transmitted_column,
            )
        )

    def _apply_inferred_acquisition_units(self) -> None:
        if self.dataframe is None:
            return
        inferred_time = self._infer_time_unit_from_selected_column()
        if inferred_time is not None:
            display_time = self._display_time_unit(inferred_time)
            if self.time_unit.currentText() != display_time:
                self._set_combo_text(self.time_unit, display_time)
                self._log(tr("log.time_unit_inferred", unit=inferred_time))

        inferred_strain = self._infer_strain_unit_from_selected_columns()
        if inferred_strain is not None:
            display_strain = self._display_strain_unit(inferred_strain)
            if self.strain_unit.currentText() != display_strain:
                self._set_combo_text(self.strain_unit, display_strain)
                self._update_voltage_conversion_enabled()
                self._log(tr("log.strain_unit_inferred", unit=inferred_strain))

    def _infer_time_unit_from_selected_column(self) -> str | None:
        if self.dataframe is None:
            return None
        column = self.time_column.currentText()
        if not column or column not in self.dataframe.columns:
            return None
        name = str(column).strip().lower()
        compact = name.replace(" ", "")
        if any(token in compact for token in ["/us", "(us)", "[us]", "microsecond"]):
            return "us"
        if any(token in compact for token in ["/ms", "(ms)", "[ms]", "millisecond"]):
            return "ms"
        if any(token in compact for token in ["/s", "(s)", "[s]", "second"]):
            return "s"

        values = pd.to_numeric(self.dataframe[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            return None
        span = float(np.nanmax(values) - np.nanmin(values))
        if 0.0 < span <= 0.1:
            return "s"
        if span >= 100.0:
            return "us"
        return None

    def _infer_strain_unit_from_selected_columns(self) -> str | None:
        if self.dataframe is None:
            return None
        columns = [self.incident_column.currentText(), self.transmitted_column.currentText()]
        names = " ".join(str(column).strip().lower() for column in columns if column)
        if any(token in names for token in ["microstrain", "(ue)", "[ue]", "/ue"]):
            return "microstrain"
        if "strain" in names and not any(token in names for token in ["micro", "(ue)", "[ue]", "/ue"]):
            return "strain"

        peaks: list[float] = []
        for column in columns:
            if not column or column not in self.dataframe.columns:
                continue
            values = pd.to_numeric(self.dataframe[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                peaks.append(float(np.nanmax(np.abs(finite))))
        if not peaks:
            return None
        peak = max(peaks)
        if 1e-8 <= peak <= 0.05:
            return "strain"
        if peak >= 5.0:
            return "microstrain"
        return None

    def _connect_live_updates(self) -> None:
        self.time_column.currentTextChanged.connect(self._update_sampling_frequency_from_time_column)
        self.time_column.currentTextChanged.connect(self._schedule_raw_waveform_preview)
        self.incident_column.currentTextChanged.connect(self._schedule_raw_waveform_preview)
        self.transmitted_column.currentTextChanged.connect(self._schedule_raw_waveform_preview)
        self.time_unit.currentTextChanged.connect(self._update_sampling_frequency_from_time_column)
        self.time_unit.currentTextChanged.connect(self._schedule_raw_waveform_preview)
        self.strain_unit.currentTextChanged.connect(self._update_voltage_conversion_enabled)
        self.strain_unit.currentTextChanged.connect(self._schedule_raw_waveform_preview)
        self.sampling_frequency.valueChanged.connect(self._schedule_raw_waveform_preview)
        self.voltage_to_microstrain.valueChanged.connect(self._schedule_raw_waveform_preview)
        self.baseline_method.currentTextChanged.connect(self._schedule_preprocessing_refresh)
        self.filter_method.currentTextChanged.connect(self._schedule_preprocessing_refresh)
        self.filter_window.valueChanged.connect(self._schedule_preprocessing_refresh)
        self.filter_cutoff.valueChanged.connect(self._schedule_preprocessing_refresh)
        self.specimen_shape.currentTextChanged.connect(self._update_specimen_geometry_enabled)
        self.dispersion_enabled.toggled.connect(self._schedule_window_recalculate)
        self.dispersion_enabled.toggled.connect(self._update_optional_feature_ui)
        self.standard_wave_enabled.toggled.connect(self._schedule_window_recalculate)
        self.standard_wave_enabled.toggled.connect(self._update_optional_feature_ui)
        self.plot_style_preset.currentTextChanged.connect(self._apply_plot_style_preset_from_ui)
        self.plot_background.currentTextChanged.connect(self._update_plot_style_from_ui)
        self.plot_grid_enabled.toggled.connect(self._update_plot_style_from_ui)
        self.plot_default_width.valueChanged.connect(self._update_plot_style_from_ui)
        self.plot_color_sequence.currentIndexChanged.connect(self._update_plot_style_from_ui)
        self.plot_line_style_sequence.currentIndexChanged.connect(self._update_plot_style_from_ui)
        self.plot_curve_overrides.editingFinished.connect(self._update_plot_style_from_ui)
        self.elastic_modulus.valueChanged.connect(self._update_wave_speed_from_bar_inputs)
        self.density.valueChanged.connect(self._update_wave_speed_from_bar_inputs)
        for spin in self.window_spins:
            spin.valueChanged.connect(self._schedule_window_recalculate)
        self.save_alignment_profile_button.clicked.connect(self._save_alignment_profile_from_ui)
        self.load_alignment_profile_button.clicked.connect(self._load_selected_alignment_profile)
        self.delete_alignment_profile_button.clicked.connect(self._delete_selected_alignment_profile)

    def _update_sampling_frequency_from_time_column(self) -> None:
        if self._updating_sampling_frequency or self.dataframe is None:
            return
        time_column = self.time_column.currentText()
        if not time_column or time_column == self._auto_time_column_text():
            return
        try:
            frequency = infer_sampling_frequency_from_column(
                self.dataframe,
                time_column,
                self.time_unit.currentText(),
            )
        except Exception as exc:
            self._log(tr("log.sampling_frequency_failed", error=exc))
            return
        if frequency is None or not np.isfinite(frequency) or frequency <= 0:
            return
        self._updating_sampling_frequency = True
        try:
            self.sampling_frequency.setValue(frequency)
        finally:
            self._updating_sampling_frequency = False
        self._log(tr("log.sampling_frequency_auto", frequency=f"{frequency:.6g}"))

    def _update_wave_speed_from_bar_inputs(self) -> None:
        if self._updating_wave_speed:
            return
        wave_speed = self._computed_wave_speed_from_ui()
        if wave_speed is None:
            return
        self._updating_wave_speed = True
        try:
            self.wave_speed.setValue(wave_speed)
        finally:
            self._updating_wave_speed = False

    def _computed_wave_speed_from_ui(self) -> float | None:
        elastic_modulus_pa = self.elastic_modulus.value() * 1e9
        density_kg_m3 = self.density.value()
        if elastic_modulus_pa <= 0 or density_kg_m3 <= 0:
            return None
        return sqrt(elastic_modulus_pa / density_kg_m3)

    def _update_voltage_conversion_enabled(self, *_args) -> None:
        enabled = self.strain_unit.currentText() == "voltage"
        self.voltage_to_microstrain.setEnabled(enabled)

    def _update_optional_feature_ui(self, *_args) -> None:
        dispersion_enabled = self.dispersion_enabled.isChecked()
        standard_wave_enabled = self.standard_wave_enabled.isChecked()
        self._set_optional_tab_available(self.segment_plot, not standard_wave_enabled)
        self._set_optional_tab_available(self.dispersion_plot, dispersion_enabled)
        self._set_optional_tab_available(self.aligned_plot, not standard_wave_enabled)
        self._set_optional_tab_available(self.standard_wave_plot, standard_wave_enabled)
        self.transmitted_free_end_distance.setEnabled(standard_wave_enabled)

    def _set_optional_tab_available(self, widget: QWidget, available: bool) -> None:
        tabs = getattr(self, "center_tabs", None)
        if tabs is None:
            return
        index = tabs.indexOf(widget)
        if index < 0:
            return
        if not available and tabs.currentWidget() is widget:
            fallback = tabs.indexOf(self.raw_plot)
            if fallback < 0:
                fallback = 0
            tabs.setCurrentIndex(fallback)
        if hasattr(tabs, "setTabVisible"):
            tabs.setTabVisible(index, available)
        else:
            tabs.setTabEnabled(index, available)

    def _apply_plot_style_preset_from_ui(self, *_args) -> None:
        if self._updating_plot_style:
            return
        preset = self.plot_style_preset.currentText()
        settings = self._plot_style_preset(preset)
        self._set_plot_style_controls(settings)
        self._update_plot_style_from_ui()

    def _update_plot_style_from_ui(self, *_args) -> None:
        if self._updating_plot_style:
            return
        self._plot_style = self._plot_style_from_ui()
        self._apply_plot_style_to_all_plots()
        if self.bundle is not None:
            self._plot_all()
        elif self.dataframe is not None:
            self._plot_imported_raw_waveform()

    def _plot_style_from_ui(self) -> PlotStyleSettings:
        return PlotStyleSettings(
            preset=self.plot_style_preset.currentText(),
            background=self.plot_background.currentText().strip() or "#000000",
            foreground=self._foreground_for_background(self.plot_background.currentText()),
            grid_enabled=self.plot_grid_enabled.isChecked(),
            default_width=self.plot_default_width.value(),
            color_sequence=self._selected_sequence(self.plot_color_sequence, self._plot_style.color_sequence),
            line_style_sequence=self._selected_sequence(self.plot_line_style_sequence, self._plot_style.line_style_sequence),
            curves=self._parse_curve_overrides(self.plot_curve_overrides.text()),
        )

    def _set_plot_style_controls(self, settings: PlotStyleSettings) -> None:
        self._updating_plot_style = True
        try:
            self._set_combo_text(self.plot_style_preset, settings.preset)
            self._set_combo_text(self.plot_background, settings.background)
            self.plot_grid_enabled.setChecked(settings.grid_enabled)
            self.plot_default_width.setValue(settings.default_width)
            self._set_sequence_combo(self.plot_color_sequence, settings.color_sequence, is_color=True)
            self._set_sequence_combo(self.plot_line_style_sequence, settings.line_style_sequence, is_color=False)
            self.plot_curve_overrides.setText(self._format_curve_overrides(settings.curves))
        finally:
            self._updating_plot_style = False

    def _apply_plot_style_to_all_plots(self) -> None:
        plots = [
            self.raw_plot,
            self.segment_plot,
            self.dispersion_plot,
            self.aligned_plot,
            self.standard_wave_plot,
            self.result_plot,
        ]
        for plot in plots:
            plot.setBackground(self._plot_style.background)
            plot.showGrid(x=self._plot_style.grid_enabled, y=self._plot_style.grid_enabled, alpha=self._plot_style.grid_alpha)
            plot_item = plot.getPlotItem()
            plot_item.getAxis("bottom").setTextPen(self._plot_style.foreground)
            plot_item.getAxis("left").setTextPen(self._plot_style.foreground)
            plot_item.getAxis("right").setTextPen(self._plot_style.foreground)
            if plot_item.legend is not None:
                plot_item.legend.setBrush((255, 255, 255, 180) if self._plot_style.background.lower() != "#000000" else (0, 0, 0, 130))
                plot_item.legend.setLabelTextColor(self._plot_style.foreground)

    def _pen(self, curve_id: str, fallback_index: int = 0, *, width: float | None = None, style: str | None = None):
        curve_style = self._plot_style.curves.get(curve_id)
        color = curve_style.color if curve_style and curve_style.color else self._sequence_value(
            self._plot_style.color_sequence,
            fallback_index,
            "#1f77b4",
        )
        line_style = curve_style.line_style if curve_style and curve_style.line_style else (
            style or self._sequence_value(self._plot_style.line_style_sequence, fallback_index, "solid")
        )
        line_width = curve_style.width if curve_style and curve_style.width is not None else self._plot_style.default_width
        pen = pg.mkPen(color, width=line_width, style=self._qt_pen_style(line_style))
        if self._normalized_line_style(line_style) == "longdash":
            pen.setDashPattern([8.0, 4.0])
        return pen

    def _plot_style_preset(self, preset: str) -> PlotStyleSettings:
        curves = self._default_curve_styles()
        color_sequences = {key: values for key, _label, values in COLOR_SEQUENCE_OPTIONS}
        line_sequences = {key: values for key, _label, values in LINE_STYLE_SEQUENCE_OPTIONS}
        if preset == "dark_screen":
            return PlotStyleSettings(
                preset=preset,
                background="#000000",
                foreground="#ffffff",
                grid_enabled=True,
                grid_alpha=0.22,
                default_width=1.6,
                color_sequence=color_sequences["screen_bright"],
                line_style_sequence=line_sequences["origin_basic"],
                curves=curves,
            )
        if preset == "light_report":
            return PlotStyleSettings(
                preset=preset,
                background="#ffffff",
                foreground="#000000",
                grid_enabled=True,
                grid_alpha=0.18,
                default_width=1.4,
                color_sequence=color_sequences["okabe_ito"],
                line_style_sequence=line_sequences["origin_basic"],
                curves=curves,
            )
        if preset == "nature_style":
            return PlotStyleSettings(
                preset=preset,
                background="#ffffff",
                foreground="#000000",
                grid_enabled=False,
                grid_alpha=0.10,
                default_width=1.2,
                color_sequence=color_sequences["nature_standard"],
                line_style_sequence=line_sequences["publication_solid"],
                curves=curves,
            )
        if preset == "high_contrast":
            return PlotStyleSettings(
                preset=preset,
                background="#000000",
                foreground="#ffffff",
                grid_enabled=True,
                grid_alpha=0.30,
                default_width=2.0,
                color_sequence=color_sequences["screen_bright"],
                line_style_sequence=line_sequences["origin_basic"],
                curves=curves,
            )
        return PlotStyleSettings(curves=curves)

    def _default_curve_styles(self) -> dict[str, CurveStyle]:
        return default_curve_styles()

    def _populate_color_sequence_combo(self) -> None:
        self.plot_color_sequence.clear()
        self.plot_color_sequence.setIconSize(QSize(120, 18))
        for _key, label_key, colors in COLOR_SEQUENCE_OPTIONS:
            self._add_sequence_combo_item(self.plot_color_sequence, self._color_sequence_icon(colors), tr(label_key), colors)

    def _populate_line_style_sequence_combo(self) -> None:
        self.plot_line_style_sequence.clear()
        self.plot_line_style_sequence.setIconSize(QSize(120, 22))
        for _key, label_key, styles in LINE_STYLE_SEQUENCE_OPTIONS:
            self._add_sequence_combo_item(self.plot_line_style_sequence, self._line_style_sequence_icon(styles), tr(label_key), styles)

    def _add_sequence_combo_item(self, combo: QComboBox, icon: QIcon, label: str, values: list[str]) -> None:
        combo.addItem(icon, label, list(values))
        combo.setItemData(combo.count() - 1, ", ".join(values), Qt.ItemDataRole.ToolTipRole)

    def _selected_sequence(self, combo: QComboBox, fallback: list[str]) -> list[str]:
        data = combo.currentData()
        if isinstance(data, list) and data:
            return [str(item) for item in data if str(item)]
        return list(fallback)

    def _set_sequence_combo(self, combo: QComboBox, values: list[str], *, is_color: bool) -> None:
        sequence = [str(item).strip() for item in values if str(item).strip()]
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, list) and self._same_sequence(data, sequence):
                combo.setCurrentIndex(index)
                return
        label = tr("plot.custom_color_sequence") if is_color else tr("plot.custom_line_style_sequence")
        icon = self._color_sequence_icon(sequence) if is_color else self._line_style_sequence_icon(sequence)
        self._add_sequence_combo_item(combo, icon, label, sequence)
        combo.setCurrentIndex(combo.count() - 1)

    def _same_sequence(self, left: list[str], right: list[str]) -> bool:
        return [str(item).strip().lower() for item in left] == [str(item).strip().lower() for item in right]

    def _color_sequence_icon(self, colors: list[str]) -> QIcon:
        width = 120
        height = 18
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        values = colors or ["#000000"]
        swatch_width = width / len(values)
        for index, color in enumerate(values):
            x0 = int(round(index * swatch_width))
            x1 = int(round((index + 1) * swatch_width))
            painter.fillRect(x0, 2, max(1, x1 - x0), height - 4, QColor(color))
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawRect(0, 2, width - 1, height - 5)
        painter.end()
        return QIcon(pixmap)

    def _line_style_sequence_icon(self, styles: list[str]) -> QIcon:
        width = 120
        height = 22
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        values = styles or ["solid"]
        for index, style in enumerate(values[:5]):
            y = int(round((index + 1) * height / (min(len(values), 5) + 1)))
            pen = QPen(QColor("#222222"), 2)
            pen.setStyle(self._qt_pen_style(style))
            if self._normalized_line_style(style) == "longdash":
                pen.setDashPattern([8.0, 4.0])
            painter.setPen(pen)
            painter.drawLine(4, y, width - 4, y)
        painter.end()
        return QIcon(pixmap)

    def _parse_curve_overrides(self, text: str) -> dict[str, CurveStyle]:
        curves: dict[str, CurveStyle] = {}
        for entry in text.split(";"):
            if "=" not in entry:
                continue
            curve_id, payload = entry.split("=", 1)
            curve_id = curve_id.strip()
            if not curve_id:
                continue
            parts = [part.strip() for part in payload.split(",")]
            color = parts[0] if len(parts) >= 1 and parts[0] else None
            line_style = parts[1] if len(parts) >= 2 and parts[1] else None
            width = None
            if len(parts) >= 3 and parts[2]:
                try:
                    width = float(parts[2])
                except ValueError:
                    width = None
            curves[curve_id] = CurveStyle(color=color, line_style=line_style, width=width)
        return curves

    def _format_curve_overrides(self, curves: dict[str, CurveStyle]) -> str:
        entries = []
        for curve_id, style in sorted(curves.items()):
            values = [
                style.color or "",
                style.line_style or "",
                "" if style.width is None else f"{style.width:g}",
            ]
            entries.append(f"{curve_id}={','.join(values)}")
        return "; ".join(entries)

    def _foreground_for_background(self, background: str) -> str:
        return "#000000" if background.lower() in {"#ffffff", "#f5f5f5", "white"} else "#ffffff"

    def _sequence_value(self, values: list[str], index: int, fallback: str) -> str:
        if not values:
            return fallback
        return values[index % len(values)]

    def _normalized_line_style(self, style: str | None) -> str:
        return (style or "solid").strip().lower().replace("_", "").replace("-", "")

    def _qt_pen_style(self, style: str | None):
        normalized = self._normalized_line_style(style)
        mapping = {
            "solid": Qt.PenStyle.SolidLine,
            "dash": Qt.PenStyle.DashLine,
            "dashed": Qt.PenStyle.DashLine,
            "dashdot": Qt.PenStyle.DashDotLine,
            "dot": Qt.PenStyle.DotLine,
            "dotted": Qt.PenStyle.DotLine,
            "longdash": Qt.PenStyle.CustomDashLine,
        }
        return mapping.get(normalized, Qt.PenStyle.SolidLine)

    def _update_specimen_geometry_enabled(self, *_args) -> None:
        shape = self.specimen_shape.currentText()
        uses_diameter = shape == SpecimenShape.CYLINDER.value
        uses_width_thickness = shape in {
            SpecimenShape.RECTANGLE.value,
            SpecimenShape.SHEET.value,
            SpecimenShape.DOGBONE.value,
        }
        uses_area = shape == SpecimenShape.CUSTOM.value
        self.specimen_diameter.setEnabled(uses_diameter)
        self.specimen_width.setEnabled(uses_width_thickness)
        self.specimen_thickness.setEnabled(uses_width_thickness)
        self.specimen_area.setEnabled(uses_area)

    def _preview_dataframe(self) -> None:
        if self.dataframe is None:
            return
        preview = self.dataframe.head(80)
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for row_idx, (_, row) in enumerate(preview.iterrows()):
            for col_idx, value in enumerate(row):
                self.preview_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.preview_table.resizeColumnsToContents()

    def _plot_imported_raw_waveform(self, *_args) -> None:
        if self.dataframe is None:
            return
        if not self.incident_column.currentText() or not self.transmitted_column.currentText():
            return
        try:
            raw_preview = process_signals(
                self.dataframe,
                self._mapping_from_ui(),
                self._acquisition_from_ui(),
                PreprocessingSettings(
                    baseline_method="none",
                    filter_method="none",
                    remove_nan_rows=True,
                ),
            )
        except Exception:
            return

        processed_preview = None
        show_processed = self._preprocessing_preview_has_effect()
        if show_processed:
            try:
                processed_preview = process_signals(
                    self.dataframe,
                    self._mapping_from_ui(),
                    self._acquisition_from_ui(),
                    self._preprocessing_from_ui(),
                )
            except Exception:
                processed_preview = None

        self.raw_plot.clear()
        self._reset_legend(self.raw_plot)
        dashed = Qt.PenStyle.DashLine
        self.raw_plot.plot(
            raw_preview.time_s * 1e6,
            raw_preview.incident_strain * 1e6,
            pen=self._pen("raw.incident_raw", 0, width=1.0, style="dash"),
            name="incident raw",
        )
        self.raw_plot.plot(
            raw_preview.time_s * 1e6,
            raw_preview.transmitted_strain * 1e6,
            pen=self._pen("raw.transmitted_raw", 1, width=1.0, style="dash"),
            name="transmitted raw",
        )
        if processed_preview is not None:
            self.raw_plot.plot(
                processed_preview.time_s * 1e6,
                processed_preview.incident_strain * 1e6,
                pen=self._pen("raw.incident_processed", 0, width=1.8, style="solid"),
                name="incident preprocessed",
            )
            self.raw_plot.plot(
                processed_preview.time_s * 1e6,
                processed_preview.transmitted_strain * 1e6,
                pen=self._pen("raw.transmitted_processed", 1, width=1.8, style="solid"),
                name="transmitted preprocessed",
            )
        title = tr("plot.raw_preprocessed")
        if show_processed and processed_preview is None:
            title = tr("plot.raw_preview_failed")
        self.raw_plot.setTitle(title)
        self.raw_plot.setLabel("bottom", tr("axis.time"), units="μs")
        self.raw_plot.setLabel("left", tr("axis.strain"), units="με")

    def _plot_segment_preview_from_current_windows(self) -> None:
        if self.dataframe is None:
            return
        try:
            processed = process_signals(
                self.dataframe,
                self._mapping_from_ui(),
                self._acquisition_from_ui(),
                self._preprocessing_from_ui(),
            )
            segments = self._segments_from_windows(
                processed.time_s,
                processed.incident_strain,
                processed.transmitted_strain,
                self._windows_from_ui(),
            )
        except Exception:
            return
        self._plot_segments(segments, title=tr("plot.segment_preview_unaligned"))

    def _schedule_raw_waveform_preview(self, *_args) -> None:
        if self.dataframe is None:
            return
        self._raw_waveform_preview_timer.start()

    def _schedule_preprocessing_refresh(self, *_args) -> None:
        self._schedule_raw_waveform_preview()
        if self.bundle is not None:
            self._schedule_window_recalculate()

    def _preprocessing_preview_has_effect(self) -> bool:
        return self.baseline_method.currentText() != "none" or self.filter_method.currentText() != "none"

    def _reset_processing_state_for_new_data(self) -> None:
        self.bundle = None
        self.processed = None
        self.mapping = None
        self.summary.clear()
        self._clear_aligned_balance_axis()
        for plot in [self.raw_plot, self.segment_plot, self.dispersion_plot, self.aligned_plot, self.standard_wave_plot, self.result_plot]:
            plot.clear()
            self._reset_legend(plot)

    def _plot_all(self) -> None:
        if self.bundle is None:
            return
        b = self.bundle
        self.raw_plot.clear()
        self._reset_legend(self.raw_plot)
        self.raw_plot.plot(b.processed.time_s * 1e6, b.processed.incident_strain * 1e6, pen=self._pen("raw.incident", 0, width=1.5), name="incident")
        self.raw_plot.plot(b.processed.time_s * 1e6, b.processed.transmitted_strain * 1e6, pen=self._pen("raw.transmitted", 1, width=1.5), name="transmitted")
        self.raw_plot.setTitle(tr("plot.raw_preprocessed"))
        self.raw_plot.setLabel("bottom", tr("axis.time"), units="μs")
        self.raw_plot.setLabel("left", tr("axis.strain"), units="με")

        self._plot_segments(b.segments)

        self._plot_dispersion_comparison(b)

        self._plot_aligned_waves(b)

        self._plot_standard_wave_separation(b)

        self.result_plot.clear()
        self._reset_legend(self.result_plot)
        if self.standard_wave_enabled.isChecked() and b.standard_wave_separation is not None:
            self.result_plot.plot(
                b.standard_wave_separation.engineering_strain,
                b.standard_wave_separation.engineering_stress_pa / 1e6,
                pen=self._pen("result.standard_wave", 0, width=2.2),
                name="standard wave",
            )
        else:
            self.result_plot.plot(b.three_wave.strain, b.three_wave.engineering_stress_pa / 1e6, pen=self._pen("result.three_wave", 0, width=2), name="three-wave")
            self.result_plot.plot(b.two_wave.strain, b.two_wave.engineering_stress_pa / 1e6, pen=self._pen("result.two_wave", 1, width=2), name="two-wave")
        self.result_plot.setLabel("bottom", tr("axis.engineering_strain"))
        self.result_plot.setLabel("left", tr("axis.engineering_stress"), units="MPa")

    def _plot_standard_wave_separation(self, bundle: ProcessingBundle) -> None:
        self.standard_wave_plot.clear()
        self._reset_legend(self.standard_wave_plot)
        result = bundle.standard_wave_separation
        if result is None:
            self.standard_wave_plot.setTitle(tr("plot.standard_wave_disabled"))
            return
        time_us = result.time_s * 1e6
        self.standard_wave_plot.plot(time_us, result.gauge1_signal * 1e6, pen=self._pen("standard.gauge1_signal", 6, width=1.3), name="gauge1 signal")
        self.standard_wave_plot.plot(time_us, result.gauge1_right_going * 1e6, pen=self._pen("standard.gauge1_right", 6, width=1.3, style="dash"), name="gauge1 right")
        self.standard_wave_plot.plot(time_us, result.gauge1_left_going * 1e6, pen=self._pen("standard.gauge1_left", 7, width=1.3, style="dashdot"), name="gauge1 left")
        self.standard_wave_plot.plot(time_us, result.gauge2_signal * 1e6, pen=self._pen("standard.gauge2_signal", 0, width=1.1), name="gauge2 signal")
        self.standard_wave_plot.plot(time_us, result.gauge2_right_going * 1e6, pen=self._pen("standard.gauge2_right", 0, width=1.1, style="dash"), name="gauge2 right")
        self.standard_wave_plot.plot(time_us, result.gauge2_left_going * 1e6, pen=self._pen("standard.gauge2_left", 1, width=1.1, style="dashdot"), name="gauge2 left")
        self.standard_wave_plot.setTitle(tr("plot.standard_wave"))
        self.standard_wave_plot.setLabel("bottom", tr("axis.time"), units="μs")
        self.standard_wave_plot.setLabel("left", tr("axis.strain"), units="με")

    def _plot_aligned_waves(self, bundle: ProcessingBundle) -> None:
        self.aligned_plot.clear()
        self._reset_legend(self.aligned_plot)
        self._clear_aligned_balance_axis()

        plot_item = self.aligned_plot.getPlotItem()
        aligned = bundle.aligned
        time_us = aligned.time_s * 1e6
        plot_item.plot(time_us, aligned.incident * 1e6, pen=self._pen("aligned.incident", 0, width=1.5), name="incident")
        plot_item.plot(time_us, aligned.reflected * 1e6, pen=self._pen("aligned.reflected", 1, width=1.5), name="reflected")
        plot_item.plot(time_us, aligned.transmitted * 1e6, pen=self._pen("aligned.transmitted", 2, width=1.5), name="transmitted")
        plot_item.plot(time_us, (aligned.transmitted - aligned.reflected) * 1e6, pen=self._pen("aligned.transmitted_minus_reflected", 3, width=1.2), name="tr - re")
        plot_item.setLabel("bottom", tr("axis.aligned_time"), units="μs")
        plot_item.setLabel("left", tr("axis.strain"), units="με")

        balance_error = np.asarray(bundle.three_wave.balance_error, dtype=float)
        if len(time_us) == len(balance_error):
            right_axis = plot_item.getAxis("right")
            right_axis.setLabel(tr("axis.balance_error"))
            right_axis.show()
            balance_view = pg.ViewBox()
            self._aligned_balance_view = balance_view
            plot_item.scene().addItem(balance_view)
            right_axis.linkToView(balance_view)
            balance_view.setXLink(plot_item.vb)
            balance_view.setGeometry(plot_item.vb.sceneBoundingRect())
            if self._aligned_balance_resize_connected:
                plot_item.vb.sigResized.disconnect(self._update_aligned_balance_axis_geometry)
                self._aligned_balance_resize_connected = False
            plot_item.vb.sigResized.connect(self._update_aligned_balance_axis_geometry)
            self._aligned_balance_resize_connected = True
            balance_curve = pg.PlotDataItem(time_us, balance_error, pen=self._pen("aligned.balance_error", 7, width=1.2), name="balance error")
            self._aligned_balance_curve = balance_curve
            balance_view.addItem(balance_curve)
            if plot_item.legend is not None:
                plot_item.legend.addItem(balance_curve, tr("legend.balance_error"))
            finite = balance_error[np.isfinite(balance_error)]
            if len(finite):
                max_error = max(1.0, float(np.nanmax(finite)) * 1.05)
                balance_view.setYRange(0.0, max_error, padding=0.02)

    def _plot_segments(self, segments: WaveSegments, title: str | None = None) -> None:
        self.segment_plot.clear()
        self._reset_legend(self.segment_plot)
        self.segment_plot.plot(segments.time_s * 1e6, segments.incident * 1e6, pen=self._pen("segments.incident", 0, width=2), name="incident")
        self.segment_plot.plot(segments.time_s * 1e6, segments.reflected * 1e6, pen=self._pen("segments.reflected", 1, width=2), name="reflected")
        self.segment_plot.plot(segments.time_s * 1e6, segments.transmitted * 1e6, pen=self._pen("segments.transmitted", 2, width=2), name="transmitted")
        self.segment_plot.setTitle(title or tr("plot.segment"))
        self.segment_plot.setLabel("bottom", tr("axis.time"), units="μs")
        self.segment_plot.setLabel("left", tr("axis.strain"), units="με")

    def _plot_dispersion_comparison(self, bundle: ProcessingBundle) -> None:
        self.dispersion_plot.clear()
        self._reset_legend(self.dispersion_plot)
        enabled = bool(bundle.segments.metadata.get("dispersion_correction_enabled", False))
        before = bundle.uncorrected_segments
        after = bundle.segments
        if enabled and before is not None:
            for label, values, curve_id, index in [
                ("incident before", before.incident, "dispersion.incident_before", 0),
                ("reflected before", before.reflected, "dispersion.reflected_before", 1),
                ("transmitted before", before.transmitted, "dispersion.transmitted_before", 2),
            ]:
                self.dispersion_plot.plot(before.time_s * 1e6, values * 1e6, pen=self._pen(curve_id, index, width=1.4, style="dash"), name=label)
            for label, values, curve_id, index in [
                ("incident after", after.incident, "dispersion.incident_after", 0),
                ("reflected after", after.reflected, "dispersion.reflected_after", 1),
                ("transmitted after", after.transmitted, "dispersion.transmitted_after", 2),
            ]:
                self.dispersion_plot.plot(after.time_s * 1e6, values * 1e6, pen=self._pen(curve_id, index, width=2.0), name=label)
        else:
            self.dispersion_plot.plot(after.time_s * 1e6, after.incident * 1e6, pen=self._pen("dispersion.incident", 0, width=2), name="incident")
            self.dispersion_plot.plot(after.time_s * 1e6, after.reflected * 1e6, pen=self._pen("dispersion.reflected", 1, width=2), name="reflected")
            self.dispersion_plot.plot(after.time_s * 1e6, after.transmitted * 1e6, pen=self._pen("dispersion.transmitted", 2, width=2), name="transmitted")
        self.dispersion_plot.setTitle(tr("plot.dispersion_comparison") if enabled else tr("plot.dispersion_disabled"))
        self.dispersion_plot.setLabel("bottom", tr("axis.time"), units="μs")
        self.dispersion_plot.setLabel("left", tr("axis.strain"), units="με")

    def _clear_aligned_balance_axis(self) -> None:
        plot_item = self.aligned_plot.getPlotItem()
        if self._aligned_balance_resize_connected:
            plot_item.vb.sigResized.disconnect(self._update_aligned_balance_axis_geometry)
            self._aligned_balance_resize_connected = False
        if self._aligned_balance_view is not None:
            try:
                plot_item.scene().removeItem(self._aligned_balance_view)
            except Exception:
                pass
        self._aligned_balance_view = None
        self._aligned_balance_curve = None
        plot_item.getAxis("right").hide()

    def _update_aligned_balance_axis_geometry(self) -> None:
        if self._aligned_balance_view is None:
            return
        plot_item = self.aligned_plot.getPlotItem()
        self._aligned_balance_view.setGeometry(plot_item.vb.sceneBoundingRect())
        self._aligned_balance_view.linkedViewChanged(plot_item.vb, self._aligned_balance_view.XAxis)

    def _update_summary(self) -> None:
        if self.bundle is None:
            return
        b = self.bundle
        if b.standard_wave_separation is not None:
            standard = b.standard_wave_separation
            dispersion_enabled = bool(b.segments.metadata.get("dispersion_correction_enabled", False))
            lines = [
                tr("summary.standard_flow"),
                tr("summary.quality_grade_aux", grade=b.quality.grade),
                tr("summary.peak_stress", value=f"{standard.summary.get('peak_stress_mpa', 0):.3f}"),
                tr("summary.max_strain", value=f"{standard.summary.get('max_strain', 0):.6g}"),
                tr("summary.max_abs_strain_rate", value=f"{standard.summary.get('max_abs_strain_rate_s^-1', 0):.6g}"),
                tr("summary.transmitted_gauge_free_end", value=f"{standard.metadata.get('transmitted_gauge_to_free_end_m', 0):.6g}"),
                tr("summary.incident_gauge_specimen", value=f"{standard.metadata.get('incident_gauge_to_specimen_m', 0):.6g}"),
                tr("summary.transmitted_gauge_specimen", value=f"{standard.metadata.get('transmitted_gauge_to_specimen_m', 0):.6g}"),
                tr("summary.incident_tau", value=f"{standard.summary.get('standard_wave_incident_tau_s', 0) * 1e6:.3f}"),
                tr("summary.transmitted_tau", value=f"{standard.summary.get('standard_wave_transmitted_tau_s', 0) * 1e6:.3f}"),
                tr("summary.free_end_tau", value=f"{standard.summary.get('standard_wave_free_end_tau_s', 0) * 1e6:.3f}"),
                tr("summary.force_closure_max", value=f"{float(standard.metadata.get('force_closure_max_abs_n', 0.0)):.6g}"),
                tr("summary.gauge1_reconstruction_error", value=f"{float(standard.metadata.get('gauge1_reconstruction_max_abs_strain', 0.0)):.6g}"),
                tr("summary.gauge2_reconstruction_error", value=f"{float(standard.metadata.get('gauge2_reconstruction_max_abs_strain', 0.0)):.6g}"),
                tr("summary.dispersion_standard_note", state=self._enabled_label(dispersion_enabled)),
                "",
                tr("summary.warning_header"),
            ]
            lines.extend(self._localized_warnings(standard.warnings or b.quality.warnings))
            self.summary.setPlainText("\n".join(lines))
            return

        auto_error = b.aligned.metadata.get("auto_alignment_final_error")
        objective = b.aligned.metadata.get("auto_alignment_objective", "unknown")
        force_error = b.aligned.metadata.get("auto_alignment_final_force_balance_error")
        force_improvement = b.aligned.metadata.get("auto_alignment_force_balance_improvement")
        wave_relation_error = b.aligned.metadata.get("auto_alignment_final_wave_relation_error")
        reflected_delta = b.aligned.metadata.get("auto_alignment_reflected_delta_s", 0.0)
        transmitted_delta = b.aligned.metadata.get("auto_alignment_transmitted_delta_s", 0.0)
        auto_status = b.aligned.metadata.get("auto_alignment_status", "unknown")
        zero_applied = b.three_wave.summary.get("zero_reference_applied", 0.0) >= 0.5
        wave_head_ok = b.three_wave.summary.get("wave_head_retention_ok", 1.0) >= 0.5
        initial_stress_offset = b.three_wave.summary.get("initial_stress_offset_mpa", 0.0)
        initial_strain_offset = b.three_wave.summary.get("initial_strain_offset", 0.0)
        dispersion_enabled = bool(b.segments.metadata.get("dispersion_correction_enabled", False))
        lines = [
            tr("summary.quality_grade", grade=b.quality.grade),
            tr("summary.three_wave_peak_stress", value=f"{b.three_wave.summary.get('peak_stress_mpa', 0):.3f}"),
            tr("summary.two_wave_peak_stress", value=f"{b.two_wave.summary.get('peak_stress_mpa', 0):.3f}"),
            tr("summary.mean_force_balance_error", value=f"{b.three_wave.summary.get('mean_balance_error', 0):.3f}"),
            tr("summary.dispersion", state=self._enabled_label(dispersion_enabled)),
            tr("summary.alignment_objective", objective=self._code_label("code.alignment_objective", str(objective))),
            tr("summary.force_balance_alignment_error", value=f"{force_error:.4f}") if isinstance(force_error, float) else tr("summary.force_balance_alignment_error_missing"),
            tr("summary.force_balance_improvement", value=f"{force_improvement * 100:.1f}") if isinstance(force_improvement, float) else tr("summary.force_balance_improvement_missing"),
            tr("summary.wave_relation_error", value=f"{wave_relation_error:.4f}") if isinstance(wave_relation_error, float) else tr("summary.wave_relation_error_missing"),
            tr("summary.initial_zero_reference", state=self._enabled_label(zero_applied)),
            tr("summary.wave_head_retention", state=tr("summary.sufficient") if wave_head_ok else tr("summary.insufficient_review")),
            tr("summary.initial_stress_offset", value=f"{initial_stress_offset:.3f}"),
            tr("summary.initial_strain_offset", value=f"{initial_strain_offset:.6g}"),
            tr("summary.objective_error", value=f"{auto_error:.4f}") if isinstance(auto_error, float) else tr("summary.objective_error_missing"),
            tr("summary.auto_alignment_status", status=self._code_label("code.auto_alignment_status", str(auto_status))),
            tr("summary.incident_shift", value=f"{b.aligned.incident_shift_s * 1e6:.3f}"),
            tr("summary.reflected_shift", value=f"{b.aligned.reflected_shift_s * 1e6:.3f}"),
            tr("summary.transmitted_shift", value=f"{b.aligned.transmitted_shift_s * 1e6:.3f}"),
            tr("summary.reflected_fine_adjust", value=f"{float(reflected_delta) * 1e6:.3f}"),
            tr("summary.transmitted_fine_adjust", value=f"{float(transmitted_delta) * 1e6:.3f}"),
            "",
            tr("summary.warning_header"),
        ]
        if dispersion_enabled:
            lines[5:5] = [
                tr("summary.dispersion_rms_incident", value=f"{float(b.segments.metadata.get('dispersion_incident_rms_delta', 0.0)):.6g}"),
                tr("summary.dispersion_rms_reflected", value=f"{float(b.segments.metadata.get('dispersion_reflected_rms_delta', 0.0)):.6g}"),
                tr("summary.dispersion_rms_transmitted", value=f"{float(b.segments.metadata.get('dispersion_transmitted_rms_delta', 0.0)):.6g}"),
            ]
        lines.extend(self._localized_warnings(b.quality.warnings))
        self.summary.setPlainText("\n".join(lines))

    def _mapping_from_ui(self) -> ColumnMapping:
        time_column = self.time_column.currentText()
        if time_column == self._auto_time_column_text():
            time_column = None
        return ColumnMapping(
            time_column=time_column,
            incident_column=self.incident_column.currentText(),
            transmitted_column=self.transmitted_column.currentText(),
        )

    def _acquisition_from_ui(self) -> AcquisitionParameters:
        strain_unit = self.strain_unit.currentText()
        return AcquisitionParameters(
            sampling_frequency_hz=self.sampling_frequency.value(),
            time_unit=self.time_unit.currentText(),
            strain_unit=strain_unit,
            voltage_to_microstrain_per_volt=self.voltage_to_microstrain.value()
            if strain_unit == "voltage"
            else None,
        )

    def _preprocessing_from_ui(self) -> PreprocessingSettings:
        cutoff = self.filter_cutoff.value() if self.filter_method.currentText() == "butterworth" else None
        return PreprocessingSettings(
            baseline_method=self.baseline_method.currentText(),
            filter_method=self.filter_method.currentText(),
            filter_window_points=self.filter_window.value(),
            filter_cutoff_hz=cutoff,
        )

    def _alignment_from_ui(self) -> AlignmentSettings:
        return AlignmentSettings(
            auto_micro_adjust=self.auto_micro_align.isChecked(),
            alignment_objective=self.alignment_objective.currentText(),
        )

    def _calculation_from_ui(self) -> CalculationSettings:
        return CalculationSettings()

    def _dispersion_from_ui(self) -> DispersionSettings:
        return DispersionSettings(
            enabled=self.dispersion_enabled.isChecked(),
            poisson_ratio=0.30,
            amplitude_correction=True,
        )

    def _standard_wave_separation_from_ui(self) -> StandardWaveSeparationSettings:
        return StandardWaveSeparationSettings(
            enabled=self.standard_wave_enabled.isChecked(),
            transmitted_gauge_to_free_end_m=self.transmitted_free_end_distance.value() * 1e-3,
            incident_gauge_to_specimen_m=self.incident_distance.value() * 1e-3,
            transmitted_gauge_to_specimen_m=self.transmitted_distance.value() * 1e-3,
        )

    def _workspace_config_from_ui(self, report_title: str, include_manual_windows: bool = True) -> WorkspaceConfig:
        config = WorkspaceConfig(
            acquisition=self._acquisition_from_ui(),
            bar=self._bar_from_ui(),
            specimen=self._specimen_from_ui(),
            mapping=self.mapping or self._mapping_from_ui(),
            preprocessing=self._preprocessing_from_ui(),
            pulse_detection=self._pulse_detection_from_ui(include_manual_windows=include_manual_windows),
            alignment=self._alignment_from_ui(),
            calculation=self._calculation_from_ui(),
            dispersion=self._dispersion_from_ui(),
            wave_separation=self._standard_wave_separation_from_ui(),
            plot_style=self._plot_style_from_ui(),
            sign_convention=self._sign_from_ui(),
        )
        config.output.include_excel_workbook = True
        config.output.include_html_report = True
        config.output.include_pdf_report = True
        config.output.include_figures = True
        config.output.report_title = report_title
        return config

    def _pulse_detection_from_ui(self, include_manual_windows: bool = True) -> PulseDetectionSettings:
        fixed_windows = self._fixed_windows_from_ui(require_valid=False) if include_manual_windows else None
        return PulseDetectionSettings(fixed_windows=fixed_windows)

    def _fixed_windows_from_ui(self, require_valid: bool) -> FixedPulseWindows | None:
        incident_window, reflected_window, transmitted_window = self._windows_from_ui()
        windows = [incident_window, reflected_window, transmitted_window]
        valid = all(window.end_s > window.start_s for window in windows)
        if not valid:
            if require_valid:
                self._error(tr("error.invalid_windows_for_processing"))
            return None
        return FixedPulseWindows(
            incident=incident_window,
            reflected=reflected_window,
            transmitted=transmitted_window,
        )

    def _bar_from_ui(self) -> BarParameters:
        wave_speed = self._computed_wave_speed_from_ui()
        if wave_speed is not None:
            self.wave_speed.setValue(wave_speed)
        return BarParameters(
            incident_diameter_m=self.incident_diameter.value() * 1e-3,
            transmitted_diameter_m=self.transmitted_diameter.value() * 1e-3,
            elastic_modulus_pa=self.elastic_modulus.value() * 1e9,
            density_kg_m3=self.density.value(),
            wave_speed_m_s=wave_speed,
            incident_gauge_distance_m=self.incident_distance.value() * 1e-3,
            transmitted_gauge_distance_m=self.transmitted_distance.value() * 1e-3,
            material_name="custom",
        )

    def _specimen_from_ui(self) -> SpecimenParameters:
        shape = SpecimenShape(self.specimen_shape.currentText())
        values = {
            "shape": shape,
            "length_m": self.specimen_length.value() * 1e-3,
            "experiment_type": ExperimentType(self.experiment_type.currentText()),
            "specimen_id": self.specimen_id.text(),
        }
        if shape == SpecimenShape.CYLINDER:
            values["diameter_m"] = self.specimen_diameter.value() * 1e-3
        elif shape in {SpecimenShape.RECTANGLE, SpecimenShape.SHEET, SpecimenShape.DOGBONE}:
            values["width_m"] = self.specimen_width.value() * 1e-3
            values["thickness_m"] = self.specimen_thickness.value() * 1e-3
        elif shape == SpecimenShape.CUSTOM:
            values["area_m2"] = self.specimen_area.value() * 1e-6
        return SpecimenParameters(**values)

    def _sign_from_ui(self) -> SignConvention:
        return SignConvention(self.sign_convention.currentText())

    def _apply_workspace_config(self, config: WorkspaceConfig) -> None:
        acquisition = config.acquisition
        if acquisition.sampling_frequency_hz:
            self.sampling_frequency.setValue(float(acquisition.sampling_frequency_hz))
        self._set_combo_text(self.time_unit, self._display_time_unit(str(acquisition.time_unit)))
        self._set_combo_text(self.strain_unit, self._display_strain_unit(str(acquisition.strain_unit)))
        if acquisition.voltage_to_microstrain_per_volt:
            self.voltage_to_microstrain.setValue(float(acquisition.voltage_to_microstrain_per_volt))
        self._update_voltage_conversion_enabled()

        preprocessing = config.preprocessing
        self._set_combo_text(self.baseline_method, preprocessing.baseline_method)
        self._set_combo_text(self.filter_method, preprocessing.filter_method)
        self.filter_window.setValue(int(preprocessing.filter_window_points))
        if preprocessing.filter_cutoff_hz:
            self.filter_cutoff.setValue(float(preprocessing.filter_cutoff_hz))

        bar = config.bar
        self.incident_diameter.setValue(bar.incident_diameter_m * 1e3)
        self.transmitted_diameter.setValue(bar.transmitted_diameter_m * 1e3)
        self.elastic_modulus.setValue(bar.elastic_modulus_pa / 1e9)
        self.density.setValue(bar.density_kg_m3)
        self.incident_distance.setValue(bar.incident_gauge_distance_m * 1e3)
        self.transmitted_distance.setValue(bar.transmitted_gauge_distance_m * 1e3)
        self._update_wave_speed_from_bar_inputs()

        self.standard_wave_enabled.setChecked(config.wave_separation.enabled)
        self.transmitted_free_end_distance.setValue(config.wave_separation.transmitted_gauge_to_free_end_m * 1e3)
        self._plot_style = config.plot_style
        self._set_plot_style_controls(config.plot_style)
        self._apply_plot_style_to_all_plots()

        specimen = config.specimen
        self.specimen_id.setText(specimen.specimen_id)
        self._set_combo_text(self.specimen_shape, specimen.shape.value)
        if specimen.diameter_m:
            self.specimen_diameter.setValue(specimen.diameter_m * 1e3)
        if specimen.width_m:
            self.specimen_width.setValue(specimen.width_m * 1e3)
        if specimen.thickness_m:
            self.specimen_thickness.setValue(specimen.thickness_m * 1e3)
        if specimen.area_m2:
            self.specimen_area.setValue(specimen.area_m2 * 1e6)
        self.specimen_length.setValue(specimen.length_m * 1e3)
        self._set_combo_text(self.experiment_type, specimen.experiment_type.value)
        self._update_specimen_geometry_enabled()

        if config.mapping is not None:
            if config.mapping.time_column is None:
                self._set_combo_text(self.time_column, self._auto_time_column_text())
            else:
                self._set_combo_text(self.time_column, config.mapping.time_column)
            self._set_combo_text(self.incident_column, config.mapping.incident_column)
            self._set_combo_text(self.transmitted_column, config.mapping.transmitted_column)
            self.mapping = config.mapping

        self.auto_micro_align.setChecked(config.alignment.auto_micro_adjust)
        self._set_combo_text(self.alignment_objective, config.alignment.alignment_objective)
        self._set_combo_text(self.sign_convention, config.sign_convention.value)
        self.dispersion_enabled.setChecked(config.dispersion.enabled)

        fixed = config.pulse_detection.fixed_windows
        if fixed is not None:
            self._set_window_spin_values(
                [
                    fixed.incident.start_s,
                    fixed.incident.end_s,
                    fixed.reflected.start_s,
                    fixed.reflected.end_s,
                    fixed.transmitted.start_s,
                    fixed.transmitted.end_s,
                ]
            )

        if self.dataframe is not None and self.processed is not None:
            self._rebuild_from_current_windows(show_errors=False, log_success=False)
        self._update_optional_feature_ui()

    def _set_combo_text(self, combo: QComboBox, value: str) -> None:
        if combo.findText(value) < 0:
            combo.addItem(value)
        combo.setCurrentText(value)

    def _display_time_unit(self, value: str) -> str:
        if value in {"us", "μs", "µs"}:
            return "μs"
        return value

    def _display_strain_unit(self, value: str) -> str:
        if value in {"microstrain", "ue", "με", "µε"}:
            return "με"
        return value

    def _auto_time_column_text(self) -> str:
        return tr("placeholder.auto_time_column")

    def _set_spin_if_present(self, spin: QDoubleSpinBox, values: dict[str, object], key: str) -> None:
        if key in values:
            spin.setValue(float(values[key]))

    def _default_report_folder_name(self) -> str:
        if self.source_path:
            name = Path(self.source_path).stem
        else:
            name = self.specimen_id.text() or "sample"
        return f"{safe_output_name(name)}_report"

    def _set_window_spins(self, segments) -> None:
        self._set_window_spin_values([
            segments.incident_window.start_s,
            segments.incident_window.end_s,
            segments.reflected_window.start_s,
            segments.reflected_window.end_s,
            segments.transmitted_window.start_s,
            segments.transmitted_window.end_s,
        ])

    def _set_window_spin_values(self, values_s: list[float]) -> None:
        self._updating_window_spins = True
        try:
            for spin, value in zip(self.window_spins, values_s):
                spin.setValue(value * 1e6)
        finally:
            self._updating_window_spins = False

    def _windows_from_ui(self) -> tuple[PulseWindow, PulseWindow, PulseWindow]:
        values = [spin.value() * 1e-6 for spin in self.window_spins]
        return (
            PulseWindow(start_s=values[0], end_s=values[1], label="incident", confidence=1.0),
            PulseWindow(start_s=values[2], end_s=values[3], label="reflected", confidence=1.0),
            PulseWindow(start_s=values[4], end_s=values[5], label="transmitted", confidence=1.0),
        )

    def _segments_from_windows(self, time_s, incident, transmitted, windows) -> WaveSegments:
        incident_window, reflected_window, transmitted_window = windows
        return WaveSegments(
            time_s=time_s,
            incident=_mask(time_s, incident, incident_window),
            reflected=_mask(time_s, incident, reflected_window),
            transmitted=_mask(time_s, transmitted, transmitted_window),
            incident_window=incident_window,
            reflected_window=reflected_window,
            transmitted_window=transmitted_window,
        )

    def _validate_segments_for_alignment(self, segments: WaveSegments) -> None:
        counts = self._segment_point_counts(segments)
        too_short = {label: count for label, count in counts.items() if count < 2}
        if not too_short:
            return
        details = tr("punctuation.list_separator").join(
            tr("error.segment_point_count", label=label, count=count) for label, count in too_short.items()
        )
        raise ValueError(
            tr("error.segment_alignment_detail", details=details)
            + tr("error.segment_alignment_range_hint")
        )

    def _segment_point_counts(self, segments: WaveSegments) -> dict[str, int]:
        return {
            tr("label.segment.incident"): int(np.count_nonzero(np.isfinite(segments.incident))),
            tr("label.segment.reflected"): int(np.count_nonzero(np.isfinite(segments.reflected))),
            tr("label.segment.transmitted"): int(np.count_nonzero(np.isfinite(segments.transmitted))),
        }

    def _friendly_recalculate_error(self, exc: Exception) -> str:
        message = str(exc)
        if "Not enough points in one or more wave segments for alignment" in message:
            return (
                tr("error.segment_alignment_short")
                + tr("error.segment_alignment_window_hint")
            )
        return message

    def _log(self, message: str) -> None:
        self.log.append(message)

    def _error(self, message: str) -> None:
        self._log(tr("log.error", message=message))
        QMessageBox.critical(self, tr("dialog.error.title"), message)

    def _enabled_label(self, enabled: bool) -> str:
        return tr("value.enabled") if enabled else tr("value.disabled")

    def _code_label(self, prefix: str, value: str) -> str:
        key = f"{prefix}.{value}"
        translated = tr(key)
        return value if translated == key else translated

    def _localized_warnings(self, warnings: list[str]) -> list[str]:
        return [tr_message(warning) for warning in warnings] if warnings else [tr("summary.no_warnings")]

    def _reset_legend(self, plot: pg.PlotWidget) -> None:
        plot_item = plot.getPlotItem()
        if plot_item.legend is not None:
            plot_item.legend.clear()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


def _spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSingleStep(10 ** (-decimals) if decimals > 0 else 1.0)
    return spin


def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def _combo(values: list[str], current: str) -> QComboBox:
    combo = QComboBox()
    combo.addItems(values)
    combo.setCurrentText(current)
    return combo


def _mask(time_s: np.ndarray, signal: np.ndarray, window: PulseWindow) -> np.ndarray:
    masked = np.full_like(signal, np.nan, dtype=float)
    mask = (time_s >= window.start_s) & (time_s <= window.end_s)
    masked[mask] = signal[mask]
    return masked
