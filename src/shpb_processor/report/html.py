from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from shpb_processor.i18n import tr, tr_message

if TYPE_CHECKING:
    from shpb_processor.processing import ProcessingBundle


def write_html_report(
    path: str | Path,
    bundle: ProcessingBundle,
    title: str | None = None,
    include_figures: bool = True,
) -> Path:
    title = title or tr("report.default_title")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = output.parent / "figures"
    figures: list[tuple[str, Path]] = []
    if include_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        figures = _write_figures(figures_dir, bundle)

    report = [
        "<!doctype html>",
        f"<html lang=\"{tr('report.html_lang')}\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:32px;line-height:1.45;color:#1f2933}",
        "h1,h2{color:#111827} table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
        "th,td{border:1px solid #d8dee9;padding:6px 8px;text-align:left;font-size:13px}",
        "th{background:#f3f6fb}.warning{color:#9a3412}.grade{font-weight:bold;font-size:18px}",
        "img{max-width:100%;border:1px solid #e5e7eb;margin:8px 0 20px}",
        "code{background:#f3f4f6;padding:1px 4px;border-radius:3px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        _summary_section(bundle),
        _quality_section(bundle),
        _figure_section(figures, output.parent),
        _processing_report_section(bundle),
        f"<h2>{html.escape(tr('report.section.machine_readable_quality'))}</h2>",
        f"<pre>{html.escape(_quality_json(bundle))}</pre>",
        "</body></html>",
    ]
    output.write_text("\n".join(report), encoding="utf-8")
    return output


def _write_figures(figures_dir: Path, bundle: ProcessingBundle) -> list[tuple[str, Path]]:
    figures: list[tuple[str, Path]] = []
    standard = bundle.standard_wave_separation
    processed = figures_dir / "processed_signals.svg"
    _write_svg_plot(
        processed,
        tr("report.figure.processed_signals"),
        tr("report.axis.time_us"),
        tr("report.axis.strain_microstrain"),
        [
            ("incident", bundle.processed.time_s * 1e6, bundle.processed.incident_strain * 1e6, "#1f77b4"),
            ("transmitted", bundle.processed.time_s * 1e6, bundle.processed.transmitted_strain * 1e6, "#2ca02c"),
        ],
    )
    figures.append((tr("report.figure.processed_signals"), processed))

    if standard is not None:
        wave_sep = figures_dir / "standard_wave_separation.svg"
        _write_svg_plot(
            wave_sep,
            tr("report.figure.standard_wave_separation"),
            tr("report.axis.time_us"),
            tr("report.axis.strain_microstrain"),
            [
                ("gauge1 signal", standard.time_s * 1e6, standard.gauge1_signal * 1e6, "#272727"),
                ("gauge1 right", standard.time_s * 1e6, standard.gauge1_right_going * 1e6, "#0F4D92"),
                ("gauge1 left", standard.time_s * 1e6, standard.gauge1_left_going * 1e6, "#B64342"),
                ("gauge2 signal", standard.time_s * 1e6, standard.gauge2_signal * 1e6, "#767676"),
                ("gauge2 right", standard.time_s * 1e6, standard.gauge2_right_going * 1e6, "#42949E"),
                ("gauge2 left", standard.time_s * 1e6, standard.gauge2_left_going * 1e6, "#9A4D8E"),
            ],
        )
        figures.append((tr("report.figure.standard_wave_separation"), wave_sep))

        stress = figures_dir / "stress_strain.svg"
        _write_svg_plot(
            stress,
            tr("report.figure.standard_wave_stress_strain"),
            tr("report.axis.engineering_strain"),
            tr("report.axis.engineering_stress_mpa"),
            [
                ("standard wave", standard.engineering_strain, standard.engineering_stress_pa / 1e6, "#272727"),
            ],
        )
        figures.append((tr("report.figure.standard_wave_stress_strain"), stress))

        force = figures_dir / "standard_wave_force_balance.svg"
        _write_svg_plot(
            force,
            tr("report.figure.standard_wave_forces"),
            tr("report.axis.time_us"),
            tr("report.axis.force_n"),
            [
                ("incident end", standard.time_s * 1e6, standard.specimen_force_from_incident_end_n, "#0F4D92"),
                ("transmitted end", standard.time_s * 1e6, standard.specimen_force_from_transmitted_end_n, "#B64342"),
            ],
        )
        figures.append((tr("report.figure.standard_wave_forces"), force))
        return figures

    aligned = figures_dir / "aligned_waves.svg"
    _write_svg_plot(
        aligned,
        tr("report.figure.aligned_waves"),
        tr("report.axis.aligned_time_us"),
        tr("report.axis.strain_microstrain"),
        [
            ("incident", bundle.aligned.time_s * 1e6, bundle.aligned.incident * 1e6, "#1f77b4"),
            ("reflected", bundle.aligned.time_s * 1e6, bundle.aligned.reflected * 1e6, "#d62728"),
            ("transmitted", bundle.aligned.time_s * 1e6, bundle.aligned.transmitted * 1e6, "#2ca02c"),
            ("tr - re", bundle.aligned.time_s * 1e6, (bundle.aligned.transmitted - bundle.aligned.reflected) * 1e6, "#ff9900"),
        ],
    )
    figures.append((tr("report.figure.aligned_waves"), aligned))

    if _dispersion_enabled(bundle) and bundle.uncorrected_segments is not None:
        dispersion = figures_dir / "dispersion_comparison.svg"
        before = bundle.uncorrected_segments
        after = bundle.segments
        _write_svg_plot(
            dispersion,
            tr("report.figure.dispersion_comparison"),
            tr("report.axis.time_us"),
            tr("report.axis.strain_microstrain"),
            [
                ("incident before", before.time_s * 1e6, before.incident * 1e6, "#1f77b4"),
                ("incident after", after.time_s * 1e6, after.incident * 1e6, "#1f77b4"),
                ("reflected before", before.time_s * 1e6, before.reflected * 1e6, "#d62728"),
                ("reflected after", after.time_s * 1e6, after.reflected * 1e6, "#d62728"),
                ("transmitted before", before.time_s * 1e6, before.transmitted * 1e6, "#2ca02c"),
                ("transmitted after", after.time_s * 1e6, after.transmitted * 1e6, "#2ca02c"),
            ],
        )
        figures.append((tr("report.figure.dispersion_comparison"), dispersion))

    stress = figures_dir / "stress_strain.svg"
    _write_svg_plot(
        stress,
        tr("report.figure.engineering_stress_strain"),
        tr("report.axis.engineering_strain"),
        tr("report.axis.engineering_stress_mpa"),
        [
            ("three-wave", bundle.three_wave.strain, bundle.three_wave.engineering_stress_pa / 1e6, "#7b3294"),
            ("two-wave", bundle.two_wave.strain, bundle.two_wave.engineering_stress_pa / 1e6, "#008837"),
        ],
    )
    figures.append((tr("report.figure.engineering_stress_strain"), stress))

    balance = figures_dir / "force_balance.svg"
    _write_svg_plot(
        balance,
        tr("report.figure.force_balance_error"),
        tr("report.axis.aligned_time_us"),
        tr("report.axis.relative_error"),
        [
            ("balance error", bundle.aligned.time_s * 1e6, bundle.three_wave.balance_error, "#6b7280"),
        ],
    )
    figures.append((tr("report.figure.force_balance_error"), balance))
    return figures


def _summary_section(bundle: ProcessingBundle) -> str:
    standard = bundle.standard_wave_separation
    if standard is not None:
        rows = [
            (tr("report.summary.processing_mode"), tr("report.mode.standard_wave")),
            (tr("report.summary.quality_grade"), f"{bundle.quality.grade} ({tr('report.summary.input_auxiliary_diagnostics')})"),
            (tr("report.summary.review_status"), _status_label(bundle.quality.status)),
            (tr("report.summary.dispersion_correction"), _enabled_label(_dispersion_enabled(bundle))),
            (tr("report.summary.standard_peak_stress_mpa"), _fmt(standard.summary.get("peak_stress_mpa"))),
            (tr("report.summary.standard_max_strain"), _fmt(standard.summary.get("max_strain"))),
            (tr("report.summary.standard_max_abs_strain_rate"), _fmt(standard.summary.get("max_abs_strain_rate_s^-1"))),
            (tr("report.summary.force_closure_max_n"), _fmt(standard.metadata.get("force_closure_max_abs_n"))),
            (tr("report.summary.incident_tau_us"), _fmt(float(standard.summary.get("standard_wave_incident_tau_s", 0.0)) * 1e6)),
            (tr("report.summary.transmitted_tau_us"), _fmt(float(standard.summary.get("standard_wave_transmitted_tau_s", 0.0)) * 1e6)),
            (tr("report.summary.free_end_tau_us"), _fmt(float(standard.summary.get("standard_wave_free_end_tau_s", 0.0)) * 1e6)),
        ]
    else:
        rows = [
            (tr("report.summary.processing_mode"), tr("report.mode.legacy_three_two_wave")),
            (tr("report.summary.quality_grade"), bundle.quality.grade),
            (tr("report.summary.review_status"), _status_label(bundle.quality.status)),
            (tr("report.summary.dispersion_correction"), _enabled_label(_dispersion_enabled(bundle))),
            (tr("report.summary.three_wave_peak_stress_mpa"), _fmt(bundle.three_wave.summary.get("peak_stress_mpa"))),
            (tr("report.summary.two_wave_peak_stress_mpa"), _fmt(bundle.two_wave.summary.get("peak_stress_mpa"))),
            (tr("report.summary.mean_force_balance_error"), _fmt(bundle.three_wave.summary.get("mean_balance_error"))),
            (tr("report.summary.max_strain"), _fmt(bundle.three_wave.summary.get("max_strain"))),
            (tr("report.summary.max_abs_strain_rate"), _fmt(bundle.three_wave.summary.get("max_abs_strain_rate_s^-1"))),
        ]
    return f"<h2>{html.escape(tr('report.section.summary'))}</h2>" + _table(
        [tr("report.table.item"), tr("report.table.value")],
        rows,
    )


def _dispersion_enabled(bundle: ProcessingBundle) -> bool:
    return bool(bundle.segments.metadata.get("dispersion_correction_enabled", False))


def _enabled_label(enabled: bool) -> str:
    return tr("value.enabled") if enabled else tr("value.disabled")


def _status_label(status: str) -> str:
    key = f"report.status.{status}"
    translated = tr(key)
    return status if translated == key else translated


def _quality_section(bundle: ProcessingBundle) -> str:
    warnings = [tr_message(warning) for warning in bundle.quality.warnings] or [tr("summary.no_warnings")]
    rows = [(warning,) for warning in warnings]
    metrics = [(key, _fmt(value)) for key, value in sorted(bundle.quality.metrics.items())]
    return (
        f"<h2>{html.escape(tr('report.section.quality_assessment'))}</h2>"
        f"<p class=\"grade\">{html.escape(tr('report.quality.grade_status', grade=bundle.quality.grade, status=_status_label(bundle.quality.status)))}</p>"
        + _table([tr("report.table.warning")], rows, css_class="warning")
        + _table([tr("report.table.metric"), tr("report.table.value")], metrics)
    )


def _figure_section(figures: list[tuple[str, Path]], report_dir: Path) -> str:
    if not figures:
        return ""
    parts = [f"<h2>{html.escape(tr('report.section.figures'))}</h2>"]
    for title, path in figures:
        rel = path.relative_to(report_dir).as_posix()
        parts.append(f"<h3>{html.escape(title)}</h3>")
        parts.append(f"<img src=\"{html.escape(rel)}\" alt=\"{html.escape(title)}\">")
    return "\n".join(parts)


def _processing_report_section(bundle: ProcessingBundle) -> str:
    rows = [(str(row.get("item", "")), str(row.get("value", ""))) for row in bundle.report_rows]
    return f"<h2>{html.escape(tr('report.section.processing_provenance'))}</h2>" + _table(
        [tr("report.table.item"), tr("report.table.value")],
        rows,
    )


def _write_svg_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Iterable[tuple[str, np.ndarray, np.ndarray, str]],
) -> None:
    width, height = 920, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 22, 42, 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    cleaned = []
    for name, x, y, color in series:
        x_arr, y_arr = _finite_xy(x, y)
        if len(x_arr) > 1200:
            step = int(np.ceil(len(x_arr) / 1200))
            x_arr = x_arr[::step]
            y_arr = y_arr[::step]
        if len(x_arr):
            cleaned.append((name, x_arr, y_arr, color))

    if not cleaned:
        path.write_text(_empty_svg(width, height, title), encoding="utf-8")
        return

    x_all = np.concatenate([item[1] for item in cleaned])
    y_all = np.concatenate([item[2] for item in cleaned])
    xmin, xmax = _range(x_all)
    ymin, ymax = _range(y_all)

    def sx(values: np.ndarray) -> np.ndarray:
        return margin_left + (values - xmin) / max(xmax - xmin, 1e-30) * plot_w

    def sy(values: np.ndarray) -> np.ndarray:
        return margin_top + (ymax - values) / max(ymax - ymin, 1e-30) * plot_h

    parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{width / 2}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-family=\"Arial\">{html.escape(title)}</text>",
        f"<rect x=\"{margin_left}\" y=\"{margin_top}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"#fbfcfe\" stroke=\"#d1d5db\"/>",
    ]
    for frac in np.linspace(0, 1, 5):
        x = margin_left + frac * plot_w
        y = margin_top + frac * plot_h
        parts.append(f"<line x1=\"{x:.1f}\" y1=\"{margin_top}\" x2=\"{x:.1f}\" y2=\"{margin_top + plot_h}\" stroke=\"#eef2f7\"/>")
        parts.append(f"<line x1=\"{margin_left}\" y1=\"{y:.1f}\" x2=\"{margin_left + plot_w}\" y2=\"{y:.1f}\" stroke=\"#eef2f7\"/>")
    for name, x_arr, y_arr, color in cleaned:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx(x_arr), sy(y_arr)))
        dash = " stroke-dasharray=\"6 4\"" if "before" in name else ""
        parts.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\"{dash} points=\"{points}\"/>")
    parts.append(f"<text x=\"{width / 2}\" y=\"{height - 16}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"Arial\">{html.escape(xlabel)}</text>")
    parts.append(f"<text transform=\"translate(18,{height / 2}) rotate(-90)\" text-anchor=\"middle\" font-size=\"13\" font-family=\"Arial\">{html.escape(ylabel)}</text>")
    parts.append(f"<text x=\"{margin_left}\" y=\"{height - 36}\" font-size=\"11\" font-family=\"Arial\">x: {_fmt(xmin)} to {_fmt(xmax)}</text>")
    parts.append(f"<text x=\"{margin_left + 220}\" y=\"{height - 36}\" font-size=\"11\" font-family=\"Arial\">y: {_fmt(ymin)} to {_fmt(ymax)}</text>")
    legend_x = margin_left + 14
    legend_y = margin_top + 20
    for index, (name, _, _, color) in enumerate(cleaned):
        y = legend_y + index * 18
        dash = " stroke-dasharray=\"6 4\"" if "before" in name else ""
        parts.append(f"<line x1=\"{legend_x}\" y1=\"{y}\" x2=\"{legend_x + 24}\" y2=\"{y}\" stroke=\"{color}\" stroke-width=\"2\"{dash}/>")
        parts.append(f"<text x=\"{legend_x + 30}\" y=\"{y + 4}\" font-size=\"12\" font-family=\"Arial\">{html.escape(name)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _range(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return 0.0, 1.0
    if abs(maximum - minimum) < 1e-30:
        pad = max(abs(maximum) * 0.1, 1.0)
        return minimum - pad, maximum + pad
    pad = 0.05 * (maximum - minimum)
    return minimum - pad, maximum + pad


def _table(headers: list[str], rows: list[tuple[object, ...]], css_class: str = "") -> str:
    class_attr = f" class=\"{css_class}\"" if css_class else ""
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _quality_json(bundle: ProcessingBundle) -> str:
    return json.dumps(bundle.quality.model_dump(mode="json"), ensure_ascii=False, indent=2)


def _fmt(value: object) -> str:
    if isinstance(value, (int, float, np.floating)):
        if np.isfinite(float(value)):
            return f"{float(value):.6g}"
    return "" if value is None else str(value)


def _empty_svg(width: int, height: int, title: str) -> str:
    return (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">"
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
        f"<text x=\"{width / 2}\" y=\"{height / 2}\" text-anchor=\"middle\" font-family=\"Arial\">"
        f"{html.escape(title)}: no finite data</text></svg>"
    )
