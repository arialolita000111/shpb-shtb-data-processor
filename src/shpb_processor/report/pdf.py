from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from shpb_processor.processing import ProcessingBundle


@dataclass
class _PdfImage:
    data: bytes
    width: int
    height: int


@dataclass
class _PdfPage:
    title: str
    lines: list[str]
    image: _PdfImage | None = None


def write_pdf_report(
    path: str | Path,
    bundle: ProcessingBundle,
    title: str = "SHPB/SHTB processing report",
) -> Path:
    """Write an illustrated PDF report with mode-aware summaries and plots."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = _report_pages(bundle, title)
    _write_pdf(output, pages)
    return output


def _report_pages(bundle: ProcessingBundle, title: str) -> list[_PdfPage]:
    pages = [_PdfPage(_ascii(title), _summary_lines(bundle))]
    pages.extend(_figure_pages(bundle))
    pages.extend(_provenance_pages(bundle))
    return pages


def _summary_lines(bundle: ProcessingBundle) -> list[str]:
    standard = bundle.standard_wave_separation
    dispersion_enabled = bool(bundle.segments.metadata.get("dispersion_correction_enabled", False))
    lines = [
        "Summary",
        f"Processing mode: {'standard SHPB wave separation' if standard is not None else 'legacy three-wave/two-wave'}",
        f"Quality grade: {bundle.quality.grade}",
        f"Review status: {bundle.quality.status}",
        f"Dispersion correction: {'enabled' if dispersion_enabled else 'disabled'}",
    ]
    if standard is not None:
        lines.extend(
            [
                f"Standard wave peak stress (MPa): {_fmt(standard.summary.get('peak_stress_mpa'))}",
                f"Standard wave max strain: {_fmt(standard.summary.get('max_strain'))}",
                f"Standard wave max abs strain rate (s^-1): {_fmt(standard.summary.get('max_abs_strain_rate_s^-1'))}",
                f"Force closure max abs (N): {_fmt(standard.metadata.get('force_closure_max_abs_n'))}",
                f"Incident propagation time (us): {_fmt(float(standard.summary.get('standard_wave_incident_tau_s', 0.0)) * 1e6)}",
                f"Transmitted propagation time (us): {_fmt(float(standard.summary.get('standard_wave_transmitted_tau_s', 0.0)) * 1e6)}",
                f"Free-end propagation time (us): {_fmt(float(standard.summary.get('standard_wave_free_end_tau_s', 0.0)) * 1e6)}",
                "",
                "Generated primary files",
                "standard_wave_separation.csv",
                "standard_wave_results.csv",
                "results.csv (standard_wave)",
                "summary.csv (standard_wave)",
            ]
        )
    else:
        lines.extend(
            [
                f"Three-wave peak stress (MPa): {_fmt(bundle.three_wave.summary.get('peak_stress_mpa'))}",
                f"Two-wave peak stress (MPa): {_fmt(bundle.two_wave.summary.get('peak_stress_mpa'))}",
                f"Mean force balance error: {_fmt(bundle.three_wave.summary.get('mean_balance_error'))}",
                f"Max strain: {_fmt(bundle.three_wave.summary.get('max_strain'))}",
                f"Max abs strain rate (s^-1): {_fmt(bundle.three_wave.summary.get('max_abs_strain_rate_s^-1'))}",
                "",
                "Generated primary files",
                "processed_signals.csv",
                "results.csv (three_wave, two_wave)",
                "summary.csv",
            ]
        )
    lines.extend(["report.html", "report.pdf", "result.xlsx", "quality_report.json", "", "Warnings"])
    lines.extend(bundle.quality.warnings or ["No warnings."])
    if standard is not None and standard.warnings:
        lines.extend(["", "Standard wave warnings", *standard.warnings])
    return [_ascii(line) for line in lines]


def _figure_pages(bundle: ProcessingBundle) -> list[_PdfPage]:
    pages = [
        _PdfPage(
            "Processed strain signals",
            ["Raw input after unit conversion and selected preprocessing."],
            _plot_image(
                "Processed strain signals",
                "time (us)",
                "strain (microstrain)",
                [
                    ("incident", bundle.processed.time_s * 1e6, bundle.processed.incident_strain * 1e6, "#0F4D92"),
                    ("transmitted", bundle.processed.time_s * 1e6, bundle.processed.transmitted_strain * 1e6, "#8BCF8B"),
                ],
            ),
        )
    ]
    standard = bundle.standard_wave_separation
    if standard is not None:
        pages.append(
            _PdfPage(
                "Standard SHPB wave separation",
                ["Separated right- and left-going strain waves at gauge locations."],
                _plot_image(
                    "Standard SHPB wave separation",
                    "time (us)",
                    "strain (microstrain)",
                    [
                        ("gauge1 signal", standard.time_s * 1e6, standard.gauge1_signal * 1e6, "#272727"),
                        ("gauge1 right", standard.time_s * 1e6, standard.gauge1_right_going * 1e6, "#0F4D92"),
                        ("gauge1 left", standard.time_s * 1e6, standard.gauge1_left_going * 1e6, "#B64342"),
                        ("gauge2 signal", standard.time_s * 1e6, standard.gauge2_signal * 1e6, "#767676"),
                        ("gauge2 right", standard.time_s * 1e6, standard.gauge2_right_going * 1e6, "#42949E"),
                        ("gauge2 left", standard.time_s * 1e6, standard.gauge2_left_going * 1e6, "#9A4D8E"),
                    ],
                ),
            )
        )
        pages.append(
            _PdfPage(
                "Standard wave engineering stress-strain",
                ["Primary result in standard wave separation mode."],
                _plot_image(
                    "Standard wave engineering stress-strain",
                    "engineering strain",
                    "engineering stress (MPa)",
                    [
                        ("standard wave", standard.engineering_strain, standard.engineering_stress_pa / 1e6, "#272727"),
                    ],
                ),
            )
        )
        pages.append(
            _PdfPage(
                "Standard wave specimen-end forces",
                ["Incident-end and transmitted-end specimen forces reconstructed from separated waves."],
                _plot_image(
                    "Standard wave specimen-end forces",
                    "time (us)",
                    "force (N)",
                    [
                        ("incident end", standard.time_s * 1e6, standard.specimen_force_from_incident_end_n, "#0F4D92"),
                        ("transmitted end", standard.time_s * 1e6, standard.specimen_force_from_transmitted_end_n, "#B64342"),
                    ],
                ),
            )
        )
        return pages

    pages.append(
        _PdfPage(
            "Aligned waves",
            ["Legacy pulse windows after propagation/alignment corrections."],
            _plot_image(
                "Aligned waves",
                "aligned time (us)",
                "strain (microstrain)",
                [
                    ("incident", bundle.aligned.time_s * 1e6, bundle.aligned.incident * 1e6, "#0F4D92"),
                    ("reflected", bundle.aligned.time_s * 1e6, bundle.aligned.reflected * 1e6, "#B64342"),
                    ("transmitted", bundle.aligned.time_s * 1e6, bundle.aligned.transmitted * 1e6, "#8BCF8B"),
                    ("tr - re", bundle.aligned.time_s * 1e6, (bundle.aligned.transmitted - bundle.aligned.reflected) * 1e6, "#42949E"),
                ],
            ),
        )
    )
    if bool(bundle.segments.metadata.get("dispersion_correction_enabled", False)) and bundle.uncorrected_segments is not None:
        before = bundle.uncorrected_segments
        after = bundle.segments
        pages.append(
            _PdfPage(
                "Dispersion correction comparison",
                ["Dashed-like before/after distinction is represented by paired labels."],
                _plot_image(
                    "Dispersion correction comparison",
                    "time (us)",
                    "strain (microstrain)",
                    [
                        ("incident before", before.time_s * 1e6, before.incident * 1e6, "#0F4D92"),
                        ("incident after", after.time_s * 1e6, after.incident * 1e6, "#3775BA"),
                        ("reflected before", before.time_s * 1e6, before.reflected * 1e6, "#B64342"),
                        ("reflected after", after.time_s * 1e6, after.reflected * 1e6, "#E9A6A1"),
                        ("transmitted before", before.time_s * 1e6, before.transmitted * 1e6, "#8BCF8B"),
                        ("transmitted after", after.time_s * 1e6, after.transmitted * 1e6, "#42949E"),
                    ],
                ),
            )
        )
    pages.append(
        _PdfPage(
            "Engineering stress-strain",
            ["Primary legacy comparison between three-wave and two-wave calculations."],
            _plot_image(
                "Engineering stress-strain",
                "engineering strain",
                "engineering stress (MPa)",
                [
                    ("three-wave", bundle.three_wave.strain, bundle.three_wave.engineering_stress_pa / 1e6, "#9A4D8E"),
                    ("two-wave", bundle.two_wave.strain, bundle.two_wave.engineering_stress_pa / 1e6, "#42949E"),
                ],
            ),
        )
    )
    pages.append(
        _PdfPage(
            "Force balance error",
            ["Relative force balance error used by legacy three-wave diagnostics."],
            _plot_image(
                "Force balance error",
                "aligned time (us)",
                "relative error",
                [("balance error", bundle.aligned.time_s * 1e6, bundle.three_wave.balance_error, "#4D4D4D")],
            ),
        )
    )
    return pages


def _provenance_pages(bundle: ProcessingBundle) -> list[_PdfPage]:
    rows = [f"{row.get('item', '')}: {row.get('value', '')}" for row in bundle.report_rows[:120]]
    if not rows:
        return []
    chunks = [rows[index : index + 42] for index in range(0, len(rows), 42)]
    return [_PdfPage("Selected processing provenance", [_ascii(line) for line in chunk]) for chunk in chunks]


def _plot_image(
    title: str,
    xlabel: str,
    ylabel: str,
    series: Iterable[tuple[str, np.ndarray, np.ndarray, str]],
) -> _PdfImage:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("Illustrated PDF reports require Pillow. Install project dependencies first.") from exc

    width, height = 1100, 520
    margin_left, margin_right, margin_top, margin_bottom = 92, 34, 58, 76
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(ImageFont, 18)
    small = _font(ImageFont, 14)
    title_font = _font(ImageFont, 24)

    cleaned = []
    for name, x, y, color in series:
        x_arr, y_arr = _finite_xy(x, y)
        if len(x_arr) > 1500:
            step = int(np.ceil(len(x_arr) / 1500))
            x_arr = x_arr[::step]
            y_arr = y_arr[::step]
        if len(x_arr):
            cleaned.append((name, x_arr, y_arr, color))

    draw.text((width / 2, 18), _ascii(title), fill="#111827", font=title_font, anchor="ma")
    draw.rectangle([margin_left, margin_top, margin_left + plot_w, margin_top + plot_h], fill="#fbfcfe", outline="#d1d5db")
    for index in range(5):
        frac = index / 4
        x = margin_left + frac * plot_w
        y = margin_top + frac * plot_h
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill="#eef2f7", width=1)
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill="#eef2f7", width=1)

    if cleaned:
        x_all = np.concatenate([item[1] for item in cleaned])
        y_all = np.concatenate([item[2] for item in cleaned])
        xmin, xmax = _range(x_all)
        ymin, ymax = _range(y_all)

        def sx(values: np.ndarray) -> np.ndarray:
            return margin_left + (values - xmin) / max(xmax - xmin, 1e-30) * plot_w

        def sy(values: np.ndarray) -> np.ndarray:
            return margin_top + (ymax - values) / max(ymax - ymin, 1e-30) * plot_h

        for name, x_arr, y_arr, color in cleaned:
            points = list(zip(sx(x_arr), sy(y_arr)))
            if len(points) >= 2:
                draw.line(points, fill=color, width=3, joint="curve")
        draw.text((margin_left, height - 48), f"x: {_fmt(xmin)} to {_fmt(xmax)}", fill="#4b5563", font=small)
        draw.text((margin_left + 300, height - 48), f"y: {_fmt(ymin)} to {_fmt(ymax)}", fill="#4b5563", font=small)
    else:
        draw.text((width / 2, height / 2), "No finite data", fill="#4b5563", font=font, anchor="mm")

    draw.text((width / 2, height - 22), _ascii(xlabel), fill="#111827", font=font, anchor="ma")
    draw.text((18, height / 2), _ascii(ylabel), fill="#111827", font=font, anchor="mm")
    legend_x, legend_y = margin_left + 18, margin_top + 18
    for index, (name, _x, _y, color) in enumerate(cleaned[:10]):
        y = legend_y + index * 21
        draw.line([(legend_x, y), (legend_x + 30, y)], fill=color, width=4)
        draw.text((legend_x + 40, y - 8), _ascii(name), fill="#111827", font=small)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    return _PdfImage(data=buffer.getvalue(), width=width, height=height)


def _font(image_font_module, size: int):
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return image_font_module.truetype(name, size)
        except OSError:
            continue
    return image_font_module.load_default()


def _write_pdf(path: Path, pages: list[_PdfPage]) -> None:
    page_width = 595.0
    page_height = 842.0
    margin_x = 54.0
    objects: list[bytes] = []
    page_refs: list[int] = []

    font_ref = 1
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page in pages:
        image_ref = None
        if page.image is not None:
            image_ref = len(objects) + 1
            objects.append(_image_object(page.image))
        content_ref = len(objects) + 1
        page_ref = len(objects) + 2
        stream = _page_stream(page, page_width, page_height, margin_x)
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
        resources = f"/Font << /F1 {font_ref} 0 R >>"
        if image_ref is not None:
            resources += f" /XObject << /Im1 {image_ref} 0 R >>"
        objects.append(
            (
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {page_width:g} {page_height:g}] "
                f"/Resources << {resources} >> /Contents {content_ref} 0 R >>"
            ).encode("ascii")
        )
        page_refs.append(page_ref)

    pages_ref = len(objects) + 1
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode("ascii"))
    catalog_ref = len(objects) + 1
    objects.append(f"<< /Type /Catalog /Pages {pages_ref} 0 R >>".encode("ascii"))

    patched = [raw.replace(b"/Parent 0 0 R", f"/Parent {pages_ref} 0 R".encode("ascii")) for raw in objects]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, raw in enumerate(patched, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(raw)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(patched) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(patched) + 1} /Root {catalog_ref} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(payload))


def _image_object(image: _PdfImage) -> bytes:
    header = (
        f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(image.data)} >>\n"
        "stream\n"
    ).encode("ascii")
    return header + image.data + b"\nendstream"


def _page_stream(page: _PdfPage, page_width: float, page_height: float, margin_x: float) -> bytes:
    parts: list[str] = []
    y = page_height - 56.0
    parts.append(_text_command(_ascii(page.title), margin_x, y, 16))
    y -= 30.0
    for line in page.lines[:34]:
        parts.append(_text_command(_ascii(line)[:110], margin_x, y, 10))
        y -= 14.0
    if page.image is not None:
        max_w = page_width - 2 * margin_x
        max_h = 360.0
        scale = min(max_w / page.image.width, max_h / page.image.height)
        draw_w = page.image.width * scale
        draw_h = page.image.height * scale
        x = (page_width - draw_w) / 2.0
        image_y = max(70.0, y - draw_h - 12.0)
        parts.append(f"q {draw_w:.3f} 0 0 {draw_h:.3f} {x:.3f} {image_y:.3f} cm /Im1 Do Q")
    return "\n".join(parts).encode("ascii")


def _text_command(text: str, x: float, y: float, size: int) -> str:
    return f"BT /F1 {size} Tf {x:.3f} {y:.3f} Td ({_escape_pdf_text(text)}) Tj ET"


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


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _ascii(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def _fmt(value: object) -> str:
    if isinstance(value, (int, float, np.floating)):
        if np.isfinite(float(value)):
            return f"{float(value):.6g}"
    return "" if value is None else str(value)
