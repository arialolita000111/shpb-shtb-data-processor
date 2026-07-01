from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from shpb_processor.models.base import ProcessorModel


class TableLoadOptions(ProcessorModel):
    sheet_name: str | int | None = 0
    header: int | None | str = "auto"
    skiprows: int = Field(default=0, ge=0)
    delimiter: str | None = None
    n_preview_rows: int = Field(default=5000, gt=0)


class TableLoadResult(ProcessorModel):
    dataframe: pd.DataFrame
    path: str
    sheet_name: str | int | None = None
    columns: list[str]
    stats: dict[str, dict[str, float]]
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_table(path: str | Path, options: TableLoadOptions | None = None) -> TableLoadResult:
    options = options or TableLoadOptions()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise ValueError(f"File is empty: {source}")

    suffix = source.suffix.lower()
    if suffix in {".csv", ".txt", ".dat"}:
        dataframe = _read_delimited(source, options)
        sheet_name = None
    elif suffix in {".xlsx", ".xls"}:
        dataframe = _read_excel(source, options)
        sheet_name = options.sheet_name
    else:
        raise ValueError(f"Unsupported file format: {source.suffix}")

    dataframe = _normalize_columns(dataframe)
    if dataframe.empty:
        raise ValueError(f"No tabular data found in {source}")

    return TableLoadResult(
        dataframe=dataframe,
        path=str(source),
        sheet_name=sheet_name,
        columns=list(dataframe.columns),
        stats=_numeric_stats(dataframe),
        metadata={"rows": int(len(dataframe)), "columns": int(len(dataframe.columns))},
    )


def _read_delimited(source: Path, options: TableLoadOptions) -> pd.DataFrame:
    header = _header_arg(options)
    if options.delimiter:
        return _read_csv_with_encoding_fallback(
            source,
            sep=options.delimiter,
            header=header,
            skiprows=options.skiprows,
        )

    if header == "infer":
        header = _detect_header(source, options.skiprows)
    return _read_csv_with_encoding_fallback(
        source,
        sep=None,
        engine="python",
        header=header,
        skiprows=options.skiprows,
        comment="#",
    )


def _read_excel(source: Path, options: TableLoadOptions) -> pd.DataFrame:
    header = _excel_header_arg(source, options)
    dataframe = pd.read_excel(
        source,
        sheet_name=options.sheet_name,
        header=header,
        skiprows=options.skiprows,
    )
    if isinstance(dataframe, dict):
        if not dataframe:
            raise ValueError(f"No sheets found in {source}")
        return next(iter(dataframe.values()))
    return dataframe


def _header_arg(options: TableLoadOptions) -> int | None | str:
    if options.header == "auto":
        return "infer"
    return options.header


def _excel_header_arg(source: Path, options: TableLoadOptions) -> int | list[int] | None:
    if options.header != "auto":
        if options.header == "infer":
            return 0
        return options.header

    preview = pd.read_excel(
        source,
        sheet_name=options.sheet_name,
        header=None,
        skiprows=options.skiprows,
        nrows=1,
    )
    if isinstance(preview, dict):
        preview = next(iter(preview.values())) if preview else pd.DataFrame()
    if preview.empty:
        return 0

    values = [value for value in preview.iloc[0].tolist() if pd.notna(value)]
    if not values:
        return 0

    numeric_count = 0
    for value in values:
        try:
            float(value)
            numeric_count += 1
        except (TypeError, ValueError):
            pass
    return None if numeric_count == len(values) else 0


def _detect_header(source: Path, skiprows: int) -> int | None:
    with source.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for _ in range(skiprows):
            next(handle, None)
        first_line = next(handle, "")
    tokens = [token.strip() for token in first_line.replace("\t", ",").replace(";", ",").split(",") if token.strip()]
    if not tokens:
        return "infer"
    numeric = 0
    for token in tokens:
        try:
            float(token)
            numeric += 1
        except ValueError:
            pass
    return None if numeric == len(tokens) else 0


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    normalized: list[str] = []
    used: dict[str, int] = {}
    for index, column in enumerate(df.columns):
        name = str(column).replace("\ufeff", "").strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"col_{index + 1}"
        count = used.get(name, 0)
        used[name] = count + 1
        normalized.append(name if count == 0 else f"{name}_{count + 1}")
    df.columns = normalized
    return df


def _numeric_stats(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in dataframe.columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.notna().any():
            stats[column] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0)),
            }
    return stats


def _read_csv_with_encoding_fallback(source: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(source, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(source, **kwargs)
