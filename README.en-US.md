# SHPB/SHTB Data Processor

SHPB/SHTB Data Processor is a reproducible Python toolkit for Split Hopkinson Pressure Bar and Split Hopkinson Tension Bar data processing. It provides a desktop GUI, a command-line batch workflow, and a Python API for converting raw strain-gauge signals into stress, strain, strain-rate, force-balance diagnostics, quality records, and report files.

## Main Features

- Import CSV, TXT, DAT, XLSX, and XLS files.
- Detect time, incident-bar, and transmitted-bar columns automatically.
- Apply baseline correction, filtering, pulse detection, and wave alignment.
- Compute two-wave, three-wave, and optional standard-wave-separation results.
- Optionally apply frequency-domain stress-wave dispersion correction.
- Export CSV tables, Excel workbooks, JSON quality records, SVG figures, HTML reports, and PDF reports.
- Save GUI-reviewed parameters as reusable batch-processing templates.

## Installation

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Launch the GUI

```powershell
python -m shpb_processor
```

On Windows, you can also run:

```powershell
run_app.bat
```

## Batch Processing

Use `examples/batch_config.json` as a starting configuration:

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

Each processed sample folder may contain processed signals, result tables, quality records, an Excel workbook, figures, and HTML/PDF reports.

## Experimental Data

The cleaned experimental datasets are stored in `experimental_test_data/`. File and folder names are in English. Experiment details are provided as paired Chinese and English text files named `experiment_details_zh-CN.txt` and `experiment_details_en-US.txt`.

## Citation

Please cite the software using the metadata in `CITATION.cff`. Update release-specific identifiers before making a formal archived release.
