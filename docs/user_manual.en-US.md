# User Manual

## 1. Installation

Install Python 3.10 or later, then install the software from the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Run the test suite after installation:

```powershell
python -m pytest
```

## 2. GUI Workflow

Launch the desktop interface:

```powershell
python -m shpb_processor
```

Typical workflow:

1. Open a CSV, TXT, DAT, XLSX, or XLS data file.
2. Confirm or adjust the detected time, incident-bar, and transmitted-bar columns.
3. Set acquisition, bar, specimen, preprocessing, alignment, and calculation parameters.
4. Run automatic processing.
5. Review pulse windows, alignment, force balance, and warning messages.
6. Export reports and result files.

## 3. Batch Workflow

Batch processing uses a JSON or YAML configuration file. Start with:

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

The output directory contains per-sample result folders and a batch summary.

## 4. Output Files

Depending on configuration, the software can export:

- `processed_signals.csv`
- `results.csv`
- `summary.csv`
- `processing_report.csv`
- `quality_report.json`
- `result.xlsx`
- `report.html`
- `report.pdf`
- `figures/*.svg`
- `config.json`

## 5. Experimental Data

Cleaned experimental datasets are in `experimental_test_data/`. Names are in English. Experiment details are provided in both Chinese and English.

## 6. Notes

Warnings and quality grades are diagnostic aids. Experimental judgment is still required when selecting pulse windows, interpreting poor force balance, or comparing material responses.
