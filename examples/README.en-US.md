# Examples

This folder contains synthetic and local example data for testing and demonstrating the processing workflow.

Synthetic files:

- `synthetic_shpb_ideal`
- `synthetic_shpb_noisy`
- `synthetic_shpb_unbalanced`
- `synthetic_shpb_overlap`

Dispersion-correction examples:

- `synthetic_shpb_dispersion_long_clean`
- `synthetic_shpb_dispersion_long_noisy`
- `synthetic_shpb_dispersion_large_bar`

Each dispersion example has companion configuration or metadata files. These examples validate software behavior and are not material-property measurements.

Run a demonstration batch:

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

Each sample output folder can include HTML/PDF reports, SVG figures, CSV tables, an Excel workbook, and `quality_report.json`.
