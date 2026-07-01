# Changelog / 更新日志

## Unreleased / 未发布

- Cleaned generated caches, local build artifacts, generated output folders, and non-release materials from the public release tree.
- Added bilingual README and user-manual files.
- Renamed the experimental data tree and data files to English names.
- Added paired Chinese and English experiment-detail files for the experimental datasets.
- Added `examples/batch_config.json` as the standard batch-processing example configuration.

- 清理缓存、本地构建产物、生成输出目录以及非发布材料。
- 补充中英文 README 和使用手册。
- 将实验数据目录和数据文件改为英文命名。
- 为实验数据补充中文和英文两版实验详情文件。
- 新增 `examples/batch_config.json` 作为标准批处理示例配置。

## 1.0.0 - 2026-05-21

- Added project metadata, license, citation, contribution, and user documentation.
- Added JSON/YAML workspace configuration for reproducible processing.
- Added command-line batch processing with per-sample output folders.
- Added optional frequency-domain stress-wave dispersion correction.
- Added HTML reports and SVG figures for processed signals, aligned waves, stress-strain response, and force balance.
- Added GUI report-package export and compact PDF report generation.
- Expanded quality assessment with pass/review/fail status, wave overlap, transmitted-pulse strength, alignment confidence, and dispersion metadata.
- Preserved the desktop GUI and existing Python API while adding stable batch interfaces.

## 0.1.0

- Initial desktop workflow for SHPB/SHTB data import, preprocessing, pulse detection, alignment, two-wave and three-wave calculation, preview, and Excel export.
