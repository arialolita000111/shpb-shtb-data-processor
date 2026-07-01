# 945 Steel Compression Batch Input / 945 钢压缩批处理输入

## English

This folder contains 945 steel impact-compression input files for batch processing.

Files:

- `945steel_compression_01.xls`
- `945steel_compression_02.xlsx`
- `945steel_compression_03.xlsx`
- `945steel_compression_04.xlsx`
- `945steel_compression_05.xlsx`
- `945steel_compression_06.xlsx`
- `945steel_compression_auto_windows_batch_config.json`

The JSON file stores reusable batch-processing parameters, including column mapping, pulse windows, bar/specimen settings, preprocessing, alignment, calculation, and output options.

Example command from the repository root:

```powershell
shpb-processor batch --config experimental_test_data/impact_compression/steel_945/steel_945_compression_batch_input/945steel_compression_auto_windows_batch_config.json --input experimental_test_data/impact_compression/steel_945/steel_945_compression_batch_input --output output/steel_945_demo
```

## 中文

本目录包含 945 钢冲击压缩实验的批处理输入文件。

文件包括：

- `945steel_compression_01.xls`
- `945steel_compression_02.xlsx`
- `945steel_compression_03.xlsx`
- `945steel_compression_04.xlsx`
- `945steel_compression_05.xlsx`
- `945steel_compression_06.xlsx`
- `945steel_compression_auto_windows_batch_config.json`

JSON 文件保存了可复用的批处理参数，包括列映射、脉冲窗口、杆件/试样参数、预处理、对齐、计算和输出设置。

在仓库根目录可运行：

```powershell
shpb-processor batch --config experimental_test_data/impact_compression/steel_945/steel_945_compression_batch_input/945steel_compression_auto_windows_batch_config.json --input experimental_test_data/impact_compression/steel_945/steel_945_compression_batch_input --output output/steel_945_demo
```
