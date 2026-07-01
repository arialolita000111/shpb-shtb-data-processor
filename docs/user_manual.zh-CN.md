# 使用手册

## 1. 安装

请先安装 Python 3.10 或更高版本，然后在仓库根目录运行：

```powershell
python -m pip install -e ".[dev]"
```

安装后建议运行测试：

```powershell
python -m pytest
```

## 2. 图形界面流程

启动桌面界面：

```powershell
python -m shpb_processor
```

典型流程：

1. 打开 CSV、TXT、DAT、XLSX 或 XLS 数据文件。
2. 确认或调整自动识别出的时间列、入射杆应变列和透射杆应变列。
3. 设置采集参数、杆件参数、试样参数、预处理、对齐和计算参数。
4. 运行自动处理。
5. 复核脉冲窗口、波形对齐、力平衡和警告信息。
6. 导出报告和结果文件。

## 3. 批处理流程

批处理使用 JSON 或 YAML 配置文件。可先运行：

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

输出目录中会包含每个样本的结果文件夹和批处理汇总文件。

## 4. 输出文件

根据配置，软件可以导出：

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

## 5. 实验数据

整理后的实验测试数据位于 `experimental_test_data/`。目录和数据文件采用英文命名，实验详情同时提供中文和英文版本。

## 6. 注意事项

警告信息和质量等级用于辅助判断。选择脉冲窗口、解释力平衡较差的结果、比较材料响应时，仍需要结合实验经验进行复核。
