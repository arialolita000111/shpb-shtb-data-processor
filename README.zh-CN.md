# 霍普金森压杆/拉杆实验数据处理软件

本软件是用于 SHPB/SHTB 实验数据处理的 Python 工具包，支持桌面图形界面、命令行批处理和 Python API。它可以将原始应变片信号转换为应力、应变、应变率、力平衡诊断、质量记录和报告文件。

## 主要功能

- 导入 CSV、TXT、DAT、XLSX 和 XLS 文件。
- 自动识别时间列、入射杆应变列和透射杆应变列。
- 进行基线修正、滤波、脉冲识别和波形对齐。
- 计算二波法、三波法以及可选的标准波形分离结果。
- 可选启用频域应力波色散修正。
- 导出 CSV 表格、Excel 工作簿、JSON 质量记录、SVG 图、HTML 报告和 PDF 报告。
- 将 GUI 中人工复核后的参数保存为可复用的批处理模板。

## 安装

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## 启动图形界面

```powershell
python -m shpb_processor
```

在 Windows 中也可以运行：

```powershell
run_app.bat
```

## 批处理

可从 `examples/batch_config.json` 开始配置：

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

每个处理后的样本目录可包含处理后信号、结果表、质量记录、Excel 工作簿、图件以及 HTML/PDF 报告。

## 实验数据

整理后的实验测试数据位于 `experimental_test_data/`。文件夹和数据文件采用英文命名；每个实验详情说明提供中文和英文两个版本，文件名分别为 `experiment_details_zh-CN.txt` 和 `experiment_details_en-US.txt`。

## 引用

请根据 `CITATION.cff` 中的元数据引用本软件。正式归档发布前，请更新对应版本的标识信息。
