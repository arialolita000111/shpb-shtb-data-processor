# SHPB/SHTB Data Processor

SHPB/SHTB Data Processor is a Python desktop and batch-processing toolkit for Split Hopkinson Pressure Bar and Split Hopkinson Tension Bar strain-gauge data.

霍普金森压杆/拉杆实验数据处理软件用于将 SHPB/SHTB 原始应变信号转换为应力、应变、应变率、力平衡诊断和可追溯报告。

## Documentation / 文档

- English README: [README.en-US.md](README.en-US.md)
- 中文 README：[README.zh-CN.md](README.zh-CN.md)
- English user manual: [docs/user_manual.en-US.md](docs/user_manual.en-US.md)
- 中文使用手册：[docs/user_manual.zh-CN.md](docs/user_manual.zh-CN.md)

## Quick Start / 快速开始

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m shpb_processor
```

Batch example / 批处理示例：

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

Experimental datasets are stored under `experimental_test_data/`.

实验测试数据位于 `experimental_test_data/`。
