# Contributing / 贡献指南

## English

Contributions should preserve the project's main scientific contract: processing must be traceable, parameterized, and testable.

### Development Setup

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

### Contribution Rules

- Keep formulas and assumptions visible in code, docs, and exported reports.
- Add or update tests for any change that affects numerical results.
- Do not replace calibrated experimental judgment with unchecked automatic decisions.
- Keep public configuration and output schemas backward compatible within a release series.
- Document new instrument-specific adapters, file formats, and correction models with example data whenever possible.

### Reporting Issues

When reporting a bug, include the input file format, relevant column names, workspace configuration, `quality_report.json`, software version, expected behavior, and observed behavior.

## 中文

贡献内容应保持本项目的核心科学约束：处理流程必须可追溯、参数化并且可测试。

### 开发环境

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

### 贡献规则

- 在代码、文档和导出报告中保留公式与假设。
- 任何影响数值结果的改动都应新增或更新测试。
- 不要用未经检查的自动判断替代经过校准的实验判断。
- 同一发布系列内应尽量保持公开配置和输出格式向后兼容。
- 新增仪器适配、文件格式或修正模型时，应尽可能配套示例数据。

### 问题反馈

反馈问题时，请包含输入文件格式、相关列名、工作区配置、`quality_report.json`、软件版本、期望行为和实际行为。
