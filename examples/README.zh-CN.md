# 示例数据

本目录包含用于测试和演示处理流程的合成数据与本地示例数据。

合成数据文件：

- `synthetic_shpb_ideal`
- `synthetic_shpb_noisy`
- `synthetic_shpb_unbalanced`
- `synthetic_shpb_overlap`

色散修正示例：

- `synthetic_shpb_dispersion_long_clean`
- `synthetic_shpb_dispersion_long_noisy`
- `synthetic_shpb_dispersion_large_bar`

每个色散示例配有相应的配置或元数据文件。这些示例用于验证软件流程，不代表真实材料性能测量。

运行批处理演示：

```powershell
shpb-processor batch --config examples/batch_config.json --input examples --output output/demo
```

每个样本输出目录可包含 HTML/PDF 报告、SVG 图、CSV 表、Excel 工作簿和 `quality_report.json`。
