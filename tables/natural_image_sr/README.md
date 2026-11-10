# SAFMN与CAMixerSR结果表

[Excel工作簿](experiments.xlsx)包含FFT对照、CAMixerSR训练曲线、全部指标、Set5逐图结果和本地推理五个工作表。两张曲线随源单元格更新。

| 文件 | 内容 |
|---|---|
| [metrics.csv](metrics.csv) | 41条历史测试与验证指标，逐行保留数据集、通道、权重键 |
| [training_validation.csv](training_validation.csv) | 24条定期验证和1条结束重复验证 |
| [set5_per_image.csv](set5_per_image.csv) | FFT/无FFT在5k和30k的20条逐图指标与预测路径 |
| [set5_mean.csv](set5_mean.csv) | 4组Set5五图平均 |
| [local_inference.csv](local_inference.csv) | bird普通参数、bird EMA、butterfly普通参数的CPU输出 |
| [test_logs.csv](test_logs.csv) | 24份日志的位置、PSNR事件数和SHA256 |
| [set5_files.csv](set5_files.csv) | 10张HR/LR原图的尺寸和SHA256 |

## 更新

```bash
python tools/build_natural_sr_tables.py
python tools/verify_natural_sr.py
```

[基础模块](../../foundations/natural-image-sr/README.md) · [对比图](../../foundations/natural-image-sr/figures/README.md)
