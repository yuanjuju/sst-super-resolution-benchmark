# 作者权重推理

## 目标

检查 SAFMN、CAMixerSR 的公开权重能否被正确加载，完成数据读取、推理和指标计算。
此主题使用作者权重，与本仓库后续从头训练的结果分开。

## 配置与方法

| 检查 | 配置 | 权重键 |
| --- | --- | --- |
| SAFMN / Set5 | [Set5 ×4](../../configs/safmn/test/SAFMN_Set5_x4_cloud.yml) | `params`，测试默认值 |
| SAFMN / DIV2K | [DIV2K val100](../../configs/safmn/test/SAFMN_DIV2K_val100_pretrained.yml) | `params`，测试默认值 |
| CAMixerSR / Set5 | [Set5 ×4](../../configs/camixersr/test/CAMixerSR_Set5_x4_cloud.yml) | 显式 `params_ema` |

均为 ×4、裁边 4；Set5 报告 Y 通道 PSNR / SSIM，DIV2K 报告 RGB PSNR。
Set5 使用第三方镜像的五对图像，来源见 [第三方说明](../../THIRD_PARTY.md)。
模型实现、预训练权重归原作者，项目来源与固定提交见 [项目首页](../../README.md)。

## 结果

| 作者权重 | 数据 / 通道 | PSNR / dB | SSIM | 证据类型 |
| --- | --- | ---: | ---: | --- |
| SAFMN `params` | Set5 / Y | 32.1836 | 0.8948 | 原始测试日志 |
| CAMixerSR `params_ema` | Set5 / Y | 32.5533 | 0.8997 | 原始测试日志 |
| SAFMN `params` | DIV2K val100 / RGB | 28.9615 | — | 历史指标记录与配置 |

## 证据

- [SAFMN Set5 日志](../../results/logs/SAFMN_Set5_x4_cloud.log)：权重路径、图像数及分数。
- [CAMixerSR Set5 日志](../../results/logs/CAMixerSR_Set5_x4_cloud.log)：权重键、图像数及分数。
- [指标汇总](../../results/metrics.csv)：`author_pretrained` 各行。
- [历史实验记录](../../docs/experiment-notes.md)：作者权重与自己训练权重的比较口径。

SAFMN DIV2K 28.9615 dB 保留为历史测量；仓库当前没有收录对应的独立原始测试日志。
不能将自己的 30k 测试日志用作这项作者权重测量的原始证据。

## 限制

这些结果验证公开权重在本地协议下可运行，没有训练或更新作者权重。
RGB 与 Y 通道指标分开解释，Set5 五图均值与 DIV2K 百图均值也不能混合比较。
两种网络所加载的权重键不同，表格不是受控的架构性能排名。
这些是本项目的协议检查，不代表重新完成作者的论文训练。

下一步：[SAFMN 从头训练与续训](../safmn-training-4090/README.md)。

[返回实验主题索引](../README.md)
