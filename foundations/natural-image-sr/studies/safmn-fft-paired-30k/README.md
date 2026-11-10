# SAFMN：相同环境与初始权重的 FFT 配对实验

## 目标

检验 FFT 损失在短训早期的收益，到 30000 步时是否仍然存在。
这是一轮新的 RTX 5090 实验；两组从共同随机初始权重开始，不恢复旧 4090 检查点。

## 配置与方法

- [FFT 组配置](../../configs/safmn/train/SAFMN_train_DF2K_x4_pair_fft_30000.yml)：L1 + 0.05 FFTLoss。
- [无 FFT 组配置](../../configs/safmn/train/SAFMN_train_DF2K_x4_pair_nofft_30000.yml)：仅 L1。
- 同一张 5090、同一个 SAFMN 独立环境；种子 10、batch 64、Adam、EMA 0.999。
- 同一份 DF2K 图块；两配置除名称和 FFT 选项外相同，各训练 30000 步。
- 保留原 500k 余弦周期，每 5000 步保存并验证；共同初始化不是预训练模型。
- [顺序运行管理器](../../scripts/training/run_safmn_fft_pair.py)、[预检](../../scripts/preflight/preflight_safmn_fft_pair.py)、[评估](../../scripts/evaluation/evaluate_safmn_fft_pair.py)。

## 结果

DIV2K_val100，`params_ema`，RGB PSNR，裁边 4：

| 步数 | L1 + FFT / dB | 仅 L1 / dB | 差值 / dB |
| --- | ---: | ---: | ---: |
| 5000 | 28.1437 | 27.7348 | +0.4089 |
| 30000 | **28.7956** | **28.7050** | **+0.0906** |

六个定期检查点的 FFT 组分数均较高，早期差距缩小；完整趋势见 [验证曲线](../../results/safmn_fft_pair_30k/results/training_curve.png)。
Set5 五图 30k EMA 均值：FFT / 无 FFT 的 RGB PSNR 为 30.0542 / 29.9539 dB，
Y PSNR 为 31.9488 / 31.8489 dB；SSIM 与逐图结果见下列 CSV。

## 证据

- [完整实验说明](../../docs/SAFMN-FFT-paired-30k.md)：设计、协议、全部检查点和解释边界。
- 训练原始日志：[FFT](../../results/safmn_fft_pair_30k/fft.stdout.log)、[无 FFT](../../results/safmn_fft_pair_30k/nofft.stdout.log)。
- 结果表：[12 条定期验证](../../results/safmn_fft_pair_30k/results/training_metrics.csv)、[Set5 均值](../../results/safmn_fft_pair_30k/results/set5_mean.csv)、[逐图指标](../../results/safmn_fft_pair_30k/results/set5_per_image.csv)。
- 图片：[Butterfly 对比](../../results/safmn_fft_pair_30k/results/comparisons/butterfly.png)、[其他对比](../../results/safmn_fft_pair_30k/results/comparisons/)。
- 验收：[FFT 检查点](../../results/safmn_fft_pair_30k/fft.checkpoint_verification.json)、[无 FFT 检查点](../../results/safmn_fft_pair_30k/nofft.checkpoint_verification.json)、[最终核验](../../results/safmn_fft_pair_30k/final_verification.json)。
- 来源：[评估清单与哈希](../../results/safmn_fft_pair_30k/results/evaluation.json)、[环境](../../results/safmn_fft_pair_30k/environment.txt)、[预检](../../results/safmn_fft_pair_30k/preflight.json)、[源码提交](../../results/safmn_fft_pair_30k/source_revision.txt)。

归档 [suite](../../results/safmn_fft_pair_30k/) 保留当次配置和脚本快照；顶层脚本为可复用入口。
两份配置与 suite/configs 的原始运行配置相同；归档脚本可能与后续改进版本不同。

## 实验设置

本阶段为单种子配对30k，两组使用共同初值和一致训练设置。

[返回实验主题索引](../README.md)
