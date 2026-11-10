# SAFMN：RTX 4090 训练、续训与早期损失对照

## 目标

跑通 DF2K 数据、BasicSR 训练、检查点保存、状态恢复和 EMA 验证，并检查 FFT 损失的短训影响。
这是早期 4090 实验；后续 [5090 配对实验](../safmn-fft-paired-30k/README.md) 使用全新权重。

## 配置与方法

- [首个 5000 步](../../configs/safmn/train/SAFMN_train_DF2K_x4_sub_first5000.yml)：L1 + 0.05 FFTLoss。
- [长周期续训配置](../../configs/safmn/train/SAFMN_train_DF2K_x4_sub_cloud.yml)：从 5k 恢复，实际在 30k 停止。
- [无 FFT 的 5000 步](../../configs/safmn/train/SAFMN_train_DF2K_x4_sub_nofft_5000.yml)：独立训练、仅 L1。
- [停训守护脚本](../../scripts/training/stop_safmn_at_50000.py)：历史实验的检查点与验证分数停训规则。

网络为 SAFMN ×4，batch 64，Adam，EMA 0.999；使用 DF2K 配对图块。
保留 500k 余弦学习率周期。实验目录和日志名中的 `500K` 不表示实际完成了 50 万步。

## 结果

| 测量 | 数据 / 通道 / 权重 | PSNR / dB |
| --- | --- | ---: |
| 含 FFT，5k | DIV2K val100 / RGB / EMA | 28.1530 |
| 无 FFT，5k | DIV2K val100 / RGB / EMA | 27.7030 |
| 含 FFT，30k | DIV2K val100 / RGB / EMA | 28.7915 |
| 含 FFT，30k 独立复测 | DIV2K val100 / RGB / `params` | 28.6901 |
| 含 FFT，30k 独立复测 | Set5 / Y / `params` | 31.7874 |

以上均裁边 4；30k EMA 独立复测也是 28.7915 dB。早期 5k 的 FFT 差值为 +0.4500 dB。

## 证据

- 训练日志：[首个 5k](../../results/logs/first5000.stdout.log)、[5k→30k 续训](../../results/logs/full500k_resume.stdout.log)、[无 FFT 5k](../../results/logs/nofft5000.stdout.log)。
- 独立复测：[30k 普通参数](../../results/logs/SAFMN_DIV2K_val100_30000.stdout.log)、[30k EMA](../../results/logs/SAFMN_DIV2K_val100_30000_ema.stdout.log)、[30k Set5](../../results/logs/SAFMN_Set5_x4_30000.stdout.log)。
- 复测配置：[普通参数](../../configs/safmn/test/SAFMN_DIV2K_val100_30000.yml)、[EMA](../../configs/safmn/test/SAFMN_DIV2K_val100_30000_ema.yml)、[Set5](../../configs/safmn/test/SAFMN_Set5_x4_30000.yml)。
- [详细记录与检查点哈希](../../docs/experiment-notes.md)、[指标汇总](../../results/metrics.csv)。

旧 30k 模型 SHA-256 为 `c8f372864a1479fb555d1f2a4793b31ce5d1e6a55741ca2393b37606dc0370f3`。
[本地学习样例](../../learning/examples/README.md) 使用这份检查点，包含普通参数与 EMA 单图比较。
5k Set5 的独立测试日志已补入：[31.4363 dB / 0.8828](../../results/logs/SAFMN_Set5_x4_5000.stdout.log)。
10k 普通参数 DIV2K 28.5927 保留在历史指标表中。

## 实验设置

本阶段训练至30k；训练验证用EMA，独立测试按各配置记录params或params_ema。

[返回实验主题索引](../README.md)
