# 实验主题索引

## 目标

从研究问题进入配置、实测结果和原始证据，区分作者权重推理、自己的训练及有控制条件的对照实验。
这些入口组织现有材料；配置、日志和评估产物仍保留各自路径。

## 配置与方法

| 主题 | 要回答的问题 | 入口 |
| --- | --- | --- |
| 作者权重推理 | 能否正确加载两个项目并完成统一口径的测试？ | [预训练推理](pretrained-inference/README.md) |
| SAFMN 早期训练 | 数据、保存、续训、EMA 验证如何构成完整流程？ | [RTX 4090 训练](safmn-training-4090/README.md) |
| CAMixerSR 短训 | 如何将第二个网络及路由损失接入 BasicSR？ | [RTX 5090 训练](camixersr-training-5090/README.md) |
| SAFMN FFT 对照 | 相同环境和初始化下，FFT 损失的早期优势是否保留？ | [各 30k 步配对实验](safmn-fft-paired-30k/README.md) |
| 本地 CPU 学习 | 尺寸、参数更新及实际单图输出怎样对应？ | [学习练习](../learning/README.md) |

配置按 [SAFMN](../configs/safmn/) / [CAMixerSR](../configs/camixersr/) 和 train / test 分类。
脚本按 [数据](../scripts/data/)、[环境](../scripts/environment/)、[预检](../scripts/preflight/)、[训练](../scripts/training/)、[评估](../scripts/evaluation/) 分类。

## 结果

| 实验 | 主要测量 | 应怎样理解 |
| --- | --- | --- |
| 4090 SAFMN 30k | DIV2K EMA RGB PSNR 28.7915 dB | 从早期 5k 恢复的历史训练 |
| 5090 CAMixerSR 5k | DIV2K EMA RGB PSNR 27.5543 dB | 第二个项目的单次短训 |
| 5090 SAFMN 配对 30k | FFT / 无 FFT 为 28.7956 / 28.7050 dB | 同环境、共同初值的单种子对照 |

所有以上 DIV2K 指标均裁边 4；具体权重、数据和评估口径见对应主题。
早期 SAFMN 检查点与后续配对实验的检查点分开保存，不能按相同步数合并。

## 证据

- [完整指标表](../results/metrics.csv)：跨主题汇总，重复评估依靠 notes 区分。
- [历史实验记录](../docs/experiment-notes.md)：数据来源、4090 训练与 CAMixerSR 短训。
- [配对实验记录](../docs/SAFMN-FFT-paired-30k.md)：共同初始化、验收及评估协议。
- [数据归档哈希](../results/dataset_sha256.txt)：训练数据准备的共享来源记录。
- [精选单图样例](../learning/examples/README.md)：使用早期 4090 SAFMN 30k 权重。

## 限制

原始日志与检查点核验是可追溯证据；仅有历史 CSV / 配置的指标会在主题内注明。
不同硬件、batch、优化设置或评估口径的数值不直接组成网络性能排名。
这些是短训和单种子测量，没有完成作者完整 500k 训练，也没有建立多种子统计结论。
仓库不提供数据集、虚拟环境或二进制检查点；路径与运行记录用于说明实验来源。

[返回项目首页](../README.md)
