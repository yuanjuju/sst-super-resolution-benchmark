# 结果与证据索引

先根据实验入口确认运行归属，再读取指标。`metrics.csv` 是跨实验汇总，不能忽略其 `checkpoint`、`weight_key`、`dataset`、`channel` 和 `notes` 直接排序。

| 内容组 | 指标入口 | 原始证据 | 解释 |
| --- | --- | --- | --- |
| 作者权重推理 | [汇总 CSV](metrics.csv) 中 `author_pretrained` 行 | [SAFMN Set5](logs/SAFMN_Set5_x4_cloud.log)、[CAMixerSR Set5](logs/CAMixerSR_Set5_x4_cloud.log) | [预训练推理](../studies/pretrained-inference/README.md) |
| 早期 SAFMN 4090 训练与复测 | [汇总 CSV](metrics.csv) 中 SAFMN `scratch_*` 行 | [5k](logs/first5000.stdout.log)、[续训至 30k](logs/full500k_resume.stdout.log)、[无 FFT 5k](logs/nofft5000.stdout.log) | [早期训练](../studies/safmn-training-4090/README.md) |
| CAMixerSR 5090 5k 短训 | [结果 JSON](camixersr5000.result.json) | [训练日志](logs/camixersr5000.stdout.log)、[检查点核验](camixersr5000.checkpoint_verification.json) | [CAMixerSR 训练](../studies/camixersr-training-5090/README.md) |
| SAFMN 同环境 FFT 配对 30k | [验证 CSV](safmn_fft_pair_30k/results/training_metrics.csv)、[Set5 均值](safmn_fft_pair_30k/results/set5_mean.csv) | [套件归档](safmn_fft_pair_30k/)、[最终验收](safmn_fft_pair_30k/final_verification.json) | [受控损失对照](../studies/safmn-fft-paired-30k/README.md) |
| CPU 单图学习样例 | [精选图与测量](../learning/examples/README.md) | 精选 JSON 保留权重、架构及原记录哈希 | 来自早期 4090 30k 权重，与新配对实验分开 |

## 数据来源

[dataset_sha256.txt](dataset_sha256.txt) 记录下载的数据压缩包哈希，服务全部训练实验。实际数据不随仓库提供。

## 原始记录说明

- 旧日志名称 `full500k_resume` 表示原计划；实际训练止于 30k。
- 新套件中的 `results/` 是原云端目录结构的一部分，收录 12 条定期验证、20 条 Set5 逐图指标、4 条均值、20 张预测、5 张对比图及曲线。
- CAMixerSR 5k 的主记录为 27.5543 dB；27.5540 dB 属于训练结束后的附加验证。
- 历史 CSV 中部分测量没有独立原始测试日志，具体见对应实验入口。已有其他日志不能替代这些缺失的来源。
- 哈希和验收 JSON 用于匹配原始文件；模型与训练状态本身保留在实验存储中。
