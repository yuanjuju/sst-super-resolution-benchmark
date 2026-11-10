# 文档导航

本目录保存详细方法与实测记录；按实验问题浏览请从 [实验索引](../studies/README.md) 开始。

## 方法与执行记录

| 内容 | 文档 | 配套入口 |
| --- | --- | --- |
| 数据准备、早期 SAFMN 训练和权重复测 | [实验笔记](experiment-notes.md) | [数据与云端工具](../scripts/README.md) |
| CAMixerSR 注册、环境适配与 5k 训练 | [CAMixerSR 短训记录](CAMixerSR-short-training-plan.md) | [正式配置](../configs/camixersr/train/CAMixerSR_train_DF2K_x4_cloud_5000.yml) |
| SAFMN FFT 损失的同环境 30k 对照 | [配对实验报告](SAFMN-FFT-paired-30k.md) | [完整评估与验收](../results/safmn_fft_pair_30k/) |
| 前向计算、参数更新和单图推理 | [本地学习指南](../learning/README.md) | [精选推理图](../learning/examples/README.md) |

## 阅读顺序

1. 在 [实验索引](../studies/README.md) 确认研究问题、训练预算和所用检查点。
2. 查看对应配置，区分数据设置、网络、任务管理类、损失和 EMA。
3. 从 [结果索引](../results/README.md) 找到指标、曲线、原始日志和检查点核验记录。
4. 结合报告中的限制解释结果，再设计下一项实验。

## 指标标签

- `params` 是训练网络的普通参数；`params_ema` 是指数移动平均参数。
- DIV2K_val100 在这些训练记录中使用 RGB PSNR、裁边 4。
- Set5 同时存在单图、五图均值以及 RGB/Y 两类口径，比较前需核对标签。
- 定期检查点验证与训练结束附加验证分别记录，不能当作两次独立训练结果。
- 预训练推理、从头短训、受控损失对照属于不同实验范围。

仓库的内容组织和后续提交规则见 [目录约定](repository-structure.md)。
