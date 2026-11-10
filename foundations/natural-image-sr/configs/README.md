# 配置索引

配置按网络和用途分为 `safmn/`、`camixersr/`，每种网络再分 `train/` 与 `test/`。文件保留实测时的实验名和云端绝对路径；换机器前须核对数据、索引、初始化和检查点路径。

`model_type` 指训练／评估管理类，`network_g.type` 指网络本身。SAFMN 使用其 checkout 中支持 FFT 损失的 `SRModel`；CAMixerSR 正式训练使用扩展的 `CAModel`，预训练推理配置则使用 `SRModel`。

## SAFMN 训练：`safmn/train/`

### 同环境配对 30k 实验

| 配置 | 实验角色 |
| --- | --- |
| [SAFMN_train_DF2K_x4_pair_fft_30000.yml](safmn/train/SAFMN_train_DF2K_x4_pair_fft_30000.yml) | L1 + 0.05 FFTLoss，30k |
| [SAFMN_train_DF2K_x4_pair_nofft_30000.yml](safmn/train/SAFMN_train_DF2K_x4_pair_nofft_30000.yml) | 仅 L1，30k |

这两份配置属于同一轮 5090 单种子对照：batch 64、相同环境与公共随机初始化，除名称和 FFT 损失选项外结构化内容一致。`pretrain_network_g` 在这里指向随机初值 `common_init.pth`，不表示加载已训练模型；两组不恢复旧 optimizer/state。

默认云端部署时，分别复制为 suite 下的 `configs/train_fft_30000.yml` 与 `configs/train_nofft_30000.yml`。顺序与依赖见[脚本索引](../scripts/README.md)，结论见[配对实验记录](../docs/SAFMN-FFT-paired-30k.md)。

### 早期 4090 实验

| 配置 | 实际用途 |
| --- | --- |
| [SAFMN_train_DF2K_x4_sub_first5000.yml](safmn/train/SAFMN_train_DF2K_x4_sub_first5000.yml) | 含 FFT 的首次 5k 训练阶段 |
| [SAFMN_train_DF2K_x4_sub_cloud.yml](safmn/train/SAFMN_train_DF2K_x4_sub_cloud.yml) | 原 500k 目标的续训配置；该次实际在 30k 按短训规则停止，并未完成 500k |
| [SAFMN_train_DF2K_x4_sub_nofft_5000.yml](safmn/train/SAFMN_train_DF2K_x4_sub_nofft_5000.yml) | 同阶段关闭 FFT 的 5k 对照 |

这些是较早环境中的历史实验，不能与后续 5090 配对配置混合成一组受控对照。详见[实验笔记](../docs/experiment-notes.md)。

## SAFMN 测试：`safmn/test/`

| 配置 | 检查点与评估口径 |
| --- | --- |
| [SAFMN_DIV2K_val100_pretrained.yml](safmn/test/SAFMN_DIV2K_val100_pretrained.yml) | 作者预训练 `params`；DIV2K RGB PSNR，裁边 4 |
| [SAFMN_DIV2K_val100_30000.yml](safmn/test/SAFMN_DIV2K_val100_30000.yml) | 早期 4090 的 30k `params`；DIV2K RGB PSNR，裁边 4 |
| [SAFMN_DIV2K_val100_30000_ema.yml](safmn/test/SAFMN_DIV2K_val100_30000_ema.yml) | 同一早期 30k 检查点，明确加载 `params_ema`；DIV2K RGB PSNR，裁边 4 |
| [SAFMN_Set5_x4_cloud.yml](safmn/test/SAFMN_Set5_x4_cloud.yml) | 作者预训练 `params`；Set5 Y PSNR/SSIM，裁边 4 |
| [SAFMN_Set5_x4_5000.yml](safmn/test/SAFMN_Set5_x4_5000.yml) | 早期 5k `params`；Set5 Y PSNR/SSIM，裁边 4 |
| [SAFMN_Set5_x4_30000.yml](safmn/test/SAFMN_Set5_x4_30000.yml) | 早期 30k `params`；Set5 Y PSNR/SSIM，裁边 4 |

未写 `path.param_key_g` 的 SAFMN 直接测试默认加载 `params`；训练验证使用 EMA 时，必须选相同权重键才能对齐分数。上述 5k／30k YAML 指向早期实验，不是新的 paired30k 检查点。新配对实验的 Set5 评估由[统一评估脚本](../scripts/evaluation/evaluate_safmn_fft_pair.py)完成，加载各组的 `params_ema`。

## CAMixerSR：`camixersr/`

| 配置 | 状态与用途 |
| --- | --- |
| [train/CAMixerSR_train_DF2K_x4_cloud_5000.yml](camixersr/train/CAMixerSR_train_DF2K_x4_cloud_5000.yml) | **实际正式训练配置**：5k、batch 16、AdamW、L1 + 路由约束、EMA；训练管理类为 `CAModel` |
| [train/CAMixerSR_train_DF2K_x4_cloud_1000.yml](camixersr/train/CAMixerSR_train_DF2K_x4_cloud_1000.yml) | **已弃用的初始计划**：正式训练前被 5k 配置替代，保留用于说明计划变化，未作为独立正式 1k 训练执行 |
| [test/CAMixerSR_Set5_x4_cloud.yml](camixersr/test/CAMixerSR_Set5_x4_cloud.yml) | 作者预训练 `params_ema`；Set5 Y PSNR/SSIM，裁边 4；不是本仓库 5k 检查点的测试 |

5k 实验中的 1k 验证点属于 5k 训练过程，不是上述弃用计划的独立结果。环境、网络注册与训练入口补丁见 [CAMixerSR 运行记录](../docs/CAMixerSR-short-training-plan.md)。

## 使用与证据边界

- 训练／测试入口来自各自的上游项目，需使用正确的 BasicSR 导入路径与项目环境；不要仅根据安装包版本推断实际执行的源码。
- RGB 与 Y 指标、普通参数与 EMA 参数、不同硬件和优化设置的实验分别记录，不混成直接网络排名。
- `configs/` 是配置入口；[results/safmn_fft_pair_30k/configs](../results/safmn_fft_pair_30k/configs) 保存配对实验当次使用的原始配置快照，保持内容不变以核验来源哈希。
