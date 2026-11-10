# CAMixerSR：RTX 5090 上的 5000 步短训

## 目标

检查同一类 BasicSR 流程怎样接入第二个网络，以及作者如何扩展训练管理类来处理路由约束。
本主题是从随机初始化训练的 5k 实验；作者预训练权重测试见 [预训练推理](../pretrained-inference/README.md)。

## 配置与方法

- [正式 5000 步配置](../../configs/camixersr/train/CAMixerSR_train_DF2K_x4_cloud_5000.yml)：单卡 batch 16、DF2K 图块、种子 10。
- 训练管理类为作者 `CAModel`，优化 L1 与路由约束；AdamW，EMA 0.999。
- 原长周期学习率节点保留，本次 5000 步内学习率没有衰减。
- [训练入口补丁](../../patches/CAMixerSR_training_entry.patch)：注册项目扩展并保存 ratio 属性。
- [独立环境脚本](../../scripts/environment/setup_camixersr_5090.sh)、[真实批次预检](../../scripts/preflight/preflight_camixersr_5090.py)。

使用项目内 `.venv-5090`，PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、BasicSR 1.4.2。
[初始 1000 步配置](../../configs/camixersr/train/CAMixerSR_train_DF2K_x4_cloud_1000.yml) 仅是被替代的计划，没有作为正式训练运行。

## 结果

DIV2K_val100，EMA，RGB PSNR，裁边 4：

| 更新步数 | 定期验证 PSNR / dB |
| --- | ---: |
| 1000 | 17.3931 |
| 2000 | 22.1289 |
| 3000 | 25.4935 |
| 4000 | 26.8954 |
| 5000 | **27.5543** |

结束附加验证为 **27.5540 dB**，没有新增参数更新；它与定期验证分别保留、不平均。
训练循环用时 35 分 46 秒；记录的最终模型、状态、进程和 GPU 释放核验通过。

## 证据

- [训练日志](../../results/logs/camixersr5000.stdout.log)：实际迭代、损失、定期与附加验证。
- [机器可读结果](../../results/camixersr5000.result.json)：两类验证分别标记，27.5543 是主结果。
- [检查点核验](../../results/camixersr5000.checkpoint_verification.json)：state iter=5000，params / params_ema 可读且有限。
- [启动记录](../../results/camixersr5000.launch.json)、[环境版本](../../results/camixersr_5090_environment.txt)、[预检结果](../../results/camixersr_5090_preflight.json)。
- [详细训练记录](../../docs/CAMixerSR-short-training-plan.md)：源版本、兼容问题、配置与权重哈希。

## 实验设置

本阶段为单次5k、batch16、AdamW、L1与路由约束，按定期验证记录主结果。

[返回实验主题索引](../README.md)
