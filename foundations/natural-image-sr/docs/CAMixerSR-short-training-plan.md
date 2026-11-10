# CAMixerSR：云端 5000 步短训记录

状态：2026-10-22，5000 步训练和最终验收已完成，已记录各次 EMA 验证分数，未见训练报错。
启动 PID 为 276，现已退出；最终模型和训练状态可读，GPU 计算进程已释放。

用户追加要求“可以多训练点”后，正式目标从初始 1000 步提高为 5000 步，
使用 `configs/camixersr/train/CAMixerSR_train_DF2K_x4_cloud_5000.yml`。
初始 1000 步配置仅保留为计划记录，不是本次正式启动配置。

## 学习目的

验证同一类 BasicSR 训练流程如何承载第二个网络，并观察作者如何扩展通用任务管理类。
本次关注数据加载、两项损失、优化、EMA 验证和保存，并观察 5000 步内的学习趋势。

| 设置 | 实际配置 |
| --- | --- |
| 网络 | CAMixerSR ×4，ratio 0.5，随机初始化 |
| 任务管理类 | 作者的 CAModel，继承 BasicSR SRModel |
| 训练数据 | 复用云端 DF2K 的 139,995 对 HR/LR 图块及现有索引 |
| 数据设置 | 单卡、batch 16（GPU 预检通过）、HR 随机图块 256×256 |
| 训练预算 | 固定 5000 步，完成后退出 |
| 优化 | AdamW，初始学习率 0.0005，weight decay 0.0001，betas (0.9, 0.99) |
| 损失 | L1 加作者实现的路由约束 |
| 随机种子 | 10 |
| EMA | 衰减系数 0.999 |
| 验证 | DIV2K val100，RGB PSNR，裁边 4 |
| 验证和保存频率 | 每 1000 步验证，保存模型、训练状态及日志 |

原仓库的长周期学习率节点予以保留，因此本次只覆盖其初始学习率阶段。
单卡 batch 16、DF2K 数据和短训预算也不等同于作者的示例训练方案。
这些结果不能作为与 SAFMN 的公平架构性能排名。

## 本地检查发现的训练兼容问题

对应原仓库提交：`d690b12bb2a0a185dc0049538b377d569f520497`。

1. 通用 SRModel 在推理时可以接收图像输出，但 CAMixerSR 训练时返回图像和路由约束两项。
   作者的 CAModel 正是为此提供的扩展；短训配置选择 CAModel。
2. 本地原始训练入口尚未显式导入项目扩展模块，需要导入本项目的 archs、models，完成注册。
3. CAMixerSR 的训练分支读取 `self.ratio`，但原始构造函数没有保存该属性。
   补丁将构造参数 ratio 保存到该属性，不增加可学习参数。

以上两处源码调整集中在 `patches/CAMixerSR_training_entry.patch`。
已核对云端代码并应用补丁，保留原始文件和来源记录；随后 GPU 预检和正式训练日志均通过各自阶段的检查。

## 执行顺序及验收

1. 已核对 GPU、原项目虚拟环境、安装版 BasicSR 的实际导入位置、已有数据和索引。
2. 已检查并应用两处兼容修改，保留上游来源和修改记录。
3. 已用真实图块小批次检查前向计算、L1 和路由损失、反向传播及显存。
4. 正式日志已记录 5000 步训练和每 1000 步验证，未见训练报错。
5. 已用本项目 `.venv-5090` 的 PyTorch 读取第 5000 步模型和训练状态：state 的 iter 为 5000；权重包含 params 和 params_ema，两者各有 718 个张量、765,322 个参数且数值全部有限。PID 276 已不存在，GPU 计算进程列表为空。

本次可用 GPU 改为 RTX 5090 32 GB，驱动版本 590.44.01。
原项目环境为 PyTorch 2.1.1+cu121，其编译架构列表不包含 5090 的 sm_120。
因此改用独立环境 `/root/fsas/sr-repro/CAMixerSR/.venv-5090`，
安装 PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128 / BasicSR 1.4.2。
Blackwell 支持依据 [PyTorch 2.7 官方发布说明](https://pytorch.org/blog/pytorch-2-7/)，
具体 cu128 版本配对见 [官方历史安装说明](https://pytorch.org/get-started/previous-versions/#v271)。

环境安装脚本为 `scripts/environment/setup_camixersr_5090.sh`，依赖记录为 `requirements-camixersr-5090.txt`。
该环境使用 headless OpenCV，并将 BasicSR 的旧版 torchvision 灰度函数导入改为当前接口。
容器没有系统 ensurepip，脚本使用原项目的 pip 引导新环境，包安装目标仍为新环境。
GPU 预检脚本 `scripts/preflight/preflight_camixersr_5090.py` 使用真实配对图块完成三次诊断更新及 EMA 前向；
这些诊断参数会丢弃，正式 5000 步从固定种子的随机初始化重新开始。

数据复用现有云盘，没有为本次训练重新下载数据。
日志与实验目录和 SAFMN 分开，实验根目录为：
`/root/fsas/sr-repro/CAMixerSR/codes/experiments/CAMixerSR_DF2K_x4_cloud_5000/`。

## 本次实测和记录

- 原仓库提交已在云端核对为 `d690b12bb2a0a185dc0049538b377d569f520497`。
- GPU 为 RTX 5090，计算能力 12.0；新环境架构列表包含 sm_120，CUDA 前向和反向计算通过。
- 预检使用真实 DF2K 图块，输入 `[16,3,64,64]`，目标和 EMA 输出均为 `[16,3,256,256]`。
- CAMixerSR 参数量为 765,322；三次预检更新的两项损失均为有限值，参数确实发生变化。
- PyTorch 统计的峰值已分配显存为 10.96 GiB；预热后同批次的纯计算约 0.24–0.26 秒/步。
- 正式训练初段包含读图的实际迭代耗时约 0.5 秒/步；这不是与其他硬件或网络的受控速度比较。
- 正式日志为 `/root/fsas/sr-repro/camixersr5000.stdout.log`，PID 文件为 `/root/fsas/sr-repro/camixersr5000.pid`。
- 本地保留环境版本 `results/camixersr_5090_environment.txt`、预检 `results/camixersr_5090_preflight.json`、启动记录 `results/camixersr5000.launch.json`。
- 5000 步配置 SHA-256：`32ada9a33c571a0df521c5ca66e4f50eb94d03a019fefe1088ece60a70ba36d4`。
- 源码兼容补丁 SHA-256：`88b0c06a9833ecde6562c1760420d6fccae8bce23a89fe0ec578ef78c0a8697a`。

## 验证结果

验证使用 DIV2K_val100，RGB PSNR，裁边 4，EMA 权重。

| 训练步数 | EMA PSNR / dB | 记录类型 |
| --- | ---: | --- |
| 1000 | 17.3931 | 定期验证 |
| 2000 | 22.1289 | 定期验证 |
| 3000 | 25.4935 | 定期验证 |
| 4000 | 26.8954 | 定期验证 |
| 5000 | 27.5543 | 第 5000 步检查点对应的定期验证，主记录值 |
| 训练结束附加验证 | 27.5540 | 单独保留的再次验证，不与主记录值平均 |

训练日志记录用时 35 分 46 秒；结束附加验证于北京时间 2026-10-22 20:28:39 完成，检查点和进程验收于 20:29:38 完成。
结束附加验证来自训练入口在循环结束后再次调用 validation，没有增加训练步数。
两次末尾验证分数均按日志原值分别保留，没有平均；尚未对 0.0003 dB 差异作严格归因。
检查点可读性、training state 内容、训练进程退出和 GPU 释放均已核验通过。
本次是单次 5000 步短训，batch、优化等设置与此前 SAFMN 不同，不能据此给两个网络作公平性能排名，也不代表作者完整训练的结论。
