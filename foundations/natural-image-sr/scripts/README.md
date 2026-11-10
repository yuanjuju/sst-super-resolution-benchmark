# 脚本索引

按数据准备、环境、预检、训练和评估组织。命令中的相对路径以仓库根目录为起点；云端脚本默认使用 `/root/fsas/sr-repro`，运行前需准备对应的上游仓库、数据和项目环境。

阅读已有结论可直接查看 [SAFMN 配对 30k 记录](../docs/SAFMN-FFT-paired-30k.md)和 [CAMixerSR 5k 记录](../docs/CAMixerSR-short-training-plan.md)。以下脚本是实测流程与可调整工具，不是空白机器上的一键安装器。

## 1. 数据准备：`data/`

按下表顺序执行，后一步使用前一步的输出。

| 顺序 | 脚本 | 输入与作用 |
| --- | --- | --- |
| 1 | [download_training_data.py](data/download_training_data.py) | 下载 DIV2K、Flickr2K 压缩包，支持分段续传并记录校验信息 |
| 2 | [extract_training_data.py](data/extract_training_data.py) | 将压缩包解压到数据目录 |
| 3 | [prepare_df2k_pairs.py](data/prepare_df2k_pairs.py) | 核对图像数量及 ×4 尺寸关系，建立 DF2K HR/LR 配对目录 |
| 4 | [prepare_safmn_subimages.py](data/prepare_safmn_subimages.py) | 调用 SAFMN 上游裁图 worker，生成对齐图块，支持按源图续作 |
| 5 | [build_meta_info.py](data/build_meta_info.py) | 检查图块并生成 BasicSR 训练索引 |

```bash
python scripts/data/download_training_data.py datasets/_archives
python scripts/data/extract_training_data.py datasets
python scripts/data/prepare_df2k_pairs.py datasets
python scripts/data/prepare_safmn_subimages.py datasets
python scripts/data/build_meta_info.py datasets/DF2K --expected-count 139995
```

裁图前需在仓库根目录的 `SAFMN/` 准备指定版本的上游 checkout，并使用已安装依赖的项目 Python。配对、裁图和索引脚本保留在同一目录，供它们共享 PNG 尺寸检查函数。

## 2. 项目环境：`environment/`

| 脚本 | 依赖与输出 |
| --- | --- |
| [setup_camixersr_5090.sh](environment/setup_camixersr_5090.sh) | 建立 `CAMixerSR/.venv-5090`，安装 5090 对应的 PyTorch/CUDA 依赖；需要云端已有的 `.venv` 引导 pip，以及根目录的 `requirements-camixersr-5090.txt` |
| [setup_safmn_pair_5090.py](environment/setup_safmn_pair_5090.py) | 从已验证的 CAMixerSR Python 3.10 环境离线复制包，建立独立的 `SAFMN/.venv-5090`，处理 torchvision 兼容导入并记录版本 |

SAFMN 环境脚本依赖前一个已验证环境，拒绝覆盖已有目标环境。两个项目的环境保留在各自目录，原环境及系统 Python 不作为安装目标。

## 3. 真实批次预检：`preflight/`

| 脚本 | 检查范围 |
| --- | --- |
| [preflight_camixersr_5090.py](preflight/preflight_camixersr_5090.py) | 使用 `CAModel` 和正式 5k 配置，检查 batch 16、L1/路由损失、参数更新与 EMA 输出 |
| [preflight_safmn_fft_pair.py](preflight/preflight_safmn_fft_pair.py) | 检查两份配对配置，生成公共随机初值，分别完成三次 batch 64 更新并核对 EMA 输出 |

预检会实际使用 GPU 并执行少量参数更新。诊断参数随后丢弃；正式训练重新初始化。SAFMN 预检会生成 `common_init.pth`、`preflight.json`，已有初始化文件时拒绝覆盖。

## 4. 训练管理：`training/`

| 脚本 | 用途 |
| --- | --- |
| [run_safmn_fft_pair.py](training/run_safmn_fft_pair.py) | 顺序运行 FFT／无 FFT 各 30k；先校验配置与初始化，再训练、验收全部检查点并调用评估。拒绝覆盖，不自动重试或续训 |
| [stop_safmn_at_50000.py](training/stop_safmn_at_50000.py) | 早期 SAFMN 实验的专用停训守护；包含当时的 PID、目录及判定条件，复用前必须核对并调整 |

### SAFMN runner 的云端部署约定

默认 suite 为 `/root/fsas/sr-repro/safmn_fft_pair_30k`。仓库中的功能分类与云端 suite 布局不同，启动前需将文件放到 runner 约定的位置：

| 仓库文件 | 云端 suite 中的位置 |
| --- | --- |
| [evaluate_safmn_fft_pair.py](evaluation/evaluate_safmn_fft_pair.py) | `evaluate_safmn_fft_pair.py`，直接位于 suite 根目录 |
| [FFT 30k 配置](../configs/safmn/train/SAFMN_train_DF2K_x4_pair_fft_30000.yml) | `configs/train_fft_30000.yml` |
| [无 FFT 30k 配置](../configs/safmn/train/SAFMN_train_DF2K_x4_pair_nofft_30000.yml) | `configs/train_nofft_30000.yml` |

还需已通过的 `preflight.json`、与其哈希一致的 `common_init.pth`，以及 `SAFMN/.venv-5090`。runner 不会自动从仓库分类目录寻找或部署评估脚本。

当前 runner 会向评估脚本透传 `--root`、`--suite`。预检脚本及 YAML 仍保留原云端布局，采用其他目录时须同步检查配置中的数据、初始化与实验路径。

## 5. 统一评估：`evaluation/`

[evaluate_safmn_fft_pair.py](evaluation/evaluate_safmn_fft_pair.py) 使用 SAFMN checkout 的 BasicSR，在两组训练完成后生成：

- 每组 6 个定期检查点的 DIV2K RGB EMA 验证记录，排除训练结束后的重复验证。
- 两组 5k／30k 的 Set5 五图 RGB/Y PSNR、SSIM，以及逐图记录和五图算术均值。
- 20 张预测、5 张共享中心 ROI 对照图、训练曲线，以及含 SHA-256 的 `evaluation.json`。

它使用 `params_ema`、uint8 图像和裁边 4；不执行训练，也不进行速度基准测试。Pillow 用于对照图，训练曲线优先使用 matplotlib。

## 当前工具与历史归档

`scripts/` 保存当前可维护版本；[results/safmn_fft_pair_30k](../results/safmn_fft_pair_30k) 中的同名脚本是当次运行的原始快照。原始文件、配置及测量产物保持原样，供哈希与实验来源核查；当前工具调整不反写历史快照。
