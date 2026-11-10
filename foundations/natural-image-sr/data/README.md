# 自然图像数据

| 数据 | 用途 | 本模块提供 |
|---|---|---|
| Set5 ×4 | 五图测试、单图推理 | [5张HR和5张LR](../datasets/Set5/) · [尺寸与SHA256](manifest.json) |
| DIV2K | 800张训练、100张验证 | [官方数据](https://data.vision.ee.ethz.ch/cvl/DIV2K/) · 下载脚本 |
| Flickr2K | 与DIV2K合成DF2K | [数据来源](https://cv.snu.ac.kr/research/EDSR/) · 下载脚本 |
| DF2K图块 | SAFMN/CAMixerSR训练 | 139,995对，HR480/LR120；训练随机裁取HR256/LR64 |

Set5来自[eugenesiow/Set5镜像](https://huggingface.co/datasets/eugenesiow/Set5)，沿用当时测试的原始文件。训练数据通过[scripts/data](../scripts/data/)下载、解压、配对及裁图；文件校验见[dataset_sha256.txt](../results/dataset_sha256.txt)。

本地学习脚本直接读取`datasets/Set5`。本模块使用×4、裁边4，指标表逐行标注RGB/Y和params/params_ema。
