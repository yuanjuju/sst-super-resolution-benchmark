# 本地BasicSR与SAFMN练习

脚本使用CPU，依赖安装到本模块的`.venv`，Set5直接读取模块内的`datasets/Set5`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r learning/requirements.txt
.venv/bin/python learning/01_forward_and_update.py --help
```

## 单图推理

先按[权重说明](../weights/README.md)下载SAFMN早期30k文件到`downloads`，再运行：

```bash
.venv/bin/python learning/02_infer_image.py \
  --input datasets/Set5/LR_bicubic/X4/bird.png \
  --gt datasets/Set5/HR/bird.png --weight-key params_ema
```

默认输出为`learning/outputs/<图名>_30k`。通过`--weight-key params`切换普通参数；网络源码位于`sources/SAFMN`。

## 已保存的结果

| 样例 | RGB PSNR / dB | 入口 |
|---|---:|---|
| bird普通参数 | 32.2033 | [全部输出](../results/local-inference/bird_30k/) |
| bird EMA | 32.3384 | [全部输出](../results/local-inference/bird_30k_ema/) |
| butterfly普通参数 | 26.4593 | [全部输出](../results/local-inference/butterfly_30k/) |

[普通参数与EMA对照图](examples/bird_params_vs_ema/comparison.png) · [全部指标](../../../tables/natural_image_sr/local_inference.csv)

指标使用uint8 RGB、裁边4。Pillow bicubic作为本地插值参照，耗时保留单次CPU前向记录。
