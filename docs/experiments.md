# 实验总览

共118项任务记录，110项完成、8项失败。初期目录中与公开源码阶段完全相同的一份OISST×4记录已合并。

| 阶段 | 完成 / 任务 | 评测 | 结果 | 图表 |
|---|---:|---|---|---|
| 24 WAM结构实现 | 9/9 | 200张测试图 | [CSV](../tables/stages/01_wmsr_reconstructed.csv) · [逐图记录](../experiments/01_wmsr_reconstructed/runs/) | [查看](../figures/01_wmsr_reconstructed/) |
| 公开4 WAM实现 | 9/9 | 200张测试图 | [CSV](../tables/stages/02_wmsr_public.csv) · [逐图记录](../experiments/02_wmsr_public/runs/) | [查看](../figures/02_wmsr_public/) |
| 补充实验 | 78/86 | 200张测试图 | [CSV](../tables/stages/03_extended.csv) · [逐图记录](../experiments/03_extended/runs/) | [查看](../figures/03_extended/) |
| MambaIRv2-Light | 9/9 | 200张测试图 | [CSV](../tables/stages/04_mambairv2.csv) · [逐图记录](../experiments/04_mambairv2/runs/) | [查看](../figures/04_mambairv2/) |
| PFT与WMSR 1000步 | 2/2 | 80张训练内验证图 | [CSV](../tables/stages/05_pft_pilot.csv) · [逐图记录](../experiments/05_pft_pilot/runs/) | [查看](../figures/05_pft_pilot/) |
| ESC 1000步 | 1/1 | 80张训练内验证图 | [CSV](../tables/stages/06_esc_pilot.csv) · [逐图记录](../experiments/06_esc_pilot/runs/) | [查看](../figures/06_esc_pilot/) |
| ESC与WMSR 5000步 | 2/2 | 80张训练内验证图 | [CSV](../tables/stages/07_esc_wmsr_5000.csv) · [逐图记录](../experiments/07_esc_wmsr_5000/runs/) | [查看](../figures/07_esc_wmsr_5000/) |

## 数据与表格

正式测试结果见[tables/test200_results.csv](../tables/test200_results.csv)，预实验raw与EMA见[tables/development80_results.csv](../tables/development80_results.csv)。每条汇总记录指向对应任务config和final_results/pilot_results；逐图指标和训练CSV保留原精度。

## 方法比较

| 数据集 / 倍率 | WMSR | MambaIRv2-Light | 新模型减WMSR / dB |
|---|---:|---:|---:|
| HYCOM ×2 | 62.5142 | 55.2807 | -7.2335 |
| HYCOM ×3 | 58.3087 | 55.7984 | -2.5103 |
| HYCOM ×4 | 55.9512 | 55.0222 | -0.9290 |
| OISST ×2 | 66.2286 | 60.3210 | -5.9075 |
| OISST ×3 | 59.4712 | 56.6363 | -2.8349 |
| OISST ×4 | 54.0727 | 53.1714 | -0.9012 |
| GHRSST ×2 | 73.0821 | 62.2266 | -10.8555 |
| GHRSST ×3 | 69.0416 | 56.5610 | -12.4805 |
| GHRSST ×4 | 66.7912 | 60.1689 | -6.6223 |

## PFT和ESC预实验

| 阶段 / 模型 | 更新 | 参数 | PSNR / dB | RMSE / ℃ |
|---|---:|---|---:|---:|
| PFT与WMSR 1000步 / PFT | 1000 | raw | 35.0625 | 0.88026 |
| PFT与WMSR 1000步 / PFT | 1000 | ema | 22.9880 | 4.09418 |
| PFT与WMSR 1000步 / WMSR | 1000 | raw | 51.9284 | 0.14025 |
| PFT与WMSR 1000步 / WMSR | 1000 | ema | 23.6977 | 3.68373 |
| ESC 1000步 / ESC | 1000 | raw | 49.2422 | 0.18058 |
| ESC 1000步 / ESC | 1000 | ema | 30.3781 | 1.51172 |
| ESC与WMSR 5000步 / ESC | 5000 | raw | 45.9947 | 0.24354 |
| ESC与WMSR 5000步 / ESC | 5000 | ema | 53.5225 | 0.11703 |
| ESC与WMSR 5000步 / WMSR | 5000 | raw | 54.0029 | 0.11589 |
| ESC与WMSR 5000步 / WMSR | 5000 | ema | 51.6466 | 0.13926 |

## 失败记录

8项CRAN任务在非有限损失后退出。保留配置、训练片段和错误日志，汇总分数留空。详见[failures.json](../experiments/03_extended/failures.json)。CUDA、前后向、梯度、恢复和回收等检查在[checks](../checks/)中。

## 自然图像基础实验

[SAFMN、CAMixerSR与BasicSR](../foundations/natural-image-sr/README.md)收录作者权重测试、早期4090训练、5090 FFT配对30k、CAMixerSR 5k和本地CPU推理。查看[41条指标与工作簿](../tables/natural_image_sr/README.md)、[全部效果图](../foundations/natural-image-sr/figures/README.md)。
