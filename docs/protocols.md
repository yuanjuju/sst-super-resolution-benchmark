# 实验协议

## 数据和指标

HYCOM、OISST、GHRSST各1000张HR样本，固定800训练/200测试；HR为96×96，×2/×3/×4的LR分别为48/32/24。以PyTorch bicubic、antialias=True、align_corners=False生成LR。温度按[-2,40]℃固定映射并复制为三通道。

- PSNR/SSIM：BT.601 Y通道，range255，裁去scale像素边界后逐图计算、取均值。
- 温度RMSE：预测三通道取均值并反变换到摄氏度，同样裁边；汇总逐图MSE再开方。
- 温度MAE：逐图计算后取均值。HYCOM另保留NP、AGM、NIO、EWP四区域结果。

## 训练设置

| 阶段 | 划分 | 更新次数 | 优化器和学习率 | 损失 |
|---|---|---|---|---|
| 24 WAM实现、公开4 WAM实现 | 800/200 | 10000 | Adam 1e-3，StepLR20/0.5 | 0.1L1+0.9FFL |
| 补充方法比较中的WMSR | 800/200 | 10000 | 同上 | 0.1L1+0.9FFL |
| EDSR、ELAN、DAT、SAFMN、MambaIR、CRAN | 800/200 | 10000 | 同上 | L1 |
| WMSR消融 | 800/200 | 2000 | 同上 | 0.1L1+0.9FFL |
| 通道、组数和损失扫描 | 800/200 | 10000 | 同上 | 见每组config |
| MambaIRv2-Light与5090 WMSR | 800/200 | 10000 | 同上 | 0.1L1+0.9FFL |
| PFT、ESC与WMSR预实验 | 720/80，取自原训练800 | 1000或5000 | AdamW2e-4，warmup100后恒定 | L1、EMA0.999 |

所有阶段seed10、有效batch8。预实验采用micro-batch2、梯度累计4、FP32且两处TF32关闭；5000步对照从1000步完整断点恢复优化器、EMA、采样顺序及RNG。历史4090和5090环境分别保留在environment目录。

## 版本记录

- 初期24 WAM模型按论文结构描述实现；公开WMSR为固定上游4 WAM网络。这两套模型分别位于01、02阶段。
- 86项补充实验中，六种比较网络用L1，WMSR用混合损失；具体变体和参数量保留在config及profiles表。
- MambaIRv2-Light保留原生Gumbel路由，完整评测前固定CPU/CUDA seed10。
- PFT保留完整网络，SMM反向CUDA中的两处累加改为atomicAdd；补丁、源码哈希和梯度检查均保留。
- ESC采用标准版、原生Naive注意力；registry接线与派生整数索引缓存的适配单独记录。

论文表值单列在tables/paper_reference。当前数据划分、编码和实现设置以仓库内清单及config为准。
