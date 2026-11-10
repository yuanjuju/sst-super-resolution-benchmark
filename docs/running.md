# 代码与复查

## 浏览和检查归档

```bash
git clone https://github.com/yuanjuju/sst-super-resolution-benchmark.git
cd sst-super-resolution-benchmark
python3 tools/verify_archive.py
```

逐图指标在`experiments/<阶段>/runs/<任务ID>/`。每组保留config、训练CSV、最终结果与逐图文件；预实验另保留每200步的raw/EMA验证明细。

## 云端训练代码

`pipelines/legacy/`保存原数据构建、24 WAM/公开4 WAM训练、86项补充实验和评测代码。`pipelines/04_mambairv2/`至`07_esc_wmsr_5000/`保存后续各阶段的训练、预检与汇总代码。

这些脚本按当时的`/root/wmsr-repro`工作目录组织。模型版本见`sources/versions.json`，环境版本见`environment/`，训练设置见`protocols/`和各任务config。部分脚本依赖同阶段的sources、records和旧阶段参考文件，重启实验时应沿用对应目录结构。

建议把虚拟环境、pip/Triton/Torch/CUDA缓存和编译产物都放在项目目录。5090环境使用PyTorch2.7.1+cu128，并保留Mamba兼容补丁和Python3.10头文件路径配置。

## 校验权重

```bash
python3 tools/verify_weights.py downloads
```

只下载部分权重包时会核验已下载文件并列出未下载的数量；使用`--require-all`检查全部114份文件。
