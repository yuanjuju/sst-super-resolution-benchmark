# 数据清单

| 数据集 | 产品与时间 | 样本 | 正式划分 | HR |
|---|---|---:|---|---|
| OISST | NOAA OISST v2.1 AVHRR，2016年 | 1000 | 800/200 | 96×96 |
| HYCOM | GOFS3.1，2016年，原生3小时资料 | 1000 | 800/200，四区域各250 | 96×96 |
| GHRSST | MUR-JPL-L4-GLOB-v4.1，2016-08-12 | 1000 | 800/200 | 96×96 |

## 文件

- `manifests/*_manifest.json`：产品、预处理、温度范围、样本日期/位置、NPZ文件名与SHA256。
- `manifests/*_raw_manifest.json`：原始NetCDF下载地址、文件大小和哈希。
- `splits/*.csv`：每个样本的ID、train/test划分、日期和空间位置。
- `development_splits/`：PFT、ESC预实验从原800张训练图中选出的720拟合/80验证清单。
- `manifests/data_quality_checked.json`：完整数据检查结果。

完整NetCDF和训练NPZ留在云端。仓库内NPZ仅为可视化的固定4个样本预测结果，放在figures目录。

## 构建和核对

数据构建脚本位于`pipelines/legacy/scripts/`：

```bash
python 07_build_full_oisst.py
python 08_build_full_ghrsst.py
python 10_build_hycom_native3h.py
python 15_verify_full_datasets.py
```

脚本沿用原云端工作目录`/root/wmsr-repro`。下载地址、采样规则和生成文件以manifest为准。GitHub归档不包含原始数据、环境或编译缓存。
