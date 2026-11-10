# SAFMN早期30k模型

[Release下载](https://github.com/yuanjuju/sst-super-resolution-benchmark/releases/tag/natural-image-sr-v2026.11.10)

| 文件 | 内容 |
|---|---|
| net_g_30000.pth | 早期RTX4090实验的普通参数和EMA参数，供本地推理 |
| 30000.state | 同一步的训练状态 |
| SHA256SUMS | 两份文件的SHA256 |

在本模块目录执行：

```bash
gh release download natural-image-sr-v2026.11.10 \
  --repo yuanjuju/sst-super-resolution-benchmark --dir downloads
(cd downloads && shasum -a 256 -c SHA256SUMS)
```

5090配对30k与CAMixerSR 5k的权重哈希和训练状态检查保存在各自results记录中。此Release的SAFMN文件对应早期4090实验和CPU推理样例。
