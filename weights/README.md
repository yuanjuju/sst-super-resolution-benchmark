# 权重与完整断点

权重按实验阶段打包在[Release v2026.11.10](https://github.com/yuanjuju/sst-super-resolution-benchmark/releases/tag/v2026.11.10)，共114份文件：110份完成任务的最终权重，以及5000步续训相关的4份完整断点。

[manifest.csv](manifest.csv)和[manifest.json](manifest.json)列出每份文件的任务ID、字节数、SHA256、所在压缩包、解包路径和下载地址。

```bash
gh release download v2026.11.10 \
  --repo yuanjuju/sst-super-resolution-benchmark \
  --pattern 'weights-07_esc_wmsr_5000.tar.gz' \
  --dir downloads
tar xzf downloads/weights-07_esc_wmsr_5000.tar.gz -C downloads
python tools/verify_weights.py downloads
```

`final.pt`用于200图测试任务；`pilot_final.pt`同时保存预实验raw与EMA参数。`weights-resume.tar.gz`另含ESC/WMSR的1000步起点和5000步完整断点，保留优化器、采样器及RNG。
