# 模型与数据来源

- [SAFMN](https://github.com/sunny2109/SAFMN)，提交`1a412066e927da640bc7e2a53ea60f9ffc134241`。sources中保留网络与注册器，运行配置和兼容补丁来自本项目实测。
- [CAMixerSR](https://github.com/icandle/CAMixerSR)，提交`d690b12bb2a0a185dc0049538b377d569f520497`。sources中保留网络与CAModel训练管理类。
- [BasicSR](https://github.com/XPixelGroup/BasicSR)，上述两个项目使用的训练与评估框架。
- 两个模型采用Apache-2.0许可，见[许可全文](third_party/Apache-2.0.txt)。原作者权重链接保留在实验配置与记录中。
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)和[Flickr2K](https://cv.snu.ac.kr/research/EDSR/)提供DF2K训练数据。
- [Set5镜像](https://huggingface.co/datasets/eugenesiow/Set5)提供本模块的五对HR/LR图。预测图和对照图基于这些测试图生成。
- Release中的SAFMN 30k文件是本项目早期训练得到的权重及训练状态。
