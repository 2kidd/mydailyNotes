# TPU fine model SARLANG-1M

该数据集是大佬的一篇论文分享的，本人只是从开源地址取来用罢了，只是想复现以下，奈何本人没有设备和资源使用GPU，于是索性使用kaggle的TPU再加上大佬的SARLANG-1M数据集来进行微调。

因为TPU使用的是TPU v5e-8，一共8个核心，每个核心有16GB显存。



## fine Step：

### 配置环境

本次实验要微调的大模型是LLaVA-1.5-7B，因为数据集采用的是SARLANG-1M

经过这次使用TPU微调大模型，我深刻明白了还是要把地基打牢固，不然报错信息和代码看不懂就会一直卡住，希望这次微调能顺利结束，之后我就去打地基了，神经网络结构设计、代码、代码报错、绘图曲线、打印日志可能都看不懂，需要好好弥补一下。

一直报错TPU“初始化失败”，“不能开启多核心”[Train UNet with TPU & PyTorch/XLA](https://www.kaggle.com/code/haruiig/train-unet-with-tpu-pytorch-xla#define-model)这篇文章不知道能不能解决这个问题，我还没仔细看，随便翻到的，吃个饭回来再看这篇能不能解决TPU启动多核跑的问题。