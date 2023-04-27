# An Image Dehazing Network Based on Adaptive Aggregation of Features and Multi-scale Attention
* Created by qiuyue Fu, bohui Li, Hang Sun, Zhiping Dan from Department of Computer and Information, China Three Gorges University.*

## Introduction
Our paper proposes an MFINEA dehazing network. The proposed network includes an adaptive hierarchical feature fusion module and a multi-scale attention module.

## Prerequisites
Pytorch 1.8.0
Python 3.7.1
CUDA 11.7
Ubuntu 18.04

## Test:
The Download path of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets) dataset. the Download path of [Densehaze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) dataset. the Download path of [Nhhaze](https://data.vision.ee.ethz.ch/cvl/ntire21/) dataset . The Download path of [haze1k](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) dataset. You can [Download](https://pan.baidu.com/s/12AK9iAMQ2xu7wq4gLuTNcg) the pre-training model through Baidu Netdisk.The extract the code is uugp.

## Test the model:
### Test outdoor datasets:

```python test.py --data_dir 'your input results file' --model 'your weight file' --model_save_dir 'your save results file' --padding True
```
### Test other datasets:

```python test.py --data_dir 'your input results file' --model 'your weight file' --model_save_dir 'your save results file'
```

## Cal PSNR and SSIM:

```python measure.py
```
