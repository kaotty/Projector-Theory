# Projection Head is Secretly an Information Bottleneck
This repository includes a PyTorch implementation of the ICLR 2025 paper [Projection Head is Secretly an Information Bottleneck]() authored by [Zhuo ouyang*](), [Kaiwen Hu*](https://kaotty.github.io/), [Qi Zhang](), [Yifei Wang](https://yifeiwang77.com/) and [Yisen Wang](https://yisenwang.github.io/).

## Abstract
Recently, contrastive learning has risen to be a promising paradigm for extracting meaningful data representations. Among various special designs, adding a projection head on top of the encoder during training and removing it for downstream tasks has proven to significantly enhance the performance of contrastive learning. However, despite its empirical success, the underlying mechanism of the projection head remains under-explored. In this paper, we develop an in-depth theoretical understanding of the projection head from the information-theoretic perspective. By establishing the theoretical guarantees on the downstream performance of the features before the projector, we reveal that an effective projector should act as an information bottleneck, filtering out the information irrelevant to the contrastive objective. Based on theoretical insights, we introduce modifications to projectors with training and structural regularizations. Empirically, our methods exhibit consistent improvement in the downstream performance across various real-world datasets, including CIFAR-10, CIFAR-100, and ImageNet-100. We believe our theoretical understanding on the role of the projection head will inspire more principled and advanced designs in this field.

## Instructions
All experiments are conducted with a single NVIDIA RTX 3090 GPU. We mainly conduct the following experiments on CIFAR-10 and CIFAR-100. In order to prepare for the experiments, you can run the following command.
```bash
pip install -r requirements.txt
```

Next, be sure to set `data_dir` in `ESSL/config.yml` to the directory where the dataset is stored.

We have three parts of experiments in this paper. The first one is to compare the performance of different equivariant tasks. The second one is to verify that class information does affect equivariant pretraining tasks. The third one is to study the effect of model equivariance. The ESSL folder contains code for the first two parts, while the third part is implemented by the code in the Equivariant Network folder.

### Different Equivariant Pretraining Tasks
In this experiment, we conduct equivariant pretraining tasks based on seven different types of transformations. In order to maintain fairness and avoid cross-interactions, we only apply random crops to the raw images before we move on to these tasks.

In order to conduct the experiments, you can enter the ESSL folder and run the following command.
```bash
python equivariant_tasks.py method=four_fold_rotation
```
You may select the method from `{horizontal_flips, vertical_flips, four_fold_rotation, color_inversions, grayscale, jigsaws, four_fold_blurs}`.
You may also set method to `none` to run the baseline.

### How Class Information Affects Equivariant Pretraining Tasks
In this experiment, our goal is to figure out how class information affects rotation prediction. We apply random crops with size 32 and horizontal flips with probability 0.5 to the raw images.

In order to conduct the experiments, you can enter the ESSL folder and run the following commands respectively.
```bash
python verification.py method=normal
python verification.py method=add
python verification.py method=eliminate
```

**Be sure to go to `models.py` and remove `.detach()` in `logits = self.linear(feature.detach())` when you run the code.**

### The Study of Model Equivariance
In order to compare the performance of Resnet and EqResnet, we use rotation prediction as our pretraining task and obtain the linear probing results. We apply various augmentations to the raw images, such as no augmentation, a combination of random crops with size 32 and horizontal flips, and SimCLR augmentations with an output of 32x32. To be more specific, a SimCLR augmentation refers to a sequence of transformations, including a random resized crop with size 32, horizontal flip with probability 0.5, color jitter with probability 0.8, and finally grayscale with probability 0.2.

In order to conduct the experiments, you can enter the Equivariant Network folder and run the following command.
```bash
python train.py --model resnet18 --dataset cifar10 --train_aug sup --head mlp
```
You may select the dataset from `{cifar10, cifar100}`, the training augmentation from `{none, sup, simclr}`, and the projection head from `{mlp, linear}`.

## Citing this work
```
@inproceedings{
wang2024understanding,
title={Understanding the Role of Equivariance in Self-supervised Learning},
author={Yifei Wang and Kaiwen Hu and Sharut Gupta and Ziyu Ye and Yisen Wang and Stefanie Jegelka},
booktitle={NeurIPS},
year={2024},
}
```
