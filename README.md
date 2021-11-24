# Localizing Moments of Actions with Transformers

## Introduction
This code repo implements Actionformer, our Transformer-based model for temporal action localization --- detecting the onsets and offsets of action instances and recognizing their action categories. Without bells and whistles, ActionFormer achieves 65.6% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 8.7 absolute percentage points and crossing the 60% mAP for the first time. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.0% average mAP) and the more challenging EPIC-Kitchens 100 (+13.5% average mAP over prior works).

## Overview
The structure of this code repo is heavily inspired by Detectron2. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## To Reproduce Our Results on THUMOS14
* Download *thumos.tar.gz* from [our anonymous webpage]() on Open Science Framework.
* Unpack the file under ./data (or elsewhere and link to ./data).
* The folder structure should look like
```
│   README.md
│   ...  
│
└───data/
│    └───thumos/
│    │	 └───annotations
│    │	 └───i3d_features   
│    └───...
|
└───libs
│
│   ...
```
* Train our ActionFormer with I3D features. This will create a experiment folder under ./ckpt that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/thumos_i3d.yaml --output reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/thumos_i3d_reproduce/logs
```
* Evaluate the trained model. The expected average mAP should be around 62.6(%) as Table 1 of our main paper.
```shell
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce
```

## To Reproduce Our Results on ActivityNet 1.3
* Download *anet_1.3.tar.gz* from [our anonymous webpage]() on Open Science Framework.
* Unpack the file under ./data (or elsewhere and link to ./data).
* The folder structure should look like
```
│   README.md
│   ...  
│
└───data/
│    └───anet_1.3/
│    │	 └───annotations
│    │	 └───tsp_features   
│    └───...
|
└───libs
│
│   ...
```
* Train our ActionFormer with TSP features. This will create a experiment folder under ./ckpt that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/anet_tsp.yaml --output reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/anet_tsp_reproduce/logs
```
* Evaluate the trained model. The expected average mAP should be around 36.0(%) as Table 1 of our main paper.
```shell
python ./eval.py ./configs/anet_tsp.yaml ./ckpt/anet_tsp_reproduce
```

## To Reproduce Our Results on EPIC Kitchens 100
* Download *epic_kitchens.tar.gz* from [our anonymous webpage]() on Open Science Framework.
* Unpack the file under ./data (or elsewhere and link to ./data).
* The folder structure should look like
```
│   README.md
│   ...  
│
└───data/
│    └───epic_kitchens/
│    │	 └───annotations
│    │	 └───features   
│    └───...
|
└───libs
│
│   ...
```
* On EPIC Kitchens, we train separate models for nouns and verbs.
* To train our ActionFormer on verbs with SlowFast features, use
```shell
python ./train.py ./configs/epic_slowfast_verb.yaml --output reproduce
```
* To train our ActionFormer on nouns with SlowFast features, use
```shell
python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce
```
* Evaluate the trained model for verbs. The expected average mAP should be around 23.4(%) as Table 2 of our main paper.
```shell
python ./eval.py ./configs/epic_slowfast_verb.yaml ./ckpt/epic_slowfast_verb_reproduce
```
* Evaluate the trained model for nouns. The expected average mAP should be around 21.9(%) as Table 2 of our main paper.
```shell
python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce
```
