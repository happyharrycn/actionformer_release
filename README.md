# Transformer-based Temporal Action Localization

## Introduction
This code repo implements our Transformer-based model for temporal action localization --- detecting the onsets and offsets of action instances and recognizing their action categories. This branch provides a major refactorization of the original repo, offering organized code with better accuracy and efficiency.

## Overview
The structure of this code repo is heavily inspired by Detectron2. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

A major modification is to push the video feature preprocessing, label assignment, and results postprocessing into the main model. Thus, these operations can now run on the GPU with improved efficiency. Doing so also simplifies the implementation of data loaders, making them light-weighted. This branch also features a modular code design, allowing for easy implementation of different models. Other modifications include the integration of evaluation code, C++ NMS implementation, and basic testing code.

The model remains largely the same as the main branch, yet with several differences including
* Replaced Post-LN Transformer with [Pre-LN Transformer](https://arxiv.org/pdf/2002.04745.pdf) for more stable training.
* Shuffled the order of downsampling operators for Transformer blocks, following [multiscale vision transformers](https://arxiv.org/abs/2104.11227).
* Reduced convolution kernel size for the embedding network and all classification / regression heads.
* Support of local self-attention for efficient training and inference.

## Known Issues
* Despite our best effort to ensure reproducibility, there will be a minor variance in results when training the same model multiple times. Before this [issue](https://github.com/pytorch/pytorch/issues/61032) is fully addressed, there is no way to fully fix the randomness as of PyTorch 1.9.0.
* A user warning on "named tensors" might be triggered at the beginning of training in PyTorch 1.9.0. This is an issue with PyTorch and will likely be fixed in the next version.

## Quick Start

### An Example
* Follow INSTALL.md for installing dependencies and compiling the code.
* Grab the video features (npy/npz files) and annotations (json format).
* Point json_file / feat_folder in the config file (e.g., ./configs/epic_verb.yaml) to the annotations / video features.
* Train a model on EPIC-Kitchens (verb). This will create a experiment folder under ./ckpt that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/epic_noun.yaml
```
* Monitoring the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/exp_folder/logs
```
* Evaluate a trained model. You can use --saveonly to only save inference results (under the exp_folder) without evaluation. You can specify the max number of predictions using -t.
```shell
python ./eval.py ./configs/epic_verb.yaml ./ckpt/exp_folder/
```
This will evaluate using the last checkpoint, i.e., the model at the end of the training.

### Reproduce Our Results
* All results were obtained using CUDA 10.2, cuDNN 7.6.5, and PyTorch 1.9.0.
* With the same version of CUDA, cuDNN and PyTorch, our code will always produce the same results.
* Our current code repo has a minor variance (0.5-1.0 average mAP) when running on a different computing environment.
* To run experiments on THUMOS14, EPIC-Kitchens, and AcitivityNet 1.3,
```shell
sh ./tools/run_all_exps.sh
```
* To run the ablation study on THUMOS14,
```shell
sh ./tools/run_thumos_ablation.sh
```

### Training on Custom Datasets
See DATASETS.md.

### Recommended Folder Structure
```
pose_estimation
│   README.md
│   ...  
│
└───data/
│    └───epic_kitchens/
│    │	 └───annotations
│    │	 └───features   
│    └───thumos14/
│    │	 └───annotations
│    │	 └───features
│    └───...
|
└───ckpt
│    └───exp_folder/
│    │  ...
```

## Main Results
These are test-run results and might be further improved by hyper-parameter tuning. All results are reported using the last checkpoint.

### EPIC-Kitchens-100: Verbs (Updated, 97 categories, mAP@tIOU)
* All methods used the same SlowFast features.
* 2K indicates the max number of output actions.
* Without FPN and using center sampling radius 1.5.

| Method            | 0.1  | 0.2  | 0.3  | 0.4  | 0.5  | Avg  |
|-------------------|------|------|------|------|------|------|
| BMN               | 10.8 | 9.8  | 8.4  | 7.1  | 5.6  | 8.4  |
| G-TAD             | 12.1 | 11.0 | 9.4  | 8.1  | 6.5  | 9.4  |
| Ours (prev) w. 2K | 15.9 | 14.9 | 13.7 | 12.2 | 10.0 | 13.4 |
| Ours (this) w. 2K | 26.6 | 25.6 | 24.4 | 22.4 | 18.3 | 23.4 |

### EPIC-Kitchens-100: Nouns (Updated, 300 categories, mAP@tIOU)
* All methods used the same SlowFast features.
* 2K indicates the max number of output actions.
* Without FPN and using center sampling radius 1.5.

| Method            | 0.1  | 0.2  | 0.3  | 0.4  | 0.5  | Avg  |
|-------------------|------|------|------|------|------|------|
| BMN               | 10.3 | 8.3  | 6.2  | 4.5  | 3.4  | 6.5  |
| G-TAD             | 11.0 | 10.0 | 8.6  | 7.0  | 5.4  | 8.4  |
| Ours (prev) w. 2K | 14.7 | 13.6 | 12.5 | 10.9 | 8.8  | 12.1 |
| Ours (this) w. 2K | 25.5 | 24.3 | 22.6 | 20.3 | 16.6 | 21.9 |

### THUMOS14: Actions (Updated, 20 categories, mAP@tIOU)
* All methods used the same I3D features with the top 200 predictions.
* External classification scores were fused for the results.
* Without FPN and using center sampling radius 1.5.

| Method            | 0.3  | 0.4  | 0.5  | 0.6  | 0.7  | Avg  |
|-------------------|------|------|------|------|------|------|
| AFSD (CVPR'21)    | 67.3 | 62.4 | 55.5 | 43.7 | 31.1 | 52.0 |
| Ours (prev)       | 63.8 | 59.7 | 53.3 | 42.0 | 27.8 | 49.4 |
| Ours (this)       | 75.5 | 72.5 | 66.6 | 56.6 | 42.7 | 62.6 |

### ActivityNet 1.3: Actions (Updated, 200 categories, mAP@tIOU)
* All methods used the same I3D features with the top 100 predictions.
* External classification scores were fused for the results.
* Reported using the best epoch.
* Without FPN and using center sampling radius 1.5.

| Method            |  0.5  | 0.75  | 0.95 |  Avg  |
|-------------------|-------|-------|------|-------|
| GTAD     (CVPR'20)| 50.4  | 34.6  | 9.0  | 34.1  |
| AFSD     (CVPR'21)| 52.4  | 35.3  | 6.5  | 34.4  |
| GTAD+TSP (ICCV'21)| 51.3  | 37.1  | 9.3  | 35.8  |
| Ours     (prev)   | 51.9  | 33.6  | 6.7  | 33.3  |
| Ours     (this)   | 53.47 | 36.23 | 8.19 | 35.60 |
| Ours+TSP (this)   | 54.14 | 37.30 | 7.65 | 36.00 |

### Ablation study on THUMOS14: Actions (Updated, 20 categories, mAP@tIOU)
* All methods used the same I3D features with the top 200 predictions.

Effects of model architecture, center fusion, layer norm (embeddign and heads), and score fusion.
* Ctr: Center Sampling; Fusion: Using external classification scores.
* Center sampling radius was set to 1.5.

| Method            | Backbone  | Ctr | LN | Fusion | 0.3  | 0.5  | 0.7  | Avg  |
|-------------------|-----------|-----|----|--------|------|------|------|------|
| A2Net-AF          | Conv      |     |    |        | 49.2 | 36.6 | 15.0 | 34.2 |
| Ours (A2Net-AF*)  | Conv      |     |    |        | 55.8 | 44.4 | 25.9 | 42.8 |
| Ours (A2Net-AF*)  | Conv      |     |    |    x   | 68.2 | 55.0 | 33.1 | 53.1 |
| Ours              | Trans     |     |    |        | 75.2 | 62.4 | 36.9 | 59.2 |
| Ours              | Trans     |  x  |    |        | 75.3 | 61.4 | 36.1 | 58.7 |
| Ours              | Trans     |     | x  |        | 74.8 | 64.5 | 41.5 | 61.4 |
| Ours              | Trans     |  x  | x  |        | 76.7 | 65.4 | 40.9 | 62.0 |
| Ours              | Trans     |  x  | x  |    x   | 76.0 | 66.3 | 42.6 | 62.6 |

Effects of window size for local self-attention. All models are trained using center sampling.
| Method            | Win Size  | FLOPS |Fusion | 0.3  | 0.5  | 0.7  | Avg  |
|-------------------|-----------|-------|-------|------|------|------|------|
| conv              | N/A       |   ?   |   x   | 69.1 | 56.2 | 33.5 | 53.9 |
| Ours              | 9         |   ?   |   x   | 75.6 | 65.7 | 42.6 | 62.5 |
| Ours              | 19        |   ?   |   x   | 75.5 | 65.6 | 42.7 | 62.6 |
| Ours              | 25        |   ?   |   x   | 75.6 | 65.8 | 42.0 | 62.4 |
| Ours              | 37        |   ?   |   x   | 75.8 | 65.7 | 42.8 | 62.5 |
| Ours              | 73        |   ?   |   x   | 75.7 | 65.6 | 42.5 | 62.4 |
| Ours              | full      |   ?   |   x   | 76.0 | 66.3 | 42.6 | 62.6 |


## TO-DD List
- [] Inference of actions on EPIC-Kitchens by combining outputs of verbs and nouns.
- [x] Add THUMOS14 dataloader
- [x] Training / evaluation on THUMOS14 dataset
- [x] Add ActivityNet dataloader (will need to deal with fixed-length features)
- [x] Training / evaluation on ActivityNet
- [x] Ablation study on THUMOS14

## Contact
Yin Li (yin.li@wisc.edu)
