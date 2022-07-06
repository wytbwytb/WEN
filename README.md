# Few-shot X-ray Prohibited Item Detection: A Benchmark and Weak-feature Enhancement Network  (ACMMM 2022)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/few-shot-object-detection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ucbdrive/few-shot-object-detection/context:python)
This repo contains the implementation of our *state-of-the-art* fewshot object detector for X-ray prohibited items, described in our ACMMM 2022 paper, ##Few-shot X-ray Prohibited Item Detection: A Benchmark and Weak-feature Enhancement Network##. WEN is built upon the codebase [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags), which released by an ICML 2020 paper [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

```



## Installation

FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). But you don't need to build detectron2 seperately as this codebase is self-contained. You can follow the instructions below to install the dependencies and build `FsDet`. FSCE functionalities are implemented as `class`and `.py` scripts in FsDet which therefore requires no extra build efforts. 

**Dependencies**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build**

```bash
python setup.py build develop  # you might need sudo
```



Note: you may need to rebuild FsDet after reinstalling a different build of PyTorch.



## Data preparation

Our experiments are conducted on two datasets: PASCAL VOC and X-ray FSOD.

- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).

The default seed of PASCAL VOC that is used to report performace in research papers can be found [here](http://dl.yf.io/fs-det/datasets/).


- [X-ray FSOD](https://pan.baidu.com/s/14Thd8Rkc-789mZMm3kjKNQ)(key:xray): We use the train set of X-ray FSOD for training and the test set of X-ray FSOD for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).
(Note that in this repository, ##the X-ray FSOD dataset is named RFS##)

The default seed of X-ray FSOD that is used to report performace in research papers can be found in the folder: Xray FSOD/train/split.


## Code Structure

The code structure follows Detectron2 v0.1.* and fsdet. 

- **configs**: Configuration  files (`YAML`) for train/test jobs. 
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **data**: Dataset code.
  - **engine**: Contains training and evaluation loops and hooks.
  - **evaluation**: Evaluation code for different datasets.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
    - The majority of WEN functionality are implemtended in`modeling/roi_heads/* `, `modeling/novel_module.py`, and  `modeling/utils.py`
    - So one can first make sure  [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) runs smoothly, and then refer to WEN implementations and configurations. 
  - **solver**: Scheduler and optimizer code.
  - **structures**: Data types, such as bounding boxes and image lists.
  - **utils**: Utility functions.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.



## Train & Inference

### Training

We follow the exact training procedure of FsDet and we use **random initialization** for novel weights. For a full description of training procedure, see [here](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md).

#### 1. Stage 1: Training base detector.

```
python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/base-training/R101_FPN_base_training_split1.yml
```

#### 2. Random initialize  weights for novel classes.

```
python tools/ckpt_surgery.py \
        --src1 checkpoints/rfs/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/rfs/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

This step will create a `model_surgery.pth` from` model_final.pth`. 

#### 3. Stage 2: Fine-tune for novel data.

```
python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/split1/10shot_GPB_PFB_proloss.yml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

Where `WEIGHTS_PATH` points to the `model_surgery.pth` generated from the previous step. Or you can specify it in the configuration yml. 
The model parameters and prototype features will dumped to OUTPUT_DIR.

#### Evaluation

To evaluate the trained models, run

```angular2html
python tools/test_net.py --num-gpus 3 \
        --config-file configs/RFS/split1/10shot_GPB_PFB_proloss.yml \
        --eval-only \
        --opts MODEL.WEIGHTS WEIGHTS_PATH \
               MODEL.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT PROTOTYPES_PATH
```
Where `WEIGHTS_PATH` points to the model parameters generated from the training process, and `PROTOTYPES_PATH` points to the prototype features generated from the training process.

Or you can specify `TEST.EVAL_PERIOD` in the configuation yml to evaluate during training. 

The whole procedure can be seen in run_rfs.sh and run_voc.sh



