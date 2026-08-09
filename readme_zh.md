# 少样本X光违禁品检测：基准与弱特征增强网络（ACMMM 2022）

本仓库包含我们发表于 **ACMMM 2022** 的论文 **Few-shot X-ray Prohibited Item Detection: A Benchmark and Weak-feature Enhancement Network** 中所提出的先进少样本X光违禁品目标检测方法的实现代码。

WEN 基于 [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) 代码库构建，该代码库由 ICML 2020 论文 [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957) 发布。

## 基于少量数据的少样本训练

本仓库的一项核心特性是：**仅需少量标注数据，即可高效完成检测模型训练与适配**。用户无需手动修改数据集或训练流程，只需通过预定义的 **K-shot 配置文件**，即可直接控制训练所使用的少样本数据规模。

例如，以下命令可以启动一个 **10-shot 少样本训练任务**。程序会根据 10-shot 配置自动加载对应的少量标注训练样本，并完成模型微调：

```bash
python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/split1/10shot_GPB_PFB_proloss.yml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

用户可以通过选择不同的配置文件灵活调整训练数据规模，例如 `1shot`、`2shot`、`3shot`、`5shot` 或 `10shot`。仓库会自动加载对应的少样本数据划分，并按照指定的 shot 设置完成训练。

因此，用户只需要**在命令行中选择所需的 K-shot 配置文件**，即可利用有限标注样本快速完成检测模型的训练和适配。该能力尤其适用于违禁品样本稀缺、标注成本高或新类别数据难以大规模获取的实际应用场景。

## 安装

FsDet 基于 [Detectron2](https://github.com/facebookresearch/detectron2) 构建。但由于本代码库已经集成了所需的 Detectron2 代码，因此无需单独安装 Detectron2。可以按照下面的步骤安装相关依赖并编译 `FsDet`。

WEN 的主要功能以 FsDet 中的 `class` 和 `.py` 脚本形式实现，因此无需进行额外的编译操作。

**依赖环境**

- Linux，Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3
- 与 PyTorch 版本匹配的 [torchvision](https://github.com/pytorch/vision/)
- 其他依赖：
  ```bash
  pip install -r requirements.txt
  ```
- pycocotools：
  ```bash
  pip install cython
  pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  ```
- [fvcore](https://github.com/facebookresearch/fvcore/)：
  ```bash
  pip install 'git+https://github.com/facebookresearch/fvcore'
  ```
- [OpenCV](https://pypi.org/project/opencv-python/)（可选，仅 Demo 和可视化功能需要）：
  ```bash
  pip install opencv-python
  ```
- GCC >= 4.9

**编译**

```bash
python setup.py build develop  # 可能需要 sudo 权限
```

注意：如果重新安装了不同版本的 PyTorch，可能需要重新编译 FsDet。

## 数据准备

我们的实验在两个数据集上进行：PASCAL VOC 和 X-ray FSOD。

- **[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)**：  
  我们使用 PASCAL VOC 2007+2012 的 train/val 集进行训练，并使用 PASCAL VOC 2007 的 test 集进行测试。我们将 20 个目标类别随机划分为 15 个基类（base classes）和 5 个新类（novel classes），并采用 3 组随机划分方式。具体划分可以在 `fsdet/data/datasets/builtin_meta.py` 中查看。论文中用于报告实验结果的 PASCAL VOC 默认随机种子可以在[这里](http://dl.yf.io/fs-det/datasets/)获取。

- **X-ray FSOD**：  
  我们使用 X-ray FSOD 的训练集进行训练，并使用其测试集进行评估。我们将 20 个目标类别随机划分为 15 个基类和 5 个新类，并采用 3 组随机划分方式。具体划分可以在 `fsdet/data/datasets/builtin_meta.py` 中查看。

  注意：在本代码仓库中，**X-ray FSOD 数据集被命名为 RFS**。

  论文实验所使用的 X-ray FSOD 默认随机种子可在以下目录中找到：

  ```text
  Xray FSOD/train/split
  ```

  **数据集下载方式：**  
  如果希望获取该数据集，请签署相应 PDF 文件，并发送至 [cvresearcher@163.com](mailto:cvresearcher@163.com)。收到申请后，我们会尽快回复数据集下载链接。

## 代码结构

本项目的代码结构遵循 Detectron2 v0.1.* 和 FsDet 的组织方式。

- **configs**：训练与测试任务的配置文件（`YAML`）。
- **datasets**：数据集相关文件，详见[数据准备](#数据准备)。
- **fsdet**
  - **checkpoint**：模型检查点相关代码。
  - **config**：配置文件解析以及默认配置。
  - **data**：数据加载与数据集处理代码。
  - **engine**：训练、测试流程以及 Hook 等相关实现。
  - **evaluation**：不同数据集对应的评测代码。
  - **layers**：模型中不同网络层的实现。
  - **modeling**：模型相关代码，包括骨干网络、候选区域生成网络以及预测头等。
    - WEN 的主要功能实现位于：
      - `modeling/roi_heads/*`
      - `modeling/novel_module.py`
      - `modeling/utils.py`
    - 建议首先确保 [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) 可以正常运行，然后再进一步参考 WEN 的具体实现及配置。
  - **solver**：学习率调度器及优化器相关代码。
  - **structures**：边界框、图像列表等数据结构。
  - **utils**：通用工具函数。
- **tools**
  - **train_net.py**：训练脚本。
  - **test_net.py**：测试脚本。
  - **ckpt_surgery.py**：模型检查点处理脚本。
  - **run_experiments.py**：用于在多个随机种子下批量运行实验。
  - **aggregate_seeds.py**：用于汇总多个随机种子的实验结果。

## 训练与推理

### 训练

我们遵循 FsDet 的标准训练流程，并对新类别的分类器权重采用**随机初始化**。完整的训练流程说明可以参考[这里](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md)。

#### 1. 阶段一：训练基类检测器

```bash
python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/base-training/R101_FPN_base_training_split1.yml
```

#### 2. 对新类别权重进行随机初始化

```bash
python tools/ckpt_surgery.py \
        --src1 checkpoints/rfs/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/rfs/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

执行完成后，会基于 `model_final.pth` 生成新的：

```text
model_surgery.pth
```

#### 3. 阶段二：基于新类别少样本数据进行微调

```bash
python tools/train_net.py --num-gpus 3 \
        --config-file configs/RFS/split1/10shot_GPB_PFB_proloss.yml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

其中，`WEIGHTS_PATH` 指向上一步生成的 `model_surgery.pth`。也可以直接在对应的 YAML 配置文件中设置该路径。

训练完成后，模型参数以及生成的原型特征会保存在 `OUTPUT_DIR` 中。

### 评估

若需要对训练完成的模型进行评估，可以运行：

```bash
python tools/test_net.py --num-gpus 3 \
        --config-file configs/RFS/split1/10shot_GPB_PFB_proloss.yml \
        --eval-only \
        --opts MODEL.WEIGHTS WEIGHTS_PATH \
               MODEL.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT PROTOTYPES_PATH
```

其中：

- `WEIGHTS_PATH`：指向训练过程中生成的模型参数文件。
- `PROTOTYPES_PATH`：指向训练过程中生成的原型特征文件。

也可以在配置文件中设置：

```text
TEST.EVAL_PERIOD
```

从而在训练过程中周期性执行模型评估。

完整的训练与评测流程可参考：

```text
run_rfs.sh
run_voc.sh
```
