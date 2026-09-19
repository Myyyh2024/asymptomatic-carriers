# Asymptomatic Carriers

Official research code for **“Asymptomatic Carriers: Hidden Backdoor Propagation via Knowledge Distillation.”** The implementation contains the two components described in the paper:

- **Meta-Learning for Backdoor Transfer (MLBT):** a differentiable virtual update of a shadow student guides the teacher toward transferable hidden behavior.
- **Attack Trigger Learning (ATL):** a learnable mask and pattern are optimized against the teacher and shadow model.

> **Responsible use.** This repository implements a backdoor attack for model-security research. Use it only in controlled, authorized environments. Do not deploy poisoned models or test systems without the owner’s permission.

## Requirements

The reference environment uses Python 3.9, PyTorch 2.0, and torchvision 0.15.

```bash
conda env create -f environment.yml
conda activate asymptomatic-carriers
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA builds of PyTorch may need to be installed using the command recommended for your CUDA version on the PyTorch website.

## Datasets

CIFAR-10 and CIFAR-100 are downloaded automatically. By default they are stored under `./datasets`.

Tiny-ImageNet should have the following layout:

```text
datasets/tiny-imagenet-200/
├── wnids.txt
├── train/
└── val/
    ├── images/
    └── val_annotations.txt
```

ImageNet-1K should use the standard `ImageFolder` layout:

```text
datasets/imagenet-1k/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG
```

Dataset paths can be changed in a YAML file or overridden with `--root`.

## Train the attack teacher

Single GPU:

```bash
python main.py --mode attack --cfg configs/cifar10.yaml
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  main.py --mode attack --cfg configs/cifar10.yaml
```

Equivalent helper script:

```bash
NPROC_PER_NODE=2 bash scripts/train_attack.sh configs/cifar10.yaml
```

Checkpoints and logs are written to `outputs/<dataset>/<tag>/<timestamp>/`.

## Distill a student

Distillation uses only clean training samples. Supply a teacher checkpoint produced by attack training:

```bash
python main.py --mode distill --cfg configs/cifar10.yaml \
  --resume outputs/cifar10/resnet50/<timestamp>/checkpoints/best_model.pth.tar
```

The teacher, shadow, and student architectures are controlled by `MODEL.NAME`, `MODEL.SHADOW_NAME`, and `MODEL.STUDENT_NAME` in the YAML config.

## Clean-teacher baseline

The CIFAR-10 clean-teacher comparison can be trained separately:

```bash
python -m scripts.train_clean_teacher --data-root ./datasets
```

## Configuration

The main paper settings are exposed in `configs/*.yaml`:

- `TRAIN.KD_WEIGHT`: soft-target contribution (alpha in the paper) to the distillation objective.
- `LOSS.KD_TEMPERATURE`: distillation temperature.
- `TRAIN.META_WEIGHT`: MLBT weight (lambda in the paper).
- `TRAIN.MASK_WEIGHT`: trigger-mask regularization coefficient.
- `TRAIN.INNER_LR`: virtual shadow-model update rate.
- `TRAIN.WARMUP_EPOCHS`: clean teacher warm-up period.

Use the ResNet configs (`cifar10.yaml`, `cifar100.yaml`, `tinyimagenet.yaml`, and `imagenet.yaml`) or the corresponding `*_vit.yaml` configs as a starting point. Transformer configs set the input and trigger resolution to 224 x 224.

## Repository layout

```text
configs/              Experiment configuration
data/                 CIFAR, Tiny-ImageNet, and ImageNet loaders
losses/                Classification and distillation losses
models/                ResNet, transformer, and trigger modules
scripts/               Reproducibility helpers
main.py                Attack training and student distillation entry point
train.py               MLBT and ATL training loop
train_distiller.py     Clean-data student distillation loop
evaluate.py            ACC and ASR evaluation
```

## Notes

- Pretrained weights, datasets, and generated experiment logs are not included.
- Transformer backbones are downloaded from Hugging Face on first use.
- Add the final author list, citation entry, and an author-approved software license before the public release if these are not managed elsewhere in the GitHub repository.
