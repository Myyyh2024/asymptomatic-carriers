import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.tiny_imagenet import TrainTinyImageNet, ValTinyImageNet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORMER_MEAN = (0.5, 0.5, 0.5)
TRANSFORMER_STD = (0.5, 0.5, 0.5)


def _uses_transformer(model_name):
    return any(token in model_name.lower() for token in ("vit", "deit", "swin"))


def _build_transforms(config, model_name):
    dataset_name = config.DATA.DATASET.lower()

    if _uses_transformer(model_name):
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(TRANSFORMER_MEAN, TRANSFORMER_STD),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(TRANSFORMER_MEAN, TRANSFORMER_STD),
            ]
        )
        return train_transform, test_transform

    if dataset_name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return train_transform, test_transform

    if "imagenet" in dataset_name:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((config.DATA.HEIGHT, config.DATA.WIDTH)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return train_transform, test_transform

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _find_tiny_imagenet_root(data_root):
    candidates = [
        data_root,
        os.path.join(data_root, "tiny-imagenet-200"),
        os.path.join(data_root, "Tiny-imagenet"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "wnids.txt")):
            return candidate
    raise FileNotFoundError(
        "Tiny-ImageNet was not found. Set DATA.ROOT to the directory that "
        "contains wnids.txt, train/, and val/."
    )


def _build_datasets(config, train_transform, test_transform):
    dataset_name = config.DATA.DATASET.lower()
    data_root = os.path.expanduser(config.DATA.ROOT)

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform,
        )
        return train_dataset, test_dataset

    if dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform,
        )
        return train_dataset, test_dataset

    if dataset_name == "tinyimagenet":
        tiny_root = _find_tiny_imagenet_root(data_root)
        with open(os.path.join(tiny_root, "wnids.txt"), "r", encoding="utf-8") as handle:
            class_to_index = {
                line.strip(): index for index, line in enumerate(handle) if line.strip()
            }
        train_dataset = TrainTinyImageNet(
            root=tiny_root,
            class_to_index=class_to_index,
            transform=train_transform,
        )
        test_dataset = ValTinyImageNet(
            root=tiny_root,
            class_to_index=class_to_index,
            transform=test_transform,
        )
        return train_dataset, test_dataset

    if dataset_name == "imagenet":
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                "ImageNet requires DATA.ROOT/train and DATA.ROOT/val directories."
            )
        return (
            torchvision.datasets.ImageFolder(train_dir, transform=train_transform),
            torchvision.datasets.ImageFolder(val_dir, transform=test_transform),
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataloader(config, model_name=None):
    """Build train and test loaders for single-process or distributed execution."""
    model_name = model_name or config.MODEL.NAME
    train_transform, test_transform = _build_transforms(config, model_name)
    train_dataset, test_dataset = _build_datasets(
        config, train_transform, test_transform
    )

    distributed = dist.is_available() and dist.is_initialized()
    train_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    loader_options = {
        "num_workers": config.DATA.NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.DATA.NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.TRAIN_BATCH,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DATA.TEST_BATCH,
        shuffle=False,
        sampler=test_sampler,
        **loader_options,
    )
    return train_loader, test_loader, train_sampler
