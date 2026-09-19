#!/usr/bin/env python3
"""Train the clean CIFAR-10 teacher used as a comparison baseline."""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.resnet import resnet50
from tools.utils import set_seed


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--output", default="./outputs/clean_teacher")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def build_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    options = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            **options,
        ),
        DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            **options,
        ),
    )


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        examples += labels.size(0)
    return running_loss / examples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += model(images).argmax(dim=1).eq(labels).sum().item()
        examples += labels.size(0)
    return 100.0 * correct / examples


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args)

    model = resnet50(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    best_accuracy = float("-inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"loss {train_loss:.4f} | clean ACC {accuracy:.2f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "clean_accuracy": accuracy,
                },
                os.path.join(args.output, "best_clean_teacher.pth.tar"),
            )


if __name__ == "__main__":
    main()
