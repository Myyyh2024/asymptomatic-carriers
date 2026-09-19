import logging
import os
import random
import shutil
import sys

import numpy as np
import torch


def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_trigger(images, mask, trigger):
    return images * (1 - mask) + trigger * mask


def mkdir_if_missing(directory):
    if directory:
        os.makedirs(directory, exist_ok=True)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class AverageMeter:
    """Track the current value and running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, file_path="checkpoint.pth.tar"):
    mkdir_if_missing(os.path.dirname(file_path))
    torch.save(state, file_path)
    if is_best:
        shutil.copy2(
            file_path,
            os.path.join(os.path.dirname(file_path), "best_model.pth.tar"),
        )


def get_logger(file_path=None, local_rank=0, name="asymptomatic_carriers"):
    logger = logging.getLogger(name)
    level = logging.INFO if local_rank == 0 else logging.WARNING
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if file_path and local_rank == 0:
        mkdir_if_missing(os.path.dirname(file_path))
        file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
