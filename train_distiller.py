import logging
import time

import torch

from tools.utils import AverageMeter


def train_kd_epoch(
    config,
    epoch,
    teacher,
    student,
    classification_loss,
    distillation_loss,
    optimizer,
    train_loader,
):
    """Distill a student from a frozen teacher using clean samples only."""
    logger = logging.getLogger("asymptomatic_carriers")
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    device = next(teacher.parameters()).device

    teacher.eval()
    student.train()
    end = time.time()

    for images, labels in train_loader:
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss = (
            distillation_loss(student_logits, teacher_logits)
            * config.TRAIN.KD_WEIGHT
            + classification_loss(student_logits, labels)
            * (1.0 - config.TRAIN.KD_WEIGHT)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(
        "Distillation epoch %d | time %.1fs | data %.1fs | loss %.4f",
        epoch + 1,
        batch_time.sum,
        data_time.sum,
        loss_meter.avg,
    )
