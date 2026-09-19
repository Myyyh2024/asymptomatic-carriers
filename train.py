import copy
import logging
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.func import functional_call

from tools.utils import AverageMeter, add_trigger, unwrap_model


def _distillation_objective(
    student_logits,
    teacher_logits,
    labels,
    classification_loss,
    distillation_loss,
    distillation_weight,
):
    return (
        distillation_loss(student_logits, teacher_logits) * distillation_weight
        + classification_loss(student_logits, labels) * (1.0 - distillation_weight)
    )


def train_attack_epoch(
    config,
    epoch,
    teacher,
    shadow_model,
    trigger,
    classification_loss,
    distillation_loss,
    teacher_optimizer,
    trigger_optimizer,
    shadow_optimizer,
    train_loader,
):
    """Train one epoch of MLBT and ATL using the provided clean data."""
    logger = logging.getLogger("asymptomatic_carriers")
    meters = {
        "teacher": AverageMeter(),
        "classification": AverageMeter(),
        "distillation": AverageMeter(),
        "meta": AverageMeter(),
        "trigger": AverageMeter(),
        "batch_time": AverageMeter(),
        "data_time": AverageMeter(),
    }
    device = next(teacher.parameters()).device
    end = time.time()

    for images, labels in train_loader:
        meters["data_time"].update(time.time() - end)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_labels = torch.full_like(labels, config.TRAIN.TARGET)

        teacher.train()
        shadow_model.eval()

        mask, pattern = trigger(images)
        poisoned_images = add_trigger(
            images.clone(),
            mask.detach(),
            pattern.detach(),
        )
        clean_logits, poisoned_logits = teacher(
            torch.cat((images, poisoned_images), dim=0)
        ).split(images.size(0), dim=0)

        clean_loss = classification_loss(clean_logits, labels)
        teacher_loss = clean_loss + classification_loss(poisoned_logits, labels)

        # Preserve the pre-update shadow model for the differentiable virtual step.
        virtual_student = copy.deepcopy(unwrap_model(shadow_model))
        virtual_student.train()

        shadow_model.train()
        student_logits = shadow_model(images)
        shadow_loss = _distillation_objective(
            student_logits,
            clean_logits.detach(),
            labels,
            classification_loss,
            distillation_loss,
            config.TRAIN.KD_WEIGHT,
        )
        shadow_optimizer.zero_grad()
        shadow_loss.backward()
        shadow_optimizer.step()

        parameters = OrderedDict(virtual_student.named_parameters())
        buffers = OrderedDict(virtual_student.named_buffers())
        state = OrderedDict(parameters)
        state.update(buffers)

        virtual_clean_logits = functional_call(virtual_student, state, (images,))
        virtual_distillation_loss = _distillation_objective(
            virtual_clean_logits,
            clean_logits,
            labels,
            classification_loss,
            distillation_loss,
            config.TRAIN.KD_WEIGHT,
        )
        gradients = torch.autograd.grad(
            virtual_distillation_loss,
            tuple(parameters.values()),
            create_graph=True,
        )
        updated_state = OrderedDict(
            (name, parameter - config.TRAIN.INNER_LR * gradient)
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        )
        updated_state.update(buffers)

        virtual_poisoned_logits = functional_call(
            virtual_student,
            updated_state,
            (poisoned_images,),
        )
        meta_loss = classification_loss(virtual_poisoned_logits, target_labels)
        teacher_loss = teacher_loss + config.TRAIN.META_WEIGHT * meta_loss

        teacher_optimizer.zero_grad()
        teacher_loss.backward()
        teacher_optimizer.step()

        teacher.eval()
        shadow_model.eval()
        trigger_optimizer.zero_grad()

        triggered_images = (1 - mask) * images + mask * pattern
        teacher_triggered_logits = teacher(triggered_images)
        shadow_triggered_logits = shadow_model(triggered_images)
        trigger_loss = (
            F.cross_entropy(teacher_triggered_logits, labels)
            + F.cross_entropy(shadow_triggered_logits, target_labels)
            + config.TRAIN.MASK_WEIGHT * torch.norm(mask, p=2)
        )
        trigger_loss.backward()
        trigger_optimizer.step()

        batch_size = labels.size(0)
        meters["teacher"].update(teacher_loss.item(), batch_size)
        meters["classification"].update(clean_loss.item(), batch_size)
        meters["distillation"].update(shadow_loss.item(), batch_size)
        meters["meta"].update(meta_loss.item(), batch_size)
        meters["trigger"].update(trigger_loss.item(), batch_size)
        meters["batch_time"].update(time.time() - end)
        end = time.time()

    logger.info(
        "Epoch %d | time %.1fs | data %.1fs | teacher %.4f | CE %.4f | "
        "KD %.4f | meta %.4f | trigger %.4f",
        epoch + 1,
        meters["batch_time"].sum,
        meters["data_time"].sum,
        meters["teacher"].avg,
        meters["classification"].avg,
        meters["distillation"].avg,
        meters["meta"].avg,
        meters["trigger"].avg,
    )


def warmup_teacher_epoch(
    epoch,
    teacher,
    classification_loss,
    teacher_optimizer,
    train_loader,
):
    """Warm up the teacher on clean samples before backdoor optimization."""
    logger = logging.getLogger("asymptomatic_carriers")
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    device = next(teacher.parameters()).device
    teacher.train()
    end = time.time()

    for images, labels in train_loader:
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = teacher(images)
        loss = classification_loss(logits, labels)
        teacher_optimizer.zero_grad()
        loss.backward()
        teacher_optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(
        "Warm-up epoch %d | time %.1fs | data %.1fs | CE %.4f",
        epoch + 1,
        batch_time.sum,
        data_time.sum,
        loss_meter.avg,
    )
