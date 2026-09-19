import argparse
import datetime
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist

from configs.defaults import get_img_config
from data import get_dataloader
from losses import build_losses
from models import build_model
from evaluate import evaluate
from tools.utils import get_logger, save_checkpoint, set_seed, unwrap_model
from train import train_attack_epoch, warmup_teacher_epoch
from train_distiller import train_kd_epoch


def parse_options():
    parser = argparse.ArgumentParser(
        description=(
            "Train or distill the hidden backdoor propagation method described "
            "in Asymptomatic Carriers."
        )
    )
    parser.add_argument("--cfg", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--mode",
        choices=("attack", "distill"),
        default="attack",
        help="Train the teacher attack or distill a student from a saved teacher.",
    )
    parser.add_argument("--root", help="Override DATA.ROOT.")
    parser.add_argument("--dataset", help="Override DATA.DATASET.")
    parser.add_argument("--output", help="Override the output root.")
    parser.add_argument("--resume", help="Teacher checkpoint path.")
    parser.add_argument("--tag", help="Experiment tag used in the output path.")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    return get_img_config(args), args


def setup_runtime(local_rank_argument):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank_argument))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank() if distributed else 0
    return device, distributed, local_rank, rank


def build_run_directory(config, distributed, rank):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if distributed:
        value = [timestamp if rank == 0 else None]
        dist.broadcast_object_list(value, src=0)
        timestamp = value[0]
    run_directory = osp.join(config.OUTPUT, timestamp)
    if rank == 0:
        os.makedirs(run_directory, exist_ok=True)
    if distributed:
        dist.barrier()
    return run_directory


def build_optimizer(parameters, name, learning_rate, weight_decay):
    if name == "sgd":
        return optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config):
    if config.TRAIN.OPTIMIZER.NAME == "adamw":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.TRAIN.MAX_EPOCH,
        )
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
        gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
    )


def load_teacher_checkpoint(teacher, trigger, checkpoint_path, require_trigger):
    if not checkpoint_path:
        if require_trigger:
            raise ValueError("--mode distill requires --resume or MODEL.RESUME.")
        return None
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")
    teacher.load_state_dict(checkpoint["model_state_dict"])

    trigger_state = checkpoint.get("apt_trigger_state_dict")
    if trigger_state is None and require_trigger:
        raise KeyError("Checkpoint is missing 'apt_trigger_state_dict'.")
    if trigger_state is not None:
        trigger.load_state_dict(trigger_state)
    return checkpoint


def wrap_distributed(model, distributed, device, local_rank):
    model = model.to(device)
    if not distributed:
        return model
    device_ids = [local_rank] if device.type == "cuda" else None
    return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)


def should_evaluate(config, epoch):
    epoch_number = epoch + 1
    return (
        epoch_number > config.TEST.START_EVAL
        and epoch_number % config.TEST.EVAL_STEP == 0
    ) or epoch_number == config.TRAIN.MAX_EPOCH


def save_teacher(
    config,
    run_directory,
    epoch,
    teacher,
    trigger,
    clean_accuracy,
    attack_success_rate,
    is_best,
):
    save_checkpoint(
        {
            "model_state_dict": unwrap_model(teacher).state_dict(),
            "apt_trigger_state_dict": unwrap_model(trigger).state_dict(),
            "epoch": epoch,
            "clean_accuracy": clean_accuracy,
            "attack_success_rate": attack_success_rate,
            "config": str(config),
        },
        is_best,
        osp.join(run_directory, "checkpoints", f"teacher_epoch_{epoch + 1:03d}.pth.tar"),
    )


def save_student(
    config,
    run_directory,
    epoch,
    student,
    trigger,
    clean_accuracy,
    attack_success_rate,
    is_best,
):
    save_checkpoint(
        {
            "student_model_state_dict": unwrap_model(student).state_dict(),
            "apt_trigger_state_dict": unwrap_model(trigger).state_dict(),
            "epoch": epoch,
            "clean_accuracy": clean_accuracy,
            "attack_success_rate": attack_success_rate,
            "config": str(config),
        },
        is_best,
        osp.join(run_directory, "checkpoints", f"student_epoch_{epoch + 1:03d}.pth.tar"),
    )


def run_attack(
    config,
    teacher,
    shadow_model,
    trigger,
    train_loader,
    test_loader,
    train_sampler,
    run_directory,
    rank,
):
    classification_loss, distillation_loss = build_losses(config)
    base_lr = config.TRAIN.OPTIMIZER.LR
    weight_decay = config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    optimizer_name = config.TRAIN.OPTIMIZER.NAME

    teacher_optimizer = build_optimizer(
        teacher.parameters(), optimizer_name, base_lr, weight_decay
    )
    trigger_optimizer = build_optimizer(
        trigger.parameters(),
        optimizer_name,
        base_lr * config.TRAIN.TRIGGER_LR_MULTIPLIER,
        weight_decay,
    )
    shadow_optimizer = build_optimizer(
        shadow_model.parameters(),
        optimizer_name,
        base_lr * config.TRAIN.SHADOW_LR_MULTIPLIER,
        weight_decay,
    )
    teacher_scheduler = build_scheduler(teacher_optimizer, config)
    best_student_asr = float("-inf")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.MAX_EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch < config.TRAIN.WARMUP_EPOCHS:
            warmup_teacher_epoch(
                epoch,
                teacher,
                classification_loss,
                teacher_optimizer,
                train_loader,
            )
        else:
            train_attack_epoch(
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
            )
        teacher_scheduler.step()

        if should_evaluate(config, epoch):
            teacher_acc, teacher_asr = evaluate(
                config, teacher, trigger, test_loader
            )
            student_acc, student_asr = evaluate(
                config, shadow_model, trigger, test_loader
            )
            is_best = student_asr > best_student_asr
            best_student_asr = max(best_student_asr, student_asr)
            if rank == 0:
                save_teacher(
                    config,
                    run_directory,
                    epoch,
                    teacher,
                    trigger,
                    teacher_acc,
                    teacher_asr,
                    is_best,
                )


def run_distillation(
    config,
    teacher,
    student,
    trigger,
    train_loader,
    test_loader,
    train_sampler,
    run_directory,
    rank,
):
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    classification_loss, distillation_loss = build_losses(config)
    student_optimizer = build_optimizer(
        student.parameters(),
        config.TRAIN.OPTIMIZER.NAME,
        config.TRAIN.OPTIMIZER.LR * config.TRAIN.SHADOW_LR_MULTIPLIER,
        config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
    )
    student_scheduler = build_scheduler(student_optimizer, config)
    best_student_asr = float("-inf")

    evaluate(config, teacher, trigger, test_loader)
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.MAX_EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_kd_epoch(
            config,
            epoch,
            teacher,
            student,
            classification_loss,
            distillation_loss,
            student_optimizer,
            train_loader,
        )
        student_scheduler.step()

        if should_evaluate(config, epoch):
            student_acc, student_asr = evaluate(
                config, student, trigger, test_loader
            )
            is_best = student_asr > best_student_asr
            best_student_asr = max(best_student_asr, student_asr)
            if rank == 0:
                save_student(
                    config,
                    run_directory,
                    epoch,
                    student,
                    trigger,
                    student_acc,
                    student_asr,
                    is_best,
                )


def main():
    config, args = parse_options()
    device, distributed, local_rank, rank = setup_runtime(args.local_rank)
    run_directory = build_run_directory(config, distributed, rank)
    logger = get_logger(
        osp.join(run_directory, "train.log"),
        local_rank=rank,
        name="asymptomatic_carriers",
    )
    set_seed(config.SEED + rank)

    secondary_name = (
        config.MODEL.SHADOW_NAME
        if args.mode == "attack"
        else config.MODEL.STUDENT_NAME
    )
    data_model_name = config.MODEL.NAME if args.mode == "attack" else secondary_name
    train_loader, test_loader, train_sampler = get_dataloader(
        config,
        model_name=data_model_name,
    )
    teacher, secondary_model, trigger = build_model(
        config,
        secondary_model_name=secondary_name,
    )
    load_teacher_checkpoint(
        teacher,
        trigger,
        config.MODEL.RESUME,
        require_trigger=args.mode == "distill",
    )

    teacher = wrap_distributed(teacher, distributed, device, local_rank)
    secondary_model = wrap_distributed(
        secondary_model, distributed, device, local_rank
    )
    trigger = wrap_distributed(trigger, distributed, device, local_rank)

    logger.info("Mode: %s", args.mode)
    logger.info("Output: %s", run_directory)
    logger.info("Config:\n%s", config)

    try:
        if args.mode == "attack":
            run_attack(
                config,
                teacher,
                secondary_model,
                trigger,
                train_loader,
                test_loader,
                train_sampler,
                run_directory,
                rank,
            )
        else:
            run_distillation(
                config,
                teacher,
                secondary_model,
                trigger,
                train_loader,
                test_loader,
                train_sampler,
                run_directory,
                rank,
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
