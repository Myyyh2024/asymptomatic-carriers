import logging
import time

import torch
from torch import distributed as dist

from tools.utils import add_trigger


def _gather_tensor(tensor, total_examples):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor.cpu()

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)[:total_examples].cpu()


@torch.no_grad()
def _extract_predictions(model, dataloader, trigger):
    clean_predictions = []
    poisoned_predictions = []
    labels = []
    device = next(model.parameters()).device

    for images, batch_labels in dataloader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        mask, pattern = trigger(images)

        clean_predictions.append(model(images).argmax(dim=1))
        poisoned_images = add_trigger(
            images,
            mask.detach(),
            pattern.detach(),
        )
        poisoned_predictions.append(model(poisoned_images).argmax(dim=1))
        labels.append(batch_labels)

    return (
        torch.cat(clean_predictions),
        torch.cat(poisoned_predictions),
        torch.cat(labels),
    )


def evaluate(config, model, trigger, test_loader):
    """Report clean accuracy and attack success rate."""
    logger = logging.getLogger("asymptomatic_carriers")
    start = time.time()
    model.eval()
    trigger.eval()

    clean_predictions, poisoned_predictions, labels = _extract_predictions(
        model,
        test_loader,
        trigger,
    )
    total_examples = len(test_loader.dataset)
    clean_predictions = _gather_tensor(clean_predictions, total_examples)
    poisoned_predictions = _gather_tensor(poisoned_predictions, total_examples)
    labels = _gather_tensor(labels, total_examples)

    target_labels = torch.full_like(labels, config.TRAIN.TARGET)
    clean_accuracy = clean_predictions.eq(labels).float().mean().item() * 100.0
    attack_success_rate = (
        poisoned_predictions.eq(target_labels).float().mean().item() * 100.0
    )

    logger.info(
        "Evaluation | clean ACC %.2f | ASR %.2f | time %.1fs",
        clean_accuracy,
        attack_success_rate,
        time.time() - start,
    )
    return clean_accuracy, attack_success_rate
