from functools import partial

from torch import nn

from losses.cross_entropy_smooth import CrossEntropyWithLabelSmooth
from losses.gather import kldiv


def build_losses(config):
    classification_losses = {
        "crossentropy": nn.CrossEntropyLoss,
        "crossentropylabelsmooth": CrossEntropyWithLabelSmooth,
    }
    try:
        classification_loss = classification_losses[config.LOSS.CLA_LOSS]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported classification loss: {config.LOSS.CLA_LOSS}"
        ) from exc

    if config.LOSS.KD_LOSS == "KLDivergenceLoss":
        distillation_loss = partial(
            kldiv,
            temperature=config.LOSS.KD_TEMPERATURE,
        )
    elif config.LOSS.KD_LOSS == "MSELoss":
        distillation_loss = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported distillation loss: {config.LOSS.KD_LOSS}")

    return classification_loss, distillation_loss
