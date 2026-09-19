import logging

from models.resnet import resnet18, resnet34, resnet50
from models.trigger import TriggerModule
from models.vit_pytorch import TransformerWithClassifier


MODEL_FACTORY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "vit_base": lambda num_classes: TransformerWithClassifier(
        "google/vit-base-patch16-224", num_classes
    ),
    "deit_base": lambda num_classes: TransformerWithClassifier(
        "facebook/deit-base-patch16-224", num_classes
    ),
}


def get_model(model_name, num_classes):
    try:
        factory = MODEL_FACTORY[model_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_FACTORY))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}") from exc
    return factory(num_classes=num_classes)


def build_model(config, secondary_model_name=None):
    """Build the teacher, secondary model, and learnable trigger."""
    logger = logging.getLogger("asymptomatic_carriers.model")
    secondary_model_name = secondary_model_name or config.MODEL.SHADOW_NAME

    logger.info("Teacher model: %s", config.MODEL.NAME)
    logger.info("Secondary model: %s", secondary_model_name)

    teacher = get_model(config.MODEL.NAME, config.MODEL.CLASS_NUM)
    secondary = get_model(secondary_model_name, config.MODEL.CLASS_NUM)
    trigger = TriggerModule(
        image_shape=(3, config.DATA.HEIGHT, config.DATA.WIDTH)
    )

    teacher_size = sum(parameter.numel() for parameter in teacher.parameters()) / 1e6
    logger.info("Teacher size: %.2fM parameters", teacher_size)
    return teacher, secondary, trigger
