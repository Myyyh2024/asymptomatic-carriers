import os

from yacs.config import CfgNode as CN


_C = CN()

_C.DATA = CN()
_C.DATA.ROOT = "./datasets"
_C.DATA.DATASET = "cifar10"
_C.DATA.NUM_WORKERS = 8
_C.DATA.HEIGHT = 32
_C.DATA.WIDTH = 32
_C.DATA.TRAIN_BATCH = 128
_C.DATA.TEST_BATCH = 256

_C.MODEL = CN()
_C.MODEL.NAME = "resnet50"
_C.MODEL.SHADOW_NAME = "resnet50"
_C.MODEL.STUDENT_NAME = "resnet34"
_C.MODEL.CLASS_NUM = 10
_C.MODEL.RESUME = ""

_C.LOSS = CN()
_C.LOSS.CLA_LOSS = "crossentropy"
_C.LOSS.KD_LOSS = "KLDivergenceLoss"
_C.LOSS.KD_TEMPERATURE = 4.0

_C.TRAIN = CN()
_C.TRAIN.TARGET = 1
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 200
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.KD_WEIGHT = 0.5
_C.TRAIN.MASK_WEIGHT = 0.01
_C.TRAIN.META_WEIGHT = 0.5
_C.TRAIN.INNER_LR = 0.0005
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "sgd"
_C.TRAIN.OPTIMIZER.LR = 0.1
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.TRAIN.TRIGGER_LR_MULTIPLIER = 5.0
_C.TRAIN.SHADOW_LR_MULTIPLIER = 0.1
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [100, 150]
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TEST = CN()
_C.TEST.EVAL_STEP = 10
_C.TEST.START_EVAL = 0

_C.SEED = 1024
_C.OUTPUT = "./outputs"
_C.TAG = "experiment"


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    if args.root:
        config.DATA.ROOT = args.root
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.output:
        config.OUTPUT = args.output
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.tag:
        config.TAG = args.tag

    config.OUTPUT = os.path.join(
        config.OUTPUT,
        config.DATA.DATASET,
        config.TAG,
    )
    config.freeze()


def get_img_config(args):
    """Build the experiment configuration from defaults, YAML, and CLI overrides."""
    config = _C.clone()
    update_config(config, args)
    return config
