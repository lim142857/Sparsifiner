import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from default_config import get_cfg_defaults


# merge parameters from opts in arguments
def overwrite_configs_from_args(cfg, args):
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size


def overwrite_args_from_configs(cfg, args):
    # Model
    if cfg.MODEL.NAME:
        args.model = cfg.MODEL.NAME

    # MLFlow
    if cfg.MLFLOW.EXPERIMENT_NAME:
        args.experiment_name = cfg.MLFLOW.EXPERIMENT_NAME
    if cfg.MLFLOW.MLFLOW_CKPT_SAVE_FREQ:
        args.mlflow_ckpt_save_freq = cfg.MLFLOW.MLFLOW_CKPT_SAVE_FREQ

    # Dataset
    if cfg.DATASET.DATA_SET:
        args.data_set = cfg.DATASET.DATA_SET
    if cfg.DATASET.DATA_PATH:
        args.data_path = cfg.DATASET.DATA_PATH

    # Data
    if cfg.DATA.INPUT_SIZE:
        args.input_size = cfg.DATA.INPUT_SIZE

    # Training
    if cfg.TRAIN.BATCH_SIZE:
        args.batch_size = cfg.TRAIN.BATCH_SIZE
    if cfg.TRAIN.EPOCHS:
        args.epochs = cfg.TRAIN.EPOCHS

    # Testing
    if cfg.TEST.EVAL_ONLY:
        args.eval = cfg.TEST.EVAL_ONLY

    # Optimizer
    if cfg.OPTIM.LR:
        args.lr = cfg.OPTIM.LR
        args.lr_scale = cfg.OPTIM.BONE_LR_SCALE
        args.min_lr = cfg.OPTIM.MIN_LR
        args.weight_decay = cfg.OPTIM.WEIGHT_DECAY

    # Loss
    args.use_l1 = cfg.LOSS.USE_L1
    args.l1_weight = cfg.LOSS.L1.START_WEIGHT
    args.l1_weight_end = cfg.LOSS.L1.END_WEIGHT
    # args.ratio_weight = cfg.LOSS.RATIO_WEIGHT
    # args.ratio_weight_end = cfg.LOSS.RATIO_WEIGHT_END

    if cfg.TRAIN.EPOCHS:
        args.epochs = cfg.TRAIN.EPOCHS

    return args


# Return cfg with parameters
def load_config(args):
    cfg = get_cfg_defaults()

    # overwrite_configs_from_args(cfg, args)

    if args.config_path is not None:
        cfg.merge_from_file(args.config_path)
    # if args.opts is not None:
    #     cfg.merge_from_list(args.opts)

    args = overwrite_args_from_configs(cfg, args)

    return args, cfg
