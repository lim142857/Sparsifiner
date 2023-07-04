"""Configs."""
from yacs.config import CfgNode as CN
import os

# Config definition
_C = CN()

# -----------------------------------------------------------------------------
# MLFLOW/WANDB options
# -----------------------------------------------------------------------------
_C.MLFLOW = CN()
_C.MLFLOW.EXPERIMENT_NAME = "imnet_default"
_C.MLFLOW.MLFLOW_CKPT_SAVE_FREQ = 10

# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 320
_C.TRAIN.EPOCHS = 30

# -----------------------------------------------------------------------------
# Testing options
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL_ONLY = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "sparsifiner"
_C.MODEL.VISUAL = False

# -----------------------------------------------------------------------------
# Sparsifiner options
# -----------------------------------------------------------------------------
_C.SPAR = CN()
_C.SPAR.REDUCE_N_FACTOR = 0
_C.SPAR.REDUCE_C_FACTOR = 0
_C.SPAR.SHARE_INOUT_PROJ = False

# Pruning options
_C.SPAR.ATTN_KEEP_RATE = 1.0
_C.SPAR.ATTN_KEEP_RATE_LIST = None
_C.SPAR.TOKEN_KEEP_RATE_BASE = 1.0
_C.SPAR.TOKEN_PRUNING_LOC = [3, 6, 9]
_C.SPAR.PRUNE_ATTN_MATRIX_ROW = False
_C.SPAR.PRUNE_ATTN_MATRIX_COL = False
_C.SPAR.SLOW_FAST_PATH = False
_C.SPAR.ZERO_OUT_PRUNED = False

# Pruning thresholds
_C.SPAR.BASIS_THRESHOLD = 0.0

_C.SPAR.BASIS_COEF = CN()
_C.SPAR.BASIS_COEF.USE_TOPK = False
_C.SPAR.BASIS_COEF.TOPK = 8
_C.SPAR.BASIS_COEF.THRESHOLD = 0.0

_C.SPAR.ATTN_SCORE = CN()
_C.SPAR.ATTN_SCORE.USE_TOPK = False
_C.SPAR.ATTN_SCORE.THRESHOLD = 0.0

# Output sparsity
_C.SPAR.OUT_BASIS_SPARSITY = False
_C.SPAR.OUT_BASIS_COEF_SPARSITY = False
_C.SPAR.OUT_ATTN_MASK_SPARSITY = False

# Architecture
_C.SPAR.ARCH = "dense"
_C.SPAR.ARCH_2 = CN()
_C.SPAR.ARCH_2.SOFTMAX_ON_APPROX_ATTN = False
_C.SPAR.ARCH_2.RESCALE_APPROX_ATTN = True
_C.SPAR.ARCH_2.LEARNED_SCALE_APPROX_ATTN = False
_C.SPAR.POST_SOFTMAX = False

# TODO: Remove this option
_C.SPAR.TRAIN_MASK_PREDICTOR_ONLY = False

# Differentiable Topk
_C.SPAR.TOPK_NUM_SAMPLES = 500
_C.SPAR.TOPK_SIGMA_INIT = 0.05
_C.SPAR.TOPK_SIGMA_MAX = 0.05
_C.SPAR.TOPK_SIGMA_DECAY = False

# -----------------------------------------------------------------------------
# Sparsifiner training options
# -----------------------------------------------------------------------------

_C.SPAR.TRAIN_STRATEGY = CN()
_C.SPAR.TRAIN_STRATEGY.INDICATOR = [['attn_recon', 0, 8], ['cls_distill', 0,  12], ['attn_self_distill', 1,  8], ['cls_distill', 1, 22]]
_C.SPAR.TRAIN_STRATEGY.ATTN_RECON = CN()
_C.SPAR.TRAIN_STRATEGY.ATTN_RECON.EPOCHS_LIST = [8]
_C.SPAR.TRAIN_STRATEGY.ATTN_RECON.WARMUP_EPOCHS = 1
_C.SPAR.TRAIN_STRATEGY.ATTN_RECON.BASE_LR = 5e-4
_C.SPAR.TRAIN_STRATEGY.ATTN_RECON.FINAL_LR = 1e-6

_C.SPAR.TRAIN_STRATEGY.CLS = CN()
_C.SPAR.TRAIN_STRATEGY.CLS.EPOCHS_LIST = []
_C.SPAR.TRAIN_STRATEGY.CLS.WARMUP_EPOCHS = 3
_C.SPAR.TRAIN_STRATEGY.CLS.BASE_LR = 1e-4
_C.SPAR.TRAIN_STRATEGY.CLS.FINAL_LR = 1e-6

_C.SPAR.TRAIN_STRATEGY.CLS_DISTILL = CN()
_C.SPAR.TRAIN_STRATEGY.CLS_DISTILL.EPOCHS_LIST = [12, 22]
_C.SPAR.TRAIN_STRATEGY.CLS_DISTILL.WARMUP_EPOCHS = 3
_C.SPAR.TRAIN_STRATEGY.CLS_DISTILL.BASE_LR = 1e-4
_C.SPAR.TRAIN_STRATEGY.CLS_DISTILL.FINAL_LR = 1e-6

_C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL = CN()
_C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL.EPOCHS_LIST = [8]
_C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL.WARMUP_EPOCHS = 1
_C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL.BASE_LR = 1e-4
_C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL.FINAL_LR = 1e-6

_C.SPAR.TRAIN_STRATEGY.JOINT = CN()
_C.SPAR.TRAIN_STRATEGY.JOINT.EPOCHS_LIST = []
_C.SPAR.TRAIN_STRATEGY.JOINT.WARMUP_EPOCHS = 1
_C.SPAR.TRAIN_STRATEGY.JOINT.BASE_LR = 1e-4
_C.SPAR.TRAIN_STRATEGY.JOINT.FINAL_LR = 1e-6

# Self distillation weight schedule?
# _C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL_WEIGHT_START = 0.00000001
# _C.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL_WEIGHT_END = 0.0

# -----------------------------------------------------------------------------
# Vision Transformer options
# -----------------------------------------------------------------------------
_C.VIT = CN()
_C.VIT.TOP_K = 0
_C.VIT.AS_TEACHER = True
_C.VIT.RETURN_ATTN = True

# -----------------------------------------------------------------------------
# Linformer options
# -----------------------------------------------------------------------------
_C.LIN = CN()
_C.LIN.REDUCE_N_FACTOR = 8.0

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_SET = "IMNET"
HOME = os.environ.get('HOME')
_C.DATASET.DATA_PATH = HOME + "/ILSVRC2012"

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.INPUT_SIZE = 224

# -----------------------------------------------------------------------------
# Loss Options
# -----------------------------------------------------------------------------
_C.LOSS = CN()

# Basis sparsity loss using L1 norm
_C.LOSS.USE_L1 = False
_C.LOSS.L1 = CN()
_C.LOSS.L1.START_WEIGHT = 0.0
_C.LOSS.L1.END_WEIGHT = 0.0

_C.LOSS.USE_CLS = True
_C.LOSS.CLS = CN()
_C.LOSS.CLS.WEIGHT = 1.0

_C.LOSS.USE_CLS_DISTILL = True
_C.LOSS.CLS_DISTILL = CN()
_C.LOSS.CLS_DISTILL.WEIGHT = 0.5

_C.LOSS.USE_TOKEN_DISTILL = True
_C.LOSS.TOKEN_DISTILL = CN()
_C.LOSS.TOKEN_DISTILL.WEIGHT = 0.5
_C.LOSS.TOKEN_DISTILL.CRITERION = "MSE"  # MSE or KL

_C.LOSS.USE_ATTN_RECON = False
_C.LOSS.ATTN_RECON = CN()
_C.LOSS.ATTN_RECON.WEIGHT = 0.01
_C.LOSS.ATTN_RECON.CRITERION = "MSE"  # MSE or KL or CE

_C.LOSS.USE_ATTN_SELF_DISTILL = False
_C.LOSS.ATTN_SELF_DISTILL = CN()
_C.LOSS.ATTN_SELF_DISTILL.WEIGHT = 0.05
_C.LOSS.ATTN_SELF_DISTILL.CRITERION = "MSE"  # MSE or KL or CE

_C.LOSS.USE_LEAST_K = False
_C.LOSS.LEAST_K = CN()
_C.LOSS.LEAST_K.WEIGHT = 0.01
_C.LOSS.LEAST_K.BUDGET_RATIO = 4

_C.LOSS.USE_RATIO_LOSS = False
_C.LOSS.RATIO_LOSS = CN()
_C.LOSS.RATIO_LOSS.RATIO = 0.25
_C.LOSS.RATIO_LOSS.WEIGHT = 2.0
_C.LOSS.RATIO_LOSS.WEIGHT_END = None

_C.LOSS.USE_MAX_BUDGET = False
_C.LOSS.MAX_BUDGET = CN()
_C.LOSS.MAX_BUDGET.TOKEN_LEVEL_BUDGET_RATIO = 4
_C.LOSS.MAX_BUDGET.TOKEN_LEVEL_BUDGET_LOSS_WEIGHT = 2.0
_C.LOSS.MAX_BUDGET.BATCH_LEVEL_BUDGET_RATIO = 4
_C.LOSS.MAX_BUDGET.BATCH_LEVEL_BUDGET_LOSS_WEIGHT = 2.0
_C.LOSS.MAX_BUDGET.COLUMN = False

# TODO: discard this
# Sparsifiner training
_C.LOSS.USE_DISTILLATION_SCHEDULER = False
_C.LOSS.DISTILLATION_SCHEDULER = CN()
_C.LOSS.DISTILLATION_SCHEDULER.ATTN_RECON_EPOCHS = 10
_C.LOSS.DISTILLATION_SCHEDULER.USE_ATTN_SELF_DISTILL = True

# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.LR = 5e-4
_C.OPTIM.MIN_LR = 1e-6
_C.OPTIM.BONE_LR_SCALE = 0.01
_C.OPTIM.WEIGHT_DECAY = 0.05

# -----------------------------------------------------------------------------
# Experiment Description
# -----------------------------------------------------------------------------
_C.DESCRIPTION = CN()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
# Check assertions for the cfg parameters and return the cfg
def _assert_and_infer_cfg(cfg):
    # assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    # assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    return cfg


def get_cfg_defaults():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
