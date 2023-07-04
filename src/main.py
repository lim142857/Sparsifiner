# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path


from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
# from optim_factory_discard import create_optimizer
# from optim_factory_discard import LayerDecayValueAssigner

from timm.optim import create_optimizer
from optim_factory import create_optimizer_v2  # Optimizer factory that handles L1 regularization

from datasets import build_dataset, build_transform
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from losses import NewLoss
from calc_flops import calc_flops, throughput

from models.vit import VisionTransformerTeacher, VisionTransformer
from models.lvvit import LVViTTeacher
from models.linformer import Linformer
from models.sparsifiner import Sparsifiner
from models.sparsifiner_lvvit import SparsifinerLVViT

from config.config_load_merge import load_config

import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic training script', add_help=False)
    # Config file
    parser.add_argument("--config_path", default=None, type=str, help="Path to the config file")

    parser.add_argument('--arch', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')  # TODO: what is this?

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=True,
                        help='Using ema to eval during training.')  # TODO: why this is helpful, do we report ema results as the final results?

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Loss parameters
    parser.add_argument('--l1_weight', type=float, default=0.05,
                        help='l1 weight (default: 0.05)')
    parser.add_argument('--l1_weight_end', type=float, default=0.05, help="""Final value of the
           l1 loss weight. We use a cosine schedule for l1 loss weight and using a larger decay by
           introducing sparsity.""")
    parser.add_argument('--ratio_weight', type=float, default=2,
                        help='ratio weight (default: 0.5)')
    parser.add_argument('--ratio_weight_end', type=float, default=2, help="""Final value of the
           ratio loss weight. We use a cosine schedule for l1 loss weight and using a larger decay by
           introducing sparsity.""")

    # LR parameters
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=utils.str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=utils.str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=utils.str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=utils.str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=utils.str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")  # TODO: is this used in DeiT?

    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--base_rate', type=float, default='0.9')

    # Mlflow
    # TODO: move to config file.
    parser.add_argument("--experiment_name", default="test", help="mlflow experiment name")
    parser.add_argument("--mlflow_ckpt_save_freq", default=10, type=int, help="mlflow checkpoint save frequency")
    return parser


def main(args):
    args, cfg = load_config(args)
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    args.nb_classes = 1000

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Set up base criterion
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    # Set up model
    print(args.model)
    SPARSE_RATIO = [args.base_rate, args.base_rate - 0.2, args.base_rate - 0.4]
    if args.model == 'deit_small_patch16_224':
        KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
        model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, cfg=cfg)
        pretrained = torch.load('./pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, cfg=cfg)
    elif args.model == 'linformer':
        model = Linformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            reduce_n_factor=4, share_kv_proj=False, cfg=cfg)
        pretrained = torch.load('./pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, cfg=cfg)
    elif args.model == 'sparsifiner':
        token_keep_rate_base = cfg.SPAR.TOKEN_KEEP_RATE_BASE
        token_keep_rate = [
            token_keep_rate_base,
            token_keep_rate_base ** 2,
            token_keep_rate_base ** 3,
        ]
        token_keep_rate_list = (
                [1.0] * 3
                + [token_keep_rate[0]] * 3
                + [token_keep_rate[1]] * 3
                + [token_keep_rate[2]] * 3
        )
        if cfg.SPAR.ATTN_KEEP_RATE_LIST is not None:
            attn_keep_rate_list = cfg.SPAR.ATTN_KEEP_RATE_LIST
        else:
            attn_keep_rate_list = [cfg.SPAR.ATTN_KEEP_RATE] * 12
        pruning_loc = cfg.SPAR.TOKEN_PRUNING_LOC
        token_pruning_loc_indicator = [False] * len(token_keep_rate_list)
        for i in pruning_loc:
            token_pruning_loc_indicator[i] = True
        model = Sparsifiner(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            distill=True,
            attn_keep_rate_list=attn_keep_rate_list,
            token_keep_rate_list=token_keep_rate_list,
            token_pruning_loc_indicator=token_pruning_loc_indicator,
            reduce_n_factor=cfg.SPAR.REDUCE_N_FACTOR,
            reduce_c_factor=cfg.SPAR.REDUCE_C_FACTOR,
            share_inout_proj=cfg.SPAR.SHARE_INOUT_PROJ,
            cfg=cfg,
        )
        pretrained = torch.load('./pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, cfg=cfg)
    elif args.model == 'sparsifiner_lvvit':
        if cfg.SPAR.ATTN_KEEP_RATE_LIST is not None:
            attn_keep_rate_list = cfg.SPAR.ATTN_KEEP_RATE_LIST
        else:
            attn_keep_rate_list = [cfg.SPAR.ATTN_KEEP_RATE] * 16
        print('attn_keep_rate_list', attn_keep_rate_list)
        model = SparsifinerLVViT(
            patch_size=16,
            embed_dim=384,
            depth=16,
            num_heads=6,
            mlp_ratio=3,
            p_emb="4_2",
            skip_lam=2.0,
            return_dense=True,
            mix_token=True,
            distill=True,
            attn_keep_rate_list=attn_keep_rate_list,
            reduce_n_factor=cfg.SPAR.REDUCE_N_FACTOR,
            reduce_c_factor=cfg.SPAR.REDUCE_C_FACTOR,
            share_inout_proj=cfg.SPAR.SHARE_INOUT_PROJ,
            cfg=cfg,
        )
        pretrained = torch.load("./pretrained/lvvit_s-26M-224-83.3.pth.tar", map_location="cpu")
        teacher_model = LVViTTeacher(
            patch_size=16,
            embed_dim=384,
            depth=16,
            num_heads=6,
            mlp_ratio=3.0,
            p_emb="4_2",
            skip_lam=2.0,
            return_dense=True,
            mix_token=True,
            cfg=cfg,
        )
    elif args.model == 'deit_tiny_patch16_384':
        model = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, cfg=cfg)
        pretrained = torch.load('./pretrained/deit_tiny_patch16_224-a1311bcf.pth', map_location='cpu')
        teacher_model = None
    elif args.model == 'deit_small_patch16_384':
        model = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, cfg=cfg)
        pretrained = torch.load('./pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        teacher_model = None

    # Load pretrained weights
    if 'deit' in args.model or 'sparsifiner' in args.model or 'linformer' in args.model:
        if 'sparsifiner_lvvit' not in args.model:
            pretrained = pretrained['model']

    # interpolate position embedding
    # Load from 224 model and scale the pos-embedding to 384
    pos_embed_checkpoint = pretrained["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        print(f"Interpolating position embedding from {orig_size} to {new_size}")
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        pretrained["pos_embed"] = new_pos_embed

    utils.load_state_dict(model, pretrained)

    # Load teacher model weights
    if teacher_model is not None:
        utils.load_state_dict(teacher_model, pretrained)
        teacher_model.eval()
        teacher_model = teacher_model.to(device)
        print('success load teacher model weight')

    # Set criterion
    if 'sparsifiner' in args.model:
        criterion = NewLoss(teacher_model, base_criterion=base_criterion, cfg=cfg)
    elif 'linformer' in args.model:
        criterion = NewLoss(teacher_model, base_criterion=base_criterion, cfg=cfg)
    else:
        criterion = NewLoss(teacher_model, base_criterion=base_criterion, cfg=cfg)


    model.eval()
    if utils.is_main_process():
        flops = calc_flops(model, args.input_size)
        print('FLOPs: {}'.format(flops))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print('number of params:', n_parameters)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # # interpolate position embedding
        # pos_embed_checkpoint = checkpoint_model["pos_embed"]
        # embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = model.patch_embed.num_patches
        # num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # # class_token and dist_token are kept unchanged
        # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # # only the position tokens are interpolated
        # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # pos_tokens = pos_tokens.reshape(
        #     -1, orig_size, orig_size, embedding_size
        # ).permute(0, 3, 1, 2)
        # pos_tokens = torch.nn.functional.interpolate(
        #     pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        # )
        # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        # checkpoint_model["pos_embed"] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    if utils.is_main_process() and args.throughput:
        print('# throughput test')
        image = torch.randn(32, 3, args.input_size, args.input_size)
        throughput(image, model)
        del image
        import sys
        sys.exit(1)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    # Create backbone optimizer list
    strategy = {
        'attn_self_distill': cfg.SPAR.TRAIN_STRATEGY.ATTN_SELF_DISTILL,
        'attn_recon': cfg.SPAR.TRAIN_STRATEGY.ATTN_RECON,
        'cls_distill': cfg.SPAR.TRAIN_STRATEGY.CLS_DISTILL,
        'cls': cfg.SPAR.TRAIN_STRATEGY.CLS,
        'joint': cfg.SPAR.TRAIN_STRATEGY.JOINT,
    }
    print("Create Cosine LR scheduler and backbone optimizer")
    for params in model.named_parameters():
        if 'mask_predictor' in params[0]:
            params[1].requires_grad = False
        else:
            params[1].requires_grad = True
    bb_optimizer_list = []
    bb_scheduler_list = []
    for i in range(len(strategy['cls'].EPOCHS_LIST)):
        bb_optimizer_list += [create_optimizer(args, model)]
        _num_epochs = strategy['cls'].EPOCHS_LIST[i]
        bb_scheduler_list += [utils.cosine_scheduler(
            strategy['cls'].BASE_LR,
            strategy['cls'].FINAL_LR,
            _num_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=strategy['cls'].WARMUP_EPOCHS,
            warmup_steps=-1,
        )]
        print("strategy['cls'].FINAL_LR", strategy['cls'].FINAL_LR)
        print("cls bb_scheduler_list[%d] = %s" % (i, len(bb_scheduler_list[i])))
        print("cls bb_scheduler_list[%d] = %s" % (i, str(bb_scheduler_list[i])))
    for i in range(len(strategy['cls_distill'].EPOCHS_LIST)):
        bb_optimizer_list += [create_optimizer(args, model)]
        _num_epochs = strategy['cls_distill'].EPOCHS_LIST[i]
        bb_scheduler_list += [utils.cosine_scheduler(
            strategy['cls_distill'].BASE_LR,
            strategy['cls_distill'].FINAL_LR,
            _num_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=strategy['cls_distill'].WARMUP_EPOCHS,
            warmup_steps=-1,
        )]
        print("strategy['cls_distill'].FINAL_LR", strategy['cls_distill'].FINAL_LR)
        print("cls_distill bb_scheduler_list[%d] = %s" % (i, len(bb_scheduler_list[i])))
        print("cls_distill bb_scheduler_list[%d] = %s" % (i, str(bb_scheduler_list[i])))

    # Create mask predictor optimizer list
    print("Create Cosine LR scheduler and mask predictor optimizer")
    for params in model.named_parameters():
        if 'mask_predictor' in params[0]:
            params[1].requires_grad = True
        else:
            params[1].requires_grad = False
    mp_optimizer_list = []
    mp_scheduler_list = []
    for i in range(len(strategy['attn_recon'].EPOCHS_LIST)):
        mp_optimizer_list += [create_optimizer_v2(args, model, cfg=cfg)]
        _num_epochs = strategy['attn_recon'].EPOCHS_LIST[i]
        mp_scheduler_list += [utils.cosine_scheduler(
            strategy['attn_recon'].BASE_LR,
            strategy['attn_recon'].FINAL_LR,
            _num_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=strategy['attn_recon'].WARMUP_EPOCHS,
            warmup_steps=-1,
        )]
        print("attn_recon mp_scheduler_list[%d] = %s" % (i, len(mp_scheduler_list[i])))
        print("attn_recon mp_scheduler_list[%d] = %s" % (i, str(mp_scheduler_list[i])))
    for i in range(len(strategy['attn_self_distill'].EPOCHS_LIST)):
        mp_optimizer_list += [create_optimizer_v2(args, model, cfg=cfg)]
        _num_epochs = strategy['attn_self_distill'].EPOCHS_LIST[i]
        mp_scheduler_list += [utils.cosine_scheduler(
            strategy['attn_self_distill'].BASE_LR,
            strategy['attn_self_distill'].FINAL_LR,
            _num_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=strategy['attn_self_distill'].WARMUP_EPOCHS,
            warmup_steps=-1,
        )]
        print("attn_self_distill mp_scheduler_list[%d] = %s" % (i, len(mp_scheduler_list[i])))
        print("attn_self_distill mp_scheduler_list[%d] = %s" % (i, str(mp_scheduler_list[i])))

    # Build the strategy indicator
    strategy_indicator = []
    for i in range(len(cfg.SPAR.TRAIN_STRATEGY.INDICATOR)):
        (strategy_name, optimizer_order, epoch_offset, num_epochs) = cfg.SPAR.TRAIN_STRATEGY.INDICATOR[i]
        strategy_indicator += [(strategy_name, optimizer_order, epoch_offset)] * num_epochs
    assert len(strategy_indicator) == args.epochs, "The total epochs must be the sum of each strategy epochs"

    # Set back requires_grad to True
    for params in model.named_parameters():
        params[1].requires_grad = True

    joint_optimizer_list = []
    joint_scheduler_list = []
    print("Create Cosine LR scheduler and Joint optimizer")
    for i in range(len(strategy['joint'].EPOCHS_LIST)):
        joint_optimizer_list += [create_optimizer_v2(args, model, cfg=cfg)]
        _num_epochs = strategy['joint'].EPOCHS_LIST[i]
        joint_scheduler_list += [utils.cosine_scheduler(
            strategy['joint'].BASE_LR,
            strategy['joint'].FINAL_LR,
            _num_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=strategy['attn_recon'].WARMUP_EPOCHS,
            warmup_steps=-1,
        )]
        print("joint_scheduler_list[%d] = %s" % (i, len(joint_scheduler_list[i])))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used

    print("Use Cosine WD scheduler")
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    print("criterion = %s" % str(criterion))

    # L1 sparse prior
    if cfg.LOSS.USE_L1:
        l1_schedule_values = utils.cosine_scheduler(
            cfg.LOSS.L1.START_WEIGHT,
            cfg.LOSS.L1.END_WEIGHT,
            args.epochs,
            num_training_steps_per_epoch
        )
        print("Max L1W = %.7f, Min L1W = %.7f" % (max(l1_schedule_values), min(l1_schedule_values)))
    else:
        l1_schedule_values = None

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    # max_accuracy, max_accuracy_ema = utils.auto_load_model(
    #     args=args, model=model, model_without_ddp=model_without_ddp,
    #     optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Evaluation only
    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, cfg=cfg)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()

    # back up weights from cfg
    cls_weight_backup = cfg.LOSS.CLS.WEIGHT
    attn_self_distill_weight_backup = cfg.LOSS.ATTN_SELF_DISTILL.WEIGHT
    token_distill_weight_backup = cfg.LOSS.TOKEN_DISTILL.WEIGHT

    for epoch in range(args.start_epoch, args.epochs):
        print("epoch = %d" % epoch)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        (strategy_name, optimizer_order, epoch_offset) = strategy_indicator[epoch]
        if strategy_name == 'attn_recon':
            print("Use attn_recon strategy, mp optimizer_order = %d" % optimizer_order)
            cfg.LOSS.USE_CLS = True
            cfg.LOSS.CLS.WEIGHT = 0.0  # DDP issue
            cfg.LOSS.USE_CLS_DISTILL = False
            cfg.LOSS.USE_TOKEN_DISTILL = True
            cfg.LOSS.TOKEN_DISTILL.WEIGHT = 0.0
            cfg.LOSS.USE_ATTN_SELF_DISTILL = False
            cfg.LOSS.USE_ATTN_RECON = True
            optimizer = mp_optimizer_list[optimizer_order]
            lr_schedule_values = mp_scheduler_list[optimizer_order]
        elif strategy_name == 'attn_self_distill':
            print("Use attn_self_distill strategy, mp optimizer_order = %d" % optimizer_order)
            cfg.LOSS.USE_CLS = True
            cfg.LOSS.CLS.WEIGHT = 0.0  # DDP issue
            cfg.LOSS.USE_CLS_DISTILL = False
            cfg.LOSS.USE_TOKEN_DISTILL = False
            cfg.LOSS.USE_ATTN_SELF_DISTILL = True
            cfg.LOSS.ATTN_SELF_DISTILL.WEIGHT = attn_self_distill_weight_backup
            cfg.LOSS.USE_ATTN_RECON = False
            optimizer = mp_optimizer_list[optimizer_order]
            lr_schedule_values = mp_scheduler_list[optimizer_order]
        elif strategy_name == 'cls':
            print("Use cls strategy, bb optimizer_order = %d" % optimizer_order)
            cfg.LOSS.USE_CLS = True
            cfg.LOSS.CLS.WEIGHT = cls_weight_backup
            cfg.LOSS.USE_CLS_DISTILL = False
            cfg.LOSS.USE_TOKEN_DISTILL = False
            cfg.LOSS.USE_ATTN_SELF_DISTILL = False
            cfg.LOSS.USE_ATTN_RECON = False
            optimizer = bb_optimizer_list[optimizer_order]
            lr_schedule_values = bb_scheduler_list[optimizer_order]
        elif strategy_name == 'cls_distill':
            print("Use cls_distill strategy, bb optimizer_order = %d" % optimizer_order)
            cfg.LOSS.USE_CLS = True
            cfg.LOSS.CLS.WEIGHT = cls_weight_backup
            cfg.LOSS.USE_CLS_DISTILL = True
            cfg.LOSS.USE_TOKEN_DISTILL = True
            cfg.LOSS.TOKEN_DISTILL.WEIGHT = token_distill_weight_backup
            cfg.LOSS.USE_ATTN_SELF_DISTILL = True
            cfg.LOSS.ATTN_SELF_DISTILL.WEIGHT = 0.0
            cfg.LOSS.USE_ATTN_RECON = False
            optimizer = bb_optimizer_list[optimizer_order]
            lr_schedule_values = bb_scheduler_list[optimizer_order]
        elif strategy_name == 'joint':
            print("Use joint strategy, joint optimizer_order = %d" % optimizer_order)
            cfg.LOSS.USE_CLS = True
            cfg.LOSS.CLS.WEIGHT = cls_weight_backup  # DDP issue
            cfg.LOSS.USE_CLS_DISTILL = True
            cfg.LOSS.USE_TOKEN_DISTILL = True
            cfg.LOSS.TOKEN_DISTILL.WEIGHT = token_distill_weight_backup
            cfg.LOSS.USE_ATTN_SELF_DISTILL = True
            cfg.LOSS.ATTN_SELF_DISTILL.WEIGHT = attn_self_distill_weight_backup
            cfg.LOSS.USE_ATTN_RECON = False
            optimizer = joint_optimizer_list[optimizer_order]
            lr_schedule_values = joint_scheduler_list[optimizer_order]
        else:
            raise ValueError("Invalid strategy")
        start_steps = (epoch + epoch_offset) * num_training_steps_per_epoch
        print('start_steps = %d' % start_steps)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=start_steps,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp,
            l1_schedule_values=l1_schedule_values,
            cfg=cfg
        )

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, cfg=cfg)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema, best_acc=max_accuracy,
                        best_acc_ema=max_accuracy_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, cfg=cfg)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema, best_acc=max_accuracy,
                            best_acc_ema=max_accuracy_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        # Save metric evaluation results here using for e.g. MLflow or Wandb
        # if utils.is_main_process():
        #    log_stats['max_accuracy'] = max_accuracy
        #    wandb.log(log_stats)

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, best_acc=max_accuracy,
                    best_acc_ema=max_accuracy_ema)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
