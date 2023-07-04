# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                    l1_schedule_values=None,
                    cfg=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    # Initialize the L1 schedule
    if l1_schedule_values is not None:
        l1_weight = l1_schedule_values[0]

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:  # TODO: this might be a bug, try to reproduce EViT results.
            print("Finished training for this epoch")
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                # if epoch < param_group['fix_step']:
                #     param_group["lr"] = 0.
                # elif lr_schedule_values is not None:
                #     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Update L1 weight
        if l1_schedule_values is not None:
            if data_iter_step % update_freq == 0:
                l1_weight = l1_schedule_values[it]
            l1_parameters = []
            for params in model.named_parameters():
                if 'proj_back_n' in params[0]:
                    l1_parameters.append(params[1].view(-1))
            l1_loss = torch.norm(torch.cat(l1_parameters), 1)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                out_dict = model(samples)
                output = out_dict['x']
                loss, loss_component = criterion(samples, out_dict, targets, cfg)
                if l1_schedule_values is not None:
                    loss += l1_weight * l1_loss
        else:  # full precision
            out_dict = model(samples)
            output = out_dict['x']
            loss, loss_component = criterion(samples, out_dict, targets, cfg)
            if l1_schedule_values is not None:
                loss += l1_weight * l1_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        # detect_nan = 0.
        # for name, param in model.named_parameters():
        #     if torch.isnan(param.grad).any():
        #         print(name, param.grad)
        #         detect_nan = 1.
        # parameters = [p for p in model.parameters() if p.requires_grad]
        # if len(parameters) == 0:
        #     total_norm = 0.0
        # else:
        #     device = parameters[0].grad.device
        #     total_norm = torch.norm(
        #         torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
        # print(f"total_norm: {total_norm}")

        torch.cuda.synchronize()

        # Update metric logger
        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        if cfg.LOSS.USE_CLS:
            metric_logger.update(cls_loss=loss_component['cls_loss'])
        if cfg.LOSS.USE_CLS_DISTILL:
            metric_logger.update(cls_distill_loss=loss_component['cls_distill_loss'])
        if cfg.LOSS.USE_TOKEN_DISTILL:
            metric_logger.update(token_distill_loss=loss_component['token_distill_loss'])
        if cfg.LOSS.USE_ATTN_RECON:
            metric_logger.update(attn_recon_loss=loss_component['attn_recon_loss'])
        if cfg.LOSS.USE_ATTN_SELF_DISTILL:
            metric_logger.update(attn_self_distill_loss=loss_component['attn_self_distill_loss'])
        if cfg.LOSS.USE_LEAST_K:
            metric_logger.update(least_k_loss=loss_component['least_k_loss'])
        if cfg.LOSS.USE_RATIO_LOSS:
            metric_logger.update(ratio_loss=loss_component['ratio_loss'])
        if cfg.LOSS.USE_L1:
            metric_logger.update(l1_weight=l1_weight)
            metric_logger.update(l1_loss=l1_loss * l1_weight)
            metric_logger.update(uw_l1_loss=l1_loss)
        metric_logger.update(class_acc=class_acc)
        # metric_logger.update(total_norm=total_norm)
        # metric_logger.update(detect_nan=detect_nan)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        # TODO: remove this
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(cls_loss=loss_part[0], head="loss")
            # log_writer.update(ratio_loss=loss_part[1], head="loss")
            # log_writer.update(cls_distill_loss=loss_part[2], head="loss")
            # log_writer.update(token_distill_loss=loss_part[3], head="loss")
            # log_writer.update(layer_mse_loss=loss_part[4], head="loss")
            # log_writer.update(feat_distill_loss=loss_part[5], head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, cfg=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    i = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        i += 1
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                out_dict = model(images)
                output = out_dict['x']
                loss = criterion(output, target)
        else:
            out_dict = model(images)
            output = out_dict['x']
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if cfg.SPAR.OUT_ATTN_MASK_SPARSITY:
            attn_mask_sparsity = out_dict['attn_mask_sparsity']
            metric_logger.meters['attn_mask_sparsity'].update(attn_mask_sparsity.item(), n=batch_size)
        if cfg.SPAR.OUT_BASIS_SPARSITY:
            basis_sparsity = out_dict['basis_sparsity']
            metric_logger.meters['basis_sparsity'].update(basis_sparsity.item(), n=batch_size)
        if cfg.SPAR.OUT_BASIS_COEF_SPARSITY:
            basis_coef_sparsity = out_dict['basis_coef_sparsity']
            metric_logger.meters['basis_coef_sparsity'].update(basis_coef_sparsity.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    if cfg.SPAR.OUT_ATTN_MASK_SPARSITY:
        print('* attn_mask_sparsity {sparsity.global_avg:.3f}:'.format(sparsity=metric_logger.attn_mask_sparsity))
    if cfg.SPAR.OUT_BASIS_SPARSITY:
        print('* basis_sparsity {sparsity.global_avg:.3f}:'.format(sparsity=metric_logger.basis_sparsity))
    if cfg.SPAR.OUT_BASIS_COEF_SPARSITY:
        print('* basis_coef_sparsity {sparsity.global_avg:.3f}:'.format(sparsity=metric_logger.basis_coef_sparsity))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
