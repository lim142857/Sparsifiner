"""
loss function for Sparsifiner training
"""
import torch
from torch.nn import functional as F
from torch import nn


class NewLoss(nn.Module):
    def __init__(self,
                 teacher_model,
                 base_criterion: torch.nn.Module,
                 cfg=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion

    def forward(self, inputs, out_dict, labels, cfg):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            out_dict: The output dictionary of the student model
            labels: the labels for the base criterion
            cfg: the config file
        """
        cfg_spar = cfg.SPAR
        cfg = cfg.LOSS

        cls_s = out_dict['x']
        token_s = out_dict['features']

        B, N, C = token_s.size()  # N = 196

        # 1. CLS loss
        if cfg.USE_CLS:
            cls_loss = self.base_criterion(cls_s, labels)

        # 2. CLS distillation loss
        if self.teacher_model is not None:
            with torch.no_grad():
                cls_t, token_t, attn_t_list = self.teacher_model(inputs)

        if cfg.USE_CLS_DISTILL:
            cls_distill_loss = F.kl_div(
                F.log_softmax(cls_s, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        # 3. Last Layer Token distillation loss
        if cfg.USE_TOKEN_DISTILL:
            token_s = token_s.reshape(B * N, C)
            token_t = token_t.reshape(B * N, C)
            if 'token_mask_list' in out_dict:
                if not cfg_spar.SLOW_FAST_PATH:
                    token_mask_list = out_dict['token_mask_list']
                    token_mask = token_mask_list[-1]
                    if token_mask is not None:
                        token_mask = token_mask.reshape(B * N).bool()
                        token_s = token_s[token_mask]
                        token_t = token_t[token_mask]
            if cfg.TOKEN_DISTILL.CRITERION == "MSE":
                token_distill_loss = F.mse_loss(token_s, token_t)
                # token_distill_loss = torch.pow(token_s - token_t, 2).mean()
            elif cfg.TOKEN_DISTILL.CRITERION == "KL":
                token_distill_loss = F.kl_div(
                    F.log_softmax(token_s, dim=-1),
                    F.log_softmax(token_t, dim=-1),
                    reduction='batchmean',
                    log_target=True
                )
            else:
                raise NotImplementedError

        # 4. Attention reconstruction loss
        if cfg.USE_ATTN_RECON:
            approx_attn_list = out_dict['approx_attn_list']
            attn_recon_loss = 0.0
            for i, (attn_s, attn_t) in enumerate(zip(approx_attn_list, attn_t_list)):
                attn_t = attn_t[..., 1:, :]  # attn_t: B, H, 196, 197
                if cfg.ATTN_RECON.CRITERION == "MSE":
                    attn_recon_loss_layer_i = F.mse_loss(attn_s, attn_t, reduction='sum')
                elif cfg.ATTN_RECON.CRITERION == "CE":
                    attn_recon_loss_layer_i = F.cross_entropy(attn_s, attn_t)  # TODO: upgrade torch to 1.10
                elif cfg.ATTN_RECON.CRITERION == "KL":
                    # attn_s is logist, attn_t in [0,1]
                    attn_recon_loss_layer_i = F.kl_div(
                        F.log_softmax(attn_s, dim=-1),
                        attn_t,
                        reduction='batchmean',
                        log_target=False
                    )
                else:
                    raise NotImplementedError
                attn_recon_loss = attn_recon_loss + attn_recon_loss_layer_i
            attn_recon_loss /= len(approx_attn_list)

        # 5. Attention Self distillation loss
        if cfg.USE_ATTN_SELF_DISTILL:
            approx_attn_list = out_dict['approx_attn_list']
            unmasked_attn_list = out_dict['unmasked_attn_list']
            attn_self_distill_loss = 0.0
            for i, (approx_attn, unmasked_attn) in enumerate(zip(approx_attn_list, unmasked_attn_list)):
                unmasked_attn = unmasked_attn[..., 1:, :]
                if cfg.ATTN_SELF_DISTILL.CRITERION == "MSE":
                    attn_self_distill_loss_layer_i = F.mse_loss(approx_attn, unmasked_attn, reduction='sum')
                else:
                    raise NotImplementedError
                attn_self_distill_loss = attn_self_distill_loss + attn_self_distill_loss_layer_i
            attn_self_distill_loss /= len(approx_attn_list)

        # 6. Maximum Budget Loss
        # TODO: fix this
        if cfg.USE_MAX_BUDGET:
            attn_mask_list = out_dict['attn_mask_list']
            H = attn_t_list[0].shape[1]
            token_level_budget = (N + 1) // cfg.MAX_BUDGET.TOKEN_LEVEL_BUDGET_RATIO
            batch_level_budget = B * H * (N + 1) // cfg.MAX_BUDGET.BATCH_LEVEL_BUDGET_RATIO
            token_level_budget_loss = 0.0
            batch_level_budget_loss = 0.0
            for i, hard_attn_mask in enumerate(attn_mask_list):
                # hard_attn_mask shape of [B, H, N+1, N+1]
                hard_attn_mask = hard_attn_mask[..., 1:, :]  # Remove the cls token connectivity map

                # each query token has an adaptive budget
                exceed = hard_attn_mask.sum(-1) - token_level_budget  # [B, H, N]
                penalty = torch.maximum(torch.zeros_like(exceed), exceed).mean()  # [B * H * N]
                token_level_budget_loss = penalty + token_level_budget_loss

                # batch level budget approximation for the global budget
                exceed = hard_attn_mask.sum() - batch_level_budget  # [B * H * N]
                penalty = torch.maximum(torch.zeros_like(exceed), exceed)  # [B * H * N]
                batch_level_budget_loss = penalty + batch_level_budget_loss
            batch_level_budget_loss /= len(attn_mask_list)
            token_level_budget_loss /= len(attn_mask_list)

        # 7. Least K Loss
        if cfg.USE_LEAST_K:
            attn_mask_list = out_dict['attn_mask_list']
            least_k_loss = 0.0
            token_level_budget = (N + 1) // cfg.LEAST_K.BUDGET_RATIO
            for i, attn_mask in enumerate(attn_mask_list):
                # hard_cont_mask shape of [B, H, N+1, N+1]
                attn_mask = attn_mask[..., 1:, :]  # Remove the cls token connectivity map

                # each query token has an adaptive budget
                attn_mask_least_k = \
                torch.topk(attn_mask, N + 1 - token_level_budget, dim=-1, largest=False, sorted=False)[0]
                least_k_loss = least_k_loss + attn_mask_least_k.abs().mean()  # [B * H * N]
            least_k_loss /= len(attn_mask_list)

        # 8. Ratio loss.
        if cfg.USE_RATIO_LOSS:
            attn_mask_list = out_dict['attn_mask_list']
            ratio_loss = 0.0
            target_ratio = cfg.RATIO_LOSS.RATIO
            for i, attn_mask in enumerate(attn_mask_list):
                attn_mask_ratio = attn_mask.mean(-1)
                ratio_loss = ratio_loss + ((attn_mask_ratio - target_ratio) ** 2).mean()
            ratio_loss /= len(attn_mask_list)

        # Total loss
        loss = 0.0
        loss_component = {}
        if cfg.USE_CLS:
            loss = loss + cls_loss * cfg.CLS.WEIGHT
            loss_component['cls_loss'] = cls_loss * cfg.CLS.WEIGHT
        if cfg.USE_CLS_DISTILL:
            loss += cls_distill_loss * cfg.CLS_DISTILL.WEIGHT
            loss_component['cls_distill_loss'] = cls_distill_loss * cfg.CLS_DISTILL.WEIGHT
        if cfg.USE_TOKEN_DISTILL:
            loss += token_distill_loss * cfg.TOKEN_DISTILL.WEIGHT
            loss_component['token_distill_loss'] = token_distill_loss * cfg.TOKEN_DISTILL.WEIGHT
        if cfg.USE_ATTN_RECON:
            loss += attn_recon_loss * cfg.ATTN_RECON.WEIGHT
            loss_component['attn_recon_loss'] = attn_recon_loss * cfg.ATTN_RECON.WEIGHT
        if cfg.USE_ATTN_SELF_DISTILL:
            loss += attn_self_distill_loss * cfg.ATTN_SELF_DISTILL.WEIGHT
            loss_component['attn_self_distill_loss'] = attn_self_distill_loss * cfg.ATTN_SELF_DISTILL.WEIGHT
        # if cfg.USE_MAX_BUDGET:
        #     loss += token_level_budget_loss * cfg.MAX_BUDGET.TOKEN_LEVEL_BUDGET_LOSS_WEIGHT
        #     loss += batch_level_budget_loss * cfg.MAX_BUDGET.BATCH_LEVEL_BUDGET_LOSS_WEIGHT
        #     loss_component['token_level_budget_loss'] = token_level_budget_loss
        #     loss_component['batch_level_budget_loss'] = batch_level_budget_loss
        if cfg.USE_LEAST_K:
            loss += least_k_loss * cfg.LEAST_K.WEIGHT
            loss_component['least_k_loss'] = least_k_loss * cfg.LEAST_K.WEIGHT
        if cfg.USE_RATIO_LOSS:
            loss += ratio_loss * cfg.RATIO_LOSS.WEIGHT
            loss_component['ratio_loss'] = ratio_loss * cfg.RATIO_LOSS.WEIGHT

        return loss, loss_component
