"""
Sparsifiner
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def compute_sparsity(x):
    total_num = torch.numel(x)
    num_non_zero = torch.count_nonzero(x)
    num_zero = total_num - num_non_zero
    sparsity = num_zero / total_num
    return sparsity


class MaskPredictor(nn.Module):
    """ Mask Predictor using Low rank MHA"""

    def __init__(self,
                 dim,
                 num_heads=8,
                 num_tokens=197,
                 attn_keep_rate=0.25,
                 reduce_n_factor=8,
                 reduce_c_factor=2,
                 share_inout_proj=False,
                 qk_scale=None,
                 cfg=None
                 ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.reduced_c = self.head_dim // reduce_c_factor
        self.reduced_n = int(num_tokens // reduce_n_factor)
        self.scale = qk_scale or self.num_heads ** -0.5
        self.proj_c_q = nn.Linear(self.head_dim, self.reduced_c)
        self.proj_c_k = nn.Linear(self.head_dim, self.reduced_c)

        self.proj_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
        # trunc_normal_(self.proj_back_n, std=.02, a=0.)
        trunc_normal_(self.proj_n, std=.02)
        if share_inout_proj:
            self.proj_back_n = self.proj_n
        else:
            self.proj_back_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
            trunc_normal_(self.proj_back_n, std=.02)

        self.basis_threshold = nn.Threshold(cfg.SPAR.BASIS_THRESHOLD, 0.)
        self.basis_coef_threshold = nn.Threshold(cfg.SPAR.BASIS_COEF.THRESHOLD, 0.)

        self.attn_budget = math.ceil(attn_keep_rate * num_tokens)
        self.cfg = cfg

    def forward(self, q, k, token_mask=None):
        # TODO: Perform full self-attention if attn_budget > token_budget
        if token_mask is not None:
            token_budget = token_mask[0].sum(dim=-1)
            self.attn_budget = token_budget if token_budget < self.attn_budget else self.attn_budget

        out_dict = {}
        cfg = self.cfg.SPAR

        B, H, N, C = q.shape
        assert self.num_tokens == N
        q, k = self.proj_c_q(q), self.proj_c_k(k)  # [B, H, N, c]
        if token_mask is not None:
            # token_mask: [B, N-1]
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)

        k = k.permute(0, 1, 3, 2)  # [B, H, c, N]
        k = k @ self.proj_n  # [B, H, c, k]

        # TODO: should call this only once during inference.
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            basis = self.proj_back_n.permute(1, 0)
        else:
            basis = self.proj_back_n.permute(1, 0)
            # basis[basis.abs() <= cfg.BASIS_THRESHOLD] = 0.
            # For Linear attention visualization
            basis = self.basis_threshold(basis.abs())

        # Compute low-rank approximation of the attention matrix
        # q: [B, H, N, C]   k: [B, H, c, K]
        cheap_attn = (q @ k) * self.scale  # [B, H, N, K]
        cheap_attn = cheap_attn[..., 1:, :]  # [B, H, N-1, K] remove cls token
        basis_coef = cheap_attn.softmax(dim=-1)  # [B, H, N-1, K] coef is naturally sparse
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            approx_attn = basis_coef @ basis  # [B, H, N-1, N]
        else:
            if cfg.BASIS_COEF.USE_TOPK:
                basis_coef_topk, basis_coef_topk_indices = basis_coef.topk(cfg.BASIS_COEF.TOPK, sorted=False)
                basis_coef = torch.zeros_like(basis_coef, device=basis_coef.device)
                basis_coef.scatter_(-1, basis_coef_topk_indices, basis_coef_topk)
            elif cfg.BASIS_COEF.THRESHOLD > 0:
                # basis_coef[basis_coef <= cfg.BASIS_COEF.THRESHOLD] = 0.
                basis_coef = self.basis_coef_threshold(basis_coef)
            approx_attn = basis_coef @ basis  # [B, H, N-1, N]

        # Zero out attention connectivity columns corresponding to inactive tokens
        attn_score = approx_attn.clone()  # [B, H, N-1, N]
        if token_mask is not None:
            attn_score[..., 1:].masked_fill_(~token_mask[:, None, None, :], float('-inf'))  # [B, H, N-1, N]

        # Generate columns of instance dependent sparse attention connectivity pattern
        if cfg.ATTN_SCORE.USE_TOPK:
            # Top-k attention connectivity
            topk_cont_indices = torch.topk(attn_score, self.attn_budget, sorted=False)[1]  # [B, H, N-1, num_cont]
            attn_mask = torch.zeros_like(attn_score, dtype=attn_score.dtype, device=attn_score.device)
            attn_mask.scatter_(-1, topk_cont_indices, True)  # [B, H, N-1, N]
        elif cfg.ATTN_SCORE.THRESHOLD > 0:
            # Threshold attention connectivity
            attn_mask = torch.where(attn_score <= cfg.ATTN_SCORE.THRESHOLD, 0., 1.)
        else:
            raise NotImplementedError

        # Zero out attention connectivity rows corresponding to inactive tokens
        if token_mask is not None and cfg.PRUNE_ATTN_MATRIX_ROW:
            attn_mask *= token_mask[:, None, :, None]  # [B, H, N-1, N]

        # Add cls token back to attn mask
        cls_mask = torch.ones(B, H, 1, N, dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=2)  # [B, H, N, N]
        attn_mask.detach_()  # TODO: No gradient for attn_mask

        out_dict['basis_coef'] = basis_coef
        out_dict['approx_attn'] = approx_attn
        out_dict['attn_mask'] = attn_mask
        if not self.training:
            if cfg.OUT_BASIS_SPARSITY:
                out_dict['basis_sparsity'] = compute_sparsity(basis)
            if cfg.OUT_BASIS_COEF_SPARSITY:
                out_dict['basis_coef_sparsity'] = compute_sparsity(basis_coef)
            if cfg.OUT_ATTN_MASK_SPARSITY:
                out_dict['attn_mask_sparsity'] = compute_sparsity(attn_mask)
        return out_dict


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_tokens=197,
            num_heads=8,
            attn_keep_rate=0.25,
            token_keep_rate=0.50,
            token_pruning_this_layer=False,
            reduce_n_factor=8,
            reduce_c_factor=2,
            share_inout_proj=False,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            cfg=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_predictor = MaskPredictor(
            dim,
            num_heads=num_heads,
            num_tokens=num_tokens,
            attn_keep_rate=attn_keep_rate,
            reduce_n_factor=reduce_n_factor,
            reduce_c_factor=reduce_c_factor,
            share_inout_proj=share_inout_proj,
            cfg=cfg,
        )
        self.token_pruning_this_layer = token_pruning_this_layer
        self.token_keep_rate = token_keep_rate
        self.token_budget = math.ceil(token_keep_rate * (num_tokens - 1))
        self.cfg = cfg

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        # https://discuss.pytorch.org/t/how-to-implement-the-exactly-same-softmax-as-f-softmax-by-pytorch/44263/9
        B, H, N, N = attn.size()
        attn_policy = policy
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, prev_token_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v shape of [B, H, N, C]

        # Zero out key query values corresponding to inactive tokens
        if prev_token_mask is not None and not self.cfg.LOSS.USE_ATTN_RECON:
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            v[..., 1:, :] = v[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)

        out_dict = self.mask_predictor(q, k, prev_token_mask)
        attn_mask = out_dict['attn_mask']

        attn = (q @ k.transpose(-2, -1)) * self.scale
        unmasked_attn = attn.clone().softmax(dim=-1)
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            attn = attn.softmax(dim=-1)  # Don't distort the token value when reconstructing attention
        else:
            # attn = self.softmax_with_policy(attn, attn_mask)
            attn.masked_fill_(~attn_mask.bool(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = torch.nan_to_num(attn)  # Some rows are pruned and filled with '-inf'

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        out_dict['token_mask'] = prev_token_mask
        if self.token_pruning_this_layer and not self.cfg.LOSS.USE_ATTN_RECON:  # TODO: refactor this
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            token_score = cls_attn.mean(dim=1)  # [B, N-1]
            if prev_token_mask is not None:
                token_score = token_score.masked_fill(~prev_token_mask, float('-inf'))
            topk_token_indices = torch.topk(token_score, self.token_budget, sorted=False)[1]  # [B, left_tokens]
            new_token_mask = torch.zeros_like(token_score, dtype=torch.bool, device=token_score.device)
            new_token_mask.scatter_(-1, topk_token_indices, True)  # [B, N-1]
            out_dict['token_mask'] = new_token_mask  # TODO: would masked_fill be faster than indices fill?

        new_val = {'x': x, 'masked_attn': attn, 'unmasked_attn': unmasked_attn}
        out_dict.update(new_val)
        return out_dict


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_tokens,
            num_heads,
            attn_keep_rate,
            token_keep_rate,
            token_pruning_this_layer,
            reduce_n_factor,
            reduce_c_factor,
            share_inout_proj,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            cfg=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            attn_keep_rate=attn_keep_rate,
            token_keep_rate=token_keep_rate,
            token_pruning_this_layer=token_pruning_this_layer,
            reduce_n_factor=reduce_n_factor,
            reduce_c_factor=reduce_c_factor,
            share_inout_proj=share_inout_proj,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cfg=cfg,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.cfg = cfg

    def forward(self, x, prev_token_mask=None):
        # x: [B, N, C] and token_mask: [B, N-1]
        out_dict = self.attn(self.norm1(x), prev_token_mask)
        new_x = out_dict["x"]
        x = x + self.drop_path(new_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        new_token_mask = out_dict['token_mask']
        if new_token_mask is not None and not self.cfg.LOSS.USE_ATTN_RECON:
            x[:, 1:].masked_fill_(~new_token_mask[..., None], 0.)
        new_val = {"x": x}
        out_dict.update(new_val)
        return out_dict


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Sparsifiner(nn.Module):
    """
    PyTorch impl of : `Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers`  -
        https://arxiv.org/abs/2303.13755
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            hybrid_backbone=None,
            norm_layer=None,
            distill=False,
            attn_keep_rate_list=None,
            token_keep_rate_list=None,
            token_pruning_loc_indicator=None,
            reduce_n_factor=8,
            reduce_c_factor=2,
            share_inout_proj=False,
            cfg=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print('## Sparsifiner ##')
        self.cfg = cfg

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_tokens=num_patches + 1,
                num_heads=num_heads,
                attn_keep_rate=attn_keep_rate_list[i],
                token_keep_rate=token_keep_rate_list[i],
                token_pruning_this_layer=token_pruning_loc_indicator[i],
                reduce_n_factor=reduce_n_factor,
                reduce_c_factor=reduce_c_factor,
                share_inout_proj=share_inout_proj,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                cfg=self.cfg,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.num_heads = num_heads

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.distill = distill

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        cfg = self.cfg.SPAR
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        basis_coef_list = []  # Analogy to cheap_attn in V3
        approx_attn_list = []
        attn_mask_list = []
        masked_attn_list = []
        unmasked_attn_list = []
        token_mask_list = []

        # Sparsity Ratio
        ave_basis_coef_sparsity = 0.
        ave_basis_sparsity = 0.
        ave_attn_mask_sparsity = 0.

        token_mask = None
        for i, blk in enumerate(self.blocks):
            out_dict = blk(x, token_mask)
            x = out_dict['x']
            token_mask = out_dict['token_mask']

            basis_coef_list.append(out_dict['basis_coef'])
            approx_attn_list.append(out_dict['approx_attn'])
            attn_mask_list.append(out_dict['attn_mask'])
            masked_attn_list.append(out_dict['masked_attn'])
            unmasked_attn_list.append(out_dict['unmasked_attn'])
            token_mask_list.append(out_dict['token_mask'])

            if not self.training:
                if 'basis_coef_sparsity' in out_dict:
                    ave_basis_coef_sparsity += out_dict['basis_coef_sparsity']
                if 'basis_sparsity' in out_dict:
                    ave_basis_sparsity += out_dict['basis_sparsity']
                if 'attn_mask_sparsity' in out_dict:
                    ave_attn_mask_sparsity += out_dict['attn_mask_sparsity']

        ave_basis_coef_sparsity /= len(self.blocks)
        ave_basis_sparsity /= len(self.blocks)
        ave_attn_mask_sparsity /= len(self.blocks)

        x = self.norm(x)
        features = x[:, 1:]
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)

        out_dict = {
            'x': x,
            'features': features,
            'basis_coef_list': basis_coef_list,
            'approx_attn_list': approx_attn_list,
            'attn_mask_list': attn_mask_list,
            'masked_attn_list': masked_attn_list,
            'unmasked_attn_list': unmasked_attn_list,
            'token_mask_list': token_mask_list
        }

        if not self.training:
            if cfg.OUT_BASIS_SPARSITY:
                out_dict['basis_sparsity'] = ave_basis_sparsity
            if cfg.OUT_BASIS_COEF_SPARSITY:
                out_dict['basis_coef_sparsity'] = ave_basis_coef_sparsity
            if cfg.OUT_ATTN_MASK_SPARSITY:
                out_dict['attn_mask_sparsity'] = ave_attn_mask_sparsity
        return out_dict


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
