"""
Sparsifiner:
- Use LVVIT as backbone
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}


class GroupLinear(nn.Module):
    """
    Group Linear operator
    """

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)

        self.group_weight = nn.Parameter(torch.zeros(self.groups, self.group_in_dim, self.group_out_dim))
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t, b, d = x.size()
        x = x.view(t, b, self.groups, int(d / self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)).reshape(t, b, self.out_dim) + self.group_bias
        return out

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    """
    MLP with support to use group linear operator
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features, group)
            self.fc2 = GroupLinear(hidden_features, out_features, group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_groups, embed_dim, eps=1e-5, affine=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, embed_dim, eps, affine)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.gn(x)
        x = x.view(B, T, C)
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

    def forward(self, q, k):
        out_dict = {}
        cfg = self.cfg.SPAR

        B, H, N, C = q.shape
        assert self.num_tokens == N
        q, k = self.proj_c_q(q), self.proj_c_k(k)  # [B, H, N, c]
        k = k.permute(0, 1, 3, 2)  # [B, H, c, N]
        k = k @ self.proj_n  # [B, H, c, k]

        # TODO: should call this only once during inference.
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            basis = self.proj_back_n.permute(1, 0)
        else:
            basis = self.proj_back_n.permute(1, 0)
            # basis[basis.abs() <= cfg.BASIS_THRESHOLD] = 0. # Can't use in-place operation on tensor needs gradient.
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

        # Generate sparse attention mask
        attn_score = approx_attn.clone()  # [B, H, N-1, N]

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
        self.cfg = cfg

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v shape of [B, H, N, C]

        out_dict = self.mask_predictor(q, k)
        attn_mask = out_dict['attn_mask']

        # trick here to make q@k.t more stable
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        unmasked_attn = attn.clone().softmax(dim=-1)
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            attn = attn.softmax(dim=-1)  # Don't distort the token value when reconstructing attention
        else:
            # attn = self.softmax_with_policy(attn, attn_mask)
            attn.masked_fill_(~attn_mask.bool(), float('-inf'))
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        new_val = {'x': x, 'masked_attn': attn, 'unmasked_attn': unmasked_attn}
        out_dict.update(new_val)
        return out_dict


class Block(nn.Module):
    """
    Pre-layernorm transformer block
    """

    def __init__(
            self,
            dim,
            num_tokens,
            num_heads,
            attn_keep_rate,
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
            group=1,
            skip_lam=1.,  # Modified
            cfg=None,
    ):
        super().__init__()
        self.skip_lam = skip_lam
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            attn_keep_rate=attn_keep_rate,
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

    def forward(self, x):
        # x: [B, N, C]
        out_dict = self.attn(self.norm1(x))
        new_x = out_dict["x"]
        x = x + self.drop_path(new_x) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        new_val = {"x": x}
        out_dict.update(new_val)
        return out_dict


class MHABlock(nn.Module):
    """
    Multihead Attention block with residual branch
    """

    def __init__(
            self,
            dim,
            num_tokens,
            num_heads,
            attn_keep_rate,
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
            group=1,
            skip_lam=1.,  # Modified
            cfg=None,
    ):
        super().__init__()
        self.skip_lam = skip_lam
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            attn_keep_rate=attn_keep_rate,
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
        self.cfg = cfg

    def forward(self, x):
        # x: [B, N, C]
        out_dict = self.attn(self.norm1(x * self.skip_lam))
        new_x = out_dict["x"]
        x = x + self.drop_path(new_x) / self.skip_lam
        new_val = {"x": x}
        out_dict.update(new_val)
        return out_dict


class FFNBlock(nn.Module):
    """
    Feed forward network with residual branch
    """

    def __init__(
            self,
            dim,
            num_tokens,
            num_heads,
            attn_keep_rate,
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
            group=1,
            skip_lam=1.,  # Modified
            cfg=None,
    ):
        super().__init__()
        self.skip_lam = skip_lam
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            group=group,
        )
        self.cfg = cfg

    def forward(self, x):
        # x: [B, N, C]
        out_dict = {"x": x}
        x = x + self.drop_path(self.mlp(self.norm2(x * self.skip_lam))) / self.skip_lam
        new_val = {"x": x}
        out_dict.update(new_val)
        return out_dict


class PatchEmbedNaive(nn.Module):
    """
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

    # def flops(self):
    #     img_size = self.img_size[0]
    #     block_flops = dict(
    #         proj=img_size * img_size * 3 * self.embed_dim,
    #     )
    #     return sum(block_flops.values())


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.proj(x)
        return x


class PatchEmbed4_2(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


class PatchEmbed4_2_128(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    # def flops(self):
    #     img_size = self.img_size[0]
    #     block_flops = dict(
    #         conv1=img_size / 2 * img_size / 2 * 3 * 128 * 7 * 7,
    #         conv2=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
    #         conv3=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
    #         proj=img_size / 2 * img_size / 2 * 128 * self.embed_dim,
    #     )
    #     return sum(block_flops.values())


def get_block(block_type, **kargs):
    if block_type == 'mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type == 'ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type == 'tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay == 'fix':
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


class SparsifinerLVViT(nn.Module):
    """SparsifinerLVViT with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
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
            qkv_bias=False,  # Modified
            qk_scale=None,
            representation_size=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            drop_path_decay='linear',
            hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            p_emb='4_2',
            skip_lam=1.0,
            order=None,
            mix_token=False,
            return_dense=False,
            distill=False,
            attn_keep_rate_list=None,
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

        print('## SparsifinerLVViT ##')
        self.cfg = cfg

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes == 0 else num_classes

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb == '4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb == '4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                              embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_tokens=num_patches + 1,
                    num_heads=num_heads,
                    attn_keep_rate=attn_keep_rate_list[i],
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
                    skip_lam=skip_lam,
                    cfg=self.cfg,
                )
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                          dim=embed_dim,
                          num_tokens=num_patches + 1,
                          num_heads=num_heads,
                          attn_keep_rate=attn_keep_rate_list[i],
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
                          skip_lam=skip_lam,
                          cfg=self.cfg,
                          )
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.return_dense = return_dense
        self.mix_token = mix_token

        if return_dense:
            self.aux_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        self.distill = distill

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # Modified
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        basis_coef_list = []  # Analogy to cheap_attn in V3
        approx_attn_list = []
        attn_mask_list = []
        masked_attn_list = []
        unmasked_attn_list = []

        # Sparsity Ratio
        ave_basis_coef_sparsity = 0.
        ave_basis_sparsity = 0.
        ave_attn_mask_sparsity = 0.

        for i, blk in enumerate(self.blocks):
            out_dict = blk(x)
            x = out_dict['x']

            basis_coef_list.append(out_dict['basis_coef'])
            approx_attn_list.append(out_dict['approx_attn'])
            attn_mask_list.append(out_dict['attn_mask'])
            masked_attn_list.append(out_dict['masked_attn'])
            unmasked_attn_list.append(out_dict['unmasked_attn'])

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
        x_cls = self.head(x[:, 0])
        x_aux = self.aux_head(x[:, 1:])
        final_pred = x_cls + 0.5 * x_aux.max(1)[0]

        out_dict = {
            'features': x_aux,
            'basis_coef_list': basis_coef_list,
            'approx_attn_list': approx_attn_list,
            'attn_mask_list': attn_mask_list,
            'masked_attn_list': masked_attn_list,
            'unmasked_attn_list': unmasked_attn_list
        }
        if self.training:
            out_dict['x'] = x_cls
        else:
            out_dict['x'] = final_pred

        if not self.training:
            if cfg.OUT_BASIS_SPARSITY:
                out_dict['basis_sparsity'] = ave_basis_sparsity
            if cfg.OUT_BASIS_COEF_SPARSITY:
                out_dict['basis_coef_sparsity'] = ave_basis_coef_sparsity
            if cfg.OUT_ATTN_MASK_SPARSITY:
                out_dict['attn_mask_sparsity'] = ave_attn_mask_sparsity
        return out_dict
