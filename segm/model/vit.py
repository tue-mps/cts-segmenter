"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from segm.model.utils import init_weights, resize_pos_embed
from segm.model.blocks import Block

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights


from segm.model.efficient import images_to_patches, policy_indices_by_policynet_pred, policy_indices_no_sharing


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        policy_method,
        policy_schedule=None,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        policynet_ckpt=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        self.policy_method = policy_method
        self.policy_schedule = policy_schedule

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, policynet_pred=None, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        # pre-process the images and reduce number of patches
        if self.policy_method == 'policy_net':
            policy_indices = policy_indices_by_policynet_pred(im, patch_size=PS,
                                                              policy_schedule=self.policy_schedule,
                                                              policynet_pred=policynet_pred)
        elif self.policy_method == 'no_sharing':
            policy_indices = policy_indices_no_sharing(im, patch_size=PS)
        else:
            raise NotImplementedError('only policy_net is currently supported as a policy method.')

        x, policy_code = images_to_patches(im, patch_size=PS, policy_indices=policy_indices)
        num_patches = policy_code.size(1)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed  # pos_embed at base scale, without grouping
        num_extra_tokens = 1 + self.distilled

        posemb_tok, posemb_grid = (
            pos_embed[:, :num_extra_tokens],
            pos_embed[0, num_extra_tokens:],
        )
        posemb_grid = posemb_grid.reshape(1, H // PS, W // PS, -1).permute(0, 3, 1, 2)
        # positional embedding is reshaped in image format, and grouped using the same strategy
        # patch size is set to 1, since 1*1 size for each patch with base size
        posemb_grid = posemb_grid.expand(B, -1, -1, -1)
        posemb_grid, _ = images_to_patches(posemb_grid, patch_size=1, policy_indices=policy_indices)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(B, num_patches, -1)
        pos_embed = torch.cat([posemb_tok.expand(B, -1, -1), posemb_grid], dim=1)
        x = x + pos_embed
        x = self.dropout(x)

        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
        x = self.norm(x)

        if return_features:
            return x, policy_code, policy_indices

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)

        return x, policy_code, policy_indices

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
