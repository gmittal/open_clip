"""FLAVA model"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Transformer
from .utils import to_2tuple


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.proj

        return x  # x[:, 0, :] is [CLS_I]


class EmbeddingTransformer(nn.Module):
    def __init__(
            self,
            block_size: int,  # n_ctx
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.output_dim = output_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = nn.LayerNorm(width)

        self.positional_embedding = nn.Parameter(scale * torch.randn(block_size + 1, width))

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, n_ctx + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if key_padding_mask is not None:
            # always attend to CLS_T token
            attend_cls = torch.zeros(x.shape[0], 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([attend_cls, key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.proj

        return x  # x[:, 0, :] is [CLS_T]


@dataclass
class FLAVAVisionCfg:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio = 4
    output_dim: int = 768  # output dim of transformer, not embed_dim

    layer_norm_eps: float = 1e-12
    use_image_masking: bool = True


@dataclass
class FLAVATextCfg:
    block_size: int = 512
    vocab_size: int = 50259  # GPT-2 vocab + 2 special tokens (pad, mask)
    width: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio = 4
    output_dim: int = 768

    layer_norm_eps: float = 1e-12,


@dataclass
class FLAVAMultimodalCfg:
    width: int = 768
    layers: int = 6
    heads: int = 12
    mlp_ratio = 4
    output_dim: int = 768

    layer_norm_eps: float = 1e-12,


class FLAVA(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: FLAVAVisionCfg,
            text_cfg: FLAVATextCfg,
            multimodal_cfg: FLAVAMultimodalCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        del quick_gelu  # unused
        del cast_dtype  # TODO(gmittal): implement this

        if isinstance(vision_cfg, dict):
            vision_cfg = FLAVAVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = FLAVATextCfg(**text_cfg)
        if isinstance(multimodal_cfg, dict):
            multimodal_cfg = FLAVAMultimodalCfg(**multimodal_cfg)

        self.context_length = text_cfg.block_size
        grid_size = vision_cfg.image_size // vision_cfg.patch_size
        self.mm_context_length = grid_size * grid_size + self.context_length

        self.visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_cfg.heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=vision_cfg.output_dim,
            act_layer=nn.GELU,
        )

        self.text_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.language = EmbeddingTransformer(
            block_size=self.context_length,
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            mlp_ratio=text_cfg.mlp_ratio,
            output_dim=text_cfg.output_dim,
            act_layer=nn.GELU,
        )
        self.text_masked_lm_head = nn.Sequential(
            nn.Linear(text_cfg.output_dim, text_cfg.width),
            nn.GELU(),
            nn.Linear(text_cfg.width, text_cfg.vocab_size),
        )

        self.multimodal = EmbeddingTransformer(
            block_size=self.mm_context_length,
            width=multimodal_cfg.width,
            layers=multimodal_cfg.layers,
            heads=multimodal_cfg.heads,
            mlp_ratio=multimodal_cfg.mlp_ratio,
            output_dim=multimodal_cfg.output_dim,
            act_layer=nn.GELU,
        )
        self.mm_masked_lm_head = nn.Sequential(
            nn.Linear(multimodal_cfg.output_dim, multimodal_cfg.width),
            nn.GELU(),
            nn.Linear(multimodal_cfg.width, text_cfg.vocab_size),
        )
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, multimodal_cfg.width),
            nn.GELU(),
            nn.Linear(multimodal_cfg.width, 1),
        )

        self.image_to_mm_projection = nn.Linear(vision_cfg.output_dim, multimodal_cfg.width)
        self.text_to_mm_projection = nn.Linear(text_cfg.output_dim, multimodal_cfg.width)

        self.image_projection = nn.Linear(vision_cfg.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_cfg.output_dim, embed_dim)
        self.mm_projection = nn.Linear(multimodal_cfg.output_dim, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.language.grad_checkpointing(enable)
        self.multimodal.grad_checkpointing(enable)

    def encode_image(self, image, normalize=False):
        features = self.visual(image)
        features = self.image_projection(features[:, 0, :])
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, padding_mask=None, normalize=False):
        embed = self.text_embedding(text)
        features = self.language(embed, key_padding_mask=padding_mask)
        features = self.text_projection(features[:, 0, :])
        return F.normalize(features, dim=-1) if normalize else features

    def encode_multimodal(self, image, text, padding_mask=None, normalize=False):
        text_embed = self.text_embedding(text)
        text_hidden = self.language(text_embed, key_padding_mask=padding_mask)

        image_hidden = self.visual(image)

        mm_image_hidden = self.image_to_mm_projection(image_hidden[:, 1:, :])
        mm_text_hidden = self.text_to_mm_projection(text_hidden[:, 1:, :])

        mm_padding_mask = None
        if padding_mask is not None:
            mm_padding_mask = ~(padding_mask.type(torch.bool))
            im_hidden_length = image_hidden.shape[1] - 1  # ignore CLS_I token
            im_pad_mask = torch.zeros(mm_padding_mask.shape[0], im_hidden_length, dtype=torch.bool, device=mm_padding_mask.device)
            mm_padding_mask = torch.cat([im_pad_mask, mm_padding_mask], dim=1)

        mm_inputs = torch.cat([mm_image_hidden, mm_text_hidden], dim=1)
        mm_hidden = self.multimodal(mm_inputs, key_padding_mask=mm_padding_mask)
        mm_features = self.mm_projection(mm_hidden[:, 0, :])
        return F.normalize(mm_features, dim=-1) if normalize else mm_features

    def forward(
        self,
        *,
        image,
        text,
        text_input_mask,
        text_masked,
        image_text_match_images,
    ):
        ####################
        ##### LANGUAGE #####
        ####################

        text_padding_mask = ~(text_input_mask.type(torch.bool))

        text_embed = self.text_embedding(text)
        text_masked_embed = self.text_embedding(text_masked)

        # Contrastive task
        text_hidden = self.language(text_embed, key_padding_mask=text_padding_mask)
        text_features = self.text_projection(text_hidden[:, 0, :])
        text_features = F.normalize(text_features, dim=-1)

        # MLM task
        text_masked_hidden = self.language(text_masked_embed, key_padding_mask=text_padding_mask)
        text_masked_recon_logits = self.text_masked_lm_head(text_masked_hidden[:, 1:, :])

        ####################
        ###### VISION ######
        ####################

        # Contrastive task
        image_hidden = self.visual(image)
        image_features = self.image_projection(image_hidden[:, 0, :])
        image_features = F.normalize(image_features, dim=-1)

        # TODO: MAE task

        ####################
        #### MULTIMODAL ####
        ####################

        mm_padding_mask = ~(text_input_mask.type(torch.bool))
        im_hidden_length = image_hidden.shape[1] - 1  # ignore CLS_I token
        im_pad_mask = torch.zeros(mm_padding_mask.shape[0], im_hidden_length, dtype=torch.bool, device=mm_padding_mask.device)
        mm_padding_mask = torch.cat([im_pad_mask, mm_padding_mask], dim=1)

        # ITM task (corrupted images, uncorrupted text)
        itm_im_hidden = self.visual(image_text_match_images)
        mm_itm_im_hidden = self.image_to_mm_projection(itm_im_hidden[:, 1:, :])
        mm_text_hidden = self.text_to_mm_projection(text_hidden[:, 1:, :])
        mm_inputs = torch.cat([mm_itm_im_hidden, mm_text_hidden], dim=1)  # [*, image_ctx + text_ctx, d_mm]

        mm_hidden = self.multimodal(mm_inputs, key_padding_mask=mm_padding_mask)
        mm_features = self.mm_projection(mm_hidden[:, 0, :])
        itm_pred = self.itm_head(mm_features)

        # MLM task (uncorrupted images, corrupted text)
        mm_image_hidden = self.image_to_mm_projection(image_hidden[:, 1:, :])
        mm_text_masked_hidden = self.text_to_mm_projection(text_masked_hidden[:, 1:, :])
        mm_inputs_masked = torch.cat([mm_image_hidden, mm_text_masked_hidden], dim=1)

        mm_hidden_masked = self.multimodal(mm_inputs_masked, key_padding_mask=mm_padding_mask)
        mm_text_hidden = mm_hidden_masked[:, 1+im_hidden_length:, :]
        mm_text_masked_recon_logits = self.mm_masked_lm_head(mm_text_hidden)

        # TODO: MVLM task (https://openreview.net/pdf?id=ZhuXksSJYWn)

        return image_features, \
               text_features, \
               self.logit_scale.exp(), \
               text_masked_recon_logits, \
               itm_pred, \
               mm_text_masked_recon_logits
