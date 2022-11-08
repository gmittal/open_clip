"""FLAVA model"""
# from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .model import Transformer, LayerNorm
from .utils import to_2tuple


class ImageEncoder(nn.Module):
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
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
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


class TransformerEncoder(nn.Module):
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
        self.ln_pre = LayerNorm(width)

        self.positional_embedding = nn.Parameter(scale * torch.randn(block_size + 1, width))

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, n_ctx + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
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
    hidden_size: int = 768  # output dim of transformer, not embed_dim

    layer_norm_eps: float = 1e-12
    use_image_masking: bool = True


@dataclass
class FLAVATextCfg:
    block_size: int = 512
    vocab_size: int = 50257
    width: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio = 4
    hidden_size: int = 768

    layer_norm_eps: float = 1e-12,
    pad_token_id: int = 0,


@dataclass
class FLAVAMultimodalCfg:
    width: int = 768
    layers: int = 6
    heads: int = 12
    mlp_ratio = 4
    hidden_size: int = 768

    layer_norm_eps: float = 1e-12,


class FLAVA(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: FLAVAVisionCfg,
            text_cfg: FLAVATextCfg,
            multimodal_cfg: FLAVAMultimodalCfg,
            quick_gelu: bool = False,
    ):
        super().__init__()

        del quick_gelu  # unused

        if isinstance(vision_cfg, dict):
            vision_cfg = FLAVAVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = FLAVATextCfg(**text_cfg)
        if isinstance(multimodal_cfg, dict):
            multimodal_cfg = FLAVAMultimodalCfg(**multimodal_cfg)

        self.context_length = text_cfg.block_size
        grid_size = vision_cfg.image_size // vision_cfg.patch_size
        self.mm_context_length = grid_size * grid_size + self.context_length

        self.image_encoder = ImageEncoder(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_cfg.heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=vision_cfg.hidden_size,
            act_layer=nn.GELU,
        )

        self.text_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.text_encoder = TransformerEncoder(
            block_size=self.context_length,
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            mlp_ratio=text_cfg.mlp_ratio,
            output_dim=text_cfg.hidden_size,
            act_layer=nn.GELU,
        )

        self.mm_encoder = TransformerEncoder(
            block_size=self.mm_context_length,
            width=multimodal_cfg.width,
            layers=multimodal_cfg.layers,
            heads=multimodal_cfg.heads,
            mlp_ratio=multimodal_cfg.mlp_ratio,
            output_dim=multimodal_cfg.hidden_size,
            act_layer=nn.GELU,
        )

        self.image_to_mm_projection = nn.Linear(vision_cfg.hidden_size, multimodal_cfg.width)
        self.text_to_mm_projection = nn.Linear(text_cfg.hidden_size, multimodal_cfg.width)

        self.image_projection = nn.Linear(vision_cfg.hidden_size, embed_dim)
        self.text_projection = nn.Linear(text_cfg.hidden_size, embed_dim)
        self.mm_projection = nn.Linear(multimodal_cfg.hidden_size, embed_dim)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, return_sequences=False):
        hidden_state = self.image_encoder(image)
        if not return_sequences:
            cls_i = hidden_state[:, 0, :]
            return self.image_projection(cls_i)
        return hidden_state

    def encode_text(self, text, return_sequences=False):
        x = self.text_embedding(text)  # [batch_size, n_ctx, d_model]
        attn_mask = torch.ones(self.context_length + 1, self.context_length + 1, device=text.device)
        hidden_state = self.text_encoder(x, attn_mask)
        if not return_sequences:
            cls_t = hidden_state[:, 0, :]
            return self.text_projection(cls_t)
        return hidden_state

    def encode_multimodal(self, image, text, return_sequences=False):
        image_hidden = self.image_encoder(image)  # TODO: add image mask

        embed_text = self.text_embedding(text)

        # TODO: check this!
        text_attn_mask = torch.ones(self.context_length + 1, self.context_length + 1, device=text.device)
        text_hidden = self.text_encoder(embed_text, text_attn_mask)

        image_hidden = self.image_to_mm_projection(image_hidden)[:, 1:, :]
        text_hidden = self.text_to_mm_projection(text_hidden)[:, 1:, :]
        x = torch.cat([image_hidden, text_hidden], dim=1)  # [*, image_ctx + text_ctx, d_mm]
        mm_attn_mask = torch.ones(self.mm_context_length + 1, self.mm_context_length + 1, device=x.device)
        mm_hidden = self.mm_encoder(x, mm_attn_mask)
        if not return_sequences:
            cls_mm = mm_hidden[:, 0, :]
            return self.mm_projection(cls_mm)
        return mm_hidden

    def forward(self, image, text):
        # TODO: this is going to be complicated with all of the masks, losses, etc.

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        # image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        # text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()
