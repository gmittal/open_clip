"""FLAVA model"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
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
    vocab_size: int = 50257
    width: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio = 4
    output_dim: int = 768

    layer_norm_eps: float = 1e-12,
    pad_token_id: int = 0,


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

        self.image_to_mm_projection = nn.Linear(vision_cfg.output_dim, multimodal_cfg.width)
        self.text_to_mm_projection = nn.Linear(text_cfg.output_dim, multimodal_cfg.width)

        self.image_projection = nn.Linear(vision_cfg.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_cfg.output_dim, embed_dim)
        self.mm_projection = nn.Linear(multimodal_cfg.output_dim, embed_dim)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.language.grad_checkpointing(enable)
        self.multimodal.grad_checkpointing(enable)

    def encode_image(self, image, return_sequences=False):
        pass

    def encode_text(self, text, return_sequences=False):
        pass

    def encode_multimodal(self, image, text, return_sequences=False):
        image_hidden = self.visual(image)  # TODO: add image mask

        embed_text = self.text_embedding(text)

        # TODO: check this!
        text_attn_mask = torch.ones(self.context_length + 1, self.context_length + 1, device=text.device)
        text_hidden = self.language(embed_text, text_attn_mask)

        if not return_sequences:
            cls_mm = mm_hidden[:, 0, :]
            return self.mm_projection(cls_mm)
        return mm_hidden

    def forward(
        self,
        *,
        image,
        text,
        text_input_mask,
        text_masked,
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

        # MLM task
        text_masked_hidden = self.language(text_masked_embed, key_padding_mask=text_padding_mask)
        text_masked_recon = self.masked_lm_head(text_masked_hidden[:, 1:, :])

        ####################
        ###### VISION ######
        ####################

        # Contrastive task
        image_hidden = self.visual(image)
        image_features = self.image_projection(image_hidden[:, 0, :])

        # TODO: add logit_scale for softmax temperature for CLIP
        # TODO: add masked autoencoder

        ####################
        #### MULTIMODAL ####
        ####################

        mm_image_hidden = self.image_to_mm_projection(image_hidden)[:, 1:, :]
        mm_text_hidden = self.text_to_mm_projection(text_hidden)[:, 1:, :]
        mm_text_masked_hidden = self.text_to_mm_projection(text_masked_hidden)[:, 1:, :]

        x = torch.cat([image_hidden, text_hidden], dim=1)  # [*, image_ctx + text_ctx, d_mm]
        mm_attn_mask = torch.ones(self.mm_context_length + 1, self.mm_context_length + 1, device=x.device)
        mm_hidden = self.multimodal(x, mm_attn_mask)

        # TODO: add image-text matching
        # TODO: add support in data loader that with itm_probability
        # a random text

        return image_features, text_features, text_masked_recon
