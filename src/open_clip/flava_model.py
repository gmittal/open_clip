"""FLAVA model"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import _build_text_tower
from .transformer import LayerNorm, Transformer, VisionTransformer
from .utils import to_2tuple


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedVisionTransformer(VisionTransformer):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.initialize_weights()

    # def initialize_weights(self):
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.positional_embedding.shape[-1], int(self.grid_size[0]), cls_token=True)
    #     self.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float())

    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.conv1.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     torch.nn.init.normal_(self.class_embedding, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # add pos embed w/o cls token
        x = x + self.positional_embedding[1:, :].to(x.dtype)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.class_embedding.to(x.dtype) + self.positional_embedding[:1, :].to(x.dtype)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply transformer blocks
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x, mask, ids_restore


class MaskedVisionDecoder(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.positional_embedding.shape[-1], int(self.grid_size[0]), cls_token=True)
        self.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.class_embedding, std=.02)

    def forward(self, x):
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        x = x[:, 1:, :]  # remove cls token
        return x


class MultimodalTransformer(nn.Module):
    def __init__(
            self,
            grid_size: int,
            text_context_length: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.output_dim = output_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = norm_layer(width)

        self.num_heads = heads

        context_length = 1 + grid_size * grid_size + text_context_length + 1
        self.positional_embedding = nn.Parameter(scale * torch.randn(context_length, width))
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.class_embedding, std=0.02)

        # pos_embed = get_2d_sincos_pos_embed(self.positional_embedding.shape[-1], int(self.grid_size[0]), cls_token=True)
        # self.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float())

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, 1 + context_length, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # always attend to CLS_M token
        attend_cls = torch.ones(x.shape[0], 1, dtype=torch.long, device=attn_mask.device)
        attn_mask = torch.cat([attend_cls, attn_mask], dim=1).bool()
        key_padding_mask = ~attn_mask

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.proj

        return x  # x[:, 0, :] is [CLS]


@dataclass
class FLAVAVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)

    # MAE parameters
    mae_mask_ratio: float = 0.75
    mae_decoder_layers: int = 2
    mae_decoder_width: int = 512
    mae_decoder_heads: int = 4


@dataclass
class FLAVATextCfg:
    context_length: int = 77
    unimodal_context_length: int = 512
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    hf_model_config: dict = None
    proj: str = None
    pooler_type: str = 'identity_pooler'
    output_tokens: bool = False


@dataclass
class FLAVAMultimodalCfg:
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value

    # Product-of-Experts
    poe_mlm: bool = False
    poe_mae: bool = False


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: FLAVAVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = MaskedVisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        global_average_pool=vision_cfg.global_average_pool,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return visual


def _build_vision_mae_decoder(
        vision_cfg: FLAVAVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    out_channels = vision_cfg.patch_size ** 2 * 3
    visual = MaskedVisionDecoder(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.mae_decoder_width,
        layers=vision_cfg.mae_decoder_layers,
        heads=vision_cfg.mae_decoder_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        global_average_pool=vision_cfg.global_average_pool,
        output_dim=out_channels,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return visual


def _build_multimodal_tower(
        embed_dim: int,
        multimodal_cfg: FLAVAMultimodalCfg,
        grid_size: int,
        text_context_length: int,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    multimodal = MultimodalTransformer(
        grid_size=grid_size,
        text_context_length=text_context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        mlp_ratio=multimodal_cfg.mlp_ratio,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return multimodal


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

        # Vision encoder
        vision_cfg = FLAVAVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        grid_size = vision_cfg.image_size // vision_cfg.patch_size
        visual_context_length = grid_size * grid_size + 1  # +1 for CLS_I token

        # Text encoder
        text_cfg = FLAVATextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        text_context_length = text_cfg.context_length  # includes CLS_T token

        # unimodal MLM
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.text.config.vocab_size),
        )

        # unimodal MAE
        self.mae_mask_ratio = vision_cfg.mae_mask_ratio
        self.patch_mask_token = nn.Parameter(vision_cfg.mae_decoder_width ** -0.5 * torch.randn(vision_cfg.mae_decoder_width))
        self.mae_decoder = _build_vision_mae_decoder(vision_cfg, quick_gelu, cast_dtype)

        # Multimodal encoder
        multimodal_context_length = visual_context_length + text_context_length + 1  # +1 for CLS_M token
        multimodal_cfg = FLAVAMultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        self.multimodal = _build_multimodal_tower(
            embed_dim,
            multimodal_cfg,
            grid_size,
            text_context_length,
            quick_gelu,
            cast_dtype,
        )
        self.image_to_mm_projection = nn.Linear(embed_dim, multimodal_cfg.width)
        self.text_to_mm_projection = nn.Linear(embed_dim, multimodal_cfg.width)
        self.poe_mlm = multimodal_cfg.poe_mlm
        self.poe_mae = multimodal_cfg.poe_mae

        # Cross-modal MLM
        self.mm_mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.text.config.vocab_size),
        )

        # Cross-modal MAE
        self.mm_mae_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, vision_cfg.patch_size ** 2 * 3, bias=True),  # patch reconstruction
        )

        # ITM
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # TODO: should this be width?
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
        )

        # Output projections
        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

        # Contrastive logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.patch_mask_token, std=0.02)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.grad_checkpointing(enable)
        self.multimodal.grad_checkpointing(enable)

    def encode_image(self, image, normalize=False):
        h_image, _, _ = self.visual(image, mask_ratio=0)
        cls_i = h_image[:, 0, :]
        image_encoding = self.image_projection(cls_i)
        return F.normalize(image_encoding, dim=-1) if normalize else image_encoding

    def encode_text(self, text, normalize=False):
        h_text = self.text(text)
        cls_t = h_text[:, 0, :]
        text_encoding = self.text_projection(cls_t)
        return F.normalize(text_encoding, dim=-1) if normalize else text_encoding

    def encode_multimodal(self, image, text, normalize=False):
        h_image, _, _ = self.visual(image, mask_ratio=0)
        h_text = self.text(text)
        mm_h_image = self.image_to_mm_projection(h_image)
        mm_h_text = self.text_to_mm_projection(h_text)

        # Multimodal attention mask (ignore text padding tokens)
        mm_text_attn_mask = (text != self.text.config.pad_token_id).long()
        mm_vis_attn_mask = torch.ones(*h_image.shape[:2], dtype=torch.long, device=mm_text_attn_mask.device)
        mm_attn_mask = torch.cat([mm_vis_attn_mask, mm_text_attn_mask], dim=1)

        mm_input = torch.cat([mm_h_image, mm_h_text], dim=1)
        h_m = self.multimodal(mm_input, attn_mask=mm_attn_mask)
        return h_m[:, 0, :]

    def forward_itm(self, image, text):
        h_image, _, _ = self.visual(image, mask_ratio=0)
        h_text = self.text(text)
        mm_h_image = self.image_to_mm_projection(h_image)
        mm_h_text = self.text_to_mm_projection(h_text)

        # Multimodal attention mask (ignore text padding tokens)
        mm_text_attn_mask = (text != self.text.config.pad_token_id).long()
        mm_vis_attn_mask = torch.ones(*h_image.shape[:2], dtype=torch.long, device=mm_text_attn_mask.device)
        mm_attn_mask = torch.cat([mm_vis_attn_mask, mm_text_attn_mask], dim=1)

        mm_input = torch.cat([mm_h_image, mm_h_text], dim=1)
        h_m = self.multimodal(mm_input, attn_mask=mm_attn_mask)
        cls_m = h_m[:, 0, :]
        return self.itm_head(cls_m)

    def forward_mlm(self, *, text_masked, mlm_labels):
        h_masked_text = self.text(text_masked)
        mlm_logits = self.mlm_head(self.text_projection(h_masked_text[:, 1:, :]))
        return {
            'mlm_logits': mlm_logits,
            'mlm_labels': mlm_labels[:, 1:],  # remove cls token from labels
        }

    def forward_mae(self, *, image):
        h_masked_image, mae_mask, ids_restore = self.visual(image, mask_ratio=self.mae_mask_ratio)

        mae_mask_tokens = self.patch_mask_token.repeat(
            h_masked_image.shape[0],
            ids_restore.shape[1] + 1 - h_masked_image.shape[1],
            1,
        )
        image_with_mask_tokens = torch.cat([h_masked_image[:, 1:, :], mae_mask_tokens], dim=1)  # no cls token
        image_with_mask_tokens = torch.gather(
            image_with_mask_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, h_masked_image.shape[2]),
        )  # unshuffle
        image_with_mask_tokens = torch.cat([h_masked_image[:, :1, :], image_with_mask_tokens], dim=1)  # append cls token

        mae_logits = self.mae_decoder(self.image_projection(image_with_mask_tokens))

        return {
            'mae_logits': mae_logits,
            'mae_mask': mae_mask,
            'image': image,
        }

    def forward_flava(self, *, image, text, text_masked, mlm_labels):
        h_text = self.text(text)
        h_masked_text = self.text(text_masked)
        h_image, _, _ = self.visual(image, mask_ratio=0)

        # CLIP
        cls_t = h_text[:, 0, :]
        text_encoding = self.text_projection(cls_t)
        text_encoding = F.normalize(text_encoding, dim=-1)
        cls_i = h_image[:, 0, :]
        image_encoding = self.image_projection(cls_i)
        image_encoding = F.normalize(image_encoding, dim=-1)

        # MAE encoder
        h_masked_image, mae_mask, ids_restore = self.visual(image, mask_ratio=self.mae_mask_ratio)

        # Multimodal attention mask (ignore text padding tokens)
        mm_text_attn_mask = (text != self.text.config.pad_token_id).long()
        mm_vis_attn_mask = torch.ones(*h_image.shape[:2], dtype=torch.long, device=mm_text_attn_mask.device)
        mm_attn_mask = torch.cat([mm_vis_attn_mask, mm_text_attn_mask], dim=1)

        # Multimodal inputs
        mm_h_image = self.image_to_mm_projection(h_image)
        mm_h_text = self.text_to_mm_projection(h_text)
        mm_h_masked_text = self.text_to_mm_projection(h_masked_text)

        # Image-text matching (ITM)
        logits = self.logit_scale * (image_encoding @ text_encoding.T)
        local_bs = h_text.shape[0]
        pos_idx = torch.arange(local_bs // 2).to(h_text.device)

        # ITM: positives
        mm_pos = self.multimodal(
            torch.cat([mm_h_image, mm_h_text], dim=1),
            attn_mask=mm_attn_mask,
        )

        # ITM: select a negative image for each text
        weights_t2i = torch.softmax(logits.T, dim=1).fill_diagonal_(0)
        idx_t2i = torch.multinomial(weights_t2i, 1).view(-1)
        idx_t2i = idx_t2i[: local_bs // 2]
        mm_neg_h_image = torch.index_select(mm_h_image, 0, idx_t2i)
        mm_pos_h_text = torch.index_select(mm_h_text, 0, pos_idx)
        mm_neg_t2i = self.multimodal(
            torch.cat([mm_neg_h_image, mm_pos_h_text], dim=1),
            attn_mask=mm_attn_mask[: local_bs // 2],
        )

        # ITM: select a negative text for each image
        weights_i2t = torch.softmax(logits, dim=1).fill_diagonal_(0)
        idx_i2t = torch.multinomial(weights_i2t, 1).view(-1)
        idx_i2t = idx_i2t[: local_bs // 2]
        mm_pos_h_image = torch.index_select(mm_h_image, 0, pos_idx)
        mm_neg_h_text = torch.index_select(mm_h_text, 0, idx_i2t)
        mm_neg_i2t = self.multimodal(
            torch.cat([mm_pos_h_image, mm_neg_h_text], dim=1),
            attn_mask=mm_attn_mask[: local_bs // 2],
        )

        # ITM: compute logits and labels
        itm_vl_embed = torch.cat([mm_pos, mm_neg_t2i, mm_neg_i2t], dim=0)
        itm_logits = self.itm_head(itm_vl_embed[:, 0, :])
        itm_labels = torch.cat([
            torch.ones(mm_pos.shape[0], dtype=torch.long, device=mm_pos.device),
            torch.zeros(mm_neg_t2i.shape[0] + mm_neg_i2t.shape[0], dtype=torch.long, device=mm_pos.device)
        ], dim=0)


        """
        # Multimodal cross-attention MLM (full images, masked text)
        mm_mlm_input = torch.cat([mm_h_image, mm_h_masked_text], dim=1)
        h_m_mlm = self.multimodal(mm_mlm_input, attn_mask=mm_attn_mask)
        mm_masked_text_pred = h_m_mlm[:, 1 + mm_h_image.shape[1]:, :]
        mm_mlm_logits = self.mm_mlm_head(self.mm_projection(mm_masked_text_pred[:, 1:, :]))  # remove cls token

        # Multimodal cross-attention MAE task (masked images, full text)
        mae_mask_tokens = self.patch_mask_token.repeat(
            h_masked_image.shape[0],
            ids_restore.shape[1] + 1 - h_masked_image.shape[1],
            1,
        )
        image_with_mask_tokens = torch.cat([h_masked_image[:, 1:, :], mae_mask_tokens], dim=1)  # no cls token
        image_with_mask_tokens = torch.gather(
            image_with_mask_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, h_masked_image.shape[2]),
        )  # unshuffle
        image_with_mask_tokens = torch.cat([h_masked_image[:, :1, :], image_with_mask_tokens], dim=1)  # append cls token

        mm_image_with_mask_tokens = self.image_to_mm_projection(image_with_mask_tokens)
        mm_mae_input = torch.cat([mm_image_with_mask_tokens, mm_h_text], dim=1)
        h_m_mae = self.multimodal(mm_mae_input, attn_mask=mm_attn_mask)
        mm_masked_patches_pred = h_m_mae[:, 1:mm_image_with_mask_tokens.shape[1] + 1, :]
        mm_mae_logits = self.mm_mae_head(self.mm_projection(mm_masked_patches_pred[:, 1:, :]))  # remove cls token

        # Product-of-Experts (Hinton '02)
        # https://aclanthology.org/2021.acl-long.522.pdf
        if self.poe_mlm:
            uni_mlm_logits = self.mlm_head(self.text_projection(h_masked_text[:, 1:, :]))
            assert uni_mlm_logits.shape == mm_mlm_logits.shape
            mm_mlm_logits = mm_mlm_logits + uni_mlm_logits
        if self.poe_mae:
            uni_mae_logits = self.mae_decoder(self.image_projection(image_with_mask_tokens))
            assert uni_mae_logits.shape == mm_mae_logits.shape
            mm_mae_logits = mm_mae_logits + uni_mae_logits
        """

        return {
            # contrastive outputs
            'image_features': image_encoding,
            'text_features': text_encoding,
            'logit_scale': self.logit_scale.exp(),

            # multimodal outputs
            'itm_logits': itm_logits,
            # 'mm_mlm_logits': mm_mlm_logits,
            # 'mm_mae_logits': mm_mae_logits,

            # labels
            'mlm_labels': mlm_labels[:, 1:],  # remove cls token from labels
            'itm_labels': itm_labels,
            'mae_mask': mae_mask,
            'image': image,
        }

    def forward(
        self,
        *,
        image=None,
        text=None,
        text_masked=None,

        # passthrough
        mlm_labels=None,

        # unimodal flags
        unimodal_mae=False,
        unimodal_mlm=False,

        output_dict=True,
    ):
        if unimodal_mlm:
            assert text_masked is not None and mlm_labels is not None
            return self.forward_mlm(
                text_masked=text_masked,
                mlm_labels=mlm_labels,
            )
        elif unimodal_mae:
            assert image is not None
            return self.forward_mae(image=image)
        else:
            assert image is not None and \
                   text is not None and \
                   mlm_labels is not None
            return self.forward_flava(
                image=image,
                text=text,
                text_masked=text_masked,
                mlm_labels=mlm_labels,
            )
