# MLPPixelShuffleDPTHead: DPT-style multi-scale tokens → fusion, but replaces the original ``refinenet1``
# block (heavy 3×3 fusion + 2× bilinear) with a lighter path: 1×1 ``l1_residual_fuse`` + 1×1 ``refinenet1_ps_mlp``
# + ``pixel_shuffle(u)`` where ``u = pixel_shuffle_factor`` (default 2).

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_head import _make_fusion_block, _make_scratch, custom_interpolate
from .head_act import activate_head
from .head_utils import create_uv_grid, position_grid_to_embed


class MLPPixelShuffleDPTHead(nn.Module):
    """
    Dense prediction head with the same **token → multi-scale maps** stage as ``DPTHead`` (``LayerNorm``,
    ``projects``, ``resize_layers``), then the same **first three** fusion blocks: ``refinenet4`` … ``refinenet2``.

    **Difference vs** ``DPTHead``: the last fusion/step-up originally done by ``refinenet1`` + ``output_conv1``
    is replaced by ``l1_residual_fuse`` + ``refinenet1_ps_mlp`` + ``pixel_shuffle(self.u)``. There is **no**
    ``output_conv1`` / ``output_conv2`` stack.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "sigmoid",
        features: int = 128,
        out_channels: Optional[List[int]] = None,
        intermediate_layer_idx: Optional[List[int]] = None,
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        pixel_shuffle_factor: int = 2,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = [128, 256, 512, 512]
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]

        self.patch_size = patch_size
        self.activation = activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.u = int(pixel_shuffle_factor)
        if self.u < 1:
            raise ValueError("pixel_shuffle_factor must be >= 1")

        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)

        self.l1_residual_fuse = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        _out_c = features * (self.u**2) if self.feature_only else output_dim * (self.u**2)
        self.refinenet1_ps_mlp = nn.Conv2d(
            features,
            _out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        render_h: int,
        render_w: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        H, W = render_h, render_w
        S = aggregated_tokens_list[0].shape[1]
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, (H, W))
        raise NotImplementedError

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = out + self.l1_residual_fuse(layer_1_rn)
        out = self.refinenet1_ps_mlp(out)
        out = F.pixel_shuffle(out, self.u)
        del layer_1_rn, layer_1

        return out

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        size: Tuple[int, int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        H, W = size
        B, S = aggregated_tokens_list[0].shape[:2]

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        interp_h = int(patch_h * self.patch_size / self.down_ratio)
        interp_w = int(patch_w * self.patch_size / self.down_ratio)

        out: List[torch.Tensor] = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1

        fused = self.scratch_forward(out)

        out = custom_interpolate(
            fused,
            (interp_h, interp_w),
            mode="bilinear",
            align_corners=True,
        )
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        preds, masks = activate_head(out, activation=self.activation)
        preds = preds.view(B, S, *preds.shape[1:])
        masks = masks.view(B, S, *masks.shape[1:])
        return preds, masks
