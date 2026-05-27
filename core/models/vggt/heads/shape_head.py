# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Shape head: iterative SMPL-X-like shape/betas refinement (AdaLN trunk + deltas; no camera activations).

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.models.vggt.layers.block import Block
from core.models.vggt.layers.mlp import Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale and shift modulation (DiT-style)."""
    return x * (1 + scale) + shift


class ShapeHead(nn.Module):
    """
    Predicts a shape/beta vector from token features via iterative refinement (AdaLN + trunk + residual delta),
    mirroring CameraHead but without pose/camera activations (plain Euclidean regression).
    """

    def __init__(
        self,
        dim_in: int,
        target_dim: int,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth

        self.trunk_blocks = nn.ModuleList(
            [
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_shape_tokens = nn.Parameter(torch.zeros(1, 1, target_dim))
        self.embed_shape = nn.Linear(target_dim, dim_in)

        self.shapeLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.shape_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=target_dim,
            drop=0,
        )

    def _trunk_forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.trunk_blocks:
            x, _ = block(x)
        return x

    def trunk_fn(self, shape_tokens: torch.Tensor, num_iterations: int) -> List[torch.Tensor]:
        """
        Args:
            shape_tokens: [B, S, C], S is typically 1.
            num_iterations: number of refine steps.

        Returns:
            List of cumulative shape predictions [B, target_dim] per iteration (no activation).
        """
        B, S, C = shape_tokens.shape
        pred_shape: Optional[torch.Tensor] = None
        pred_shape_list: List[torch.Tensor] = []

        for _ in range(num_iterations):
            if pred_shape is None:
                module_input = self.embed_shape(self.empty_shape_tokens.expand(B, S, -1))
            else:
                pred_shape = pred_shape.detach()
                module_input = self.embed_shape(pred_shape)

            if module_input.dim() == 2:
                module_input = module_input.unsqueeze(1)
            if module_input.shape[1] == 1 and S > 1:
                module_input = module_input.expand(B, S, C)

            shift_msa, scale_msa, gate_msa = self.shapeLN_modulation(module_input).chunk(3, dim=-1)

            shape_tokens_modulated = gate_msa * modulate(self.adaln_norm(shape_tokens), shift_msa, scale_msa)
            shape_tokens_modulated = shape_tokens_modulated + shape_tokens

            shape_tokens_modulated = self._trunk_forward(shape_tokens_modulated)
            pred_shape_delta = self.shape_branch(self.trunk_norm(shape_tokens_modulated))

            if pred_shape is None:
                pred_shape = pred_shape_delta
            else:
                pred_shape = pred_shape + pred_shape_delta

            if pred_shape.shape[1] == 1:
                pred_shape_list.append(pred_shape.squeeze(1))
            else:
                pred_shape_list.append(pred_shape.mean(dim=1))

        return pred_shape_list

    def forward(self, shape_tokens: torch.Tensor, num_iterations: int = 4) -> List[torch.Tensor]:
        """
        Args:
            shape_tokens: [B, C] or [B, 1, C] aggregated shape-token features.
            num_iterations: refinement steps.

        Returns:
            List of shape predictions [B, target_dim] per iteration.
        """
        if shape_tokens.dim() == 2:
            shape_tokens = shape_tokens.unsqueeze(1)
        shape_tokens = self.token_norm(shape_tokens)
        return self.trunk_fn(shape_tokens, num_iterations)

    def forward_from_sequence(
        self,
        tokens: torch.Tensor,
        shape_token_idx: int = 0,
        num_iterations: int = 4,
    ) -> List[torch.Tensor]:
        """
        Extract the shape slot from multi-view tokens, mean over views, then refine.

        Args:
            tokens: [B, V, P, C] (invalid views are expected to be dropped upstream).
            shape_token_idx: index of the shape token along P (matches VGGTAggregator.layout).
            num_iterations: passed to ``forward``.

        Returns:
            One list entry per refinement stage, each tensor ``[B, target_dim]``.
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B, V, P, C], got shape {tuple(tokens.shape)}")
        shape_slot = tokens[:, :, shape_token_idx, :]
        shape_agg = shape_slot.mean(dim=1)
        return self.forward(shape_agg, num_iterations=num_iterations)
