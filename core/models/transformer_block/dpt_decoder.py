# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-07-17 23:07:39
# @Function      : efficient point-image transformer block using merge and unmerge strategy

import sys
from typing import Any, Dict

sys.path.append("./")
import torch
import torch.nn as nn
from core.models.heads.dpt_head import DPTHead
from core.models.heads.mlp_head import MLPPixelShuffleDPTHead
from core.models.transformer_block.transformer_dit import (
    SD3MMJointRopeTransformerBlock,
    SD3MMJointTransformerBlockV2,
)
from core.models.vggt_transformer import PositionGetter
from core.structures.slidewindows import SlidingAverageTimer
from diffusers.utils import is_torch_version
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.

    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class EfficientDPTDecoder(nn.Module):
    """
    EfficientDPTDecoder

    An efficient point-image transformer decoder block based on a merge and unmerge strategy, designed for unified point and image representation processing.
    This class stacks multiple SD3MMJointTransformerBlockV2 layers to process encoded tokens, supporting features such as gradient checkpointing,
    configurable contextual attention, and patch size adaptation.

    Args:
        enc_channels (int): Number of channels per encoder token.
        img_dim (int): Dimension of image feature space.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        enc_patch_size (int, optional): Patch size for image encoding. Defaults to 14.
        gradient_checkpointing (bool, optional): Enable gradient checkpointing for reduced memory. Defaults to False.
        depth (int, optional): Number of transformer layers. Defaults to 4.
        merge_ratio (float, optional): Merge ratio for fused token interaction. Defaults to 0.5.
        **kwargs: Additional arguments (e.g., 'accumulate').

    """

    def __init__(
        self,
        enc_channels,
        img_dim,
        num_heads=16,
        enc_patch_size=14,
        gradient_checkpointing=False,
        depth=4,
        merge_ratio=0.5,
        **kwargs,
    ):
        """
        Initialize the EfficientDPTDecoder module.

        This initializes a stack of SD3MMJointTransformerBlockV2 layers for efficient point-image
        transformer decoding, setting up the transformer blocks, decoder head, and relevant configurations.

        Args:
            enc_channels (int): Number of channels for encoder tokens.
            img_dim (int): Image feature dimension.
            num_heads (int, optional): Number of attention heads for transformer blocks. Defaults to 16.
            enc_patch_size (int, optional): Patch size used for image encoding. Defaults to 14.
            gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing. Defaults to False.
            depth (int, optional): Number of transformer layers. Defaults to 4.
            merge_ratio (float, optional): Ratio for merging operations inside the transformer. Defaults to 0.5.
            **kwargs: Additional keyword arguments, such as 'accumulate', for controlling accumulation strategy.
        """

        super().__init__()
        self.enc_channels = enc_channels
        self.enc_patch_size = enc_patch_size
        # Transformer blocks
        self.transformer_block = nn.ModuleList()
        for i in range(depth):
            self.transformer_block.append(
                SD3MMJointTransformerBlockV2(
                    pv_dim=enc_channels,
                    img_dim=img_dim,
                    dim=img_dim,
                    num_heads=num_heads,
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=False,
                )
            )

        self.depth = depth
        # Decoder
        self.build_decoder()

        self.gradient_checkpointing = gradient_checkpointing
        self.accumulate = kwargs.get("accumulate", "concat")
        self.ratio = merge_ratio

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """
        Helper function to create a tokenizer for patch embedding.

        Args:
            in_channels (int): Number of input channels per patch.
            patch_size (int): Size of each patch (assumed square).
            d_model (int): Output embedding dimension for each patch.

        Returns:
            nn.Sequential: Sequential module that rearranges and linearly projects input patches.
        """
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def build_decoder(self):
        """
        Build the DPT decoder head for efficient multi-view prediction.

        Sets up and registers the dense predictor module (DPTHead) with the proper
        intermediate layer mappings based on the decoder depth, supporting multi-stage
        supervision and efficient dense prediction for input features.

        This method determines which intermediate transformer blocks' outputs to use
        for decoding, enabling hierarchical feature fusion at multiple levels.
        """
        intermediate_mapping = {
            1: [0, 0, 0, 0],
            4: [0, 1, 2, 3],
            8: [1, 3, 6, 7],
            12: [1, 3, 6, 11],
            16: [3, 7, 12, 15],
            24: [4, 11, 17, 23],
        }

        # DPT decoder
        self.dense_predictor = DPTHead(
            dim_in=128, intermediate_layer_idx=intermediate_mapping[self.depth]
        )

    def _merge_mv(self, mv_tokens, mode="enc", depth_i=0):
        """
        Merge multi-view tokens by concatenating the sequence and patch dimensions
        when in encoder mode and accumulation strategy is 'concat'.

        Args:
            mv_tokens (Tensor): Input tokens of shape (b, s, p, c)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder
            depth_i (int): Depth index, used for potential future logic

        Returns:
            Tuple[Tensor, Tensor]:
                - Merged tokens of shape (b, s*p, c)
                - Merge size mask of shape (b, s*p, 1), all ones
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        # In encoder mode, merge the sequence and patch dimensions
        register_tokens, mv_tokens = mv_tokens[:, :, :5], mv_tokens[:, :, 5:]
        mv_tokens = rearrange(mv_tokens, "b s p c -> b (s p) c")

        return register_tokens, mv_tokens

    def _demerge_mv(
        self, merge_mv_tokens, register_tokens, mv_views, unmerge_func, mode="enc"
    ):
        """
        Reverse the merging of multi-view tokens by reshaping back to original
        multi-view layout when in encoder mode and using 'concat' accumulation.

        Args:
            mv_tokens (Tensor): Placeholder for output (not used when merging)
            merge_mv_tokens (Tensor): Merged tokens of shape (b, s*p, c)
            mv_p (int): Number of patches per view (used to unflatten)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder

        Returns:
            Tensor: Demerged tokens of shape (b, s, p, c)
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        merge_mv_tokens = unmerge_func(merge_mv_tokens)

        # Only demerge in encoder mode; otherwise, keep as-is
        mv_tokens = rearrange(merge_mv_tokens, "b (s p) c -> b s p c", s=mv_views)

        # merge mv tokens with register_tokens

        return torch.cat([register_tokens, mv_tokens], dim=2)

    def forward_multi_modality_layer(
        self, layer, x, cond, motion_emb=None, pos_emb=None
    ):
        """
        Forward a single multi-modality transformer layer with support for motion and positional embeddings,
        and handling of gradient checkpointing as needed.

        Args:
            layer (nn.Module): The transformer layer/block to be applied.
            x (Tensor): Query tokens/features (e.g., shape [B, N, C] or [V, P, C]).
            cond (Tensor): Conditional/context tokens for cross-attention (shape compatible with layer).
            motion_emb (Tensor, optional): Motion embedding tensor, if required by the layer. Default: None.
            pos_emb (Tensor, optional): Positional embedding tensor, if required by the layer. Default: None.

        Returns:
            Tuple[Tensor, Tensor]:
                - Output query tokens/features (same shape as x).
                - Output context/conditional tokens (same shape as cond, possibly updated).
        """

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            x, cond = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                x,
                cond,
                motion_emb,
                pos_emb,
                **ckpt_kwargs,
            )
        else:
            x, cond = layer(
                x,
                cond,
                motion_emb,
                pos_emb,
            )

        return x, cond

    def predict_rgbs(self, cond_imgs, query_imgs, motion_emb, render_h, render_w):
        """
        Predict RGB images and masks from conditional images, query images, and motion embeddings.

        Args:
            cond_imgs (torch.Tensor): Conditional/context images of shape [B, V, P, C].
            query_imgs (torch.Tensor): Query images to render, typically as [V, C, H, W].
            motion_emb (torch.Tensor): Motion embedding tensor, shape [B, D_mod] or similar.
            render_h (int): Rendered image height.
            render_w (int): Rendered image width.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - rgb: Predicted RGB images, shape depends on dense_predictor.
                - mask: Corresponding mask predictions of rendered images.
        """

        query_tokens = rearrange(query_imgs, "v c h w -> v (h w) c")
        target_views = query_tokens.shape[0]
        cond_imgs = rearrange(cond_imgs, "b v p c -> b (v p) c")
        cond_imgs = repeat(cond_imgs, "b np d -> (b vt) np d", vt=target_views)
        motion_emb = repeat(motion_emb, "b d -> (b vt) d", vt=target_views)

        # decoder_only transformer
        intermediate_tokens = []
        for transformer in self.transformer_block:
            query_tokens, cond_imgs = self.forward_multi_modality_layer(
                transformer, query_tokens, cond_imgs, motion_emb
            )
            intermediate_tokens.append(query_tokens.unsqueeze(0))

        rgb, mask = self.dense_predictor(intermediate_tokens, render_h, render_w)

        return rgb, mask

    def forward(
        self,
        cond_imgs_list: torch.Tensor = None,
        batch_query_imgs: torch.Tensor = None,
        motion_emb_list: torch.Tensor = None,
        render_h: int = None,
        render_w: int = None,
        pos_emb_list: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass of the transformer model.
        Args:
            query_points (torch.Tensor): Input tensor of shape [N, L, D].
            cond_imgs (torch.Tensor, optional):  [B S P C] from pretrained_models
            motion_emb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """
        batch = len(cond_imgs_list)
        pred_rgbs = []
        pred_masks = []
        for batch_id in range(batch):
            cond_imgs = cond_imgs_list[batch_id]
            pos_emb = pos_emb_list[batch_id] if pos_emb_list is not None else None

            query_imgs = batch_query_imgs[batch_id]
            motion_emb = motion_emb_list[batch_id]

            pred_rgb, pred_mask = self.predict_rgbs(
                cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb
            )
            pred_rgbs.append(pred_rgb)
            pred_masks.append(pred_mask)

        return torch.cat(pred_rgbs, dim=0), torch.cat(pred_masks, dim=0)


class PatchEfficientDPTDecoder(EfficientDPTDecoder):
    """
    PatchEfficientDPTDecoder

    An extension of EfficientDPTDecoder tailored for patch-based processing. This class implements
    a transformer-based decoder module that efficiently handles image tokens in a patchified manner,
    with support for multi-head self-attention, flexible context fusion strategies, and hierarchical prediction.
    """

    def __init__(
        self,
        enc_channels,
        img_dim,
        num_heads=16,
        enc_patch_size=14,
        gradient_checkpointing=False,
        depth=4,
        merge_ratio=0.5,
        **kwargs,
    ):
        """
        Initializes the PatchEfficientDPTDecoder.

        This constructor extends the EfficientDPTDecoder for efficient patch-based transformer decoding.
        It sets up the model for patchified input, multi-head self-attention, and hierarchical decoder configuration.

        Args:
            enc_channels (int): Number of encoder channels per image patch.
            img_dim (int): Feature dimensionality per patch token.
            num_heads (int, optional): Number of transformer self-attention heads. Defaults to 16.
            enc_patch_size (int, optional): Patch size for patchifying the input images. Defaults to 14.
            gradient_checkpointing (bool, optional): If True, enables gradient checkpointing for reduced memory usage. Defaults to False.
            depth (int, optional): Number of stacked transformer layers. Defaults to 4.
            merge_ratio (float, optional): Ratio controlling fusion strategy for patch tokens. Defaults to 0.5.
            **kwargs: Additional configurable arguments (e.g., 'accumulate' strategy).
        """

        super().__init__(
            enc_channels=enc_channels,
            img_dim=img_dim,
            num_heads=num_heads,
            enc_patch_size=enc_patch_size,
            gradient_checkpointing=gradient_checkpointing,
            depth=depth,
            merge_ratio=merge_ratio,
            **kwargs,
        )

        # Override transformer blocks for patch-based processing
        self.transformer_block = nn.ModuleList()
        for i in range(depth):
            self.transformer_block.append(
                SD3MMJointRopeTransformerBlock(
                    dim=img_dim,
                    num_heads=num_heads,
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                )
            )

        self.patchify = self._create_tokenizer(enc_channels, enc_patch_size, img_dim)
        self.position_getter = PositionGetter()

    def predict_rgbs(
        self, cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb=None
    ):
        """
        Predict RGB values from input images using the patch-based efficient DPT decoder.

        Args:
            cond_imgs (torch.Tensor): Conditional images tensor of shape [B, V, P, C].
            query_imgs (torch.Tensor): Query images tensor to be patchified and embedded.
            motion_emb (torch.Tensor): Motion embedding tensor.
            render_h (int): Render output height.
            render_w (int): Render output width.
            pos_emb (torch.Tensor, optional): Optional positional embedding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted RGB images and mask tensors.
        """

        query_tokens = self.patchify(query_imgs.unsqueeze(0))

        target_views = query_tokens.shape[0]
        cond_imgs = rearrange(cond_imgs, "b v p c -> b (v p) c")
        cond_imgs = repeat(cond_imgs, "b np d -> (b vt) np d", vt=target_views)
        motion_emb = repeat(motion_emb, "b d -> (b vt) d", vt=target_views)

        # decoder_only transformer
        intermediate_tokens = []

        for transformer in self.transformer_block:
            query_tokens, cond_imgs = self.forward_multi_modality_layer(
                transformer, query_tokens, cond_imgs, motion_emb, pos_emb=None
            )
            intermediate_tokens.append(query_tokens.unsqueeze(0))
        rgb, mask = self.dense_predictor(intermediate_tokens, render_h, render_w)

        return rgb, mask

    def build_decoder(self):
        """
        Build the DPT decoder head for efficient multi-view prediction.

        Sets up and registers the dense predictor module (DPTHead) with the proper
        intermediate layer mappings based on the decoder depth, supporting multi-stage
        supervision and efficient dense prediction for input features.

        This method determines which intermediate transformer blocks' outputs to use
        for decoding, enabling hierarchical feature fusion at multiple levels.
        """
        intermediate_mapping = {
            1: [0, 0, 0, 0],
            4: [0, 1, 2, 3],
            8: [1, 3, 6, 7],
            12: [1, 3, 6, 11],
            16: [3, 7, 12, 15],
            24: [4, 11, 17, 23],
        }

        # DPT decoder
        self.dense_predictor = DPTHead(
            dim_in=1024,
            patch_size=self.enc_patch_size,
            intermediate_layer_idx=intermediate_mapping[self.depth],
        )


class PatchRopeEfficientDPTDecoder(PatchEfficientDPTDecoder):
    def forward_multi_modality_layer(
        self, transformer, query_tokens, cond_imgs, motion_emb, pos_emb
    ):
        """
        Forward pass for a single multi-modal transformer layer.

        Processes the query tokens with conditional image features, motion embeddings,
        and optional position embeddings using the given transformer block.

        Args:
            transformer (nn.Module): Transformer layer/block to process tokens.
            query_tokens (Tensor): Input query tokens of shape (B, N, C).
            cond_imgs (Tensor): Conditional image features for fusion.
            motion_emb (Tensor): Motion embedding features.
            pos_emb (Tensor, optional): Optional positional embeddings.

        Returns:
            Tuple[Tensor, Tensor]:
                - Output query tokens processed by the transformer block.
                - Output conditional image features (for chaining/fusion).
        """

        if pos_emb is not None:
            # Some transformers accept pos_emb, so pass if present
            out_query, out_cond = transformer(
                query_tokens, cond_imgs, motion_emb, pos_emb
            )
        else:
            out_query, out_cond = transformer(query_tokens, cond_imgs, motion_emb)
        return out_query, out_cond

    def predict_rgbs(
        self, cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb=None
    ):
        """
        Predict RGB images and masks using the DPT decoder with patch-based multi-modal input.

        This function processes conditional images, query images, motion embeddings, and optional
        position embeddings to predict target RGB images and their associated masks. It first
        computes patch tokens for the target query images, prepares the conditional and positional
        embeddings for each view, and then sequentially applies each transformer block on the
        input tokens. The intermediate transformer outputs are fed to the DPTHead to generate
        dense RGB predictions and masks.

        Args:
            cond_imgs (Tensor): Conditional image features of shape [B, V, P, C], where
                B is batch size, V is the number of views, P is the number of patches/tokens
                per view, and C is the feature dimension.
            query_imgs (Tensor): The target query images for which RGBs are to be predicted,
                shape [B, C, H, W].
            motion_emb (Tensor): Motion embeddings of shape [B, D], used for temporal or
                pose information.
            render_h (int): The height of the rendered output image.
            render_w (int): The width of the rendered output image.
            pos_emb (Tensor, optional): Position embeddings for the queries. If provided, must
                be compatible with target patches and used for spatial alignment.

        Returns:
            Tuple[Tensor, Tensor]:
                - rgb (Tensor): The predicted RGB outputs for each query image, shape [B, 3, H, W].
                - mask (Tensor): The foreground/background mask for each prediction, shape [B, 1, H, W].
        """

        query_tokens = self.patchify(query_imgs.unsqueeze(0))

        if pos_emb is not None:
            B, C, H, W = query_imgs.shape
            PH = H // self.enc_patch_size
            PW = W // self.enc_patch_size
            target_pos_emb = self.position_getter(B, PH, PW, device=pos_emb.device)

            # pose view embedding
            B, N, P = pos_emb.shape
            max_offset_h = pos_emb[..., 0].max()
            max_offset_w = pos_emb[..., 1].max()
            view_ids = torch.arange(B).view(B, 1, 1).to(pos_emb)
            offset_x = view_ids * max_offset_w
            offset_y = view_ids * max_offset_h
            offset_windows = torch.cat([offset_y, offset_x], dim=-1)
            offset_windows = offset_windows + view_ids
            pos_emb = pos_emb + offset_windows
            pos_emb[..., 0] += PH
            pos_emb[..., 1] += PW

        target_views = query_tokens.shape[0]
        cond_imgs = rearrange(cond_imgs, "b v p c -> b (v p) c")
        cond_imgs = repeat(cond_imgs, "b np d -> (b vt) np d", vt=target_views)
        motion_emb = repeat(motion_emb, "b d -> (b vt) d", vt=target_views)

        if pos_emb is not None:
            pos_emb = rearrange(pos_emb, "v l d -> 1 (v l) d")
            pos_emb = repeat(pos_emb, "b l d -> (b vt) l d", vt=target_views)
            pos_emb = torch.cat([target_pos_emb, pos_emb], dim=1)

        # decoder_only transformer
        intermediate_tokens = []
        for transformer in self.transformer_block:
            query_tokens, cond_imgs = self.forward_multi_modality_layer(
                transformer, query_tokens, cond_imgs, motion_emb, pos_emb
            )
            intermediate_tokens.append(query_tokens.unsqueeze(0))

        rgb, mask = self.dense_predictor(intermediate_tokens, render_h, render_w)

        return rgb, mask


class PatchDPTDecoderOnly(EfficientDPTDecoder):
    """
    PatchDPTDecoderOnly

    An efficient DPT transformer decoder specialized for patch-based multi-view queries.
    This decoder extends EfficientDPTDecoder to provide tokenization for per-patch processing,
    spatial position encoding, and dense RGB/mask prediction for high-resolution downstream tasks.

    Args:
        enc_channels (int): Number of input feature channels.
        img_dim (int): Transformer input/output dimension.
        num_heads (int, optional): Number of transformer attention heads.
        enc_patch_size (int, optional): Patch size for tokenization.
        gradient_checkpointing (bool, optional): Enable gradient checkpointing to reduce memory usage.
        depth (int, optional): Number of transformer layers.
        merge_ratio (float, optional): Patch-token merge ratio.
        **kwargs: Extra keyword arguments for decoder/head configuration.

    This class implements patch-level input tokenization and view-aware position embedding,
    supporting efficient multi-view dense prediction (e.g., in generative reconstruction and rendering).
    """

    def __init__(
        self,
        enc_channels,
        img_dim,
        num_heads=16,
        enc_patch_size=14,
        gradient_checkpointing=False,
        depth=4,
        merge_ratio=0.5,
        **kwargs,
    ):
        super().__init__(
            enc_channels=enc_channels,
            img_dim=img_dim,
            num_heads=num_heads,
            enc_patch_size=enc_patch_size,
            gradient_checkpointing=gradient_checkpointing,
            depth=depth,
            merge_ratio=merge_ratio,
            **kwargs,
        )
        self.patchify = self._create_tokenizer(enc_channels, enc_patch_size, img_dim)
        self.position_getter = PositionGetter()

    def build_decoder(self):
        """
        Build the DPT decoder head for efficient multi-view prediction.

        Sets up and registers the dense predictor module (DPTHead) with the proper
        intermediate layer mappings based on the decoder depth, supporting multi-stage
        supervision and efficient dense prediction for input features.

        This method determines which intermediate transformer blocks' outputs to use
        for decoding, enabling hierarchical feature fusion at multiple levels.
        """
        intermediate_mapping = {
            1: [0, 0, 0, 0],
            4: [0, 1, 2, 3],
            8: [1, 3, 6, 7],
            12: [1, 3, 6, 11],
            16: [3, 7, 12, 15],
            24: [4, 11, 17, 23],
        }

        # DPT decoder
        self.dense_predictor = DPTHead(
            dim_in=1024,
            patch_size=self.enc_patch_size,
            intermediate_layer_idx=intermediate_mapping[self.depth],
        )

    def predict_rgbs(
        self, cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb=None
    ):
        """
        Predict RGB values and mask from input images using the DPT decoder (decoder only, without transformer blocks).

        Args:
            cond_imgs (torch.Tensor): Conditional images tensor of shape [B, V, P, C].
            query_imgs (torch.Tensor): Query images tensor to be patchified and embedded.
            motion_emb (torch.Tensor): Motion embedding tensor.
            render_h (int): Render output height.
            render_w (int): Render output width.
            pos_emb (torch.Tensor, optional): Optional positional embedding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted RGB images and mask tensors.
        """

        query_tokens = self.patchify(query_imgs.unsqueeze(0))
        intermediate_tokens = [query_tokens.unsqueeze(0)]
        rgb, mask = self.dense_predictor(intermediate_tokens, render_h, render_w)

        return rgb, mask


class PatchDPT4DecoderOnly(PatchDPTDecoderOnly):
    """
    PatchDPT4DecoderOnly

    A decoder-only patch-based DPT decoder. This class implements a DPTHead-based decoder for dense prediction,
    operating solely on the patchified query images and their features, without the use of any transformer blocks.
    It is suitable for use as a decoder head in patch-based neural architectures.
    """

    def __init__(
        self,
        enc_channels,
        img_dim,
        num_heads=16,
        enc_patch_size=14,
        gradient_checkpointing=False,
        depth=4,
        merge_ratio=0.5,
        **kwargs,
    ):
        """
        Initializes the PatchDPT4DecoderOnly.

        This class is a decoder-only version for patch-based DPT decoding, operating on patchified queries without transformer blocks.

        Args:
            enc_channels (int): Number of encoder channels per image patch.
            img_dim (int): Feature dimensionality per patch token.
            num_heads (int, optional): Number of transformer heads (unused, for API compatibility).
            enc_patch_size (int, optional): Patch size for patchifying input images. Defaults to 14.
            gradient_checkpointing (bool, optional): If True, enables gradient checkpointing. Defaults to False.
            depth (int, optional): Number of depth stages (used only for grouping outputs, not for transformer blocks). Defaults to 4.
            merge_ratio (float, optional): Patch merging ratio (unused in decoder-only). Defaults to 0.5.
            **kwargs: Additional optional keyword arguments.
        """

        super().__init__(
            enc_channels=enc_channels,
            img_dim=img_dim,
            num_heads=num_heads,
            enc_patch_size=enc_patch_size,
            gradient_checkpointing=gradient_checkpointing,
            depth=depth,
            merge_ratio=merge_ratio,
            **kwargs,
        )
        self.patchify = self._create_tokenizer(enc_channels, enc_patch_size, img_dim)
        self.position_getter = PositionGetter()

    def build_decoder(self, img_dim=1024):
        """
        Build the DPT decoder head for efficient multi-view prediction.

        Sets up and registers the dense predictor module (DPTHead) with the proper
        intermediate layer mappings based on the decoder depth, supporting multi-stage
        supervision and efficient dense prediction for input features.

        This method determines which intermediate transformer blocks' outputs to use
        for decoding, enabling hierarchical feature fusion at multiple levels.
        """
        intermediate_mapping = {
            1: [0, 0, 0, 0],
            4: [0, 1, 2, 3],
            8: [1, 3, 6, 7],
            12: [1, 3, 6, 11],
            16: [3, 7, 12, 15],
            24: [4, 11, 17, 23],
        }

        # DPT decoder
        self.dense_predictor = DPTHead(
            dim_in=1024 // self.depth,
            patch_size=self.enc_patch_size,
            intermediate_layer_idx=intermediate_mapping[self.depth],
        )

    def predict_rgbs(
        self, cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb=None
    ):
        """
        Predict RGB values and mask from input images using the DPT decoder (decoder only, without transformer blocks).

        Args:
            cond_imgs (torch.Tensor): Conditional images tensor of shape [B, V, P, C].
            query_imgs (torch.Tensor): Query images tensor to be patchified and embedded.
            motion_emb (torch.Tensor): Motion embedding tensor.
            render_h (int): Render output height.
            render_w (int): Render output width.
            pos_emb (torch.Tensor, optional): Optional positional embedding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted RGB images and mask tensors.
        """
        query_tokens = self.patchify(query_imgs.unsqueeze(0))
        intermediate_tokens = torch.chunk(query_tokens, self.depth, dim=-1)
        intermediate_tokens = [
            query_tokens.unsqueeze(0) for query_tokens in intermediate_tokens
        ]

        rgb, mask = self.dense_predictor(intermediate_tokens, render_h, render_w)

        return rgb, mask


class PatchMLPPixelShuffleDPT4DecoderOnly(PatchDPT4DecoderOnly):
    """Same pipeline as ``PatchDPT4DecoderOnly``, but uses ``MLPPixelShuffleDPTHead`` (partial scratch + MLP + PixelShuffle)."""

    def __init__(
        self,
        enc_channels,
        img_dim,
        num_heads=16,
        enc_patch_size=14,
        gradient_checkpointing=False,
        depth=4,
        merge_ratio=0.5,
        pixel_shuffle_factor=2,
        **kwargs,
    ):
        self.pixel_shuffle_factor = pixel_shuffle_factor
        super().__init__(
            enc_channels=enc_channels,
            img_dim=img_dim,
            num_heads=num_heads,
            enc_patch_size=enc_patch_size,
            gradient_checkpointing=gradient_checkpointing,
            depth=depth,
            merge_ratio=merge_ratio,
            **kwargs,
        )

    def build_decoder(self, img_dim=1024):
        intermediate_mapping = {
            1: [0, 0, 0, 0],
            4: [0, 1, 2, 3],
            8: [1, 3, 6, 7],
            12: [1, 3, 6, 11],
            16: [3, 7, 12, 15],
            24: [4, 11, 17, 23],
        }
        self.dense_predictor = MLPPixelShuffleDPTHead(
            dim_in=1024 // self.depth,
            patch_size=self.enc_patch_size,
            intermediate_layer_idx=intermediate_mapping[self.depth],
            pixel_shuffle_factor=self.pixel_shuffle_factor,
        )


if __name__ == "__main__":

    pi_model = PatchDPTDecoderOnly(enc_channels=128, img_dim=1024, depth=1)
    pi_model.cuda()

    cond_imgs_list = [torch.randn(1, 4, 2565, 1024).cuda().float()]
    query_imgs = torch.randn(1, 1, 128, int(1024// 14 * 14), int(1024// 14 * 14)).cuda()
    motion_emb_list = [torch.randn(1, 1024).cuda().float()]

    slide_window = SlidingAverageTimer(window_size=10)

    for i in range(100):
        with torch.no_grad():
            slide_window.start()
            output = pi_model(cond_imgs_list, query_imgs, motion_emb_list, 1024, 1024)
            torch.cuda.synchronize()
            slide_window.record()
            print(f"{slide_window}")
