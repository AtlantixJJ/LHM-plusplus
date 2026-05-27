# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2026-03-10 10:00:00
# @Function      : LHM++ Gradio App inference logic

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from core.utils.hf_hub import wrap_model_hub

logger = logging.getLogger(__name__)

# Default batch size for inference
DEFAULT_BATCH_SIZE = 40


def parse_app_configs(
    model_cards: Dict[str, Dict[str, str]],
) -> Tuple[DictConfig, DictConfig]:
    """Parse model configuration from environment variables and config files.

    Returns:
        A tuple of (cfg, cfg_train) containing merged configurations.
    """
    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    app_model_name = os.environ.get("APP_MODEL_NAME")
    if app_model_name is None:
        raise NotImplementedError("APP_MODEL_NAME environment variable must be set")

    model_card = model_cards[app_model_name]
    model_path = model_card["model_path"]
    model_config = model_card["model_config"]

    cli_cfg.model_name = model_path

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        use_pred_render = OmegaConf.select(
            cfg_train, "model.use_pred_shape_for_render", default=False
        )
        cfg.use_smplx_shape_estimator = not bool(use_pred_render)
        if not cfg.use_smplx_shape_estimator:
            logger.warning(
                "SMPL-X FREE model: DO NOT USE SMPL-X PRIOR "
                "(PoseEstimator / SMPL-X shape-from-image estimation is disabled)."
            )
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except AttributeError:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high

        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def build_app_model(cfg: DictConfig) -> torch.nn.Module:
    """Build and load the LHM model from pretrained weights.

    Args:
        cfg: Configuration object containing model_name and other parameters.

    Returns:
        Loaded LHM model ready for inference.
    """
    from core.models import model_dict

    model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
    model = model_cls.from_pretrained(cfg.model_name)
    return model


@torch.no_grad()
def inference_results(
    model: torch.nn.Module,
    ref_img_tensors: torch.Tensor,
    smplx_params: Dict[str, torch.Tensor],
    motion_seq: Dict[str, Any],
    video_size: int = 40,
    ref_imgs_bool: Optional[torch.Tensor] = None,
    visualized_center: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
    infer_output_renderer: str = "neural",
) -> np.ndarray:
    """Run inference on a motion sequence with batching to prevent OOM.

    Args:
        model: LHM model for human animation.
        ref_img_tensors: Reference image tensors of shape (N, C, H, W).
        smplx_params: SMPL-X parameters for the initial pose.
        motion_seq: Dictionary containing motion sequence data.
        video_size: Total number of frames to render.
        ref_imgs_bool: Boolean mask indicating which reference images to use.
        visualized_center: If True, crops output to subject bounds with 10% padding.
        batch_size: Number of frames to process in each batch.
        device: Device to run inference on.
        infer_output_renderer: ``"neural"`` applies the refinement decoder after GS
            rasterization; ``"gs"`` outputs rasterized Gaussian splat RGB only.

    Returns:
        Rendered RGB frames as numpy array of shape (T, H, W, 3).
    """
    offset_list = motion_seq.get("offset_list")
    ori_h, ori_w = motion_seq.get("ori_size", (512, 512))
    output_rgb = torch.ones((ori_h, ori_w, 3))
    ref_imgs_bool = torch.ones(
        ref_img_tensors.shape[0], dtype=torch.bool, device=device
    )

    smplx_params_dev = {k: v.to(device) for k, v in smplx_params.items()}
    use_pred_render = getattr(model, "use_pred_shape_for_render", False)

    model_outputs = model.infer_single_view(
        ref_img_tensors.unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params=smplx_params_dev,
        ref_imgs_bool=ref_imgs_bool.unsqueeze(0),
        return_pred_shape=use_pred_render,
    )

    pred_shape = None
    if len(model_outputs) == 8:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
            pred_shape,
        ) = model_outputs
    elif len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
        ) = model_outputs
    else:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
        ) = model_outputs
        pos_emb = None

    if pred_shape is not None and use_pred_render:
        smplx_for_animation = type(model).smplx_params_with_pred_shape_betas(
            smplx_params_dev, pred_shape
        )
        animation_betas = smplx_for_animation["betas"]
    else:
        animation_betas = smplx_params_dev["betas"]

    batch_smplx_params = {
        "betas": animation_betas,
        "transform_mat_neutral_pose": transform_mat_neutral_pose,
    }

    frame_varying_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "focal",
        "princpt",
        "img_size_wh",
        "expr",
    ]

    batch_rgb_list = []
    batch_mask_list = []
    num_batches = (video_size + batch_size - 1) // batch_size

    for batch_idx in range(0, video_size, batch_size):
        current_batch = batch_idx // batch_size + 1
        print(f"Processing batch {current_batch}/{num_batches}")

        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_idx : batch_idx + batch_size
                ].to(device)
                for key in frame_varying_keys
            }
        )

        mask_seqs = (
            motion_seq.get("masks", [])[batch_idx : batch_idx + batch_size]
            if "masks" in motion_seq
            else None
        )

        anim_kwargs = {
            "gs_model_list": gs_model_list,
            "query_points": query_points,
            "smplx_params": batch_smplx_params,
            "render_c2ws": motion_seq["render_c2ws"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_intrs": motion_seq["render_intrs"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_bg_colors": motion_seq["render_bg_colors"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "gs_hidden_features": gs_hidden_features,
            "image_latents": image_latents,
            "motion_emb": motion_emb,
        }

        if pos_emb is not None:
            anim_kwargs["pos_emb"] = pos_emb
        if offset_list is not None:
            anim_kwargs["offset_list"] = offset_list[batch_idx : batch_idx + batch_size]
        if mask_seqs is not None:
            anim_kwargs["mask_seqs"] = mask_seqs
        if output_rgb is not None:
            anim_kwargs["output_rgb"] = output_rgb
        anim_kwargs["infer_output_renderer"] = infer_output_renderer

        batch_rgb, batch_mask = model.animation_infer(**anim_kwargs)
        batch_rgb_list.append((batch_rgb.clamp(0, 1) * 255).to(torch.uint8).numpy())
        batch_mask_list.append((batch_mask.clamp(0, 1) * 255).to(torch.uint8).numpy())
    
    print("End of inference")
    
    if visualized_center:
        mask_numpy = np.concatenate(batch_mask_list, axis=0)
        h_indices, w_indices = np.where(mask_numpy > 0.25)[1:]

        if len(h_indices) > 0 and len(w_indices) > 0:
            top, bottom = h_indices.min(), h_indices.max()
            left, right = w_indices.min(), w_indices.max()

            center_y, center_x = (top + bottom) / 2, (left + right) / 2
            height, width = bottom - top, right - left
            new_height, new_width = height * 1.1, width * 1.1

            top_new = max(0, int(center_y - new_height / 2))
            bottom_new = int(center_y + new_height / 2)
            left_new = max(0, int(center_x - new_width / 2))
            right_new = int(center_x + new_width / 2)

            rgb = np.concatenate(batch_rgb_list, axis=0)
            output = rgb[:, top_new:bottom_new, left_new:right_new]
        else:
            output = np.concatenate(batch_rgb_list, axis=0)
    else:
        output = np.concatenate(batch_rgb_list, axis=0)

    return output
