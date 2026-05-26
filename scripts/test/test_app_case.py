#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Run app inference case directly without Gradio UI for debugging

"""
Run app.py inference case directly without Gradio UI.

Executes the same logic as the Gradio "Generate" button for debugging.
Use this to test inference without manually clicking through the UI.

Usage:
    # Default case: yuliang images + TAICHI motion (auto-downloads prior + model if needed)
    python scripts/test/test_app_case.py

    # Specify model, images, motion
    python scripts/test/test_app_case.py --model_name LHMPP-700M-PixelShuffle \
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png" \
        --motion_video "./motion_video/Dance_I/Dance_I.mp4" \
        --motion_size 120 --ref_view 8

    # Override model path (skip AutoModelQuery, use local checkpoint)
    python scripts/test/test_app_case.py --model_path ./exps/checkpoints/LHMPP-Released-v0.1

    # Save to custom output dir (default: debug/app_test)
    python scripts/test/test_app_case.py --output_dir ./my_output
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import imageio.v3 as iio
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

torch._dynamo.config.disable = True

from app import get_motion_video_fps, prior_model_check
from core.utils.model_card import MODEL_CONFIG
from core.utils.model_download_utils import AutoModelQuery
from scripts.download_motion_video import motion_video_check
from scripts.inference.utils import easy_memory_manager

# ==================== Configuration ====================
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_VIDEO_BITRATE = "10M"
MACRO_BLOCK_SIZE = 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Run app inference case directly")
    parser.add_argument(
        "--model_name",
        type=str,
        default="LHMPP-700M-PixelShuffle",
        choices=[
            "LHMPP-700M-PixelShuffle",
            "LHMPP-700M-SMPLX-FREE",
            "LHMPP-700M",
            "LHMPPS-700M",
        ],  # LHMPP-700MC coming soon
        help="Model to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model path (e.g. ./exps/releases_migrated/LHMPP-Released-v0.1)",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="./assets/example_multi_images/00000_yuliang_*.png",
        help="Glob pattern for input images",
    )
    parser.add_argument(
        "--motion_video",
        type=str,
        default="./motion_video/TaiChi/TaiChi.mp4",
        help="Path to motion video (same dir must contain smplx_params/)",
    )
    parser.add_argument(
        "--ref_view",
        type=int,
        default=8,
        help="Number of reference views to use",
    )
    parser.add_argument(
        "--motion_size",
        type=int,
        default=120,
        help="Number of frames to render",
    )
    parser.add_argument(
        "--render_fps",
        type=int,
        default=30,
        help="Output video FPS (fallback if samurai_visualize.mp4 not found)",
    )
    parser.add_argument(
        "--visualized_center",
        action="store_true",
        help="Crop output to subject bounds with 10%% padding",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug/app_test",
        help="Output directory (default: debug/app_test)",
    )
    args = parser.parse_args()

    # Env setup (same as app.py)
    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": args.model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    from core.datasets.data_utils import SrcImagePipeline
    from core.utils.app_utils import get_motion_information, prepare_input_and_output
    from engine.pose_estimation.pose_estimator import PoseEstimator
    from scripts.inference.app_inference import (
        build_app_model,
        inference_results,
        parse_app_configs,
    )

    # Build model_cards (same as app.py): AutoModelQuery or --model_path override
    prior_model_check(save_dir="./pretrained_models")
    motion_video_check(save_dir=".")
    model_config = MODEL_CONFIG[args.model_name]
    if args.model_path:
        model_path = args.model_path
    else:
        auto_query = AutoModelQuery(save_dir="./pretrained_models")
        model_path = auto_query.query(args.model_name)
    model_cards = {
        args.model_name: {
            "model_path": model_path,
            "model_config": model_config,
        }
    }

    print(f"[1/6] Loading config and model...")
    processing_list = [
        dict(
            name="PadRatioWithScale",
            target_ratio=5 / 3,
            tgt_max_size_list=[840],
            val=True,
        ),
    ]
    dataset_pipeline = SrcImagePipeline(*processing_list)

    accelerator = Accelerator()
    cfg, _ = parse_app_configs(model_cards)

    lhmpp = build_app_model(cfg)
    lhmpp.to("cuda")
    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cpu"
        )
        pose_estimator.device = "cuda"

    # Load images
    print(f"[2/6] Loading images from {args.image_glob}...")
    image_paths = sorted(glob.glob(args.image_glob))[:8]
    if not image_paths:
        raise FileNotFoundError(f"No images found for glob: {args.image_glob}")
    imgs_pil = [Image.open(p) for p in image_paths]
    # Format expected by obtain_ref_imgs: list of (img,) or gallery format
    image_for_prepare = [(np.asarray(img),) for img in imgs_pil]

    # Motion video
    if not os.path.isfile(args.motion_video):
        raise FileNotFoundError(f"Motion video not found: {args.motion_video}")
    print(f"[3/6] Using motion: {args.motion_video}")

    # Working dir: always use persistent output_dir (default: debug/app_test)
    output_dir = args.output_dir or "debug/app_test"
    os.makedirs(output_dir, exist_ok=True)

    class NamedDir:
        pass

    working_dir = NamedDir()
    working_dir.name = os.path.abspath(output_dir)

    print(f"[4/6] Preparing input and motion...")
    imgs, _, motion_path, _, dump_video_path = prepare_input_and_output(
        image=image_for_prepare,
        video=None,
        ref_view=args.ref_view,
        video_params=args.motion_video,
        working_dir=working_dir,
        dataset_pipeline=dataset_pipeline,
        cfg=cfg,
    )

    motion_name, motion_seqs = get_motion_information(
        motion_path, cfg, motion_size=args.motion_size
    )
    video_size = len(motion_seqs["motion_seqs"])
    print(f"  Motion: {motion_name}, frames: {video_size}")

    print(f"[5/6] Running inference...")
    device = "cuda"
    dtype = torch.float32
    if pose_estimator is not None:
        with torch.no_grad():
            with easy_memory_manager(pose_estimator, device="cuda"):
                shape_pose = pose_estimator(imgs[0])
        assert shape_pose.is_full_body, f"Input image invalid: {shape_pose.msg}"

    img_np = np.stack(imgs) / 255.0
    ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
    smplx_params = motion_seqs["smplx_params"]
    if pose_estimator is not None:
        smplx_params["betas"] = torch.tensor(
            shape_pose.beta, dtype=dtype, device=device
        ).unsqueeze(0)
    else:
        smplx_params = smplx_params.copy()

    rgbs = inference_results(
        lhmpp,
        ref_imgs_tensor,
        smplx_params,
        motion_seqs,
        video_size=video_size,
        visualized_center=args.visualized_center,
        device=device,
    )

    # Get FPS (same as app.py: prefer samurai_visualize.mp4 in motion dir)
    motion_dir = os.path.dirname(args.motion_video)
    samurai_path = os.path.join(motion_dir, "samurai_visualize.mp4")
    video_for_fps = samurai_path if os.path.isfile(samurai_path) else args.motion_video
    render_fps = get_motion_video_fps(video_for_fps, default=args.render_fps)

    print(f"[6/6] Saving video ({render_fps} fps) to {dump_video_path}...")
    iio.imwrite(
        dump_video_path,
        rgbs,
        fps=render_fps,
        codec=DEFAULT_VIDEO_CODEC,
        pixelformat=DEFAULT_PIXEL_FORMAT,
        bitrate=DEFAULT_VIDEO_BITRATE,
        macro_block_size=MACRO_BLOCK_SIZE,
    )
    print(f"Done. Video saved to: {dump_video_path}")

    # Output is kept in output_dir for inspection (no cleanup)


if __name__ == "__main__":
    main()
