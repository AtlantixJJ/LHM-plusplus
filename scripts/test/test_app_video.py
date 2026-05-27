#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Test app video input inference without Gradio UI

"""
Test app.py inference with video input (no Gradio UI).

Extracts reference frames from input video and runs the same inference
logic as the Gradio "Generate" button. Auto-downloads prior models and
LHM++ weights if not present.

Usage:
    python scripts/test/test_app_video.py
    # Default input_video: ./assets/example_videos/yuliang.mp4

    python scripts/test/test_app_video.py --input_video ./assets/example_videos/woman.mp4

    # Override model path (skip AutoModelQuery)
    python scripts/test/test_app_video.py --input_video ./video.mp4 --model_path ./exps/checkpoints/LHMPP-Released-v0.1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import imageio.v3 as iio
import numpy as np
import torch
from accelerate import Accelerator

torch._dynamo.config.disable = True

from app import get_motion_video_fps, prior_model_check
from scripts.download_motion_video import motion_video_check
from core.utils.model_card import MODEL_CONFIG
from core.utils.model_download_utils import AutoModelQuery

DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_VIDEO_BITRATE = "10M"
MACRO_BLOCK_SIZE = 16


def _resolve_video_path(video) -> str:
    """Resolve video path from Gradio Video component value (str or dict)."""
    if isinstance(video, dict):
        return video.get("path") or video.get("name") or ""
    return video if isinstance(video, str) else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test app.py inference with video input (no Gradio UI)"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="./assets/example_videos/yuliang.mp4",
        help="Path to input human video (source for reference frames)",
    )
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
        help="Override model path (skip AutoModelQuery, use local checkpoint)",
    )
    parser.add_argument(
        "--motion_video",
        type=str,
        default="./motion_video/BasketBall_II/BasketBall_II.mp4",
        help="Path to motion video. Default: BasketBall_II",
    )
    parser.add_argument(
        "--ref_view",
        type=int,
        default=8,
        help="Number of reference frames to extract from input video",
    )
    parser.add_argument(
        "--motion_size",
        type=int,
        default=120,
        help="Number of motion frames to render",
    )
    parser.add_argument(
        "--render_fps",
        type=int,
        default=30,
        help="Output video FPS (fallback)",
    )
    parser.add_argument(
        "--visualized_center",
        action="store_true",
        help="Center crop output with padding",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug/video_test",
        help="Output directory",
    )
    args = parser.parse_args()

    input_video = _resolve_video_path(args.input_video)
    if not input_video or not os.path.isfile(input_video):
        raise FileNotFoundError(
            f"Input video not found: {args.input_video}. "
            "Provide a video containing a full-body human."
        )

    motion_video_check(save_dir=".")

    # Default motion video
    motion_video = args.motion_video
    if not motion_video or not os.path.isfile(motion_video):
        motion_dir = "./motion_video"
        if os.path.isdir(motion_dir):
            candidates = []
            for name in sorted(os.listdir(motion_dir)):
                subdir = os.path.join(motion_dir, name)
                if not os.path.isdir(subdir):
                    continue
                mp4 = os.path.join(subdir, name + ".mp4")
                if os.path.isfile(mp4):
                    candidates.append(mp4)
            if candidates:
                motion_video = candidates[0]
                print(f"[*] Using default motion video: {motion_video}")
        if not motion_video or not os.path.isfile(motion_video):
            raise FileNotFoundError(
                f"Motion video not found: {args.motion_video}. "
                "Specify --motion_video or ensure motion_video/<name>/<name>.mp4 exists."
            )

    # Env (same as app.py)
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
    from scripts.inference.utils import easy_memory_manager

    # Build model_cards (same as app.py): AutoModelQuery or --model_path override
    prior_model_check(save_dir="./pretrained_models")
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

    print("[1/5] Loading model...")
    lhmpp = build_app_model(cfg)
    lhmpp.to("cuda")
    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cpu"
        )
        pose_estimator.device = "cuda"

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    class WorkingDir:
        name = output_dir

    working_dir = WorkingDir()

    print(f"[2/5] Extracting reference frames from input video: {input_video}")
    imgs, save_sample_imgs, motion_path, dump_image_dir, dump_video_path = (
        prepare_input_and_output(
            image=None,
            video=input_video,
            ref_view=args.ref_view,
            video_params=motion_video,
            working_dir=working_dir,
            dataset_pipeline=dataset_pipeline,
            cfg=cfg,
        )
    )
    print(f"  Extracted {len(imgs)} reference frames, saved to {save_sample_imgs}")

    print("[3/5] Loading motion sequences...")
    motion_name, motion_seqs = get_motion_information(
        motion_path, cfg, motion_size=args.motion_size
    )
    video_size = len(motion_seqs["motion_seqs"])
    print(f"  Motion: {motion_name}, frames: {video_size}")

    print("[4/5] Running inference...")
    device = "cuda"
    dtype = torch.float32
    if pose_estimator is not None:
        with torch.no_grad():
            with easy_memory_manager(pose_estimator, device="cuda"):
                shape_pose = pose_estimator(imgs[0])
        if not shape_pose.is_full_body:
            raise ValueError(f"Input video invalid: {shape_pose.msg}")
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
    motion_dir = os.path.dirname(motion_video)
    samurai_path = os.path.join(motion_dir, "samurai_visualize.mp4")
    video_for_fps = samurai_path if os.path.isfile(samurai_path) else motion_video
    render_fps = get_motion_video_fps(video_for_fps, default=args.render_fps)

    print(f"[5/5] Saving video ({render_fps} fps) to {dump_video_path}...")
    iio.imwrite(
        dump_video_path,
        rgbs,
        fps=render_fps,
        codec=DEFAULT_VIDEO_CODEC,
        pixelformat=DEFAULT_PIXEL_FORMAT,
        bitrate=DEFAULT_VIDEO_BITRATE,
        macro_block_size=MACRO_BLOCK_SIZE,
    )
    print(f"Done. Output: {dump_video_path}")


if __name__ == "__main__":
    main()
