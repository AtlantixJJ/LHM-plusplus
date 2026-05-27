#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Function      : Export Gaussian Splatting as PLY (T-pose or SMPL-X frame; CLI + loaders)

"""
Export canonical (T-pose) Gaussian Splatting as PLY.

This script mirrors the model / image / SMPL-X shape loading path used by
``scripts/test/test_app_case.py`` and ``scripts/inference/app_inference.py``,
then runs ``infer_single_view`` → ``inference_gs`` → ``GaussianModel.save_ply``.

Supported models must appear in ``core.utils.model_card.GS_RENDER_SUPPORTED_MODEL_NAMES``
(models with Gaussian/3DGS output). Using any other ``--model_name`` is rejected before load.

Usage:
    # 仅图像：合成相机 + 规范 T-pose（``--pose_dir`` 默认为空）
    python scripts/inference/to_gs_ply.py \\
        --model_name LHMPP-700M-PixelShuffle \\
        --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle \\
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png"

    # 指定某一帧 SMPL-X JSON：用该文件内相机与姿态做 animation 导出（不经视频/mask 管线）
    python scripts/inference/to_gs_ply.py \\
        --pose_dir "./motion_video/BasketBall_I/smplx_params/00014.json" \\
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

_LHM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LHM_ROOT)

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

torch._dynamo.config.disable = True


def _load_module(unique_name: str, rel_path: str) -> Any:
    """Load a project module by path.

    Avoids ``import scripts...`` because a PyPI distribution named ``scripts`` can
    shadow the project's ``scripts`` package.
    """
    path = os.path.join(_LHM_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ai = _load_module("_lhmpp_app_inference", "scripts/inference/app_inference.py")
build_app_model = _ai.build_app_model
parse_app_configs = _ai.parse_app_configs

from core.utils.model_card import (
    GS_RENDER_SUPPORTED_MODEL_NAMES,
    MODEL_CONFIG,
    model_supports_gs_render,
)
from core.utils.model_download_utils import AutoModelQuery


def _require_gs_output_model(model_name: str) -> None:
    """Fail fast if checkpoint is not wired for Gaussian export (infer_single_view + inference_gs)."""
    if model_supports_gs_render(model_name):
        return
    allowed = ", ".join(GS_RENDER_SUPPORTED_MODEL_NAMES) or "(none configured)"
    raise ValueError(
        f"T-pose Gaussian PLY export requires a GS-output model (--model_name in "
        f"GS_RENDER_SUPPORTED_MODEL_NAMES). Got {model_name!r}; supported: {allowed}. "
        f"Extend the allowlist in core/utils/model_card.py when a new GS build is validated."
    )


def _parse_smplx_raw(smplx_raw_data: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Match ``core.runners.infer.utils._parse_smplx_param`` without importing the runners package."""
    return {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }


def _gs_model_name_choices() -> list[str]:
    if not GS_RENDER_SUPPORTED_MODEL_NAMES:
        raise RuntimeError(
            "GS_RENDER_SUPPORTED_MODEL_NAMES is empty in core/utils/model_card.py; "
            "add at least one model with 3DGS output."
        )
    return list(GS_RENDER_SUPPORTED_MODEL_NAMES)


def _effective_pose_dir(args: argparse.Namespace) -> str | None:
    """Unset, empty string, or whitespace ``--pose_dir`` => no pose file (T-pose / synthetic camera)."""
    v = getattr(args, "pose_dir", None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _derive_pose_seq_folder_name(pose_json_path: str) -> str:
    """``.../BasketBall_I/smplx_params/00014.json`` -> ``BasketBall_I``."""
    p = Path(pose_json_path).resolve()
    if p.parent.name == "smplx_params":
        return _sanitize_export_folder_name(p.parent.parent.name)
    return _sanitize_export_folder_name(p.parent.name)


def _sanitize_export_folder_name(name: str) -> str:
    """Single path segment for `{folder_name}.ply` (no slashes, trimmed)."""
    name = (name or "").strip()
    name = os.path.basename(name.replace("\\", "/"))
    for c in '/:*?"<>|':
        name = name.replace(c, "_")
    name = name.strip("._") or "tpose_export"
    return name[:200]


def _derive_input_image_folder_name(args: argparse.Namespace) -> str:
    """Subfolder / parent-dir name for reference images (no motion in path)."""
    if args.images_dir is not None:
        return _sanitize_export_folder_name(
            os.path.basename(os.path.normpath(args.images_dir))
        )
    paths = _resolve_image_paths(args)
    if paths:
        parent = os.path.dirname(os.path.abspath(paths[0]))
        base = os.path.basename(parent)
        if base and base not in (".", ""):
            return _sanitize_export_folder_name(base)
        return _sanitize_export_folder_name(
            os.path.splitext(os.path.basename(paths[0]))[0]
        )
    return "ref_images"


def _derive_folder_name_for_tpose_output(args: argparse.Namespace) -> str:
    """Filename stem for image-only export under ``outputs/tpose_output/``."""
    return _derive_input_image_folder_name(args)


def default_export_output_path(args: argparse.Namespace) -> str:
    """T-pose: ``outputs/tpose_output/{ref}.ply``; posed: ``outputs/animation_output/{seq}/{ref}_{frame}.ply``."""
    pose_path = _effective_pose_dir(args)
    if pose_path is None:
        folder = _derive_folder_name_for_tpose_output(args)
        return os.path.join(_LHM_ROOT, "outputs", "tpose_output", f"{folder}.ply")
    pose_path = str(Path(pose_path).expanduser())
    motion_key = _derive_pose_seq_folder_name(pose_path)
    img_key = _derive_input_image_folder_name(args)
    frame_stem = _sanitize_export_folder_name(Path(pose_path).stem)
    fname = f"{img_key}_{frame_stem}.ply"
    return os.path.join(
        _LHM_ROOT,
        "outputs",
        "animation_output",
        motion_key,
        fname,
    )


def prior_model_check(save_dir: str = "./pretrained_models") -> None:
    """Same behavior as ``app.prior_model_check`` without importing Gradio (``app.py``)."""
    human_model_path = os.path.join(save_dir, "human_model_files")
    if os.path.exists(human_model_path):
        return
    if os.path.islink(human_model_path):
        try:
            os.unlink(human_model_path)
            print("Removed broken symlink: human_model_files")
        except OSError as e:
            print(f"Failed to remove broken symlink: {e}")
    print("Prior models not found or invalid. Downloading...")
    auto_query = AutoModelQuery(save_dir=save_dir)
    auto_query.download_all_prior_models()
    print("Prior models ready.")


@contextmanager
def _easy_memory_manager(model: torch.nn.Module, device: str = "cuda"):
    """Same behavior as ``scripts.inference.utils.easy_memory_manager`` without importing Gradio."""
    del device
    if torch.cuda.is_available():
        dev = f"cuda:{torch.cuda.current_device()}"
    else:
        dev = "cpu"
    model.to(dev)
    try:
        yield model
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    gs_names = _gs_model_name_choices()
    parser = argparse.ArgumentParser(
        description="Load LHM++ and export Gaussian Splatting as PLY (T-pose or SMPL-X frame; same path as app / test_app_case)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=gs_names[0],
        choices=gs_names,
        help=(
            "Model card key (configs + HF/MS ids). Restricted to "
            "GS_RENDER_SUPPORTED_MODEL_NAMES in core/utils/model_card.py."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local checkpoint directory; skips AutoModelQuery when set.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="./assets/example_multi_images/00000_yuliang_*.png",
        help="Glob for reference images (ignored if --images_dir is set).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory of images (*.png, *.jpg, ...); overrides --image_glob when set.",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="",
        help=(
            "Path to one SMPL-X parameter JSON (e.g. motion_video/BasketBall_I/smplx_params/00014.json). "
            "Optional matching FLAME file: ../flame_params/<same_name>.json. "
            "Default empty => canonical T-pose export with a synthetic camera (no motion file)."
        ),
    )
    parser.add_argument(
        "--ref_view",
        type=int,
        default=8,
        help="Number of reference views (same as Gradio / test_app_case).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Target PLY path. Default when ``--pose_dir`` is empty: "
            "<repo>/outputs/tpose_output/{ref_images_parent}.ply; "
            "with a pose JSON: <repo>/outputs/animation_output/{seq}/{ref_images_parent}_{json_stem}.ply"
        ),
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for intermediates (default: <output_parent>/debug/tpose_gs_work).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model and tensors.",
    )
    return parser


def _list_images_from_dir(images_dir: str, max_views: int) -> list[str]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(images_dir, pat)))
    paths = sorted(set(paths))[:max_views]
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {images_dir}")
    return paths


def _resolve_image_paths(args: argparse.Namespace) -> list[str]:
    if args.images_dir is not None:
        return _list_images_from_dir(args.images_dir, max_views=8)
    paths = sorted(glob.glob(args.image_glob))[:8]
    if not paths:
        raise FileNotFoundError(f"No images matched glob: {args.image_glob}")
    return paths


def _synthetic_render_hw(cfg) -> tuple[int, int]:
    """Match ``prepare_motion_seqs_eval`` sizing: ``tgt_h = tgt_size * aspect_standard``."""
    render_res = int(cfg.get("render_size", 420))
    aspect_standard = 5.0 / 3.0
    tgt_h = int(round(render_res * aspect_standard))
    tgt_w = render_res
    return tgt_h, tgt_w


def _load_pose_minimal(pose: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as ``core.runners.infer.utils._load_pose`` (no package import)."""
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()
    c2w = torch.eye(4).float()
    return c2w, intrinsic


def _stack_smplx_params_list_minimal(
    smplx_params_list: list[dict[str, torch.Tensor]], shape_param: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Same as ``_stack_smplx_params_list`` in infer utils."""
    from collections import defaultdict

    stacked: dict[str, list] = defaultdict(list)
    for sp in smplx_params_list:
        for k, v in sp.items():
            stacked[k].append(v)
    result = {k: torch.stack(v) for k, v in stacked.items()}
    result["betas"] = shape_param
    return result


def _to_motion_batch_dict_minimal(
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
    bg_colors: torch.Tensor,
    smplx_params: dict[str, torch.Tensor],
    rgbs: list,
    vis_motion: bool,
) -> dict[str, Any]:
    """Same as ``_to_motion_batch_dict`` with ``vis_motion=False`` (no SMPL mesh render)."""
    if vis_motion:
        raise NotImplementedError("vis_motion True requires full infer utils")
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
    rgbs_out = rgbs.unsqueeze(0) if len(rgbs) > 0 else rgbs
    return {
        "render_c2ws": c2ws.unsqueeze(0),
        "render_intrs": intrs.unsqueeze(0),
        "render_bg_colors": bg_colors.unsqueeze(0),
        "smplx_params": smplx_params,
        "rgbs": rgbs_out,
        "vis_motion_render": None,
    }


def _build_synthetic_motion_seq(cfg) -> dict[str, Any]:
    """One-frame motion dict for ``infer_single_view`` (synthetic camera + neutral pose)."""

    tgt_h, tgt_w = _synthetic_render_hw(cfg)
    Wf, Hf = float(tgt_w), float(tgt_h)
    fx = max(Wf, Hf)
    fy = max(Wf, Hf)
    cx = Wf / 2.0
    cy = Hf / 2.0

    zf = torch.float32
    betas = torch.zeros(10, dtype=zf)
    frame: dict[str, torch.Tensor] = {
        "betas": betas,
        "root_pose": torch.zeros(3, dtype=zf),
        "body_pose": torch.zeros(21, 3, dtype=zf),
        "jaw_pose": torch.zeros(3, dtype=zf),
        "leye_pose": torch.zeros(3, dtype=zf),
        "reye_pose": torch.zeros(3, dtype=zf),
        "lhand_pose": torch.zeros(15, 3, dtype=zf),
        "rhand_pose": torch.zeros(15, 3, dtype=zf),
        "trans": torch.zeros(3, dtype=zf),
        "expr": torch.zeros(100, dtype=zf),
        "focal": torch.tensor([fx, fy], dtype=zf),
        "princpt": torch.tensor([cx, cy], dtype=zf),
        "img_size_wh": torch.tensor([Wf, Hf], dtype=zf),
    }
    c2w, intr = _load_pose_minimal(frame)
    stacked = _stack_smplx_params_list_minimal([frame], betas)
    c2ws = c2w.unsqueeze(0)
    intrs = intr.unsqueeze(0)
    bg_colors = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    base = _to_motion_batch_dict_minimal(c2ws, intrs, bg_colors, stacked, [], vis_motion=False)
    base["offset_list"] = [[1.0, 1.0, 0.0, 0.0]]
    base["ori_size"] = (tgt_h, tgt_w)
    base["motion_seqs"] = []
    return base


def _build_motion_seq_from_pose_json(json_path: str, cfg: Any) -> dict[str, Any]:
    """Single-frame ``motion_seq`` dict from one SMPL-X JSON (no video / mask / ``get_motion_information``)."""
    path = str(Path(json_path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--pose_dir must be an existing SMPL-X JSON file: {path}")
    if not path.lower().endswith(".json"):
        raise ValueError(
            f"--pose_dir must point to a .json file (e.g. .../smplx_params/00014.json), got {path!r}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    p = Path(path)
    if p.parent.name == "smplx_params":
        flame_path = p.parent.parent / "flame_params" / p.name
    else:
        flame_path = p.parent / "flame_params" / p.name
    if flame_path.is_file():
        with open(flame_path, "r", encoding="utf-8") as f:
            flame_params = json.load(f)
        raw["expr"] = flame_params["expcode"]
        pc = flame_params["posecode"]
        raw["jaw_pose"] = pc[3:6] if len(pc) >= 6 else [0.0, 0.0, 0.0]
        ec = flame_params["eyecode"]
        raw["leye_pose"] = ec[:3] if len(ec) >= 3 else [0.0, 0.0, 0.0]
        raw["reye_pose"] = ec[3:6] if len(ec) >= 6 else [0.0, 0.0, 0.0]

    if "expr" not in raw:
        raw["expr"] = [0.0] * 100

    smplx_param = _parse_smplx_raw(raw)
    c2w, intr = _load_pose_minimal(smplx_param)
    shape_param = smplx_param["betas"]
    stacked = _stack_smplx_params_list_minimal([smplx_param], shape_param)
    c2ws = c2w.unsqueeze(0)
    intrs = intr.unsqueeze(0)
    bg_colors = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    base = _to_motion_batch_dict_minimal(c2ws, intrs, bg_colors, stacked, [], vis_motion=False)
    if "img_size_wh" in smplx_param:
        wh = smplx_param["img_size_wh"]
        tgt_w, tgt_h = int(wh[0].item()), int(wh[1].item())
    else:
        tgt_h, tgt_w = _synthetic_render_hw(cfg)
    base["offset_list"] = [[1.0, 1.0, 0.0, 0.0]]
    base["ori_size"] = (tgt_h, tgt_w)
    base["motion_seqs"] = []
    return base


def slice_motion_seq_to_single_frame(
    motion_seq: dict[str, Any], frame_idx: int = 0
) -> dict[str, Any]:
    """Keep the first (or given) frame on the time dimension for cameras and SMPL-X."""
    out: dict[str, Any] = dict(motion_seq)
    sl = slice(frame_idx, frame_idx + 1)
    out["render_c2ws"] = motion_seq["render_c2ws"][:, sl]
    out["render_intrs"] = motion_seq["render_intrs"][:, sl]
    out["render_bg_colors"] = motion_seq["render_bg_colors"][:, sl]
    rgbs = motion_seq.get("rgbs")
    if isinstance(rgbs, torch.Tensor) and rgbs.numel() > 0 and rgbs.dim() >= 2:
        out["rgbs"] = rgbs[:, sl]

    smplx_in = motion_seq["smplx_params"]
    smplx: dict[str, torch.Tensor] = {}
    time_varying = {
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "expr",
        "focal",
        "princpt",
        "img_size_wh",
    }
    for k, v in smplx_in.items():
        if k in time_varying and v.dim() >= 2 and v.shape[1] > 1:
            smplx[k] = v[:, sl].clone()
        else:
            smplx[k] = v.clone()

    out["smplx_params"] = smplx
    off = motion_seq.get("offset_list")
    if isinstance(off, list) and len(off) > frame_idx:
        out["offset_list"] = [off[frame_idx]]
    mseq = motion_seq.get("motion_seqs")
    if isinstance(mseq, list) and len(mseq) > frame_idx:
        out["motion_seqs"] = [mseq[frame_idx]]
    msks = motion_seq.get("masks")
    if isinstance(msks, list) and len(msks) > frame_idx:
        out["masks"] = [msks[frame_idx]]
    return out


def cano_body_pose_template(
    device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Same canonical body_pose tweaks as ``BaseGSRender._prepare_smplx_data`` (last frame)."""
    bp = torch.zeros(1, 1, 21, 3, device=device, dtype=dtype)
    # leg
    bp[0, 0, 0, -1] = math.pi / 12
    bp[0, 0, 1, -1] = -math.pi / 12
    # hands
    bp[0, 0, 15, -1] = -math.pi / 6
    bp[0, 0, 16, -1] = math.pi / 6
    return bp


def build_tpose_smplx_params(
    motion_seq_one: dict[str, Any],
    transform_mat_neutral_pose: torch.Tensor,
    merged_betas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """One-frame T-pose SMPL-X dict: canonical pose + merged betas + neutral transform."""
    sp: dict[str, torch.Tensor] = {}
    for k, v in motion_seq_one["smplx_params"].items():
        sp[k] = v.to(device=device, dtype=dtype)

    sp["betas"] = merged_betas.to(device=device, dtype=dtype)
    sp["transform_mat_neutral_pose"] = transform_mat_neutral_pose.to(
        device=device, dtype=dtype
    )

    z31 = torch.zeros(1, 1, 3, device=device, dtype=dtype)
    z_hand = torch.zeros(1, 1, 15, 3, device=device, dtype=dtype)
    n_expr = sp["expr"].shape[-1]
    sp["root_pose"] = z31
    sp["body_pose"] = cano_body_pose_template(device, dtype)
    sp["jaw_pose"] = z31
    sp["leye_pose"] = z31
    sp["reye_pose"] = z31
    sp["lhand_pose"] = z_hand
    sp["rhand_pose"] = z_hand
    sp["trans"] = z31
    sp["expr"] = torch.zeros(1, 1, n_expr, device=device, dtype=dtype)
    return sp


def build_animation_frame_smplx_params(
    motion_seq_one: dict[str, Any],
    transform_mat_neutral_pose: torch.Tensor,
    merged_betas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """First-frame motion SMPL-X for GS export (clip frame 0 pose, not canonical T-pose)."""
    sp: dict[str, torch.Tensor] = {}
    for k, v in motion_seq_one["smplx_params"].items():
        sp[k] = v.to(device=device, dtype=dtype)

    sp["betas"] = merged_betas.to(device=device, dtype=dtype)
    sp["transform_mat_neutral_pose"] = transform_mat_neutral_pose.to(
        device=device, dtype=dtype
    )
    return sp


@torch.no_grad()
def run_tpose_export(
    model: torch.nn.Module,
    ref_imgs_tensor: torch.Tensor,
    motion_seq: dict[str, Any],
    device: str,
    output_ply: str,
    *,
    export_animation_pose: bool = False,
) -> None:
    """``infer_single_view`` → ``inference_gs`` → ``GaussianModel.save_ply``.

    SMPL-X for the forward pass is taken from the first frame of ``motion_seq`` (see
    ``slice_motion_seq_to_single_frame``), so it matches cameras and body pose length.

    If ``export_animation_pose`` is True (non-empty ``--pose_dir``), the final GS matches that
    JSON pose. Otherwise it uses canonical T-pose in SMPL-X angles.
    """
    dev = torch.device(device)
    use_pred_render = getattr(model, "use_pred_shape_for_render", False)
    motion_one = slice_motion_seq_to_single_frame(motion_seq, frame_idx=0)

    render_c2ws = motion_one["render_c2ws"].to(dev)
    render_intrs = motion_one["render_intrs"].to(dev)
    render_bg_colors = motion_one["render_bg_colors"].to(dev)
    smplx_dev = {k: v.to(dev) for k, v in motion_one["smplx_params"].items()}

    ref_batch = ref_imgs_tensor.unsqueeze(0)
    ref_mask = torch.ones(
        ref_imgs_tensor.shape[0], dtype=torch.bool, device=dev
    ).unsqueeze(0)

    model_outputs = model.infer_single_view(
        ref_batch,
        None,
        None,
        render_c2ws=render_c2ws,
        render_intrs=render_intrs,
        render_bg_colors=render_bg_colors,
        smplx_params=smplx_dev,
        ref_imgs_bool=ref_mask,
        return_pred_shape=use_pred_render,
    )

    pred_shape = None
    if len(model_outputs) == 8:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            _image_feats,
            _motion_emb,
            _pos_emb,
            pred_shape,
        ) = model_outputs
    elif len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            _image_feats,
            _motion_emb,
            _pos_emb,
        ) = model_outputs
    else:
        raise RuntimeError(f"Unexpected infer_single_view outputs: {len(model_outputs)}")

    merged = type(model).smplx_params_with_pred_shape_betas(smplx_dev, pred_shape)
    merged_betas = merged["betas"]

    dtype = merged_betas.dtype
    if export_animation_pose:
        gs_smplx = build_animation_frame_smplx_params(
            motion_one,
            transform_mat_neutral_pose,
            merged_betas,
            dev,
            dtype,
        )
    else:
        gs_smplx = build_tpose_smplx_params(
            motion_one,
            transform_mat_neutral_pose,
            merged_betas,
            dev,
            dtype,
        )

    view_idx = 0
    renderer = model.renderer
    if export_animation_pose:
        # ``model.inference_gs`` → ``inference_cano_gs`` only appends ``cano_models`` (last template row
        # from ``_prepare_smplx_data``), which looks like T-pose. Use ``animate_gs_model`` like
        # ``forward_animate_gs`` and take the first posed view (``anim_models[0]``).
        smplx_view = renderer.get_single_view_smpl_data(gs_smplx, view_idx)
        smplx_one = renderer._get_single_batch_data(smplx_view, 0)
        anim_models, _ = renderer.animate_gs_model(
            gs_model_list[0],
            query_points["neutral_coords"][0],
            smplx_one,
            debug=False,
            mesh_meta=query_points["mesh_meta"],
        )
        if not anim_models:
            cano_gs = model.inference_gs(
                gs_model_list,
                query_points,
                gs_smplx,
                render_c2ws,
                render_intrs,
                render_bg_colors,
                gs_hidden_features,
                pad_forward=False,
            )
        else:
            cano_gs = anim_models[0]
    else:
        cano_gs = model.inference_gs(
            gs_model_list,
            query_points,
            gs_smplx,
            render_c2ws,
            render_intrs,
            render_bg_colors,
            gs_hidden_features,
            pad_forward=False,
        )
    out_abs = os.path.abspath(output_ply)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cano_gs.save_ply(out_abs)
    if export_animation_pose:
        print(f"Saved SMPL-X pose Gaussian PLY to {out_abs}")
    else:
        print(f"Saved T-pose canonical Gaussian PLY to {out_abs}")


def setup_loaders_and_inputs(args: argparse.Namespace):
    """Load cfg, model, reference images, motion tensors, and optional shape-from-image betas.

    If ``--pose_dir`` is empty, uses a single-frame synthetic camera and neutral SMPL-X pose.
    If ``--pose_dir`` points to one SMPL-X JSON, loads that frame only (no video / mask pipeline).
    """
    _require_gs_output_model(args.model_name)
    from core.utils.app_utils import obtain_ref_imgs

    from engine.pose_estimation.pose_estimator import PoseEstimator

    device = args.device
    dtype = torch.float32

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

    _ = Accelerator()
    cfg, _ = parse_app_configs(model_cards)

    model = build_app_model(cfg)
    model.to(device)

    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        pose_estimator = PoseEstimator("./pretrained_models/human_model_files/", device="cpu")
        pose_estimator.device = device

    image_paths = _resolve_image_paths(args)
    imgs_pil = [Image.open(p) for p in image_paths]
    image_for_prepare = [(np.asarray(img),) for img in imgs_pil]

    out_parent = os.path.dirname(os.path.abspath(args.output))
    if not out_parent:
        out_parent = os.getcwd()
    work_dir_path = args.work_dir or os.path.join(out_parent, "debug", "tpose_gs_work")
    os.makedirs(work_dir_path, exist_ok=True)
    working_dir = SimpleNamespace()
    working_dir.name = os.path.abspath(work_dir_path)

    imgs = obtain_ref_imgs(image_for_prepare, ref_view=args.ref_view)
    sample_imgs = np.concatenate(imgs, axis=1)
    save_sample_imgs = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(sample_imgs) as img:
        img.save(save_sample_imgs)

    pose_json = _effective_pose_dir(args)
    if pose_json is not None:
        motion_seqs = _build_motion_seq_from_pose_json(pose_json, cfg)
    else:
        motion_seqs = _build_synthetic_motion_seq(cfg)

    if pose_estimator is not None:
        with torch.no_grad():
            with _easy_memory_manager(pose_estimator, device=device):
                shape_pose = pose_estimator(imgs[0])
        if not shape_pose.is_full_body:
            raise ValueError(f"Input image invalid for shape estimator: {shape_pose.msg}")

    img_np = np.stack(imgs) / 255.0
    ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
    smplx_params = motion_seqs["smplx_params"].copy()
    if pose_estimator is not None:
        smplx_params["betas"] = torch.tensor(shape_pose.beta, dtype=dtype, device=device).unsqueeze(0)
    motion_seqs["smplx_params"] = smplx_params

    return model, cfg, ref_imgs_tensor, smplx_params, motion_seqs, pose_estimator, device


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None:
        args.output = default_export_output_path(args)

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": args.model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    (
        model,
        cfg,
        ref_imgs_tensor,
        smplx_params,
        motion_seqs,
        pose_estimator,
        device,
    ) = setup_loaders_and_inputs(args)

    run_tpose_export(
        model,
        ref_imgs_tensor,
        motion_seqs,
        device=device,
        output_ply=os.path.abspath(args.output),
        export_animation_pose=_effective_pose_dir(args) is not None,
    )


if __name__ == "__main__":
    main()
