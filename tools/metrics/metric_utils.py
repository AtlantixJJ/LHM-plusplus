#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared photometric metric helpers (PSNR / SSIM / LPIPS with GT mask).

Used by DNA MV, synthetic MV, and dynamic animation benchmark scripts. Each benchmark
script owns its own directory layout and aggregation; this module only provides I/O,
masked metrics, and small filesystem helpers.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

FRAME_DIR_RE = re.compile(r"^\d{5}$")

PSNR_DEFINITION = (
    "MSE(mean) over (pred*mask, gt*mask) in [0,1], PSNR=10 log10(1/MSE)"
)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(h)


def load_rgb(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return arr


def load_mask_float(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    if arr.ndim != 2:
        arr = arr[..., 0]
    return np.clip(arr, 0.0, 1.0)


def masked_psnr_mse(pred: np.ndarray, gt: np.ndarray, mask_hw: np.ndarray) -> float:
    """Scalar PSNR for ``pred*mask`` vs ``gt*mask`` (global MSE over H×W×C)."""
    m = mask_hw[..., None]
    pm = pred * m
    gm = gt * m
    err = np.mean((pm - gm) ** 2)
    if err <= 1e-20:
        return 99.0
    return float(10.0 * np.log10(1.0 / err))


def masked_ssim(pred: np.ndarray, gt: np.ndarray, mask_hw: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    m = mask_hw[..., None]
    pm = np.clip(pred * m, 0.0, 1.0)
    gm = np.clip(gt * m, 0.0, 1.0)
    try:
        return float(
            structural_similarity(gm, pm, data_range=1.0, channel_axis=-1)
        )
    except TypeError:
        return float(
            structural_similarity(gm, pm, data_range=1.0, multichannel=True)
        )


def batched_lpips_masked(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    masks: List[np.ndarray],
    lpips_fn: torch.nn.Module,
    device: torch.device,
) -> List[float]:
    if not preds:
        return []
    batch = []
    for pr, gt, mk in zip(preds, gts, masks):
        m = mk[..., None]
        pm = np.clip(pr * m, 0.0, 1.0)
        gm = np.clip(gt * m, 0.0, 1.0)
        x = torch.from_numpy(pm).float().permute(2, 0, 1)
        y = torch.from_numpy(gm).float().permute(2, 0, 1)
        batch.append((x, y))
    xs = torch.stack([b[0] for b in batch], dim=0).to(device) * 2.0 - 1.0
    ys = torch.stack([b[1] for b in batch], dim=0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_fn(xs, ys)
    return [float(d[i].item()) for i in range(d.shape[0])]


def metrics_triple(
    pred: np.ndarray,
    gt: np.ndarray,
    mask_hw: np.ndarray,
    lpips_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    psnr = masked_psnr_mse(pred, gt, mask_hw)
    ssim = masked_ssim(pred, gt, mask_hw)
    lp = batched_lpips_masked([pred], [gt], [mask_hw], lpips_fn, device)[0]
    return psnr, ssim, lp


def branch_record(psnr: float, ssim: float, lpips: float) -> Dict[str, float]:
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


def tight_bbox_ltrb(
    mask_hw: np.ndarray, thresh: float = 0.5, pad: int = 0
) -> Optional[Tuple[int, int, int, int]]:
    """Inclusive (left, top, right, bottom) pixel indices; None if empty."""
    ys, xs = np.where(mask_hw > thresh)
    if len(xs) == 0:
        return None
    l, r = int(xs.min()), int(xs.max())
    t, b = int(ys.min()), int(ys.max())
    h, w = mask_hw.shape[:2]
    if pad:
        l = max(0, l - pad)
        t = max(0, t - pad)
        r = min(w - 1, r + pad)
        b = min(h - 1, b + pad)
    return l, t, r, b


def crop_array(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    l, t, r, b = bbox
    return arr[t : b + 1, l : r + 1, ...]


def list_scene_dirs(root: str) -> List[str]:
    names = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and not name.startswith("."):
            names.append(name)
    return names


def list_frame_dirs(scene_path: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(scene_path)):
        if not FRAME_DIR_RE.match(name):
            continue
        p = os.path.join(scene_path, name)
        if os.path.isdir(p):
            out.append(name)
    return out


def paired_stems(pred_dir: str, gt_dir: str, mask_dir: str) -> List[str]:
    if not os.path.isdir(pred_dir):
        return []
    stems = []
    for f in sorted(os.listdir(pred_dir)):
        if not f.lower().endswith(".png"):
            continue
        gp = os.path.join(gt_dir, f)
        mp = os.path.join(mask_dir, f)
        if os.path.isfile(gp) and os.path.isfile(mp):
            stems.append(f)
    return stems
