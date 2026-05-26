# -*- coding: utf-8 -*-
"""Helpers for LHMPP dynamic benchmark masked motion inference + unpad save layout.

Adapted from ``scripts/inference/infer_eval_video_benchmark.py`` (motion bundles + ``infer_single_view`` cache).

Upstream references: ``infer_eval_video_benchmark.obtain_motion_sequence``,
``prepare_motion_seqs_benchmark_animation_native`` (dynamic animation; no mask pad-before-render),
``infer_eval_video_benchmark.inference_results`` (mask + board path).
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.datasets.evaluation import DYNAMIC_BENCHMARK_REF_SUBDIR
from core.runners.infer.utils import (
    _load_pose,
    center_crop,
    render_smplx_mesh,
    scale_intrs,
)

_LOG = logging.getLogger(__name__)
_LEGACY_REF_SUBDIR = "crop_sr_src_imgs"
_REF_GT_RESIZE_ONLY_SUBDIRS = frozenset(
    {DYNAMIC_BENCHMARK_REF_SUBDIR, _LEGACY_REF_SUBDIR, "sr_imgs_png"}
)


def _cfg_ref_imgs_subdir(cfg: Any, default: str = DYNAMIC_BENCHMARK_REF_SUBDIR) -> str:
    for key in ("motion_ref_imgs_subdir", "motion_sr_inputs_subdir"):
        raw = _cfg_get(cfg, key, None)
        if raw is not None:
            sub = str(raw).strip()
            if sub:
                return sub
    return default


def _cfg_ref_source_height(cfg: Any, default: int) -> int:
    for key in ("motion_ref_source_tgt_max_size", "motion_sr_inputs_source_tgt_max_size"):
        raw = _cfg_get(cfg, key, None)
        if raw is not None:
            return int(raw)
    return int(default)


def obtain_motion_sequence_from_paths(motion_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load SMPL-X (+ optional FLAME) JSON files in path order."""
    import json

    smplx_list: List[Dict[str, Any]] = []
    for motion in motion_paths:
        with open(motion) as reader:
            smplx_params = json.load(reader)
        flame_path = motion.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as reader:
                flame_params = json.load(reader)
            smplx_params["expr"] = torch.FloatTensor(flame_params["expcode"])
            smplx_params["jaw_pose"] = torch.FloatTensor(flame_params["posecode"][3:])
            smplx_params["leye_pose"] = torch.FloatTensor(flame_params["eyecode"][:3])
            smplx_params["reye_pose"] = torch.FloatTensor(flame_params["eyecode"][3:])
        else:
            smplx_params["expr"] = torch.FloatTensor([0.0] * 100)
        smplx_list.append(smplx_params)
    return smplx_list





# ViT-{S,B,L}/14 and Dinov2Wrapper ``downsample_ratio`` use patch size 14.
_SR_REF_ENCODER_PATCH_DEFAULT = 14

def patch_processing_pipeline_ref_resolution(
    ds_kw: MutableMapping[str, Any],
    *,
    tgt_height: int,
    encoder_patch: int = _SR_REF_ENCODER_PATCH_DEFAULT,
) -> Tuple[int, int]:
    """Set ``PadRatio*`` canvas height so ``source_rgbs`` match ``--src-height`` (default 1036).

    :class:`~openlrm.datasets.data_utils.PadRatioWithScale` already rounds **width** to multiples of 14.
    **Height** is taken verbatim from ``tgt_max_size_list`` — it must be divisible by ``encoder_patch``
    (DINO ViT /14) so the encoder does not silently shrink inside ``_preprocess_image``.

    Returns ``(height, width)`` of the padded canvas after patching (same rule as ``PadRatioWithScale``).
    """
    ep = int(encoder_patch)
    if ep <= 0:
        raise ValueError(f"encoder_patch must be positive, got {encoder_patch}")
    th = int(tgt_height)
    if th % ep != 0:
        raise ValueError(
            f"reference canvas height {th} must be divisible by DINO patch size {ep} "
            f"(e.g. 1036 = 74×14, 840 = 60×14)."
        )
    pipe = ds_kw.get("processing_pipeline")
    if not pipe:
        raise ValueError("ds_kw missing processing_pipeline; cannot patch reference resolution.")
    patched = False
    ratio = 5.0 / 3.0
    for step in pipe:
        name = step.get("name")
        if name not in ("PadRatioWithScale", "PadRatio"):
            continue
        raw_r = step.get("target_ratio")
        if raw_r is not None:
            ratio = float(eval(raw_r)) if isinstance(raw_r, str) else float(raw_r)
        step["tgt_max_size_list"] = [th]
        patched = True
        break
    if not patched:
        raise ValueError(
            "processing_pipeline has no PadRatioWithScale/PadRatio step; "
            "cannot set reference tensor resolution."
        )
    tw = int(th / ratio) // ep * ep
    mul = int(ds_kw.get("multiply") or ep)
    if th % mul != 0:
        _LOG.warning(
            "Reference height %s is not divisible by dataset.multiply=%s — motion/intrinsics helpers "
            "may expect multiples of multiply.",
            th,
            mul,
        )
    return th, tw


class BenchmarkAnimationGSCache(NamedTuple):
    """One ``infer_single_view`` reconstruction (refs + anchor motion) reused for all timeline frames.

    ``betas`` is set only when ``use_pred_shape`` is True (predicted body shape repeated every
    animation frame). When False, animation uses **motion JSON** ``betas`` from ``motion_seqs``.
    """

    gs_model_list: Any
    query_points: Any
    transform_mat_neutral_pose: torch.Tensor
    gs_hidden_features: torch.Tensor
    image_latents: Any
    motion_emb: torch.Tensor
    pos_emb: Any
    betas: Optional[torch.Tensor]


def _smpl_motion_dict_for_infer(
    item: Mapping[str, Any],
    motion_seqs: Mapping[str, Any],
    *,
    betas_override: Optional[torch.Tensor] = None,
    merge_item_camera: bool = True,
    betas_from_motion_json: bool = False,
) -> Dict[str, Any]:
    """Merge dataset ``item`` + benchmark ``motion_seqs`` SMPL-X dict.

    For masked animation infer, ``motion_seqs`` carries ``render_intrs`` / SMPL tensors aligned with the
    native mask pipeline; set ``merge_item_camera=False`` so dataset ``focal`` / ``princpt`` /
    ``img_size_wh`` do not override JSON motion-pack intrinsics.

    Body ``betas`` use ``betas_override`` when given; otherwise keep ``motion_seqs["smplx_params"]["betas"]``
    (animation JSON shape) unchanged.

    ``betas_from_motion_json`` is **deprecated** and ignored (kept so older scripts that still pass this
    keyword do not raise ``TypeError``)."""

    _ = betas_from_motion_json  # legacy kwarg; default is motion JSON betas unless override.
    smpl_motion: Dict[str, Any] = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in motion_seqs["smplx_params"].items()
    }
    if betas_override is not None:
        smpl_motion["betas"] = (
            betas_override.clone() if torch.is_tensor(betas_override) else betas_override
        )

    if not merge_item_camera:
        return smpl_motion

    for cam_key in ("focal", "princpt", "img_size_wh"):
        if cam_key in smpl_motion or cam_key not in item:
            continue
        v = item[cam_key]
        if not torch.is_tensor(v):
            continue
        if v.dim() == 1:
            smpl_motion[cam_key] = v.unsqueeze(0).unsqueeze(0)
        else:
            smpl_motion[cam_key] = v.unsqueeze(0)
    return smpl_motion


def _collapse_betas_for_infer_single_view(smplx_params: Mapping[str, Any]) -> Dict[str, Any]:
    """``infer_single_view`` expects ``betas`` as ``[B, D]``; after merge, betas may be ``[B, T, D]``."""
    out = dict(smplx_params)
    b = out.get("betas")
    if torch.is_tensor(b) and b.dim() == 3:
        out["betas"] = b[:, 0, :].contiguous()
    return out


def _mask_to_hw(mask: Any) -> np.ndarray:
    """Collapse pred/GS masks to 2D (H,W). Handles HW, HWC, CHW and singleton dims."""
    x = np.asarray(mask)
    while x.ndim > 2:
        if x.shape[-1] == 1:
            x = x[..., 0]
            continue
        if x.shape[0] == 1:
            x = x[0]
            continue
        if x.ndim != 3:
            break
        c0, c1, c2 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if c2 <= 8 and c2 <= c1 and c2 <= c0:
            x = x[..., 0]
            continue
        if c0 <= 8 and c0 <= c1 and c0 <= c2:
            x = x[0]
            continue
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Could not reduce mask to 2D: shape={np.asarray(mask).shape}")
    return x


def save_scene_reference_strip_png(
    output_root: str,
    uid: str,
    source_rgbs: torch.Tensor,
    ref_imgs_bool: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """Concatenate validation reference views into one row image ``{uid}/ref_img.png`` (once per run).

    Uses ``item['source_rgbs']`` (``[P, 3, H, W]``) from the benchmark sampler; skips slots where
    ``ref_imgs_bool`` is falsy (padding to ``ref_img_size``). Rows are resized to a common height
    before a horizontal concatenate.
    """

    sr = source_rgbs.detach().cpu().float().clamp(0, 1)
    if sr.dim() != 4 or int(sr.shape[1]) != 3:
        raise ValueError(f"source_rgbs expected [N,3,H,W], got {tuple(sr.shape)}")
    n_v = int(sr.shape[0])

    patches_u8: List[np.ndarray] = []
    rb = (
        ref_imgs_bool.detach().cpu().reshape(-1)
        if ref_imgs_bool is not None and torch.is_tensor(ref_imgs_bool)
        else None
    )

    for i in range(n_v):
        if rb is not None and i < rb.numel() and float(rb[i].item()) == 0.0:
            continue
        if rb is None and float(sr[i].abs().sum()) < 1e-8:
            continue
        hwc_i = (
            np.transpose(sr[i].numpy(), (1, 2, 0)) * 255.0
        ).clip(0, 255).round().astype(np.uint8)
        patches_u8.append(hwc_i)

    if not patches_u8:
        return None

    target_h = max(p.shape[0] for p in patches_u8)
    row_parts: List[np.ndarray] = []
    for pic in patches_u8:
        if pic.shape[0] == target_h:
            row_parts.append(pic)
            continue
        scale = float(target_h) / float(pic.shape[0])
        new_w = max(1, int(round(pic.shape[1] * scale)))
        row_parts.append(
            cv2.resize(pic, (new_w, target_h), interpolation=cv2.INTER_AREA),
        )

    strip_u8 = np.concatenate(row_parts, axis=1)
    uid_dir = os.path.join(output_root, uid)
    os.makedirs(uid_dir, exist_ok=True)
    out_path = os.path.join(uid_dir, "ref_img.png")
    Image.fromarray(strip_u8).save(out_path)
    return out_path


@torch.no_grad()
def _infer_lhm_animation_forward_cache(
    lhm: torch.nn.Module,
    batch: Dict[str, Any],
    smplx_params: Mapping[str, Any],
    motion_seq: Mapping[str, Any],
    *,
    ref_imgs_bool: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Tuple[Any, ...]:
    """One ``infer_single_view`` with ``return_pred_shape`` for downstream ``animation_infer``."""
    smpl_in = _collapse_betas_for_infer_single_view(smplx_params)
    return lhm.infer_single_view(
        batch["source_rgbs"].unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params={k: v.to(device) for k, v in smpl_in.items()},
        ref_imgs_bool=ref_imgs_bool,
        return_pred_shape=True,
    )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default)
        return default if v is None else v
    except Exception:
        return getattr(cfg, key, default)


def _pad_rgb_hwc_u8_to_aspect_standard(rgb_hwc: np.ndarray, aspect_standard: float) -> np.ndarray:
    """Letterbox uint8 RGB ``H×W×3`` so ``H/W ≈ aspect_standard`` (dataset height/width convention)."""
    x = np.asarray(rgb_hwc)
    if x.ndim != 3 or int(x.shape[2]) != 3:
        raise ValueError(f"expected HWC RGB, got shape={getattr(x, 'shape', None)}")
    h, w = int(x.shape[0]), int(x.shape[1])
    if h <= 0 or w <= 0:
        return x
    tgt = float(aspect_standard)
    cur = float(h) / float(max(w, 1))
    if abs(cur - tgt) < 1e-3:
        return x
    border_val = (255, 255, 255)
    if cur < tgt:
        new_h = int(round(float(w) * tgt))
        pad_total = max(0, new_h - h)
        pt, pb = pad_total // 2, pad_total - pad_total // 2
        return cv2.copyMakeBorder(x, pt, pb, 0, 0, cv2.BORDER_CONSTANT, value=border_val)
    new_w = int(round(float(h) / tgt))
    pad_total = max(0, new_w - w)
    pl, pr = pad_total // 2, pad_total - pad_total // 2
    return cv2.copyMakeBorder(x, 0, 0, pl, pr, cv2.BORDER_CONSTANT, value=border_val)


def patch_benchmark_item_ref_pad_resize(
    item: Dict[str, Any],
    *,
    motion_scene_root: str,
    ref_subdir: str,
    aspect_standard: float,
) -> None:
    """Replace ``source_rgbs`` with disk reference crops (**inference only**).

    Paths are ``{motion_scene_root}/{ref_subdir}/{frame}.png`` (stem variants via
    :func:`_resolve_gt_image_path_with_variants`), using ``item["ref_camera_ids"]`` — no extra path list.
    Pipeline: read RGB → pad to ``aspect_standard`` (letterbox, white) → resize to tensor ``H×W``.
    """
    src = item.get("source_rgbs")
    rc = item.get("ref_camera_ids")
    if src is None or not torch.is_tensor(src) or rc is None or not torch.is_tensor(rc):
        return
    root = os.path.abspath(os.path.expanduser(str(motion_scene_root)))
    sub = str(ref_subdir).strip("/\\") or DYNAMIC_BENCHMARK_REF_SUBDIR
    frame_ids = rc.detach().cpu().reshape(-1).long().tolist()
    n = min(len(frame_ids), int(src.shape[0]))
    if n <= 0:
        return
    out = src.detach().clone()
    tw, th = int(src.shape[-1]), int(src.shape[-2])
    device = src.device
    dtype = src.dtype
    rb = item.get("ref_imgs_bool")
    rb_flat = rb.detach().cpu().reshape(-1) if rb is not None and torch.is_tensor(rb) else None
    for i in range(n):
        if rb_flat is not None and i < rb_flat.numel() and float(rb_flat[i].item()) == 0.0:
            continue
        stem = str(int(frame_ids[i]))
        sr_path = _resolve_gt_image_path_with_variants(root, sub, stem)
        if sr_path is None:
            raise FileNotFoundError(
                f"SR ref not found for frame_id={stem} under {os.path.join(root, sub)} "
                f"(tried stem variants)."
            )
        bgr = cv2.imread(sr_path)
        if bgr is None:
            raise FileNotFoundError(f"SR reference image unreadable: {sr_path}")
        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_u8 = _pad_rgb_hwc_u8_to_aspect_standard(rgb_u8, aspect_standard)
        rgb_u8 = cv2.resize(rgb_u8, (tw, th), interpolation=cv2.INTER_AREA)
        chw = (
            torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(device=device, dtype=dtype)
            .clamp(0, 1)
        )
        out[i] = chw
    item["source_rgbs"] = out.clamp(0, 1)


def _motion_scene_root_from_mask_path(mask_path: str) -> str:
    """``.../{scene}/samurai_seg/{stem}.png`` → ``.../{scene}``."""
    return os.path.abspath(os.path.join(os.path.dirname(mask_path), ".."))


def _resolve_gt_image_path(motion_root: str, subdir: str, stem: str) -> Optional[str]:
    """Return first existing ``{motion_root}/{subdir}/{stem}.{ext}``."""
    base = os.path.join(motion_root, subdir, stem)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = base + ext
        if os.path.isfile(p):
            return p
    return None


def _gt_stem_filename_candidates(stem: str) -> List[str]:
    """Try common frame filename variants (``00028`` vs ``28.png`` on disk, different zero-padding)."""
    s = stem.strip()
    if not s:
        return []
    name, _ext = os.path.splitext(s)
    if not name:
        return [s]
    out: List[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(name)
    if name.isdigit():
        n = int(name)
        add(str(n))
        for width in (2, 3, 4, 5, 6, 8):
            add(f"{n:0{width}d}")
    return out


def _resolve_gt_image_path_with_variants(
    lookup_root: str, subdir: str, stem: str
) -> Optional[str]:
    """Like :func:`_resolve_gt_image_path` but tries :func:`_gt_stem_filename_candidates`."""
    sub = subdir.strip("/\\") if subdir else ""
    folder = os.path.join(lookup_root, sub) if sub else lookup_root
    for stem_c in _gt_stem_filename_candidates(stem):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = os.path.join(folder, stem_c + ext)
            if os.path.isfile(p):
                return p
    return None


def _stem_to_frame_id(stem: str) -> Optional[int]:
    name, _ = os.path.splitext(stem.strip())
    return int(name) if name.isdigit() else None


def _nearest_numeric_frame_image(folder: str, target_id: int) -> Optional[str]:
    """Pick ``{fid}.(png|jpg|…)`` under ``folder`` minimizing ``abs(fid - target_id)`` (tie → smaller fid)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    best_p: Optional[str] = None
    best_d = 10**18
    best_fid: Optional[int] = None
    for fn in names:
        base, ext = os.path.splitext(fn)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            continue
        if not base.isdigit():
            continue
        fid = int(base)
        d = abs(fid - target_id)
        if d < best_d:
            best_d = d
            best_p = os.path.join(folder, fn)
            best_fid = fid
        elif d == best_d and best_fid is not None and fid < best_fid:
            best_p = os.path.join(folder, fn)
            best_fid = fid
    return best_p


def _resolve_crop_sr_gt_path(
    *,
    motion_root: str,
    gt_img_subdir: str,
    stem: str,
    sr_lookup_abs: Optional[str],
    nearest_if_missing: bool,
) -> Optional[str]:
    """Exact SR crop match (variants), optionally nearest numeric frame; scene folder then shared lookup."""
    roots: List[Tuple[str, str]] = [(motion_root, gt_img_subdir)]
    if sr_lookup_abs:
        ap = os.path.abspath(os.path.expanduser(str(sr_lookup_abs).strip()))
        if ap:
            roots.append((ap, ""))

    for lookup_root, sub in roots:
        sub_path = os.path.join(lookup_root, sub) if sub else lookup_root
        if not os.path.isdir(sub_path):
            continue
        hit = _resolve_gt_image_path_with_variants(lookup_root, sub, stem)
        if hit is not None:
            return hit

    target_id = _stem_to_frame_id(stem)
    if nearest_if_missing and target_id is not None:
        for lookup_root, sub in roots:
            sub_path = os.path.join(lookup_root, sub) if sub else lookup_root
            if not os.path.isdir(sub_path):
                continue
            hit = _nearest_numeric_frame_image(sub_path, target_id)
            if hit is not None:
                return hit

    return None


def _legacy_parsing_gt_bgr(gt_path: str, mask: np.ndarray) -> np.ndarray:
    """Mask-weighted composite on white background (native full-frame GT)."""
    gt_img = cv2.imread(gt_path)
    if gt_img is None:
        raise FileNotFoundError(f"failed to read GT image: {gt_path}")
    parsing_img = (gt_img / 255.0) * (mask[..., None] / 255.0) + (
        1.0 - mask[..., None] / 255.0
    )
    return (parsing_img * 255.0).astype(np.uint8)


def _infer_native_roi_tw_th(
    render_h: int, render_w: int, scale_x: float, scale_y: float
) -> Tuple[int, int]:
    """Match ``infer_model_animation._resize_dsize_tw_th``: resize render plane → native ROI ``(tw,th)``."""
    tw = max(1, int(round(float(render_w) / float(scale_x))))
    th = max(1, int(round(float(render_h) / float(scale_y))))
    return tw, th


def _paste_gt_roi_on_native_canvas(
    roi_bgr: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    offset_y: float,
    offset_x: float,
) -> np.ndarray:
    """Paste ``roi_bgr`` (already resized to native ROI size) onto a white ``canvas_h×canvas_w`` board."""
    board = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    th, tw = int(roi_bgr.shape[0]), int(roi_bgr.shape[1])
    oy, ox = int(round(offset_y)), int(round(offset_x))
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(canvas_h, oy + th), min(canvas_w, ox + tw)
    if y1 <= y0 or x1 <= x0:
        return board
    ry0, rx0 = y0 - oy, x0 - ox
    ry1, rx1 = ry0 + (y1 - y0), rx0 + (x1 - x0)
    patch = roi_bgr[ry0:ry1, rx0:rx1]
    board[y0:y1, x0:x1] = patch
    return board


def _virtual_pad_offsets_from_ratio(
    unpad_h: int, unpad_w: int, pad_ratio: float
) -> Tuple[int, int]:
    """Same top-left offset as ``img_center_padding``; used to map JSON intrinsics to native pixels."""
    h_pad = round((1.0 + float(pad_ratio)) * unpad_h)
    w_pad = round((1.0 + float(pad_ratio)) * unpad_w)
    offset_h = (h_pad - unpad_h) // 2
    offset_w = (w_pad - unpad_w) // 2
    return offset_h, offset_w


def _verify_native_motion_pack_resolution_chain(
    motion_seqs_ret: Mapping[str, Any], aspect_standard: float
) -> None:
    """Ensure ``scale_*`` and resized masks match a crop on native masks (not a padded canvas)."""
    u0 = motion_seqs_ret["unpad_mask"][0]
    m0 = motion_seqs_ret["masks"][0]
    ox = motion_seqs_ret["offset_list"][0]
    sx, sy = float(ox[0]), float(ox[1])
    crop_ref, _, _ = center_crop(u0, None, aspect_standard, (1.0, 1.0))
    ch, cw = int(crop_ref.shape[0]), int(crop_ref.shape[1])
    inferred_h = float(m0.shape[0]) / sy
    inferred_w = float(m0.shape[1]) / sx
    dh, dw = abs(inferred_h - ch), abs(inferred_w - cw)
    oh, ow = motion_seqs_ret["ori_size"]
    if (int(oh), int(ow)) != (int(u0.shape[0]), int(u0.shape[1])):
        _LOG.warning(
            "ori_size %s vs first native mask %s (expected equal for native motion pack)",
            (oh, ow),
            u0.shape[:2],
        )
    if dw > 1.5 or dh > 1.5:
        _LOG.warning(
            "native motion pack resolution chain mismatch: center_crop size %s vs inferred from scales %s "
            "(dh=%.3f dw=%.3f); check JSON intrinsics / pad_ratio / mask alignment.",
            (ch, cw),
            (inferred_h, inferred_w),
            dh,
            dw,
        )
    else:
        _LOG.debug(
            "native motion pack resolution OK: native=%s crop=%s resized=%s scale=(%.4f,%.4f) tgt_render uses native chain",
            u0.shape[:2],
            (ch, cw),
            m0.shape[:2],
            sx,
            sy,
        )


def prepare_motion_seqs_benchmark_animation_native(
    motion_seqs: Sequence[Mapping[str, Any]],
    mask_paths: Sequence[str],
    bg_color: float,
    aspect_standard: float,
    enlarge_ratio: Sequence[float],
    render_image_res: Any,
    need_mask: bool,
    multiply: int = 14,
    tgt_size: int = 384,
    vis_motion: bool = False,
    motion_size: int = 3000,
    specific_id_list: Any = None,
    res_scale: float = 1.0,
    *,
    assume_json_camera_in_padded_canvas: bool = True,
    gt_img_subdir: str = "imgs_png",
    gt_crop_sr_resize_only: bool = False,
    gt_sr_lookup_abs: Optional[str] = None,
    gt_sr_fallback_to_imgs_png: bool = True,
    gt_sr_nearest_if_missing: bool = False,
) -> Dict[str, Any]:
    """Like ``prepare_motion_seqs_benchmark`` but **without** ``img_center_padding`` on masks.

    Masks stay at disk **native** resolution; JSON focal / principal point are shifted from the virtual
    padded canvas (``pad_ratio``) into native pixel coordinates before ``center_crop`` and ``scale_intrs``.
    ``ori_size`` is the first native frame HW; ``unpad_offset`` is ``[0,0,H,W]`` per frame (full board
    equals native; no separate padded canvas).

    ``gt_crop_sr_resize_only`` + ``gt_img_subdir`` (e.g. ``sr_imgs_png``): GT tiles are already
    cropped / aspect-fixed on disk (super-resolved crops). Skip mask-based cropping on GT; resize once
    to the same native ROI size used when pasting neural outputs, then embed into the full native board.

    Sparse SR timelines: set ``motion_gt_sr_fallback_to_imgs_png`` (default True) to use ``imgs_png``
    when the SR stem is missing; or ``motion_gt_sr_nearest_if_missing`` to pick the closest numbered
    frame in the SR folder; optional ``motion_gt_sr_lookup_abs`` adds a shared directory (tried after
    ``{scene}/{subdir}``).

    Parameters ``enlarge_ratio``, ``render_image_res``, ``need_mask``, ``specific_id_list``, and
    ``res_scale`` mirror the benchmark API but are unused here (training parity only).
    """
    _ = (enlarge_ratio, render_image_res, need_mask, specific_id_list, res_scale)

    motion_seqs = motion_seqs[:motion_size]
    mask_paths = mask_paths[:motion_size]

    c2ws: List[torch.Tensor] = []
    intrs: List[torch.Tensor] = []
    bg_colors: List[float] = []
    masks: List[np.ndarray] = []
    offset_list: List[List[float]] = []
    unpad_offset_list: List[List[int]] = []
    unpad_mask_list: List[np.ndarray] = []
    unpad_gt_img_list: List[np.ndarray] = []
    ori_h, ori_w = None, None
    smplx_params: List[Mapping[str, Any]] = []

    for idx, (smplx_raw_data, mask_path) in enumerate(zip(motion_seqs, mask_paths)):
        try:
            assert os.path.exists(mask_path)
        except Exception:
            continue
        stem = os.path.splitext(os.path.basename(mask_path))[0]
        motion_root = _motion_scene_root_from_mask_path(mask_path)

        mask_u8 = cv2.imread(mask_path)
        if mask_u8 is None:
            raise FileNotFoundError(f"failed to read mask: {mask_path}")
        mask = mask_u8[..., 0]

        pad_ratio = float(smplx_raw_data["pad_ratio"])
        smplx_param = {
            k: torch.FloatTensor(v)
            for k, v in smplx_raw_data.items()
            if "pad_ratio" not in k
        }
        c2w, intrinsic = _load_pose(smplx_param)

        unpad_h, unpad_w = mask.shape[0], mask.shape[1]
        unpad_mask_list.append(mask.copy())

        if ori_w is None:
            ori_h, ori_w = unpad_h, unpad_w

        if assume_json_camera_in_padded_canvas:
            off_h, off_w = _virtual_pad_offsets_from_ratio(unpad_h, unpad_w, pad_ratio)
            intrinsic[0, 2] -= float(off_w)
            intrinsic[1, 2] -= float(off_h)

        crop_mask, offset_x, offset_y = center_crop(mask, None, aspect_standard, (1.0, 1.0))

        intrinsic[0, 2] -= offset_x
        intrinsic[1, 2] -= offset_y

        new_tgt_size = [
            max(tgt_size * aspect_standard, crop_mask.shape[0]),
            max(tgt_size, crop_mask.shape[1]),
        ]
        new_tgt_size = (
            int(new_tgt_size[0] / multiply) * multiply,
            int(new_tgt_size[1] / multiply) * multiply,
        )
        scale_x = new_tgt_size[1] / crop_mask.shape[1]
        scale_y = new_tgt_size[0] / crop_mask.shape[0]
        intrinsic = scale_intrs(intrinsic, scale_x, scale_y)

        crop_mask = cv2.resize(
            crop_mask,
            dsize=(new_tgt_size[1], new_tgt_size[0]),
            interpolation=cv2.INTER_AREA,
        )

        if gt_crop_sr_resize_only:
            gt_path = _resolve_crop_sr_gt_path(
                motion_root=motion_root,
                gt_img_subdir=gt_img_subdir,
                stem=stem,
                sr_lookup_abs=gt_sr_lookup_abs,
                nearest_if_missing=gt_sr_nearest_if_missing,
            )
            if gt_path is None and gt_sr_fallback_to_imgs_png:
                legacy_path = _resolve_gt_image_path_with_variants(
                    motion_root, "imgs_png", stem
                )
                if legacy_path is None:
                    legacy_path = mask_path.replace("samurai_seg", "imgs_png")
                _LOG.info(
                    "motion GT: SR missing stem=%s scene=%s subdir=%s → imgs_png %s",
                    stem,
                    motion_root,
                    gt_img_subdir,
                    legacy_path,
                )
                parsing_img = _legacy_parsing_gt_bgr(legacy_path, mask)
            elif gt_path is None:
                raise FileNotFoundError(
                    f"motion GT ({gt_img_subdir}): no image for stem={stem} under {motion_root}"
                    + (
                        f" (also checked motion_gt_sr_lookup_abs={gt_sr_lookup_abs!r})"
                        if gt_sr_lookup_abs
                        else ""
                    )
                    + "; disable motion_gt_sr_fallback_to_imgs_png only if you supply dense SR tiles."
                )
            else:
                sr_bgr = cv2.imread(gt_path)
                if sr_bgr is None:
                    raise FileNotFoundError(f"failed to read GT image: {gt_path}")
                rh, rw = int(crop_mask.shape[0]), int(crop_mask.shape[1])
                tw, th = _infer_native_roi_tw_th(rh, rw, scale_x, scale_y)
                roi_u8 = cv2.resize(sr_bgr, (tw, th), interpolation=cv2.INTER_AREA)
                parsing_img = _paste_gt_roi_on_native_canvas(
                    roi_u8, unpad_h, unpad_w, offset_y, offset_x
                )
        else:
            legacy_path = _resolve_gt_image_path_with_variants(
                motion_root, gt_img_subdir, stem
            )
            if legacy_path is None:
                legacy_path = mask_path.replace("samurai_seg", gt_img_subdir)
            parsing_img = _legacy_parsing_gt_bgr(legacy_path, mask)

        unpad_gt_img_list.append(parsing_img)

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        smplx_params.append(smplx_param)
        masks.append(crop_mask)
        offset_list.append([scale_x, scale_y, offset_x, offset_y])
        unpad_offset_list.append([0, 0, unpad_h, unpad_w])

    c2ws_t = torch.stack(c2ws, dim=0)
    intrs_t = torch.stack(intrs, dim=0)
    bg_colors_t = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )

    smplx_params_tmp: Dict[str, List[Any]] = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)
    smplx_out = dict(smplx_params_tmp)
    # Per-frame betas from motion JSON (may differ across timeline); not replaced by dataset ref betas.
    if "betas" in smplx_out:
        b0 = smplx_out["betas"][0:1]
        smplx_out["betas"] = b0.expand(smplx_out["betas"].shape[0], -1).contiguous()

    if vis_motion:
        motion_render = render_smplx_mesh(smplx_out, intrs_t)
    else:
        motion_render = None

    for k, v in smplx_out.items():
        smplx_out[k] = v.unsqueeze(0)
    c2ws_t = c2ws_t.unsqueeze(0)
    intrs_t = intrs_t.unsqueeze(0)
    bg_colors_t = bg_colors_t.unsqueeze(0)

    motion_seqs_ret: Dict[str, Any] = {}
    motion_seqs_ret["render_c2ws"] = c2ws_t
    motion_seqs_ret["render_intrs"] = intrs_t
    motion_seqs_ret["render_bg_colors"] = bg_colors_t
    motion_seqs_ret["smplx_params"] = smplx_out
    motion_seqs_ret["rgbs"] = []
    motion_seqs_ret["vis_motion_render"] = motion_render
    motion_seqs_ret["motion_seqs"] = motion_seqs
    motion_seqs_ret["offset_list"] = offset_list
    motion_seqs_ret["masks"] = masks
    motion_seqs_ret["ori_size"] = (ori_h, ori_w)
    motion_seqs_ret["unpad_offset"] = unpad_offset_list
    motion_seqs_ret["unpad_mask"] = unpad_mask_list
    motion_seqs_ret["unpad_gt_img_list"] = unpad_gt_img_list

    _verify_native_motion_pack_resolution_chain(motion_seqs_ret, aspect_standard)
    return motion_seqs_ret


def query_target_motion_for_frames(
    motion_path: str, frame_indices: Sequence[int], cfg: Any
) -> Dict[str, Any]:
    """Build motion tensors for exactly ``frame_indices`` (SAMURAI + SMPL-X under ``motion_path``).

    Unpacked rec_mv data typically uses **the same zero-padded stem per frame**, e.g.
    ``smplx_params/00001.json``, ``samurai_seg/00001.png``, and (via native motion pack)
    ``imgs_png/{stem}.png`` aligned with ``samurai_seg``. **Default** GT root is ``imgs_png``.
    **Reference** views are sampled from ``{scene}/ref_imgs_png/`` (see ``motion_ref_imgs_subdir``).
    **Timeline GT** defaults to ``imgs_png`` + mask composite (full native frame).

    ROI resize+paste GT from pre-cropped tiles is opt-in via ``motion_gt_img_subdir`` +
    ``motion_gt_crop_sr_resize_only: true`` (or ``motion_gt_prefer_crop_sr``).

    Config (OmegaConf / merged infer cfg):

    - ``motion_ref_imgs_subdir``: per-scene folder for reference PNGs (default ``ref_imgs_png``).
    - ``motion_ref_source_tgt_max_size``: PadRatio reference height (CLI ``--src-height``).
    - ``motion_gt_img_subdir``: force GT subdirectory (e.g. ``imgs_png``, ``ref_imgs_png``).
    - ``motion_gt_crop_sr_resize_only``: ROI resize+paste layout for GT (explicit override).
    - ``motion_gt_sr_lookup_abs``: optional absolute path to a shared reference-tile folder
      (checked after ``{scene}/{subdir}``).
    - ``motion_gt_sr_fallback_to_imgs_png`` (default ``True``): if SR tile missing for a frame, use
      ``imgs_png`` + mask composite instead of failing (sparse SR timelines).
    - ``motion_gt_sr_nearest_if_missing``: if True, pick closest numbered ``*.png`` in SR dirs before
      imgs_png fallback.
    """

    ordered = [int(x) for x in frame_indices]
    smplx_path = os.path.join(motion_path, "smplx_params")
    mask_root = os.path.join(motion_path, "samurai_seg")

    stem_to_path = {}
    for p in sorted(glob.glob(os.path.join(smplx_path, "*.json"))):
        mid = int(os.path.splitext(os.path.basename(p))[0])
        stem_to_path[mid] = p

    missing = [t for t in ordered if t not in stem_to_path]
    if missing:
        raise FileNotFoundError(
            f"motion_path={motion_path}: missing smplx json for frames {missing}"
        )

    query_motion_paths: List[str] = []
    query_mask_paths: List[str] = []
    for t in ordered:
        json_path = stem_to_path[t]
        # e.g. 00001.json -> samurai_seg/00001.png (zero-padded stems, not 1.png)
        file_stem = os.path.splitext(os.path.basename(json_path))[0]
        query_motion_paths.append(json_path)
        query_mask_paths.append(os.path.join(mask_root, f"{file_stem}.png"))
    for mp in query_mask_paths:
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"mask PNG not found: {mp}")

    smplx_list = obtain_motion_sequence_from_paths(query_motion_paths)
    render_size = int(_cfg_get(cfg, "render_size", 420))

    subdir_ov = _cfg_get(cfg, "motion_gt_img_subdir", None)
    prefer_ref = bool(_cfg_get(cfg, "motion_gt_prefer_crop_sr", False))
    ref_root = os.path.join(motion_path, DYNAMIC_BENCHMARK_REF_SUBDIR)
    legacy_ref_root = os.path.join(motion_path, _LEGACY_REF_SUBDIR)

    if subdir_ov is not None:
        gt_img_subdir = str(subdir_ov).strip() or "imgs_png"
    elif prefer_ref and os.path.isdir(ref_root):
        gt_img_subdir = DYNAMIC_BENCHMARK_REF_SUBDIR
    elif prefer_ref and os.path.isdir(legacy_ref_root):
        gt_img_subdir = _LEGACY_REF_SUBDIR
    else:
        gt_img_subdir = "imgs_png"

    sr_resize_ov = _cfg_get(cfg, "motion_gt_crop_sr_resize_only", None)
    if sr_resize_ov is not None:
        gt_crop_sr_resize_only = bool(sr_resize_ov)
    else:
        gt_crop_sr_resize_only = gt_img_subdir in _REF_GT_RESIZE_ONLY_SUBDIRS

    if gt_crop_sr_resize_only:
        _LOG.debug(
            "motion GT: subdir=%s resize_only ROI paste (pre-cropped tiles; no mask crop on GT)",
            gt_img_subdir,
        )
    else:
        _LOG.debug("motion GT: imgs_png + mask composite (subdir=%s)", gt_img_subdir)

    motion_seqs_ret = prepare_motion_seqs_benchmark_animation_native(
        smplx_list,
        mask_paths=query_mask_paths,
        bg_color=1.0,
        aspect_standard=5.0 / 3,
        enlarge_ratio=[1.0, 1.0],
        tgt_size=render_size,
        render_image_res=render_size,
        need_mask=_cfg_get(cfg, "motion_img_need_mask", False),
        vis_motion=_cfg_get(cfg, "vis_motion", False),
        motion_size=100000,
        specific_id_list=None,
        res_scale=1.0,
        assume_json_camera_in_padded_canvas=_cfg_get(
            cfg, "motion_json_camera_in_padded_canvas", True
        ),
        gt_img_subdir=gt_img_subdir,
        gt_crop_sr_resize_only=gt_crop_sr_resize_only,
        gt_sr_lookup_abs=_cfg_get(cfg, "motion_gt_sr_lookup_abs", None),
        gt_sr_fallback_to_imgs_png=bool(
            _cfg_get(cfg, "motion_gt_sr_fallback_to_imgs_png", True)
        ),
        gt_sr_nearest_if_missing=bool(
            _cfg_get(cfg, "motion_gt_sr_nearest_if_missing", False)
        ),
    )
    return motion_seqs_ret


def _crop_unpad_roi(
    arr: np.ndarray, offset_h: int, offset_w: int, h: int, w: int
) -> np.ndarray:
    """Crop the leading H×W plane (HW or HWC)."""
    return arr[offset_h : offset_h + h, offset_w : offset_w + w]


def _align_gt_or_mask_with_predictions(
    arr: np.ndarray,
    *,
    offset_h: int,
    offset_w: int,
    uh: int,
    uw: int,
    board_h: int,
    board_w: int,
) -> np.ndarray:
    """Motion-pack GT/mask are usually **already native** ``uh×uw`` (disk); they are not painted on
    the padded canvas. Only predictions are full-board and need ``offset`` cropping to remove pad.

    If ``arr`` already matches ``(uh, uw)``, return as-is. If it matches the prediction canvas
    ``(board_h, board_w)``, apply the same ROI crop as neural/GS.
    """
    a = np.asarray(arr)
    ah, aw = int(a.shape[0]), int(a.shape[1])
    if ah == uh and aw == uw:
        return a
    if ah == board_h and aw == board_w:
        return _crop_unpad_roi(a, offset_h, offset_w, uh, uw)
    if ah >= offset_h + uh and aw >= offset_w + uw:
        return _crop_unpad_roi(a, offset_h, offset_w, uh, uw)
    return a


def save_unpadded_quads_under_view_dirs(
    output_root: str,
    uid: str,
    ref_view_count: int,
    stem: str,
    neural_rgb_u8: np.ndarray,
    neural_mask_u8: np.ndarray,
    unpad_offset: Sequence[int],
    unpad_gt_u8: np.ndarray,
    unpad_mask_u8: np.ndarray,
    *,
    gs_rgb_f: Optional[np.ndarray] = None,
    gs_mask_f: Optional[np.ndarray] = None,
    export_gs: bool = False,
) -> None:
    """Writes ``gt/*`` under ``uid`` and per-view ``neural_*``; ``gs_*`` only when ``export_gs``.

    **Unpad predictions:** ``neural_*`` / ``gs_*`` are rasterized on the full inference board
    (possibly larger than native); we crop with ``unpad_offset`` to native ``uh×uw``.

    **GT:** ``unpad_gt_img_list`` / ``unpad_mask`` from the motion pack are normally **already**
    native-resolution tiles (``uh×uw``) from disk — we do **not** re-apply pad offsets to them.
    If they ever match the full board size, the same ROI crop as predictions is applied.
    """

    offset_h, offset_w, uh, uw = (int(unpad_offset[0]), int(unpad_offset[1]), int(unpad_offset[2]), int(unpad_offset[3]))

    bh = int(np.asarray(neural_rgb_u8).shape[0])
    bw = int(np.asarray(neural_rgb_u8).shape[1])

    base = os.path.join(output_root, uid)
    gt_rgb_dir = os.path.join(base, "gt", "rgb")
    gt_mask_dir = os.path.join(base, "gt", "mask")
    vid = f"view_{int(ref_view_count):03d}"
    vdir = os.path.join(base, vid)
    os.makedirs(gt_rgb_dir, exist_ok=True)
    os.makedirs(gt_mask_dir, exist_ok=True)
    for sub in ("neural_rgb", "neural_mask"):
        os.makedirs(os.path.join(vdir, sub), exist_ok=True)
    if export_gs:
        for sub in ("gs_rgb", "gs_mask"):
            os.makedirs(os.path.join(vdir, sub), exist_ok=True)

    neu_r = _crop_unpad_roi(neural_rgb_u8, offset_h, offset_w, uh, uw)
    neu_m_col = _mask_to_hw(neural_mask_u8)
    neu_m = _crop_unpad_roi(np.asarray(neu_m_col, dtype=np.uint8), offset_h, offset_w, uh, uw)

    gt_bgr = _align_gt_or_mask_with_predictions(
        unpad_gt_u8,
        offset_h=offset_h,
        offset_w=offset_w,
        uh=uh,
        uw=uw,
        board_h=bh,
        board_w=bw,
    )
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    gt_m = _align_gt_or_mask_with_predictions(
        unpad_mask_u8,
        offset_h=offset_h,
        offset_w=offset_w,
        uh=uh,
        uw=uw,
        board_h=bh,
        board_w=bw,
    )

    Image.fromarray(neu_r).save(os.path.join(vdir, "neural_rgb", stem))
    Image.fromarray(neu_m).save(os.path.join(vdir, "neural_mask", stem))
    if export_gs:
        if gs_rgb_f is None or gs_mask_f is None:
            raise ValueError("export_gs=True requires gs_rgb_f and gs_mask_f")
        gs_hw3 = gs_rgb_f[..., :3] if gs_rgb_f.shape[-1] >= 3 else gs_rgb_f
        gs_u8 = np.clip(gs_hw3 * 255.0, 0.0, 255.0).round().astype(np.uint8)
        gs_u8 = _crop_unpad_roi(gs_u8, offset_h, offset_w, uh, uw)
        gsm_hw = _mask_to_hw(np.asarray(gs_mask_f))
        gsm_u8 = (np.clip(gsm_hw, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        gsm_u8 = _crop_unpad_roi(gsm_u8, offset_h, offset_w, uh, uw)
        Image.fromarray(gs_u8).save(os.path.join(vdir, "gs_rgb", stem))
        Image.fromarray(gsm_u8).save(os.path.join(vdir, "gs_mask", stem))
    Image.fromarray(gt_rgb).save(os.path.join(gt_rgb_dir, stem))
    Image.fromarray(gt_m).save(os.path.join(gt_mask_dir, stem))


@torch.no_grad()
def build_benchmark_animation_gs_cache(
    lhm: torch.nn.Module,
    item: Mapping[str, Any],
    motion_seqs: Mapping[str, Any],
    *,
    device: str = "cuda",
    use_pred_shape: bool = False,
) -> BenchmarkAnimationGSCache:
    """Run ``infer_single_view`` once per scene (same refs); reuse for all timeline frames.

    ``use_pred_shape``: if true, replace SMPL-X betas with the model's ``pred_shape`` from
    ``infer_single_view`` (via :meth:`smplx_params_with_pred_shape_betas`). The cache stores that
    tensor for :func:`benchmark_mask_animation_infer_from_gs_cache`. If false (default), the cache
    leaves ``betas`` unset there; animation uses **motion JSON** betas unless ``use_pred_shape`` is True.
    """

    smpl_motion = _smpl_motion_dict_for_infer(
        item,
        motion_seqs,
        merge_item_camera=False,
        betas_from_motion_json=not use_pred_shape,
    )
    fc = _infer_lhm_animation_forward_cache(
        lhm,
        dict(item),
        smpl_motion,
        dict(motion_seqs),
        ref_imgs_bool=item["ref_imgs_bool"].unsqueeze(0),
        device=device,
    )
    (
        gs_model_list,
        query_points,
        transform_mat_neutral_pose,
        gs_hidden_features,
        image_latents,
        motion_emb,
        pos_emb,
        pred_shape,
    ) = fc

    betas = smpl_motion["betas"]
    if torch.is_tensor(betas) and betas.dim() == 3:
        betas = betas[:, 0, :].contiguous()
    if (
        use_pred_shape
        and pred_shape is not None
        and hasattr(lhm, "smplx_params_with_pred_shape_betas")
    ):
        betas = lhm.smplx_params_with_pred_shape_betas({"betas": betas}, pred_shape)["betas"]
    if not use_pred_shape:
        betas = None

    return BenchmarkAnimationGSCache(
        gs_model_list=gs_model_list,
        query_points=query_points,
        transform_mat_neutral_pose=transform_mat_neutral_pose,
        gs_hidden_features=gs_hidden_features,
        image_latents=image_latents,
        motion_emb=motion_emb,
        pos_emb=pos_emb,
        betas=betas,
    )


@torch.no_grad()
def benchmark_mask_animation_infer_from_gs_cache(
    lhm: torch.nn.Module,
    item: Mapping[str, Any],
    motion_seqs: Mapping[str, Any],
    cache: BenchmarkAnimationGSCache,
    *,
    batch_size: int = 40,
    device: str = "cuda",
    use_pred_shape: bool = False,
    export_gs_outputs: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked ``animation_infer`` using a pre-built :class:`BenchmarkAnimationGSCache`.

    Default (``export_gs_outputs=False``): neural renderer only. With ``export_gs_outputs=True``,
    also returns GS raster ``comp_rgb`` / ``comp_mask`` pasted on the native board.
    """

    smpl_motion = _smpl_motion_dict_for_infer(
        item,
        motion_seqs,
        betas_override=cache.betas if use_pred_shape else None,
        merge_item_camera=False,
    )
    camera_size = int(smpl_motion["root_pose"].shape[1])

    offset_list = motion_seqs["offset_list"]
    ori_h, ori_w = motion_seqs["ori_size"]
    output_rgb = torch.ones((ori_h, ori_w, 3))

    rgb_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    gs_rgb_list: List[np.ndarray] = []
    gs_mask_list: List[np.ndarray] = []

    keys = [
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

    transform_mat_neutral_pose = cache.transform_mat_neutral_pose

    for batch_i in range(0, camera_size, batch_size):
        bch = min(batch_size, camera_size - batch_i)
        # ``get_single_view_smpl_data`` / SMPL-X ``blend_shapes`` expect ``betas`` as ``[1, D]``, not
        # ``[1, T, D]``: shape is shared across the batched motion frames; poses vary per column.
        if use_pred_shape:
            if cache.betas is None:
                raise RuntimeError("use_pred_shape=True requires cache.betas from build_benchmark_animation_gs_cache")
            pred_b = cache.betas.to(device)
            if pred_b.dim() == 3:
                batch_betas = pred_b[:, 0, :].contiguous()
            else:
                batch_betas = pred_b
            if batch_betas.dim() == 1:
                batch_betas = batch_betas.unsqueeze(0)
        else:
            mb = smpl_motion["betas"]
            if not torch.is_tensor(mb):
                mb = torch.as_tensor(mb, dtype=torch.float32)
            if mb.dim() == 3:
                batch_betas = mb[:, 0, :].contiguous()
            elif mb.dim() == 2:
                batch_betas = mb
            else:
                batch_betas = mb.reshape(1, -1)
            batch_betas = batch_betas.to(device)
            if batch_betas.dim() == 1:
                batch_betas = batch_betas.unsqueeze(0)

        batch_smplx_params = {
            "betas": batch_betas,
            "transform_mat_neutral_pose": transform_mat_neutral_pose,
        }
        for key in keys:
            if key not in smpl_motion:
                continue
            batch_smplx_params[key] = smpl_motion[key][
                :, batch_i : batch_i + batch_size
            ].to(device)

        masks_slice = motion_seqs["masks"][batch_i : batch_i + batch_size]
        offs_slice = offset_list[batch_i : batch_i + batch_size]

        anim_out = lhm.animation_infer(
            cache.gs_model_list,
            cache.query_points,
            batch_smplx_params,
            render_c2ws=motion_seqs["render_c2ws"][:, batch_i : batch_i + batch_size].to(
                device
            ),
            render_intrs=motion_seqs["render_intrs"][:, batch_i : batch_i + batch_size].to(
                device
            ),
            render_bg_colors=motion_seqs["render_bg_colors"][
                :, batch_i : batch_i + batch_size
            ].to(device),
            gs_hidden_features=cache.gs_hidden_features,
            image_latents=cache.image_latents,
            motion_emb=cache.motion_emb,
            pos_emb=cache.pos_emb,
            offset_list=offs_slice,
            mask_seqs=masks_slice,
            output_rgb=output_rgb,
            return_gs_outputs=export_gs_outputs,
            infer_output_renderer="neural",
        )

        if export_gs_outputs:
            batch_rgb, batch_mask, batch_gs_rgb, batch_gs_mask = anim_out
            gs_rgb_list.append(batch_gs_rgb.clamp(0, 1).detach().cpu().numpy())
            gs_mask_list.append(batch_gs_mask.clamp(0, 1).detach().cpu().numpy())
        else:
            batch_rgb, batch_mask = anim_out

        rgb_list.append(
            (batch_rgb.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
        )
        mask_list.append(
            (batch_mask.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
        )

    pred_rgb = np.concatenate(rgb_list, axis=0)
    pred_mask = np.concatenate(mask_list, axis=0)
    if export_gs_outputs:
        return (
            pred_rgb,
            pred_mask,
            np.concatenate(gs_rgb_list, axis=0),
            np.concatenate(gs_mask_list, axis=0),
        )
    return pred_rgb, pred_mask


@torch.no_grad()
def benchmark_mask_inference_with_gs(
    lhm: torch.nn.Module,
    item: Mapping[str, Any],
    motion_seqs: Mapping[str, Any],
    *,
    batch_size: int = 40,
    device: str = "cuda",
    use_pred_shape: bool = False,
    export_gs_outputs: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked ``animation_infer`` (one full ``infer_single_view`` + animation batches)."""

    cache = build_benchmark_animation_gs_cache(
        lhm, item, motion_seqs, device=device, use_pred_shape=use_pred_shape
    )
    return benchmark_mask_animation_infer_from_gs_cache(
        lhm,
        item,
        motion_seqs,
        cache,
        batch_size=batch_size,
        device=device,
        use_pred_shape=use_pred_shape,
        export_gs_outputs=export_gs_outputs,
    )
