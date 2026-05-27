# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-14 19:22:22
# @Function      : Util Tools for Inference.

import json
import math
import os
import pdb
from collections import defaultdict

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def dump_json(
    save_path: str,
    smplx_params: dict,
    motion_id: int,
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
) -> None:
    """Saves SMPL-X params, cameras, and intrinsics to a JSON file.

    Args:
        save_path (str): Output file path.
        smplx_params (dict): SMPL-X parameter dict (tensors converted to lists).
        motion_id (int): Identifier for the motion sequence.
        c2ws (torch.Tensor): Camera-to-world matrices [B, F, 4, 4].
        intrs (torch.Tensor): Intrinsic matrices [B, F, 4, 4].
    """
    save_json = {
        "smplx_params": {},
        "motion_id": motion_id,
        "c2ws": c2ws.detach().cpu().numpy().tolist(),
        "intrs": intrs.detach().cpu().numpy().tolist(),
    }
    for key, v in smplx_params.items():
        if isinstance(v, torch.Tensor):
            save_json["smplx_params"][key] = v.detach().cpu().numpy().tolist()
        else:
            save_json["smplx_params"][key] = v

    with open(save_path, "w") as f:
        json.dump(save_json, f, indent=2)


def generate_rotation_matrix_y(degrees):
    """Construct a right-handed rotation matrix for rotation about the Y-axis.

    This function converts the input angle from degrees to radians and returns
    a 3x3 rotation matrix that rotates vectors around the Y-axis following the
    right-hand rule. The implementation is numerically stable for typical use
    in computer vision and graphics pipelines.

    Args:
        degrees (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix with dtype float32.
    """
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def scale_intrs(intrs, ratio_x, ratio_y):
    """Scale camera intrinsics under image resizing.

    This updates the focal lengths and principal point coordinates according to
    the provided horizontal and vertical scale ratios. The function supports both
    batched intrinsics of shape [B, 4] or [B, 3, 3]/[B, 4, 4] style (with the first
    two rows representing fx/cx and fy/cy), and single-frame intrinsics.

    Args:
        intrs (numpy.ndarray or torch.Tensor): Intrinsic parameters; the first two
            rows are expected to be [fx, cx] and [fy, cy] style when rank >= 2.
        ratio_x (float): Horizontal scale factor applied to width-related terms.
        ratio_y (float): Vertical scale factor applied to height-related terms.

    Returns:
        Same type as input: Scaled intrinsics with updated focal and principal points.
    """
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def _snap_hw_to_multiply(h: float, w: float, multiply: int) -> tuple:
    """Snap height and width down to nearest multiple of multiply."""
    h_snap = int(h / multiply) * multiply
    w_snap = int(w / multiply) * multiply
    return (h_snap, w_snap)


def calc_new_tgt_size(cur_hw, tgt_size, multiply):
    """Compute target height/width while preserving aspect ratio.

    The shortest side is scaled to `tgt_size` and both dimensions are then snapped
    down to the nearest multiple of `multiply`. Scale ratios along height and width
    are returned for downstream geometric updates.

    Args:
        cur_hw (tuple[int, int]): Current (H, W).
        tgt_size (int): Desired length for the shorter side before snapping.
        multiply (int): Granularity to which both H and W are aligned (e.g., 8/16).

    Returns:
        tuple: ((H_tgt, W_tgt), ratio_y, ratio_x).
    """
    ratio = tgt_size / min(cur_hw)
    tgt_h, tgt_w = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
    tgt_hw = _snap_hw_to_multiply(tgt_h, tgt_w, multiply)
    ratio_y, ratio_x = tgt_hw[0] / cur_hw[0], tgt_hw[1] / cur_hw[1]
    return tgt_hw, ratio_y, ratio_x


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    """Compute target size given a standard aspect ratio constraint.

    The output height/width are determined by enforcing `H/W ≈ aspect_standard` and
    limiting the longer side by `tgt_size`, followed by snapping to `multiply`.

    Args:
        cur_hw (tuple[int, int]): Current (H, W).
        aspect_standard (float): Target aspect ratio H/W.
        tgt_size (int): Scalar controlling the longer side length.
        multiply (int): Granularity to which the size is aligned.

    Returns:
        tuple: ((H_tgt, W_tgt), ratio_y, ratio_x).
    """
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_h, tgt_w = tgt_size * aspect_standard, tgt_size
    tgt_hw = _snap_hw_to_multiply(tgt_h, tgt_w, multiply)
    ratio_y, ratio_x = tgt_hw[0] / cur_hw[0], tgt_hw[1] / cur_hw[1]
    return tgt_hw, ratio_y, ratio_x


def _expand_bbox_and_clip(
    x_min, x_max, y_min, y_max, scale=1.1, img_w=None, img_h=None
):
    """Expand bbox by scale around center and clip to image bounds."""
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    new_width = (x_max - x_min) * scale
    new_height = (y_max - y_min) * scale
    x_min_new = center_x - new_width / 2
    x_max_new = center_x + new_width / 2
    y_min_new = center_y - new_height / 2
    y_max_new = center_y + new_height / 2
    if img_w is not None and img_h is not None:
        x_min_new = np.clip(x_min_new, 0, img_w)
        x_max_new = np.clip(x_max_new, 0, img_w)
        y_min_new = np.clip(y_min_new, 0, img_h)
        y_max_new = np.clip(y_max_new, 0, img_h)
    return int(x_min_new), int(x_max_new), int(y_min_new), int(y_max_new)


def img_center_padding(img_np, pad_ratio):
    """Pad an image around its center with zeros by a ratio of its size.

    Padding is applied symmetrically so the original image remains centered in
    the padded canvas. The function returns the padded image and the (y, x)
    offsets of the original top-left corner within the new canvas.

    Args:
        img_np (numpy.ndarray): Input image of shape [H, W] or [H, W, C].
        pad_ratio (float): Relative padding amount w.r.t. the original size.

    Returns:
        tuple: (padded_image, (offset_h, offset_w)).
    """

    ori_h, ori_w = img_np.shape[:2]

    h = round((1 + pad_ratio) * ori_h)
    w = round((1 + pad_ratio) * ori_w)

    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((h, w, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((h, w), dtype=np.uint8)
    offset_h, offset_w = (h - img_np.shape[0]) // 2, (w - img_np.shape[1]) // 2

    img_pad_np[
        offset_h : offset_h + img_np.shape[0],
        offset_w : offset_w + img_np.shape[1],
    ] = img_np

    return img_pad_np, (offset_h, offset_w)


def _load_pose(pose):
    """Build camera-to-world and intrinsic matrices from a pose dict.

    The intrinsic matrix is assembled from focal lengths and principal points.
    The camera-to-world matrix is initialized as identity here, serving as a
    placeholder for downstream updates.

    Args:
        pose (dict): Dictionary containing keys 'focal' and 'princpt'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (c2w [4x4], intrinsics [4x4]).
    """
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    c2w = torch.eye(4).float()

    return c2w, intrinsic


def resize_image_keepaspect_np(img, max_tgt_size):
    """Resize while preserving aspect ratio so the longer side equals max_tgt_size.

    This behavior mirrors PIL ImageOps.contain, ensuring that neither dimension
    exceeds `max_tgt_size` and the original aspect ratio is preserved.

    Args:
        img (numpy.ndarray): Input image array.
        max_tgt_size (int): Target size for the longer side after resizing.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    """Center crop an image around the masked foreground under an aspect constraint.

    The crop is centered at the image midpoint and expanded to tightly include the
    foreground region defined by the binary mask while enforcing an aspect ratio.
    An optional random enlargement is applied within specified bounds to provide
    spatial context.

    Args:
        img (numpy.ndarray): RGB image of shape [H, W, 3].
        mask (numpy.ndarray): Binary mask of shape [H, W] with foreground > 0.
        aspect_standard (float): Target H/W aspect to enforce for the crop.
        enlarge_ratio (tuple[float, float]): Min/Max random scale for enlargement.

    Returns:
        tuple: (cropped_img, cropped_mask, offset_x, offset_y).
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    half_w = max(abs(center_x - x_min), abs(center_x - x_max))
    half_h = max(abs(center_y - y_min), abs(center_y - y_max))
    half_w_raw = half_w
    half_h_raw = half_h
    aspect = half_h / half_w

    if aspect >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    # not exceed original image
    if half_h > center_y:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > center_x:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = (
            np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
            + enlarge_ratio_min
        )
        half_h, half_w = round(enlarge_ratio_cur * half_h), round(
            enlarge_ratio_cur * half_w
        )

    assert half_h <= center_y
    assert half_w <= center_x
    assert abs(half_h / half_w - aspect_standard) < 0.03

    offset_x = center_x - half_w
    offset_y = center_y - half_h

    new_img = img[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
    new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]

    return new_img, new_mask, offset_x, offset_y


def center_crop(mask, bbox, aspect_standard, enlarge_ratio):
    """Center crop a mask around its foreground while remaining within bounds.

    Expands the foreground bbox by scale 1.2 and returns a centered crop.

    Args:
        mask (numpy.ndarray): Binary mask of shape [H, W].
        aspect_standard (float): Target H/W aspect (reserved).
        enlarge_ratio (tuple[float, float]): Reserved.

    Returns:
        tuple: (cropped_mask, offset_x, offset_y).
    """

    if bbox is None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("empty mask")
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
    else:
        x_min, y_min, x_max, y_max = bbox.to_whwh().get_box()

    x_min, x_max, y_min, y_max = _expand_bbox_and_clip(
        x_min,
        x_max,
        y_min,
        y_max,
        scale=1.2,
        img_w=mask.shape[1],
        img_h=mask.shape[0],
    )

    h, w = mask.shape[0], mask.shape[1]
    center_x, center_y = w // 2, h // 2
    half_w = max(abs(center_x - x_min), abs(center_x - x_max))
    half_h = max(abs(center_y - y_min), abs(center_y - y_max))

    # Clamp to valid crop bounds (avoids assert failure when bbox touches edge
    # with odd image dimensions: e.g. H=511, half_h=256 > center_y=255)
    half_w = min(half_w, center_x, w - 1 - center_x)
    half_h = min(half_h, center_y, h - 1 - center_y)
    half_w = max(0, half_w)
    half_h = max(0, half_h)

    offset_x = center_x - half_w
    offset_y = center_y - half_h
    new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
    return new_mask, offset_x, offset_y


def preprocess_image(
    rgb_path,
    mask_path,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """Preprocess a single RGB image and optional mask for inference.

    The pipeline optionally pads the image, estimates or loads an alpha mask,
    composites the background color, resizes while preserving aspect ratio,
    performs a center crop guided by the mask, adjusts intrinsics accordingly,
    and finally resizes to the render target size aligned to a multiple.

    Args:
        rgb_path (str): Path to an RGB image.
        mask_path (str or None): Optional path to a mask image; when None and the
            input has no alpha channel, a background removal method is used.
        intr (numpy.ndarray or torch.Tensor or None): Intrinsic matrix to be updated in-place.
        pad_ratio (float): Center padding ratio applied before processing.
        bg_color (float): Background color value for compositing, in [0, 1].
        max_tgt_size (int): Max side length for the initial keep-aspect resize.
        aspect_standard (float): Target aspect ratio H/W for cropping.
        enlarge_ratio (tuple[float, float]): Min/Max random enlargement around the mask.
        render_tgt_size (int): Base target size for final rendering.
        multiply (int): Alignment multiple for final H/W.
        need_mask (bool): If False, a unit mask is used as placeholder.

    Returns:
        tuple[torch.Tensor, torch.Tensor, same-as-intr]:
            (rgb_tensor [1,3,H,W], mask_tensor [1,1,H,W], updated intrinsics or None).
    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()
    if pad_ratio > 0:
        rgb, offset_size = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0  # normalize to [0, 1]
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_path is not None:
                mask = np.array(Image.open(mask_path))
            else:
                from rembg import remove

                # rembg cuda version -> error
                mask = remove(rgb_raw[:, :, (2, 1, 0)])[:, :, -1]  # rembg expects BGR
            if pad_ratio > 0:
                mask, offset_size = img_center_padding(mask, pad_ratio)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]  # [H, W]
    else:
        # just placeholder
        mask = np.ones_like(rgb[:, :, 0])

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr


def extract_imgs_from_video(video_file: str, save_root: str, fps: int) -> None:
    """Extracts frames from a video at fixed intervals and saves as images.

    Args:
        video_file (str): Path to the video file.
        save_root (str): Directory to save extracted frames.
        fps (int): Sampling interval; every ``fps``-th frame is saved.
    """
    vr = decord.VideoReader(video_file)
    for i in range(0, len(vr), fps):
        frame = vr[i].asnumpy()
        save_path = os.path.join(save_root, f"{i:05d}.jpg")
        cv2.imwrite(save_path, frame[:, :, (2, 1, 0)])


def predict_motion_seqs_from_images(image_folder: str, save_root, fps=6):
    """Predict SMPL-X motion sequences from an image folder or video file.

    When an input video is provided, frames are first extracted. The function
    then invokes an external multi-human motion regressor to produce SMPL-X
    parameters for each frame.

    Args:
        image_folder (str): Path to an image directory or a video file.
        save_root (str): Directory where results and intermediates are stored.
        fps (int): Frame extraction interval for videos.

    Returns:
        tuple[str, str]: (path_to_smplx_results, path_to_image_folder_used).
    """
    id_name = os.path.splitext(os.path.basename(image_folder))[0]
    if os.path.isfile(image_folder) and (
        image_folder.endswith("mp4") or image_folder.endswith("move")
    ):
        save_frame_root = os.path.join(save_root, "extracted_frames", id_name)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root, exist_ok=True)
            extract_imgs_from_video(
                video_file=image_folder, save_root=save_frame_root, fps=fps
            )
        else:
            pass  # frames already extracted
        image_folder = save_frame_root

    image_folder_abspath = os.path.abspath(image_folder)
    save_smplx_root = image_folder + "_smplx_params_mhmr"
    if not os.path.exists(save_smplx_root):
        cmd = f"cd thirdparty/multi-hmr &&  python infer_batch.py  --data_root {image_folder_abspath}  --out_folder {image_folder_abspath} --crop_head   --crop_hand   --pad_ratio 0.2 --smplify"
        os.system(cmd)
    return save_smplx_root, image_folder


def _stack_smplx_params_list(
    smplx_params_list: list, shape_param: torch.Tensor
) -> dict:
    """Stack list of per-frame SMPL-X dicts into a single dict of batched tensors."""
    stacked = defaultdict(list)
    for sp in smplx_params_list:
        for k, v in sp.items():
            stacked[k].append(v)
    result = {k: torch.stack(v) for k, v in stacked.items()}
    result["betas"] = shape_param
    return result


def _to_motion_batch_dict(c2ws, intrs, bg_colors, smplx_params, rgbs, vis_motion):
    """Add batch dim and build standard motion batch dict. render_smplx_mesh expects [N,...]."""
    motion_render = render_smplx_mesh(smplx_params, intrs) if vis_motion else None

    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
    rgbs_out = rgbs.unsqueeze(0) if len(rgbs) > 0 else rgbs
    return {
        "render_c2ws": c2ws.unsqueeze(0),
        "render_intrs": intrs.unsqueeze(0),
        "render_bg_colors": bg_colors.unsqueeze(0),
        "smplx_params": smplx_params,
        "rgbs": rgbs_out,
        "vis_motion_render": motion_render,
    }


def render_smplx_mesh(
    smplx_params, render_intrs, human_model_path="./pretrained_models/human_model_files"
):
    from core.models.rendering.smplx import smplx
    from core.models.rendering.smplx.vis_utils import render_mesh

    layer_arg = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }

    smplx_layer = smplx.create(
        human_model_path,
        "smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=100,
        use_pca=False,
        use_face_contour=False,
        flat_hand_mean=True,
        **layer_arg,
    )

    body_pose = smplx_params["body_pose"]
    num_view = body_pose.shape[0]
    shape_param = smplx_params["betas"]
    if "expr" not in smplx_params:
        # supports v2.0 data format
        smplx_params["expr"] = torch.zeros((num_view, 100))

    output = smplx_layer(
        global_orient=smplx_params["root_pose"],
        body_pose=smplx_params["body_pose"].view(num_view, -1),
        left_hand_pose=smplx_params["lhand_pose"].view(num_view, -1),
        right_hand_pose=smplx_params["rhand_pose"].view(num_view, -1),
        jaw_pose=smplx_params["jaw_pose"],
        leye_pose=smplx_params["leye_pose"],
        reye_pose=smplx_params["reye_pose"],
        expression=smplx_params["expr"],
        betas=smplx_params["betas"].unsqueeze(0).repeat(num_view, 1),  # 10 blendshape
        transl=smplx_params["trans"],
        face_offset=None,
        joint_offset=None,
    )

    smplx_face = smplx_layer.faces.astype(np.int64)
    mesh_render_list = []
    for v_idx in range(num_view):
        intr = render_intrs[v_idx]
        cam_param = {
            "focal": torch.tensor([intr[0, 0], intr[1, 1]]),
            "princpt": torch.tensor([intr[0, 2], intr[1, 2]]),
        }
        render_shape = int(cam_param["princpt"][1] * 2), int(
            cam_param["princpt"][0] * 2
        )  # require h, w
        mesh_render, is_bkg = render_mesh(
            output.vertices[v_idx],
            smplx_face,
            cam_param,
            np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255,
            return_bg_mask=True,
        )
        mesh_render = mesh_render.astype(np.uint8)
        mesh_render_list.append(mesh_render)
    mesh_render = np.stack(mesh_render_list)
    return mesh_render


def prepare_motion_single(
    motion_seqs_dir,
    image_name,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    vis_motion=False,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.
        vis_motion (bool, optional): Flag indicating whether to visualize motion. Defaults to False.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions
            'vis_motion_render': rendering smplx motion

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """

    motion_seqs = [
        os.path.join(
            motion_seqs_dir,
            image_name.replace(".jpg", ".png").replace(".png", ".json"),
        )
        for _ in range(4)
    ]
    axis_list = [0, 60, 180, 240]

    # source images
    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = _parse_smplx_param(smplx_raw_data)

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)

        if "expr" in smplx_raw_data:
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as f:
                flame_param = json.load(f)
                smplx_param["expr"] = torch.FloatTensor(flame_param["expcode"])

                # replace with flame's jaw_pose
                smplx_param["jaw_pose"] = torch.FloatTensor(flame_param["posecode"][3:])
                smplx_param["leye_pose"] = torch.FloatTensor(flame_param["eyecode"][:3])
                smplx_param["reye_pose"] = torch.FloatTensor(flame_param["eyecode"][3:])

        else:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        root_rotate_matrix = axis_angle_to_matrix(smplx_param["root_pose"])
        rotate = generate_rotation_matrix_y(axis_list[idx])
        rotate = torch.from_numpy(rotate).float()
        rotate = rotate @ root_rotate_matrix
        new_rotate_axis = matrix_to_axis_angle(rotate)

        smplx_param["root_pose"] = new_rotate_axis

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    rgbs = torch.cat(rgbs, dim=0) if len(rgbs) > 0 else []

    smplx_params = _stack_smplx_params_list(smplx_params, shape_param)
    base = _to_motion_batch_dict(c2ws, intrs, bg_colors, smplx_params, rgbs, vis_motion)
    return base


def _parse_smplx_param(smplx_raw_data):
    """Parse SMPL-X from raw JSON dict, excluding pad_ratio."""
    return {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }


def _process_mask_frame_for_eval(
    mask,
    bbox,
    pad_ratio,
    intrinsic,
    tgt_size,
    aspect_standard,
    multiply,
    res_scale=1.0,
    resize_to_1024=True,
    mask_already_padded=False,
    rendering_with_pad=True,
):
    """Resize, pad, crop mask and update intrinsics. Returns (intrinsic, crop_mask, offset_entry, ori_size)."""
    if resize_to_1024:
        resize_scale = 1024 / max(mask.shape)
        resize_w = int(mask.shape[1] * resize_scale)
        resize_h = int(mask.shape[0] * resize_scale)
        if res_scale != 1.0:
            resize_w = int(resize_w * res_scale)
            resize_h = int(resize_h * res_scale)
            resize_scale *= res_scale
        if bbox is not None:
            bbox = bbox.scale_bbox(mask.shape[1], mask.shape[0], resize_w, resize_h)

        mask = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        intrinsic = scale_intrs(intrinsic, resize_scale, resize_scale)

    if mask_already_padded:
        mask_pad = mask
    else:
        # without using pad rendering space.
        if rendering_with_pad:
            # TODO: bbox should be scaled to the padded mask
            raise NotImplementedError("Padding with bbox is not implemented")
            mask_pad, _ = img_center_padding(mask, pad_ratio)
        else:
            _, offset_size = img_center_padding(mask, pad_ratio)
            mask_pad = mask
            intrinsic[0, 2] -= offset_size[1]
            intrinsic[1, 2] -= offset_size[0]

    ori_size = mask_pad.shape[:2]
    crop_mask, offset_x, offset_y = center_crop(mask_pad, bbox, 5 / 3.0, 1.0)
    intrinsic[0, 2] -= offset_x
    intrinsic[1, 2] -= offset_y

    new_tgt_size = _snap_hw_to_multiply(
        max(tgt_size * aspect_standard, crop_mask.shape[0]),
        max(tgt_size, crop_mask.shape[1]),
        multiply,
    )
    scale_x = new_tgt_size[1] / crop_mask.shape[1]
    scale_y = new_tgt_size[0] / crop_mask.shape[0]
    intrinsic = scale_intrs(intrinsic, scale_x, scale_y)
    crop_mask = cv2.resize(
        crop_mask,
        dsize=(new_tgt_size[1], new_tgt_size[0]),
        interpolation=cv2.INTER_AREA,
    )
    return intrinsic, crop_mask, [scale_x, scale_y, offset_x, offset_y], ori_size


def prepare_motion_seqs_eval(
    motion_seqs,
    mask_paths,
    bbox_list,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=14,
    tgt_size=384,
    vis_motion=False,
    motion_size=3000,
    specific_id_list=None,
    res_scale=1.0,
):
    """Prepare evaluation motion sequences from per-frame SMPL-X and mask data."""
    motion_seqs = motion_seqs[:motion_size]
    mask_paths = mask_paths[:motion_size]
    bbox_list = bbox_list[:motion_size]


    c2ws, intrs, rgbs, bg_colors, masks, offset_list = [], [], [], [], [], []
    ori_h, ori_w = None, None
    smplx_params = []
    shape_param = None

    for idx, (smplx_raw_data, mask_path) in enumerate(zip(motion_seqs, mask_paths)):
        if not os.path.exists(mask_path):
            continue
        bbox = bbox_list[idx]
        mask = cv2.imread(mask_path)[..., 0]
        pad_ratio = float(smplx_raw_data["pad_ratio"])
        smplx_param = _parse_smplx_param(smplx_raw_data)
        c2w, intrinsic = _load_pose(smplx_param)

        intrinsic, crop_mask, offset_entry, frame_ori_size = (
            _process_mask_frame_for_eval(
                mask,
                bbox,
                pad_ratio,
                intrinsic,
                tgt_size,
                aspect_standard,
                multiply,
                res_scale=res_scale,
                resize_to_1024=True,
                mask_already_padded=False,
                rendering_with_pad=False,
            )
        )
        if ori_w is None:
            ori_h, ori_w = frame_ori_size
        if idx == 0:
            shape_param = smplx_param["betas"]
        if "expr" not in smplx_param:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        smplx_params.append(smplx_param)
        masks.append(crop_mask)
        offset_list.append(offset_entry)

    if not c2ws:
        raise RuntimeError(
            "prepare_motion_seqs_eval: no valid frames (empty c2ws). "
            "Usually every SMPL-X JSON under smplx_params needs a matching PNG in samurai_seg; "
            "missing masks are skipped until the tensor list is empty."
        )

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    rgbs = torch.cat(rgbs, dim=0) if len(rgbs) > 0 else []

    smplx_params = _stack_smplx_params_list(smplx_params, shape_param)
    base = _to_motion_batch_dict(c2ws, intrs, bg_colors, smplx_params, rgbs, vis_motion)
    base["motion_seqs"] = motion_seqs
    base["offset_list"] = offset_list
    base["masks"] = masks
    base["ori_size"] = (ori_h, ori_w)
    return base


def prepare_motion_seqs_benchmark(
    motion_seqs,
    mask_paths,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=14,
    tgt_size=384,
    vis_motion=False,
    motion_size=3000,  # only support 12s videos
    specific_id_list=None,
    res_scale=1.0,  # whether to scale the rendering resolution
):
    """
    Prepare benchmark motion sequences dictionary similarly to prepare_motion_seqs_eval,
    but also keeps unpadded masks and GT images in output for evaluation protocols.

    Args:
        motion_seqs (list[dict]): List of per-frame SMPL-X parameter dicts.
        mask_paths (list[str]): Paths to binary mask images for each frame.
        bg_color (float or array): Background color used for rendering masked regions.
        aspect_standard (float): Expected aspect ratio H/W for output renders.
        enlarge_ratio (float): Ratio by which to expand crop around human in mask.
        render_image_res (tuple): (H, W) target render resolution.
        need_mask (bool): Whether masks are required for output.
        multiply (int): Size granularity (outputs are multiples of this).
        tgt_size (int): Target longer/shorter side for resizing.
        vis_motion (bool): Whether to produce visualization renders.
        motion_size (int): Max number of frames to process.
        specific_id_list (list[int] or None): Restricts frames by id if set.
        res_scale (float): Additional output down/upscaling factor.

    Returns:
        motion_seqs_ret (dict): {
            "render_c2ws": [...],            # [B, T, 4, 4]
            "render_intrs": [...],           # [B, T, 4, 4]
            "render_bg_colors": [...],       # [B, T, 3]
            "smplx_params": {...},           # dict of param_name: tensor
            "rgbs": [...],                   # [B, T, 3, H, W] or []
            "vis_motion_render": ...,
            "motion_seqs": [...],            # original motion seqs list
            'offset_list': [...],            # list of [scale_x, scale_y, ox, oy]
            'masks': [...],                  # list of masks after padding/cropping
            'ori_size': (H, W),              # original image size after padding
            'unpad_mask_list': [...],        # masks before padding/cropping
            'unpad_gt_img_list': [...],      # GT rgb images before padding/cropping
            'unpad_offset_list': [...],      # offset/pad tracking for restore
        }
    """

    motion_seqs = motion_seqs[:motion_size]
    mask_paths = mask_paths[:motion_size]

    # source images
    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    offset_list = []  # scale_x, scale_y, offset_x, offset_y
    unpad_offset_list = []  # offset_h, offset_w, unpad_h, unpad_w
    unpad_mask_list = []
    unpad_gt_img_list = []
    ori_h, ori_w = None, None
    smplx_params = []
    shape_param = None

    gt_replace = ("samurai_seg", "imgs_png")
    for idx, (smplx_raw_data, mask_path) in enumerate(zip(motion_seqs, mask_paths)):
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path)[..., 0]
        gt_img = cv2.imread(mask_path.replace(gt_replace[0], gt_replace[1]))
        parsing_img = (gt_img / 255.0) * (mask[..., None] / 255.0) + (
            1 - mask[..., None] / 255.0
        )
        unpad_gt_img_list.append((parsing_img * 255).astype(np.uint8))
        unpad_h, unpad_w = mask.shape[:2]
        unpad_mask_list.append(mask.copy())

        pad_ratio = float(smplx_raw_data["pad_ratio"])
        mask_pad, offset_size = img_center_padding(mask, pad_ratio)
        unpad_offset_list.append([offset_size[0], offset_size[1], unpad_h, unpad_w])
        if ori_w is None:
            ori_h, ori_w = mask_pad.shape[:2]

        smplx_param = _parse_smplx_param(smplx_raw_data)
        c2w, intrinsic = _load_pose(smplx_param)
        if idx == 0:
            shape_param = smplx_param["betas"]

        intrinsic, crop_mask, offset_entry, _ = _process_mask_frame_for_eval(
            mask_pad,
            0.0,
            intrinsic,
            tgt_size,
            aspect_standard,
            multiply,
            res_scale=res_scale,
            resize_to_1024=False,
            mask_already_padded=True,
        )

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        smplx_params.append(smplx_param)
        masks.append(crop_mask)
        offset_list.append(offset_entry)

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    rgbs = torch.cat(rgbs, dim=0) if len(rgbs) > 0 else []

    smplx_params = _stack_smplx_params_list(smplx_params, shape_param)
    base = _to_motion_batch_dict(c2ws, intrs, bg_colors, smplx_params, rgbs, vis_motion)
    base["motion_seqs"] = motion_seqs
    base["offset_list"] = offset_list
    base["masks"] = masks
    base["ori_size"] = (ori_h, ori_w)
    base["unpad_offset"] = unpad_offset_list
    base["unpad_mask"] = unpad_mask_list
    base["unpad_gt_img_list"] = unpad_gt_img_list
    return base
