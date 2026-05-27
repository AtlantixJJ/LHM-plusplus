# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-06-04 20:43:18
# @Function      : Base GSRender Class

import copy
import math
import os
import pdb
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply

from core.models.rendering.gaussian_decoder.crossattn_decoder import DecoderCrossAttn
from core.models.rendering.skinnings import (
    SMPLXDiffusedVoxelSkinning,
    SMPLXVoxelSkinning,
)
from core.models.rendering.utils.typing import *
from core.models.rendering.utils.utils import MLP
from core.modules.embed import PointEmbed
from core.outputs.output import GaussianAppOutput
from core.structures.camera import Camera, generate_rotation_matrix_y
from core.structures.gaussian_model import GaussianModel


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


class BaseGSRender(nn.Module):
    # Base class for 3D Gaussian Splatting (GS) renderer.
    def __init__(
        self,
        human_model_path,
        subdivide_num,
        smpl_type,
        feat_dim,
        query_dim,
        use_rgb,
        sh_degree,
        xyz_offset_max_step,
        mlp_network_config,
        expr_param_dim,
        shape_param_dim,
        clip_scaling=0.2,
        cano_pose_type=0,
        decoder_mlp=False,
        skip_decoder=False,
        fix_opacity=False,
        fix_rotation=False,
        decode_with_extra_info=None,
        gradient_checkpointing=False,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,  # only use for dense_smaple_smplx
        gs_deform_scale=0.005,
        render_features=False,
    ):
        """

        Args:
            human_model_path (str): Path to human model files.
            subdivide_num (int): Subdivision number for base mesh.
            smpl_type (str): Type of SMPL/SMPL-X/other model to use.
            feat_dim (int): Dimension of feature embeddings.
            query_dim (int): Dimension of query points/features.
            use_rgb (bool): Whether to use RGB channels.
            sh_degree (int): Spherical harmonics degree for appearance.
            xyz_offset_max_step (float): Max offset per step for position.
            mlp_network_config (dict or None): MLP configuration for feature mapping.
            expr_param_dim (int): Expression parameter dimension.
            shape_param_dim (int): Shape parameter dimension.
            clip_scaling (float, optional): Output scaling for decoder. Default 0.2.
            cano_pose_type (int, optional): Canonical pose type. Default 0.
            decoder_mlp (bool, optional): Use MLP in decoder cross-attention. Default False.
            skip_decoder (bool, optional): Whether to skip decoder and cross-attn layers. Default False.
            fix_opacity (bool, optional): Fix opacity during training. Default False.
            fix_rotation (bool, optional): Fix rotation during training. Default False.
            decode_with_extra_info (dict or None, optional): Provide extra info to decoder. Default None.
            gradient_checkpointing (bool, optional): Enable gradient checkpointing. Default False.
            apply_pose_blendshape (bool, optional): Apply pose blendshape. Default False.
            dense_sample_pts (int, optional): Dense sample points for mesh/voxel. Default 40000.
            gs_deform_scale (float, optional): Deformation scale for Gaussian Splatting. Default 0.005.
            render_features (bool, optional): Output additional features in renderer. Default False.
        """

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.smpl_type = smpl_type
        assert self.smpl_type in [
            "smplx_skirt",
            "smplx_voxel",
            "smplx_diffused_voxel",
            "mesh_voxel",
            "mesh_smpl_voxel",
            "mesh_smpl_flame_voxel",
        ]

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        self.render_features = render_features

        # Initialize SMPLX model based on type
        smplx_models = {
            "smplx_voxel": SMPLXVoxelSkinning,
            "smplx_diffused_voxel": SMPLXDiffusedVoxelSkinning,
        }

        model_kwargs = {
            "human_model_path": human_model_path,
            "gender": "neutral",
            "subdivide_num": subdivide_num,
            "shape_param_dim": shape_param_dim,
            "expr_param_dim": expr_param_dim,
            "cano_pose_type": cano_pose_type,
            "apply_pose_blendshape": apply_pose_blendshape,
        }
        if self.smpl_type in [
            "smplx_skirt",
            "smplx_voxel",
            "smplx_diffused_voxel",
            "mesh_voxel",
            "mesh_smpl_voxel",
            "mesh_smpl_flame_voxel",
        ]:
            model_kwargs["dense_sample_points"] = dense_sample_pts
        if self.smpl_type in ["smplx_skirt", "smplx_diffused_voxel"]:
            model_kwargs["voxel_weights_path"] = (
                "./pretrained_models/voxel_grid/cano_1_volume.npz"
            )

        self.smplx_model = smplx_models[self.smpl_type](**model_kwargs)

        if not self.skip_decoder:
            self.pcl_embed = PointEmbed(dim=query_dim)
            self.decoder_cross_attn = DecoderCrossAttn(
                query_dim=query_dim,
                context_dim=feat_dim,
                num_heads=1,
                mlp=decoder_mlp,
                decode_with_extra_info=decode_with_extra_info,
            )

        self.mlp_network_config = mlp_network_config

        # using to mapping transformer decode feature to regression features. as decode feature is processed by NormLayer.
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

    def forward_single_view(
        self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
    ):
        # This function renders a single view of a given GaussianModel using a specified camera and background color.
        # Args:
        #     gs (GaussianModel): The Gaussian model to be rendered.
        #     viewpoint_camera (Camera): Camera object describing the viewpoint for rendering.
        #     background_color (Optional[Float[Tensor, "3"]]): The background color for rendering. If None, defaults may be used.
        #     ret_mask (bool, optional): Whether to return a mask along with the rendered result. Defaults to True.
        # Returns:
        #     Output of the rasterizer and any computed masks, as a dictionary or tensor depending on the implementation.

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
        }

        return ret

    def _transform_points(
        self,
        smplx_data: Dict[str, Tensor],
        query_points: Tensor,
        offset_xyz: Tensor,
        device: torch.device,
        mesh_meta: dict,
    ) -> Dict[str, Tensor]:
        """
        Transforms query points and their predicted offsets from canonical (neutral) space
        to posed space coordinates using the SMPL-X model and transformation matrices.

        Args:
            smplx_data (Dict[str, Tensor]): Dictionary containing SMPL-X model data, including
                pose parameters and transformation matrices.
            query_points (Tensor): Query points in canonical (neutral) space. Shape: [N, 3].
            offset_xyz (Tensor): Predicted per-point offsets to apply to query_points. Shape: [N, 3].
            device (torch.device): Target device for computation.
            mesh_meta (dict): Metadata for mesh partitioning/region masking.

        Returns:
            Dict[str, Tensor]: Dictionary containing transformed coordinates and related data.
        """

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                (query_points + offset_xyz)
                .unsqueeze(0)
                .expand(smplx_data["body_pose"].shape[0], -1, -1)
            )
            transform_mat = (
                smplx_data["transform_mat_neutral_pose"]
                .unsqueeze(0)
                .expand(smplx_data["body_pose"].shape[0], -1, -1, -1)
            )

            points = {
                "neutral_coords": mean_3d,
                "transform_mat_to_null_pose": transform_mat,
                "mesh_meta": mesh_meta,
            }

            return self.smplx_model.transform_to_posed_verts_from_neutral_pose(
                points, smplx_data, device=device
            )

    def _compute_rotations(
        self,
        transform_matrix: Tensor,
        rotation_neutral: Tensor,
        device: torch.device,
        mesh_meta: dict,
    ) -> Tensor:

        # Computes the rotation quaternions for transforming points from neutral pose to posed space.
        #
        # Args:
        #     transform_matrix (Tensor): Transformation matrices from neutral to posed space. Shape: [B, N, 4, 4].
        #     rotation_neutral (Tensor): Neutral-space per-point quaternions. Shape: [N, 4].
        #     device (torch.device): Device on which computation is performed.
        #     mesh_meta (dict): Mesh region metadata including constraint information.
        #
        # Returns:
        #     Tensor: Combined quaternions (posed-space) for each point. Shape: [B, N, 4].

        transform_rotation = transform_matrix[:, :, :3, :3]
        rigid_rotation = F.normalize(matrix_to_quaternion(transform_rotation), dim=-1)

        return quaternion_multiply(
            rigid_rotation,
            rotation_neutral.unsqueeze(0).expand(transform_matrix.shape[0], -1, -1),
        )

    def _create_gaussian_models(
        self,
        posed_coords: Tensor,
        gs_attr: GaussianAppOutput,
        rotations: Tensor,
        num_views: int,
    ) -> Tuple[List, List]:
        """
        Constructs lists of GaussianModel instances for each view, given posed coordinates, Gaussian attributes, rotations, and number of views.

        Args:
            posed_coords (Tensor): Tensor of posed coordinates for each view. Shape: [num_views, N, 3].
            gs_attr (GaussianAppOutput): Output containing Gaussian attributes such as opacity, scaling, and sh coefficients.
            rotations (Tensor): Rotation quaternions per view. Shape: [num_views, N, 4].
            num_views (int): Number of views.

        Returns:
            Tuple[List[GaussianModel], List[GaussianModel]]:
                - gs_list: List of GaussianModel for each posed view (not including canonical as last).
                - cano_gs_list: List with a single GaussianModel for the canonical (last) view.
        """

        gs_list, cano_gs_list = [], []

        for i in range(num_views):
            gs = GaussianModel(
                xyz=posed_coords[i],
                opacity=gs_attr.opacity,
                rotation=rotations[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                use_rgb=self.gs_net.use_rgb,
            )
            (cano_gs_list if i == num_views - 1 else gs_list).append(gs)

        return gs_list, cano_gs_list

    def animate_gs_model(
        self,
        gs_attr: GaussianAppOutput,
        query_points,
        smplx_data,
        debug=False,
        mesh_meta=None,
    ):
        """
        Animates the Gaussian Splatting (GS) model by transforming canonical (neutral) points and attributes into the posed space using SMPL-X model deformations.

        Args:
            gs_attr (GaussianAppOutput): Gaussian attribute output for canonical points, including offset positions, opacity, rotation, scaling, and appearance.
            query_points (Tensor): Canonical query point coordinates, shape (N, 3).
            smplx_data (dict): SMPL-X input data for the current animation frame, including body pose, shape, etc.
            debug (bool, optional): If True, use debug mode (e.g., force all opacities to 1.0, use identity rotations). Default: False.
            mesh_meta (dict, optional): Additional mesh region meta-information (e.g., for constraints). Default: None.

        Returns:
            Tuple[List[GaussianModel], List[GaussianModel]]:
                - gs_list: List of posed-space GaussianModel instances (one per camera/view except canonical view).
                - cano_gs_list: List of canonical-space GaussianModel instances (last view is canonical).
        """

        device = gs_attr.offset_xyz.device

        if debug:
            N = gs_attr.offset_xyz.shape[0]
            gs_attr.xyz = torch.zeros_like(gs_attr.offset_xyz)
            gs_attr.opacity = torch.ones((N, 1), device=device)
            gs_attr.rotation = matrix_to_quaternion(
                torch.eye(3, device=device).expand(N, 3, 3)
            )

        # build cano_dependent_pose
        merge_smplx_data = self._prepare_smplx_data(smplx_data)

        posed_points = self._transform_points(
            merge_smplx_data, query_points, gs_attr.offset_xyz, device, mesh_meta
        )
        rotation_pose_verts = self._compute_rotations(
            posed_points["transform_mat_posed"],
            gs_attr.rotation,
            device,
            posed_points["mesh_meta"],
        )

        return self._create_gaussian_models(
            posed_points["posed_coords"],
            gs_attr,
            rotation_pose_verts,
            merge_smplx_data["body_pose"].shape[0],
        )

    def forward_animate_gs(
        self,
        gs_attr_list: List[GaussianAppOutput],
        query_points: Dict[str, Tensor],
        smplx_data: Dict[str, Tensor],
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        df_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Animate and render Gaussian Splatting (GS) models for a batch of frames/views.

        Args:
            gs_attr_list (List[GaussianAppOutput]):
                List of Gaussian attribute outputs, one per batch item. Each element contains predicted Gaussian parameters such as offset positions, opacity, rotation, scaling, and appearance for canonical points.
            query_points (Dict[str, Tensor]):
                Dictionary containing query information, must include:
                    - 'neutral_coords': Tensor of canonical coordinate positions, shape [B, N, 3].
                    - 'mesh_meta': (Optional) Dictionary with mesh region meta-info as required by the skinning/posing models.
            smplx_data (Dict[str, Tensor]):
                Dictionary containing per-batch SMPL-X (or similar model) data for the current animation/frame. Used for pose and shape transformation.
            c2w (Float[Tensor, "B Nv 4 4"]):
                Camera-to-world matrices for the views to render (B: batch, Nv: number of views).
            intrinsic (Float[Tensor, "B Nv 4 4"]):
                Intrinsic camera matrices, shape matches c2w.
            height (int):
                Height of output render images (in pixels).
            width (int):
                Width of output render images (in pixels).
            background_color (Optional[Float[Tensor, "B Nv 3"]], default=None):
                Optional RGB background color per batch/view.
            debug (bool, optional):
                If True, enables debug behavior (e.g., simplifies opacities, disables poses, saves debug visualizations).
            df_data (Optional[Dict], default=None):
                Optional dictionary of additional deformation/feature data.
            **kwargs:
                Additional keyword arguments. Can optionally contain 'features' key for render feature maps.

        Returns:
            Dict[str, Tensor]:
                Dictionary of rendered outputs, including:
                - Main render outputs (images, masks, etc.), canonically organized and batched.
                - '3dgs': List of all canonical-space GaussianModel instances for the batch.
        """

        batch_size = len(gs_attr_list)
        out_list, cano_out_list = [], []
        query_points_pos = query_points["neutral_coords"]
        mesh_meta = query_points["mesh_meta"]

        gs_list = []
        for b in range(batch_size):
            # Animate GS models
            anim_models, cano_models = self.animate_gs_model(
                gs_attr_list[b],
                query_points_pos[b],
                self._get_single_batch_data(smplx_data, b),
                debug=debug,
                mesh_meta=mesh_meta,
            )

            gs_list.extend(cano_models)

            features = (
                kwargs["features"][b] if kwargs.get("features") is not None else None
            )

            # Render animated views
            out_list.append(
                self._render_views(
                    anim_models[: c2w.shape[1]],  # Only keep requested views
                    c2w[b],
                    intrinsic[b],
                    height,
                    width,
                    background_color[b] if background_color is not None else None,
                    debug,
                    features=features,
                )
            )

            # Render canonical view
            cano_out_list.append(
                self._render_canonical(
                    cano_models,
                    c2w[b],
                    intrinsic[b],
                    background_color[b] if background_color is not None else None,
                    debug,
                )
            )

        results = self._combine_outputs(out_list, cano_out_list)
        results["3dgs"] = gs_list

        return results

    def forward_gs(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: dict,
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Forward pass to obtain per-point Gaussian attributes.

        Args:
            gs_hidden_features (Float[Tensor, "B Np Cp"]): Gaussian hidden features for each batch.
            query_points (dict): Dictionary containing query points information, such as 'neutral_coords' and 'mesh_meta'.
            smplx_data: SMPL-X data per batch, containing pose, shape, and other model parameters.
            additional_features (Optional[dict], optional): Additional features (like per-point or per-image features). Default is None.
            debug (bool, optional): If True, enables debug mode. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            gs_attr_list: List of dictionaries, each with Gaussian attributes for the batch.
            query_points: Updated query_points dict.
            smplx_data: Updated smplx_data dict (may include additional transforms).
        """

        batch_size = gs_hidden_features.shape[0]

        # obtain gs_features embedding, cur points position, and also smplx params
        query_gs_features, query_points_pos, smplx_data = self.query_latent_feat(
            query_points["neutral_coords"],
            smplx_data,
            gs_hidden_features,
            additional_features,
        )

        # TODO support batch mesh_meta
        mesh_meta = query_points["mesh_meta"]

        gs_attr_list = []
        for b in range(batch_size):
            if isinstance(query_gs_features, dict):
                gs_attr = self.forward_gs_attr(
                    query_gs_features["coarse"][b],
                    query_points_pos[b],
                    None,
                    debug,
                    x_fine=query_gs_features["fine"][b],
                    mesh_meta=mesh_meta,
                )
            else:
                gs_attr = self.forward_gs_attr(
                    query_gs_features[b],
                    query_points_pos[b],
                    None,
                    debug,
                    mesh_meta=mesh_meta,
                )
            gs_attr_list.append(gs_attr)

        return gs_attr_list, query_points, smplx_data

    def forward(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: dict,
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Forward pass for the GS renderer.

        Args:
            gs_hidden_features (Float[Tensor, "B Np Cp"]): Latent features representing the 3D object (batch, num_points, channels).
            query_points (dict): Dictionary containing query points and related metadata for rendering.
            smplx_data: Dictionary containing SMPL-X parameters needed to query canonical/posed coordinates.
            c2w (Float[Tensor, "B Nv 4 4"]): Camera-to-world transformation matrices (batch, num_views, 4, 4).
            intrinsic (Float[Tensor, "B Nv 4 4"]): Intrinsic camera parameter matrices (batch, num_views, 4, 4).
            height (int): Image height for rendering.
            width (int): Image width for rendering.
            additional_features (Optional[Float[Tensor, "B C H W"]], optional): Extra features to be used (default: None).
            background_color (Optional[Float[Tensor, "B Nv 3"]], optional): Background color per view (default: None).
            debug (bool, optional): Whether to enable debug visualization (default: False).
            **kwargs: Additional arguments for downstream rendering or feature flow.

        Returns:
            out (dict): Dictionary containing rendered outputs, such as RGB images, masks, and attribute lists.
        """

        # need shape_params of smplx_data to get querty points and get "transform_mat_neutral_pose"
        # only forward gs params

        gs_attr_list, query_points, smplx_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            smplx_data=smplx_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            smplx_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs["df_data"],
            features=gs_hidden_features if kwargs["is_refine"] else None,
        )

        out["gs_attr"] = gs_attr_list
        out["mesh_meta"] = query_points["mesh_meta"]

        return out

    def inference_cano_gs(
        self,
        gs_attr_list: List[GaussianAppOutput],
        query_points: Dict[str, Tensor],
        smplx_data: Dict[str, Tensor],
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        df_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Inference function to obtain canonical-space GaussianModel instances.

        Args:
            gs_attr_list (List[GaussianAppOutput]):
                List of Gaussian attribute outputs, one per batch item. Each element contains predicted Gaussian parameters such as offset positions, opacity, rotation, scaling, and appearance for canonical points.
            query_points (Dict[str, Tensor]):
                Dictionary containing query information, must include:
                    - 'neutral_coords': Tensor of canonical coordinate positions, shape [B, N, 3].
                    - 'mesh_meta': (Optional) Dictionary with mesh region meta-info as required by the skinning/posing models.
            smplx_data (Dict[str, Tensor]):
                Dictionary containing per-batch SMPL-X (or similar model) data for the current animation/frame. Used for pose and shape transformation.
            c2w (Float[Tensor, "B Nv 4 4"]):
                Camera-to-world matrices for the views to render (B: batch, Nv: number of views).
            intrinsic (Float[Tensor, "B Nv 4 4"]):
                Intrinsic camera matrices, shape matches c2w.
            height (int):
                Height of output render images (in pixels).
            width (int):
                Width of output render images (in pixels).
            background_color (Optional[Float[Tensor, "B Nv 3"]], default=None):
                Optional RGB background color per batch/view.
            debug (bool, optional):
                If True, enables debug behavior (e.g., simplifies opacities, disables poses, saves debug visualizations).
            df_data (Optional[Dict], default=None):
                Optional dictionary of additional deformation/feature data.
            **kwargs:
                Additional keyword arguments.

        Returns:
            List[GaussianModel]:
                List of canonical-space GaussianModel instances for the batch (from each canonical view).
        """

        batch_size = len(gs_attr_list)
        out_list, cano_out_list = [], []
        query_points_pos = query_points["neutral_coords"]
        mesh_meta = query_points["mesh_meta"]

        return_posed_gaussians = bool(kwargs.pop("return_posed_gaussians", False))

        gs_list = []
        for b in range(batch_size):
            # Animate GS models
            anim_models, cano_models = self.animate_gs_model(
                gs_attr_list[b],
                query_points_pos[b],
                self._get_single_batch_data(smplx_data, b),
                debug=debug,
                mesh_meta=mesh_meta,
            )

            # `_prepare_smplx_data` stacks [actual_pose, template_row]; posed coords are anim_models,
            # the last row is a deliberate canonical-style template (cano_models).
            if return_posed_gaussians and anim_models:
                gs_list.extend(anim_models)
            else:
                gs_list.extend(cano_models)

        return gs_list

    ############################################ Auxiliary  Function ########################################
    def _prepare_smplx_data(self, smplx_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cano_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
        ]

        merge_data = {
            k: torch.cat([smplx_data[k], torch.zeros_like(smplx_data[k][:1])], dim=0)
            for k in cano_keys
        }

        # Special handling for body pose
        if "body_pose" in merge_data:
            # leg
            merge_data["body_pose"][-1, 0, -1] = math.pi / 12
            merge_data["body_pose"][-1, 1, -1] = -math.pi / 12

            # hands
            merge_data["body_pose"][-1, 15, -1] = -math.pi / 6
            merge_data["body_pose"][-1, 16, -1] = math.pi / 6

        merge_data["betas"] = smplx_data["betas"]
        merge_data["transform_mat_neutral_pose"] = smplx_data[
            "transform_mat_neutral_pose"
        ]

        return merge_data

    def get_query_points(self, query_pts_path, smplx_data, device):

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                query_points = self.smplx_model.get_query_points(
                    query_pts_path, smplx_data, device=device
                )
                transform_mat_neutral_pose = query_points["transform_mat_to_null_pose"]

        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 55, 4, 4]
        )
        return query_points, smplx_data

    def get_single_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[
                bidx
            ]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[
                    bidx : bidx + 1
                ]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch

    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}
        for k, v in smpl_data.items():
            assert v.shape[0] == 1
            if (
                k == "betas"
                or (k == "joint_offset")
                or (k == "face_offset")
                or (k == "transform_mat_neutral_pose")
            ):
                smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
            else:
                smpl_data_single_view[k] = v[
                    :, vidx : vidx + 1
                ]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
        return smpl_data_single_view

    def decoder_cross_attn_wrapper(self, pcl_embed, latent_feat, extra_info):
        gs_feats = self.decoder_cross_attn(
            pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
        )
        return gs_feats

    def query_latent_feat(
        self,
        positions: Float[Tensor, "*B N1 3"],
        smplx_data,
        latent_feat: Float[Tensor, "*B N2 C"],
        extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, smplx_data = self.get_query_points(smplx_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)

            gs_feats = self.decoder_cross_attn_wrapper(
                pcl_embed, latent_feat, extra_info
            )

        return gs_feats, positions, smplx_data

    def _combine_outputs(
        self, out_list: List[Dict], cano_out_list: List[Dict]
    ) -> Dict[str, Tensor]:

        batch_size = len(out_list)

        combined = defaultdict(list)
        for out in out_list:
            # Collect render outputs
            for render_item in out["render"]:
                for k, v in render_item.items():
                    combined[k].append(v)

        # Reshape and permute tensors
        result = {
            k: torch.stack(v).view(batch_size, -1, *v[0].shape).permute(0, 1, 4, 2, 3)
            for k, v in combined.items()
            if torch.stack(v).dim() >= 4
        }

        return result

    def _get_single_batch_data(
        self, data: Dict[str, Tensor], bidx: int
    ) -> Dict[str, Tensor]:
        return {
            k: (
                v[bidx : bidx + 1]
                if k in ["betas", "joint_offset", "face_offset"]
                else v[bidx]
            )
            for k, v in data.items()
        }

    def _debug_save_image(self, tensor: Tensor, prefix: str = ""):
        import cv2

        img = (tensor.detach().cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
        cv2.imwrite(f"{prefix}debug.png" if prefix else "debug.png", img)

    def _render_views(
        self,
        gs_list: List[GaussianModel],
        c2w: Tensor,
        intrinsic: Tensor,
        height: int,
        width: int,
        bg_color: Optional[Tensor],
        debug: bool,
        **kwargs,
    ) -> Dict[str, Tensor]:

        # obtain device
        self.device = gs_list[0].xyz.device

        gs_mask_list = [gs.CloneMaskGaussian(self.gs_deform_scale) for gs in gs_list]
        results = defaultdict(list)

        for v_idx, (gs, gs_mask) in enumerate(zip(gs_list, gs_mask_list)):

            if self.render_features:
                render_features = kwargs["features"]
            else:
                render_features = None

            camera = Camera.from_c2w(c2w[v_idx], intrinsic[v_idx], height, width)

            results["render"].append(
                self.forward_single_view(
                    gs, camera, bg_color[v_idx], features=render_features
                )
            )
            results["mask"].append(
                self.forward_single_view(gs_mask, camera, bg_color[v_idx])["comp_mask"]
            )

            if debug and v_idx == 0:
                self._debug_save_image(results["render"][-1]["comp_rgb"])

        return results

    def _render_canonical(
        self,
        cano_models: List[GaussianModel],
        c2w: Tensor,
        intrinsic: Tensor,
        bg_color: Optional[Tensor],
        debug: bool,
    ) -> Dict[str, Tensor]:
        cano_results = defaultdict(list)
        for degree, gs in zip(
            [0, 90, 180, 270], self._rotate_canonical(cano_models[0])
        ):

            camera = Camera.from_c2w_center_modfied(c2w[0], intrinsic[0], 768, 768)
            view_result = self.forward_single_view(gs, camera, bg_color[0])
            cano_results["render"].append(view_result)

            if debug:
                self._debug_save_image(view_result["comp_rgb"], f"cano_{degree}")

        return cano_results

    def _rotate_canonical(self, gs: GaussianModel) -> List[GaussianModel]:
        rotated = []
        for degree in [0, 90, 180, 270]:
            gs_copy = gs.clone()
            _R = torch.eye(3).to(gs.xyz)
            _R[-1, -1] *= -1
            _R[1, 1] *= -1
            self_R = torch.from_numpy(generate_rotation_matrix_y(degree)).to(_R.device)
            R = self_R @ _R

            gs_copy.xyz = (R @ gs_copy.xyz.T).T
            gs_copy.xyz -= (aabb(gs_copy.xyz)[0] + aabb(gs_copy.xyz)[1]) / 2
            gs_copy.rotation = quaternion_multiply(
                matrix_to_quaternion(R), gs_copy.rotation
            )
            gs_copy.xyz[..., -1] += 2.5
            rotated.append(gs_copy)
        return rotated

    ############################################ Auxiliary  Function ########################################
