# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Gradio app.py for LHM++

import argparse
import glob
import logging
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import ffmpegio
import gradio as gr
import imageio.v3 as iio
import numpy as np
import spaces
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image

torch._dynamo.config.disable = True

# Suppress specific FutureWarning emitted by spconv regarding torch.cuda.amp.custom_bwd
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=(
        r"`torch\.cuda\.amp\.custom_bwd\(args\.\.\.\)` is deprecated\. "
        r"Please use `torch\.amp\.custom_bwd\(args\.\.\., device_type='cuda'\)` instead\."
    ),
    module=r"spconv\.pytorch\.functional",
)

from core.datasets.data_utils import SrcImagePipeline
from core.utils.app_utils import (
    get_motion_information,
    prepare_examples_ordered,
    prepare_input_and_output,
    split_image,
)
from core.utils.model_card import (
    GS_RENDER_SUPPORTED_MODEL_NAMES,
    MODEL_CONFIG,
    model_supports_gs_render,
)
from core.utils.model_download_utils import AutoModelQuery
from engine.pose_estimation.pose_estimator import PoseEstimator
from scripts.download_motion_video import motion_video_check
from scripts.inference.app_inference import (
    build_app_model,
    inference_results,
    parse_app_configs,
)
from scripts.inference.utils import (
    assert_input_image,
    easy_memory_manager,
    get_image_base64,
    prepare_working_dir,
)

logger = logging.getLogger(__name__)

# ==================== Configuration Constants ====================
MOTION_SIZE: int = 120
MV_IMAGES_THRESHOLD: int = 8
NUM_VIEWS: int = 2
DEFAULT_FPS: int = 30
DEFAULT_VIDEO_CODEC: str = "libx264"
DEFAULT_PIXEL_FORMAT: str = "yuv420p"
DEFAULT_VIDEO_BITRATE: str = "10M"
MACRO_BLOCK_SIZE: int = 16

APP_RENDER_BACKEND_MAP = {
    "neural_render": "neural",
    "gs_render": "gs",
}


def resolve_cli_render_backend(
    args: argparse.Namespace, model_name: str
) -> Tuple[str, bool]:
    """Resolve initial Gradio render mode from ``--neural`` / ``--gs`` and model capability.

    Returns:
        (initial_radio_value, gs_render_supported_for_this_model).
    """
    gs_ok = model_supports_gs_render(model_name)
    if getattr(args, "gs", False):
        if gs_ok:
            return "gs_render", True
        logger.warning(
            "当前模型 %r 不支持 gs_render；当前白名单为：%s；已强制使用 neural_render。",
            model_name,
            ", ".join(GS_RENDER_SUPPORTED_MODEL_NAMES),
        )
        return "neural_render", gs_ok
    return "neural_render", gs_ok


def get_motion_video_fps(
    video_params: Union[str, Dict[str, Any]], default: int = DEFAULT_FPS
) -> int:
    """Get FPS from samurai_visualize.mp4 in the motion video directory.

    The render FPS is determined by the motion video's samurai_visualize.mp4
    to ensure temporal consistency with the source motion.

    Args:
        video_params: Path to the motion video (e.g. girl.mp4) or Gradio video
            value. The samurai_visualize.mp4 is expected in the same directory.
        default: Fallback FPS when samurai_visualize.mp4 is missing or invalid.

    Returns:
        FPS as int (rounded from video's frame_rate).
    """
    try:
        path = video_params
        if isinstance(video_params, dict):
            path = video_params.get("path") or video_params.get("name") or ""
        if not path or not isinstance(path, str):
            return default
        vst = ffmpegio.probe.video_streams_basic(path)
        if not vst or "frame_rate" not in vst[0]:
            return default
        fr = vst[0]["frame_rate"]
        if hasattr(fr, "numerator") and hasattr(fr, "denominator"):
            fps_val = fr.numerator / fr.denominator if fr.denominator else default
        else:
            fps_val = float(fr) if fr else default
        return max(1, round(fps_val))
    except Exception:
        return default


def demo_lhmpp(
    lhmpp: Optional[torch.nn.Module],
    pose_estimator: Optional[PoseEstimator],
    dataset_pipeline: SrcImagePipeline,
    cfg: DictConfig,
    initial_render_backend: str = "neural_render",
    gs_render_available: bool = True,
) -> None:
    """Initialize and launch the Gradio demo interface."""

    render_backend_choices: List[str] = (
        ["neural_render", "gs_render"]
        if gs_render_available
        else ["neural_render"]
    )
    if initial_render_backend not in render_backend_choices:
        initial_render_backend = "neural_render"

    @spaces.GPU(duration=200)
    def core_fn(
        image: Any,
        video: Any,
        ref_view: int,
        video_params: Union[str, Dict[str, Any]],
        visualized_center: bool,
        working_dir: Any,
        motion_size: int,
        render_fps: int,
        render_backend: str,
    ) -> Tuple[Any, Optional[str], None, List[Image.Image]]:
        """Core inference function for the Gradio interface."""
        import gradio as gr

        device = "cuda"
        dtype = torch.float32

        try:
            (
                imgs,
                input_imgs_path,
                motion_path,
                dump_image_dir,
                dump_video_path,
            ) = prepare_input_and_output(
                image, video, ref_view, video_params, working_dir, dataset_pipeline, cfg
            )
        except ValueError as e:
            raise gr.Error(str(e)) from e

        motion_name, motion_seqs = get_motion_information(motion_path, cfg, motion_size)
        video_size = len(motion_seqs["motion_seqs"])

        if pose_estimator is not None:
            with torch.no_grad():
                with easy_memory_manager(pose_estimator, device="cuda"):
                    shape_pose = pose_estimator(imgs[0])
            assert shape_pose.is_full_body, (
                f"The input image is illegal, {shape_pose.msg}"
            )

        img_np = np.stack(imgs) / 255.0
        ref_imgs_tensor = (
            torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
        )
        smplx_params = motion_seqs["smplx_params"]
        if pose_estimator is not None:
            smplx_params["betas"] = torch.tensor(
                shape_pose.beta, dtype=dtype, device=device
            ).unsqueeze(0)
        else:
            smplx_params = smplx_params.copy()

        backend_choice = render_backend
        if backend_choice == "gs_render" and not gs_render_available:
            logger.warning(
                "当前模型不支持 gs_render，已改用 neural_render。"
            )
            backend_choice = "neural_render"

        rgbs = inference_results(
            lhmpp,
            ref_imgs_tensor,
            smplx_params,
            motion_seqs,
            video_size=video_size,
            visualized_center=visualized_center,
            device=device,
            infer_output_renderer=APP_RENDER_BACKEND_MAP[backend_choice],
        )

        render_fps_actual = get_motion_video_fps(video_params, default=render_fps)

        iio.imwrite(
            dump_video_path,
            rgbs,
            fps=render_fps_actual,
            codec=DEFAULT_VIDEO_CODEC,
            pixelformat=DEFAULT_PIXEL_FORMAT,
            bitrate=DEFAULT_VIDEO_BITRATE,
            macro_block_size=MACRO_BLOCK_SIZE,
        )
        print(f"Saved video to {dump_video_path}")
        # Convert selected imgs to PIL for display; preserve original gallery
        selected_pils = [
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)) for img in imgs
        ]
        return image, dump_video_path, None, selected_pils

    gallery_css = """
    .gradio-container { max-width: 95vw !important; }
    .image-prompt-gallery { min-height: 120px !important; }
    .image-prompt-gallery img { object-fit: contain !important; }
    #selected_input_gallery { min-height: 240px !important; overflow: visible !important; }
    #selected_input_gallery .thumbnail-item img,
    #selected_input_gallery img { object-fit: contain !important; }
    #image_examples, #image_examples .wrap, #image_examples .contain,
    #image_examples .block, #image_examples > div {
        max-width: none !important; width: 100% !important; box-sizing: border-box !important;
    }
    #image_examples { width: 100% !important; }
    #image_examples .wrap, #image_examples .contain,
    #image_examples > div > div, #image_examples [class*="example"] {
        width: 100% !important; display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 12px !important;
    }
    #image_examples .example, #image_examples [class*="example"] > * {
        width: 100% !important; min-width: 0 !important; min-height: 160px !important;
    }
    #image_examples img {
        width: 100% !important; height: auto !important;
        max-height: 200px !important; object-fit: contain !important;
    }
    .lhmpp-header { text-align: center !important; margin-bottom: 16px !important; }
    .lhmpp-header .header-logo { height: 140px !important; display: block !important; margin: 0 auto 12px auto !important; }
    .lhmpp-header .publication-title { margin-bottom: 0 !important; font-size: 2.2rem !important; }
    .lhmpp-badges { display: flex !important; flex-wrap: wrap !important; justify-content: center !important; gap: 8px !important; }
    .lhmpp-badges a img { height: 32px !important; }
    .video-row { gap: 16px !important; }
    .video-row .video-wrapper { min-width: 0 !important; flex: 1 !important; }
    .video-row video, .video-row .gr-video { width: 100% !important; max-width: 100% !important; }
    .section-label { font-size: 0.95em !important; margin-bottom: 6px !important; }
    """
    with gr.Blocks(
        title="LHM++",
        analytics_enabled=False,
        css=gallery_css,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="purple"),
    ) as demo:
        logo_url = "./assets/LHM++_logo.png"
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(
            f"""
            <div class="lhmpp-header" style="margin-bottom: 12px;">
                <img src="{logo_base64}" class="header-logo" alt="LHM++ Logo"/>
                <h2 class="publication-title"><b style="color: #7B2FBE">L</b><b style="color: #9B59B6">H</b><b style="color: #C084FC">M</b><b style="background: linear-gradient(135deg, #C084FC, #E879F9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.1em;">++</b>: <b style="color: #6D28D9">A</b>n <b style="color: #7C3AED">E</b>fficient <b style="color: #8B5CF6">L</b>arge <b style="color: #9B59B6">H</b>uman <b style="color: #A78BFA">R</b>econstruction <b style="color: #B794F4">M</b>odel for <b style="color: #C084FC">P</b>ose-free <b style="color: #D6BCFA">I</b>mages to <b style="color: #E879F9">3D</b></h2>
            </div>
            <div class="lhmpp-badges" style="margin-bottom: 16px;">
                <a title="Website" href="https://lingtengqiu.github.io/LHM++/" target="_blank" rel="noopener noreferrer">
                    <img src="https://www.obukhov.ai/img/badges/badge-website.svg" alt="Website">
                </a>
                <a title="Read Paper" href="https://arxiv.org/pdf/2506.13766v2" target="_blank" rel="noopener noreferrer">
                    <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg" alt="Paper">
                </a>
                <a title="GitHub" href="https://github.com/aigc3d/LHM-plusplus" target="_blank" rel="noopener noreferrer">
                    <img src="https://img.shields.io/github/stars/aigc3d/LHM-plusplus?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="GitHub">
                </a>
                <a title="Video" href="https://www.youtube.com/watch?v=Nipf3jdSi34" target="_blank" rel="noopener noreferrer">
                    <img src="https://img.shields.io/badge/YouTube-QiuLingteng-red?logo=youtube" alt="YouTube">
                </a>
            </div>
            """
        )

        gr.HTML(
            """<p style="color: #c00; font-size: 0.9em; margin-bottom: 12px; font-weight: 500;">
            ⚠️ Human images only. Motion video max 1000 frames. Default model LHMPP-700M-PixelShuffle requires ≥8 GB GPU. Videos may exhibit white occlusion artifacts due to cropping with the original mask, and some motion estimation may have slight jittering.
            </p>"""
        )

        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", scale=3, min_width=320):
                with gr.Accordion(label="⚙️ Input Settings", open=False):
                    with gr.Row():
                        num_images = gr.Slider(
                            label="Ref. Images",
                            minimum=1,
                            maximum=8,
                            value=8,
                            step=1,
                            elem_id="num_input_images",
                            interactive=True,
                            scale=1,
                        )
                        motion_size = gr.Slider(
                            label="Motion Frames",
                            minimum=30,
                            maximum=1000,
                            value=MOTION_SIZE,
                            step=2,
                            elem_id="motion_size",
                            interactive=True,
                            scale=1,
                        )
                    with gr.Row():
                        render_fps = gr.Slider(
                            label="FPS",
                            minimum=10,
                            maximum=120,
                            value=DEFAULT_FPS,
                            step=1,
                            elem_id="render_fps",
                            interactive=True,
                            scale=1,
                        )
                        center_visualizen = gr.Checkbox(
                            label="Center Crop", value=False, scale=1
                        )
                        render_backend = gr.Radio(
                            label="Output Renderer",
                            choices=render_backend_choices,
                            value=initial_render_backend,
                            elem_id="render_backend",
                            scale=2,
                        )

                with gr.Tabs(elem_id="core_input_image"):
                    with gr.Tab(label="Input Video or Images"):
                        image_prompt = gr.Image(
                            label="Image Prompt",
                            format="png",
                            visible=False,
                            image_mode="RGBA",
                            type="pil",
                            height=360,
                        )
                        default_imgs = glob.glob(
                            "./assets/example_multi_images/00000_yuliang_*.png"
                        )
                        default_imgs = sorted(default_imgs)[:MV_IMAGES_THRESHOLD]
                        default_imgs = [Image.open(img) for img in default_imgs]
                        multiimage_prompt = gr.Gallery(
                            label="Image Prompt",
                            format="png",
                            type="pil",
                            height=280,
                            columns=8,
                            value=default_imgs,
                            object_fit="contain",
                            elem_classes="image-prompt-gallery",
                        )
                        video_upload_input = gr.Video(
                            label="Or Input Video",
                            height=300,
                            interactive=True,
                            value=None,
                            elem_id="Or video input.",
                        )
                        gr.Markdown(
                            "Upload multi-view human images or a video.",
                            elem_classes="small-hint",
                        )

                _all_examples = prepare_examples_ordered()
                examples = gr.Examples(
                    examples=_all_examples,
                    inputs=[image_prompt],
                    fn=lambda img: split_image(img, NUM_VIEWS),
                    outputs=[multiimage_prompt],
                    run_on_click=True,
                    examples_per_page=8,
                    elem_id="image_examples",
                )
                submit = gr.Button(
                    "Generate", elem_id="core_generate", variant="primary", size="lg"
                )

            examples_video = os.listdir("./motion_video/")
            examples = [
                os.path.join("./motion_video/", ex, "samurai_visualize.mp4")
                for ex in examples_video
            ]
            examples = list(filter(lambda x: os.path.exists(x), examples))
            examples = sorted(examples)
            new_examples = []
            for ex in examples:
                video_basename = os.path.basename(os.path.dirname(ex))
                input_video = os.path.join(os.path.dirname(ex), video_basename + ".mp4")
                if not os.path.exists(input_video):
                    shutil.copyfile(ex, input_video)
                new_examples.append(input_video)

            with gr.Column(variant="panel", scale=5, min_width=400):
                gr.Markdown(
                    "**Selected Input** (used for inference):",
                    elem_classes="section-label",
                )
                selected_input_tag = gr.Gallery(
                    label="Selected Input Images (used for inference)",
                    format="png",
                    type="pil",
                    height=200,
                    columns=8,
                    value=[],
                    object_fit="contain",
                    show_label=False,
                    elem_id="selected_input_gallery",
                )
                gr.Markdown("**Motion & Output**", elem_classes="section-label")
                with gr.Row(variant="default", elem_classes="video-row"):
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Motion Video",
                            height=320,
                            interactive=False,
                            value=(
                                new_examples[1]
                                if len(new_examples) > 5
                                else (new_examples[0] if new_examples else None)
                            ),
                            autoplay=True,
                        )
                    with gr.Column(scale=2):
                        output_video = gr.Video(
                            label="Rendered Video",
                            format="mp4",
                            height=480,
                            autoplay=True,
                        )
                gr.Examples(
                    examples=new_examples,
                    inputs=[video_input],
                    examples_per_page=8,
                )

        working_dir = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[multiimage_prompt],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[
                multiimage_prompt,
                video_upload_input,
                num_images,
                video_input,
                center_visualizen,
                working_dir,
                motion_size,
                render_fps,
                render_backend,
            ],
            outputs=[
                multiimage_prompt,
                output_video,
                video_upload_input,
                selected_input_tag,
            ],
        )

        demo.queue()
        demo.launch(server_name="0.0.0.0")


def prior_model_check(save_dir: str = "./pretrained_models") -> None:
    """Check if prior models exist with valid targets; if not, download and link.

    Prior models include human_model_files, voxel_grid, arcface, BiRefNet, etc.
    - If human_model_files exists and target is valid: skip.
    - If symlink exists but target is missing (broken): remove symlink, download, re-link.
    - If not present: download and link.
    """
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


def get_parse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LHM-gradio: Large Animatable Human Model"
    )
    parser.add_argument(
        "--model_name",
        default="LHMPP-700M-PixelShuffle",
        type=str,
        choices=[
            "LHMPP-700M-PixelShuffle",
            "LHMPP-700M-SMPLX-FREE",
            "LHMPP-700M",
            "LHMPPS-700M",
        ],  # LHMPP-700MC coming soon
        help="Model name",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    render_mode = parser.add_mutually_exclusive_group()
    render_mode.add_argument(
        "--neural",
        action="store_true",
        help="Use neural_render (default if neither flag is set)",
    )
    render_mode.add_argument(
        "--gs",
        action="store_true",
        help="Prefer gs_render at startup (LHMPP-700M-PixelShuffle / LHMPP-700M-SMPLX-FREE; others fall back to neural_render)",
    )
    return parser.parse_args()


def launch_gradio_app() -> None:
    """Launch the Gradio application."""
    args = get_parse()
    model_name = args.model_name

    initial_render_backend, gs_render_available = resolve_cli_render_backend(
        args, model_name
    )

    prior_model_check(save_dir="./pretrained_models")
    motion_video_check(save_dir=".")

    # Get model path from AutoModelQuery (downloads from HuggingFace/ModelScope if needed)
    auto_query = AutoModelQuery(save_dir="./pretrained_models")
    model_path: str = auto_query.query(model_name)
    model_config: str = MODEL_CONFIG[model_name]
    model_cards: Dict[str, Dict[str, Any]] = {
        model_name: {
            "model_path": model_path,
            "model_config": model_config,
        }
    }

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    processing_list: List[Dict[str, Any]] = [
        dict(
            name="PadRatioWithScale",
            target_ratio=5 / 3,
            tgt_max_size_list=[840],
            val=True,
        ),
    ]

    accelerator = Accelerator()
    cfg, _ = parse_app_configs(model_cards)

    dataset_pipeline = SrcImagePipeline(*processing_list)

    if args.debug:
        lhmpp = None
        pose_estimator = None
    else:
        lhmpp = build_app_model(cfg)
        lhmpp.to("cuda")
        if cfg.get("use_smplx_shape_estimator", True):
            pose_estimator = PoseEstimator(
                "./pretrained_models/human_model_files/", device="cpu"
            )
            pose_estimator.device = "cuda"
        else:
            pose_estimator = None

    demo_lhmpp(
        lhmpp,
        pose_estimator,
        dataset_pipeline,
        cfg,
        initial_render_backend=initial_render_backend,
        gs_render_available=gs_render_available,
    )


if __name__ == "__main__":
    launch_gradio_app()