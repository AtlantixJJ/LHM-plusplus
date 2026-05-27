# -*- coding: utf-8 -*-
# @Function: LHM++ model registry for HuggingFace and ModelScope

ModelScope_Prior_MODEL_CARD = {
    "LHMPP": "Damo_XR_Lab/LHMPP-Prior",
}
HuggingFace_Prior_MODEL_CARD = {
    "LHMPP": "3DAIGC/LHMPP-Prior",
}
ModelScope_MODEL_CARD = {
    # SMPLX-FREE / ShapeHead + GS export: set repo id when published; until then use local checkpoint + --model_path
    "LHMPP-700M-PixelShuffle": "Damo_XR_Lab/LHMPP-700M-PixelShuffle",  # default; SMPLX-FREE + MLPPixelShuffle head + GS
    "LHMPP-700M-SMPLX-FREE": "Damo_XR_Lab/LHMPP-700M-SMPLX-FREE",
    "LHMPP-700M": "Damo_XR_Lab/LHMPP-700M",
    # "LHMPP-700MC": "Damo_XR_Lab/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "Damo_XR_Lab/LHMPPS-700M",
}

HuggingFace_MODEL_CARD = {
    # Publish repo id when releasing; otherwise use local weights (e.g. test_app_case.py --model_path).
    "LHMPP-700M-PixelShuffle": "3DAIGC/LHMPP-700M-PixelShuffle",  # default; SMPLX-FREE + MLPPixelShuffle head + GS
    "LHMPP-700M-SMPLX-FREE": "3DAIGC/LHMPP-700M-SMPLX-FREE",
    "LHMPP-700M": "3DAIGC/LHMPP-700M",
    # "LHMPP-700MC": "3DAIGC/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "3DAIGC/LHMPPS-700M",
}

MODEL_CONFIG = {
    "LHMPP-700M-PixelShuffle": "./configs/train/LHMPP-pixelshuffle.yaml",
    "LHMPP-700M-SMPLX-FREE": "./configs/LHMPP-anyview-SMPLX-FREE.yaml",
    "LHMPP-700M": "./configs/train/LHMPP-any-view.yaml",
    # "LHMPP-700MC": "./configs/train/LHMPP-any-view-convhead.yaml",  # coming soon
    "LHMPPS-700M": "./configs/train/LHMPP-any-view-DPTS.yaml",
}

MEMORY_MODEL_CARD = {
    "LHMPP-700M-PixelShuffle": 8000,  # 8G, default
    "LHMPP-700M-SMPLX-FREE": 8000,  # 8G
    "LHMPP-700M": 8000,  # 8G
    # "LHMPP-700MC": 8000,  # 8G, coming soon
    "LHMPPS-700M": 8000,  # 8G
}

# App / inference: gs_render (Gaussian raster only, no neural refiner) and
# scripts/inference/to_gs_ply.py (GS PLY export, T-pose or SMPL-X frame).
# SMPLX-FREE variants (ShapeHead + featbacksplat GS); neural_renderer type may differ.
# First entry is the CLI default for to_gs_ply.py (--model_name).
GS_RENDER_SUPPORTED_MODEL_NAMES = [
    "LHMPP-700M-PixelShuffle",
    "LHMPP-700M-SMPLX-FREE",
]


def model_supports_gs_render(model_name: str) -> bool:
    return model_name in GS_RENDER_SUPPORTED_MODEL_NAMES