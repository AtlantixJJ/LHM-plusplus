"""Animation inference patch for benchmark GS export.

:class:`~core.models.modeling_humana4o_lrm.ModelHumanA4OLRM` already implements
``return_gs_outputs`` on :meth:`animation_infer`. ``patch_model_animation_infer`` is kept for
import compatibility with benchmark scripts; it is a no-op when the base method accepts
``return_gs_outputs``.
"""

from __future__ import annotations

import inspect

import torch.nn as nn


def patch_model_animation_infer(model: nn.Module) -> nn.Module:
    """Ensure ``animation_infer`` supports ``return_gs_outputs`` (no-op when already present)."""
    sig = inspect.signature(getattr(model, "animation_infer", None))
    if "return_gs_outputs" not in sig.parameters:
        raise RuntimeError(
            "ModelHumanA4OLRM.animation_infer is missing return_gs_outputs; "
            "upgrade core.models.modeling_humana4o_lrm."
        )
    return model
