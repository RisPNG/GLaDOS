"""Vision processing components."""

from .vision_config import VisionConfig
from .vision_request import VisionRequest
from .vision_state import VisionState

__all__ = ["FastVLM", "VisionConfig", "VisionProcessor", "VisionRequest", "VisionState"]


def __getattr__(name: str) -> object:
    if name == "FastVLM":
        from .fastvlm import FastVLM

        return FastVLM
    if name == "VisionProcessor":
        from .vision_processor import VisionProcessor

        return VisionProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
