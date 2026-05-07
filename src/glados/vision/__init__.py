"""Vision processing components."""

from .vision_config import CameraVisionConfig, ScreenVisionConfig, VisionConfig
from .vision_request import VisionRequest
from .vision_state import VisionState

__all__ = [
    "CameraVisionConfig",
    "CameraVisionProcessor",
    "FastVLM",
    "ScreenVisionConfig",
    "ScreenVisionProcessor",
    "VisionConfig",
    "VisionProcessor",
    "VisionRequest",
    "VisionState",
]


def __getattr__(name: str) -> object:
    if name == "FastVLM":
        from .fastvlm import FastVLM

        return FastVLM
    if name == "VisionProcessor":
        from .vision_processor import VisionProcessor

        return VisionProcessor
    if name == "CameraVisionProcessor":
        from .vision_processor import CameraVisionProcessor

        return CameraVisionProcessor
    if name == "ScreenVisionProcessor":
        from .screen_processor import ScreenVisionProcessor

        return ScreenVisionProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
