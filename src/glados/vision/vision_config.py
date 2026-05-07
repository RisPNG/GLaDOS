from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraVisionConfig(BaseModel):
    """Configuration for webcam/camera vision."""

    enabled: bool = Field(default=True, description="Enable camera vision.")

    camera_index: int = Field(
        default=0,
        ge=0,
        description="The index of the camera to use for capturing images. Use 0 if only one camera is connected.",
    )
    capture_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Interval in seconds between image captures. Tune this to your own system.",
    )
    resolution: int = Field(
        default=384,
        gt=0,
        description="Resolution (in pixels) used for scene-change detection. FastVLM handles its own resize.",
    )
    scene_change_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum normalized difference between frames to trigger VLM inference. "
            "0=always process, 1=never process."
        ),
    )
    max_tokens: int = Field(
        default=64,
        gt=0,
        le=512,
        description="Maximum tokens to generate in the background camera description.",
    )


class ScreenVisionConfig(BaseModel):
    """Configuration for monitor/screen vision."""

    enabled: bool = Field(default=False, description="Enable monitor/screen vision.")
    backend: Literal["mss", "dxcam"] = Field(
        default="mss",
        description="Screen capture backend. MSS is cross-platform; DXcam is Windows-only and optional.",
    )
    analyzer: Literal["fastvlm", "openai_compatible"] = Field(
        default="fastvlm",
        description="Background screen understanding backend.",
    )
    tool_analyzer: Literal["same", "fastvlm", "openai_compatible"] = Field(
        default="same",
        description="On-demand screen_look backend. Use 'same' to reuse analyzer.",
    )
    model: str = Field(
        default="bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M",
        description="OpenAI-compatible VLM model id for screen vision.",
    )
    completion_url: str | None = Field(
        default=None,
        description="OpenAI-compatible VLM endpoint for screen vision.",
    )
    tool_model: str | None = Field(
        default=None,
        description="Optional model id override for screen_look.",
    )
    tool_completion_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible endpoint override for screen_look.",
    )
    tool_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for the screen_look OpenAI-compatible endpoint.",
    )
    monitors: Literal["all", "primary"] | list[int] = Field(
        default="all",
        description="Monitor selection. Use 'all', 'primary', or 1-based MSS monitor indexes.",
    )
    thumbnail_interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Interval in seconds between low-resolution change checks.",
    )
    capture_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Minimum seconds between full screen VLM summaries for the same monitor.",
    )
    resolution: int = Field(
        default=384,
        gt=0,
        description="Resolution used for screen change detection.",
    )
    scene_change_threshold: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Minimum normalized difference between thumbnails to trigger screen inference.",
    )
    max_tokens: int = Field(
        default=128,
        gt=0,
        le=512,
        description="Maximum tokens to generate in the background screen description.",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Timeout for OpenAI-compatible screen vision HTTP requests.",
    )

    def effective_tool_analyzer(self) -> Literal["fastvlm", "openai_compatible"]:
        if self.tool_analyzer == "same":
            return self.analyzer
        return cast(Literal["fastvlm", "openai_compatible"], self.tool_analyzer)


class VisionConfig(BaseModel):
    """Configuration for camera and screen vision sources."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True, description="Enable all configured vision sources.")
    model_dir: Path | None = Field(
        default=None,
        description="Path to FastVLM ONNX model directory. Uses default if None.",
    )
    camera: CameraVisionConfig | None = None
    screen: ScreenVisionConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_camera_config(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        legacy_camera_keys = {
            "camera_index",
            "capture_interval_seconds",
            "resolution",
            "scene_change_threshold",
            "max_tokens",
        }
        if "camera" not in data and any(key in data for key in legacy_camera_keys):
            data_dict = dict(data)
            data_dict["camera"] = {
                key: data_dict[key]
                for key in legacy_camera_keys
                if key in data_dict
            }
            return data_dict
        return data

    @model_validator(mode="after")
    def _default_to_camera_for_legacy_configs(self) -> VisionConfig:
        if self.camera is None and self.screen is None:
            self.camera = CameraVisionConfig()
        return self

    def enabled_sources(self) -> set[str]:
        if not self.enabled:
            return set()
        sources: set[str] = set()
        if self.camera and self.camera.enabled:
            sources.add("camera")
        if self.screen and self.screen.enabled:
            sources.add("screen")
        return sources

    def is_enabled(self) -> bool:
        return bool(self.enabled_sources())
