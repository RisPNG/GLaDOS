import threading

import numpy as np
import pytest

from glados.vision import screen_processor
from glados.vision.screen_processor import ScreenVisionProcessor
from glados.vision.vision_config import ScreenVisionConfig, VisionConfig
from glados.vision.vision_state import VisionState


def test_legacy_vision_config_defaults_to_camera() -> None:
    config = VisionConfig.model_validate(
        {
            "camera_index": 1,
            "capture_interval_seconds": 3,
        }
    )

    assert config.camera is not None
    assert config.camera.camera_index == 1
    assert config.camera.capture_interval_seconds == 3
    assert config.enabled_sources() == {"camera"}


def test_nested_screen_config_can_enable_both_sources() -> None:
    config = VisionConfig.model_validate(
        {
            "camera": {"enabled": True},
            "screen": {"enabled": True, "monitors": [1, 2]},
        }
    )

    assert config.enabled_sources() == {"camera", "screen"}
    assert config.screen is not None
    assert config.screen.monitors == [1, 2]


def test_screen_config_can_split_background_and_tool_analyzers() -> None:
    config = VisionConfig.model_validate(
        {
            "screen": {
                "enabled": True,
                "analyzer": "fastvlm",
                "tool_analyzer": "openai_compatible",
                "completion_url": "http://localhost:12331",
            }
        }
    )

    assert config.screen is not None
    assert config.screen.analyzer == "fastvlm"
    assert config.screen.effective_tool_analyzer() == "openai_compatible"


def test_vision_state_formats_camera_and_screen_sources() -> None:
    state = VisionState()
    state.update("User at desk.", source="camera")
    state.update("VS Code terminal shows an error.", source="screen", key="monitor_1")

    message = state.as_message()

    assert message is not None
    assert "[vision:camera] User at desk." in message["content"]
    assert "[vision:screen:monitor_1] VS Code terminal shows an error." in message["content"]


def test_screen_openai_compatible_posts_image_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "Visible terminal error."}}]}

    def fake_post(url: str, **kwargs: object) -> DummyResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(screen_processor.requests, "post", fake_post)
    processor = ScreenVisionProcessor(
        vision_state=VisionState(),
        processing_active_event=threading.Event(),
        shutdown_event=threading.Event(),
        config=ScreenVisionConfig(
            enabled=True,
            analyzer="openai_compatible",
            completion_url="http://localhost:12331",
        ),
    )

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = processor._get_description(1, frame, "Read screen.", 32)

    assert result == "Visible terminal error."
    assert captured["url"] == "http://localhost:12331/v1/chat/completions"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    payload = kwargs["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M"
    content = payload["messages"][1]["content"]
    assert content[0]["text"] == "Read screen."
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
