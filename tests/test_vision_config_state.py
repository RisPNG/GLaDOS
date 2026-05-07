from glados.vision.vision_config import VisionConfig
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


def test_vision_state_formats_camera_and_screen_sources() -> None:
    state = VisionState()
    state.update("User at desk.", source="camera")
    state.update("VS Code terminal shows an error.", source="screen", key="monitor_1")

    message = state.as_message()

    assert message is not None
    assert "[vision:camera] User at desk." in message["content"]
    assert "[vision:screen:monitor_1] VS Code terminal shows an error." in message["content"]
