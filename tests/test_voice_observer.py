import time

import numpy as np

from glados.core.voice_observer import VoiceObservation, VoiceObserver, VoiceObserverConfig


def test_voice_observer_basic_features_and_prompt() -> None:
    config = VoiceObserverConfig(enabled=True)
    observer = VoiceObserver(config)
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    audio = (0.13 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    observation = observer.analyze(audio, "This bug is back again", sample_rate=sample_rate)

    assert observation is not None
    assert observation.basic["loudness"] == "raised"
    assert observation.basic["speaking_rate"] in {"normal", "fast"}
    prompt = observer.as_prompt()
    assert prompt is not None
    assert "[hearing]" in prompt
    assert '"input_mode": "speech"' in prompt
    assert "your own hearing" in prompt
    assert "do not have access" in prompt
    assert "soft signal" in prompt
    assert "loudness" in prompt


def test_voice_observer_context_expires() -> None:
    observer = VoiceObserver(VoiceObserverConfig(enabled=True, max_context_age_s=0.01))
    observer.analyze(np.ones(1600, dtype=np.float32) * 0.02, "hello", sample_rate=16000)

    time.sleep(0.02)

    assert observer.as_prompt() is None


def test_voice_observation_event_description_prefers_compact_signals() -> None:
    observation = VoiceObservation(
        transcript="I am fine",
        duration_s=1.2,
        basic={"loudness": "quiet", "speaking_rate": "slow", "energy_trend": "falling"},
        pitch={"mean_f0_hz": 143.2, "pitch_variation": "low"},
        emotion={"label": "sad", "confidence": 0.72},
    )

    description = observation.to_event_description()

    assert description.startswith("I heard the user's speech as:")
    assert "emotion=sad" in description
    assert "loudness=quiet" in description
    assert "mean_pitch=143Hz" in description


def test_voice_observer_parses_emotion_labels_scores() -> None:
    observer = VoiceObserver(VoiceObserverConfig(enabled=True))

    parsed = observer._parse_emotion_result(
        [{"labels": ["neutral", "angry", "sad"], "scores": [0.1, 0.7, 0.2]}]
    )

    assert parsed["label"] == "angry"
    assert parsed["confidence"] == 0.7
    assert parsed["alternatives"] == {"sad": 0.2, "neutral": 0.1}
