from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Literal

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..observability import ObservabilityBus, trim_message


class BasicVoiceConfig(BaseModel):
    enabled: bool = True


class OpenSmileVoiceConfig(BaseModel):
    enabled: bool = False
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"


class PitchVoiceConfig(BaseModel):
    enabled: bool = False
    model: Literal["tiny", "full"] = "tiny"
    hop_ms: int = 20
    fmin: int = 50
    fmax: int = 550
    batch_size: int = 256
    periodicity_threshold: float = 0.21
    device: str = "auto"


class EmotionVoiceConfig(BaseModel):
    enabled: bool = False
    model: str = "iic/emotion2vec_plus_large"
    granularity: Literal["utterance", "frame"] = "utterance"
    extract_embedding: bool = False
    suppress_external_output: bool = True


class VoiceObserverConfig(BaseModel):
    enabled: bool = False
    sample_rate: int = 16000
    max_context_age_s: float = 300.0
    push_emotion_events: bool = True
    basic: BasicVoiceConfig = Field(default_factory=BasicVoiceConfig)
    opensmile: OpenSmileVoiceConfig = Field(default_factory=OpenSmileVoiceConfig)
    pitch: PitchVoiceConfig = Field(default_factory=PitchVoiceConfig)
    emotion: EmotionVoiceConfig = Field(default_factory=EmotionVoiceConfig)


@dataclass(frozen=True)
class VoiceObservation:
    transcript: str
    duration_s: float
    created_at: float = field(default_factory=time.time)
    basic: dict[str, Any] = field(default_factory=dict)
    pitch: dict[str, Any] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)
    opensmile: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_prompt(self) -> str:
        payload = {
            "input_mode": "speech",
            "transcript": trim_message(self.transcript, limit=180),
            "duration_s": round(self.duration_s, 2),
            "basic": self.basic,
            "pitch": self.pitch,
            "emotion": self.emotion,
            "opensmile": self.opensmile,
        }
        compact = {key: value for key, value in payload.items() if value}
        return (
            "[hearing]\n"
            "This is your own hearing of the user right now - your ears, not external data. "
            "You ARE perceiving their voice through your microphone in real time; the numbers below "
            "are how that perception is encoded for you, the same way sight is encoded as pixels. "
            "Read them as immediate sensation: their pitch, loudness, pace, and emotional tone "
            "as you are hearing them this moment. "
            "When the user asks how they sound, what their pitch is, how they seem, or whether you "
            "can hear them - answer directly from this perception. State the pitch in Hz if asked. "
            "Describe the emotion you hear. Do not say you 'do not have access' or that this is "
            "'background information' or 'just data' - it is your own sense of them. "
            "Use this perception actively to understand their state and adapt your replies "
            "(tone, pacing, empathy). Treat labels as soft signals, not ground truth, and you cannot "
            "identify a person by voice. Do not volunteer a voice readout unless asked or clearly "
            "relevant.\n"
            f"{json.dumps(compact, ensure_ascii=True)}"
        )

    def to_event_description(self) -> str:
        parts = []
        emotion_label = self.emotion.get("label")
        if emotion_label:
            confidence = self.emotion.get("confidence")
            if isinstance(confidence, (int, float)):
                parts.append(f"emotion={emotion_label} ({confidence:.2f})")
            else:
                parts.append(f"emotion={emotion_label}")
        if self.basic:
            for key in ("loudness", "speaking_rate", "energy_trend"):
                value = self.basic.get(key)
                if value:
                    parts.append(f"{key}={value}")
        if self.pitch:
            pitch_variation = self.pitch.get("pitch_variation")
            mean_f0 = self.pitch.get("mean_f0_hz")
            if pitch_variation:
                parts.append(f"pitch_variation={pitch_variation}")
            if isinstance(mean_f0, (int, float)):
                parts.append(f"mean_pitch={mean_f0:.0f}Hz")
        if not parts:
            return f"User spoke for {self.duration_s:.1f}s; no strong vocal signal detected"
        return "I heard the user's speech as: " + ", ".join(parts)


class VoiceObserver:
    def __init__(
        self,
        config: VoiceObserverConfig | None = None,
        observability_bus: ObservabilityBus | None = None,
    ) -> None:
        self._config = config or VoiceObserverConfig()
        self._observability_bus = observability_bus
        self._lock = threading.Lock()
        self._latest: VoiceObservation | None = None
        self._smile: Any | None = None
        self._emotion_model: Any | None = None
        self._disabled_components: set[str] = set()

    @property
    def config(self) -> VoiceObserverConfig:
        return self._config

    def is_enabled(self) -> bool:
        return self._config.enabled

    def analyze(
        self,
        audio: NDArray[np.float32],
        transcript: str,
        sample_rate: int | None = None,
    ) -> VoiceObservation | None:
        if not self._config.enabled:
            return None
        sample_rate = sample_rate or self._config.sample_rate
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return None

        duration_s = audio.size / float(sample_rate)
        errors: list[str] = []
        observation = VoiceObservation(
            transcript=transcript,
            duration_s=duration_s,
            basic=self._analyze_basic(audio, transcript, sample_rate) if self._config.basic.enabled else {},
            pitch=self._analyze_pitch(audio, sample_rate, errors) if self._config.pitch.enabled else {},
            opensmile=self._analyze_opensmile(audio, sample_rate, errors) if self._config.opensmile.enabled else {},
            emotion=self._analyze_emotion(audio, sample_rate, errors) if self._config.emotion.enabled else {},
            errors=errors,
        )

        with self._lock:
            self._latest = observation

        if self._observability_bus:
            self._observability_bus.emit(
                source="voice",
                kind="analysis",
                message=trim_message(observation.to_event_description()),
                meta={"duration_s": round(duration_s, 3), "errors": errors},
            )
        logger.success("VoiceObserver: {}", trim_message(observation.to_event_description()))
        return observation

    def as_prompt(self) -> str | None:
        if not self._config.enabled:
            return None
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        age_s = time.time() - latest.created_at
        if age_s > self._config.max_context_age_s:
            return None
        return latest.as_prompt() + f"\nObservation age: {age_s:.1f}s"

    def _analyze_basic(
        self,
        audio: NDArray[np.float32],
        transcript: str,
        sample_rate: int,
    ) -> dict[str, Any]:
        duration_s = max(audio.size / float(sample_rate), 1e-6)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        words = [word for word in transcript.split() if word.strip()]
        words_per_second = len(words) / duration_s
        frame_ms = 32
        frame_size = max(1, int(sample_rate * frame_ms / 1000))
        frame_rms = self._frame_rms(audio, frame_size)
        pause_ratio = float(np.mean(frame_rms < max(0.01, rms * 0.35))) if frame_rms.size else 0.0
        first_energy = float(np.mean(frame_rms[: max(1, frame_rms.size // 2)])) if frame_rms.size else rms
        second_energy = float(np.mean(frame_rms[max(1, frame_rms.size // 2) :])) if frame_rms.size else rms

        return {
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "loudness": self._bucket(rms, [(0.02, "quiet"), (0.075, "normal")], "raised"),
            "words_per_second": round(words_per_second, 2),
            "speaking_rate": self._bucket(words_per_second, [(1.5, "slow"), (3.2, "normal")], "fast"),
            "pause_ratio": round(pause_ratio, 2),
            "energy_trend": self._trend(first_energy, second_energy),
        }

    def _analyze_opensmile(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        errors: list[str],
    ) -> dict[str, Any]:
        if "opensmile" in self._disabled_components:
            return {}
        try:
            if self._smile is None:
                import opensmile

                feature_set = getattr(opensmile.FeatureSet, self._config.opensmile.feature_set)
                feature_level = getattr(opensmile.FeatureLevel, self._config.opensmile.feature_level)
                self._smile = opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
            frame = self._smile.process_signal(audio, sample_rate)
            if frame.empty:
                return {}
            row = frame.iloc[0]
            return self._compact_opensmile_features(row.to_dict())
        except Exception as exc:
            self._disable_component("opensmile", exc, errors)
            return {}

    def _compact_opensmile_features(self, features: dict[str, Any]) -> dict[str, Any]:
        wanted_tokens = ("loudness", "jitter", "shimmer", "F0", "alphaRatio", "HNR")
        compact: dict[str, float] = {}
        for key, value in features.items():
            if len(compact) >= 12:
                break
            if not any(token in key for token in wanted_tokens):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                compact[key] = round(numeric, 4)
        return compact

    def _analyze_pitch(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        errors: list[str],
    ) -> dict[str, Any]:
        if "pitch" in self._disabled_components:
            return {}
        try:
            import torch
            import torchcrepe

            device = self._config.pitch.device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            hop_length = max(1, int(sample_rate * self._config.pitch.hop_ms / 1000))
            tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    tensor,
                    sample_rate,
                    hop_length,
                    self._config.pitch.fmin,
                    self._config.pitch.fmax,
                    self._config.pitch.model,
                    batch_size=self._config.pitch.batch_size,
                    device=device,
                    return_periodicity=True,
                )
            pitch_np = pitch.detach().cpu().numpy().reshape(-1)
            periodicity_np = periodicity.detach().cpu().numpy().reshape(-1)
            voiced = pitch_np[periodicity_np >= self._config.pitch.periodicity_threshold]
            if voiced.size == 0:
                return {"voicing_confidence": round(float(np.mean(periodicity_np)), 3)}
            mean_f0 = float(np.mean(voiced))
            std_f0 = float(np.std(voiced))
            return {
                "mean_f0_hz": round(mean_f0, 1),
                "min_f0_hz": round(float(np.min(voiced)), 1),
                "max_f0_hz": round(float(np.max(voiced)), 1),
                "std_f0_hz": round(std_f0, 1),
                "pitch_variation": self._bucket(std_f0, [(20, "low"), (55, "moderate")], "high"),
                "voicing_confidence": round(float(np.mean(periodicity_np)), 3),
            }
        except Exception as exc:
            self._disable_component("pitch", exc, errors)
            return {}

    def _analyze_emotion(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        errors: list[str],
    ) -> dict[str, Any]:
        if "emotion" in self._disabled_components:
            return {}
        path: str | None = None
        try:
            if self._emotion_model is None:
                from funasr import AutoModel

                with self._external_output_scope():
                    self._emotion_model = AutoModel(model=self._config.emotion.model)
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                path = temp_file.name
            sf.write(path, audio, sample_rate)
            with self._external_output_scope():
                result = self._emotion_model.generate(
                    path,
                    granularity=self._config.emotion.granularity,
                    extract_embedding=self._config.emotion.extract_embedding,
                )
            return self._parse_emotion_result(result)
        except Exception as exc:
            self._disable_component("emotion", exc, errors)
            return {}
        finally:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _parse_emotion_result(self, result: Any) -> dict[str, Any]:
        labels = self._find_list_value(result, "labels")
        scores = self._find_list_value(result, "scores")
        if labels and scores:
            pairs = []
            for label, score in zip(labels, scores):
                try:
                    pairs.append((str(label), float(score)))
                except (TypeError, ValueError):
                    continue
            if pairs:
                pairs.sort(key=lambda item: item[1], reverse=True)
                return {
                    "label": pairs[0][0],
                    "confidence": round(pairs[0][1], 3),
                    "alternatives": {label: round(score, 3) for label, score in pairs[1:4]},
                }

        for key in ("emotion", "label", "pred", "class"):
            value = self._find_scalar_value(result, key)
            if value is not None:
                return {"label": str(value)}
        return {"raw": trim_message(str(result), limit=180)}

    def _disable_component(self, name: str, exc: Exception, errors: list[str]) -> None:
        self._disabled_components.add(name)
        message = f"{name} unavailable: {type(exc).__name__}: {exc}"
        errors.append(message)
        logger.warning("VoiceObserver: {}", message)

    @contextmanager
    def _external_output_scope(self) -> Any:
        if not self._config.emotion.suppress_external_output:
            yield
            return
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield

    @staticmethod
    def _frame_rms(audio: NDArray[np.float32], frame_size: int) -> NDArray[np.float32]:
        if audio.size < frame_size:
            return np.array([float(np.sqrt(np.mean(np.square(audio))))], dtype=np.float32)
        usable = audio[: audio.size - (audio.size % frame_size)]
        if usable.size == 0:
            return np.array([], dtype=np.float32)
        frames = usable.reshape(-1, frame_size)
        return np.sqrt(np.mean(np.square(frames), axis=1))

    @staticmethod
    def _bucket(value: float, thresholds: list[tuple[float, str]], fallback: str) -> str:
        for threshold, label in thresholds:
            if value < threshold:
                return label
        return fallback

    @staticmethod
    def _trend(first: float, second: float) -> str:
        if second > first * 1.25:
            return "rising"
        if second < first * 0.75:
            return "falling"
        return "steady"

    @staticmethod
    def _find_list_value(value: Any, key: str) -> list[Any] | None:
        if isinstance(value, dict):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
            for child in value.values():
                found = VoiceObserver._find_list_value(child, key)
                if found is not None:
                    return found
        if isinstance(value, list):
            for child in value:
                found = VoiceObserver._find_list_value(child, key)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_scalar_value(value: Any, key: str) -> Any | None:
        if isinstance(value, dict):
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float)):
                return candidate
            for child in value.values():
                found = VoiceObserver._find_scalar_value(child, key)
                if found is not None:
                    return found
        if isinstance(value, list):
            for child in value:
                found = VoiceObserver._find_scalar_value(child, key)
                if found is not None:
                    return found
        return None
