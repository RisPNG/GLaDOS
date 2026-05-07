"""Screen vision processor for monitor screenshots and desktop summaries."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from ..autonomy import EventBus
from ..autonomy.events import VisionUpdateEvent
from ..observability import ObservabilityBus, trim_message
from .constants import SCREEN_DEFAULT_PROMPT
from .fastvlm import FastVLM
from .vision_config import ScreenVisionConfig
from .vision_request import VisionRequest
from .vision_state import VisionState


class ScreenCaptureError(RuntimeError):
    """Raised when screen capture is unavailable."""


class ScreenVisionProcessor:
    """Captures monitor screenshots, detects changes, and summarizes them with a VLM."""

    def __init__(
        self,
        vision_state: VisionState,
        processing_active_event: threading.Event,
        shutdown_event: threading.Event,
        config: ScreenVisionConfig,
        model_dir: Path | None = None,
        model: FastVLM | None = None,
        request_queue: queue.Queue[VisionRequest] | None = None,
        event_bus: EventBus | None = None,
        observability_bus: ObservabilityBus | None = None,
    ) -> None:
        self.vision_state = vision_state
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.config = config
        self._request_queue = request_queue
        self._event_bus = event_bus
        self._observability_bus = observability_bus
        self._model = model or (FastVLM(model_dir) if config.analyzer == "fastvlm" else None)
        self._last_frames: dict[int, NDArray[np.uint8]] = {}
        self._last_features: dict[int, NDArray[np.float32]] = {}
        self._prompt_cache: dict[tuple[int, str, int], str] = {}
        self._last_descriptions: dict[int, str] = {}
        self._last_inference_at: dict[int, float] = {}
        self._capture_backend: Any | None = None

    def run(self) -> None:
        logger.info("ScreenVisionProcessor thread started.")
        try:
            while not self.shutdown_event.is_set():
                loop_started = time.perf_counter()

                if self._process_tool_request():
                    self._sleep(loop_started)
                    continue

                try:
                    monitor_ids = self._monitor_ids()
                except ScreenCaptureError as exc:
                    logger.error("ScreenVisionProcessor: {}", exc)
                    self._sleep(loop_started)
                    continue

                for monitor_id in monitor_ids:
                    if self.shutdown_event.is_set():
                        break
                    frame = self._grab_monitor(monitor_id)
                    if frame is None:
                        continue

                    processed = self._preprocess_frame(frame)
                    change_score = self._scene_change_score(monitor_id, processed)
                    now = time.time()
                    recently_processed = (
                        now - self._last_inference_at.get(monitor_id, 0.0)
                    ) < self.config.capture_interval_seconds
                    if (
                        monitor_id in self._last_frames
                        and change_score <= self.config.scene_change_threshold
                    ) or recently_processed:
                        self._last_frames[monitor_id] = processed.copy()
                        continue

                    self._last_frames[monitor_id] = processed.copy()
                    self._clear_monitor_cache(monitor_id)
                    description = self._get_description(
                        monitor_id,
                        frame,
                        prompt=SCREEN_DEFAULT_PROMPT,
                        max_tokens=self.config.max_tokens,
                    )
                    if description:
                        key = self._monitor_key(monitor_id)
                        self.vision_state.update(
                            description,
                            source="screen",
                            key=key,
                            metadata={"monitor": monitor_id},
                        )
                        logger.success("Screen vision snapshot updated ({}): {}", key, description)
                        self._publish_update(monitor_id, description, change_score)
                        self._last_descriptions[monitor_id] = description
                        self._last_inference_at[monitor_id] = now

                self._sleep(loop_started)
        except Exception as exc:
            logger.exception("ScreenVisionProcessor uncaught exception: {}", exc)
        finally:
            self._close_backend()
            logger.info("ScreenVisionProcessor thread finished.")

    def _process_tool_request(self) -> bool:
        if self._request_queue is None:
            return False

        try:
            request = self._request_queue.get_nowait()
        except queue.Empty:
            return False

        try:
            monitor_ids = [request.monitor] if request.monitor else self._monitor_ids()
            monitor_ids = [monitor_id for monitor_id in monitor_ids if monitor_id is not None]
        except ScreenCaptureError as exc:
            request.response_queue.put(f"error: screen capture unavailable - {exc}")
            return True

        if not monitor_ids:
            request.response_queue.put("error: no screen monitors available")
            return True

        results: list[str] = []
        for monitor_id in monitor_ids:
            frame = self._grab_monitor(monitor_id)
            if frame is None:
                results.append(f"{self._monitor_key(monitor_id)}: error: failed to capture screen")
                continue

            processed = self._preprocess_frame(frame)
            change_score = self._scene_change_score(monitor_id, processed)
            reuse_cached = (
                monitor_id in self._last_features
                and change_score <= self.config.scene_change_threshold
            )
            prompt = request.prompt.strip() if request.prompt else ""
            cache_key = (monitor_id, prompt, int(request.max_tokens))
            if reuse_cached:
                description = self._prompt_cache.get(cache_key)
                if description is None:
                    description = self._model.describe_from_features(
                        self._last_features[monitor_id],
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                    )
                    if description:
                        self._prompt_cache[cache_key] = description
            else:
                self._last_frames[monitor_id] = processed.copy()
                self._clear_monitor_cache(monitor_id)
                description = self._get_description(
                    monitor_id,
                    frame,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                )

            if description:
                results.append(f"{self._monitor_key(monitor_id)}: {description}")
            else:
                results.append(f"{self._monitor_key(monitor_id)}: error: screen vision inference failed")

        request.response_queue.put("\n".join(results))
        return True

    def _monitor_ids(self) -> list[int]:
        if self.config.backend == "dxcam":
            return self._dxcam_monitor_ids()
        return self._mss_monitor_ids()

    def _mss_monitor_ids(self) -> list[int]:
        try:
            import mss
        except ImportError as exc:
            raise ScreenCaptureError("install the 'mss' package or use camera-only vision") from exc

        if self._capture_backend is None or self._capture_backend.__class__.__module__.split(".")[0] != "mss":
            self._close_backend()
            self._capture_backend = mss.mss()

        monitor_count = max(0, len(self._capture_backend.monitors) - 1)
        if monitor_count == 0:
            raise ScreenCaptureError("no monitors found")

        return self._select_monitor_ids(monitor_count)

    def _dxcam_monitor_ids(self) -> list[int]:
        try:
            import dxcam  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ScreenCaptureError("install 'dxcam' or set screen.backend to 'mss'") from exc

        if self._capture_backend is None or self._capture_backend.__class__.__module__.split(".")[0] != "dxcam":
            self._close_backend()
            self._capture_backend = {"module": dxcam, "cameras": {}}

        # DXcam uses zero-based outputs. Expose them as 1-based monitor ids to match MSS/config.
        output_count = len(dxcam.output_info())
        if output_count == 0:
            raise ScreenCaptureError("no DXcam outputs found")
        return self._select_monitor_ids(output_count)

    def _select_monitor_ids(self, monitor_count: int) -> list[int]:
        monitors = self.config.monitors
        if monitors == "all":
            return list(range(1, monitor_count + 1))
        if monitors == "primary":
            return [1]
        selected = []
        for monitor_id in monitors:
            if 1 <= int(monitor_id) <= monitor_count:
                selected.append(int(monitor_id))
        return selected

    def _grab_monitor(self, monitor_id: int) -> NDArray[np.uint8] | None:
        try:
            if self.config.backend == "dxcam":
                return self._grab_monitor_dxcam(monitor_id)
            return self._grab_monitor_mss(monitor_id)
        except Exception as exc:
            logger.warning("ScreenVisionProcessor: Failed to capture monitor {}: {}", monitor_id, exc)
            return None

    def _grab_monitor_mss(self, monitor_id: int) -> NDArray[np.uint8]:
        self._mss_monitor_ids()
        monitor = self._capture_backend.monitors[monitor_id]
        shot = self._capture_backend.grab(monitor)
        bgra = np.array(shot, dtype=np.uint8)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def _grab_monitor_dxcam(self, monitor_id: int) -> NDArray[np.uint8] | None:
        self._dxcam_monitor_ids()
        dxcam_module = self._capture_backend["module"]
        cameras: dict[int, Any] = self._capture_backend["cameras"]
        camera = cameras.get(monitor_id)
        if camera is None:
            camera = dxcam_module.create(output_idx=monitor_id - 1, output_color="BGR")
            cameras[monitor_id] = camera
        return camera.grab()

    def _preprocess_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        target_resolution = self.config.resolution
        height, width = frame.shape[:2]
        max_dim = max(height, width)

        if max_dim <= target_resolution:
            return frame

        scale = target_resolution / float(max_dim)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _scene_change_score(self, monitor_id: int, current_frame: NDArray[np.uint8]) -> float:
        last_frame = self._last_frames.get(monitor_id)
        if last_frame is None or current_frame.shape != last_frame.shape:
            return 1.0
        diff = cv2.absdiff(current_frame, last_frame)
        return float(np.mean(diff)) / 255.0

    def _get_description(
        self,
        monitor_id: int,
        frame: NDArray[np.uint8],
        prompt: str,
        max_tokens: int,
    ) -> str | None:
        if self.config.analyzer != "fastvlm":
            logger.error(
                "ScreenVisionProcessor: analyzer '{}' is configured but only FastVLM is implemented.",
                self.config.analyzer,
            )
            return None
        if self._model is None:
            logger.error("ScreenVisionProcessor: FastVLM model is unavailable.")
            return None

        prompt = prompt.strip() if prompt else SCREEN_DEFAULT_PROMPT
        try:
            padded = self._pad_to_square(frame)
            vision_features = self._model.encode_image(padded)
            description = self._model.describe_from_features(
                vision_features,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            self._last_features[monitor_id] = vision_features
            if description:
                self._prompt_cache[(monitor_id, prompt, int(max_tokens))] = description
            return description
        except Exception as exc:
            logger.error("FastVLM screen inference failed: {}", exc)
            return None

    @staticmethod
    def _pad_to_square(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        if height == width:
            return frame
        size = max(height, width)
        canvas = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
        top = (size - height) // 2
        left = (size - width) // 2
        canvas[top : top + height, left : left + width] = frame
        return canvas

    def _sleep(self, loop_started: float) -> None:
        elapsed = time.perf_counter() - loop_started
        sleep_time = max(0.0, self.config.thumbnail_interval_seconds - elapsed)
        if sleep_time:
            self.shutdown_event.wait(timeout=sleep_time)

    def _publish_update(self, monitor_id: int, description: str, change_score: float) -> None:
        key = self._monitor_key(monitor_id)
        if self._event_bus:
            self._event_bus.publish(
                VisionUpdateEvent(
                    description=description,
                    prev_description=self._last_descriptions.get(monitor_id),
                    change_score=change_score,
                    captured_at=time.time(),
                    source="screen",
                    key=key,
                )
            )
        if self._observability_bus:
            self._observability_bus.emit(
                source=f"vision.screen.{key}",
                kind="update",
                message=trim_message(description),
                meta={"change_score": round(change_score, 4), "monitor": monitor_id},
            )

    def _clear_monitor_cache(self, monitor_id: int) -> None:
        self._last_features.pop(monitor_id, None)
        self._prompt_cache = {
            cache_key: value
            for cache_key, value in self._prompt_cache.items()
            if cache_key[0] != monitor_id
        }

    @staticmethod
    def _monitor_key(monitor_id: int) -> str:
        return f"monitor_{monitor_id}"

    def _close_backend(self) -> None:
        if self._capture_backend is None:
            return
        try:
            if isinstance(self._capture_backend, dict):
                for camera in self._capture_backend.get("cameras", {}).values():
                    close = getattr(camera, "release", None)
                    if callable(close):
                        close()
            else:
                close = getattr(self._capture_backend, "close", None)
                if callable(close):
                    close()
        finally:
            self._capture_backend = None
