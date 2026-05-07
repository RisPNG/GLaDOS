import queue
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from .config import AutonomyConfig
from .emotion_state import EmotionEvent
from .event_bus import EventBus
from .events import TaskUpdateEvent, TimeTickEvent, VisionUpdateEvent
from .interaction_state import InteractionState
from .slots import TaskSlotStore
from ..observability import ObservabilityBus, trim_message
from ..vision.vision_state import VisionState
from ..core.audio_state import AudioState
from ..core.llm_tracking import InFlightCounter

if TYPE_CHECKING:
    from .agents.emotion_agent import EmotionAgent


class AutonomyLoop:
    # Scene change threshold for triggering emotion events
    VISION_EMOTION_THRESHOLD = 0.3
    # Treat user as still speaking if VAD fired within this window (debounce ASR finalization).
    USER_SPEAKING_GRACE_S = 1.5

    def __init__(
        self,
        config: AutonomyConfig,
        event_bus: EventBus,
        interaction_state: InteractionState,
        vision_state: VisionState | None,
        slot_store: TaskSlotStore,
        llm_queue: queue.Queue[dict[str, Any]],
        processing_active_event: threading.Event,
        currently_speaking_event: threading.Event,
        shutdown_event: threading.Event,
        observability_bus: ObservabilityBus | None = None,
        inflight_counter: InFlightCounter | None = None,
        emotion_agent: "EmotionAgent | None" = None,
        pause_time: float = 0.1,
        audio_state: AudioState | None = None,
        priority_queue: queue.Queue[dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._interaction_state = interaction_state
        self._vision_state = vision_state
        self._slot_store = slot_store
        self._llm_queue = llm_queue
        self._processing_active_event = processing_active_event
        self._currently_speaking_event = currently_speaking_event
        self._shutdown_event = shutdown_event
        self._observability_bus = observability_bus
        self._inflight_counter = inflight_counter
        self._emotion_agent = emotion_agent
        self._pause_time = pause_time
        self._audio_state = audio_state
        self._priority_queue = priority_queue
        self._last_prompt_ts = 0.0
        self._last_scene: str | None = None
        self._idle_nudge_turns = 0
        self._last_seen_user_age_s: float | None = None

    def set_emotion_agent(self, agent: "EmotionAgent") -> None:
        """Set the emotion agent for vision event forwarding."""
        self._emotion_agent = agent

    def run(self) -> None:
        logger.info("AutonomyLoop thread started.")
        while not self._shutdown_event.is_set():
            try:
                event = self._event_bus.get(timeout=self._pause_time)
            except queue.Empty:
                continue

            if isinstance(event, TaskUpdateEvent) and not event.notify_user:
                continue

            if self._should_skip():
                continue

            if (
                isinstance(event, TimeTickEvent)
                and self._config.coalesce_ticks
                and self._pending_autonomy()
            ):
                continue

            prompt = self._build_prompt(event)
            if not prompt:
                continue
            self._dispatch(prompt)
        logger.info("AutonomyLoop thread finished.")

    def _should_skip(self) -> bool:
        if self._currently_speaking_event.is_set():
            return True
        if self._user_voice_active():
            return True
        if self._priority_pending():
            return True
        since_user = self._interaction_state.seconds_since_user()
        since_assistant = self._interaction_state.seconds_since_assistant()
        if since_user is not None:
            if since_assistant is None or since_user < since_assistant:
                return True
            if self._config.cooldown_s > 0 and since_user < self._config.cooldown_s:
                return True
        if self._config.cooldown_s <= 0:
            return False
        return (time.time() - self._last_prompt_ts) < self._config.cooldown_s

    def _user_voice_active(self) -> bool:
        if self._audio_state is None:
            return False
        snap = self._audio_state.snapshot()
        if snap.vad_active:
            return True
        if snap.last_vad_active_at <= 0:
            return False
        return (time.time() - snap.last_vad_active_at) < self.USER_SPEAKING_GRACE_S

    def _priority_pending(self) -> bool:
        if self._priority_queue is None:
            return False
        try:
            return self._priority_queue.qsize() > 0
        except NotImplementedError:
            return False

    def _dispatch(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        if self._observability_bus:
            self._observability_bus.emit(
                source="autonomy",
                kind="dispatch",
                message=trim_message(prompt),
            )
        logger.success("Autonomy dispatch: {}", trim_message(prompt))
        payload = {
            "role": "user",
            "content": prompt,
            "autonomy": True,
            "_enqueued_at": time.time(),
            "_lane": "autonomy",
        }
        if not self._enqueue_llm(payload):
            logger.warning("Autonomy dispatch dropped: LLM queue is full.")
            return
        self._processing_active_event.set()
        self._last_prompt_ts = time.time()

    def _build_prompt(self, event: object) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        since_user = self._interaction_state.seconds_since_user()
        since_assistant = self._interaction_state.seconds_since_assistant()
        since_user_text = f"{since_user:.1f}" if since_user is not None else "unknown"
        since_assistant_text = f"{since_assistant:.1f}" if since_assistant is not None else "unknown"
        idle_nudge = self._idle_nudge_guidance(
            since_user,
            since_assistant,
            reserve_turn=isinstance(event, TimeTickEvent),
        )
        if isinstance(event, TimeTickEvent) and idle_nudge.startswith("exhausted"):
            return ""

        scene = self._current_scene()
        prev_scene = self._last_scene or "unknown"
        change_score = "unknown"

        if isinstance(event, VisionUpdateEvent):
            prev_scene = event.prev_description or "unknown"
            event_label = event.source if event.key is None else f"{event.source}:{event.key}"
            scene = f"{event_label}: {event.description}"
            all_scene = self._current_scene()
            if all_scene and all_scene != event.description:
                scene = f"{scene}\nAll current visual context:\n{all_scene}"
            change_score = f"{event.change_score:.4f}"
            self._last_scene = scene
            # Push vision event to emotion agent if change is significant
            if self._emotion_agent and event.change_score >= self.VISION_EMOTION_THRESHOLD:
                self._push_vision_emotion(event)
        elif isinstance(event, TimeTickEvent):
            if scene:
                self._last_scene = scene
        elif isinstance(event, TaskUpdateEvent):
            if scene:
                self._last_scene = scene

        tasks = self._task_summary()
        try:
            return self._config.tick_prompt.format(
                now=now,
                since_user=since_user_text,
                since_assistant=since_assistant_text,
                prev_scene=prev_scene,
                scene=scene or "unknown",
                change_score=change_score,
                idle_nudge=idle_nudge,
                tasks=tasks,
            )
        except KeyError:
            return self._config.tick_prompt

    def _idle_nudge_guidance(
        self,
        since_user: float | None,
        since_assistant: float | None,
        reserve_turn: bool = False,
    ) -> str:
        threshold = self._config.idle_nudge_after_s
        if threshold <= 0:
            return "disabled"
        self._refresh_idle_episode(since_user)
        if since_user is None:
            return "wait - no user input yet"
        if since_assistant is None:
            return "wait - assistant has not replied yet"
        idle_for = min(since_user, since_assistant)
        if idle_for >= threshold:
            max_turns = self._config.idle_nudge_max_turns
            if self._idle_nudge_turns >= max_turns:
                return (
                    f"exhausted - already made {self._idle_nudge_turns} idle nudge attempt"
                    f"{'s' if self._idle_nudge_turns != 1 else ''} in this silence episode; "
                    "stay quiet until the user speaks"
                )
            turn_number = self._idle_nudge_turns + 1
            if reserve_turn:
                self._idle_nudge_turns = turn_number
            if turn_number >= max_turns:
                return (
                    f"final - silence has lasted {idle_for:.1f}s (threshold {threshold:.1f}s); "
                    f"idle nudge {turn_number}/{max_turns}. Required action: call `speak` with one brief "
                    "leave-you-to-it line that fits the conversation. Do not ask a question. "
                    "After this, stay silent until the user speaks."
                )
            return (
                f"eligible - silence has lasted {idle_for:.1f}s "
                f"(threshold {threshold:.1f}s); idle nudge {turn_number}/{max_turns}. "
                "Required action: call `speak` with a brief check-in or topic continuation. "
                "Do not repeat recent wording."
            )
        return f"wait - silence {idle_for:.1f}s below threshold {threshold:.1f}s"

    def _refresh_idle_episode(self, since_user: float | None) -> None:
        if since_user is None:
            self._idle_nudge_turns = 0
            self._last_seen_user_age_s = None
            return
        if self._last_seen_user_age_s is not None and since_user + 0.5 < self._last_seen_user_age_s:
            self._idle_nudge_turns = 0
        self._last_seen_user_age_s = since_user

    def _current_scene(self) -> str | None:
        if self._vision_state is None:
            return "camera disabled"
        return self._vision_state.snapshot()

    def _task_summary(self) -> str:
        slots = self._slot_store.list_slots()
        if not slots:
            return "none"
        lines = []
        for slot in slots:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            meta_parts = []
            if slot.importance is not None:
                meta_parts.append(f"importance={slot.importance:.2f}")
            if slot.confidence is not None:
                meta_parts.append(f"confidence={slot.confidence:.2f}")
            if slot.next_run is not None:
                meta_parts.append(f"next_run={slot.next_run:.0f}")
            meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"{slot.title}: {slot.status}{summary_text}{meta_text}")
        return "\n".join(lines)

    def _enqueue_llm(self, item: dict[str, Any]) -> bool:
        try:
            self._llm_queue.put_nowait(item)
        except queue.Full:
            return False
        return True

    def _pending_autonomy(self) -> bool:
        inflight = self._inflight_counter.value() if self._inflight_counter else 0
        try:
            queued = self._llm_queue.qsize()
        except NotImplementedError:
            queued = 0
        return (inflight + queued) > 0

    def update_slot(
        self,
        slot_id: str,
        title: str,
        status: str,
        summary: str,
        notify_user: bool = True,
        updated_at: float | None = None,
    ) -> None:
        self._slot_store.update_slot(
            slot_id=slot_id,
            title=title,
            status=status,
            summary=summary,
            notify_user=notify_user,
            updated_at=updated_at,
        )

    def _push_vision_emotion(self, event: VisionUpdateEvent) -> None:
        """Push a vision-related emotion event."""
        # Describe the scene change for emotional processing
        if event.prev_description and event.description:
            description = (
                f"{event.source} scene changed from '{event.prev_description}' to "
                f"'{event.description}' (change={event.change_score:.2f})"
            )
        elif event.description:
            description = f"New {event.source} scene observed: '{event.description}'"
        else:
            description = f"{event.source} scene change detected (change={event.change_score:.2f})"

        emotion_event = EmotionEvent(
            source="vision",
            description=description,
        )
        self._emotion_agent.push_event(emotion_event)
        logger.debug("Pushed vision emotion event: {}", description)
