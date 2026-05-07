import queue
import threading
import time

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.event_bus import EventBus
from glados.autonomy.interaction_state import InteractionState
from glados.autonomy.loop import AutonomyLoop
from glados.autonomy.slots import TaskSlotStore


def _loop(state: InteractionState, cooldown_s: float = 20.0) -> AutonomyLoop:
    return AutonomyLoop(
        config=AutonomyConfig(enabled=True, cooldown_s=cooldown_s),
        event_bus=EventBus(),
        interaction_state=state,
        vision_state=None,
        slot_store=TaskSlotStore(),
        llm_queue=queue.Queue(),
        processing_active_event=threading.Event(),
        currently_speaking_event=threading.Event(),
        shutdown_event=threading.Event(),
    )


def _set_interaction_times(
    state: InteractionState,
    *,
    user_age_s: float | None,
    assistant_age_s: float | None,
) -> None:
    now = time.time()
    with state._lock:
        state._last_user_ts = None if user_age_s is None else now - user_age_s
        state._last_assistant_ts = None if assistant_age_s is None else now - assistant_age_s


def test_autonomy_skips_when_user_turn_is_waiting_for_assistant() -> None:
    state = InteractionState()
    _set_interaction_times(state, user_age_s=35.0, assistant_age_s=60.0)

    assert _loop(state)._should_skip() is True


def test_autonomy_skips_during_user_cooldown_after_assistant_reply() -> None:
    state = InteractionState()
    _set_interaction_times(state, user_age_s=5.0, assistant_age_s=1.0)

    assert _loop(state, cooldown_s=20.0)._should_skip() is True


def test_autonomy_runs_after_user_cooldown_and_assistant_reply() -> None:
    state = InteractionState()
    _set_interaction_times(state, user_age_s=30.0, assistant_age_s=10.0)

    assert _loop(state, cooldown_s=20.0)._should_skip() is False


def test_idle_nudge_waits_below_threshold() -> None:
    state = InteractionState()
    _set_interaction_times(state, user_age_s=45.0, assistant_age_s=45.0)
    loop = _loop(state)

    assert "wait" in loop._idle_nudge_guidance(45.0, 45.0)


def test_idle_nudge_becomes_eligible_after_threshold() -> None:
    state = InteractionState()
    _set_interaction_times(state, user_age_s=75.0, assistant_age_s=75.0)
    loop = _loop(state)

    assert "eligible" in loop._idle_nudge_guidance(75.0, 75.0)
    assert "threshold" in loop._build_prompt(object())
