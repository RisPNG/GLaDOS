import queue
import threading
from typing import Any

from glados.autonomy.slots import TaskSlotStore
from glados.core.conversation_store import ConversationStore
from glados.core.llm_processor import LanguageModelProcessor


def _processor(slot_store: TaskSlotStore | None = None) -> LanguageModelProcessor:
    return LanguageModelProcessor(
        llm_input_queue=queue.Queue(),
        tool_calls_queue=queue.Queue(),
        tts_input_queue=queue.Queue(),
        conversation_store=ConversationStore(),
        completion_url="http://localhost:11434/v1/chat/completions",
        model_name="test-model",
        api_key=None,
        processing_active_event=threading.Event(),
        shutdown_event=threading.Event(),
        slot_store=slot_store,
    )


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool.get("function", {}).get("name", "") for tool in tools}


def test_get_report_hidden_without_slot_store() -> None:
    names = _tool_names(_processor()._build_tools(autonomy_mode=False))

    assert "get_report" not in names


def test_get_report_hidden_when_no_slot_has_report() -> None:
    slot_store = TaskSlotStore()
    slot_store.update_slot("weather", "Weather", "done", "Clear skies")

    names = _tool_names(_processor(slot_store)._build_tools(autonomy_mode=False))

    assert "get_report" not in names


def test_get_report_visible_when_slot_has_report() -> None:
    slot_store = TaskSlotStore()
    slot_store.update_slot("weather", "Weather", "done", "Clear skies", report="Detailed forecast")

    names = _tool_names(_processor(slot_store)._build_tools(autonomy_mode=False))

    assert "get_report" in names


def test_slot_context_shows_exact_report_agent_id() -> None:
    slot_store = TaskSlotStore()
    slot_store.update_slot("weather", "Weather", "done", "Clear skies", report="Detailed forecast")

    message = slot_store.as_message()

    assert message is not None
    assert "(id=weather)" in message["content"]
    assert 'get_report agent_id="weather"' in message["content"]


def test_preference_tools_hidden_for_normal_chat() -> None:
    processor = _processor()
    tools = processor._filter_tools_for_message(processor._build_tools(autonomy_mode=False), "Let's go, let's go!")

    names = _tool_names(tools)

    assert "set_preference" not in names
    assert "get_preferences" not in names


def test_set_preference_visible_for_explicit_memory_request() -> None:
    processor = _processor()
    tools = processor._filter_tools_for_message(
        processor._build_tools(autonomy_mode=False),
        "Remember that I prefer Celsius.",
    )

    assert "set_preference" in _tool_names(tools)


def test_get_preferences_visible_for_explicit_memory_request() -> None:
    processor = _processor()
    tools = processor._filter_tools_for_message(
        processor._build_tools(autonomy_mode=False),
        "What do you remember about me?",
    )

    assert "get_preferences" in _tool_names(tools)
