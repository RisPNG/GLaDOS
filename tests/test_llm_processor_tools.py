import queue
import threading
import time
from typing import Any

from glados.autonomy.slots import TaskSlotStore
from glados.core.context import ContextBuilder, SessionClockContext, format_current_time_context
from glados.core.conversation_store import ConversationStore
from glados.core.llm_processor import LanguageModelProcessor
from glados.vision.vision_state import VisionState


def _processor(
    slot_store: TaskSlotStore | None = None,
    context_builder: ContextBuilder | None = None,
    vision_sources: set[str] | None = None,
) -> LanguageModelProcessor:
    vision_state = VisionState() if vision_sources else None
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
        vision_state=vision_state,
        slot_store=slot_store,
        context_builder=context_builder,
        vision_sources=vision_sources,
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


def test_visual_tools_are_source_gated() -> None:
    names = _tool_names(_processor(vision_sources={"camera"})._build_tools(autonomy_mode=False))

    assert "camera_look" in names
    assert "screen_look" not in names


def test_screen_tool_visible_for_screen_question() -> None:
    processor = _processor(vision_sources={"screen"})
    tools = processor._filter_tools_for_message(
        processor._build_tools(autonomy_mode=False),
        "What error is on my screen?",
    )

    assert "screen_look" in _tool_names(tools)


def test_camera_tool_visible_for_camera_question() -> None:
    processor = _processor(vision_sources={"camera"})
    tools = processor._filter_tools_for_message(
        processor._build_tools(autonomy_mode=False),
        "What do I look like on camera?",
    )

    assert "camera_look" in _tool_names(tools)


def test_priority_messages_include_registered_time_context() -> None:
    builder = ContextBuilder()
    builder.register("time", lambda: format_current_time_context())
    messages = _processor(context_builder=builder)._build_messages(autonomy_mode=False)

    assert any(
        message["role"] == "system"
        and "[time]" in message["content"]
        and "local system clock" in message["content"]
        for message in messages
    )


def test_priority_messages_include_registered_session_context() -> None:
    builder = ContextBuilder()
    builder.register("session", SessionClockContext().as_prompt)
    messages = _processor(context_builder=builder)._build_messages(autonomy_mode=False)

    assert any(
        message["role"] == "system"
        and "[session]" in message["content"]
        and "Elapsed session time" in message["content"]
        for message in messages
    )


def test_autonomy_tool_calls_do_not_pollute_conversation_history() -> None:
    processor = _processor()
    tool_call = {"function": {"name": "do_nothing", "arguments": "{}"}, "id": "call_1"}

    processor._process_tool_call([tool_call], autonomy_mode=True, tool_names={"do_nothing"})

    assert processor._conversation_store.snapshot() == []
    queued = processor.tool_calls_queue.get(timeout=1)
    assert queued["autonomy"] is True
    assert queued["function"]["name"] == "do_nothing"


def test_autonomy_tool_result_does_not_trigger_followup_llm_call(mocker) -> None:
    processor = _processor()
    processor.processing_active_event.set()
    post_mock = mocker.patch("glados.core.llm_processor.requests.post")
    processor.llm_input_queue.put(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "success",
            "type": "function_call_output",
            "autonomy": True,
            "_allow_tools": False,
            "_terminal_tool_result": True,
        }
    )

    thread = threading.Thread(target=processor.run)
    thread.start()

    deadline = time.time() + 2
    while time.time() < deadline and not processor.llm_input_queue.empty():
        time.sleep(0.01)

    processor.shutdown_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    post_mock.assert_not_called()
    assert processor._conversation_store.snapshot() == []
