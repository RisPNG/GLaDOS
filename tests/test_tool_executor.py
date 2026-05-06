import json
import queue
import threading
import time
from unittest.mock import ANY, Mock

from glados.core.tool_executor import ToolExecutor


def _make_executor(
    llm_queue: queue.Queue,
    tool_calls_queue: queue.Queue,
    processing_active_event: threading.Event,
    shutdown_event: threading.Event,
) -> ToolExecutor:
    return ToolExecutor(
        llm_queue_priority=llm_queue,
        llm_queue_autonomy=queue.Queue(),
        tool_calls_queue=tool_calls_queue,
        processing_active_event=processing_active_event,
        shutdown_event=shutdown_event,
    )


def test_run_shutdown_event(caplog):
    """
    Test that the run method exits when shutdown event is set.
    """
    llm_queue = queue.Queue()
    tool_calls_queue = queue.Queue()
    processing_active_event = threading.Event()
    shutdown_event = threading.Event()
    shutdown_event.set()
    executor = _make_executor(llm_queue, tool_calls_queue, processing_active_event, shutdown_event)

    caplog.set_level("INFO")
    executor.run()

    # Check that the logger messages are present
    assert "ToolExecutor thread started." in caplog.text
    assert "ToolExecutor thread finished." in caplog.text


def test_tool_call_discarded_if_processing_inactive(mocker, caplog):
    """
    Test that a tool call is discarded if processing_active_event is not set.
    """
    mock_tool = Mock()
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {
        "test tool": mock_tool
    })
    test_tool = "test tool"
    llm_queue = queue.Queue()
    tool_calls_queue = queue.Queue()

    tool_call = {
        "function": {"name": test_tool, "arguments": "{}"},
        "id": "123"
    }

    tool_calls_queue.put(tool_call)
    processing_active_event = threading.Event()
    shutdown_event = threading.Event()

    executor = _make_executor(llm_queue, tool_calls_queue, processing_active_event, shutdown_event)
    thread = threading.Thread(target=executor.run)
    thread.start()

    timeout = 2
    start_time = time.time()
    expected_message = "ToolExecutor: Interruption signal active, discarding tool call."

    while time.time() - start_time < timeout:
        if expected_message in caplog.text:
            break
        time.sleep(0.1)

    # Check that the correct message is logged and no tool instance is created
    assert "ToolExecutor: Interruption signal active, discarding tool call." in caplog.text
    shutdown_event.set()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "Thread is still running after the test timeout"
    mock_tool.assert_not_called()


def test_process_valid_tool_call(mocker, caplog):
    """
    Test processing of a valid tool call.
    """
    mock_tool_instance = Mock()

    def run_tool(tool_call_id, args):
        tool_result_queue = mock_tool.call_args.kwargs["llm_queue"]
        tool_result_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "success",
                "type": "function_call_output",
            }
        )

    mock_tool_instance.run.side_effect = run_tool
    mock_tool = Mock(return_value=mock_tool_instance)
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {
        "test tool": mock_tool
    })
    test_tool = "test tool"
    llm_queue = queue.Queue()
    tool_calls_queue = queue.Queue()

    tool_call = {
        "function": {"name": test_tool, "arguments": "{\"key\": \"value\"}"},
        "id": "123"
    }

    tool_calls_queue.put(tool_call)
    processing_active_event = threading.Event()
    processing_active_event.set()
    shutdown_event = threading.Event()

    executor = _make_executor(llm_queue, tool_calls_queue, processing_active_event, shutdown_event)
    thread = threading.Thread(target=executor.run)
    thread.start()

    timeout = 2
    start_time = time.time()
    expected_message = "ToolExecutor: finished test tool"

    while time.time() - start_time < timeout:
        if expected_message in caplog.text:
            break
        time.sleep(0.1)

    assert "ToolExecutor: Received tool call" in caplog.text
    mock_tool.assert_called_once_with(llm_queue=ANY, tool_config={})
    mock_tool_instance.run.assert_called_once_with(tool_call["id"], json.loads(tool_call["function"]["arguments"]))
    output = llm_queue.get(timeout=1)
    assert output["role"] == "tool"
    assert output["tool_call_id"] == "123"
    assert output["content"] == "success"
    assert output["_lane"] == "priority"
    assert output["_allow_tools"] is False
    assert "autonomy" not in output
    shutdown_event.set()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "Thread is still running after the test timeout"

def test_json_decode_error(mocker, caplog):
    """
    Test handling of invalid JSON in tool arguments.
    """
    mock_tool_instance = Mock()
    mock_tool = Mock(return_value=mock_tool_instance)
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {
        "test tool": mock_tool
    })
    test_tool = "test tool"
    llm_queue = queue.Queue()
    tool_calls_queue = queue.Queue()

    tool_call = {
        "function": {"name": "test tool", "arguments": "invalid_json"},
        "id": "123"
    }

    tool_calls_queue.put(tool_call)
    processing_active_event = threading.Event()
    processing_active_event.set()
    shutdown_event = threading.Event()

    executor = _make_executor(llm_queue, tool_calls_queue, processing_active_event, shutdown_event)
    thread = threading.Thread(target=executor.run)
    thread.start()

    timeout = 2
    start_time = time.time()
    expected_message = "ToolExecutor: finished test tool"

    while time.time() - start_time < timeout:
        if expected_message in caplog.text:
            break
        time.sleep(0.1)

    assert "ToolExecutor: Failed to parse non-JSON tool call args: " in caplog.text
    mock_tool.assert_called_once_with(llm_queue=ANY, tool_config={})
    mock_tool_instance.run.assert_called_once_with(tool_call["id"], {})
    shutdown_event.set()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "Thread is still running after the test timeout"

def test_unknown_tool(mocker, caplog):
    """
    Test handling of an unknown tool (not in all_tools).
    """
    mock_tool_instance = Mock()
    mock_tool = Mock(return_value=mock_tool_instance)
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {
        "test tool": mock_tool
    })
    test_tool = "test tool"
    llm_queue = queue.Queue()
    tool_calls_queue = queue.Queue()

    tool_call = {
        "function": {"name": "unknown tool", "arguments": "{}"},
        "id": "123"
    }

    tool_calls_queue.put(tool_call)
    processing_active_event = threading.Event()
    processing_active_event.set()
    shutdown_event = threading.Event()

    executor = _make_executor(llm_queue, tool_calls_queue, processing_active_event, shutdown_event)
    thread = threading.Thread(target=executor.run)
    thread.start()

    timeout = 2
    start_time = time.time()
    expected_message = "ToolExecutor: error: no tool named unknown tool is available"

    while time.time() - start_time < timeout:
        if expected_message in caplog.text:
            break
        time.sleep(0.1)

    # Check that the error message is logged and the LLM queue is updated
    assert "ToolExecutor: error: no tool named unknown tool is available" in caplog.text
    output = llm_queue.get()
    assert output["role"] == "tool"
    assert output["tool_call_id"] == "123"
    assert output["content"] == "error: no tool named unknown tool is available"
    assert output["type"] == "function_call_output"
    assert output["_lane"] == "priority"
    assert output["_allow_tools"] is False
    assert isinstance(output["_enqueued_at"], float)
    mock_tool.assert_not_called()
    shutdown_event.set()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "Thread is still running after the test timeout"
