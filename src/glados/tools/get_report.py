import queue
from typing import Any

tool_definition = {
    "type": "function",
    "function": {
        "name": "get_report",
        "description": (
            "Get a detailed report for an autonomy task slot that is already listed in [tasks] with "
            "'report available'. Do not use for conversation history, memory, or prior chat lookup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Exact task slot id shown in [tasks] (e.g., 'weather', 'hn_top', 'emotion').",
                }
            },
            "required": ["agent_id"],
        },
    },
}


class GetReport:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.tool_config = tool_config or {}

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        agent_id = call_args.get("agent_id", "")
        slot_store = self.tool_config.get("slot_store")

        if not slot_store:
            content = "No autonomy task slots are available. This tool is not for conversation history."
        elif not agent_id:
            content = "agent_id is required. Use the exact task slot id from [tasks]."
        else:
            slot = slot_store.get_slot(agent_id)
            if slot is None:
                content = f"No task slot found for agent_id '{agent_id}'. This tool is not for conversation history."
            elif slot.report:
                content = slot.report
            else:
                content = f"No detailed report available for '{agent_id}'. Summary: {slot.summary}"

        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )
