import queue
from typing import Any

tool_definition = {
    "type": "function",
    "function": {
        "name": "get_report",
        "description": (
            "Get a readable report from a subagent. Use this when the user asks for a report, "
            "status, details, or a summary from a background subagent such as weather, hn_top, "
            "emotion, compaction, or observer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the subagent (e.g., 'weather', 'hn_top', 'emotion', 'compaction', 'observer')",
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
        agent_id = str(call_args.get("agent_id", "")).strip()
        slot_store = self.tool_config.get("slot_store")

        if not slot_store:
            content = (
                "Report system unavailable: autonomy slots are not configured. "
                "Autonomy must be enabled before subagent reports can be retrieved."
            )
        elif not agent_id:
            content = "Report request failed: agent_id is required."
        else:
            slot = slot_store.get_slot(agent_id)
            if slot is None:
                available = ", ".join(sorted(s.slot_id for s in slot_store.list_slots()))
                if not available:
                    available = "none"
                content = (
                    f"No report slot found for agent '{agent_id}'. "
                    f"Available report slots: {available}."
                )
            else:
                meta_parts = []
                if slot.importance is not None:
                    meta_parts.append(f"importance={slot.importance:.2f}")
                if slot.confidence is not None:
                    meta_parts.append(f"confidence={slot.confidence:.2f}")
                if slot.next_run is not None:
                    meta_parts.append(f"next_run={slot.next_run:.0f}s")

                meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""
                report_text = slot.report.strip() if slot.report else ""

                if report_text:
                    content = (
                        f"Report for {slot.title} [{slot.slot_id}]\n"
                        f"Status: {slot.status}{meta_text}\n"
                        f"Summary: {slot.summary}\n\n"
                        f"{report_text}"
                    )
                else:
                    content = (
                        f"Report for {slot.title} [{slot.slot_id}]\n"
                        f"Status: {slot.status}{meta_text}\n"
                        f"Summary: {slot.summary}\n\n"
                        "No detailed report has been generated for this slot yet. "
                        "Use the summary above as the current report."
                    )

        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )