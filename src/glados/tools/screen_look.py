import queue
from typing import Any

from loguru import logger

from ..vision.constants import SCREEN_DETAIL_PROMPT
from ..vision.vision_request import VisionRequest

tool_definition = {
    "type": "function",
    "function": {
        "name": "screen_look",
        "description": "Capture the current monitor screen view and describe desktop content.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional instruction for what to inspect on the screen.",
                },
                "monitor": {
                    "type": "string",
                    "description": "Optional 1-based monitor id, such as '1' or '2'. Omit for all configured monitors.",
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Maximum tokens to generate for the description.",
                },
            },
        },
    },
}


class ScreenLook:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self._request_queue: queue.Queue[VisionRequest] | None = tool_config.get("screen_request_queue")
        self._timeout = float(tool_config.get("vision_tool_timeout", 30.0))
        self._default_prompt = tool_config.get("screen_detail_prompt", SCREEN_DETAIL_PROMPT)

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if self._request_queue is None:
            self._send_result(tool_call_id, "error: screen tool is unavailable")
            return

        prompt = str(call_args.get("prompt") or self._default_prompt).strip()
        monitor = self._parse_monitor(call_args.get("monitor"))
        max_tokens = self._parse_max_tokens(call_args.get("max_tokens", 256))
        response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        request = VisionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            response_queue=response_queue,
            monitor=monitor,
        )

        try:
            self._request_queue.put(request, timeout=self._timeout)
        except queue.Full:
            self._send_result(tool_call_id, "error: screen request queue is full")
            return

        try:
            result = response_queue.get(timeout=self._timeout)
        except queue.Empty:
            self._send_result(tool_call_id, "error: screen request timed out")
            return

        self._send_result(tool_call_id, result)

    def _send_result(self, tool_call_id: str, content: str) -> None:
        if content.startswith("error:"):
            logger.error("ScreenLook: {}", content)
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )

    @staticmethod
    def _parse_monitor(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text or text == "all":
            return None
        try:
            monitor = int(text)
        except ValueError:
            return None
        return monitor if monitor > 0 else None

    @staticmethod
    def _parse_max_tokens(value: Any) -> int:
        try:
            max_tokens = int(value)
        except (TypeError, ValueError):
            max_tokens = 256
        return max(1, min(max_tokens, 512))
