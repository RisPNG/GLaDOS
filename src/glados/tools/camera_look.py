import queue
from typing import Any

from loguru import logger

from ..vision.constants import CAMERA_DETAIL_PROMPT
from ..vision.vision_request import VisionRequest

tool_definition = {
    "type": "function",
    "function": {
        "name": "camera_look",
        "description": "Capture the current webcam/camera view and describe it.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional instruction for what to inspect in the camera view.",
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Maximum tokens to generate for the description.",
                },
            },
        },
    },
}


class CameraLook:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self._request_queue: queue.Queue[VisionRequest] | None = tool_config.get("camera_request_queue")
        self._timeout = float(tool_config.get("vision_tool_timeout", 30.0))
        self._default_prompt = tool_config.get("camera_detail_prompt", CAMERA_DETAIL_PROMPT)

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if self._request_queue is None:
            self._send_result(tool_call_id, "error: camera tool is unavailable")
            return

        prompt = str(call_args.get("prompt") or self._default_prompt).strip()
        max_tokens = self._parse_max_tokens(call_args.get("max_tokens", 256))
        response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        request = VisionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            response_queue=response_queue,
        )

        try:
            self._request_queue.put(request, timeout=self._timeout)
        except queue.Full:
            self._send_result(tool_call_id, "error: camera request queue is full")
            return

        try:
            result = response_queue.get(timeout=self._timeout)
        except queue.Empty:
            self._send_result(tool_call_id, "error: camera request timed out")
            return

        self._send_result(tool_call_id, result)

    def _send_result(self, tool_call_id: str, content: str) -> None:
        if content.startswith("error:"):
            logger.error("CameraLook: {}", content)
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )

    @staticmethod
    def _parse_max_tokens(value: Any) -> int:
        try:
            max_tokens = int(value)
        except (TypeError, ValueError):
            max_tokens = 256
        return max(1, min(max_tokens, 512))
