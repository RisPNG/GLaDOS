from __future__ import annotations

from typing import Final

# Instructions for the LLM to handle vision messages from the vision module.
# These instructions are essential for proper integration of vision observations into the conversation.
SYSTEM_PROMPT_VISION_HANDLING: Final[str] = (
    "Important vision instructions: "
    "- You may receive visual snapshots in system messages prefixed with '[vision:camera]' and '[vision:screen:<monitor>]'. Treat them as context, not user messages. "
    "- Keep camera observations separate from screen observations. Camera means the physical webcam view; screen means monitor desktop content. "
    "- Do not respond directly to visual snapshots unless the user asks about them or autonomous mode decides the event is clearly useful. "
    "- For fresh physical-world inspection, call `camera_look` with a short prompt. "
    "- For fresh desktop/monitor inspection, call `screen_look` with a short prompt. "
    "- Use visual snapshots to ground answers, mentioning only relevant or changed elements."
)

# Default prompts for FastVLM inference.
CAMERA_DEFAULT_PROMPT: Final[str] = "Describe the camera image briefly, focusing on salient elements."
CAMERA_DETAIL_PROMPT: Final[str] = "Describe the camera image in detail."
SCREEN_DEFAULT_PROMPT: Final[str] = (
    "Describe this desktop screenshot briefly. Focus on active apps, visible errors, notifications, "
    "important text, and meaningful changes. Ignore wallpaper and minor UI chrome."
)
SCREEN_DETAIL_PROMPT: Final[str] = (
    "Inspect this desktop screenshot in detail. Read visible errors or important text if possible, "
    "identify active apps/windows, and summarize only useful desktop context."
)

VISION_DEFAULT_PROMPT: Final[str] = CAMERA_DEFAULT_PROMPT
VISION_DETAIL_PROMPT: Final[str] = CAMERA_DETAIL_PROMPT
