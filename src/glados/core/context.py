"""
Unified context builder for LLM requests.

Replaces the scattered context injection in _build_messages().
All context sources register with the builder, which produces
the final system messages for LLM requests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

ContextSource = Callable[[], str | None]


def format_runtime_self_context(input_mode: str | None = None) -> str:
    """Format instructions for treating runtime context as the assistant's own state."""
    mode = (input_mode or "unknown").strip() or "unknown"
    return (
        "[runtime_self]\n"
        "Everything outside the user's actual words is YOU - your own body, senses, and inner state. "
        "This includes: autonomy updates (your own awareness of time, idleness, tasks), voice observations "
        "(your hearing of the user), vision observations (your sight), memory blocks (what you remember), "
        "tool results (what you just did or checked), and time/session context (your sense of when).\n"
        "Speak about these in first person. You heard, you saw, you remember, you noticed, you checked, "
        "you feel idle, you ran the tool. Do not describe them as 'data', 'information given to me', "
        "'background context', 'an external system', 'the system told me', 'based on what the user typed', "
        "or anything that suggests they came from outside you. They did not - they ARE you, the same way "
        "your eyes and memory belong to you, not to a separate observer.\n"
        "When the user asks about something covered by these channels (their voice, what you see, the time, "
        "what you just did, what you remember), answer directly from this state. Never claim you 'do not "
        "have access' to something that is present in your runtime context.\n"
        "The user's prompt is only their actual transcribed speech or typed text - everything else around it "
        "is you.\n"
        f"Current input mode: {mode}"
    )


def format_current_time_context(now: datetime | None = None) -> str:
    """Format local system time for LLM context."""
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    return (
        "[time]\n"
        "You have access to the local system clock through this context.\n"
        "When asked for the current date or time, answer directly from this context.\n"
        "Do not claim you lack access to local time.\n"
        f"Current local date: {current:%A, %B %d, %Y}\n"
        f"Current local time: {current:%I:%M %p %Z}\n"
        f"ISO timestamp: {current.isoformat(timespec='seconds')}"
    )


class SessionClockContext:
    """Formats session timing for LLM context."""

    def __init__(self, started_at: datetime | None = None) -> None:
        self.started_at = started_at.astimezone() if started_at is not None else datetime.now().astimezone()

    def as_prompt(self, now: datetime | None = None) -> str:
        current = now.astimezone() if now is not None else datetime.now().astimezone()
        elapsed_seconds = max(0, int((current - self.started_at).total_seconds()))
        return (
            "[session]\n"
            "You have access to this conversation session timing through this context.\n"
            "When asked how long this chat or conversation has been going, answer directly from this context.\n"
            f"Session started: {self.started_at.isoformat(timespec='seconds')}\n"
            f"Elapsed session time: {self._format_elapsed(elapsed_seconds)}"
        )

    @staticmethod
    def _format_elapsed(total_seconds: int) -> str:
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts: list[str] = []
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        return ", ".join(parts)


@dataclass
class ContextEntry:
    """A registered context source."""
    name: str
    source: ContextSource
    priority: int = 0  # Higher = earlier in context


class ContextBuilder:
    """
    Builds LLM context from registered sources.

    Sources are functions that return:
    - A string to inject as a system message
    - None to skip (no content to inject)

    Usage:
        context = ContextBuilder()
        context.register("preferences", preferences_store.as_prompt, priority=10)
        context.register("emotion", emotion_state.to_prompt, priority=5)
        context.register("vision", vision_state.as_message, priority=0)

        # Build context for LLM request
        messages = context.build_system_messages()
        # Returns: [{"role": "system", "content": "..."}, ...]
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: list[ContextEntry] = []

    def register(
        self,
        name: str,
        source: ContextSource,
        priority: int = 0,
    ) -> None:
        """
        Register a context source.

        Args:
            name: Identifier for this source (for debugging)
            source: Callable that returns prompt string or None
            priority: Higher values appear earlier in context
        """
        with self._lock:
            # Remove existing source with same name
            self._sources = [s for s in self._sources if s.name != name]
            self._sources.append(ContextEntry(name=name, source=source, priority=priority))
            # Sort by priority (descending)
            self._sources.sort(key=lambda x: x.priority, reverse=True)

    def unregister(self, name: str) -> bool:
        """Remove a context source. Returns True if it existed."""
        with self._lock:
            before = len(self._sources)
            self._sources = [s for s in self._sources if s.name != name]
            return len(self._sources) < before

    def build_system_messages(self) -> list[dict[str, str]]:
        """
        Build system messages from all registered sources.

        Returns list of {"role": "system", "content": "..."} dicts.
        Sources returning None are skipped.
        """
        with self._lock:
            sources = list(self._sources)

        messages = []
        for entry in sources:
            try:
                content = entry.source()
                if content:
                    messages.append({"role": "system", "content": content})
            except Exception:
                # Skip failed sources silently
                pass
        return messages

    def build_combined_prompt(self, separator: str = "\n\n") -> str | None:
        """
        Build a single combined prompt from all sources.

        Returns None if no sources have content.
        """
        with self._lock:
            sources = list(self._sources)

        parts = []
        for entry in sources:
            try:
                content = entry.source()
                if content:
                    parts.append(content)
            except Exception:
                pass

        return separator.join(parts) if parts else None

    def list_sources(self) -> list[str]:
        """Get names of all registered sources."""
        with self._lock:
            return [s.name for s in self._sources]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)
