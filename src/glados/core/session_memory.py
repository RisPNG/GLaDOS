from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from ..autonomy.llm_client import LLMConfig
from ..autonomy.summarization import extract_facts, summarize_messages
from ..mcp.memory_server import Fact, Summary, _save_fact, _save_summary


@dataclass(frozen=True)
class SessionMemoryResult:
    summary_saved: bool
    facts_saved: int
    message_count: int


def filter_session_messages(
    messages: list[dict[str, Any]],
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Keep real user/assistant messages for long-term memory."""
    filtered: list[dict[str, Any]] = []
    for message in messages[start_index:]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content or content.startswith("[SYSTEM:"):
            continue
        filtered.append({"role": role, "content": content})
    return filtered


def persist_session_memory(
    messages: list[dict[str, Any]],
    llm_config: LLMConfig,
    started_at: datetime,
    ended_at: datetime | None = None,
    start_index: int = 0,
    min_messages: int = 2,
) -> SessionMemoryResult:
    """Persist a session summary and extracted facts for next launches."""
    session_messages = filter_session_messages(messages, start_index=start_index)
    if len(session_messages) < min_messages:
        return SessionMemoryResult(summary_saved=False, facts_saved=0, message_count=len(session_messages))

    ended_at = ended_at.astimezone() if ended_at is not None else datetime.now().astimezone()
    started_at = started_at.astimezone()

    summary = summarize_messages(session_messages, llm_config)
    summary_saved = False
    if summary:
        _save_summary(
            Summary(
                content=summary,
                period="session",
                start_time=started_at.isoformat(timespec="seconds"),
                end_time=ended_at.isoformat(timespec="seconds"),
            )
        )
        summary_saved = True

    facts_saved = 0
    facts = extract_facts(session_messages, llm_config)
    for fact_text in facts:
        text = fact_text.strip()
        if not text:
            continue
        _save_fact(Fact(content=text, source="conversation", importance=0.6))
        facts_saved += 1

    logger.info(
        "SessionMemory: saved summary={}, facts={}, messages={}",
        summary_saved,
        facts_saved,
        len(session_messages),
    )
    return SessionMemoryResult(
        summary_saved=summary_saved,
        facts_saved=facts_saved,
        message_count=len(session_messages),
    )
