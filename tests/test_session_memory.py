from datetime import datetime, timezone

from glados.autonomy.llm_client import LLMConfig
from glados.core.session_memory import filter_session_messages, persist_session_memory


def test_filter_session_messages_skips_preprompt_and_internal_messages() -> None:
    messages = [
        {"role": "system", "content": "Persona"},
        {"role": "user", "content": "Style example only"},
        {"role": "assistant", "content": "Example"},
        {"role": "user", "content": "What did we discuss?"},
        {"role": "assistant", "content": "We discussed memory."},
        {"role": "user", "content": "[SYSTEM: User interrupted mid-response!]"},
        {"role": "tool", "content": "success"},
    ]

    filtered = filter_session_messages(messages, start_index=3)

    assert filtered == [
        {"role": "user", "content": "What did we discuss?"},
        {"role": "assistant", "content": "We discussed memory."},
    ]


def test_persist_session_memory_saves_summary_and_facts(mocker) -> None:
    saved_summaries = []
    saved_facts = []
    mocker.patch("glados.core.session_memory.summarize_messages", return_value="User discussed persistent memory.")
    mocker.patch("glados.core.session_memory.extract_facts", return_value=["User wants natural recall across sessions"])
    mocker.patch("glados.core.session_memory._save_summary", side_effect=saved_summaries.append)
    mocker.patch("glados.core.session_memory._save_fact", side_effect=saved_facts.append)

    result = persist_session_memory(
        [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "Can memory work every session?"},
            {"role": "assistant", "content": "Yes, by saving session summaries."},
        ],
        LLMConfig(url="http://localhost:12330/v1/chat/completions", model="test"),
        started_at=datetime(2026, 5, 7, 1, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
        start_index=1,
    )

    assert result.summary_saved is True
    assert result.facts_saved == 1
    assert result.message_count == 2
    assert saved_summaries[0].content == "User discussed persistent memory."
    assert saved_summaries[0].period == "session"
    assert saved_facts[0].content == "User wants natural recall across sessions"
    assert saved_facts[0].importance == 0.6


def test_persist_session_memory_skips_short_sessions(mocker) -> None:
    summarize = mocker.patch("glados.core.session_memory.summarize_messages")

    result = persist_session_memory(
        [{"role": "user", "content": "Hi"}],
        LLMConfig(url="http://localhost:12330/v1/chat/completions", model="test"),
        started_at=datetime(2026, 5, 7, 1, 0, tzinfo=timezone.utc),
    )

    summarize.assert_not_called()
    assert result.summary_saved is False
    assert result.facts_saved == 0
    assert result.message_count == 1
