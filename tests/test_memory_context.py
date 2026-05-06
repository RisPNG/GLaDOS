import json
import time

from glados.core import memory_context as memory_module
from glados.core.memory_context import MemoryContext


def test_memory_context_includes_auto_extracted_facts(tmp_path, monkeypatch) -> None:
    facts_file = tmp_path / "facts.jsonl"
    summaries_file = tmp_path / "summaries.jsonl"
    monkeypatch.setattr(memory_module, "FACTS_FILE", facts_file)
    monkeypatch.setattr(memory_module, "SUMMARIES_FILE", summaries_file)
    facts_file.write_text(
        json.dumps(
            {
                "content": "User likes long-running voice conversations",
                "source": "conversation",
                "importance": 0.6,
                "created_at": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompt = MemoryContext().as_prompt()

    assert prompt is not None
    assert "Long-term memory" in prompt
    assert "User likes long-running voice conversations" in prompt


def test_memory_context_includes_recent_summaries(tmp_path, monkeypatch) -> None:
    facts_file = tmp_path / "facts.jsonl"
    summaries_file = tmp_path / "summaries.jsonl"
    monkeypatch.setattr(memory_module, "FACTS_FILE", facts_file)
    monkeypatch.setattr(memory_module, "SUMMARIES_FILE", summaries_file)
    summaries_file.write_text(
        json.dumps(
            {
                "content": "User debugged GLaDOS time and autonomy behavior.",
                "period": "session",
                "start_time": "2026-05-06T23:00:00",
                "end_time": "2026-05-06T23:30:00",
                "created_at": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompt = MemoryContext().as_prompt()

    assert prompt is not None
    assert "Recent conversation summaries" in prompt
    assert "User debugged GLaDOS time and autonomy behavior." in prompt
