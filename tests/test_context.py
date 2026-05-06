from datetime import datetime, timedelta, timezone

from glados.core.context import ContextBuilder, SessionClockContext, format_current_time_context


def test_time_context_contains_local_clock_details() -> None:
    now = datetime(2026, 5, 6, 23, 9, 12, tzinfo=timezone(timedelta(hours=8)))

    context = format_current_time_context(now)

    assert "[time]" in context
    assert "You have access to the local system clock" in context
    assert "Wednesday, May 06, 2026" in context
    assert "11:09 PM" in context
    assert "2026-05-06T23:09:12+08:00" in context


def test_context_builder_includes_time_source() -> None:
    builder = ContextBuilder()
    builder.register("time", lambda: format_current_time_context(datetime(2026, 5, 6, 23, 9, tzinfo=timezone.utc)))

    messages = builder.build_system_messages()

    assert messages == [
        {
            "role": "system",
            "content": format_current_time_context(datetime(2026, 5, 6, 23, 9, tzinfo=timezone.utc)),
        }
    ]


def test_session_context_contains_elapsed_time() -> None:
    started = datetime(2026, 5, 6, 23, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    now = datetime(2026, 5, 6, 23, 28, 34, tzinfo=timezone(timedelta(hours=8)))

    context = SessionClockContext(started).as_prompt(now)

    assert "[session]" in context
    assert "2026-05-06T23:00:00+08:00" in context
    assert "28 minutes, 34 seconds" in context
    assert "answer directly" in context
