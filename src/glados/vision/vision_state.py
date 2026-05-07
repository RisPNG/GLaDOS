from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any


@dataclass(frozen=True)
class VisionSnapshot:
    source: str
    description: str
    key: str | None = None
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionState:
    """Thread-safe store for latest vision descriptions by source/key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[tuple[str, str | None], VisionSnapshot] = {}

    def update(
        self,
        description: str,
        source: str = "camera",
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the latest vision description."""
        with self._lock:
            self._snapshots[(source, key)] = VisionSnapshot(
                source=source,
                key=key,
                description=description,
                metadata=metadata or {},
            )

    def snapshots(self, source: str | None = None) -> list[VisionSnapshot]:
        """Return latest snapshots, optionally filtered by source."""
        with self._lock:
            values = list(self._snapshots.values())
        if source is not None:
            values = [snapshot for snapshot in values if snapshot.source == source]
        return sorted(values, key=lambda snapshot: (snapshot.source, snapshot.key or ""))

    def snapshot(self, source: str | None = None, key: str | None = None) -> str | None:
        """Return formatted latest vision description(s), if available."""
        with self._lock:
            if source is not None and key is not None:
                snapshot = self._snapshots.get((source, key))
                return snapshot.description if snapshot else None
            values = list(self._snapshots.values())

        if source is not None:
            values = [snapshot for snapshot in values if snapshot.source == source]
        if not values:
            return None

        values = sorted(values, key=lambda snapshot: (snapshot.source, snapshot.key or ""))
        return "\n".join(self._format_snapshot(snapshot) for snapshot in values)

    def as_message(self) -> dict[str, Any] | None:
        """Return the vision context as a system message or None if empty."""
        snapshots = self.snapshots()
        if not snapshots:
            return None
        content = "\n".join(self._format_snapshot(snapshot) for snapshot in snapshots)
        return {"role": "system", "content": content}

    @staticmethod
    def _format_snapshot(snapshot: VisionSnapshot) -> str:
        label = f"vision:{snapshot.source}"
        if snapshot.key:
            label = f"{label}:{snapshot.key}"
        return f"[{label}] {snapshot.description}"
