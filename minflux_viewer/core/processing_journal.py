"""
minflux_viewer.core.processing_journal
========================================
A small append-only log of every transformation applied to data during
a session, used by the *Generate Method Text* plugin to write a
methods-paragraph-style description of what the user did.

Entries are tagged by category (``"load"``, ``"filter"``, ``"transform"``,
``"analysis"``, ``"export"``, ``"plugin"``, ``"other"``) and carry an
optional dict of structured details.

The class is intentionally simple: a list of dataclass entries that
modules append to via :meth:`add` from anywhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class JournalEntry:
    timestamp: str
    category: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class ProcessingJournal:
    """Append-only history. Plain Python; no Qt dependency."""

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    # ------------------------------------------------------------------

    def add(
        self,
        category: str,
        summary: str,
        **details: Any,
    ) -> JournalEntry:
        """Append a new entry. Returns the entry for chaining."""
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            category=str(category),
            summary=str(summary),
            details=dict(details) if details else {},
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
