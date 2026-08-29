"""
``mfv.journal`` -- recording what was done, so it can be written up.

The application keeps an append-only record of every transformation applied
during a session (``core/processing_journal.py``), and the *Generate Method
Text* plugin turns it into a methods paragraph. A plugin that records its run
here appears in that paragraph automatically -- the cheapest reproducibility
any analysis will get, at one line per run.

Record the **parameters that were actually used**, not the defaults, and record
them after the user has confirmed them.
"""

from __future__ import annotations

from typing import Any

from ._base import ApiError, Namespace

#: Journal categories, matching ``core/processing_journal.py``.
CATEGORIES = ("load", "filter", "transform", "analysis", "export", "plugin", "other")


class Journal(Namespace):
    """Append to the session's processing record."""

    name = "journal"

    def record(
        self,
        category: str,
        summary: str,
        *,
        dataset: Any = None,
        **details: Any,
    ) -> None:
        """
        Add one entry.

        *category* is one of :data:`CATEGORIES`. *summary* is a single sentence
        in the past tense, naming what was done. Keyword *details* are the
        operational parameters and become the parameter block in the generated
        method text, so they should be complete enough to reproduce the run.

        Also emits a Log line, so the entry is visible while the session is
        running rather than only at write-up time -- and because the method-text
        generator reads the Log history as well as the journal.
        """
        cat = str(category).lower()
        if cat not in CATEGORIES:
            raise ApiError(
                f"Unknown journal category {category!r}. One of: {', '.join(CATEGORIES)}"
            )
        if not summary:
            raise ApiError("A journal entry needs a one-sentence summary.")

        idx = None
        ds_name = ""
        if dataset is not None:
            ds = self._dataset(dataset)
            idx = self._dataset_index(ds)
            ds_name = ds.name

        self._state.journal.add(cat, str(summary), **details)

        detail_text = ", ".join(f"{k}={v}" for k, v in details.items())
        target = f" on {ds_name!r}" if ds_name else ""
        line = f"{summary}{target}"
        if detail_text:
            line = f"{line}: {detail_text}"
        self._state.log(line, "INFO", dataset_idx=idx)

    def entries(self, *, category: str | None = None) -> list:
        """Every recorded entry, optionally filtered to one category."""
        items = list(self._state.journal)
        if category is None:
            return items
        wanted = str(category).lower()
        return [e for e in items if getattr(e, "category", "") == wanted]
