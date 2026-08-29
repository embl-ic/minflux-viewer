"""
``mfv.results`` -- the shared results table.

Implemented by extension-layer **track C**.

This is the ImageJ ``ResultsTable`` analogue and the keystone the current
scripting API is missing: without it every analysis invents its own
``QTableWidget``, which is how this application ended up with a dozen bespoke
ones. A plugin should put numbers here.

Semantics follow ImageJ where they are useful: :meth:`Results.table` with the
same *name* returns the **same** table, so a plugin run twice appends to the
table the user already has open rather than stacking windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ._base import Namespace, _todo

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_TRACK = "C"


class ResultsTable:
    """
    A named, sortable table of results, backed by a modeless viewer window.

    Columns are created on first use, so rows with differing keys are allowed;
    a cell never written reads back as empty. Values may be numbers or strings.
    """

    #: The table name, and the key under which the facade caches it.
    name: str

    def add_row(self, **columns: Any) -> "ResultsTable":
        """Append one row. Returns self, so calls can be chained."""
        raise _todo(_TRACK, "ResultsTable.add_row()")

    def add_rows(self, rows: "Sequence[Mapping[str, Any]]") -> "ResultsTable":
        """Append many rows at once -- far cheaper than a loop of :meth:`add_row`."""
        raise _todo(_TRACK, "ResultsTable.add_rows()")

    def add_column(self, name: str, values: "Sequence[Any] | None" = None) -> "ResultsTable":
        """Add a column, optionally filling it. Extends the table if *values* is longer."""
        raise _todo(_TRACK, "ResultsTable.add_column()")

    def from_arrays(self, **columns: Any) -> "ResultsTable":
        """Replace the table contents with equal-length column arrays."""
        raise _todo(_TRACK, "ResultsTable.from_arrays()")

    # -- reading back --------------------------------------------------------

    def columns(self) -> list[str]:
        """Column names, in display order."""
        raise _todo(_TRACK, "ResultsTable.columns()")

    def column(self, name: str) -> list:
        """One column of values."""
        raise _todo(_TRACK, "ResultsTable.column()")

    def rows(self) -> list[dict]:
        """Every row as a dict."""
        raise _todo(_TRACK, "ResultsTable.rows()")

    def to_dict(self) -> dict[str, list]:
        """The whole table as ``{column: values}`` -- ready for ``pandas.DataFrame``."""
        raise _todo(_TRACK, "ResultsTable.to_dict()")

    def __len__(self) -> int:
        """Number of rows."""
        raise _todo(_TRACK, "ResultsTable.__len__()")

    # -- lifecycle -----------------------------------------------------------

    def clear(self) -> "ResultsTable":
        """Remove every row, keeping the window and the column layout."""
        raise _todo(_TRACK, "ResultsTable.clear()")

    def show(self) -> "ResultsTable":
        """Show and raise the table window."""
        raise _todo(_TRACK, "ResultsTable.show()")

    def hide(self) -> "ResultsTable":
        """Hide the window without discarding the data."""
        raise _todo(_TRACK, "ResultsTable.hide()")

    def close(self) -> None:
        """Close the window and forget the table."""
        raise _todo(_TRACK, "ResultsTable.close()")

    def save_csv(self, path: str, *, separator: str = ",") -> str:
        """Write the table to *path*. Returns the path written."""
        raise _todo(_TRACK, "ResultsTable.save_csv()")


class Results(Namespace):
    """Create and find results tables."""

    name = "results"

    def table(self, name: str = "Results") -> ResultsTable:
        """
        The table called *name*, created on first use.

        The same name always returns the same table, so a plugin can append
        across runs without asking whether it already exists.
        """
        raise _todo(_TRACK, "results.table()")

    def tables(self) -> list[str]:
        """Names of every open table."""
        raise _todo(_TRACK, "results.tables()")

    def close(self, name: str) -> bool:
        """Close one table by name. Returns whether it existed."""
        raise _todo(_TRACK, "results.close()")

    def close_all(self) -> None:
        """Close every table this session opened."""
        raise _todo(_TRACK, "results.close_all()")
