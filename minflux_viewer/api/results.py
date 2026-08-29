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

import csv
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._base import ApiError, Namespace


class ResultsTable:
    """
    A named, sortable table of results, backed by a modeless viewer window.

    Columns are created on first use, so rows with differing keys are allowed;
    a cell never written reads back as empty. Values may be numbers or strings.
    """

    def __init__(self, namespace: Results, name: str) -> None:
        self._namespace = namespace
        self.name = name
        self._columns: list[str] = []
        self._rows: list[dict[str, Any]] = []
        self._window = None
        self._window_generation = 0
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise ApiError(f"Results table {self.name!r} has been closed.")

    @staticmethod
    def _column_name(name: Any) -> str:
        text = str(name).strip()
        if not text:
            raise ApiError("A results-table column name cannot be empty.")
        return text

    @staticmethod
    def _values(values: Any, *, column: str) -> list[Any]:
        shape = getattr(values, "shape", None)
        if shape is not None and len(shape) != 1:
            raise ApiError(f"Results column {column!r} must be a one-dimensional sequence.")
        if isinstance(values, (str, bytes)):
            raise ApiError(f"Results column {column!r} must be a one-dimensional sequence.")
        try:
            return list(values)
        except TypeError:
            raise ApiError(
                f"Results column {column!r} must be a one-dimensional sequence."
            ) from None

    def _notify_changed(self) -> None:
        if self._window is not None:
            from ..ui.qt_lifecycle import qobject_alive

            if qobject_alive(self._window):
                self._window.refresh()
            else:
                self._window = None

    def add_row(self, **columns: Any) -> ResultsTable:
        """Append one row. Returns self, so calls can be chained."""
        self._require_open()
        row: dict[str, Any] = {}
        for raw_name, value in columns.items():
            name = self._column_name(raw_name)
            if name not in self._columns:
                self._columns.append(name)
            row[name] = value
        self._rows.append(row)
        self._notify_changed()
        return self

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> ResultsTable:
        """Append many rows at once -- far cheaper than a loop of :meth:`add_row`."""
        self._require_open()
        pending: list[dict[str, Any]] = []
        new_columns: list[str] = []
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise ApiError("Every results-table row must be a mapping.")
            row = {}
            for raw_name, value in raw_row.items():
                name = self._column_name(raw_name)
                if name not in self._columns and name not in new_columns:
                    new_columns.append(name)
                row[name] = value
            pending.append(row)
        self._columns.extend(new_columns)
        self._rows.extend(pending)
        self._notify_changed()
        return self

    def add_column(self, name: str, values: Sequence[Any] | None = None) -> ResultsTable:
        """Add a column, optionally filling it. Extends the table if *values* is longer."""
        self._require_open()
        name = self._column_name(name)
        if name not in self._columns:
            self._columns.append(name)
        if values is not None:
            column = self._values(values, column=name)
            while len(self._rows) < len(column):
                self._rows.append({})
            for row, value in zip(self._rows, column):
                row[name] = value
        self._notify_changed()
        return self

    def from_arrays(self, **columns: Any) -> ResultsTable:
        """Replace the table contents with equal-length column arrays."""
        self._require_open()
        parsed: list[tuple[str, list[Any]]] = []
        lengths: set[int] = set()
        for raw_name, values in columns.items():
            name = self._column_name(raw_name)
            column = self._values(values, column=name)
            parsed.append((name, column))
            lengths.add(len(column))
        if len(lengths) > 1:
            sizes = ", ".join(str(size) for size in sorted(lengths))
            raise ApiError(f"Results-table columns must have equal lengths; got {sizes}.")
        length = next(iter(lengths), 0)
        self._columns[:] = [name for name, _values in parsed]
        self._rows[:] = [{name: values[row] for name, values in parsed} for row in range(length)]
        self._notify_changed()
        return self

    # -- reading back --------------------------------------------------------

    def columns(self) -> list[str]:
        """Column names, in display order."""
        self._require_open()
        return list(self._columns)

    def column(self, name: str) -> list:
        """One column of values."""
        self._require_open()
        name = self._column_name(name)
        if name not in self._columns:
            raise ApiError(f"Results table {self.name!r} has no column {name!r}.")
        return [row.get(name) for row in self._rows]

    def rows(self) -> list[dict]:
        """Every row as a dict."""
        self._require_open()
        return [{name: row.get(name) for name in self._columns} for row in self._rows]

    def to_dict(self) -> dict[str, list]:
        """The whole table as ``{column: values}`` -- ready for ``pandas.DataFrame``."""
        self._require_open()
        return {name: self.column(name) for name in self._columns}

    def __len__(self) -> int:
        """Number of rows."""
        self._require_open()
        return len(self._rows)

    # -- lifecycle -----------------------------------------------------------

    def clear(self) -> ResultsTable:
        """Remove every row, keeping the window and the column layout."""
        self._require_open()
        self._rows.clear()
        self._notify_changed()
        return self

    def show(self) -> ResultsTable:
        """Show and raise the table window."""
        self._require_open()
        from ..ui.modeless import show_modeless
        from ..ui.qt_lifecycle import qobject_alive
        from ..ui.results_table import ResultsTableWindow

        owner = self._namespace._main_window()
        if self._window is None or not qobject_alive(self._window):
            self._window_generation += 1
            generation = self._window_generation
            self._window = ResultsTableWindow(
                self.name,
                self._columns,
                self._rows,
                self.save_csv,
            )
            table_ref = weakref.ref(self)

            def _destroyed(*_args, g=generation) -> None:
                table = table_ref()
                if table is not None:
                    table._forget_window(g)

            self._window.destroyed.connect(_destroyed)
            show_modeless(self._window, owner)
        else:
            self._window.show()
            self._window.raise_()
            self._window.activateWindow()
        return self

    def _forget_window(self, generation: int) -> None:
        if generation == self._window_generation:
            self._window = None

    def hide(self) -> ResultsTable:
        """Hide the window without discarding the data."""
        self._require_open()
        if self._window is not None:
            try:
                self._window.hide()
            except RuntimeError:
                self._window = None
        return self

    def close(self) -> None:
        """Close the window and forget the table."""
        self._namespace.close(self.name)

    def _dispose(self) -> None:
        if self._closed:
            return
        self._closed = True
        window, self._window = self._window, None
        if window is not None:
            try:
                window.close()
            except RuntimeError:
                pass

    def save_csv(self, path: str, *, separator: str = ",") -> str:
        """Write the table to *path*. Returns the path written."""
        self._require_open()
        if not isinstance(separator, str) or len(separator) != 1:
            raise ApiError("CSV separator must be exactly one character.")
        target = Path(path).expanduser()
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=separator)
            writer.writerow(self._columns)
            for row in self._rows:
                writer.writerow(
                    "" if row.get(name) is None else row.get(name) for name in self._columns
                )
        return str(target)


class Results(Namespace):
    """Create and find results tables."""

    name = "results"

    def __init__(self, facade: Any) -> None:
        super().__init__(facade)
        self._tables: dict[str, ResultsTable] = {}

    def table(self, name: str = "Results") -> ResultsTable:
        """
        The table called *name*, created on first use.

        The same name always returns the same table, so a plugin can append
        across runs without asking whether it already exists.
        """
        clean = " ".join(str(name).split())
        if not clean:
            raise ApiError("A results table needs a non-empty name.")
        table = self._tables.get(clean)
        if table is None:
            table = self._tables[clean] = ResultsTable(self, clean)
        return table

    def tables(self) -> list[str]:
        """Names of every open table."""
        return list(self._tables)

    def close(self, name: str) -> bool:
        """Close one table by name. Returns whether it existed."""
        clean = " ".join(str(name).split())
        table = self._tables.pop(clean, None)
        if table is None:
            return False
        table._dispose()
        return True

    def close_all(self) -> None:
        """Close every table this session opened."""
        tables = list(self._tables.values())
        self._tables.clear()
        for table in tables:
            table._dispose()
