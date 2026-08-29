from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def facade(qt_app):
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.scripting import create_facade

    state = SimpleNamespace(datasets=[], active_dataset=None)
    api = create_facade(state)
    owner = QWidget()
    api.bind_main_window(owner)
    yield api, owner
    from minflux_viewer.ui.modeless import close_modeless

    close_modeless(owner)
    api.close_windows()
    owner.close()
    qt_app.processEvents()


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_named_table_data_operations_and_validation(facade):
    from minflux_viewer.api._base import ApiError

    results, _owner = facade
    table = results.results.table("Measurements")
    assert results.results.table("Measurements") is table

    assert table.add_row(id=1, score=2.5) is table
    table.add_rows([{"id": 2, "label": "two"}, {"id": 3, "score": 7.0}])
    table.add_column("accepted", [True, False, True, False])

    assert table.columns() == ["id", "score", "label", "accepted"]
    assert len(table) == 4
    assert table.column("id") == [1, 2, 3, None]
    assert table.rows()[1] == {
        "id": 2,
        "score": None,
        "label": "two",
        "accepted": False,
    }
    assert table.to_dict()["accepted"] == [True, False, True, False]

    table.from_arrays(x=[1, 2], y=[3.0, 4.0])
    assert table.to_dict() == {"x": [1, 2], "y": [3.0, 4.0]}
    table.clear()
    assert len(table) == 0
    assert table.columns() == ["x", "y"]

    with pytest.raises(ApiError, match="equal lengths"):
        table.from_arrays(x=[1], y=[2, 3])
    with pytest.raises(ApiError, match="no column"):
        table.column("missing")


def test_table_window_is_modeless_non_owned_sortable_and_copyable(facade, qt_app):
    from PyQt6.QtCore import QItemSelectionModel, Qt
    from PyQt6.QtGui import QFontDatabase

    results, owner = facade
    table = results.results.table("Sortable").from_arrays(value=[10, 2], label=["a", "b"])
    assert table.show() is table
    qt_app.processEvents()

    window = table._window
    assert window.parentWidget() is None
    assert window in owner._modeless_windows
    assert not window.isModal()
    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert window.table.isSortingEnabled()
    assert (
        window.table.font().family()
        == QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
    )

    window.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    qt_app.processEvents()
    model = window.table.model()
    assert model.index(0, 0).data() == "2"
    assert model.index(1, 0).data() == "10"

    selection = window.table.selectionModel()
    selection.select(
        model.index(0, 0),
        QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    selection.select(
        model.index(0, 1),
        QItemSelectionModel.SelectionFlag.Select,
    )
    window.table.copy_selection()
    assert qt_app.clipboard().text() == "2\tb"

    # Closing the presentation does not discard the named table; showing it
    # again recreates one window over the same data.
    window.close()
    qt_app.processEvents()
    assert results.results.table("Sortable") is table
    assert table.to_dict()["value"] == [10, 2]
    table.show()
    qt_app.processEvents()
    assert table._window is not window


def test_save_csv_and_explicit_close_forgets_table(facade, tmp_path):
    from minflux_viewer.api._base import ApiError

    results, _owner = facade
    table = results.results.table("CSV").add_rows(
        [{"name": "alpha", "value": 1.5}, {"name": "beta"}]
    )
    target = tmp_path / "table.csv"
    assert table.save_csv(str(target), separator=";") == str(target)
    with target.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream, delimiter=";")) == [
            ["name", "value"],
            ["alpha", "1.5"],
            ["beta", ""],
        ]

    assert results.results.tables() == ["CSV"]
    assert results.results.close("CSV")
    assert not results.results.close("CSV")
    assert results.results.tables() == []
    with pytest.raises(ApiError, match="closed"):
        table.add_row(value=2)


def test_results_window_child_process_teardown_is_clean(tmp_path):
    code = r"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication, QWidget
from minflux_viewer.scripting import create_facade
from minflux_viewer.ui.modeless import close_modeless

app = QApplication([])
owner = QWidget()
facade = create_facade(SimpleNamespace(datasets=[], active_dataset=None))
facade.bind_main_window(owner)
table = facade.results.table("Lifecycle").from_arrays(x=range(50), y=range(50))
table.show()
app.processEvents()
close_modeless(owner)
facade.close_windows()
owner.close()
app.processEvents()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "has been deleted" not in completed.stderr
