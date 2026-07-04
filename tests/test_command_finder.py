"""Fiji-style command finder: index collection, ranking, and the dialog."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _menubar(_app):
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import QMenuBar
    mb = QMenuBar()
    file_menu = mb.addMenu("&File")
    a_open = file_menu.addAction("&Open…")
    a_open.setShortcut(QKeySequence("Ctrl+O"))
    file_menu.addSeparator()
    file_menu.addAction("&Quit")
    analyze = mb.addMenu("&Analyze")
    seg = analyze.addMenu("Segmentation")
    npc = seg.addMenu("NPC")
    a_npc2d = npc.addAction("2D")
    a_off = analyze.addAction("Local Density")
    a_off.setEnabled(False)
    return mb, {"open": a_open, "npc2d": a_npc2d, "off": a_off}


def test_collect_paths_shortcuts_and_cleaning(_app):
    from minflux_viewer.ui.command_finder import collect_commands
    mb, _ = _menubar(_app)
    by_text = {e.text: e for e in collect_commands(mb)}
    assert by_text["Open"].path == "File"                  # '&' + '…' stripped
    assert by_text["Open"].shortcut == "Ctrl+O"
    assert by_text["2D"].path == "Analyze › Segmentation › NPC"
    assert by_text["Local Density"].enabled is False
    # submenu-opening actions are not commands themselves
    assert "Segmentation" not in by_text and "NPC" not in by_text


def test_filter_ranks_name_above_path(_app):
    from minflux_viewer.ui.command_finder import collect_commands, filter_commands
    entries = collect_commands(_menubar(_app)[0])
    res = filter_commands(entries, "npc")                  # 'NPC' only in the path
    assert res and res[0].text == "2D"
    assert filter_commands(entries, "open")[0].text == "Open"
    assert len(filter_commands(entries, "")) == len(entries)   # empty → all
    assert filter_commands(entries, "zzz-nope") == []


def test_dialog_filters_lists_and_runs(_app):
    from minflux_viewer.ui.command_finder import CommandFinderDialog, collect_commands
    mb, acts = _menubar(_app)
    fired = {"n": 0}
    acts["npc2d"].triggered.connect(lambda *_: fired.__setitem__("n", fired["n"] + 1))
    dlg = CommandFinderDialog(lambda: collect_commands(mb))
    try:
        dlg.set_query("npc")
        assert dlg._table.rowCount() >= 1
        assert dlg._table.item(0, 0).text() == "2D"
        assert dlg._table.item(0, 1).text() == "Analyze › Segmentation › NPC"
        dlg._run_row(0)                                    # like a double-click
        assert fired["n"] == 1                             # ran the command
    finally:
        dlg.close()


def test_excluded_menu_and_action_are_skipped(_app):
    from minflux_viewer.ui.command_finder import collect_commands
    mb, _ = _menubar(_app)
    file_menu = mb.actions()[0].menu()                     # "File"
    # An "Open Recent" submenu of file paths, flagged for exclusion.
    recent = file_menu.addMenu("Open Recent")
    recent.setProperty("command_finder_exclude", True)
    recent.addAction("D:/data/expt1.mat")
    recent.addAction("D:/data/expt2.msr")
    # A single excluded action too.
    dbg = file_menu.addAction("Secret Debug")
    dbg.setProperty("command_finder_exclude", True)

    texts = {e.text for e in collect_commands(mb)}
    assert "D:/data/expt1.mat" not in texts                # recent files gone
    assert "D:/data/expt2.msr" not in texts
    assert "Secret Debug" not in texts                     # flagged action gone
    assert "Open" in texts and "2D" in texts               # real commands remain


def test_source_of_callable(_app):
    from minflux_viewer.ui.command_finder import source_of
    src = source_of(source_of)                             # a function in this module
    assert src.endswith("command_finder.py")
    assert src in ("minflux_viewer/ui/command_finder.py", "command_finder.py")


def test_collect_reads_command_source_property(_app):
    from minflux_viewer.ui.command_finder import collect_commands
    mb, acts = _menubar(_app)
    acts["npc2d"].setProperty("command_source", "minflux_viewer/plugins/npc.py")
    by_text = {e.text: e for e in collect_commands(mb)}
    assert by_text["2D"].source == "minflux_viewer/plugins/npc.py"
    assert by_text["Open"].source == ""                    # untagged → blank


def test_dialog_four_columns_source_and_shortcut(_app):
    from minflux_viewer.ui.command_finder import CommandFinderDialog, collect_commands
    mb, acts = _menubar(_app)
    acts["npc2d"].setProperty("command_source", "plugins/npc.py")
    dlg = CommandFinderDialog(lambda: collect_commands(mb))
    try:
        assert dlg._table.columnCount() == 4
        labels = [dlg._table.horizontalHeaderItem(c).text() for c in range(4)]
        assert labels == ["Command", "Menu", "Source", "Shortcut"]
        dlg.set_query("2D")
        assert dlg._table.item(0, 2).text() == "plugins/npc.py"    # Source in col 2
        dlg.set_query("open")
        assert dlg._table.item(0, 3).text() == "Ctrl+O"            # Shortcut in col 3
    finally:
        dlg.close()


def test_reopen_after_close_and_builtin_source(_app):
    """Reopening after the dialog was closed (WA_DeleteOnClose) must not touch a
    deleted C++ object; built-in commands show their tagged source; opening clears
    the toolbar field."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.command_finder import collect_commands
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    win = MainWindow(state)
    try:
        by_text = {e.text: e for e in collect_commands(win.menuBar())}
        assert by_text["Convolution"].source.endswith("conv_segmentation_dialog.py")

        win._toolbar_search.setText("convolution")
        win._open_command_finder("convolution")
        assert win._command_finder is not None
        assert win._command_finder._filter.text() == "convolution"
        assert win._toolbar_search.text() == ""          # field reset to placeholder

        win._command_finder.close()
        _app.processEvents()                             # deleteLater + destroyed
        assert win._command_finder is None               # stale ref cleared

        win._open_command_finder("render")               # must NOT raise RuntimeError
        assert win._command_finder is not None
        assert win._command_finder._filter.text() == "render"
    finally:
        win.close()
        _app.processEvents()


def _real_window(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow
    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    return MainWindow(state)


def test_keyword_search_finds_nested_commands(_app):
    """Operational leaves are findable by tag/synonym words that are in neither the
    command name nor its menu path (the whole point of command_meta keywords)."""
    from minflux_viewer.ui.command_finder import collect_commands, filter_commands
    win = _real_window(_app)
    try:
        entries = collect_commands(win.menuBar())

        def hits(query):
            return {(e.text, e.path) for e in filter_commands(entries, query)}

        # "nuclear pore" is only a keyword on the NPC leaves (name is just "2D").
        npc = hits("nuclear pore")
        assert any(t == "2D" and p.endswith("NPC") for t, p in npc)
        # "matched filter" is a keyword on the Convolution detectors only.
        assert any("Convolution" in t for t, _ in hits("matched filter"))
        # "unmix" tags DCR channel separation (word absent from its name/path).
        assert any("DCR" in t for t, _ in hits("unmix"))
        # keywords must not leak into the name-rank: a real name still wins.
        conv = filter_commands(entries, "convolution")
        assert conv and conv[0].text.startswith("Convolution")
    finally:
        win.close()
        _app.processEvents()


def test_every_menu_command_is_traceable_to_a_source(_app):
    """Every indexed menu command (recent-files etc. are excluded from the index)
    resolves to an implementing module — the user's 'no untraceable command' rule."""
    from minflux_viewer.ui.command_finder import collect_commands
    win = _real_window(_app)
    try:
        blank = [
            e for e in collect_commands(win.menuBar())
            if not e.source and e.enabled and e.text != "(no plugins registered)"
        ]
        assert blank == [], f"commands with no Source: {[(e.text, e.path) for e in blank]}"
    finally:
        win.close()
        _app.processEvents()


def test_dialog_does_not_run_disabled(_app):
    from minflux_viewer.ui.command_finder import CommandFinderDialog, collect_commands
    mb, acts = _menubar(_app)
    fired = {"n": 0}
    acts["off"].triggered.connect(lambda *_: fired.__setitem__("n", fired["n"] + 1))
    dlg = CommandFinderDialog(lambda: collect_commands(mb))
    try:
        dlg.set_query("Local Density")
        dlg._run_row(0)
        assert fired["n"] == 0                             # disabled → not triggered
        assert "unavailable" in dlg._info.text().lower()   # shows a hint, stays open
    finally:
        dlg.close()
