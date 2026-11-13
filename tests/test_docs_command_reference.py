"""The published command reference (docs/reference/commands.md) matches the app.

The page is generated from the main window's menus and the command metadata by
``tools/build_command_reference.py``; this test regenerates it in memory and
fails when the committed page differs, naming the command to run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    """``tools/build_command_reference.py``, loaded by path (``tools`` is no package)."""
    name = "_mfv_build_command_reference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "build_command_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def _window(qapp):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.save_prefs = lambda *a, **k: None
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    state.prefs.setdefault("plugin", {})["scan_user_dir"] = False
    window = MainWindow(state)
    yield window
    window.close()
    qapp.processEvents()


def test_the_published_command_reference_matches_the_menus(_window):
    tool = _tool()
    expected = tool.build_markdown(_window)
    committed = tool.PAGE.read_text(encoding="utf-8")
    assert committed == expected, (
        "docs/reference/commands.md is out of date with the menus or "
        "ui/common/command_meta.py; regenerate it with "
        "`python tools/build_command_reference.py`.")


def _locally_excluded_names() -> set[str]:
    """What this checkout keeps out of git at its top level through
    ``.git/info/exclude``: each such file or folder name and its stem, lower
    case. Read from there rather than listed, so this test names none of them;
    a fresh clone has none. Nested patterns (tool state inside an excluded
    folder) are covered by their folder's name."""
    exclude = ROOT / ".git" / "info" / "exclude"
    if not exclude.is_file():
        return set()
    names: set[str] = set()
    for line in exclude.read_text(encoding="utf-8", errors="replace").splitlines():
        pattern = line.strip().rstrip("/")
        if not pattern or pattern.startswith(("#", "!")) or "/" in pattern or "*" in pattern:
            continue
        name = pattern.lower()
        stem = name.lstrip(".").rsplit(".", 1)[0]
        names.update(n for n in (name, stem) if len(n) >= 4)
    return names


def test_the_reference_lists_only_built_in_commands_and_names_no_private_file(_window):
    text = _tool().build_markdown(_window).lower()
    assert "### filter" in text and "### frc resolution estimation" in text
    leaked = sorted(name for name in _locally_excluded_names() if name in text)
    assert not leaked, f"the page names files this checkout keeps out of git: {leaked}"
    assert "pyminflux" not in text          # methods cite their papers
