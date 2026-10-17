"""UI capabilities have canonical owners without legacy module shims."""
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("group,stem", [
    ("analysis", "precision_analysis"), ("common", "background_tasks"),
    ("controllers", "window_controller"), ("dialogs", "filter_dialog"),
    ("plots", "histogram_window"), ("rendering", "render_window"),
    ("roi", "roi_overlay"), ("settings", "preferences_dialog"),
    ("tools", "script_editor_window"), ("widgets", "status_line"),
])
def test_capability_module_exists_and_its_old_ui_path_does_not(group, stem):
    spec = importlib.util.find_spec(f"minflux_viewer.ui.{group}.{stem}")
    assert spec is not None
    assert Path(spec.origin).parent.name == group
    assert importlib.util.find_spec(f"minflux_viewer.ui.{stem}") is None


def test_only_the_stable_application_shell_remains_at_the_ui_root():
    import minflux_viewer.ui

    root = Path(minflux_viewer.ui.__file__).parent
    assert {path.stem for path in root.glob("*.py")} == {"__init__", "main_window"}


def test_editor_script_directory_does_not_depend_on_ui_module_depth(monkeypatch, tmp_path):
    import minflux_viewer
    from minflux_viewer.ui.tools import script_editor_window

    expected = Path(minflux_viewer.__file__).resolve().parent.parent / "scripts"
    monkeypatch.setattr(script_editor_window, "__file__", str(tmp_path / "unrelated.py"))
    assert script_editor_window._project_scripts_directory() == expected
