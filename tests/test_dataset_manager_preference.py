"""Dataset Manager auto-show preference and focus behavior."""

from types import SimpleNamespace

import pytest


def test_dataset_manager_preference_roundtrips_in_data_show_section():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.preferences_dialog import PreferencesDialog

    state = AppState()
    dlg = PreferencesDialog(state)
    assert dlg._show_data_info.text() == "data info..."
    assert dlg._show_dataset_manager.text() == "Dataset Manager"

    dlg._show_dataset_manager.setChecked(True)
    dlg._apply_widgets_to_draft()
    assert dlg._draft["data"]["show_dataset_manager"] is True
    dlg.close()


class _Manager:
    def __init__(self, visible: bool):
        self.visible = visible
        self.show_calls = 0

    def isVisible(self):
        return self.visible

    def show(self):
        self.show_calls += 1
        self.visible = True


def test_auto_show_does_not_raise_or_activate_existing_manager():
    from minflux_viewer.ui.main_window import MainWindow

    manager = _Manager(visible=False)
    notified = []
    window = SimpleNamespace(
        _ds_manager=manager,
        _get_dataset_manager=lambda: manager,
        _notify_view_state_changed=lambda: notified.append(True),
    )

    MainWindow._ensure_dataset_manager_visible(window)

    assert manager.show_calls == 1
    assert notified == [True]


def test_auto_show_leaves_visible_manager_untouched():
    from minflux_viewer.ui.main_window import MainWindow

    manager = _Manager(visible=True)
    window = SimpleNamespace(
        _ds_manager=manager,
        _get_dataset_manager=lambda: manager,
        _notify_view_state_changed=lambda: None,
    )

    MainWindow._ensure_dataset_manager_visible(window)

    assert manager.show_calls == 0
