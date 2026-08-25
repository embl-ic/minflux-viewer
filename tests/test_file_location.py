"""OS file-manager integration used by recent files and Dataset Manager."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")

from minflux_viewer.ui.main_window import MainWindow


def test_windows_explorer_select_keeps_spaced_path_separate(monkeypatch):
    path = r"D:\Workspace\Microscopes\MINFLUX\sample data\2_3C_measurement.msr"
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "Popen", lambda args: calls.append(args))

    MainWindow.open_file_location(None, path)

    assert calls == [["explorer.exe", "/select,", path]]
    command_line = subprocess.list2cmdline(calls[0])
    assert command_line == f'explorer.exe /select, "{path}"'
