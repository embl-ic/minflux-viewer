"""Regression checks for the frozen-application entry point."""

from __future__ import annotations

import ast
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "run_app.py"
SPEC = Path(__file__).resolve().parents[1] / "minflux_viewer.spec"


def test_freeze_support_precedes_application_import() -> None:
    """A macOS resource tracker must exit before the GUI package is imported."""
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    main_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )

    freeze_index = next(
        index
        for index, node in enumerate(main_guard.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "multiprocessing.freeze_support"
    )
    app_import_index = next(
        index
        for index, node in enumerate(main_guard.body)
        if isinstance(node, ast.ImportFrom)
        and node.module == "minflux_viewer.__main__"
    )

    assert freeze_index < app_import_index
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "minflux_viewer.__main__"
        for node in tree.body
    )


def test_spec_guards_against_incomplete_build_environment() -> None:
    """A global PyInstaller must not silently omit required app packages."""
    source = SPEC.read_text(encoding="utf-8")

    assert '"PyQt6": "PyQt6"' in source
    assert "importlib.util.find_spec(module)" in source
    assert "Build interpreter: {sys.executable}" in source
    assert "python.exe -m pip install pyinstaller==6.19.0" in source
    assert "python.exe -m PyInstaller --clean --noconfirm" in source
