"""Regression checks for the frozen-application entry point."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "run_app.py"
SPEC = ROOT / "minflux_viewer.spec"


def _evaluate_spec():
    """Run the spec with PyInstaller's heavy builders stubbed out.

    Returns ``(namespace, calls)`` where ``calls`` maps a builder name to the
    ``(args, kwargs)`` it was invoked with, so the packaging decisions the spec
    makes can be asserted without running a real build.
    """
    calls: dict[str, list] = {}

    def _stub(name):
        class Stub:
            def __init__(self, *args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                self.binaries = []
                self.datas = []
                self.pure = []
                self.scripts = []

        Stub.__name__ = name
        return Stub

    namespace = {
        "__file__": str(SPEC),
        "SPECPATH": str(ROOT),
        "DISTPATH": str(ROOT / "dist"),
        **{name: _stub(name) for name in ("Analysis", "PYZ", "EXE", "COLLECT", "BUNDLE")},
    }
    code = compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec")
    exec(code, namespace)  # noqa: S102 - the spec is first-party build config
    return namespace, calls


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows version resource")
def test_windows_executable_carries_the_application_version() -> None:
    """The shipped .exe must report its version to Explorer / inventory tools."""
    from minflux_viewer import __version__

    namespace, calls = _evaluate_spec()
    assert namespace["VERSION"] == __version__

    info = namespace["EXE_VERSION_INFO"]
    assert info is not None, "no version resource was built for Windows"
    # The EXE must actually receive it; building it and forgetting to pass it
    # is exactly how the shipped 0.4.1 executable ended up with no version.
    exe_kwargs = calls["EXE"][0][1]
    assert exe_kwargs.get("version") is info

    major, minor, patch = (int(part) for part in __version__.split(".")[:3])
    rendered = str(info)
    assert f"{major}, {minor}, {patch}, 0" in rendered
    assert __version__ in rendered
    assert info.toRaw(), "version resource did not serialise"
