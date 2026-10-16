"""The application shell delegates capabilities without changing their call contract."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "minflux_viewer" / "ui"
CONTROLLERS = (
    "post_load_controller", "save_export_controller", "file_open_controller",
    "roi_command_controller", "overlay_controller", "window_controller", "menu_controller",
)


def _shell():
    tree = ast.parse((UI / "main_window.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow")


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_delegators_preserve_signatures_and_forward_every_argument(controller):
    methods = {n.name: n for n in _shell().body if isinstance(n, ast.FunctionDef)}
    tree = ast.parse((UI / (controller + ".py")).read_text(encoding="utf-8"))
    implementations = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert implementations
    for implementation in implementations:
        wrapper = methods[implementation.name]
        args = copy.deepcopy(implementation.args)
        first = next(iter([*args.posonlyargs, *args.args]), None)
        if first is not None and first.arg in {"self", "cls"}:
            first.annotation = None
        assert ast.dump(args) == ast.dump(wrapper.args), implementation.name
        actual_return = ast.dump(implementation.returns) if implementation.returns else None
        expected_return = ast.dump(wrapper.returns) if wrapper.returns else None
        assert actual_return == expected_return, implementation.name
        statements = wrapper.body
        if isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
            statements = statements[1:]
        assert len(statements) == 2, implementation.name
        imported, returned = statements
        assert isinstance(imported, ast.ImportFrom)
        assert imported.level == 1 and imported.module is None
        assert [(n.name, n.asname) for n in imported.names] == [(controller, "controller")]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call)
        call = returned.value
        assert ast.unparse(call.func) == "controller." + implementation.name
        forwarded = [a.arg for a in [*args.posonlyargs, *args.args]]
        if args.vararg:
            forwarded.append("*" + args.vararg.arg)
        forwarded += [a.arg + "=" + a.arg for a in args.kwonlyargs]
        if args.kwarg:
            forwarded.append("**" + args.kwarg.arg)
        expected = ast.parse("f(" + ", ".join(forwarded) + ")", mode="eval").body
        assert [ast.dump(n) for n in call.args] == [ast.dump(n) for n in expected.args]
        assert [ast.dump(n) for n in call.keywords] == [ast.dump(n) for n in expected.keywords]


def test_application_shell_stays_small_and_owns_native_qt_events():
    assert len((UI / "main_window.py").read_text(encoding="utf-8").splitlines()) < 3000
    methods = {n.name for n in _shell().body if isinstance(n, ast.FunctionDef)}
    assert {"__init__", "createPopupMenu", "eventFilter", "dragEnterEvent", "dragMoveEvent",
            "dropEvent", "showEvent", "closeEvent"} <= methods
