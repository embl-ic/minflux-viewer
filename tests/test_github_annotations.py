"""CI names each failure in a public GitHub annotation.

A failing job's log can be read only when signed in; its annotations cannot be
hidden that way, so a failure is diagnosable from the run page (or the API)
alone. See tools/github_annotations.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ga = _load("github_annotations_under_test", _ROOT / "tools" / "github_annotations.py")
runner = _load("run_qt_tests_annotations", _ROOT / "tools" / "run_qt_tests.py")


def test_only_active_on_github_actions():
    assert ga.on_github_actions({"GITHUB_ACTIONS": "true"})
    assert not ga.on_github_actions({})
    assert not ga.on_github_actions({"GITHUB_ACTIONS": "false"})


def test_a_command_is_escaped_as_github_requires():
    line = ga.error_command("FAILED tests/a.py::test_x[1, 2]",
                            "E   assert 100% == 1\nE   second: line\r")
    assert line == ("::error title=FAILED tests/a.py%3A%3Atest_x[1%2C 2]"
                    "::E   assert 100%25 == 1%0AE   second: line")
    assert "\n" not in line                     # one command is one line


def test_a_long_message_keeps_its_end():
    message = "start\n" + "x" * (ga.MAX_MESSAGE_CHARS + 50) + "\nE   the error"
    line = ga.error_command("t", message)
    assert line.endswith("%0AE   the error")
    assert "start" not in line


def test_past_the_limit_the_last_annotation_names_the_rest():
    failures = [(f"FAILED tests/t.py::test_{n}", f"E {n}") for n in range(13)]
    commands = ga.error_commands(failures)
    assert len(commands) == ga.MAX_ERRORS_PER_STEP
    assert commands[-1].startswith("::error title=4 more failure(s)::")
    for n in range(9, 13):
        assert f"test_{n}" in commands[-1]
    assert ga.error_commands([]) == []


def test_the_pytest_hook_annotates_failures_on_github_only(request, monkeypatch):
    conftest = next(plugin for plugin in request.config.pluginmanager.get_plugins()
                    if Path(getattr(plugin, "__file__", "") or "").name == "conftest.py"
                    and Path(plugin.__file__).parent.name == "tests")
    written = []
    reporter = SimpleNamespace(
        write_line=written.append,
        stats={"failed": [SimpleNamespace(nodeid="tests/t.py::test_a",
                                          longreprtext="def test_a():\nE   assert 1 == 2")],
               "error": [SimpleNamespace(nodeid="tests/u.py",
                                         longreprtext="E   ImportError: no module")],
               "passed": [SimpleNamespace(nodeid="tests/t.py::test_b")]})

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    conftest.pytest_terminal_summary(reporter)
    assert written == []                        # nothing changes off GitHub

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv(ga.QT_RUNNER_CHILD_ENV, "1")
    conftest.pytest_terminal_summary(reporter)
    assert written == []                        # the Qt runner reports for its children

    monkeypatch.delenv(ga.QT_RUNNER_CHILD_ENV)
    conftest.pytest_terminal_summary(reporter)
    assert written == [
        "::error title=FAILED tests/t.py%3A%3Atest_a::def test_a():%0AE   assert 1 == 2",
        "::error title=ERROR tests/u.py::E   ImportError: no module",
    ]


def test_a_qt_module_is_summarised_by_its_failed_lines_or_its_crash():
    failed = "....F\nFAILED tests/q.py::test_x - AssertionError: no\n1 failed, 4 passed"
    assert runner.failure_summary(failed, "failed") == \
        "FAILED tests/q.py::test_x - AssertionError: no"
    crashed = "..\nFatal Python error: Aborted\n\nCurrent thread 0x1:\n  File \"q.py\""
    summary = runner.failure_summary(crashed, "crashed")
    assert summary.startswith("Fatal Python error: Aborted")
    assert runner.failure_summary(crashed, "passed after a crash (retried)") == summary
