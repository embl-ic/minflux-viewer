"""tools/run_qt_tests.py: the per-module runner CI uses for the Qt tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_qt_tests.py"
_spec = importlib.util.spec_from_file_location("run_qt_tests", _PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_modules_are_read_from_the_collection_once_each_in_order():
    output = (
        "tests/test_b.py::test_one\n"
        "tests/test_a.py::TestX::test_two\n"
        "tests/test_b.py::test_three[param]\n"
        "\n3 tests collected in 0.10s\n"
    )
    assert runner.modules_from_collection(output) == [
        "tests/test_b.py", "tests/test_a.py"]


def test_a_native_abort_is_reported_as_a_crash_not_a_failure():
    assert runner.outcome(0) == "passed"
    assert runner.outcome(runner.NO_TESTS_COLLECTED) == "passed"
    assert runner.outcome(1) == "failed"          # tests failed
    assert runner.outcome(2) == "failed"          # interrupted / usage
    assert runner.outcome(139) == "crashed"       # segfault, as bash reports it
    assert runner.outcome(-11) == "crashed"
    assert runner.outcome(3221225477) == "crashed"  # Windows 0xC0000005


def test_an_abort_is_told_apart_from_an_internal_error_by_its_report():
    """On Windows abort() exits with 3 -- also pytest's internal-error code --
    so only faulthandler's report says which it was. Only a crash is retried."""
    assert runner.outcome(3) == "failed"
    assert runner.outcome(3, "...\nFatal Python error: Aborted\n") == "crashed"
    assert runner.outcome(1, "Windows fatal exception: access violation") == "crashed"


def test_strict_mode_fails_on_a_crash_even_when_its_retry_passed():
    """CI is strict: the application is one long-lived process, so a crash is a
    defect even when a fresh process gets past it. Locally the retry stays a
    diagnostic and a recovered module does not fail the run."""
    assert runner.exit_status(failed=0, recovered=0, strict=True) == 0
    assert runner.exit_status(failed=0, recovered=1, strict=False) == 0
    assert runner.exit_status(failed=0, recovered=1, strict=True) == 1
    assert runner.exit_status(failed=1, recovered=0, strict=False) == 1


def test_the_report_starts_at_the_crash_not_at_the_last_lines():
    output = "\n".join([*(f"line {i}" for i in range(200)),
                        "Windows fatal exception: access violation",
                        "Current thread 0x1 (most recent call first):",
                        '  File "app.py", line 1 in f'])
    excerpt = runner._report_excerpt(output)
    assert excerpt.startswith("Windows fatal exception")
    assert 'File "app.py"' in excerpt
