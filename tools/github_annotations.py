"""Name each test failure in a GitHub Actions annotation.

A failing CI job's log can be read only by someone signed in to GitHub, but
its annotations are public: they are listed on the run's summary page and
returned by the API without a login. So the test runners print every failure
as a workflow command (``::error title=...::message``), which GitHub turns into
an annotation naming the test and its error.

Active only when ``GITHUB_ACTIONS`` is ``true``; anywhere else the runners
print exactly what they did before. Pure and dependency-free: it is loaded by
file path from ``tests/conftest.py`` and ``tools/run_qt_tests.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

#: GitHub keeps at most ten error annotations per step (fifty per job).
MAX_ERRORS_PER_STEP = 10

#: Characters of a message kept. An annotation is a pointer, the log has the
#: rest; the end of a traceback is the part that names the error, so the end
#: is what is kept.
MAX_MESSAGE_CHARS = 3000

#: Set by ``tools/run_qt_tests.py`` for the pytest processes it starts, which
#: then leave the annotations to the runner (one per module, not per test).
QT_RUNNER_CHILD_ENV = "MFV_QT_RUNNER_CHILD"


def on_github_actions(env: Mapping[str, str] | None = None) -> bool:
    """True when running as a GitHub Actions step."""
    env = os.environ if env is None else env
    return env.get("GITHUB_ACTIONS") == "true"


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def error_command(title: str, message: str) -> str:
    """One ``::error`` workflow command, escaped as GitHub requires."""
    message = message.strip() or title
    if len(message) > MAX_MESSAGE_CHARS:
        message = "..." + message[-MAX_MESSAGE_CHARS:]
    return f"::error title={_escape_property(title)}::{_escape_data(message)}"


def error_commands(failures: Iterable[tuple[str, str]]) -> list[str]:
    """Commands for ``(title, message)`` failures, within GitHub's limit.

    Past the limit, the last annotation lists the titles of the rest, so no
    failure goes unnamed.
    """
    failures = list(failures)
    if len(failures) <= MAX_ERRORS_PER_STEP:
        return [error_command(title, message) for title, message in failures]
    shown = failures[:MAX_ERRORS_PER_STEP - 1]
    rest = failures[MAX_ERRORS_PER_STEP - 1:]
    commands = [error_command(title, message) for title, message in shown]
    commands.append(error_command(
        f"{len(rest)} more failure(s)", "\n".join(title for title, _ in rest)))
    return commands
