"""Run the Qt tests one module per process.

Every Qt test in ONE process aborts natively at a moving target -- a
``Fatal Python error: Aborted`` inside pytest-qt, or an access violation --
because the failure needs the state many earlier tests leave behind. A module
in a process of its own does not accumulate that state: measured 2026-10-13, all
125 Qt modules passed this way, while a single-process run of the same tests
aborted on every attempt (four of four, native display and offscreen alike).

Even alone, a module very occasionally dies the same way under parallel load
(once in a 125-module run; never in ten runs of that module on its own). So a
module that CRASHES is run once more, and a pass on the retry is reported as
such rather than hidden. A module whose tests FAIL is never retried. With
``--strict`` (what CI runs) a crash fails the run even when its retry passes:
the application is one long-lived process, so a crash is a defect wherever a
fresh process happens to get past it. Locally the retry stays a diagnostic.

The pure tests do not need any of this; run them in one process with
``pytest -m "not qt"``.

Usage::

    python tools/run_qt_tests.py              # offscreen, 4 workers
    python tools/run_qt_tests.py -j 8
    python tools/run_qt_tests.py --native     # use the real display
    python tools/run_qt_tests.py --strict     # a recovered crash still fails (CI)
    python tools/run_qt_tests.py -- -x -k lut  # extra pytest arguments

``QT_QPA_PLATFORM=offscreen`` is the default because it makes the results
independent of the display: several layout tests assert sizes that only fit
a desktop-sized screen, and OpenGL is fragile on some display paths (remote
desktop, CI runners without a GPU).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: pytest's exit code when a selection is empty -- not a failure here.
NO_TESTS_COLLECTED = 5

#: What faulthandler prints when the process dies natively. The exit code alone
#: cannot say so: an abort on Windows exits with 3, pytest's internal-error code.
CRASH_MARKERS = ("Fatal Python error", "Windows fatal exception")


def modules_from_collection(output: str) -> list[str]:
    """The test modules, in order, from ``pytest --collect-only -q`` output."""
    seen: dict[str, None] = {}
    for line in output.splitlines():
        if "::" in line:
            seen.setdefault(line.split("::", 1)[0].strip(), None)
    return list(seen)


def outcome(returncode: int, output: str = "") -> str:
    """``passed``, ``failed`` or ``crashed`` for one module's pytest run."""
    if returncode in (0, NO_TESTS_COLLECTED):
        return "passed"
    if any(marker in output for marker in CRASH_MARKERS):
        return "crashed"
    if 0 < returncode <= 4:
        return "failed"
    return "crashed"           # killed by a native fault (negative, 139, ...)


def exit_status(*, failed: int, recovered: int, strict: bool) -> int:
    """1 when any module failed, or -- in strict mode -- crashed even once."""
    return 1 if failed or (strict and recovered) else 0


def _collect(python: str, env: dict) -> list[str]:
    result = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "qt"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode not in (0, NO_TESTS_COLLECTED):
        sys.stdout.write(result.stdout + result.stderr)
        raise SystemExit("Collecting the Qt tests failed.")
    return modules_from_collection(result.stdout)


def _github_annotations():
    """``tools/github_annotations.py``, loaded by path so tests can load this file."""
    path = Path(__file__).resolve().parent / "github_annotations.py"
    spec = importlib.util.spec_from_file_location("_mfv_github_annotations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_summary(output: str, verdict: str, limit: int = 25) -> str:
    """What a module's annotation says: its FAILED/ERROR lines, else the crash
    report (or, failing both, the end of the output)."""
    if not verdict.startswith("passed"):
        lines = [line for line in output.splitlines()
                 if line.startswith(("FAILED ", "ERROR "))]
        if lines and "crash" not in verdict:
            return "\n".join(lines[:limit])
    return "\n".join(_report_excerpt(output).splitlines()[:limit])


def _report_excerpt(output: str, limit: int = 60) -> str:
    """The crash report when there is one, else the end of the output."""
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if any(marker in line for marker in CRASH_MARKERS)),
                 max(0, len(lines) - limit))
    return "\n".join(lines[start:start + limit])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-j", "--jobs", type=int,
                        default=min(4, os.cpu_count() or 1),
                        help="modules run at once (default: %(default)s)")
    parser.add_argument("--strict", action="store_true",
                        help="fail when a module crashed, even if its retry passed (CI)")
    parser.add_argument("--native", action="store_true",
                        help="use the real display instead of QT_QPA_PLATFORM=offscreen")
    parser.add_argument("pytest_args", nargs="*",
                        help="extra pytest arguments, after --")
    args = parser.parse_args(argv)

    env = dict(os.environ)
    if not args.native:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    annotations = _github_annotations()
    env[annotations.QT_RUNNER_CHILD_ENV] = "1"     # the runner annotates, per module
    python = sys.executable
    modules = _collect(python, env)
    if not modules:
        print("No Qt tests collected.")
        return 0

    basetemp = Path(tempfile.mkdtemp(prefix="mfv-qt-tests-"))

    def run_once(module: str, attempt: int):
        result = subprocess.run(
            [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "qt",
             f"--basetemp={basetemp / f'{Path(module).stem}-{attempt}'}", module,
             *args.pytest_args],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        return result.returncode, output, outcome(result.returncode, output)

    def run(module: str):
        started = time.monotonic()
        code, output, verdict = run_once(module, 1)
        if verdict == "crashed":
            retry_code, retry_output, retry_verdict = run_once(module, 2)
            if retry_verdict == "passed":
                verdict = "passed after a crash (retried)"   # keep the crash's report
            else:
                code, output, verdict = retry_code, retry_output, retry_verdict
        return module, code, output, verdict, time.monotonic() - started

    print(f"Running {len(modules)} Qt test modules, {args.jobs} at a time "
          f"(QT_QPA_PLATFORM={env.get('QT_QPA_PLATFORM', 'native')}).", flush=True)
    bad, recovered = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for module, code, output, verdict, seconds in pool.map(run, modules):
            print(f"  {verdict:7s} {seconds:6.1f} s  {module}", flush=True)
            if verdict.startswith("passed after"):
                recovered.append((module, code, output, verdict))
            elif verdict != "passed":
                bad.append((module, code, output, verdict))

    for module, code, output, verdict in recovered + bad:
        print(f"\n===== {module}: {verdict} (exit {code}) =====")
        print(_report_excerpt(output))
    summary = f"\n{len(modules) - len(bad)} of {len(modules)} Qt modules passed"
    if recovered:
        summary += f", {len(recovered)} of them only after a native crash was retried"
    print(summary + ".")
    if recovered and args.strict:
        print("Strict mode: a crash fails the run even when its retry passed.")
    if annotations.on_github_actions():
        reported = bad + (recovered if args.strict else [])
        for line in annotations.error_commands(
                (f"Qt {verdict}: {module}", failure_summary(output, verdict))
                for module, _code, output, verdict in reported):
            print(line, flush=True)
    return exit_status(failed=len(bad), recovered=len(recovered), strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
