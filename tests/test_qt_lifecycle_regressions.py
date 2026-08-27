"""Process-isolated regression tests for Qt lifetime and shutdown failures.

Qt/PyQtGraph teardown bugs often terminate the interpreter after all ordinary
assertions have passed.  Running the affected scenario in a child process makes
the native exit code an assertion in the parent process instead of taking the
whole test session down with it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WINDOWS_ERROR_MODE = 0x0001 | 0x0002 | 0x8000


def _run_python(code: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    prefix = f"""
        import ctypes
        import os
        if os.name == "nt":
            ctypes.windll.kernel32.SetErrorMode({_WINDOWS_ERROR_MODE})
    """
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(prefix + code)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _assert_clean(result: subprocess.CompletedProcess[str], scenario: str) -> None:
    assert result.returncode == 0, (
        f"{scenario} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "test_file",
    [
        "test_image_source.py",
        "test_spreadsheet_autodetect.py",
        "test_render_roi_keys.py",
        "test_roi_rotation.py",
        "test_render_fit_view.py",
        "test_lut_dialog.py",
        "test_brightness_contrast_close.py",
        "test_format_registry.py",
    ],
)
def test_qt_heavy_module_exits_cleanly(test_file: str, tmp_path: Path) -> None:
    """A green Qt module must also survive its final deferred deletes and GC."""
    target = str(Path("tests") / test_file)
    base = str(tmp_path / Path(test_file).stem)
    code = f"""
        import pytest
        raise SystemExit(pytest.main([
            "-q", "-p", "no:cacheprovider", "--basetemp", {base!r}, {target!r}
        ]))
    """
    result = _run_python(code, timeout=60)
    _assert_clean(result, test_file)


def test_render_close_never_waits_for_a_tiff_export() -> None:
    """A long export must not make RenderWindow.close() block the GUI thread."""
    code = """
        import time
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.render_window import RenderWindow

        class SlowWorker:
            def __init__(self):
                self.interruption_requested = False

            def isRunning(self):
                return True

            def requestInterruption(self):
                self.interruption_requested = True

            def wait(self, *args):
                time.sleep(0.4)
                return True

        app = QApplication([])
        window = RenderWindow(AppState())
        worker = SlowWorker()
        window._export_workers = [worker]
        started = time.perf_counter()
        window.close()
        elapsed = time.perf_counter() - started
        if elapsed >= 0.2:
            raise AssertionError(f"RenderWindow.close blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "RenderWindow close during TIFF export")


def test_main_window_close_never_waits_for_the_global_pool() -> None:
    """Application close initiates shutdown without sleeping in closeEvent."""
    code = """
        import time
        from PyQt6.QtCore import QRunnable, QThreadPool
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.main_window import MainWindow

        app = QApplication([])
        state = AppState()
        state.prefs.setdefault("file", {})["check_updates_on_startup"] = False
        state.save_prefs = lambda: None
        window = MainWindow(state)
        pool = QThreadPool.globalInstance()
        pool.start(QRunnable.create(lambda: time.sleep(0.8)))
        time.sleep(0.05)
        started = time.perf_counter()
        window.close()
        elapsed = time.perf_counter() - started
        pool.waitForDone()
        if elapsed >= 0.2:
            raise AssertionError(f"MainWindow.close blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "MainWindow close with a running global task")


def test_precision_scheduler_deletion_does_not_wait_for_running_work() -> None:
    """Deleting a scheduler must not invoke QThreadPool's blocking destructor."""
    code = """
        import time
        from PyQt6.QtCore import QCoreApplication, QEvent, QRunnable
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.ui.precision_render import PrecisionRenderScheduler

        app = QApplication([])
        scheduler = PrecisionRenderScheduler()
        pool = scheduler._pool
        pool.start(QRunnable.create(lambda: time.sleep(0.6)))
        time.sleep(0.05)
        scheduler.cancel()
        scheduler.deleteLater()
        started = time.perf_counter()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        elapsed = time.perf_counter() - started
        try:
            pool.waitForDone()
        except RuntimeError:
            pass
        if elapsed >= 0.2:
            raise AssertionError(f"scheduler deletion blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "PrecisionRenderScheduler deletion with active work")
