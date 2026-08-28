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


def test_main_window_then_image_views_survive_shared_process_gc(tmp_path: Path) -> None:
    """Guard the ViewBox.destroyed weak-registry crash seen in long sessions."""
    base = str(tmp_path / "mixed-main-image")
    targets = [
        str(Path("tests") / "test_async_file_io.py"),
        str(Path("tests") / "test_post_load.py"),
        str(Path("tests") / "test_image_source.py"),
    ]
    code = f"""
        import pytest
        raise SystemExit(pytest.main([
            "-q", "-p", "no:cacheprovider", "--basetemp", {base!r},
            *{targets!r},
        ]))
    """
    result = _run_python(code, timeout=60)
    _assert_clean(result, "mixed MainWindow and ImageView lifecycle")


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


def test_msr_dialog_close_does_not_destroy_running_parse_thread() -> None:
    code = """
        import time
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.plugins.msr_reader.msr_reader_dialog import (
            MsrReaderDialog, _ParseWorker,
        )

        app = QApplication([])
        _ParseWorker.run = lambda self: time.sleep(0.45)
        dialog = MsrReaderDialog(state=None)
        dialog._save_settings = lambda: None
        dialog._start_parse_worker("ignored.msr", ".")
        worker = dialog._worker
        time.sleep(0.05)
        started = time.perf_counter()
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        elapsed = time.perf_counter() - started
        worker.wait()
        if elapsed >= 0.2:
            raise AssertionError(f"MSR dialog close blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "MSR dialog close during parse")


def test_particle_average_close_detaches_running_task() -> None:
    code = """
        import time
        import numpy as np
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.background_tasks import shared_thread_pool
        from minflux_viewer.ui.particle_average_dialog import (
            ParticleAverageWindow, _AverageTask,
        )

        app = QApplication([])
        dialog = ParticleAverageWindow(AppState())
        task = _AverageTask(
            lambda report: (time.sleep(0.45), (np.empty((0, 3)), "done"))[1]
        )
        dialog._task = task
        pool = shared_thread_pool("particle-average", max_threads=1)
        pool.start(task)
        time.sleep(0.05)
        started = time.perf_counter()
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        elapsed = time.perf_counter() - started
        pool.waitForDone()
        if elapsed >= 0.2:
            raise AssertionError(f"Particle Average close blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "Particle Average close during work")


def test_main_window_close_detaches_running_zarr_io() -> None:
    code = """
        import time
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.background_tasks import shared_thread_pool
        from minflux_viewer.ui.main_window import MainWindow, _ZarrIoTask

        app = QApplication([])
        state = AppState()
        state.prefs.setdefault("file", {})["check_updates_on_startup"] = False
        state.save_prefs = lambda: None
        window = MainWindow(state)
        task = _ZarrIoTask(
            lambda report: (time.sleep(0.45), object())[1], description="test Zarr"
        )
        window._begin_zarr_io(task)
        time.sleep(0.05)
        started = time.perf_counter()
        window.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        elapsed = time.perf_counter() - started
        shared_thread_pool("zarr-io", max_threads=2).waitForDone()
        if elapsed >= 0.2:
            raise AssertionError(f"MainWindow close during Zarr I/O blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "MainWindow close during Zarr I/O")


def test_main_window_close_detaches_running_generic_file_io() -> None:
    code = """
        import time
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.background_tasks import BackgroundTask, shared_thread_pool
        from minflux_viewer.ui.main_window import MainWindow

        app = QApplication([])
        state = AppState()
        state.prefs.setdefault("file", {})["check_updates_on_startup"] = False
        state.save_prefs = lambda: None
        window = MainWindow(state)
        task = BackgroundTask(
            lambda report: (time.sleep(0.45), object())[1],
            description="test generic file I/O",
        )
        window._begin_file_io(task)
        time.sleep(0.05)
        started = time.perf_counter()
        window.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        elapsed = time.perf_counter() - started
        shared_thread_pool("file-io", max_threads=2).waitForDone()
        if elapsed >= 0.2:
            raise AssertionError(f"MainWindow close during file I/O blocked for {elapsed:.3f}s")
    """
    result = _run_python(code)
    _assert_clean(result, "MainWindow close during generic file I/O")


def test_tiff_window_close_defers_source_close_until_export_finishes() -> None:
    code = """
        import tempfile
        import time
        from pathlib import Path
        import numpy as np
        import tifffile
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.tiff_source import TiffImageSource
        from minflux_viewer.ui.background_tasks import BackgroundTask, shared_thread_pool
        from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

        class SourceProxy:
            def __init__(self, source):
                self._source = source
                self.close_count = 0
            def __getattr__(self, name):
                return getattr(self._source, name)
            def close(self):
                self.close_count += 1
                self._source.close()

        app = QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.tif"
            tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))
            source = SourceProxy(TiffImageSource(path))
            window = TiffViewerWindow(source)
            window._source_lease.acquire()
            task = BackgroundTask(
                lambda report: time.sleep(0.45),
                description="test TIFF export",
                finally_callback=window._source_lease.release,
            )
            window._save_tasks.append(task)
            pool = shared_thread_pool("tiff-io", max_threads=2)
            pool.start(task)
            time.sleep(0.05)
            started = time.perf_counter()
            window.close()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            elapsed = time.perf_counter() - started
            if elapsed >= 0.2:
                raise AssertionError(f"TIFF window close blocked for {elapsed:.3f}s")
            if source.close_count != 0:
                raise AssertionError("source closed while background export still held it")
            pool.waitForDone()
            if source.close_count != 1:
                raise AssertionError(f"source close count after export: {source.close_count}")
    """
    result = _run_python(code)
    _assert_clean(result, "TIFF close during source-backed export")
