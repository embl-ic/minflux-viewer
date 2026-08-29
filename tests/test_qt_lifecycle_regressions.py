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


# Native teardown faults that can leave the exit code at 0. A Qt abort usually
# changes the return code, but a caught-and-reported fault, a faulthandler dump
# from a non-fatal signal, or Qt's own "Destroyed while still running" warning
# can all pass an exit-code-only check while describing exactly the failure
# these tests exist to catch.
_FATAL_STDERR_MARKERS = (
    "Fatal Python error",
    "Windows fatal exception",
    "Current thread 0x",
    "QThread: Destroyed while thread is still running",
    "Segmentation fault",
)


def _assert_clean(result: subprocess.CompletedProcess[str], scenario: str) -> None:
    detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, (
        f"{scenario} exited with {result.returncode}\n{detail}"
    )
    found = [m for m in _FATAL_STDERR_MARKERS if m in (result.stderr or "")]
    assert not found, (
        f"{scenario} exited 0 but reported {found} on stderr\n{detail}"
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


def test_repeated_lut_plot_teardown_survives_deferred_layout_events() -> None:
    """Deleted pyqtgraph labels must not receive a later layout resize event."""
    code = """
        import gc
        import numpy as np
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.ui.lut_dialog import LutDialog

        app = QApplication([])
        pixels = np.linspace(0.0, 100.0, 1000)
        for _ in range(80):
            dialog = LutDialog(
                on_levels_changed=lambda _lo, _hi: None,
                on_cmap_changed=lambda _name, _invert: None,
            )
            dialog.load_image(
                pixels=pixels,
                data_lo=0.0,
                data_hi=100.0,
                lo=20.0,
                hi=80.0,
                cmap_name="gray",
                invert=False,
            )
            dialog.close()
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            del dialog
            gc.collect()
            app.processEvents()
    """
    result = _run_python(code, timeout=60)
    _assert_clean(result, "repeated LUT plot teardown")


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


def test_update_dialog_close_does_not_destroy_a_running_download_thread() -> None:
    """A QThread owned by the dialog aborts the process when destroyed running.

    The download worker must therefore be parentless and retained by the
    process-level registry, exactly like the MSR parse and TIFF export threads.
    """
    code = """
        import gc
        import time
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.updater import (
            ReleaseAsset, ReleaseInfo, UpdateCheckResult,
        )
        from minflux_viewer.ui.update_dialog import (
            _DownloadCancelled, _DownloadWorker, _UpdateAvailableDialog,
        )

        def fake_run(self):
            try:
                for index in range(200):
                    self._report(index, 200)   # the real cancellation checkpoint
                    time.sleep(0.01)
                self.finished_ok.emit(str(self._dest))
            except _DownloadCancelled:
                return
            except Exception as exc:
                self.failed.emit(str(exc))

        _DownloadWorker.run = fake_run

        app = QApplication([])
        asset = ReleaseAsset(name="MINFLUX-Viewer-9.9.9-win.zip", url="http://x", size=1)
        release = ReleaseInfo(
            version="9.9.9", tag="v9.9.9", name="r", notes="n",
            html_url="http://x", published_at="2026-01-01", assets=(asset,),
        )
        result = UpdateCheckResult(
            status="update_available", current_version="0.4.2", latest=release,
        )
        dialog = _UpdateAvailableDialog(result, None, state=None)
        dialog._start_install()
        worker = dialog._worker
        time.sleep(0.15)
        if worker.parent() is not None:
            raise AssertionError("download worker is owned by the dialog")
        if not worker.isRunning():
            raise AssertionError("download worker did not start")

        started = time.perf_counter()
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        del dialog
        gc.collect()
        app.processEvents()
        elapsed = time.perf_counter() - started
        if elapsed >= 0.2:
            raise AssertionError(f"update dialog close blocked for {elapsed:.3f}s")
        worker.wait()
        app.processEvents()
    """
    result = _run_python(code)
    _assert_clean(result, "update dialog close during download")
