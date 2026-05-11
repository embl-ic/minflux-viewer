"""
minflux_viewer.ui.memory_monitor
=================================
Task-Manager-style memory usage window.

Shows:
* Process RSS (resident set size) as a filled area under a line
* System available memory as a reference line
* 60-second rolling history, updated once per second
* Double-click the window → :func:`gc.collect` + trim caches, log freed memory

Dependency
----------
The window uses :mod:`psutil` for accurate cross-platform memory numbers.
If ``psutil`` is unavailable the window shows a clear message instead of
failing to open.
"""

from __future__ import annotations

import gc
import time
from collections import deque

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_SECONDS = 60
_TICK_MS         = 1000           # one sample per second
_MB              = 1024 * 1024


class MemoryMonitor(QWidget):
    """Floating memory-usage monitor — Help → Monitor memory."""

    TAG = "memory_monitor"

    def __init__(self, state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        self.setWindowTitle("Memory monitor")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.resize(480, 220)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # History buffers (process RSS and system-available)
        self._rss_hist   = deque(maxlen=_HISTORY_SECONDS)
        self._avail_hist = deque(maxlen=_HISTORY_SECONDS)
        self._t_hist     = deque(maxlen=_HISTORY_SECONDS)
        self._t0         = time.monotonic()

        try:
            import psutil  # noqa: F401
            self._psutil_ok = True
        except ImportError:
            self._psutil_ok = False

        self._build_ui()

        if self._psutil_ok:
            self._timer = QTimer(self)
            self._timer.setInterval(_TICK_MS)
            self._timer.timeout.connect(self._tick)
            self._timer.start()
            self._tick()  # populate first sample immediately

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #ccc; font-size: 11px;")
        root.addWidget(self._status)

        if not self._psutil_ok:
            warn = QLabel(
                "<b>psutil is not installed.</b><br>"
                "Install it to enable live memory monitoring:<br>"
                "<pre>poetry add psutil</pre>"
            )
            warn.setTextFormat(Qt.TextFormat.RichText)
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(warn, stretch=1)
            return

        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget(background="#1e1e1e")
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setLabel("left",   "memory", units="MB")
        self._plot.setLabel("bottom", "seconds ago")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setXRange(-_HISTORY_SECONDS, 0, padding=0)
        root.addWidget(self._plot, stretch=1)

        # Filled area for RSS
        self._rss_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color="#66b2ff", width=2),
            fillLevel=0,
            brush=pg.mkBrush(102, 178, 255, 80),   # transparent blue
        )
        self._plot.addItem(self._rss_curve)

        # Reference line at total system memory (full scale)
        self._avail_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color="#9aa0a6", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._avail_curve)

        self._status.setText(
            "Double-click the graph to run Python's garbage collector and free caches."
        )

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        import psutil
        p = psutil.Process()
        rss_mb   = p.memory_info().rss / _MB
        avail_mb = psutil.virtual_memory().available / _MB
        total_mb = psutil.virtual_memory().total / _MB

        t = time.monotonic() - self._t0
        self._t_hist.append(t)
        self._rss_hist.append(rss_mb)
        self._avail_hist.append(avail_mb)

        # X-axis: seconds ago (negative values)
        xs = [ti - t for ti in self._t_hist]
        self._rss_curve.setData(xs, list(self._rss_hist))
        self._avail_curve.setData([-_HISTORY_SECONDS, 0], [total_mb, total_mb])
        # Keep Y range at the system total so the "available" line is useful
        self._plot.setYRange(0, total_mb * 1.05, padding=0)

        self._status.setText(
            f"Process RSS: {rss_mb:,.1f} MB  |  "
            f"system available: {avail_mb:,.0f} / {total_mb:,.0f} MB  |  "
            "double-click to free memory"
        )

    # ------------------------------------------------------------------
    # Double-click: run gc + trim pyqtgraph caches
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if not self._psutil_ok:
            return
        import psutil
        before = psutil.Process().memory_info().rss

        freed_objects = gc.collect()
        # Trim pyqtgraph's internal paint caches (best-effort, no-op if api changes)
        try:
            pg.setConfigOption("imageAxisOrder", "row-major")   # harmless touch
        except Exception:
            pass

        after = psutil.Process().memory_info().rss
        freed_mb = max(0.0, (before - after) / _MB)
        msg = (
            f"gc.collect freed {freed_objects} object(s); "
            f"RSS dropped {freed_mb:,.1f} MB"
        )
        if self._state is not None:
            self._state.log(msg, "INFO")
        self._status.setText(msg)
        event.accept()
