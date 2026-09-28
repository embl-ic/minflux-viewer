"""Tracking View: trajectories as comets over a de-emphasised structure.

The picture this draws is the one the MATLAB NPC-trafficking workflow draws --
the reconstructed structure as a faint backdrop, and each cargo trajectory as a
connected line with a bright head walking along it as time runs.  It is a
**third viewer**, not a mode of Render or Scatter: it answers *when*, and it
needs the other two to step back to a context layer for that to be legible,
which is not something to do to a view someone opened to look at localizations.

Three things it has to get right:

**Time is zeroed to each trace by default.**  On the reference tracking file 230
traces of a few tens of milliseconds are scattered through 24 minutes, so played
on absolute time you would watch an empty field for minutes and almost never see
two traces at once.  Zeroed, they walk together and the comparison is the point.
Absolute time stays available for the question it does answer -- when things
actually happened.

**A structure channel and a tracking channel are different things.**  In a
two-colour experiment one channel is a scaffold that does not move and the other
is cargo that does.  ``core/tracks.py::looks_like_tracking`` separates them by
how far a trace travels end to end (measured: 104 nm on the reference file
against single-digit nm for a scaffold), and the role is then editable per
channel, because a default that cannot be corrected is a guess in disguise.

**It has to stay playable.**  Measured: windowing 0.20 ms/frame and drawing
3.3 ms/frame on the reference file (300 fps); 8.3 ms at 40,000 tail points and
33 ms at 200,000.  The tail is drawn as a handful of constant-opacity polylines
rather than per-point alpha, which measured **11x** faster at the same point
count (8.3 ms against 89.7 ms) -- pyqtgraph pays a per-point cost for a varying
brush, and a curve with a ``connect`` array is one painter path.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState
from ..core.loader import attr_values_1d
from ..core.overlay import (
    apply_display_transform_nm,
    channel_rgb,
    dataset_group_id,
    overlay_color_cycle,
)
from ..core.tracking_time import (
    TimeAxis,
    available_time_modes,
    build_time_axis,
    dataset_time_inputs,
    derived_step,
    format_time_seconds,
    time_axis_values,
    tracking_precision_from_prefs,
)
from ..core.tracks import (
    DEFAULT_TAIL_BANDS,
    TrackSet,
    band_edges,
    build_track_set,
    looks_like_tracking,
)
from .plot_format import apply_spatial_y_direction, plot_widget
from .time_slider import TimeAxisRow

#: The projections this view offers, in the order the other viewers list them.
AXIS_OPTIONS = ("XY", "XZ", "YZ", "3D")
AXIS_COLUMNS = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}

#: How the structure is shown behind the comets.
BACKDROP_MODES = ("Render", "Scatter", "None")

HEAD_SYMBOL = "star"
#: ⚠ The head has to be small enough not to cover its own tail. A MINFLUX track
#: is 100-600 nm end to end (median 104 nm on the reference file) inside a field
#: of ~10 um, so at full-field zoom a whole track is a few pixels: an 11 px star
#: hid every one of them. 7 px reads as a head while leaving the track visible,
#: and "Zoom to live tracks" is the answer to wanting more.
DEFAULT_HEAD_SIZE = 7

#: Tail opacity at the head and at the tip. The tip stays visible rather than
#: fading to nothing, because a trail that reaches zero cannot be told from one
#: that ended.
TAIL_ALPHA_HEAD = 255
TAIL_ALPHA_TIP = 70
TAIL_WIDTH = 2.0

#: Backdrop strength: enough to place the comets, not enough to compete. Offered
#: as three levels because how much structure is wanted depends on what it is --
#: a dense scaffold needs less than a sparse one to read as a reference frame.
BACKDROP_LEVELS = {"Faint": 90, "Normal": 175, "Strong": 255}
DEFAULT_BACKDROP_LEVEL = "Normal"
TAIL_COLOR_MODES = ("Channel", "Time")
_TIME_COLORS = np.asarray(
    [[68, 1, 84], [33, 145, 140], [253, 231, 37]], dtype=float)
#: ⚠ The raster is normalised with a square root, not linearly. A structure
#: render is sparse -- most non-empty pixels hold one or two localizations while
#: the 99th percentile is several -- so a linear ramp put the whole scaffold at
#: under a fifth of the available opacity and it vanished behind the comets.
BACKDROP_GAMMA = 0.5
BACKDROP_SCATTER_ALPHA = 70
BACKDROP_MAX_PX = 900
#: Sub-pixel anti-alias blur for the backdrop raster, in pixels.
BACKDROP_SIGMA_PX = 0.7
#: Above this, the backdrop scatter is thinned -- it is context, not data.
BACKDROP_MAX_POINTS = 300_000

#: A fresh view opens with this much tail, so the comets read as comets at once.
DEFAULT_TAIL_FRACTION = 0.04
_REDRAW_MS = 10


@dataclass(frozen=True)
class _TrackChannelSnapshot:
    """Non-Qt input captured on the GUI thread for one channel build."""

    dataset_idx: int
    dataset_identity: int
    x_m: np.ndarray | None
    y_m: np.ndarray | None
    z_m: np.ndarray | None
    z_scaling_factor: float
    display_transform: dict | None
    timestamps_s: np.ndarray | None
    trace_ids: np.ndarray | None
    keep: np.ndarray | None


@dataclass(frozen=True)
class _TrackChannelBuild:
    dataset_idx: int
    dataset_identity: int
    tracks: TrackSet | None
    automatic_role: str
    reason: str


@dataclass(frozen=True)
class _TrackBuildResult:
    channels: tuple[_TrackChannelBuild, ...]
    anchor_axis: TimeAxis | None
    axis_error: str


def _snapshot_coordinates(snapshot: _TrackChannelSnapshot) -> np.ndarray:
    """Build calibrated/display nanometres without dereferencing a dataset."""
    if snapshot.x_m is None or snapshot.y_m is None:
        raise ValueError("no aligned localization coordinates")
    x = np.asarray(snapshot.x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(snapshot.y_m, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("localization coordinate columns are not row-aligned")
    if snapshot.z_m is None:
        z = np.zeros(x.size, dtype=np.float64)
    else:
        z = np.asarray(snapshot.z_m, dtype=np.float64).reshape(-1)
        if z.size != x.size:
            raise ValueError("localization coordinate columns are not row-aligned")
    coords = np.empty((x.size, 3), dtype=np.float64)
    coords[:, 0] = x * 1.0e9
    coords[:, 1] = y * 1.0e9
    coords[:, 2] = z * 1.0e9 * float(snapshot.z_scaling_factor)
    if snapshot.display_transform is None:
        return coords
    return apply_display_transform_nm(coords, snapshot.display_transform)


def _build_track_payload(
    snapshots: tuple[_TrackChannelSnapshot, ...],
    *,
    anchor_idx: int,
    mode: str,
    precision_s: float,
    with_diagnostics: bool,
    role_displacement_nm: float,
    role_min_median_locs: int,
    report,
) -> _TrackBuildResult:
    """Worker-side track/axis construction; pure NumPy, no dataset and no Qt."""
    results: list[_TrackChannelBuild] = []
    anchor_axis = None
    axis_error = ""
    for snapshot in snapshots:
        report(f"Preparing channel {snapshot.dataset_idx + 1}")
        tid = snapshot.trace_ids
        tim = snapshot.timestamps_s
        if tid is None:
            reason = "no 'tid', so it has no trajectories at all"
            results.append(_TrackChannelBuild(
                snapshot.dataset_idx,
                snapshot.dataset_identity,
                None,
                "Structure",
                reason,
            ))
            if snapshot.dataset_idx == anchor_idx:
                axis_error = reason
            continue
        try:
            values, _unit, diagnostics = time_axis_values(
                tim,
                tid,
                mode=mode,
                precision_s=precision_s,
                with_diagnostics=(
                    with_diagnostics and snapshot.dataset_idx == anchor_idx),
            )
            report(f"Indexing channel {snapshot.dataset_idx + 1}")
            tracks = build_track_set(
                _snapshot_coordinates(snapshot),
                tid,
                values,
                # ``values`` is already on the chosen axis.
                relative=False,
                keep=snapshot.keep,
            )
            report(f"Classifying channel {snapshot.dataset_idx + 1}")
            is_tracking, reason = looks_like_tracking(
                tracks,
                displacement_nm=role_displacement_nm,
                min_median_locs=role_min_median_locs,
            )
            automatic = "Tracking" if is_tracking else "Structure"
            results.append(_TrackChannelBuild(
                snapshot.dataset_idx,
                snapshot.dataset_identity,
                tracks,
                automatic,
                reason,
            ))
            if snapshot.dataset_idx == anchor_idx:
                anchor_axis = build_time_axis(
                    tim,
                    tid,
                    mode=mode,
                    precision_s=precision_s,
                    keep=snapshot.keep,
                    with_diagnostics=with_diagnostics,
                    values=values,
                    diagnostics=diagnostics,
                    n_traces=tracks.n_traces,
                )
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            results.append(_TrackChannelBuild(
                snapshot.dataset_idx,
                snapshot.dataset_identity,
                None,
                "Structure",
                reason,
            ))
            if snapshot.dataset_idx == anchor_idx:
                axis_error = reason
        report(f"Finished channel {snapshot.dataset_idx + 1}")
    return _TrackBuildResult(tuple(results), anchor_axis, axis_error)


def _thin(n: int, limit: int) -> np.ndarray | slice:
    if n <= limit:
        return slice(None)
    return np.linspace(0, n - 1, limit).astype(np.int64)


class TrackingMovieDialog(QDialog):
    """Small export contract: deterministic frames over the current axis."""

    def __init__(self, default_frames: int, default_fps: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export tracking movie")
        root = QVBoxLayout(self)
        form = QFormLayout()
        self._frames = QSpinBox()
        self._frames.setRange(2, 5000)
        self._frames.setValue(int(np.clip(default_frames, 2, 5000)))
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.1, 120.0)
        self._fps.setDecimals(1)
        self._fps.setValue(float(np.clip(default_fps, 0.1, 120.0)))
        self._fps.setSuffix(" frames/s")
        form.addRow("Frames:", self._frames)
        form.addRow("Playback rate:", self._fps)
        root.addLayout(form)
        note = QLabel(
            "Frames span the complete current time axis and preserve the current "
            "tail width, projection, colours, backdrop, and selected track. The "
            "output is a multi-page RGB TIFF plus a JSON sidecar.")
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> tuple[int, float]:
        return int(self._frames.value()), float(self._fps.value())


class TrackingWindow(QWidget):
    """One tracking view per dataset: comets over a faint structure."""

    TAG = "tracking_window"

    def __init__(self, state: AppState, dataset_idx: int | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._channels: list[dict] = []
        self._tracks: dict[int, TrackSet] = {}
        self._track_reasons: dict[int, str] = {}
        tracking_prefs = state.prefs.get("tracking", {}) or {}
        backdrop_mode = str(tracking_prefs.get("backdrop_mode", "Render"))
        self._backdrop_mode = (
            backdrop_mode if backdrop_mode in BACKDROP_MODES else "Render")
        backdrop_level = str(tracking_prefs.get(
            "backdrop_level", DEFAULT_BACKDROP_LEVEL))
        self._backdrop_level = (
            backdrop_level if backdrop_level in BACKDROP_LEVELS
            else DEFAULT_BACKDROP_LEVEL)
        projection = str(tracking_prefs.get("default_projection", "XY"))
        self._axis = projection if projection in AXIS_OPTIONS else "XY"
        self._black_bg = True
        self._tail_bands = int(np.clip(
            tracking_prefs.get("tail_bands", DEFAULT_TAIL_BANDS), 1, 24))
        self._tail_width = max(float(tracking_prefs.get("tail_width", TAIL_WIDTH)), 0.1)
        self._tail_alpha_head = int(np.clip(
            tracking_prefs.get("tail_head_opacity", TAIL_ALPHA_HEAD), 0, 255))
        self._tail_alpha_tip = int(np.clip(
            tracking_prefs.get("tail_tip_opacity", TAIL_ALPHA_TIP), 0, 255))
        self._tail_fraction = float(np.clip(
            tracking_prefs.get("tail_fraction", DEFAULT_TAIL_FRACTION), 0.0001, 1.0))
        tail_color_mode = str(tracking_prefs.get("tail_color_mode", "Channel"))
        self._tail_color_mode = (
            tail_color_mode if tail_color_mode in TAIL_COLOR_MODES else "Channel")
        self._head_symbol = str(tracking_prefs.get("head_symbol", HEAD_SYMBOL))
        self._head_size = int(np.clip(
            tracking_prefs.get("head_size", DEFAULT_HEAD_SIZE), 2, 64))
        self._role_displacement_nm = max(float(
            tracking_prefs.get("role_displacement_nm", 25.0)), 0.0)
        self._role_min_median_locs = max(int(
            tracking_prefs.get("role_min_median_locs", 5)), 2)
        self._show_structure_tracks = False
        self._closing = False
        self._building_tracks = False
        self._track_generation = 0
        self._track_tasks: dict[int, object] = {}
        self._bounds: tuple[float, float, float, float] | None = None
        self._backdrop_key_cached: tuple | None = None
        self._last_frame_ms = 0.0
        self._3d_view = None
        self._3d_backdrop = None
        self._3d_head = None
        self._3d_tail_items: list = []
        self._3d_camera_initialized = False
        self._selected_track: tuple[int, object] | None = None
        self._candidate_xy = np.empty((0, 2), dtype=float)
        self._candidate_dataset_indices = np.empty(0, dtype=np.int64)
        self._candidate_trace_ids = np.empty(0, dtype=object)
        self._hover_summary = ""
        self._receiving_playhead = False
        self._published_playhead: tuple[int, object, float] | None = None
        self._movie_export: dict | None = None
        self._movie_timer = QTimer(self)
        self._movie_timer.setSingleShot(True)
        self._movie_timer.timeout.connect(self._capture_movie_frame)

        self.setWindowTitle("Tracking View")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(900, 780)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_REDRAW_MS)
        self._redraw_timer.timeout.connect(self._draw)

        self._build_ui()
        self._time_row.restore_state({
            "mode": str(tracking_prefs.get("default_axis_mode", "trace")),
            "rate_hz": float(tracking_prefs.get("playback_rate_hz", 10.0)),
            "loop": bool(tracking_prefs.get("playback_loop", True)),
            "grow": bool(tracking_prefs.get("tail_grow", False)),
            "all": True,
        })
        self._refresh_from_dataset()

        state.filter_changed.connect(self._on_dataset_changed)
        state.calibration_changed.connect(self._on_dataset_changed)
        state.overlay_transform_changed.connect(self._on_dataset_changed)
        state.attributes_changed.connect(self._on_dataset_changed)
        state.tracking_playhead_changed.connect(self._receive_playhead)

    # ------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        self._view_stack = QStackedWidget()
        self._view_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._view_stack, stretch=1)

        self._plot = plot_widget(background="k")
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        self._plot.setAspectLocked(True)
        self._plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot.customContextMenuRequested.connect(self._show_context_menu)
        self._plot.getPlotItem().getViewBox().setMenuEnabled(False)
        self._apply_y_axis_direction()
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self._mouse_proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved,
            rateLimit=20,
            slot=self._on_plot_moved,
        )
        self._view_stack.addWidget(self._plot)

        # Layer order is the whole readability of the picture: backdrop, then
        # the tail oldest-first, then the heads on top of everything.
        self._backdrop_image = pg.ImageItem()
        self._backdrop_image.setZValue(-20)
        self._plot.addItem(self._backdrop_image)
        self._backdrop_scatter = pg.ScatterPlotItem(size=2, pen=None)
        self._backdrop_scatter.setZValue(-19)
        self._plot.addItem(self._backdrop_scatter)

        self._tail_items: list[pg.PlotCurveItem] = []
        for band in range(self._tail_bands):
            item = pg.PlotCurveItem()
            item.setZValue(band)
            self._plot.addItem(item)
            self._tail_items.append(item)

        self._head_item = pg.ScatterPlotItem(
            size=self._head_size, symbol=self._head_symbol,
            pen=pg.mkPen(0, 0, 0, 160, width=0.8),
            brush=None)
        self._head_item.setZValue(50)
        self._plot.addItem(self._head_item)

        self._channel_area = QScrollArea()
        self._channel_area.setWidgetResizable(True)
        self._channel_area.setMaximumHeight(96)
        self._channel_widget = QWidget()
        self._channel_layout = QVBoxLayout(self._channel_widget)
        self._channel_layout.setContentsMargins(4, 2, 4, 2)
        self._channel_layout.setSpacing(2)
        self._channel_area.setWidget(self._channel_widget)
        root.addWidget(self._channel_area)

        # The time row is the established control: window, playback, axis mode.
        # Here the window *is* the comet -- its end is the head and its width the
        # tail -- so "grow from the start" is the cumulative trail for free.
        self._time_row = TimeAxisRow()
        self._time_row.rangeChanged.connect(self._schedule_draw)
        self._time_row.modeChanged.connect(self._on_time_mode_changed)
        self._time_row.diagnosticsRequested.connect(self._on_diagnostics_requested)
        root.addWidget(self._time_row)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)
        self._apply_background()

    def _rebuild_channel_rows(self) -> None:
        while self._channel_layout.count():
            item = self._channel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for pos, ch in enumerate(self._channels):
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)

            visible = QCheckBox(ch["name"])
            visible.setChecked(bool(ch["visible"]))
            visible.toggled.connect(
                lambda on, i=pos: self._set_channel_visible(i, on))
            lay.addWidget(visible, stretch=1)

            swatch = QLabel("  ")
            swatch.setFixedWidth(16)
            r, g, b = ch["color"]
            swatch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border: 1px solid #555;")
            lay.addWidget(swatch)

            role = QComboBox()
            role.addItems(["Tracking", "Structure"])
            role.setCurrentText(ch["role"])
            role.setToolTip(ch["reason"])
            role.currentTextChanged.connect(
                lambda text, i=pos: self._set_channel_role(i, text))
            lay.addWidget(role)
            self._channel_layout.addWidget(row)

    # -------------------------------------------------------------- dataset

    def _dataset(self):
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return None
        return self._state.datasets[self._idx]

    def dataset_idx(self) -> int | None:
        return self._idx

    def _build_channels(self) -> None:
        """Gather this dataset and, when it is an overlay, its siblings.

        The same membership rule the render view uses, so a two-colour
        experiment opens here showing both channels rather than one.
        """
        previous = {ch["dataset_idx"]: ch for ch in self._channels}
        self._channels = []
        anchor = self._dataset()
        if anchor is None:
            return
        group = dataset_group_id(anchor)
        cycle = list(overlay_color_cycle(self._state.prefs)) + ["Gray"]
        for idx, ds in enumerate(self._state.datasets):
            if not ds.has_localizations:
                continue
            if idx != self._idx and (group is None or dataset_group_id(ds) != group):
                continue
            name = str(getattr(ds, "name", f"dataset {idx}"))
            lut = ds.state.get("overlay_lut") or cycle[len(self._channels) % len(cycle)]
            old = previous.get(idx, {})
            self._channels.append({
                "dataset_idx": idx, "name": name,
                "visible": bool(old.get("visible", True)),
                "lut": lut, "color": channel_rgb(lut),
                "role": "Tracking", "reason": "",
            })
        self._ensure_tail_items(len(self._channels))

    def _ensure_tail_items(self, channel_count: int) -> None:
        """Allocate one curve per (channel, age band) to retain channel colour."""
        required = max(1, int(channel_count)) * self._tail_bands
        while len(self._tail_items) < required:
            item = pg.PlotCurveItem()
            item.setZValue(len(self._tail_items) % self._tail_bands)
            self._plot.addItem(item)
            self._tail_items.append(item)
        self._ensure_3d_tail_items(required)

    def _ensure_3d_built(self) -> bool:
        if self._3d_view is not None:
            return True
        try:
            import pyqtgraph.opengl as gl
        except ImportError as exc:
            self._state.log(f"Tracking View 3D unavailable: {exc}", "WARN")
            self._info_label.setText(
                f"3D view unavailable: PyOpenGL is not installed ({exc}).")
            return False
        view = gl.GLViewWidget()
        view.setBackgroundColor("k" if self._black_bg else "w")
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(self._show_context_menu)
        backdrop = gl.GLScatterPlotItem(pxMode=True, size=2.0)
        backdrop.setGLOptions("translucent")
        head = gl.GLScatterPlotItem(pxMode=True, size=float(self._head_size))
        head.setGLOptions("translucent")
        view.addItem(backdrop)
        view.addItem(head)
        self._3d_view = view
        self._3d_backdrop = backdrop
        self._3d_head = head
        self._view_stack.addWidget(view)
        self._ensure_3d_tail_items(max(1, len(self._tail_items)))
        return True

    def _ensure_3d_tail_items(self, required: int) -> None:
        if self._3d_view is None:
            return
        import pyqtgraph.opengl as gl

        while len(self._3d_tail_items) < int(required):
            item = gl.GLLinePlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                width=self._tail_width,
                mode="lines",
                antialias=True,
            )
            item.setGLOptions("translucent")
            self._3d_view.addItem(item)
            self._3d_tail_items.append(item)

    def _capture_track_snapshots(self) -> tuple[_TrackChannelSnapshot, ...]:
        """Capture non-Qt inputs; workers never dereference mutable datasets."""
        snapshots = []
        for channel in self._channels:
            idx = channel["dataset_idx"]
            ds = self._state.datasets[idx]

            def read(name: str):
                try:
                    value = attr_values_1d(ds, name)
                except Exception:
                    return None
                if value is None:
                    return None
                array = np.asarray(value).reshape(-1).view()
                array.flags.writeable = False
                return array if array.size else None

            tim, tid = dataset_time_inputs(ds)
            keep = np.asarray(ds.filter_mask, dtype=bool).reshape(-1).copy()
            keep.flags.writeable = False
            transform = ds.state.get("overlay_transform") or ds.state.get(
                "render_transform_2d")
            tim = None if tim is None else np.asarray(tim).reshape(-1).view()
            tid = None if tid is None else np.asarray(tid).reshape(-1).view()
            if tim is not None:
                tim.flags.writeable = False
            if tid is not None:
                tid.flags.writeable = False
            snapshots.append(_TrackChannelSnapshot(
                dataset_idx=idx,
                dataset_identity=id(ds),
                x_m=read("loc_x"),
                y_m=read("loc_y"),
                z_m=read("loc_z"),
                z_scaling_factor=float(ds.cali.z_scaling_factor),
                display_transform=copy.deepcopy(transform),
                timestamps_s=tim,
                trace_ids=tid,
                keep=keep,
            ))
        return tuple(snapshots)

    def _request_track_build(
        self,
        *,
        reset_window: bool,
        fit_view: bool,
        rebuild_rows: bool,
        with_diagnostics: bool | None = None,
    ) -> None:
        """Queue the newest immutable track snapshot and retire older generations."""
        if self._closing:
            return
        from .background_tasks import BackgroundTask, request_task_cancel

        self._track_generation += 1
        generation = self._track_generation
        for old_generation, task in tuple(self._track_tasks.items()):
            if old_generation != generation:
                request_task_cancel(task)
        try:
            snapshots = self._capture_track_snapshots()
        except Exception as exc:  # noqa: BLE001 - snapshot errors belong in the view
            self._on_track_build_failed(generation, str(exc))
            return
        if not snapshots:
            self._on_track_build_failed(generation, "no localization channels")
            return

        diagnostics = (
            self._time_row.wants_diagnostics() or reset_window
            if with_diagnostics is None else bool(with_diagnostics)
        )
        anchor_idx = int(self._idx) if self._idx is not None else -1
        mode = self._time_row.mode()
        precision = self._time_precision_s()

        def work(report):
            return _build_track_payload(
                snapshots,
                anchor_idx=anchor_idx,
                mode=mode,
                precision_s=precision,
                with_diagnostics=diagnostics,
                role_displacement_nm=self._role_displacement_nm,
                role_min_median_locs=self._role_min_median_locs,
                report=report,
            )

        anchor = self._dataset()
        name = str(getattr(anchor, "name", "dataset"))
        task = BackgroundTask(
            work,
            description=f"Index Tracking View trajectories: {name}",
            category="analysis",
        )
        task.signals.stage.connect(
            lambda text, g=generation: self._on_track_build_stage(g, text))
        task.signals.done.connect(
            lambda result, g=generation: self._on_track_build_done(
                g,
                result,
                reset_window=reset_window,
                fit_view=fit_view,
                rebuild_rows=rebuild_rows,
            ))
        task.signals.failed.connect(
            lambda message, g=generation: self._on_track_build_failed(g, message))
        task.signals.cancelled.connect(
            lambda g=generation: self._on_track_build_cancelled(g))
        task.signals.finished.connect(
            lambda g=generation, t=task: self._forget_track_task(g, t))
        self._track_tasks[generation] = task
        self._building_tracks = True
        self._clear_track_display()
        self._info_label.setText("Indexing trajectories…")
        self._start_track_task(task)

    def _start_track_task(self, task) -> None:
        """Start on the process-owned serial pool (a seam for deterministic tests)."""
        from .background_tasks import shared_thread_pool

        shared_thread_pool("tracking-index", max_threads=1).start(task)

    def _on_track_build_stage(self, generation: int, text: str) -> None:
        if generation == self._track_generation and not self._closing:
            self._info_label.setText(f"Indexing trajectories… {text}")

    def _forget_track_task(self, generation: int, task) -> None:
        if self._track_tasks.get(generation) is task:
            self._track_tasks.pop(generation, None)

    def _on_track_build_done(
        self,
        generation: int,
        result: _TrackBuildResult,
        *,
        reset_window: bool,
        fit_view: bool,
        rebuild_rows: bool,
    ) -> None:
        if generation != self._track_generation or self._closing:
            return
        channels = {ch["dataset_idx"]: ch for ch in self._channels}
        source_is_current = all(
            built.dataset_idx in channels
            and 0 <= built.dataset_idx < len(self._state.datasets)
            and id(self._state.datasets[built.dataset_idx])
            == built.dataset_identity
            for built in result.channels
        )
        if not source_is_current:
            self._on_track_build_failed(
                generation, "a source dataset changed while trajectories were indexed")
            return
        self._building_tracks = False
        self._tracks = {}
        self._track_reasons = {}
        for built in result.channels:
            channel = channels.get(built.dataset_idx)
            if channel is None:  # guarded above; keeps the type narrow below
                continue
            ds = self._state.datasets[built.dataset_idx]
            ds.state["tracking_role_auto"] = built.automatic_role
            ds.state["tracking_role_auto_reason"] = built.reason
            manual = ds.state.get("tracking_role")
            if manual in {"Tracking", "Structure"}:
                channel["role"] = manual
                channel["reason"] = (
                    "User-selected role "
                    f"(automatic: {built.automatic_role.lower()}; {built.reason})"
                )
            else:
                channel["role"] = built.automatic_role
                channel["reason"] = built.reason
            self._track_reasons[built.dataset_idx] = built.reason
            if built.tracks is not None:
                self._tracks[built.dataset_idx] = built.tracks

        if self._selected_track is not None:
            selected_idx, selected_tid = self._selected_track
            selected_tracks = self._tracks.get(selected_idx)
            if (
                selected_tracks is None
                or not np.any(selected_tracks.trace_ids == selected_tid)
            ):
                self._selected_track = None
                self._published_playhead = None

        self._backdrop_key_cached = None
        if rebuild_rows:
            self._rebuild_channel_rows()
        self._apply_time_axis(result.anchor_axis, reason=result.axis_error)
        if reset_window:
            self._open_the_gate()
        if fit_view:
            self._fit_view()
        self._draw()

    def _on_track_build_failed(self, generation: int, message: str) -> None:
        if generation != self._track_generation or self._closing:
            return
        self._building_tracks = False
        self._tracks = {}
        self._time_row.set_axis(None, reason=str(message))
        self._clear_track_display()
        self._info_label.setText(f"Could not index trajectories: {message}")
        self._state.log(f"Tracking View indexing failed: {message}", "ERROR")

    def _on_track_build_cancelled(self, generation: int) -> None:
        if generation != self._track_generation or self._closing:
            return
        self._building_tracks = False
        self._tracks = {}
        self._time_row.set_axis(None, reason="trajectory indexing cancelled")
        self._clear_track_display()
        self._info_label.setText("Trajectory indexing cancelled.")

    def _on_diagnostics_requested(self) -> None:
        axis = self._time_row.axis()
        if axis is None or axis.is_index or axis.diagnostics is not None:
            return
        self._request_track_build(
            reset_window=False,
            fit_view=False,
            rebuild_rows=False,
            with_diagnostics=True,
        )

    def _time_precision_s(self) -> float:
        try:
            return tracking_precision_from_prefs(self._state.prefs)
        except (TypeError, ValueError):
            return 1.0e-6

    def _refresh_from_dataset(self) -> None:
        ds = self._dataset()
        if ds is None:
            self._info_label.setText("No dataset.")
            return
        self.setWindowTitle(f"Tracking View — {ds.name}")
        self._build_channels()
        # A tracking view is about time, so its axis mode is chosen before the
        # tracks are indexed: trace-relative unless the dataset cannot offer it.
        modes = available_time_modes(ds)
        self._time_row.set_available_modes(modes)
        preferred = str((self._state.prefs.get("tracking", {}) or {}).get(
            "default_axis_mode", "trace"))
        if preferred in modes and self._time_row.mode() != preferred:
            self._time_row._sync_mode_combo(preferred)
        elif "trace" in modes and self._time_row.mode() not in modes:
            self._time_row._sync_mode_combo("trace")
        self._rebuild_channel_rows()
        self._request_track_build(
            reset_window=True, fit_view=True, rebuild_rows=True)

    def _apply_time_axis(self, axis, *, reason: str = "") -> None:
        """Install the worker-built anchor axis, extended across overlay tracks."""
        row = self._time_row
        if axis is None:
            row.set_axis(None, reason=reason or "no playable trajectory axis")
            return
        tracks = [t for t in self._tracks.values() if t.n_points]
        if tracks:
            if row.mode() == "absolute":
                lo = min(float(t.t_min) for t in tracks)
            else:
                lo = 0.0
            hi = max(float(t.t_max) for t in tracks)
            if hi <= lo:
                hi = lo + (1.0 if axis.is_index else max(abs(lo) * 1e-9, 1e-9))
            axis = replace(
                axis,
                lo=lo,
                hi=hi,
                step=derived_step(hi - lo, axis.interval),
                n_traces=sum(t.n_traces for t in tracks),
            )
        row.set_axis(axis)

    def _open_the_gate(self) -> None:
        """Start with a comet, not with the whole track drawn at once."""
        axis = self._time_row.axis()
        if axis is None:
            return
        self._time_row.set_window(
            axis.lo, axis.lo + max(axis.span * self._tail_fraction, axis.step))

    def _on_time_mode_changed(self, _mode: str) -> None:
        self._request_track_build(
            reset_window=True, fit_view=False, rebuild_rows=True)

    def _on_dataset_changed(self, idx: int) -> None:
        if self._closing:
            return
        if any(ch["dataset_idx"] == idx for ch in self._channels):
            self._request_track_build(
                reset_window=False, fit_view=False, rebuild_rows=True)

    def refresh_preferences(self) -> None:
        prefs = self._state.prefs.get("tracking", {}) or {}
        bands = int(np.clip(prefs.get("tail_bands", DEFAULT_TAIL_BANDS), 1, 24))
        self._tail_bands = bands
        self._ensure_tail_items(len(self._channels))
        self._tail_width = max(float(prefs.get("tail_width", TAIL_WIDTH)), 0.1)
        self._tail_alpha_head = int(np.clip(
            prefs.get("tail_head_opacity", TAIL_ALPHA_HEAD), 0, 255))
        self._tail_alpha_tip = int(np.clip(
            prefs.get("tail_tip_opacity", TAIL_ALPHA_TIP), 0, 255))
        self._tail_fraction = float(np.clip(
            prefs.get("tail_fraction", DEFAULT_TAIL_FRACTION), 0.0001, 1.0))
        tail_color_mode = str(prefs.get("tail_color_mode", "Channel"))
        self._tail_color_mode = (
            tail_color_mode if tail_color_mode in TAIL_COLOR_MODES else "Channel")
        self._head_symbol = str(prefs.get("head_symbol", HEAD_SYMBOL))
        self._head_size = int(np.clip(
            prefs.get("head_size", DEFAULT_HEAD_SIZE), 2, 64))
        self._backdrop_mode = str(prefs.get("backdrop_mode", "Render"))
        if self._backdrop_mode not in BACKDROP_MODES:
            self._backdrop_mode = "Render"
        self._backdrop_level = str(prefs.get(
            "backdrop_level", DEFAULT_BACKDROP_LEVEL))
        if self._backdrop_level not in BACKDROP_LEVELS:
            self._backdrop_level = DEFAULT_BACKDROP_LEVEL
        projection = str(prefs.get("default_projection", self._axis))
        if projection in AXIS_OPTIONS:
            self._axis = projection
        self._apply_y_axis_direction()
        self._role_displacement_nm = max(float(
            prefs.get("role_displacement_nm", 25.0)), 0.0)
        self._role_min_median_locs = max(int(
            prefs.get("role_min_median_locs", 5)), 2)
        row_state = self._time_row.state()
        row_state.update({
            "rate_hz": float(prefs.get("playback_rate_hz", 10.0)),
            "loop": bool(prefs.get("playback_loop", True)),
            "grow": bool(prefs.get("tail_grow", False)),
        })
        self._time_row.restore_state(row_state)
        self._backdrop_key_cached = None
        self._request_track_build(
            reset_window=False, fit_view=True, rebuild_rows=True)

    # -------------------------------------------------------------- drawing

    def _clear_track_display(self) -> None:
        for item in self._tail_items:
            item.setData(x=[], y=[])
        self._head_item.setData(x=[], y=[])
        self._candidate_xy = np.empty((0, 2), dtype=float)
        self._candidate_dataset_indices = np.empty(0, dtype=np.int64)
        self._candidate_trace_ids = np.empty(0, dtype=object)
        self._backdrop_image.setVisible(False)
        self._backdrop_scatter.setVisible(False)
        if self._3d_backdrop is not None:
            self._3d_backdrop.setData(pos=np.empty((0, 3), dtype=np.float32))
        if self._3d_head is not None:
            self._3d_head.setData(pos=np.empty((0, 3), dtype=np.float32))
        for item in self._3d_tail_items:
            item.setData(pos=np.empty((0, 3), dtype=np.float32))

    def _columns(self) -> tuple[int, int]:
        return AXIS_COLUMNS.get(self._axis, (0, 1))

    def _visible_channels(self, role: str) -> list[dict]:
        return [ch for ch in self._channels
                if ch["visible"] and ch["role"] == role
                and ch["dataset_idx"] in self._tracks]

    def _all_points(self) -> np.ndarray:
        parts = [self._tracks[ch["dataset_idx"]].coords
                 for ch in self._channels if ch["dataset_idx"] in self._tracks]
        if not parts:
            return np.empty((0, 3), dtype=float)
        return np.vstack(parts)

    def _fit_view(self) -> None:
        points = self._all_points()
        if self._axis == "3D":
            self._reset_3d_camera(points)
            return
        ci, cj = self._columns()
        if points.shape[0] == 0:
            self._bounds = None
            return
        x, y = points[:, ci], points[:, cj]
        pad = 0.03 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
        self._bounds = (float(x.min()) - pad, float(x.max()) + pad,
                        float(y.min()) - pad, float(y.max()) + pad)
        self._plot.setXRange(self._bounds[0], self._bounds[1], padding=0)
        self._plot.setYRange(self._bounds[2], self._bounds[3], padding=0)
        self._plot.setLabel("bottom", f"{self._axis[0]} (nm)")
        self._plot.setLabel("left", f"{self._axis[1]} (nm)")

    def _schedule_draw(self) -> None:
        if not self._closing:
            self._redraw_timer.start()

    def _draw(self) -> None:
        if self._closing:
            return
        if self._building_tracks:
            self._clear_track_display()
            return
        import time
        started = time.perf_counter()
        if self._axis == "3D":
            drawn, traces = self._draw_3d()
            self._last_frame_ms = (time.perf_counter() - started) * 1e3
            self._update_info(drawn, traces)
            self._publish_playhead()
            return
        self._draw_backdrop()
        drawn, traces = self._draw_comets()
        self._last_frame_ms = (time.perf_counter() - started) * 1e3
        self._update_info(drawn, traces)
        self._publish_playhead()

    # -- the backdrop ------------------------------------------------------

    def _backdrop_points(self) -> np.ndarray:
        """What the comets are drawn over.

        The structure channels when there are any -- the two-colour case, where
        the scaffold is the reference frame. Otherwise every localization of the
        tracking channels themselves, which is the where-has-it-all-been context
        a single-channel tracking run has instead of a scaffold.
        """
        structure = self._visible_channels("Structure")
        source = structure or self._visible_channels("Tracking")
        parts = [self._tracks[ch["dataset_idx"]].coords for ch in source]
        if not parts:
            return np.empty((0, 3), dtype=float)
        return np.vstack(parts)

    def _backdrop_key(self) -> tuple:
        """Everything the backdrop depends on -- and time is not in it.

        The structure does not move, so recomputing its raster on every frame was
        pure waste: measured 22.9 ms/frame, against 3.3 ms for the comets alone.
        """
        return (
            self._backdrop_mode, self._backdrop_level, self._axis, self._black_bg,
            self._bounds,
            tuple((ch["dataset_idx"], ch["visible"], ch["role"])
                  for ch in self._channels),
            tuple(id(self._tracks.get(ch["dataset_idx"])) for ch in self._channels),
        )

    def _draw_backdrop(self) -> None:
        key = self._backdrop_key()
        if key == self._backdrop_key_cached:
            return
        self._backdrop_key_cached = key
        self._draw_backdrop_now()

    def _draw_backdrop_3d(self) -> None:
        if self._3d_backdrop is None:
            return
        key = self._backdrop_key()
        if key == self._backdrop_key_cached:
            return
        self._backdrop_key_cached = key
        if self._backdrop_mode == "None":
            self._3d_backdrop.setData(pos=np.empty((0, 3), dtype=np.float32))
            return
        points = self._backdrop_points()
        if points.size == 0:
            self._3d_backdrop.setData(pos=np.empty((0, 3), dtype=np.float32))
            return
        pick = _thin(points.shape[0], 150_000)
        pos = np.asarray(points[pick, :3], dtype=np.float32)
        shade = 0.82 if self._black_bg else 0.24
        alpha = BACKDROP_LEVELS.get(self._backdrop_level, 175) / 255.0 * 0.45
        color = np.tile(np.asarray([shade, shade, shade, alpha], dtype=np.float32),
                        (pos.shape[0], 1))
        self._3d_backdrop.setData(pos=pos, color=color, size=2.0, pxMode=True)

    def _draw_backdrop_now(self) -> None:
        if self._backdrop_mode == "None" or self._bounds is None:
            self._backdrop_image.setVisible(False)
            self._backdrop_scatter.setVisible(False)
            return
        points = self._backdrop_points()
        ci, cj = self._columns()
        if points.shape[0] == 0:
            self._backdrop_image.setVisible(False)
            self._backdrop_scatter.setVisible(False)
            return

        if self._backdrop_mode == "Scatter":
            self._backdrop_image.setVisible(False)
            pick = _thin(points.shape[0], BACKDROP_MAX_POINTS)
            shade = 210 if self._black_bg else 60
            alpha = int(BACKDROP_LEVELS.get(self._backdrop_level, 175) * 0.55)
            self._backdrop_scatter.setData(
                x=points[pick, ci], y=points[pick, cj],
                brush=pg.mkBrush(shade, shade, shade, alpha),
                pen=None, size=2)
            self._backdrop_scatter.setVisible(True)
            return

        self._backdrop_scatter.setVisible(False)
        from .precision_render import render_gaussian_filtered

        x0, x1, y0, y1 = self._bounds
        aspect = max(y1 - y0, 1e-9) / max(x1 - x0, 1e-9)
        width = BACKDROP_MAX_PX
        height = int(np.clip(round(width * aspect), 1, BACKDROP_MAX_PX))
        px = (x1 - x0) / width
        scalar = render_gaussian_filtered(
            points[:, ci], points[:, cj], (x0, x1, y0, y1), (height, width),
            BACKDROP_SIGMA_PX * px, BACKDROP_SIGMA_PX * px)
        top = float(np.percentile(scalar[scalar > 0], 99.0)) if np.any(scalar > 0) else 1.0
        norm = np.clip(scalar / max(top, 1e-9), 0.0, 1.0) ** BACKDROP_GAMMA
        ceiling = BACKDROP_LEVELS.get(self._backdrop_level, 175)
        # A grey ramp, so the structure reads as context and the comet colours
        # are the only hues in the picture.
        rgba = np.zeros((*norm.shape, 4), dtype=np.uint8)
        level = (norm * 255).astype(np.uint8)
        if not self._black_bg:
            level = 255 - level
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = level
        rgba[..., 3] = (norm * ceiling).astype(np.uint8)
        self._backdrop_image.setImage(np.transpose(rgba, (1, 0, 2)), autoLevels=False)
        self._backdrop_image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self._backdrop_image.setVisible(True)

    # -- the comets --------------------------------------------------------

    def _comet_channels(self) -> list[dict]:
        channels = self._visible_channels("Tracking")
        if self._show_structure_tracks:
            channels = channels + self._visible_channels("Structure")
        return channels

    def _draw_comets(self) -> tuple[int, int]:
        window = self._time_row.range()
        axis = self._time_row.axis()
        channels = self._comet_channels()
        if window is None or axis is None or not channels:
            for item in self._tail_items:
                item.setData(x=[], y=[])
            self._head_item.setData(x=[], y=[])
            self._candidate_xy = np.empty((0, 2), dtype=float)
            self._candidate_dataset_indices = np.empty(0, dtype=np.int64)
            self._candidate_trace_ids = np.empty(0, dtype=object)
            return 0, 0

        t_lo, t_head = window
        ci, cj = self._columns()
        bands = band_edges(t_head, t_head - t_lo, axis.lo, self._tail_bands)

        for item in self._tail_items:
            item.setData(x=[], y=[])

        # One curve per channel and band keeps both the fast constant-opacity
        # painter path and the channel's LUT identity.
        drawn = 0
        candidate_xy: list[np.ndarray] = []
        candidate_dataset: list[np.ndarray] = []
        candidate_tid: list[np.ndarray] = []
        for channel_pos, ch in enumerate(channels):
            r, g, b = ch["color"]
            for band_pos, (lo, hi, age) in enumerate(bands):
                item = self._tail_items[channel_pos * self._tail_bands + band_pos]
                tracks = self._tracks[ch["dataset_idx"]]
                sel = tracks.window(lo, hi)
                if sel.size == 0:
                    continue
                indices = sel.indices
                connect = sel.connect
                if self._selected_track is not None:
                    selected_idx, selected_tid = self._selected_track
                    if ch["dataset_idx"] != selected_idx:
                        continue
                    selected = tracks.trace_ids[tracks.trace_codes[indices]] == selected_tid
                    indices = indices[selected]
                    connect = connect[selected]
                    if indices.size:
                        connect = np.asarray(connect, dtype=np.uint8).copy()
                        connect[-1] = 0
                if indices.size == 0:
                    continue
                pts = tracks.coords[indices]
                alpha = int(round(
                    self._tail_alpha_tip
                    + (self._tail_alpha_head - self._tail_alpha_tip) * (1.0 - age)))
                color = (
                    self._time_band_color(age)
                    if self._tail_color_mode == "Time" else (r, g, b))
                item.setData(
                    x=pts[:, ci], y=pts[:, cj], connect=connect,
                    pen=pg.mkPen(*color, alpha, width=self._tail_width))
                drawn += int(pts.shape[0])
                candidate_xy.append(pts[:, [ci, cj]])
                candidate_dataset.append(np.full(
                    indices.size, ch["dataset_idx"], dtype=np.int64))
                candidate_tid.append(tracks.trace_ids[tracks.trace_codes[indices]])

        heads_x: list[np.ndarray] = []
        heads_y: list[np.ndarray] = []
        head_brushes = []
        for ch in channels:
            tracks = self._tracks[ch["dataset_idx"]]
            sel = tracks.window(max(t_lo, axis.lo), t_head)
            if sel.n_traces == 0:
                continue
            head_indices = sel.head_indices
            head_traces = sel.head_traces
            if self._selected_track is not None:
                selected_idx, selected_tid = self._selected_track
                if ch["dataset_idx"] != selected_idx:
                    continue
                selected = tracks.trace_ids[head_traces] == selected_tid
                head_indices = head_indices[selected]
                head_traces = head_traces[selected]
            if head_indices.size == 0:
                continue
            pts = tracks.coords[head_indices]
            heads_x.append(pts[:, ci])
            heads_y.append(pts[:, cj])
            head_brushes.extend(
                pg.mkBrush(*ch["color"], 250) for _ in range(pts.shape[0]))
        if candidate_xy:
            self._candidate_xy = np.vstack(candidate_xy)
            self._candidate_dataset_indices = np.concatenate(candidate_dataset)
            self._candidate_trace_ids = np.concatenate(candidate_tid)
        else:
            self._candidate_xy = np.empty((0, 2), dtype=float)
            self._candidate_dataset_indices = np.empty(0, dtype=np.int64)
            self._candidate_trace_ids = np.empty(0, dtype=object)
        if heads_x:
            hx = np.concatenate(heads_x)
            self._head_item.setData(
                x=hx, y=np.concatenate(heads_y),
                size=self._head_size, symbol=self._head_symbol,
                brush=head_brushes,
                pen=pg.mkPen(0, 0, 0, 160, width=0.8))
            return drawn, int(hx.size)
        self._head_item.setData(x=[], y=[])
        return drawn, 0

    @staticmethod
    def _line_segments_3d(points: np.ndarray, connect: np.ndarray) -> np.ndarray:
        if points.shape[0] < 2:
            return np.empty((0, 3), dtype=np.float32)
        joined = np.asarray(connect[:-1], dtype=bool)
        if not np.any(joined):
            return np.empty((0, 3), dtype=np.float32)
        return np.stack([points[:-1][joined], points[1:][joined]], axis=1).reshape(
            -1, 3).astype(np.float32, copy=False)

    def _draw_3d(self) -> tuple[int, int]:
        if not self._ensure_3d_built():
            return 0, 0
        self._view_stack.setCurrentWidget(self._3d_view)
        self._draw_backdrop_3d()
        for item in self._3d_tail_items:
            item.setData(pos=np.empty((0, 3), dtype=np.float32))
        window = self._time_row.range()
        axis = self._time_row.axis()
        channels = self._comet_channels()
        if window is None or axis is None or not channels:
            self._3d_head.setData(pos=np.empty((0, 3), dtype=np.float32))
            return 0, 0
        t_lo, t_head = window
        bands = band_edges(t_head, t_head - t_lo, axis.lo, self._tail_bands)
        self._ensure_3d_tail_items(max(1, len(channels) * self._tail_bands))
        drawn = 0
        for channel_pos, ch in enumerate(channels):
            for band_pos, (lo, hi, age) in enumerate(bands):
                tracks = self._tracks[ch["dataset_idx"]]
                selected_window = tracks.window(lo, hi)
                indices = selected_window.indices
                connect = selected_window.connect
                if self._selected_track is not None:
                    selected_idx, selected_tid = self._selected_track
                    if ch["dataset_idx"] != selected_idx:
                        continue
                    selected = tracks.trace_ids[tracks.trace_codes[indices]] == selected_tid
                    indices = indices[selected]
                    connect = connect[selected]
                    if indices.size:
                        connect = np.asarray(connect, dtype=np.uint8).copy()
                        connect[-1] = 0
                if indices.size == 0:
                    continue
                points = tracks.coords[indices]
                segments = self._line_segments_3d(points, connect)
                if segments.size == 0:
                    continue
                base = (
                    self._time_band_color(age)
                    if self._tail_color_mode == "Time" else ch["color"])
                alpha = (
                    self._tail_alpha_tip
                    + (self._tail_alpha_head - self._tail_alpha_tip) * (1.0 - age)
                ) / 255.0
                color = tuple(value / 255.0 for value in base) + (float(alpha),)
                item = self._3d_tail_items[
                    channel_pos * self._tail_bands + band_pos]
                item.setData(
                    pos=segments, color=color, width=self._tail_width, mode="lines")
                drawn += int(indices.size)

        head_points: list[np.ndarray] = []
        head_colors: list[np.ndarray] = []
        for ch in channels:
            tracks = self._tracks[ch["dataset_idx"]]
            selected_window = tracks.window(max(t_lo, axis.lo), t_head)
            indices = selected_window.head_indices
            trace_codes = selected_window.head_traces
            if self._selected_track is not None:
                selected_idx, selected_tid = self._selected_track
                if ch["dataset_idx"] != selected_idx:
                    continue
                selected = tracks.trace_ids[trace_codes] == selected_tid
                indices = indices[selected]
            if indices.size:
                head_points.append(tracks.coords[indices])
                rgba = np.asarray((*ch["color"], 250), dtype=float) / 255.0
                head_colors.append(np.tile(rgba, (indices.size, 1)))
        if head_points:
            pos = np.vstack(head_points).astype(np.float32, copy=False)
            color = np.vstack(head_colors).astype(np.float32, copy=False)
            self._3d_head.setData(
                pos=pos, color=color, size=float(self._head_size), pxMode=True)
            if not self._3d_camera_initialized:
                self._reset_3d_camera(self._all_points())
            return drawn, int(pos.shape[0])
        self._3d_head.setData(pos=np.empty((0, 3), dtype=np.float32))
        return drawn, 0

    def _update_info(self, drawn: int, heads: int) -> None:
        axis = self._time_row.axis()
        window = self._time_row.range()
        pieces = []
        tracking = self._visible_channels("Tracking")
        structure = self._visible_channels("Structure")
        total = sum(self._tracks[ch["dataset_idx"]].n_traces for ch in tracking)
        pieces.append(f"{heads:,} of {total:,} track(s) live")
        pieces.append(f"{drawn:,} tail points")
        if structure:
            pieces.append(f"{len(structure)} structure channel(s)")
        if axis is not None and window is not None:
            pieces.append(f"t = {format_time_seconds(window[1])}"
                          f"  tail {format_time_seconds(window[1] - window[0])}")
        pieces.append(f"{self._last_frame_ms:.1f} ms/frame")
        if self._selected_track is not None:
            pieces.append(f"selected tid {self._selected_track[1]}")
        if self._hover_summary:
            pieces.append(self._hover_summary)
        self._info_label.setText("  |  ".join(pieces))

    # ---------------------------------------------------------- interaction

    @staticmethod
    def _time_band_color(age: float) -> tuple[int, int, int]:
        """Viridis-like oldest-to-newest colour for one constant-colour band."""
        position = float(np.clip(1.0 - age, 0.0, 1.0))
        scaled = position * (_TIME_COLORS.shape[0] - 1)
        left = min(int(np.floor(scaled)), _TIME_COLORS.shape[0] - 2)
        fraction = scaled - left
        rgb = (1.0 - fraction) * _TIME_COLORS[left] + fraction * _TIME_COLORS[left + 1]
        return tuple(int(round(value)) for value in rgb)

    def _nearest_candidate(
        self,
        point: QPointF,
        *,
        maximum_pixels: float,
    ) -> int | None:
        if self._candidate_xy.size == 0:
            return None
        ranges = self._plot.viewRange()
        x_span = max(float(ranges[0][1] - ranges[0][0]), np.finfo(float).eps)
        y_span = max(float(ranges[1][1] - ranges[1][0]), np.finfo(float).eps)
        dx = (self._candidate_xy[:, 0] - point.x()) / x_span * max(self._plot.width(), 1)
        dy = (self._candidate_xy[:, 1] - point.y()) / y_span * max(self._plot.height(), 1)
        squared = dx * dx + dy * dy
        nearest = int(np.argmin(squared))
        return nearest if squared[nearest] <= float(maximum_pixels) ** 2 else None

    def _event_view_point(self, event) -> QPointF | None:
        scene_pos = event[0] if isinstance(event, (tuple, list)) else event
        bounds = self._plot.getPlotItem().sceneBoundingRect()
        if scene_pos is None or not bounds.contains(scene_pos):
            return None
        return self._plot.getPlotItem().getViewBox().mapSceneToView(scene_pos)

    def _on_plot_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._event_view_point(event.scenePos())
        nearest = None if point is None else self._nearest_candidate(
            point, maximum_pixels=12.0)
        if nearest is None:
            self._selected_track = None
        else:
            self._selected_track = (
                int(self._candidate_dataset_indices[nearest]),
                self._candidate_trace_ids[nearest].item()
                if hasattr(self._candidate_trace_ids[nearest], "item")
                else self._candidate_trace_ids[nearest],
            )
        self._hover_summary = ""
        self._draw()

    def _on_plot_moved(self, event) -> None:
        point = self._event_view_point(event)
        nearest = None if point is None else self._nearest_candidate(
            point, maximum_pixels=9.0)
        summary = ""
        if nearest is not None:
            summary = self._track_summary(
                int(self._candidate_dataset_indices[nearest]),
                self._candidate_trace_ids[nearest],
            )
        if summary != self._hover_summary:
            self._hover_summary = summary
            self._update_info(0, 0) if self._building_tracks else self._draw_info_only()

    def _draw_info_only(self) -> None:
        window = self._time_row.range()
        if window is None:
            return
        drawn = int(self._candidate_xy.shape[0])
        heads = 1 if self._selected_track is not None and drawn else 0
        self._update_info(drawn, heads)

    def _track_summary(self, dataset_idx: int, trace_id) -> str:
        tracks = self._tracks.get(dataset_idx)
        if tracks is None:
            return ""
        match = np.flatnonzero(tracks.trace_ids == trace_id)
        if match.size != 1:
            return ""
        code = int(match[0])
        start = int(tracks.starts[code])
        end = start + int(tracks.counts[code])
        points = tracks.coords[start:end]
        elapsed = np.diff(tracks.t[start:end])
        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        duration = float(tracks.t[end - 1] - tracks.t[start]) if end - start > 1 else 0.0
        net = float(np.linalg.norm(points[-1] - points[0])) if end - start > 1 else 0.0
        valid = elapsed > 0
        median_speed = float(np.median(distance[valid] / elapsed[valid])) if np.any(valid) else np.nan
        speed_text = f"{median_speed:.1f} nm/{'s' if self._time_row.mode() != 'index' else 'index'}"
        return (
            f"tid {trace_id}: {end - start} loc, {duration:.4g} "
            f"{'s' if self._time_row.mode() != 'index' else 'index'}, "
            f"net {net:.1f} nm, median {speed_text}")

    def _set_tail_color_mode(self, mode: str) -> None:
        if mode in TAIL_COLOR_MODES:
            self._tail_color_mode = mode
            self._draw()

    def _clear_selected_track(self) -> None:
        self._selected_track = None
        self._published_playhead = None
        self._draw()

    def _publish_playhead(self) -> None:
        if self._selected_track is None or self._receiving_playhead:
            return
        dataset_idx, trace_id = self._selected_track
        tracks = self._tracks.get(dataset_idx)
        window = self._time_row.range()
        if tracks is None or window is None:
            return
        match = np.flatnonzero(tracks.trace_ids == trace_id)
        if match.size != 1 or not (0 <= dataset_idx < len(self._state.datasets)):
            return
        code = int(match[0])
        head_time = float(window[1])
        relative_time = head_time - float(tracks.t_start[code])
        key = (dataset_idx, trace_id, round(relative_time, 12))
        if key == self._published_playhead:
            return
        self._published_playhead = key
        self._state.tracking_playhead_changed.emit(
            self._state.datasets[dataset_idx], trace_id, relative_time, self)

    def _receive_playhead(
        self,
        dataset,
        trace_id,
        trace_time: float,
        source,
    ) -> None:
        if source is self or self._closing:
            return
        matches = [
            ch["dataset_idx"] for ch in self._channels
            if 0 <= ch["dataset_idx"] < len(self._state.datasets)
            and self._state.datasets[ch["dataset_idx"]] is dataset
        ]
        if not matches:
            return
        dataset_idx = int(matches[0])
        tracks = self._tracks.get(dataset_idx)
        if tracks is None:
            return
        code_match = np.flatnonzero(tracks.trace_ids == trace_id)
        if code_match.size != 1:
            return
        code = int(code_match[0])
        axis = self._time_row.axis()
        window = self._time_row.range()
        if axis is None or window is None:
            return
        head = float(trace_time) + float(tracks.t_start[code])
        width = max(float(window[1] - window[0]), axis.step)
        head = float(np.clip(head, axis.lo, axis.hi))
        lo = max(axis.lo, head - width)
        self._receiving_playhead = True
        try:
            self._selected_track = (dataset_idx, tracks.trace_ids[code])
            self._time_row.set_window(lo, head)
            self._draw()
        finally:
            self._receiving_playhead = False

    def _set_channel_visible(self, pos: int, on: bool) -> None:
        if 0 <= pos < len(self._channels):
            self._channels[pos]["visible"] = bool(on)
            self._schedule_draw()

    def _set_channel_role(self, pos: int, role: str) -> None:
        if 0 <= pos < len(self._channels) and role in {"Tracking", "Structure"}:
            channel = self._channels[pos]
            channel["role"] = role
            ds = self._state.datasets[channel["dataset_idx"]]
            ds.state["tracking_role"] = role
            channel["reason"] = "User-selected Tracking View role"
            self._schedule_draw()

    def _apply_background(self) -> None:
        self._plot.setBackground("k" if self._black_bg else "w")
        axis_color = "#dddddd" if self._black_bg else "#222222"
        for side in ("bottom", "left"):
            ax = self._plot.getPlotItem().getAxis(side)
            ax.setPen(pg.mkPen(axis_color))
            ax.setTextPen(pg.mkPen(axis_color))
        if self._3d_view is not None:
            self._3d_view.setBackgroundColor("k" if self._black_bg else "w")

    def _apply_y_axis_direction(self) -> None:
        _horizontal, vertical = AXIS_COLUMNS.get(self._axis, (0, 1))
        apply_spatial_y_direction(
            self._plot,
            vertical_coordinate="XYZ"[vertical],
            prefs=self._state.prefs,
            preference_key="scatter_xy_origin",
        )

    def _set_axis(self, axis: str) -> None:
        if axis in AXIS_OPTIONS and axis != self._axis:
            if axis == "3D" and not self._ensure_3d_built():
                return
            self._axis = axis
            self._apply_y_axis_direction()
            self._view_stack.setCurrentWidget(
                self._3d_view if axis == "3D" else self._plot)
            self._fit_view()
            self._draw()

    def _set_backdrop(self, mode: str) -> None:
        if mode in BACKDROP_MODES:
            self._backdrop_mode = mode
            self._draw()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        view = menu.addMenu("View as")
        for name in AXIS_OPTIONS:
            action = view.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self._axis)
            action.triggered.connect(lambda _c=False, n=name: self._set_axis(n))

        backdrop = menu.addMenu("Structure backdrop")
        backdrop.setToolTipsVisible(True)
        tips = {
            "Render": "A grey reconstruction of the structure, behind the comets",
            "Scatter": "The structure's localizations as faint points",
            "None": "Comets only",
        }
        for name in BACKDROP_MODES:
            action = backdrop.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self._backdrop_mode)
            action.setToolTip(tips[name])
            action.triggered.connect(lambda _c=False, n=name: self._set_backdrop(n))

        level = menu.addMenu("Structure brightness")
        level.setEnabled(self._backdrop_mode != "None")
        for name in BACKDROP_LEVELS:
            action = level.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self._backdrop_level)
            action.triggered.connect(
                lambda _c=False, n=name: self._set_backdrop_level(n))

        black = menu.addAction("Black background")
        black.setCheckable(True)
        black.setChecked(self._black_bg)
        black.triggered.connect(self._set_black_background)

        structure_tracks = menu.addAction("Draw structure channels as comets too")
        structure_tracks.setCheckable(True)
        structure_tracks.setChecked(self._show_structure_tracks)
        structure_tracks.setToolTip(
            "A structure channel's traces barely move, so its comets sit still. "
            "Useful to check a channel really is the scaffold.")
        structure_tracks.triggered.connect(self._set_show_structure_tracks)

        size = menu.addMenu("Head size")
        for value in (7, 9, 11, 14, 18):
            action = size.addAction(str(value))
            action.setCheckable(True)
            action.setChecked(value == self._head_size)
            action.triggered.connect(lambda _c=False, v=value: self._set_head_size(v))

        tail_color = menu.addMenu("Tail colour")
        for value in TAIL_COLOR_MODES:
            action = tail_color.addAction(value)
            action.setCheckable(True)
            action.setChecked(value == self._tail_color_mode)
            action.setToolTip(
                "Keep overlay-channel identity in the tail" if value == "Channel"
                else "Colour oldest-to-newest by time; heads retain channel identity")
            action.triggered.connect(
                lambda _c=False, v=value: self._set_tail_color_mode(v))

        clear_selection = menu.addAction("Clear selected track")
        clear_selection.setEnabled(self._selected_track is not None)
        clear_selection.triggered.connect(self._clear_selected_track)

        export_movie = menu.addAction("Export movie as TIFF…")
        export_movie.setToolTip(
            "Capture deterministic frames over the complete time axis into a "
            "multi-page RGB TIFF; capture advances through the event loop.")
        export_movie.setEnabled(
            self._time_row.axis() is not None and self._movie_export is None)
        export_movie.triggered.connect(self._show_movie_export)

        menu.addSeparator()
        zoom = menu.addAction("Zoom to live tracks")
        zoom.setToolTip(
            "Fit the view to the tracks currently on screen. A MINFLUX track is "
            "a few hundred nm inside a field of micrometres, so at full extent "
            "it is a few pixels wide.")
        zoom.triggered.connect(self._zoom_to_live)
        menu.addAction("Reset View").triggered.connect(self._reset_view)
        sender = self.sender()
        host = sender if isinstance(sender, QWidget) else self._plot
        menu.exec(host.mapToGlobal(pos))

    def _set_backdrop_level(self, name: str) -> None:
        if name in BACKDROP_LEVELS:
            self._backdrop_level = name
            self._draw()

    def _set_black_background(self, on: bool) -> None:
        self._black_bg = bool(on)
        self._apply_background()
        self._draw()

    def _set_show_structure_tracks(self, on: bool) -> None:
        self._show_structure_tracks = bool(on)
        self._draw()

    def _set_head_size(self, value: int) -> None:
        self._head_size = int(value)
        self._draw()

    def _show_movie_export(self) -> None:
        axis = self._time_row.axis()
        window = self._time_row.range()
        if axis is None or window is None or self._movie_export is not None:
            return
        row_state = self._time_row.state()
        step = float(row_state.get("step") or axis.step)
        width = max(float(window[1] - window[0]), axis.step)
        available = max(axis.span - width, 0.0)
        default_frames = max(2, min(300, int(np.ceil(available / max(step, 1e-15))) + 1))
        dialog = TrackingMovieDialog(
            default_frames, float(row_state.get("rate_hz", 10.0)), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        frames, fps = dialog.values()
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export tracking movie",
            f"{self._dataset().name if self._dataset() is not None else 'tracking'}.tif",
            "TIFF stack (*.tif *.tiff)",
        )
        if not path:
            return
        if Path(path).suffix.lower() not in {".tif", ".tiff"}:
            path += ".tif"
        self._begin_movie_export(Path(path), frame_count=frames, fps=fps)

    def _begin_movie_export(
        self,
        path: Path,
        *,
        frame_count: int,
        fps: float,
    ) -> None:
        if self._movie_export is not None:
            raise RuntimeError("A tracking movie export is already running.")
        axis = self._time_row.axis()
        window = self._time_row.range()
        if axis is None or window is None:
            raise ValueError("A playable time window is required for movie export.")
        from tifffile import TiffWriter

        target = Path(path)
        partial = target.with_name(target.name + ".partial")
        frame_count = max(int(frame_count), 2)
        fps = max(float(fps), 0.1)
        width = min(max(float(window[1] - window[0]), axis.step), axis.span)
        heads = np.linspace(axis.lo + width, axis.hi, frame_count)
        progress = QProgressDialog(
            "Capturing tracking movie…", "Cancel", 0, frame_count, self)
        progress.setWindowTitle("Export tracking movie")
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(lambda: self._finish_movie_export(cancelled=True))
        progress.show()
        self._time_row.stop()
        self._movie_export = {
            "target": target,
            "partial": partial,
            "writer": TiffWriter(partial, bigtiff=True),
            "frame_count": frame_count,
            "fps": fps,
            "heads": heads,
            "width": width,
            "index": 0,
            "saved_state": self._time_row.state(),
            "progress": progress,
            "axis_mode": self._time_row.mode(),
            "projection": self._axis,
        }
        self._movie_timer.start(0)

    def _capture_current_frame(self) -> np.ndarray:
        if self._axis == "3D" and self._3d_view is not None:
            image = self._3d_view.grabFramebuffer()
        else:
            image = self._plot.grab().toImage()
        image = image.convertToFormat(QImage.Format.Format_RGB888)
        height, width = image.height(), image.width()
        pointer = image.bits()
        pointer.setsize(image.bytesPerLine() * height)
        raw = np.frombuffer(pointer, dtype=np.uint8).reshape(
            height, image.bytesPerLine())
        return raw[:, :width * 3].reshape(height, width, 3).copy()

    def _capture_movie_frame(self) -> None:
        export = self._movie_export
        if export is None or self._closing:
            return
        try:
            index = int(export["index"])
            head = float(export["heads"][index])
            axis = self._time_row.axis()
            if axis is None:
                raise RuntimeError("The tracking time axis disappeared during export.")
            self._time_row.set_window(
                max(axis.lo, head - float(export["width"])), head)
            self._draw()
            frame = self._capture_current_frame()
            export["writer"].write(frame, photometric="rgb", metadata=None)
            export["index"] = index + 1
            export["progress"].setValue(index + 1)
            if index + 1 >= int(export["frame_count"]):
                self._finish_movie_export(cancelled=False)
            else:
                self._movie_timer.start(0)
        except Exception as exc:  # noqa: BLE001 - export error belongs in Log
            self._state.log(f"Tracking movie export failed: {exc}", "ERROR")
            self._finish_movie_export(cancelled=True)

    def _finish_movie_export(self, *, cancelled: bool) -> None:
        export = self._movie_export
        if export is None:
            return
        self._movie_export = None
        self._movie_timer.stop()
        try:
            export["writer"].close()
        except Exception:
            cancelled = True
        progress = export.get("progress")
        if progress is not None:
            progress.close()
            progress.deleteLater()
        partial = Path(export["partial"])
        target = Path(export["target"])
        if cancelled:
            partial.unlink(missing_ok=True)
            self._state.log("Tracking movie export cancelled.", "WARN")
        else:
            partial.replace(target)
            sidecar = target.with_suffix(target.suffix + ".json")
            sidecar.write_text(json.dumps({
                "format": "MINFLUX Viewer tracking movie",
                "frames": int(export["frame_count"]),
                "fps": float(export["fps"]),
                "axis_mode": export["axis_mode"],
                "projection": export["projection"],
                "head_times": np.asarray(export["heads"], dtype=float).tolist(),
                "tail_width_axis_units": float(export["width"]),
            }, indent=2), encoding="utf-8")
            self._state.log(
                f"Exported {export['frame_count']} tracking frames to {target}")
        if not self._closing:
            self._time_row.restore_state(export["saved_state"])
            self._draw()

    def _zoom_to_live(self) -> None:
        """Fit to the tracks inside the current window, not to the whole field."""
        window = self._time_row.range()
        if window is None:
            self._reset_view()
            return
        if self._axis == "3D":
            points = []
            for ch in self._comet_channels():
                selected = self._tracks[ch["dataset_idx"]].window(*window)
                if selected.size:
                    points.append(self._tracks[ch["dataset_idx"]].coords[selected.indices])
            self._reset_3d_camera(np.vstack(points) if points else np.empty((0, 3)))
            return
        ci, cj = self._columns()
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for ch in self._comet_channels():
            tracks = self._tracks[ch["dataset_idx"]]
            sel = tracks.window(*window)
            if sel.size == 0:
                continue
            pts = tracks.coords[sel.indices]
            xs.append(pts[:, ci])
            ys.append(pts[:, cj])
        if not xs:
            return
        x, y = np.concatenate(xs), np.concatenate(ys)
        pad = 0.12 * max(float(np.ptp(x)), float(np.ptp(y)), 50.0)
        self._plot.setXRange(float(x.min()) - pad, float(x.max()) + pad, padding=0)
        self._plot.setYRange(float(y.min()) - pad, float(y.max()) + pad, padding=0)
        # The backdrop is rendered to the fitted bounds, so it follows the zoom.
        self._bounds = (float(x.min()) - pad, float(x.max()) + pad,
                        float(y.min()) - pad, float(y.max()) + pad)
        self._draw()

    def _reset_view(self) -> None:
        self._fit_view()
        self._draw()

    def _reset_3d_camera(self, points: np.ndarray) -> None:
        if self._3d_view is None:
            return
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            return
        points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
        if not points.size:
            return
        center = np.mean(points, axis=0)
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        if not np.isfinite(extent) or extent <= 0:
            extent = 100.0
        self._3d_view.opts["center"] = pg.Vector(*center)
        self._3d_view.setCameraPosition(distance=max(extent * 1.5, 100.0))
        self._3d_camera_initialized = True

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
            self._state.set_active(self._idx)
        super().focusInEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        if self._movie_export is not None:
            self._finish_movie_export(cancelled=True)
        self._redraw_timer.stop()
        self._time_row.stop()
        from .background_tasks import retire_background_tasks

        retire_background_tasks(self._track_tasks.values())
        self._track_tasks.clear()
        from .qt_lifecycle import close_plot_widgets
        close_plot_widgets(self._plot)
        super().closeEvent(event)
