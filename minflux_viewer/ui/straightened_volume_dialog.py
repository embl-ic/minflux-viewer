"""
Experimental centerline-straightened projection viewer.

The window is intentionally modest: it consumes selected 3-D point/polyline ROIs
as a skeleton and shows count projections from a straightened ``(s, u, v)``
volume around that skeleton.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis.straightened_volume import StraightenedVolumeResult, straightened_count_volume
from ..core.roi import OPEN_LINE_TYPES, RoiRecord
from .plot_format import plot_widget

_GL_MAX_POINTS = 80_000
_RAW_COLOR = (0.7, 0.7, 0.7, 0.22)
_SKELETON_COLOR = (1.0, 0.85, 0.1, 1.0)
_CENTERLINE_COLOR = (0.1, 0.9, 1.0, 1.0)


class StraightenedVolumeAlongSkeletonWindow(QDialog):
    TAG = "straightened_volume_skeleton"

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self._xyz = self._load_xyz()
        self._skeleton_points = np.empty((0, 3), dtype=float)
        self._result: StraightenedVolumeResult | None = None
        self._suspend = False
        self._gl_view = None
        self._gl_items: dict[str, object] = {}
        self._gl_center = np.zeros(3, dtype=float)

        ds = self._dataset()
        self.setWindowTitle(f"Straightened Volume along Skeleton - {ds.name if ds else '?'}")
        self.resize(1280, 820)

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(150)
        self._recompute_timer.timeout.connect(self._recompute)

        self._build_ui()
        self._use_selected_rois(show_message=False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_view(), 1)

    def _build_controls(self) -> QWidget:
        left = QVBoxLayout()
        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>")
        title.setWordWrap(True)
        left.addWidget(title)
        self._data_label = QLabel(f"{self._xyz.shape[0]:,} localizations")
        self._data_label.setStyleSheet("color:#777;")
        left.addWidget(self._data_label)

        self._skeleton_label = QLabel("No skeleton ROI selected")
        self._skeleton_label.setWordWrap(True)
        left.addWidget(self._skeleton_label)

        row = QHBoxLayout()
        use_btn = QPushButton("Use Selected ROIs")
        use_btn.clicked.connect(lambda: self._use_selected_rois(show_message=True))
        row.addWidget(use_btn)
        open_btn = QPushButton("Open ROI Tools")
        open_btn.clicked.connect(self._open_roi_tools)
        row.addWidget(open_btn)
        left.addLayout(row)

        left.addWidget(self._hline())
        form = QFormLayout()
        self._step_spin = self._spin(1.0, 2000.0, 20.0, 1, 5.0, " nm")
        self._step_spin.setToolTip("Arc-length spacing between resampled skeleton planes.")
        self._step_spin.valueChanged.connect(self._schedule_recompute)
        form.addRow("Skeleton step:", self._step_spin)
        self._half_width_spin = self._spin(5.0, 10000.0, 250.0, 1, 10.0, " nm")
        self._half_width_spin.setToolTip("Half-width of the sampled cross-section square.")
        self._half_width_spin.valueChanged.connect(self._schedule_recompute)
        form.addRow("Half width:", self._half_width_spin)
        self._pixel_spin = self._spin(1.0, 1000.0, 10.0, 1, 1.0, " nm")
        self._pixel_spin.setToolTip("Bin size in the local cross-section coordinates.")
        self._pixel_spin.valueChanged.connect(self._schedule_recompute)
        form.addRow("Cross-section pixel:", self._pixel_spin)

        self._projection_combo = QComboBox()
        self._projection_combo.addItem("Sum counts", "sum")
        self._projection_combo.addItem("Max counts", "max")
        self._projection_combo.currentIndexChanged.connect(self._schedule_recompute)
        form.addRow("Projection:", self._projection_combo)

        self._frame_combo = QComboBox()
        self._frame_combo.addItem("Minimum twist", "minimum_twist")
        self._frame_combo.addItem("Global reference", "global_reference")
        self._frame_combo.currentIndexChanged.connect(self._schedule_recompute)
        form.addRow("Frame:", self._frame_combo)

        self._reference_combo = QComboBox()
        self._reference_combo.addItem("+Z", (0.0, 0.0, 1.0))
        self._reference_combo.addItem("+Y", (0.0, 1.0, 0.0))
        self._reference_combo.addItem("+X", (1.0, 0.0, 0.0))
        self._reference_combo.currentIndexChanged.connect(self._schedule_recompute)
        form.addRow("Reference:", self._reference_combo)

        self._angle_spin = self._spin(-180.0, 180.0, 0.0, 1, 5.0, " deg")
        self._angle_spin.setToolTip("Rotate the local cross-section axes around the skeleton tangent.")
        self._angle_spin.valueChanged.connect(self._schedule_recompute)
        form.addRow("Angle offset:", self._angle_spin)
        left.addLayout(form)

        recompute_btn = QPushButton("Recompute")
        recompute_btn.clicked.connect(self._recompute)
        left.addWidget(recompute_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color:#1a8;")
        left.addWidget(self._status_label)
        left.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        left.addWidget(close_btn)

        panel = QWidget()
        panel.setLayout(left)
        panel.setFixedWidth(300)
        return panel

    def _build_view(self) -> QWidget:
        grid = QGridLayout()
        grid.setSpacing(5)
        self._plots: dict[str, dict] = {}
        self._make_plot(grid, "su", "Skeleton length (nm)", "U (nm)", "S-U projection", 0, 0)
        self._make_plot(grid, "sv", "Skeleton length (nm)", "V (nm)", "S-V projection", 0, 1)
        self._make_plot(grid, "uv", "U (nm)", "V (nm)", "Accumulated cross-section", 1, 0)
        self._build_3d_preview(grid, 1, 1)
        panel = QWidget()
        panel.setLayout(grid)
        return panel

    def _make_plot(self, grid: QGridLayout, key: str, xlabel: str, ylabel: str,
                   title: str, row: int, col: int) -> None:
        plot = plot_widget(background="k")
        plot.setAspectLocked(False)
        plot.setLabel("bottom", xlabel)
        plot.setLabel("left", ylabel)
        plot.setTitle(title)
        plot.setMenuEnabled(False)
        img = pg.ImageItem()
        try:
            img.setLookupTable(pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        plot.addItem(img)
        grid.addWidget(plot, row, col)
        self._plots[key] = {"plot": plot, "img": img}

    def _build_3d_preview(self, grid: QGridLayout, row: int, col: int) -> None:
        try:
            import pyqtgraph.opengl as gl

            self._gl_view = gl.GLViewWidget()
            self._gl_view.setBackgroundColor(pg.mkColor(10, 10, 14))
            grid.addWidget(self._gl_view, row, col)
        except Exception as exc:
            label = QLabel(f"3-D preview unavailable\n({exc})")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color:#888; border:1px solid #333;")
            label.setWordWrap(True)
            grid.addWidget(label, row, col)

    @staticmethod
    def _spin(lo: float, hi: float, value: float, decimals: int, step: float,
              suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line

    @staticmethod
    def _oriented(img: np.ndarray) -> np.ndarray:
        try:
            return img.T if pg.getConfigOption("imageAxisOrder") == "row-major" else img
        except Exception:
            return img.T

    # ------------------------------------------------------------------ data
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_xyz(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty((0, 3), dtype=float)
        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            return np.empty((0, 3), dtype=float)
        if loc.ndim != 2 or loc.shape[1] < 3:
            return np.empty((0, 3), dtype=float)
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] == loc.shape[0]:
            loc = loc[mask]
        xyz = loc[:, :3]
        return xyz[np.all(np.isfinite(xyz), axis=1)]

    def _open_roi_tools(self) -> None:
        if self._owner is None:
            return
        try:
            self._owner._show_render(self._idx)
            self._owner._show_roi_manager()
        except Exception:
            pass

    def _use_selected_rois(self, *, show_message: bool) -> None:
        try:
            points, label = self._selected_roi_points()
        except Exception as exc:
            self._skeleton_points = np.empty((0, 3), dtype=float)
            self._skeleton_label.setText(str(exc))
            self._clear_outputs()
            if show_message:
                QMessageBox.information(self, "Straightened Volume along Skeleton", str(exc))
            return
        self._skeleton_points = points
        self._skeleton_label.setText(label)
        self._recompute()

    def _selected_roi_points(self) -> tuple[np.ndarray, str]:
        selected = self._state.rois.selected_records()
        if not selected:
            raise ValueError("Select 3-D point ROIs or one 3-D polyline ROI in the ROI Manager.")
        ordered_ids = {r.id for r in selected}
        records = [r for r in self._state.rois.records if r.id in ordered_ids]
        pts: list[list[float]] = []
        used_names: list[str] = []
        skipped = 0
        for record in records:
            if not self._record_belongs_to_dataset(record):
                skipped += 1
                continue
            rpts = self._points_from_record(record)
            if rpts.size == 0:
                skipped += 1
                continue
            pts.extend(rpts.tolist())
            used_names.append(record.name or record.type)
        if len(pts) < 2:
            msg = "Selected ROI(s) did not provide at least two 3-D skeleton points."
            if skipped:
                msg += f" Skipped {skipped} ROI(s)."
            raise ValueError(msg)
        arr = np.asarray(pts, dtype=float)
        text = f"{arr.shape[0]} skeleton points from {len(used_names)} ROI(s)"
        if skipped:
            text += f"; skipped {skipped}"
        return arr, text

    def _record_belongs_to_dataset(self, record: RoiRecord) -> bool:
        ctx = record.context if isinstance(record.context, dict) else {}
        rec_idx = ctx.get("dataset_idx")
        return rec_idx is None or rec_idx == self._idx

    @staticmethod
    def _points_from_record(record: RoiRecord) -> np.ndarray:
        geom = record.geometry or {}
        if record.type == "point":
            pt = np.asarray(geom.get("point", []), dtype=float).reshape(-1)
            if pt.size >= 3 and np.all(np.isfinite(pt[:3])):
                return pt[:3].reshape(1, 3)
            return np.empty((0, 3), dtype=float)
        if record.type in OPEN_LINE_TYPES or record.type == "line":
            pts = np.asarray(geom.get("points", []), dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 3:
                pts = pts[:, :3]
                return pts[np.all(np.isfinite(pts), axis=1)]
        return np.empty((0, 3), dtype=float)

    # ---------------------------------------------------------------- compute
    def _schedule_recompute(self) -> None:
        if not self._suspend:
            self._recompute_timer.start()

    def _recompute(self) -> None:
        if self._skeleton_points.shape[0] < 2:
            self._clear_outputs()
            return
        if self._xyz.shape[0] == 0:
            self._status_label.setText("No finite 3-D localizations available.")
            self._clear_outputs()
            return
        try:
            result = straightened_count_volume(
                self._xyz,
                self._skeleton_points,
                step_nm=float(self._step_spin.value()),
                half_width_nm=float(self._half_width_spin.value()),
                pixel_nm=float(self._pixel_spin.value()),
                frame_mode=self._frame_combo.currentData(),
                reference_vector=self._reference_combo.currentData(),
                angle_offset_deg=float(self._angle_spin.value()),
                projection_mode=self._projection_combo.currentData(),
            )
        except Exception as exc:
            self._status_label.setText(str(exc))
            self._clear_outputs(clear_status=False)
            return
        self._result = result
        self._draw_result(result)
        self._update_gl(result)
        self._status_label.setText(
            f"Straightened volume: {result.volume.shape[0]} x {result.volume.shape[1]} x "
            f"{result.volume.shape[2]} bins; {result.n_used:,}/{result.n_input:,} "
            "localizations inside the sampled cross-section.")

    def _clear_outputs(self, *, clear_status: bool = True) -> None:
        self._result = None
        for entry in self._plots.values():
            entry["img"].clear()
        if clear_status:
            self._status_label.setText("")
        self._clear_gl()

    def _draw_result(self, result: StraightenedVolumeResult) -> None:
        self._draw_image("su", result.projection_su, result.s_edges_nm, result.u_edges_nm)
        self._draw_image("sv", result.projection_sv, result.s_edges_nm, result.v_edges_nm)
        self._draw_image("uv", result.projection_uv, result.u_edges_nm, result.v_edges_nm)
        for entry in self._plots.values():
            entry["plot"].getViewBox().autoRange(padding=0.02)

    def _draw_image(self, key: str, img_hv: np.ndarray,
                    hedge: np.ndarray, vedge: np.ndarray) -> None:
        item = self._plots[key]["img"]
        vmax = float(np.nanmax(img_hv)) if img_hv.size else 0.0
        levels = (0.0, max(vmax, 1.0))
        item.setImage(self._oriented(np.ascontiguousarray(img_hv)), levels=levels)
        item.setRect(QRectF(float(hedge[0]), float(vedge[0]),
                            float(hedge[-1] - hedge[0]), float(vedge[-1] - vedge[0])))

    # ------------------------------------------------------------------ 3-D
    def _gl_ok(self) -> bool:
        return self._gl_view is not None

    def _clear_gl(self) -> None:
        if not self._gl_ok():
            return
        for item in list(self._gl_items.values()):
            try:
                self._gl_view.removeItem(item)
            except Exception:
                pass
        self._gl_items.clear()

    def _update_gl(self, result: StraightenedVolumeResult) -> None:
        if not self._gl_ok():
            return
        import pyqtgraph.opengl as gl

        pts = self._xyz
        if pts.shape[0] > _GL_MAX_POINTS:
            sel = np.random.default_rng(0).choice(pts.shape[0], _GL_MAX_POINTS, replace=False)
            pts = pts[sel]
        bounds = np.vstack([result.centerline_nm, pts]) if pts.size else result.centerline_nm
        self._gl_center = 0.5 * (np.nanmin(bounds, axis=0) + np.nanmax(bounds, axis=0))
        raw_local = pts - self._gl_center
        skel_local = self._skeleton_points - self._gl_center
        center_local = result.centerline_nm - self._gl_center

        raw = self._gl_items.get("raw")
        if raw is None:
            raw = gl.GLScatterPlotItem(pos=raw_local, color=_RAW_COLOR, size=2.0, pxMode=True)
            self._gl_view.addItem(raw)
            self._gl_items["raw"] = raw
        else:
            raw.setData(pos=raw_local, color=_RAW_COLOR, size=2.0)

        skel = self._gl_items.get("skeleton")
        if skel is None:
            skel = gl.GLLinePlotItem(pos=skel_local, color=_SKELETON_COLOR,
                                     width=3.0, mode="line_strip", antialias=True)
            self._gl_view.addItem(skel)
            self._gl_items["skeleton"] = skel
        else:
            skel.setData(pos=skel_local, color=_SKELETON_COLOR)

        ctr = self._gl_items.get("centerline")
        if ctr is None:
            ctr = gl.GLLinePlotItem(pos=center_local, color=_CENTERLINE_COLOR,
                                    width=1.5, mode="line_strip", antialias=True)
            self._gl_view.addItem(ctr)
            self._gl_items["centerline"] = ctr
        else:
            ctr.setData(pos=center_local, color=_CENTERLINE_COLOR)

        span = float(np.nanmax(np.ptp(bounds, axis=0))) if bounds.size else 1.0
        self._gl_view.setCameraPosition(distance=max(span * 1.8, 1.0))
