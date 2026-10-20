"""Separate, explicitly submitted 3D (XYZ) convolution workflow on the CPU."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.analysis.convolution_3d import (
    MODELS,
    VolumeRequest,
    preflight_volume,
    volume_detections,
    volume_maps,
)
from minflux_viewer.colormaps import colormap_lut
from minflux_viewer.core.computation_snapshot import input_signature
from minflux_viewer.core.inspection_raster import peak_preview
from minflux_viewer.core.roi_crop import display_xyz_filtered
from minflux_viewer.ui.common.background_tasks import (
    BackgroundTask,
    detach_task_signals,
    retire_background_tasks,
    shared_thread_pool,
)
from minflux_viewer.ui.common.dataset_binding import DatasetIndex, window_dataset_key
from minflux_viewer.ui.common.qt_lifecycle import dispose_plot_widgets, qobject_alive
from minflux_viewer.ui.plots.plot_format import apply_spatial_y_direction
from minflux_viewer.ui.rendering.inspection_layer import InspectionLayer, render_appearance
from minflux_viewer.ui.widgets.layer_brightness import LayerBrightness, percentile_levels

_PLANES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}


class Convolution3DWindow(QDialog):
    """True XYZ convolution; presentation changes never trigger another FFT."""

    _idx = DatasetIndex()

    def __init__(self, state, dataset_idx, owner=None):
        super().__init__(None)
        self._state, self._owner = state, owner
        self._idx = dataset_idx
        ds = state.datasets[dataset_idx]
        if ds.prop.num_dim != 3:
            raise ValueError("3D Convolution requires a 3D localization dataset")
        self.setWindowTitle(f"3D Convolution — {ds.name}")
        self.resize(1240, 800)
        self._closed, self._generation = False, 0
        self._task = None
        self._maps = None
        self._maps_key = None
        self._accepted_stamp = None
        self._centers, self._scores = np.empty((0, 3)), np.empty(0)
        self._projections = {}
        self._bc = LayerBrightness()
        self._xyz = display_xyz_filtered(ds)
        self._xyz.flags.writeable = False
        self._suspend = True
        self._build_ui()
        self._data_preview = InspectionLayer(self, self._plot, self._data_item,
                                             render_appearance(owner, state, dataset_idx),
                                             self._data_ready, self._error)
        self._bc.add_layer("Data", get_pixels=lambda: self._data_preview.pixels,
                           apply=lambda lo, hi: self._data_item.setLevels((lo, hi)),
                           auto=lambda: percentile_levels(self._data_preview.pixels, positive_only=True),
                           default=lambda: self._data_preview.levels)
        self._bc.add_layer("Response", get_pixels=lambda: self._shown_response,
                           apply=lambda lo, hi: self._response_item.setLevels((lo, hi)),
                           auto=lambda: (0., 1.), default=lambda: (0., 1.))
        self._shown_response = np.empty((0, 0))
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._refresh_data)
        self._restore()
        self._suspend = False
        self._central_region()
        for signal in (state.dataset_events.filter_changed, state.dataset_events.attributes_changed,
                       state.dataset_events.calibration_changed, state.dataset_events.overlay_transform_changed,
                       state.dataset_events.removed):
            signal.connect(self._source_changed)

    @staticmethod
    def _spin(value, lo=.1, hi=1e9, step=1., suffix=" nm"):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _build_ui(self):
        root = QHBoxLayout(self)
        controls = QWidget()
        form = QFormLayout(controls)
        description = QLabel("True XYZ matched filtering. Localizations up to one template radius "
                             "outside the region are included, so structures at its edge are not cut; "
                             "detections are reported inside the region only.\n"
                             "NPC template assumes rings normal to Z (no tilt search). "
                             "Set a region and exact voxels, then Run.")
        description.setWordWrap(True)
        form.addRow(description)
        self._bounds = []
        for axis in "XYZ":
            row = QHBoxLayout()
            lo, hi = self._spin(0., -1e12, 1e12), self._spin(1., -1e12, 1e12)
            row.addWidget(lo)
            row.addWidget(hi)
            for spin in (lo, hi):
                spin.valueChanged.connect(self._region_changed)
            self._bounds.append((lo, hi))
            form.addRow(f"{axis} min / max:", row)
        for text, callback in (("400 nm region near data centre", self._central_region),
                               ("Use render viewport region", self._render_region),
                               ("Use whole dataset extent", self._whole_region)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            form.addRow(button)
        self._voxels = [self._spin(4., .5, 200., .5) for _ in range(3)]
        for axis, spin in zip("XYZ", self._voxels):
            spin.valueChanged.connect(self._map_changed)
            form.addRow(f"Compute voxel {axis}:", spin)
        self._model = QComboBox()
        for key, label in MODELS.items():
            self._model.addItem(label, key)
        self._model.setCurrentIndex(self._model.findData("npc"))
        self._model.currentIndexChanged.connect(self._map_changed)
        form.addRow("3D template:", self._model)
        self._diameter = self._spin(100., 1., 5000.)
        self._width = self._spin(10., .5, 1000.)
        self._ring_separation = self._spin(50., 1., 5000.)
        for name, spin in (("Diameter:", self._diameter), ("Gaussian σ / shell width σ:", self._width),
                           ("NPC ring separation:", self._ring_separation)):
            spin.valueChanged.connect(self._map_changed)
            form.addRow(name, spin)
        self._zero, self._sqrt = QCheckBox("Zero-mean template"), QCheckBox("√ counts")
        for check in (self._zero, self._sqrt):
            check.setChecked(True)
            check.toggled.connect(self._map_changed)
            form.addRow(check)
        self._voxel_budget = QSpinBox()
        self._voxel_budget.setRange(1, 64000000)
        self._voxel_budget.setValue(8000000)
        self._voxel_budget.setToolTip("Largest grid (region plus edge margin) a run may use. Memory is "
                                      "checked against the background-work budget in Preferences > Data.")
        self._voxel_budget.valueChanged.connect(self._map_changed)
        form.addRow("Voxel limit:", self._voxel_budget)
        self._threshold, self._distance = self._spin(.3, 0., 1., .05, ""), self._spin(100.)
        self._cap = QSpinBox()
        self._cap.setRange(1, 10000)
        self._cap.setValue(500)
        for name, spin in (("Min response:", self._threshold), ("Min 3D separation:", self._distance),
                           ("Max detections:", self._cap)):
            spin.valueChanged.connect(self._threshold_changed)
            form.addRow(name, spin)
        self._run = QPushButton("Run 3D convolution")
        self._run.clicked.connect(self._submit)
        self._cancel = QPushButton("Request stop")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._stop)
        form.addRow(self._run)
        form.addRow(self._cancel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(390)
        root.addWidget(scroll)
        right = QVBoxLayout()
        self._status = QLabel("Select a region and voxel sizes, then Run.")
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        right.addWidget(self._status)
        row = QHBoxLayout()
        self._plane = QComboBox()
        self._plane.addItems(list(_PLANES))
        self._plane.currentTextChanged.connect(self._plane_changed)
        row.addWidget(self._plane)
        self._show_data, self._show_response = QCheckBox("Data"), QCheckBox("Response MIP")
        self._show_data.setChecked(True)
        self._show_response.setChecked(True)
        for check in (self._show_data, self._show_response):
            check.toggled.connect(self._visibility)
            row.addWidget(check)
        self._target = QComboBox()
        self._target.addItems(["Data", "Response"])
        row.addWidget(self._target)
        bc = QPushButton("B/C…")
        bc.clicked.connect(self._open_bc)
        row.addWidget(bc)
        copy = QPushButton("Copy render appearance")
        copy.clicked.connect(self._copy_appearance)
        row.addWidget(copy)
        right.addLayout(row)
        right.addWidget(QLabel("Inspection projections adapt to the monitor. Full XYZ voxels determine detections."))
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._data_item, self._response_item = (pg.ImageItem(axisOrder="row-major") for _ in range(2))
        self._response_item.setLookupTable(colormap_lut("inferno", alpha=False))
        self._response_item.setOpacity(.55)
        self._response_item.setZValue(1)
        self._markers = pg.ScatterPlotItem(size=12, pen=pg.mkPen("c", width=2), brush=None)
        self._markers.setZValue(2)
        for item in (self._data_item, self._response_item, self._markers):
            self._plot.addItem(item)
        right.addWidget(self._plot, 1)
        self._list = QListWidget()
        self._list.setMaximumHeight(140)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        right.addWidget(self._list)
        self._add = QPushButton("Add selected centres to ROI Manager (none selected = all)")
        self._add.setEnabled(False)
        self._add.clicked.connect(self._publish)
        right.addWidget(self._add)
        root.addLayout(right, 1)

    def _request(self):
        return VolumeRequest(tuple(tuple(s.value() for s in pair) for pair in self._bounds),
                             tuple(s.value() for s in self._voxels), self._model.currentData(),
                             self._diameter.value(), self._width.value(), self._ring_separation.value(),
                             self._zero.isChecked(), self._sqrt.isChecked(),
                             max_voxels=self._voxel_budget.value())

    def _stamp(self):
        key = window_dataset_key(self)
        ds = self._state.dataset_for(key)
        if ds is None:
            return None
        return (self._state.data_revision(key), input_signature(ds), repr(ds.state.get("overlay_transform")),
                repr(ds.state.get("render_transform_2d")))

    def _invalidate(self, *, maps):
        if self._suspend or self._closed:
            return
        self._generation += 1
        if self._task is not None:
            self._task.cancel()
        self._centers, self._scores = np.empty((0, 3)), np.empty(0)
        self._markers.clear()
        self._list.clear()
        self._add.setEnabled(False)
        self._accepted_stamp = None
        if maps:
            self._maps, self._maps_key = None, None
            self._projections.clear()
            self._response_item.clear()
        self._status.setText("Settings changed. Press Run to compute; display changes do not affect the scientific grid.")

    def _map_changed(self, *_args):
        if not self._suspend:
            self._diameter.setEnabled(self._model.currentData() != "gaussian")
            self._ring_separation.setEnabled(self._model.currentData() == "npc")
        self._invalidate(maps=True)

    def _region_changed(self, *_args):
        self._map_changed()
        if not self._suspend and not self._closed:
            self._preview_timer.start()

    def _threshold_changed(self, *_args):
        self._invalidate(maps=False)

    def _set_bounds(self, bounds):
        self._suspend = True
        for widgets, interval in zip(self._bounds, bounds):
            # Controls show 0.01 nm: round region buttons outward so a whole
            # dataset/viewport action cannot drop boundary rows by rounding.
            rounded = (np.floor(interval[0]*100.)/100., np.ceil(interval[1]*100.)/100.)
            for spin, value in zip(widgets, rounded):
                spin.setValue(value)
        self._suspend = False
        self._region_changed()

    def _central_region(self):
        if len(self._xyz):
            lo, hi = self._xyz.min(axis=0), self._xyz.max(axis=0)
            midpoint = (lo+hi)/2
            nearest = self._xyz[np.argmin(np.sum((self._xyz-midpoint)**2, axis=1))]
            half = np.minimum(np.maximum((hi-lo)/2, 2.), 200.)
            # Small axes retain their complete extent. Moving a small whole-data
            # box onto a localization instead of its midpoint clips that data.
            center = np.where(hi-lo <= 400., midpoint, np.clip(nearest, lo+half, hi-half))
            lower = np.where(hi-lo <= 400., np.minimum(center-half, lo), center-half)
            upper = np.where(hi-lo <= 400., np.maximum(center+half, hi), center+half)
            self._set_bounds(np.column_stack((lower, upper)))

    def _whole_region(self):
        if len(self._xyz):
            lo, hi = self._xyz.min(axis=0), self._xyz.max(axis=0)
            self._set_bounds(np.column_stack((lo, np.maximum(hi, lo+1.))))

    def _render_region(self):
        if self._owner is not None and self._idx is not None:
            key = window_dataset_key(self)
            for registry in (getattr(self._owner, "_render_windows", {}),
                             getattr(self._owner, "_advanced_render_windows", {})):
                for window in registry.values():
                    bounds = window.inspection_region(key)
                    if bounds is not None:
                        self._set_bounds(bounds)
                        return
        self._error("Open a render view of this dataset first, or edit the explicit XYZ bounds.")

    def _refresh_data(self):
        if self._closed:
            return
        bounds = np.asarray([[s.value() for s in pair] for pair in self._bounds])
        selected = self._xyz[((self._xyz >= bounds[:, 0]) & (self._xyz <= bounds[:, 1])).all(axis=1)]
        h, v, _ = _PLANES[self._plane.currentText()]
        xy = np.ascontiguousarray(selected[:, (h, v)])
        xy.flags.writeable = False
        self._data_preview.set_points(xy)
        self._plot.setLabel("bottom", f"{'XYZ'[h]} (nm)")
        self._plot.setLabel("left", f"{'XYZ'[v]} (nm)")
        apply_spatial_y_direction(self._plot, vertical_coordinate="XYZ"[v], prefs=self._state.prefs,
                                  preference_key="render_xy_origin")

    def _plane_changed(self, *_args):
        if not self._suspend:
            self._refresh_data()
            self._draw_response()

    def _data_ready(self):
        self._data_item.setLevels(self._bc.effective_levels("Data"))
        self._visibility()
        self._bc.refresh_if_open()

    def _visibility(self):
        self._data_item.setVisible(self._show_data.isChecked())
        self._response_item.setVisible(self._show_response.isChecked())

    def _copy_appearance(self):
        if self._idx is not None:
            self._data_preview.set_appearance(render_appearance(self._owner, self._state, self._idx))
            self._bc.reset_layer("Data")

    def _open_bc(self):
        self._bc.set_target(self._target.currentText())
        self._bc.open(self)

    def _submit(self):
        if self._closed or self._task is not None or self._idx is None:
            return
        self._accepted_stamp = None
        self._centers, self._scores = np.empty((0, 3)), np.empty(0)
        self._list.clear()
        self._markers.clear()
        request = self._request()
        stamp, generation, xyz = self._stamp(), self._generation, self._xyz
        map_key = (stamp, request)
        cached = self._maps if self._maps_key == map_key else None
        try:
            estimate = preflight_volume(request, len(xyz))
        except (ValueError, MemoryError) as exc:
            self._error(str(exc))
            return
        threshold, distance, cap = self._threshold.value(), self._distance.value(), self._cap.value()

        def work(report):
            maps = cached if cached is not None else volume_maps(xyz, request, report=report)
            centers, scores = volume_detections(maps, min_response=threshold, min_distance_nm=distance,
                                                max_detections=cap, report=report)
            if cached is None:
                maps["projections"] = {name: peak_preview(maps["response"].max(axis=axes[2]))
                                       for name, axes in _PLANES.items()}
            report("3D convolution ready")
            return maps, centers, scores

        task = BackgroundTask(work, description="3D convolution", category="analysis",
                              estimated_bytes=0 if cached is not None else estimate.host_bytes)
        self._task = task
        self._add.setEnabled(False)
        self._run.setEnabled(False)
        self._cancel.setEnabled(True)
        self._status.setText(f"Computing exact XYZ {estimate.region_shape} (with edge margin "
                             f"{estimate.image_shape}); memory estimate "
                             f"{estimate.host_bytes / 1024**2:,.1f} MiB. Preview is display-only.")

        def current():
            return (not self._closed and qobject_alive(self) and self._task is task and not task.is_cancelled
                    and generation == self._generation and stamp == self._stamp())

        def done(result):
            if not current():
                return
            self._maps, self._centers, self._scores = result
            self._maps_key, self._accepted_stamp = map_key, stamp
            self._projections = self._maps["projections"]
            self._draw_response()
            self._list.addItems([f"{i+1}: X={x:.1f}, Y={y:.1f}, Z={z:.1f} nm · r={score:.3f}"
                                 for i, ((x, y, z), score) in enumerate(zip(self._centers, self._scores))])
            self._status.setText(f"{len(self._centers)} 3D detections · {self._maps['selected_count']:,} localizations · "
                                 f"XYZ grid {self._maps['shape']} · CPU float64. "
                                 "Threshold changes reuse this volume after Run.")
            self._status.setToolTip(json.dumps(self._maps["provenance"], indent=2))
            self._add.setEnabled(bool(len(self._centers)))

        def failed(message):
            if current():
                self._error(message)

        def finished():
            detach_task_signals(task)
            if self._closed:
                return
            if task.is_cancelled:
                self._status.setText("Request stopped or invalidated; no detections published. Press Run for current settings.")
            self._task = None
            self._run.setEnabled(self._idx is not None)
            self._cancel.setEnabled(False)

        task.signals.done.connect(done)
        task.signals.failed.connect(failed)
        task.signals.finished.connect(finished)
        shared_thread_pool("convolution-segmentation").start(task)

    def _draw_response(self):
        plane = self._plane.currentText()
        if plane in self._projections and self._maps is not None:
            self._shown_response = self._projections[plane]
            h, v, _ = _PLANES[plane]
            edges = self._maps["edges"]
            self._response_item.setImage(self._shown_response.T, levels=self._bc.effective_levels("Response"))
            self._response_item.setRect(QRectF(edges[h][0], edges[v][0],
                                              edges[h][-1]-edges[h][0], edges[v][-1]-edges[v][0]))
        if len(self._centers):
            h, v, _ = _PLANES[plane]
            self._markers.setData(x=self._centers[:, h], y=self._centers[:, v])

    def _error(self, message):
        self._accepted_stamp = None
        self._status.setText(" ".join(str(message).split())[:600])
        self._status.setToolTip(str(message))
        self._add.setEnabled(False)

    def _stop(self):
        if self._task is not None:
            self._accepted_stamp = None
            self._task.cancel()
            self._add.setEnabled(False)
            self._status.setText("Stop requested; waiting for the worker's cancellation checkpoint.")

    def _source_changed(self, event):
        if self._closed or event.key != window_dataset_key(self):
            return
        self._invalidate(maps=True)
        if self._idx is None:
            self._xyz = np.empty((0, 3))
            self._run.setEnabled(False)
            self._error("Source dataset is closed.")
        else:
            self._xyz = display_xyz_filtered(self._state.datasets[self._idx])
            self._xyz.flags.writeable = False
        self._refresh_data()

    def _publish(self):
        if (self._closed or self._task is not None or self._accepted_stamp != self._stamp()
                or self._accepted_stamp is None or self._owner is None or self._idx is None):
            return
        rows = sorted(self._list.row(item) for item in self._list.selectedItems()) or list(range(len(self._centers)))
        request = asdict(self._maps["request"])
        provenance = self._maps["provenance"]
        self._owner.add_point_rois_3d(self._idx, self._centers[rows], name_prefix="Conv3D", source="convolution_3d",
                                      log_message=f"3D convolution: added {len(rows)} XYZ point ROI(s); "
                                      f"request={json.dumps(request, sort_keys=True)}; "
                                      f"threshold={self._threshold.value()}, separation_nm={self._distance.value()}, "
                                      f"cap={self._cap.value()}; FFT={json.dumps(provenance, sort_keys=True)}")

    def _restore(self):
        cfg = self._state.prefs.get("convolution_3d", {})
        if "voxel_limit" in cfg:
            self._voxel_budget.setValue(int(cfg["voxel_limit"]))
        self._model.setCurrentIndex(max(0, self._model.findData(cfg.get("model", "npc"))))
        for key, spin in (("diameter_nm", self._diameter), ("width_nm", self._width),
                          ("ring_separation_nm", self._ring_separation), ("min_response", self._threshold),
                          ("min_distance_nm", self._distance), ("max_detections", self._cap)):
            if key in cfg:
                spin.setValue(cfg[key])
        for axis, spin in zip("xyz", self._voxels):
            if f"voxel_{axis}_nm" in cfg:
                spin.setValue(cfg[f"voxel_{axis}_nm"])
        for key, check in (("zero_mean", self._zero), ("sqrt_counts", self._sqrt),
                           ("show_data", self._show_data), ("show_response", self._show_response)):
            if key in cfg:
                check.setChecked(bool(cfg[key]))

    def closeEvent(self, event):  # noqa: N802
        from minflux_viewer.ui.common.qt_lifecycle import cleanup_callback

        self._closed = True
        self._generation += 1
        self._preview_timer.stop()
        self._data_preview.close()
        if self._task is not None:
            retire_background_tasks([self._task])
            self._task = None
        self._state.prefs["convolution_3d"] = dict(voxel_limit=self._voxel_budget.value())
        self._state.prefs["convolution_3d"].update(
            model=self._model.currentData(), diameter_nm=self._diameter.value(), width_nm=self._width.value(),
            ring_separation_nm=self._ring_separation.value(), zero_mean=self._zero.isChecked(),
            sqrt_counts=self._sqrt.isChecked(), min_response=self._threshold.value(),
            min_distance_nm=self._distance.value(), max_detections=self._cap.value(),
            show_data=self._show_data.isChecked(), show_response=self._show_response.isChecked(),
            **{f"voxel_{axis}_nm": spin.value() for axis, spin in zip("xyz", self._voxels)})
        cleanup_callback(self._state.save_prefs, context="save 3D convolution preferences")
        cleanup_callback(self._bc.close, context="close 3D convolution brightness dialog")
        dispose_plot_widgets(self)
        super().closeEvent(event)
