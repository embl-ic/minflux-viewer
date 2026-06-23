"""
minflux_viewer.ui.particle_average_dialog
==========================================
**Particle averaging** tool (Analyze › Segmentation › Particle Average…).

Takes the **active dataset** and the **box ROIs in the ROI Manager** as input.
The workflow:

1. **Survey** the ROI Manager for rectangle (box) ROIs — the detected particles.
   This is the intermediate review step: add / edit / delete box ROIs in the
   render view or ROI Manager, then **Refresh** here; multi-select rows to use a
   subset (none selected = all).
2. Choose a **mode**:
   * *template-provided* — align every particle (2-D, XY) to a geometry template;
   * *template-free* — reference-free iterative alignment to a self-consistent
     average.
3. **Run** — each box ROI's localizations are cropped, centred (Z translationally),
   aligned, and pooled into an averaged super-particle shown in a new render +
   scatter (3-D-ready).

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis import conv_segmentation as cs
from ..analysis import particle_average as pa
from ..core import roi_crop
from ..core.roi_selection import rectangle_bounds, rectangle_mask

_MIN_LOCS_PER_PARTICLE = 5
_TEMPLATE_MODELS = ("ring", "disk", "gaussian")


class ParticleAverageWindow(QDialog):
    """Modeless particle-averaging tool for the active dataset + box ROIs."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Particle Average")
        self.resize(380, 640)
        self._template_spins: dict[str, QDoubleSpinBox] = {}

        self._build_ui()
        self._refresh_rois()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>")
        title.setWordWrap(True)
        root.addWidget(title)

        # --- box ROI survey ---------------------------------------------------
        roi_group = QGroupBox("Box ROIs (detections)")
        rv = QVBoxLayout(roi_group)
        self._roi_info = QLabel("")
        self._roi_info.setStyleSheet("color:#888;")
        self._roi_info.setWordWrap(True)
        rv.addWidget(self._roi_info)
        self._roi_list = QListWidget()
        self._roi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        rv.addWidget(self._roi_list, 1)
        hint = QLabel("Add / edit box ROIs in the render view or ROI Manager, then "
                      "Refresh. Select rows to use a subset (none = all).")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        rv.addWidget(hint)
        refresh = QPushButton("Refresh from ROI Manager")
        refresh.clicked.connect(self._refresh_rois)
        rv.addWidget(refresh)
        root.addWidget(roi_group, 1)

        # --- mode -------------------------------------------------------------
        mode_group = QGroupBox("Alignment mode")
        mv = QVBoxLayout(mode_group)
        self._free_radio = QRadioButton("Template-free (reference-free)")
        self._free_radio.setChecked(True)
        self._free_radio.toggled.connect(self._on_mode_changed)
        self._template_radio = QRadioButton("Template-provided (geometry)")
        mv.addWidget(self._free_radio)
        mv.addWidget(self._template_radio)

        self._template_box = QWidget()
        tb = QFormLayout(self._template_box)
        tb.setContentsMargins(16, 0, 0, 0)
        self._geom_combo = QComboBox()
        for key in _TEMPLATE_MODELS:
            self._geom_combo.addItem(cs.GEOMETRY_MODELS[key].label, key)
        self._geom_combo.currentIndexChanged.connect(lambda *_: self._rebuild_template_params())
        tb.addRow("Geometry:", self._geom_combo)
        self._template_param_form = QFormLayout()
        tb.addRow(self._template_param_form)
        self._template_box.setVisible(False)
        mv.addWidget(self._template_box)
        root.addWidget(mode_group)

        # --- parameters -------------------------------------------------------
        param_group = QGroupBox("Parameters")
        pf = QFormLayout(param_group)
        self._box_size = self._spin(8.0, 5000.0, 150.0, 0, 5.0, " nm")
        self._box_size.setToolTip("Side of the square analysis window each particle is rendered into.")
        pf.addRow("Average box:", self._box_size)
        self._pixel = self._spin(0.5, 50.0, 3.0, 1, 0.5, " nm")
        self._pixel.setToolTip("Pixel size of the alignment render.")
        pf.addRow("Pixel size:", self._pixel)
        self._angles = QSpinBox()
        self._angles.setRange(4, 360)
        self._angles.setValue(36)
        self._angles.setToolTip("Number of rotation steps in the alignment search (360/N° resolution).")
        pf.addRow("Rotation steps:", self._angles)
        self._iters = QSpinBox()
        self._iters.setRange(1, 30)
        self._iters.setValue(5)
        self._iters.setToolTip("Template-free refinement iterations.")
        pf.addRow("Iterations:", self._iters)
        root.addWidget(param_group)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#1a8;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Run Average")
        self._run_btn.clicked.connect(self._run)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self._run_btn)
        btns.addWidget(close_btn)
        root.addLayout(btns)

        self._rebuild_template_params()

    @staticmethod
    def _spin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _on_mode_changed(self) -> None:
        self._template_box.setVisible(self._template_radio.isChecked())
        self._iters.setEnabled(self._free_radio.isChecked())

    def _rebuild_template_params(self) -> None:
        while self._template_param_form.rowCount():
            self._template_param_form.removeRow(0)
        self._template_spins.clear()
        model = cs.GEOMETRY_MODELS[self._geom_combo.currentData()]
        for ps in model.params:
            spin = self._spin(ps.lo, ps.hi, ps.default, ps.decimals, ps.step, ps.suffix)
            self._template_param_form.addRow(ps.label + ":", spin)
            self._template_spins[ps.name] = spin

    # ----------------------------------------------------------- data / ROIs
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _box_records(self) -> list:
        return [r for r in self._state.rois.records if r.type == "rectangle"]

    def _refresh_rois(self) -> None:
        self._roi_list.clear()
        recs = self._box_records()
        sizes = []
        for rec in recs:
            b = rectangle_bounds(rec)
            side = max(b[1] - b[0], b[3] - b[2]) if b else 0.0
            sizes.append(side)
            item = QListWidgetItem(f"{rec.name}   {side:.0f} nm")
            item.setData(Qt.ItemDataRole.UserRole, rec.id)
            self._roi_list.addItem(item)
        self._roi_info.setText(f"{len(recs)} box ROI(s) in the ROI Manager.")
        if sizes:                       # default the analysis box to the median ROI size
            self._box_size.setValue(float(np.median(sizes)))
        self._run_btn.setEnabled(len(recs) >= 2)

    def _selected_records(self) -> list:
        ids = {it.data(Qt.ItemDataRole.UserRole) for it in self._roi_list.selectedItems()}
        recs = self._box_records()
        chosen = [r for r in recs if r.id in ids]
        return chosen if chosen else recs

    def _extract_particles(self):
        ds = self._dataset()
        coords = roi_crop.display_coords(ds)         # (N, 3) display nm
        particles = []
        for rec in self._selected_records():
            mask = rectangle_mask(coords[:, 0], coords[:, 1], rec)
            p = coords[mask]
            if p.shape[0] >= _MIN_LOCS_PER_PARTICLE:
                particles.append(p)
        return particles

    # ----------------------------------------------------------- run
    def _template_image(self, box_nm, pixel_nm):
        key = self._geom_combo.currentData()
        params = {name: spin.value() for name, spin in self._template_spins.items()}
        return key, params, pa.geometry_template_image(key, params, box_nm, pixel_nm,
                                                       sigma_nm=pixel_nm)

    def _run(self) -> None:
        ds = self._dataset()
        if ds is None:
            return
        particles = self._extract_particles()
        if len(particles) < 2:
            self._status.setText("Need at least 2 box ROIs with localizations.")
            return
        box = float(self._box_size.value())
        px = float(self._pixel.value())
        n_angles = int(self._angles.value())
        self._status.setText("Aligning…")
        self._run_btn.setEnabled(False)
        try:
            if self._template_radio.isChecked():
                key, params, tmpl = self._template_image(box, px)
                res = pa.average_particles(
                    particles, mode="template", template_img=tmpl, box_nm=box,
                    pixel_nm=px, n_angles=n_angles, sigma_nm=px)
                mode_desc = f"template '{key}'"
            else:
                res = pa.average_particles(
                    particles, mode="free", box_nm=box, pixel_nm=px, n_angles=n_angles,
                    n_iter=int(self._iters.value()), sigma_nm=px)
                mode_desc = f"template-free, {res.get('iterations', 0)} iter"
        except Exception as exc:        # pragma: no cover - defensive
            self._status.setText(f"Averaging failed: {exc}")
            self._run_btn.setEnabled(True)
            return
        finally:
            self._run_btn.setEnabled(True)

        pts = res["points"]
        n = res["n_particles"]
        score = float(np.nanmean(res["scores"])) if res["scores"].size else float("nan")
        name = f"{ds.name} — avg [{n}p]"
        log = (f"Particle average ({mode_desc}): pooled {n} particle(s) "
               f"({pts.shape[0]:,} localizations) from '{ds.name}' "
               f"(box {box:.0f} nm, pixel {px:.1f} nm, {n_angles} angles, "
               f"mean align score {score:.3f}).")
        if self._owner is not None and hasattr(self._owner, "add_particle_average_dataset"):
            self._owner.add_particle_average_dataset(pts, name=name, log_message=log)
        self._status.setText(
            f"Averaged {n} particles · {pts.shape[0]:,} locs · mean score {score:.3f}. "
            "Opened render + scatter (switch to 3-D for depth).")
