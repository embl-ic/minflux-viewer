"""
minflux_viewer.ui.particle_average_dialog
==========================================
**Particle averaging** tool (Analyze › Segmentation › Particle Average…).

A **particle accumulator**: collect detected particles (box ROIs) from the active
dataset, keep **independent re-zeroed copies** inside the dialog, then keep
appending from further datasets/files so particles from many acquisitions
accumulate into one pool that is averaged together.

Workflow:
1. Detect on a dataset (e.g. Convolution / NPC detection wanlu) so box ROIs land
   in the ROI Manager; select the rows here and **Collect** — each ROI's
   localizations are cropped, **re-zeroed** (XY to the box centre, Z to the median)
   and stored as an independent copy with their ``tid`` / sub-unit ``unit_id``
   (labels are unaffected by the coordinate re-zero, so they travel with the
   particle). The source dataset can then be closed.
2. Load the next dataset, detect, **Collect** again — the pool grows.
3. Pick an **Average method** and **Run Average** — the accumulated particles are
   pooled into one averaged super-particle (render + scatter, 3-D-ready).

Average methods:
  * *template-free* — reference-free image alignment;
  * *template-provided* — align to a geometry template;
  * *NPC two-ring (Wanlu)* — constrained two-ring fit + 8-fold phase alignment
    (uses the stored ``tid`` / ``unit_id``).

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis import conv_segmentation as cs
from ..analysis import npc_wanlu as nw
from ..analysis import particle_average as pa
from ..core import roi_crop
from ..core.app_state import format_progress_bar
from ..core.loader import attr_values_1d
from ..core.roi_selection import rectangle_bounds, rectangle_mask

_MIN_LOCS_PER_PARTICLE = 5
_TEMPLATE_MODELS = ("ring", "disk", "gaussian")


class _AverageSignals(QObject):
    progress = pyqtSignal(int, int)        # done, total
    done = pyqtSignal(object, str)         # pooled points (ndarray), description
    failed = pyqtSignal(str)


class _AverageTask(QRunnable):
    """Run an averaging closure off the UI thread, reporting progress + result."""

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _AverageSignals()

    def run(self) -> None:  # noqa: N802 - Qt API
        try:
            pts, desc = self._fn(lambda d, t: self.signals.progress.emit(int(d), int(t)))
        except Exception as exc:               # never let a worker exception escape
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(np.asarray(pts), desc)


@dataclass
class _Particle:
    """One collected particle — an independent, re-zeroed copy."""

    points: np.ndarray            # (M, 3) re-zeroed nm (XY to box centre, Z to median)
    tid: np.ndarray | None        # (M,) trace ids (labels, unaffected by re-zero)
    unit_id: np.ndarray | None    # (M,) sub-unit ids (1-based within particle), or None
    source: str                   # source dataset name
    roi_name: str                 # source ROI name
    n_locs: int


class ParticleAverageWindow(QDialog):
    """Modeless particle-accumulator + averaging tool."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._owner = owner
        self.setWindowTitle("Particle Average")
        self.resize(380, 820)
        self._particles: list[_Particle] = []
        self._template_spins: dict[str, QDoubleSpinBox] = {}
        self._task = None            # running background averaging task (keep a ref)
        self._last_pct = -1
        self._pending: dict = {}

        self._build_ui()
        self._refresh_rois()
        self._on_method_changed()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # --- collect from active dataset -------------------------------------
        add_group = QGroupBox("Add particles from active dataset")
        av = QVBoxLayout(add_group)
        self._active_label = QLabel("")
        self._active_label.setStyleSheet("color:#888;")
        self._active_label.setWordWrap(True)
        av.addWidget(self._active_label)
        self._roi_list = QListWidget()
        self._roi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._roi_list.setMaximumHeight(120)
        av.addWidget(self._roi_list)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_rois)
        collect = QPushButton("Collect selected → particles")
        collect.clicked.connect(self._collect)
        row.addWidget(refresh)
        row.addWidget(collect, 1)
        av.addLayout(row)
        hint = QLabel("Select box ROIs (none = all), then Collect. Coordinates are "
                      "re-zeroed and stored as an independent copy.")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        av.addWidget(hint)
        root.addWidget(add_group)

        # --- accumulated particles -------------------------------------------
        pool_group = QGroupBox("Collected particles (0)")
        self._pool_group = pool_group
        pv = QVBoxLayout(pool_group)
        self._pool_list = QListWidget()
        self._pool_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        pv.addWidget(self._pool_list, 1)
        prow = QHBoxLayout()
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_selected)
        clr = QPushButton("Clear all")
        clr.clicked.connect(self._clear_particles)
        prow.addWidget(rm)
        prow.addWidget(clr)
        pv.addLayout(prow)
        root.addWidget(pool_group, 1)

        # --- average method ---------------------------------------------------
        method_group = QGroupBox("Average method")
        mv = QVBoxLayout(method_group)
        self._method = QComboBox()
        self._method.addItem("Template-free (reference-free)", "free")
        self._method.addItem("Template-provided (geometry)", "template")
        self._method.addItem("NPC two-ring (Wanlu)", "wanlu")
        self._method.currentIndexChanged.connect(self._on_method_changed)
        mv.addWidget(self._method)

        # template geometry sub-params
        self._template_box = QWidget()
        tb = QFormLayout(self._template_box)
        tb.setContentsMargins(12, 0, 0, 0)
        self._geom_combo = QComboBox()
        for key in _TEMPLATE_MODELS:
            self._geom_combo.addItem(cs.GEOMETRY_MODELS[key].label, key)
        self._geom_combo.currentIndexChanged.connect(lambda *_: self._rebuild_template_params())
        tb.addRow("Geometry:", self._geom_combo)
        self._template_param_form = QFormLayout()
        tb.addRow(self._template_param_form)
        mv.addWidget(self._template_box)

        # image-method params (template-free / template-provided)
        self._image_box = QWidget()
        ib = QFormLayout(self._image_box)
        ib.setContentsMargins(12, 0, 0, 0)
        self._box_size = self._dspin(8.0, 5000.0, 150.0, 0, 5.0, " nm")
        self._pixel = self._dspin(0.5, 50.0, 3.0, 1, 0.5, " nm")
        self._angles = QSpinBox(); self._angles.setRange(4, 360); self._angles.setValue(36)
        self._iters = QSpinBox(); self._iters.setRange(1, 30); self._iters.setValue(5)
        ib.addRow("Average box:", self._box_size)
        ib.addRow("Pixel size:", self._pixel)
        ib.addRow("Rotation steps:", self._angles)
        ib.addRow("Iterations:", self._iters)
        mv.addWidget(self._image_box)

        # NPC wanlu params
        self._wanlu_box = QWidget()
        wb = QFormLayout(self._wanlu_box)
        wb.setContentsMargins(12, 0, 0, 0)
        self._diam_min = self._dspin(20.0, 200.0, 70.0, 1, 1.0, " nm")
        self._diam_max = self._dspin(20.0, 300.0, 90.0, 1, 1.0, " nm")
        self._inter_min = self._dspin(5.0, 100.0, 20.0, 1, 1.0, " nm")
        self._inter_max = self._dspin(5.0, 150.0, 40.0, 1, 1.0, " nm")
        self._inter_exp = self._dspin(0.0, 150.0, 0.0, 1, 1.0, " nm")
        self._inter_exp.setToolTip("Expected inter-ring distance for the Z fallback (0 = bounds midpoint).")
        self._z_scale = self._dspin(0.1, 2.0, 1.0, 2, 0.01, "")
        self._z_scale.setToolTip("Z flattening (MATLAB rimf); < 1 (e.g. 0.67) for un-z-corrected data.")
        self._sym = QSpinBox(); self._sym.setRange(2, 16); self._sym.setValue(8)
        wb.addRow("Diameter min:", self._diam_min)
        wb.addRow("Diameter max:", self._diam_max)
        wb.addRow("Inter-ring min:", self._inter_min)
        wb.addRow("Inter-ring max:", self._inter_max)
        wb.addRow("Inter-ring expected:", self._inter_exp)
        wb.addRow("Z scale (rimf):", self._z_scale)
        wb.addRow("Symmetry:", self._sym)
        mv.addWidget(self._wanlu_box)
        root.addWidget(method_group)

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
    def _dspin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _on_method_changed(self) -> None:
        m = self._method.currentData()
        self._template_box.setVisible(m == "template")
        self._image_box.setVisible(m in ("free", "template"))
        self._wanlu_box.setVisible(m == "wanlu")

    def _rebuild_template_params(self) -> None:
        while self._template_param_form.rowCount():
            self._template_param_form.removeRow(0)
        self._template_spins.clear()
        model = cs.GEOMETRY_MODELS[self._geom_combo.currentData()]
        for ps in model.params:
            spin = self._dspin(ps.lo, ps.hi, ps.default, ps.decimals, ps.step, ps.suffix)
            self._template_param_form.addRow(ps.label + ":", spin)
            self._template_spins[ps.name] = spin

    # ----------------------------------------------------------- active dataset
    def _active_idx(self):
        idx = self._state.active_idx
        return idx if idx is not None and 0 <= idx < len(self._state.datasets) else None

    def _active_dataset(self):
        idx = self._active_idx()
        return self._state.datasets[idx] if idx is not None else None

    def _box_records(self) -> list:
        """Rectangle ROIs belonging to the active dataset (or untagged ones)."""
        idx = self._active_idx()
        recs = []
        for r in self._state.rois.records:
            if r.type != "rectangle":
                continue
            ctx = (r.context or {}).get("dataset_idx")
            if ctx is None or ctx == idx:
                recs.append(r)
        return recs

    def _refresh_rois(self) -> None:
        ds = self._active_dataset()
        self._active_label.setText(
            f"Active: <b>{ds.name}</b>" if ds else "No active dataset.")
        self._roi_list.clear()
        for rec in self._box_records():
            b = rectangle_bounds(rec)
            side = max(b[1] - b[0], b[3] - b[2]) if b else 0.0
            item = QListWidgetItem(f"{rec.name}   {side:.0f} nm")
            item.setData(Qt.ItemDataRole.UserRole, rec.id)
            self._roi_list.addItem(item)

    def _selected_records(self) -> list:
        ids = {it.data(Qt.ItemDataRole.UserRole) for it in self._roi_list.selectedItems()}
        recs = self._box_records()
        chosen = [r for r in recs if r.id in ids]
        return chosen if chosen else recs

    # ----------------------------------------------------------- collection
    def _collect(self) -> None:
        ds = self._active_dataset()
        if ds is None:
            self._status.setText("No active dataset to collect from.")
            return
        try:
            coords = roi_crop.display_coords(ds)              # (N, 3) display nm
        except Exception:
            self._status.setText("Active dataset has no localizations.")
            return
        n = coords.shape[0]
        tid_all = attr_values_1d(ds, "tid")
        tid_all = np.asarray(tid_all).ravel() if tid_all is not None else None
        fmask = np.asarray(ds.filter_mask, dtype=bool)
        if fmask.shape[0] != n:
            fmask = np.ones(n, dtype=bool)
        tid_to_unit = self._tid_unit_map(ds)

        added = 0
        for rec in self._selected_records():
            p = self._extract_particle(ds, rec, coords, tid_all, fmask, tid_to_unit)
            if p is not None:
                self._particles.append(p)
                added += 1
        self._populate_pool()
        self._status.setText(
            f"Collected {added} particle(s) from '{ds.name}' "
            f"(total {len(self._particles)}).")

    @staticmethod
    def _tid_unit_map(ds) -> dict:
        """trace_id → sub-unit_id, from an NPC-detection (Wanlu) ``raw6`` if present."""
        stored = (ds.derived or {}).get("npc_wanlu")
        if not stored:
            return {}
        raw6 = np.asarray(stored.get("raw6", []), dtype=float)
        if raw6.ndim != 2 or raw6.shape[0] == 0 or raw6.shape[1] < 6:
            return {}
        return {float(t): float(u) for t, u in zip(raw6[:, 5], raw6[:, 4])}

    def _extract_particle(self, ds, rec, coords, tid_all, fmask, tid_to_unit):
        in_box = rectangle_mask(coords[:, 0], coords[:, 1], rec)
        mask = in_box & fmask
        if int(mask.sum()) < _MIN_LOCS_PER_PARTICLE:
            return None
        pts = coords[mask].astype(float)
        b = rectangle_bounds(rec)
        cx, cy = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
        rez = pts.copy()
        rez[:, 0] -= cx
        rez[:, 1] -= cy
        rez[:, 2] -= float(np.median(rez[:, 2]))            # re-zero Z (box has no Z)

        tid = unit = None
        if tid_all is not None and tid_all.shape[0] == coords.shape[0]:
            tid = tid_all[mask]
            if tid_to_unit:
                raw_unit = np.array([tid_to_unit.get(float(t), np.nan) for t in tid])
            else:
                raw_unit = tid.astype(float)                # fallback: each trace = a sub-unit
            # renumber to 1..K within this particle
            valid = np.isfinite(raw_unit)
            unit = np.zeros(tid.shape[0], dtype=int)
            if valid.any():
                _u, inv = np.unique(raw_unit[valid], return_inverse=True)
                unit[valid] = inv + 1
        return _Particle(points=rez, tid=tid, unit_id=unit, source=ds.name,
                         roi_name=rec.name, n_locs=int(mask.sum()))

    def _populate_pool(self) -> None:
        self._pool_list.clear()
        for i, p in enumerate(self._particles):
            tag = "tid" if p.tid is not None else "no-tid"
            if p.unit_id is not None and p.unit_id.max() > 0:
                tag += f", {int(p.unit_id.max())}u"
            self._pool_list.addItem(QListWidgetItem(
                f"{p.source} · {p.roi_name} · {p.n_locs} locs ({tag})"))
        self._pool_group.setTitle(f"Collected particles ({len(self._particles)})")

    def _remove_selected(self) -> None:
        rows = sorted((self._pool_list.row(it) for it in self._pool_list.selectedItems()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._particles):
                del self._particles[r]
        self._populate_pool()

    def _clear_particles(self) -> None:
        self._particles.clear()
        self._populate_pool()

    # ----------------------------------------------------------- run
    def _build_raw6(self, z_scale: float) -> np.ndarray:
        rows = []
        for i, p in enumerate(self._particles, start=1):
            n = p.points.shape[0]
            tid = p.tid if p.tid is not None else np.arange(n)
            unit = p.unit_id if (p.unit_id is not None and p.unit_id.max() > 0) \
                else (np.arange(n) + 1)
            tid_g = np.asarray(tid, dtype=float) + i * 1e7      # globally-unique trace ids
            z = p.points[:, 2] * float(z_scale)
            rows.append(np.column_stack([p.points[:, 0], p.points[:, 1], z,
                                         np.full(n, i, float),
                                         np.asarray(unit, dtype=float), tid_g]))
        return np.vstack(rows) if rows else np.empty((0, 6))

    def _run(self) -> None:
        if self._task is not None:
            return                                           # already averaging
        if not self._particles:
            self._status.setText("Collect at least one particle first.")
            return
        if self._owner is None or not hasattr(self._owner, "add_particle_average_dataset"):
            return
        method = self._method.currentData()
        n_in = len(self._particles)

        # Read every widget value on the UI thread, then build a closure that does
        # the heavy work off-thread (never touch widgets from the worker).
        if method == "wanlu":
            raw6 = self._build_raw6(self._z_scale.value())
            dbounds = (self._diam_min.value(), self._diam_max.value())
            ibounds = (self._inter_min.value(), self._inter_max.value())
            exp = self._inter_exp.value()
            sym = self._sym.value()

            def compute(progress):
                res = nw.average_npc_wanlu(
                    raw6, diameter_bounds=dbounds, inter_ring_bounds=ibounds,
                    expected_inter_ring=(exp if exp > 0 else None), symmetry=sym,
                    progress=progress)
                zlo, zhi = res["z_peaks"]
                desc = (f"NPC two-ring: {res['n_accepted']}/{res['n_npc']} fitted, "
                        f"Z rings {zlo:.1f}/{zhi:.1f} nm (sep {zhi - zlo:.1f})")
                return res["average_loc"][:, :3], desc
        else:
            particles = [p.points for p in self._particles]
            box, px = float(self._box_size.value()), float(self._pixel.value())
            n_ang = int(self._angles.value())
            if method == "template":
                key = self._geom_combo.currentData()
                params = {k: s.value() for k, s in self._template_spins.items()}
                tmpl = pa.geometry_template_image(key, params, box, px, sigma_nm=px)

                def compute(progress):
                    res = pa.average_particles(particles, mode="template", template_img=tmpl,
                                               box_nm=box, pixel_nm=px, n_angles=n_ang,
                                               sigma_nm=px, progress=progress)
                    return res["points"], f"template '{key}'"
            else:
                n_iter = int(self._iters.value())

                def compute(progress):
                    res = pa.average_particles(particles, mode="free", box_nm=box, pixel_nm=px,
                                               n_angles=n_ang, n_iter=n_iter, sigma_nm=px,
                                               progress=progress)
                    return res["points"], f"template-free, {res.get('iterations', 0)} iter"

        self._pending = {"n_in": n_in, "method": method, "sources": self._n_sources()}
        self._last_pct = -1
        self._state.log(f"Particle average ({method}): averaging {n_in} collected particle(s)…")
        self._state.log_progress(format_progress_bar(0.0))
        self._status.setText("Averaging… (progress in the Log; the app stays responsive)")
        self._run_btn.setEnabled(False)

        task = _AverageTask(compute)
        task.signals.progress.connect(self._on_progress)
        task.signals.done.connect(self._on_done)
        task.signals.failed.connect(self._on_failed)
        self._task = task                                    # keep a ref (QRunnable auto-deletes)
        QThreadPool.globalInstance().start(task)

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(done / total * 100)
        if pct != self._last_pct:                            # throttle to whole-percent updates
            self._last_pct = pct
            self._state.log_progress(format_progress_bar(done / total))

    def _on_done(self, pts, desc: str) -> None:
        self._task = None
        self._run_btn.setEnabled(True)
        self._state.log_progress(format_progress_bar(1.0, done=True), final=True)
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[0] == 0:
            self._status.setText("No particles passed the average. Try widening bounds / params.")
            return
        n_in = self._pending.get("n_in", len(self._particles))
        method = self._pending.get("method", "")
        log = (f"Particle average ({desc}): pooled {n_in} collected particle(s) "
               f"({pts.shape[0]:,} localizations) from {self._pending.get('sources', 1)} dataset(s).")
        try:
            self._owner.add_particle_average_dataset(
                pts[:, :3], name=f"Particle avg [{n_in}p, {method}]", log_message=log)
        except Exception as exc:                              # pragma: no cover - defensive
            self._status.setText(f"Average computed but display failed: {exc}")
            return
        self._status.setText(
            f"Averaged {n_in} particles · {pts.shape[0]:,} locs ({desc}). "
            "Opened render + scatter (switch to 3-D for depth).")

    def _on_failed(self, msg: str) -> None:
        self._task = None
        self._run_btn.setEnabled(True)
        self._state.log_progress("=" * 10 + "  FAILED  " + "=" * 10, final=True)
        self._status.setText(f"Averaging failed: {msg}")

    def _n_sources(self) -> int:
        return len({p.source for p in self._particles})
