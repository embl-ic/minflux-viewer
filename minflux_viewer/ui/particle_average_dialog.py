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
   pooled into one averaged super-particle (render view, 3-D-ready).

Average methods:
  * *template-free* — reference-free image alignment;
  * *template-provided* — align to a geometry template;
  * *NPC two-ring (Wanlu)* — constrained two-ring fit + 8-fold phase alignment
    (uses the stored ``tid`` / ``unit_id``).

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis import conv_segmentation as cs
from ..analysis import npc_wanlu as nw
from ..analysis import particle_average as pa
from ..core.app_state import format_progress_bar
from ..core.particle_extract import (
    extract_particles,
    load_any_dataset,
    records_from_roi_json,
)
from ..core.particle_set import ParticleData, load_particle_set, save_particle_set
from ..core.roi_selection import rectangle_bounds

_TEMPLATE_MODELS = ("ring", "disk", "gaussian", "npc")

# The NPC 8-fold template model is kept **local** to the particle-average plugin so
# it does not leak into the Convolution Segmentation geometry registry
# (`cs.GEOMETRY_MODELS`, which the conv dialogs enumerate). It needs no make_kernel —
# only its ParamSpecs (for the widgets) + `geometry_template_image("npc", …)`.
_NPC_TEMPLATE_PARAMS = (
    cs.ParamSpec("diameter_nm", "Diameter", 100.0, 10.0, 5000.0, 1.0, 1, " nm"),
    cs.ParamSpec("corner_nm", "Corner size", 12.0, 1.0, 500.0, 0.5, 1, " nm",
                 "Gaussian σ of each corner blob."),
    cs.ParamSpec("symmetry", "Symmetry (fold)", 8.0, 2.0, 16.0, 1.0, 0, "",
                 "Number of evenly-spaced corner blobs on the ring (NPC = 8)."),
)


def _template_model_spec(key: str):
    """(label, ParamSpec tuple) for a particle-average geometry template key."""
    if key == "npc":
        return "NPC ring (8-fold)", _NPC_TEMPLATE_PARAMS
    m = cs.GEOMETRY_MODELS[key]
    return m.label, m.params


class _AverageSignals(QObject):
    progress = pyqtSignal(int, int)        # done, total
    done = pyqtSignal(object, str, object)  # pooled points, description, extra(dict|None)
    failed = pyqtSignal(str)


class _AverageTask(QRunnable):
    """Run an averaging closure off the UI thread, reporting progress + result."""

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _AverageSignals()

    def run(self) -> None:  # noqa: N802 - Qt API
        try:
            result = self._fn(lambda d, t: self.signals.progress.emit(int(d), int(t)))
        except Exception as exc:               # never let a worker exception escape
            self.signals.failed.emit(str(exc))
            return
        pts, desc = result[0], result[1]
        extra = result[2] if len(result) > 2 else None
        self.signals.done.emit(np.asarray(pts), desc, extra)


class ParticleAverageWindow(QDialog):
    """Modeless particle-accumulator + averaging tool."""

    def __init__(self, state, dataset_idx: int | None = None, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._owner = owner
        self.setWindowTitle("Particle Average")
        self.resize(380, 820)
        self._particles: list[ParticleData] = []
        # Parallel to _particles: per-particle {channel_name: (M,3) re-zeroed points}
        # from the other overlay channels (multi-channel averaging).
        self._channels_by_particle: list[dict] = []
        self._channel_luts: dict[str, str] = {}
        self._ref_channel_name = ""
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
        prow2 = QHBoxLayout()
        save_btn = QPushButton("Save particles…")
        save_btn.clicked.connect(self._save_particles)
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._load_particles)
        prow2.addWidget(save_btn)
        prow2.addWidget(load_btn)
        pv.addLayout(prow2)
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
            self._geom_combo.addItem(_template_model_spec(key)[0], key)
        # Connect AFTER populating so the initial addItem does not fire the handler
        # before _tilt_correct exists.
        self._geom_combo.currentIndexChanged.connect(self._on_geom_changed)
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
        self._tilt_correct = QCheckBox("Correct axial tilt (3-D, per-particle PCA)")
        self._tilt_correct.setToolTip(
            "Un-tilt each particle onto its best-fit plane (PCA) before 2-D "
            "alignment, so tilted 3-D particles average correctly. No-op for 2-D data.")
        self._multichannel = QCheckBox("Combine overlay channels (apply transform)")
        self._multichannel.setToolTip(
            "When the active dataset is one channel of an overlay: align/average using "
            "this (reference) channel and apply each particle's transform to the other "
            "channel(s) too, producing a combined multi-channel averaged overlay. "
            "Default on for an overlay; only the reference channel is used to align.")
        self._multichannel.clicked.connect(lambda *_: setattr(self, "_multichannel_user_set", True))
        ib.addRow("Average box:", self._box_size)
        ib.addRow("Pixel size:", self._pixel)
        ib.addRow("Rotation steps:", self._angles)
        ib.addRow("Iterations:", self._iters)
        ib.addRow("", self._tilt_correct)
        ib.addRow("", self._multichannel)
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
        self._min_gof = self._dspin(-1.0, 1.0, 0.0, 2, 0.05, "")
        self._min_gof.setToolTip(
            "Goodness-of-fit gate (R²): only NPCs whose two-ring fit explains at "
            "least this fraction of the point spread are pooled. < 0 keeps all; "
            "higher = stricter. Every fitted NPC is still listed in the result table.")
        wb.addRow("Diameter min:", self._diam_min)
        wb.addRow("Diameter max:", self._diam_max)
        wb.addRow("Inter-ring min:", self._inter_min)
        wb.addRow("Inter-ring max:", self._inter_max)
        wb.addRow("Inter-ring expected:", self._inter_exp)
        wb.addRow("Z scale (rimf):", self._z_scale)
        wb.addRow("Symmetry:", self._sym)
        wb.addRow("Min goodness of fit:", self._min_gof)
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

    def _on_geom_changed(self) -> None:
        self._rebuild_template_params()
        # The NPC ring model targets a 3-D structure, so bundle the axial-tilt
        # correction with it (still user-toggleable).
        if self._geom_combo.currentData() == "npc" and hasattr(self, "_tilt_correct"):
            self._tilt_correct.setChecked(True)

    def _rebuild_template_params(self) -> None:
        while self._template_param_form.rowCount():
            self._template_param_form.removeRow(0)
        self._template_spins.clear()
        _label, params = _template_model_spec(self._geom_combo.currentData())
        for ps in params:
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

    def _overlay_members(self) -> list:
        """The active dataset's overlay channels ``[(name, ds, lut), …]`` (reference
        first), or ``[]`` when it is not part of a multi-channel overlay."""
        from ..core.overlay import dataset_group_id, overlay_members
        idx = self._active_idx()
        if idx is None:
            return []
        active = self._state.datasets[idx]
        if not dataset_group_id(active):
            return []
        members = overlay_members(self._state, idx)
        if len(members) < 2:
            return []
        out = []
        for i, ds in members:
            lut = ds.state.get("render_channel_lut") or ds.state.get("overlay_lut") or ""
            out.append((ds.name, ds, str(lut)))
        # reference (active) channel first
        out.sort(key=lambda t: 0 if t[1] is active else 1)
        return out

    def _should_collect_multichannel(self) -> bool:
        """At collect time: pull the other overlay channels too?"""
        return (self._multichannel.isEnabled() and self._multichannel.isChecked()
                and len(self._overlay_members()) >= 2)

    def _multichannel_active(self) -> bool:
        """At run time: do we have collected channel data + a compatible method?"""
        return (self._method.currentData() in ("free", "template")
                and self._multichannel.isChecked() and len(self._channel_luts) >= 2)

    def _multichannel_data(self) -> dict:
        """Per-channel per-particle points (parallel to ``_particles``) + luts, for
        the worker to replay the reference transform onto the other channels."""
        if not self._multichannel_active():
            return {"channels": {}, "ref_name": "", "luts": {}, "ref_lut": ""}
        n = len(self._particles)
        others = [name for name in self._channel_luts if name != self._ref_channel_name]
        channels = {
            name: [(self._channels_by_particle[i].get(name) if i < len(self._channels_by_particle) else None)
                   for i in range(n)]
            for name in others}
        return {"channels": channels, "ref_name": self._ref_channel_name,
                "luts": dict(self._channel_luts),
                "ref_lut": self._channel_luts.get(self._ref_channel_name, "")}

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
        members = self._overlay_members()
        if ds and members:
            names = ", ".join(n for n, _d, _l in members)
            self._active_label.setText(
                f"Active: <b>{ds.name}</b> — overlay of {len(members)} channels ({names})")
        else:
            self._active_label.setText(
                f"Active: <b>{ds.name}</b>" if ds else "No active dataset.")
        # Multi-channel is available (and on by default) only for an overlay.
        in_overlay = bool(members)
        self._multichannel.setEnabled(in_overlay)
        if in_overlay and not getattr(self, "_multichannel_user_set", False):
            self._multichannel.setChecked(True)
        elif not in_overlay:
            self._multichannel.setChecked(False)
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
        records = self._selected_records()
        try:
            if self._should_collect_multichannel():
                added = self._collect_multichannel(ds, records)
                extra = f" + {len(self._channel_luts) - 1} overlay channel(s)"
            else:
                new = extract_particles(ds, records)
                self._particles.extend(new)
                self._channels_by_particle.extend({} for _ in new)
                added = len(new)
                extra = ""
        except Exception as exc:
            self._status.setText(f"Collect failed: {exc}")
            return
        self._populate_pool()
        self._status.setText(
            f"Collected {added} particle(s) from '{ds.name}'{extra} "
            f"(total {len(self._particles)}).")

    def _collect_multichannel(self, ds, records) -> int:
        """Collect the reference particle **and** the co-located points from every
        other overlay channel (re-zeroed by the reference offset) per ROI."""
        from ..core.particle_extract import channel_points_in_roi
        members = self._overlay_members()
        self._ref_channel_name = ds.name
        self._channel_luts = {name: lut for name, _d, lut in members}
        others = [(name, d) for name, d, _l in members if d is not ds]
        added = 0
        for rec in records:
            ps = extract_particles(ds, [rec])
            if not ps:
                continue
            p = ps[0]
            offset = p.meta.get("rezero_offset", [0.0, 0.0, 0.0])
            chans: dict = {}
            for name, other in others:
                pts = channel_points_in_roi(other, rec, offset)
                if pts.shape[0] >= 1:
                    chans[name] = pts
            self._particles.append(p)
            self._channels_by_particle.append(chans)
            added += 1
        return added

    def _populate_pool(self) -> None:
        self._pool_list.clear()
        for i, p in enumerate(self._particles):
            tag = "tid" if p.tid is not None else "no-tid"
            if p.unit_id is not None and p.unit_id.max() > 0:
                tag += f", {int(p.unit_id.max())}u"
            nch = len(self._channels_by_particle[i]) if i < len(self._channels_by_particle) else 0
            if nch:
                tag += f", +{nch}ch"
            self._pool_list.addItem(QListWidgetItem(
                f"{p.source} · {p.roi_name} · {p.n_locs} locs ({tag})"))
        self._pool_group.setTitle(f"Collected particles ({len(self._particles)})")

    def _remove_selected(self) -> None:
        rows = sorted((self._pool_list.row(it) for it in self._pool_list.selectedItems()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._particles):
                del self._particles[r]
                if r < len(self._channels_by_particle):
                    del self._channels_by_particle[r]
        self._populate_pool()

    def _clear_particles(self) -> None:
        self._particles.clear()
        self._channels_by_particle.clear()
        self._channel_luts = {}
        self._populate_pool()

    # ----------------------------------------------------------- save / load
    def _save_particles(self) -> None:
        if not self._particles:
            self._status.setText("Collect or load at least one particle first.")
            return
        opts = ParticleSaveDialog(self).get_options()
        if opts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save particle set", "particles.h5",
            "MINFLUX particle set (*.h5);;All files (*)")
        if not path:
            return
        if not path.lower().endswith((".h5", ".hdf5")):
            path += ".h5"
        try:
            save_particle_set(path, self._particles,
                              content=opts["content"], include_raw_loc=opts["include_raw_loc"])
        except Exception as exc:
            self._status.setText(f"Save failed: {exc}")
            self._state.log(f"Particle set save failed: {exc}", "ERROR")
            return
        n_attr = len({k for p in self._particles for k in (p.attrs or {})}) \
            if opts["content"] == "all" else 0
        self._state.log(
            f"Saved {len(self._particles)} particle(s) to '{Path(path).name}' "
            f"(content={opts['content']}, raw loc={'yes' if opts['include_raw_loc'] else 'no'}, "
            f"{n_attr} attribute column(s)).")
        self._status.setText(f"Saved {len(self._particles)} particle(s) → {Path(path).name}")

    def _load_particles(self) -> None:
        spec = ParticleLoadDialog(self).get_spec()
        if spec is None:
            return
        loaded: list = []
        errors: list[str] = []
        if spec["mode"] == "particle_set":
            paths = spec["paths"]
            # Pool every file; a bad file is skipped (logged) so one failure
            # doesn't abort the whole batch.
            for p in paths:
                try:
                    loaded.extend(load_particle_set(p))
                except Exception as exc:
                    errors.append(f"{Path(p).name}: {exc}")
                    self._state.log(f"Particle load failed ({Path(p).name}): {exc}", "ERROR")
            src = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} particle-set files"
        else:
            try:
                ds = load_any_dataset(spec["data_path"], prefs=self._state.prefs)
                records = records_from_roi_json(spec["roi_path"])
                loaded = extract_particles(ds, records)
                src = f"{Path(spec['data_path']).name} + {Path(spec['roi_path']).name}"
            except Exception as exc:
                self._status.setText(f"Load failed: {exc}")
                self._state.log(f"Particle load failed: {exc}", "ERROR")
                return
        if not loaded:
            if errors:
                self._status.setText(f"Load failed: {errors[0]}"
                                     + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""))
            else:
                self._status.setText("No particles found in the selection (no box ROIs / too few locs?).")
            return
        self._particles.extend(loaded)
        self._channels_by_particle.extend({} for _ in loaded)   # loaded particles: single channel
        self._populate_pool()
        msg = (f"Loaded {len(loaded)} particle(s) from {src} "
               f"(total {len(self._particles)}).")
        if errors:
            msg += f"  {len(errors)} file(s) failed."
        self._state.log(msg)
        self._status.setText(msg)

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
            mingof = float(self._min_gof.value())

            def compute(progress):
                res = nw.average_npc_wanlu(
                    raw6, diameter_bounds=dbounds, inter_ring_bounds=ibounds,
                    expected_inter_ring=(exp if exp > 0 else None), symmetry=sym,
                    min_gof=mingof, progress=progress)
                zlo, zhi = res["z_peaks"]
                desc = (f"NPC two-ring: {res['n_accepted']}/{res['n_npc']} accepted "
                        f"(GoF≥{mingof:g}), Z rings {zlo:.1f}/{zhi:.1f} nm (sep {zhi - zlo:.1f})")
                return res["average_loc"][:, :3], desc, {"table": res.get("table", []),
                                                         "min_gof": mingof}
        else:
            particles = [p.points for p in self._particles]
            box, px = float(self._box_size.value()), float(self._pixel.value())
            n_ang = int(self._angles.value())
            tilt = bool(self._tilt_correct.isChecked())
            tilt_tag = ", tilt-corrected" if tilt else ""
            # Multi-channel: capture the other channels' per-particle points (UI
            # thread) so the worker can replay the reference's per-particle transform.
            mc = self._multichannel_data()
            n_locs = len(self._particles)

            def _channel_extra(res):
                if not mc["channels"]:
                    return {}
                mats = res.get("particle_transforms", [])
                pooled = {name: pa.pool_transformed(pl, mats) for name, pl in mc["channels"].items()}
                return {"channels": pooled, "ref_name": mc["ref_name"],
                        "luts": mc["luts"], "ref_lut": mc["ref_lut"]}

            if method == "template":
                key = self._geom_combo.currentData()
                key_label = _template_model_spec(key)[0]
                params = {k: s.value() for k, s in self._template_spins.items()}
                tmpl = pa.geometry_template_image(key, params, box, px, sigma_nm=px)

                def compute(progress):
                    res = pa.average_particles(particles, mode="template", template_img=tmpl,
                                               box_nm=box, pixel_nm=px, n_angles=n_ang,
                                               sigma_nm=px, correct_tilt=tilt, progress=progress)
                    return res["points"], f"template '{key_label}'{tilt_tag}", _channel_extra(res)
            else:
                n_iter = int(self._iters.value())

                def compute(progress):
                    res = pa.average_particles(particles, mode="free", box_nm=box, pixel_nm=px,
                                               n_angles=n_ang, n_iter=n_iter, sigma_nm=px,
                                               correct_tilt=tilt, progress=progress)
                    return (res["points"], f"template-free, {res.get('iterations', 0)} iter{tilt_tag}",
                            _channel_extra(res))

        self._pending = {"n_in": n_in, "method": method, "sources": self._n_sources()}
        self._last_pct = -1
        self._state.log(f"Particle average ({method}): averaging {n_in} collected particle(s)…")
        self._state.log_progress(format_progress_bar(0.0))
        self._state.status_progress(f"Particle averaging ({method})", 0.0)
        self._status.setText("Averaging… (progress in the main window status bar + Log; "
                             "the app stays responsive)")
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
            self._state.status_progress(
                f"Particle averaging ({self._pending.get('method', '')})", done / total)

    def _on_done(self, pts, desc: str, extra=None) -> None:
        self._task = None
        self._run_btn.setEnabled(True)
        self._state.log_progress(format_progress_bar(1.0, done=True), final=True)
        self._state.status_message.emit("Particle average: done.")
        # A per-particle fit table (NPC two-ring) is shown even if nothing was
        # pooled, so the user can see *why* (e.g. all rejected by the GoF gate).
        table = (extra or {}).get("table") if isinstance(extra, dict) else None
        if table:
            self._show_fit_table(table, (extra or {}).get("min_gof"))
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[0] == 0:
            self._status.setText("No particles passed the average / GoF gate. "
                                 "See the fit table; try widening bounds or lowering Min GoF.")
            return
        n_in = self._pending.get("n_in", len(self._particles))
        method = self._pending.get("method", "")
        channels = (extra or {}).get("channels") if isinstance(extra, dict) else None
        log = (f"Particle average ({desc}): pooled {n_in} collected particle(s) "
               f"({pts.shape[0]:,} localizations) from {self._pending.get('sources', 1)} dataset(s).")
        try:
            if channels:
                # Multi-channel: reference average + each other channel (transform
                # replayed) as a combined overlay.
                ref_name = (extra or {}).get("ref_name") or "reference"
                luts = (extra or {}).get("luts") or {}
                chan_data = {ref_name: pts[:, :3]}
                for name, cp in channels.items():
                    arr = np.asarray(cp)
                    if arr.ndim == 2 and arr.shape[0]:
                        chan_data[name] = arr[:, :3]
                log += f"  Multi-channel overlay: {', '.join(chan_data)}."
                self._owner.add_particle_average_overlay(
                    chan_data, luts=luts, name=f"Particle avg [{n_in}p, {method}]",
                    log_message=log)
            else:
                self._owner.add_particle_average_dataset(
                    pts[:, :3], name=f"Particle avg [{n_in}p, {method}]", log_message=log)
        except Exception as exc:                              # pragma: no cover - defensive
            self._status.setText(f"Average computed but display failed: {exc}")
            return
        self._status.setText(
            f"Averaged {n_in} particles · {pts.shape[0]:,} locs ({desc}). "
            "Opened render view (switch to 3-D for depth).")

    def _show_fit_table(self, table, min_gof) -> None:
        owner = self._owner if self._owner is not None else self
        win = ParticleFitTableWindow(table, min_gof=min_gof)
        from .modeless import show_modeless
        show_modeless(win, owner)
        n_acc = sum(1 for r in table if r.get("accepted"))
        self._state.log(f"Particle fit table: {n_acc}/{len(table)} NPC(s) passed the "
                        f"GoF gate (min GoF {min_gof:g}).")

    def _on_failed(self, msg: str) -> None:
        self._task = None
        self._run_btn.setEnabled(True)
        self._state.log_progress("=" * 10 + "  FAILED  " + "=" * 10, final=True)
        self._state.status_message.emit("Particle average: failed.")
        self._status.setText(f"Averaging failed: {msg}")

    def _n_sources(self) -> int:
        return len({p.source for p in self._particles})


class ParticleSaveDialog(QDialog):
    """Save-time options: how much per-localization information to store."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save particle set")
        self.setModal(True)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Store per localization:"))
        self._content = QComboBox()
        self._content.addItem("Coordinates + tid + sub-unit only (smallest)", "loc_tid")
        self._content.addItem("Coordinates + all attributes (recommended)", "all")
        self._content.setCurrentIndex(1)
        root.addWidget(self._content)
        self._raw = QCheckBox("Include raw localization (loc_x/y/z, metres) for provenance")
        self._raw.setChecked(True)
        root.addWidget(self._raw)
        hint = QLabel("xnm/ynm/znm (re-zeroed, analysis-ready) are always written. "
                      "Raw loc is provenance only — never re-corrected on load.")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        root.addWidget(hint)
        self._content.currentIndexChanged.connect(
            lambda: self._raw.setEnabled(self._content.currentData() == "all"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def get_options(self) -> dict | None:
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        content = self._content.currentData()
        return {"content": content,
                "include_raw_loc": bool(self._raw.isChecked()) and content == "all"}


class ParticleLoadDialog(QDialog):
    """Choose what to load: a particle-set .h5, or a data + detection-ROI(.json) pair."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load particles")
        self.setModal(True)
        self.resize(480, 0)
        self._set_paths: list[str] = []       # one or more particle-set files
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Source:"))
        self._mode = QComboBox()
        self._mode.addItem("Particle set file(s) (.h5)", "particle_set")
        self._mode.addItem("Data + detection ROI (.json) pair", "data_roi")
        self._mode.currentIndexChanged.connect(self._sync)
        root.addWidget(self._mode)

        self._set_row, self._set_edit = self._file_row("Particle set(s):")
        self._data_row, self._data_edit = self._file_row("Data file:")
        self._roi_row, self._roi_edit = self._file_row("ROI .json:")
        for row in (self._set_row, self._data_row, self._roi_row):
            root.addLayout(row)

        hint = QLabel("Select one or more particle-set files to pool all their "
                      "particles at once. A data + ROI pair instead crops + re-zeroes "
                      "particles from the box ROIs in the .json (e.g. detected NPCs) — "
                      "the dataset need not be open.")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        self._sync()

    def _file_row(self, label: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        edit.setReadOnly(True)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda _=False, lbl=label, e=edit: self._browse(lbl, e))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        return row, edit

    def _browse(self, label: str, edit: QLineEdit) -> None:
        if "Particle" in label:
            # Multiple particle-set files can be pooled in one load.
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select particle set file(s)", "",
                "MINFLUX particle set (*.h5 *.hdf5);;All files (*)")
            if paths:
                self._set_paths = list(paths)
                edit.setText(self._summarize(paths))
            return
        if "ROI" in label:
            filt = "ROI set (*.json);;All files (*)"
        else:
            filt = "MINFLUX data (*.mat *.npy *.csv *.tsv *.txt);;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            edit.setText(path)

    @staticmethod
    def _summarize(paths: list[str]) -> str:
        names = ", ".join(Path(p).name for p in paths)
        return names if len(paths) == 1 else f"{len(paths)} files: {names}"

    @staticmethod
    def _row_visible(row, visible: bool) -> None:
        for i in range(row.count()):
            w = row.itemAt(i).widget()
            if w is not None:
                w.setVisible(visible)

    def _sync(self) -> None:
        m = self._mode.currentData()
        self._row_visible(self._set_row, m == "particle_set")
        self._row_visible(self._data_row, m == "data_roi")
        self._row_visible(self._roi_row, m == "data_roi")

    def get_spec(self) -> dict | None:
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        return self._build_spec()

    def _build_spec(self) -> dict | None:
        if self._mode.currentData() == "particle_set":
            return {"mode": "particle_set", "paths": list(self._set_paths)} if self._set_paths else None
        data, roi = self._data_edit.text().strip(), self._roi_edit.text().strip()
        return {"mode": "data_roi", "data_path": data, "roi_path": roi} if (data and roi) else None


class ParticleFitTableWindow(QWidget):
    """Per-NPC two-ring fit metrics (recomputed on the best fit). Modeless."""

    _COLS = [
        ("NPC", "npc_id", "{:d}"),
        ("Locs", "n_locs", "{:d}"),
        ("Radius (nm)", "radius", "{:.1f}"),
        ("Rim (nm)", "rim", "{:.1f}"),
        ("Inter-ring (nm)", "inter", "{:.1f}"),
        ("Tilt (°)", "tilt", "{:.1f}"),
        ("GoF (R²)", "gof", "{:.3f}"),
        ("RMSE (nm)", "rmse", "{:.2f}"),
        ("NRMSE", "nrmse", "{:.3f}"),
        ("Accepted", "accepted", None),
    ]

    def __init__(self, table, *, min_gof=None) -> None:
        super().__init__(None)
        self._rows = list(table)
        self.setWindowTitle("Particle fit table — NPC two-ring")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(760, 440)
        root = QVBoxLayout(self)
        n_acc = sum(1 for r in self._rows if r.get("accepted"))
        head = f"{n_acc} / {len(self._rows)} NPC(s) accepted"
        if min_gof is not None:
            head += f"   (min GoF = {min_gof:g})"
        lbl = QLabel(head)
        lbl.setStyleSheet("font-weight:bold;")
        root.addWidget(lbl)

        self._tbl = QTableWidget(len(self._rows), len(self._COLS))
        self._tbl.setHorizontalHeaderLabels([c[0] for c in self._COLS])
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl.verticalHeader().setVisible(False)
        self._fill()
        root.addWidget(self._tbl, 1)

        btns = QHBoxLayout()
        save = QPushButton("Save CSV…")
        save.clicked.connect(self._save_csv)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        btns.addWidget(save)
        btns.addStretch(1)
        btns.addWidget(close)
        root.addLayout(btns)

    def _fill(self) -> None:
        from PyQt6.QtGui import QColor
        for r, row in enumerate(self._rows):
            accepted = bool(row.get("accepted"))
            for c, (_lbl, key, fmt) in enumerate(self._COLS):
                v = row.get(key)
                if key == "accepted":
                    text = "yes" if accepted else "no"
                elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                    text = "—"
                else:
                    try:
                        text = fmt.format(v)
                    except Exception:
                        text = str(v)
                item = QTableWidgetItem(text)
                if not accepted:
                    item.setForeground(QColor("#999"))
                self._tbl.setItem(r, c, item)

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save fit table", "npc_fit_table.csv",
                                              "CSV (*.csv);;All files (*)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        import csv
        keys = [c[1] for c in self._COLS]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([c[0] for c in self._COLS])
            for row in self._rows:
                writer.writerow([row.get(k) for k in keys])
