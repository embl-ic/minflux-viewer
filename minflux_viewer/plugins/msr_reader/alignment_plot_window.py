'Alignment plot window'

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.ui.plots.plot_format import apply_spatial_y_direction


class AlignmentPlotWindow(QDialog):
    _single_channel = None   # class default so bypass-__init__ tests see None

    def __init__(self, results, parent=None, data_bounds_nm=None,
                 requested_transform_type=None, single_channel=None,
                 prefs: dict | None = None):
        super().__init__(parent)
        # single_channel (dict: name/bead_ids/rids/pos_nm/drift_nm) → no alignment;
        # show the beads + data region + a per-bead drift table.
        self._single_channel = single_channel
        self.setWindowTitle(
            "Beads and data region" if single_channel is not None
            else "Beads and alignment result")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1150, 800)
        self._owner = parent
        self._prefs = prefs or {}
        self.results = results
        # Requested MBM-handling mode (may be downgraded per bead count when re-fit);
        # falls back to the effective mode already on the results.
        from minflux_viewer.core.bead_alignment import TRANSFORM_RIGID_XY_Z
        self._requested_transform_type = (
            requested_transform_type
            or (getattr(results[0], "transform_type", "") if results else "")
            or TRANSFORM_RIGID_XY_Z
        )
        # Immutable per-moving-channel bead data (all matched beads); the live re-fit
        # works on subsets of this without re-parsing the raw MBM point arrays.
        self._base = [
            {
                "ref_name": r.channel1_name,
                "mov_name": r.channel2_name,
                "bead_ids": np.asarray(r.bead_ids, dtype=np.uint32),
                "ref_xyz": np.asarray(r.channel1_xyz, dtype=np.float64),
                "mov_xyz": np.asarray(r.channel2_xyz, dtype=np.float64),
                "ref_counts": np.asarray(getattr(r, "channel1_counts", np.zeros(np.asarray(r.bead_ids).size)), dtype=np.int32),
                "mov_counts": np.asarray(getattr(r, "channel2_counts", np.zeros(np.asarray(r.bead_ids).size)), dtype=np.int32),
            }
            for r in results
        ]
        self._excluded_gris: set[int] = set()
        # Beads the owner had already excluded before this window opened (they are not
        # among the passed-in results). Writeback unions them so a window exclusion adds
        # to — never replaces — the existing selection.
        self._owner_prior_excluded = set(
            int(g) for g in (getattr(parent, "_bead_unchecked_gris", None) or set()))
        self._current: list[dict] = []
        self._row_order: list[tuple[int, int, int]] = []
        self._label_items: list = []
        self._label_items_3d: list = []
        self._suspend_bead_signals = False
        # (min_xyz, max_xyz) nm bounding box of all datasets' raw localizations,
        # drawn as a view-aware wireframe data-range box. None = bead bounds.
        self.data_bounds_nm = data_bounds_nm
        self.visibility = {}
        self._items = {}
        self._axis_indices = (0, 1)
        self._view_mode = "XY"
        self._view_3d = None
        self._grid_3d = None
        self._axis_3d = None
        self._items_3d = {}
        self._box_3d_items = []
        self._axis_3d_items = []
        self._show_3d_axis = True
        self._show_3d_bounding_box = True

        self._recompute_current()
        self.datasets = self._build_plot_datasets()

        import pyqtgraph as pg

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Visible channels:"))
        for key, meta in self.datasets.items():
            cb = QCheckBox(meta["label"])
            cb.setChecked(True)
            cb.toggled.connect(self._update_visibility)
            self.visibility[key] = cb
            controls.addWidget(cb)
        controls.addStretch(1)
        self.hover_label = QLabel("Hover near a point to see channel and coordinates (µm).")
        controls.addWidget(self.hover_label)
        layout.addLayout(controls)

        # Readout #1: fit-quality banner (RMSE per moving channel + a warning when
        # the fiducial registration is poor). Refreshed live on bead selection change.
        self.quality_banner = QLabel()
        self.quality_banner.setWordWrap(True)
        self.quality_banner.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self.quality_banner)

        self.plot = pg.PlotWidget(background="w")
        try:
            self.plot.setMenuEnabled(False)
            self.plot.getPlotItem().getViewBox().setMenuEnabled(False)
        except Exception:
            pass
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._show_context_menu)
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setAspectLocked(True)
        self.plot.addLegend()
        self.plot.getAxis("bottom").setPen("k")
        self.plot.getAxis("left").setPen("k")
        self.plot.getAxis("bottom").setTextPen("k")
        self.plot.getAxis("left").setTextPen("k")
        self._stack = QStackedWidget()
        self._stack.addWidget(self.plot)
        self._stack.setMinimumHeight(160)

        # Readout #2: per-bead residual table (worst-first) with a per-bead "Bead"
        # checkbox — un/re-checking a bead re-fits the alignment live and updates the
        # banner + plot. Δ = raw per-bead offset (reference − moving); resid = error
        # remaining after the fitted transform.
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.resid_caption = QLabel(
            "Per-bead residuals (worst first) — tick/untick a bead to exclude/include it "
            "in the fit (updates live). Δ = raw offset (ref − mov); resid = error remaining "
            "after the fitted transform, in nm:"
        )
        self.resid_caption.setWordWrap(True)
        table_layout.addWidget(self.resid_caption)
        self.resid_table = QTableWidget()
        self.resid_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.resid_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.resid_table.setAlternatingRowColors(True)
        self.resid_table.setMinimumHeight(70)
        self.resid_table.setToolTip(
            "Tick a bead to include it in the alignment fit; untick to exclude it.\n"
            "Δ = raw per-bead offset (reference − moving), nm.\n"
            "resid = error remaining after the fitted transform (aligned − reference), nm.\n"
            "Large, inconsistent residuals mean the beads disagree with each other, so a "
            "single transform cannot register the channels accurately."
        )
        table_layout.addWidget(self.resid_table)

        # Draggable split between the scatter plot and the residual table, so the user
        # can give the table more height to show more rows (drag the handle between them,
        # not just the dialog border).
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._stack)
        split.addWidget(table_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        split.setSizes([540, 220])
        self._split = split
        layout.addWidget(split, stretch=1)

        self._build_residual_table_structure()
        self.resid_table.itemChanged.connect(self._on_bead_item_changed)

        self.plot.scene().sigMouseMoved.connect(self._on_hover)
        self._draw()
        self._refresh_readouts()

    def _build_plot_datasets(self):
        if self._single_channel is not None:
            cur = self._current[0]
            return {
                "beads": {
                    "label": f"{cur['ref_name']} beads",
                    "color": "#1f77b4",
                    "xyz_nm": np.asarray(cur["ref_xyz"], dtype=np.float64),
                    "bead_ids": np.asarray(cur["bead_ids"]),
                    "symbol": "o",
                }
            }
        cmap = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        reference = self._current[0]
        datasets = {
            "reference_original": {
                "label": f"{reference['ref_name']} original",
                "color": cmap[0],
                "xyz_nm": np.asarray(reference["ref_xyz"], dtype=np.float64),
                "bead_ids": np.asarray(reference["bead_ids"]),
                "symbol": "o",
            }
        }
        for idx, result in enumerate(self._current, start=1):
            color = cmap[idx % len(cmap)]
            base = f"channel_{idx+1}"
            datasets[f"{base}_original"] = {
                "label": f"{result['mov_name']} original",
                "color": color,
                "xyz_nm": np.asarray(result["mov_xyz"], dtype=np.float64),
                "bead_ids": np.asarray(result["bead_ids"]),
                "symbol": "o",
            }
            datasets[f"{base}_aligned"] = {
                "label": f"{result['mov_name']} aligned",
                "color": color,
                "xyz_nm": np.asarray(result["aligned_all"], dtype=np.float64),
                "bead_ids": np.asarray(result["bead_ids"]),
                "symbol": "star",
            }
        return datasets

    # ------------------------------------------------------------------
    # Live re-fit on a bead subset
    # ------------------------------------------------------------------

    def _recompute_current(self):
        """(Re)fit each moving channel using only the currently-included beads,
        and record aligned positions + residuals for ALL matched beads so excluded
        beads still show how far they fall from the fit."""
        if self._single_channel is not None:
            sc = self._single_channel
            ids = np.asarray(sc["bead_ids"], dtype=np.uint32)
            excluded = np.asarray(sorted(self._excluded_gris), dtype=np.uint32)
            inc = ~np.isin(ids, excluded) if excluded.size else np.ones(ids.size, dtype=bool)
            self._current = [{
                "ref_name": sc["name"], "mov_name": sc["name"],
                "bead_ids": ids,
                "rids": list(sc.get("rids") or [str(int(g)) for g in ids]),
                "ref_xyz": np.asarray(sc["pos_nm"], dtype=np.float64),
                "drift": np.asarray(sc["drift_nm"], dtype=np.float64),
                "included": inc, "valid": True,
            }]
            return
        from minflux_viewer.core.bead_alignment import (
            MIN_BEADS_FOR_ALIGNMENT,
            align_channels_from_bead_positions,
            apply_homogeneous_transform,
        )

        excluded = np.asarray(sorted(self._excluded_gris), dtype=np.uint32)
        self._current = []
        for b in self._base:
            ids = b["bead_ids"]
            inc = ~np.isin(ids, excluded) if excluded.size else np.ones(ids.size, dtype=bool)
            cur = {
                "ref_name": b["ref_name"], "mov_name": b["mov_name"],
                "bead_ids": ids, "ref_xyz": b["ref_xyz"], "mov_xyz": b["mov_xyz"],
                "included": inc,
            }
            if int(inc.sum()) >= MIN_BEADS_FOR_ALIGNMENT:
                res = align_channels_from_bead_positions(
                    b["ref_name"], b["mov_name"], ids[inc],
                    b["ref_xyz"][inc], b["mov_xyz"][inc],
                    b["ref_counts"][inc], b["mov_counts"][inc],
                    transform_type=self._requested_transform_type)
                matrix = np.asarray(res.matrix_4x4, dtype=np.float64)
                cur["transform_type"] = res.transform_type
                cur["valid"] = True
            else:
                matrix = np.eye(4, dtype=np.float64)
                cur["transform_type"] = self._requested_transform_type
                cur["valid"] = False
            aligned_all = apply_homogeneous_transform(b["mov_xyz"], matrix)
            residual_all = aligned_all - b["ref_xyz"]
            cur["matrix_4x4"] = matrix
            cur["aligned_all"] = aligned_all
            cur["residual_all"] = residual_all
            cur["n_used"] = int(inc.sum())
            inc_res = residual_all[inc]
            if inc_res.shape[0] and cur["valid"]:
                cur["rmse_xy_nm"] = float(np.sqrt(np.mean(inc_res[:, 0] ** 2 + inc_res[:, 1] ** 2)))
                cur["rmse_z_nm"] = float(np.sqrt(np.mean(inc_res[:, 2] ** 2)))
            else:
                cur["rmse_xy_nm"] = cur["rmse_z_nm"] = float("nan")
            self._current.append(cur)

    def _recompute_and_refresh(self):
        self._recompute_current()
        self.datasets = self._build_plot_datasets()
        self._draw()
        self._refresh_readouts()
        self._writeback_selection()

    def _writeback_selection(self):
        """Persist the current bead selection to the owning MSR-reader dialog so a
        subsequent Align/Open uses it (matches the 'Show beads drift' selection)."""
        owner = self._owner
        if owner is None or not hasattr(owner, "_bead_unchecked_gris"):
            return
        try:
            combined = set(self._owner_prior_excluded) | set(int(g) for g in self._excluded_gris)
            owner._bead_unchecked_gris = combined
            if hasattr(owner, "log"):
                owner.log(f"[align] bead selection updated from result window: "
                          f"{len(combined)} bead(s) excluded "
                          f"{sorted(combined) if combined else ''}")
        except Exception:
            pass

    def _bead_label_points(self) -> dict:
        """{gri: reference-position (nm)} for the bead-ID labels (union over channels)."""
        pts: dict[int, np.ndarray] = {}
        for cur in self._current:
            for i, g in enumerate(cur["bead_ids"]):
                pts.setdefault(int(g), cur["ref_xyz"][i])
        return pts

    # ------------------------------------------------------------------
    # Readouts (banner + per-bead residual table)
    # ------------------------------------------------------------------

    def _current_reports(self) -> list[dict]:
        from minflux_viewer.core.bead_alignment import RMSE_WARN_NM

        reports = []
        for cur in self._current:
            warn = bool(cur["valid"]) and np.isfinite(cur["rmse_xy_nm"]) and cur["rmse_xy_nm"] > RMSE_WARN_NM
            reports.append({
                "reference": cur["ref_name"], "moving": cur["mov_name"],
                "n_beads": cur["n_used"], "transform_type": cur["transform_type"],
                "rmse_xy_nm": cur["rmse_xy_nm"], "rmse_z_nm": cur["rmse_z_nm"],
                "warn": warn, "valid": bool(cur["valid"]),
            })
        return reports

    def _refresh_readouts(self):
        if self._single_channel is not None:
            self._render_single_channel_banner()
            self._update_single_channel_table_values()
            return
        from minflux_viewer.core.bead_alignment import RMSE_WARN_NM

        reports = self._current_reports()
        self._render_quality_banner(reports, RMSE_WARN_NM)
        self._update_residual_table_values()

    def _render_single_channel_banner(self):
        """Banner above the table for the single-channel (no-alignment) view:
        bead count + drift (peak-to-peak) summary over the included beads."""
        cur = self._current[0]
        inc = np.asarray(cur["included"], dtype=bool)
        drift = np.asarray(cur["drift"], dtype=np.float64)
        n_total = int(inc.size)
        n_inc = int(inc.sum())
        lines = [
            f"<b>{cur['ref_name']}</b> — single channel, so <b>no alignment</b> is "
            f"needed. Showing {n_total} bead(s) and the data region; "
            f"{n_inc} bead(s) included in the drift summary."
        ]
        if n_inc:
            dxyz = np.sqrt((drift[inc] ** 2).sum(axis=1))
            med = float(np.median(dxyz))
            worst_i = int(np.argmax(dxyz))
            rids = cur.get("rids") or [str(int(g)) for g in cur["bead_ids"]]
            inc_idx = np.flatnonzero(inc)
            worst_rid = rids[inc_idx[worst_i]]
            lines.append(
                f"Per-bead drift (peak-to-peak over the acquisition): "
                f"median&nbsp;Δxyz&nbsp;=&nbsp;{med:.1f}&nbsp;nm, "
                f"max&nbsp;=&nbsp;{float(dxyz[worst_i]):.1f}&nbsp;nm (bead&nbsp;{worst_rid})."
            )
        self.quality_banner.setStyleSheet(
            "QLabel { background:#f4f7fb; border:1px solid #99b; "
            "border-radius:4px; padding:6px; }"
        )
        self.quality_banner.setText("<br>".join(lines))

    def _update_single_channel_table_values(self):
        from PyQt6.QtGui import QBrush, QColor

        cur = self._current[0]
        drift = np.asarray(cur["drift"], dtype=np.float64)
        excluded_bg = QColor("#eeeeee")
        normal_bg = QBrush(Qt.GlobalColor.transparent)
        grey_fg = QBrush(QColor("#999999"))
        black_fg = QBrush(QColor("#202020"))
        inc_mask = np.asarray(cur["included"], dtype=bool)
        dxyz_all = np.sqrt((drift ** 2).sum(axis=1))
        med = float(np.median(dxyz_all[inc_mask])) if inc_mask.any() else 0.0
        self._suspend_bead_signals = True
        for row, (ci, bi, gri) in enumerate(self._row_order):
            dx, dy, dz = drift[bi]
            dxy = float(np.hypot(dx, dy))
            dxyz = float(np.hypot(np.hypot(dx, dy), dz))
            included = bool(cur["included"][bi])
            # Δx Δy Δz Δxy Δxyz, then the (empty) resid x/y/z/|resid xy| columns.
            texts = [f"{dx:.0f}", f"{dy:.0f}", f"{dz:.0f}", f"{dxy:.0f}", f"{dxyz:.0f}",
                     "", "", "", ""]
            first_num = self._bead_col + 1
            for k, text in enumerate(texts):
                item = self.resid_table.item(row, first_num + k)
                if item is not None:
                    item.setText(text)
            comment = ""
            if included and med > 0 and dxyz > 2.0 * med:
                comment = "high drift (outlier)"
            comment_item = self.resid_table.item(row, self._comment_col)
            if comment_item is not None:
                comment_item.setText(comment)
            bead_item = self.resid_table.item(row, self._bead_col)
            if bead_item is not None:
                bead_item.setCheckState(
                    Qt.CheckState.Checked if included else Qt.CheckState.Unchecked)
            for c in range(self.resid_table.columnCount()):
                cell = self.resid_table.item(row, c)
                if cell is None:
                    continue
                cell.setForeground(grey_fg if not included else black_fg)
                cell.setBackground(excluded_bg if not included else normal_bg)
        self._suspend_bead_signals = False

    def _render_quality_banner(self, reports, warn_nm):
        lines = []
        any_warn = False
        any_invalid = False
        for rep in reports:
            if not rep["valid"]:
                any_invalid = True
                lines.append(
                    f"⛔ <b>{rep['moving']}</b> → {rep['reference']}: only {rep['n_beads']} "
                    f"bead(s) selected — need at least 2 to align. Re-check beads below.")
                continue
            warn = bool(rep["warn"])
            any_warn = any_warn or warn
            icon = "⚠" if warn else "✓"
            note = ""
            if rep["n_beads"] <= 3:
                note = (f" &nbsp;<i>(only {rep['n_beads']} bead(s) → {rep['transform_type']})</i>")
            lines.append(
                f"{icon} <b>{rep['moving']}</b> → {rep['reference']}: "
                f"{rep['n_beads']} beads, RMSE&nbsp;XY&nbsp;=&nbsp;{rep['rmse_xy_nm']:.1f}&nbsp;nm, "
                f"Z&nbsp;=&nbsp;{rep['rmse_z_nm']:.1f}&nbsp;nm{note}"
            )
        if any_warn:
            lines.append(
                f"<span style='color:#a00'>Fiducial registration is poor "
                f"(RMSE &gt; {warn_nm:.0f}&nbsp;nm, well above MINFLUX localization "
                f"precision). The beads disagree among themselves, so no single transform "
                f"can register the channels accurately. If your sample contains a common "
                f"feature/fiducial, try the <b>data centroid</b> alignment; otherwise untick "
                f"the largest-residual beads below.</span>"
            )
        if any_warn or any_invalid:
            self.quality_banner.setStyleSheet(
                "QLabel { background:#fff4f4; border:1px solid #d88; "
                "border-radius:4px; padding:6px; }"
            )
        else:
            self.quality_banner.setStyleSheet(
                "QLabel { background:#f2fbf2; border:1px solid #8c8; "
                "border-radius:4px; padding:6px; }"
            )
        self.quality_banner.setText("<br>".join(lines))

    def _build_residual_table_structure(self):
        """Build the residual-table rows + per-bead checkboxes once, in a fixed order
        (worst residual at the initial full-bead fit first). Values are filled by
        _update_residual_table_values; row order stays stable so ticking a checkbox
        does not make the row jump."""
        if self._single_channel is not None:
            self._multi = False
            self._bead_col = 0
            cur = self._current[0]
            drift = np.asarray(cur["drift"], dtype=np.float64)
            dxyz = np.sqrt((drift ** 2).sum(axis=1))
            self._row_order = [
                (0, int(bi), int(cur["bead_ids"][bi])) for bi in np.argsort(-dxyz)]
            rids = cur.get("rids") or [str(int(g)) for g in cur["bead_ids"]]
            headers = ["Bead", "Δx", "Δy", "Δz", "Δxy", "Δxyz",
                       "resid x", "resid y", "resid z", "|resid xy|", "Comment"]
            self._comment_col = len(headers) - 1
            self.resid_caption.setText(
                "Per-bead drift (peak-to-peak per axis over the acquisition), nm. "
                "Δxy/Δxyz combine the axes; the resid columns are N/A (single channel, "
                "no alignment). Untick a bead to drop it from the drift summary.")
            self.resid_table.setToolTip(
                "Δ = total drift distance of the bead per axis (max − min over the "
                "acquisition), nm; Δxy/Δxyz combine axes.\nresid columns are empty "
                "(no channel alignment for a single channel).")
            self._suspend_bead_signals = True
            self.resid_table.clear()
            self.resid_table.setColumnCount(len(headers))
            self.resid_table.setHorizontalHeaderLabels(headers)
            self.resid_table.setRowCount(len(self._row_order))
            for row, (_ci, bi, gri) in enumerate(self._row_order):
                bead_item = QTableWidgetItem(str(rids[bi]))
                bead_item.setFlags(bead_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                bead_item.setCheckState(Qt.CheckState.Checked)
                bead_item.setData(Qt.ItemDataRole.UserRole, gri)
                self.resid_table.setItem(row, 0, bead_item)
                for c in range(1, len(headers)):
                    cell = QTableWidgetItem("")
                    align = (Qt.AlignmentFlag.AlignLeft if c == self._comment_col
                             else Qt.AlignmentFlag.AlignRight)
                    cell.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                    self.resid_table.setItem(row, c, cell)
            header = self.resid_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
            self._suspend_bead_signals = False
            return
        self._multi = len(self._current) > 1
        # fixed row order: (channel_index, bead_index, gri), worst initial |resid| first
        order = []
        for ci, cur in enumerate(self._current):
            rxy = np.hypot(cur["residual_all"][:, 0], cur["residual_all"][:, 1])
            for bi in np.argsort(-rxy):
                order.append((ci, int(bi), int(cur["bead_ids"][bi])))
        self._row_order = order

        headers = (["Moving"] if self._multi else []) + [
            "Bead", "Δx", "Δy", "Δz", "resid x", "resid y", "resid z", "|resid xy|", "Comment"]
        self._comment_col = len(headers) - 1
        self._suspend_bead_signals = True
        self.resid_table.clear()
        self.resid_table.setColumnCount(len(headers))
        self.resid_table.setHorizontalHeaderLabels(headers)
        self.resid_table.setRowCount(len(order))
        bead_col = 1 if self._multi else 0
        for row, (ci, bi, gri) in enumerate(order):
            col = 0
            if self._multi:
                mv = QTableWidgetItem(str(self._current[ci]["mov_name"]))
                self.resid_table.setItem(row, col, mv)
                col += 1
            bead_item = QTableWidgetItem(str(gri))
            bead_item.setFlags(bead_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            bead_item.setCheckState(Qt.CheckState.Checked)
            bead_item.setData(Qt.ItemDataRole.UserRole, gri)
            self.resid_table.setItem(row, col, bead_item)
            for c in range(col + 1, len(headers)):
                cell = QTableWidgetItem("")
                if c == self._comment_col:  # wide text column, left-aligned
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                else:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.resid_table.setItem(row, c, cell)
        header = self.resid_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # The numeric columns hug their (2-3 digit) content; the wide "Comment" column
        # takes the remaining width so it, not "|resid xy|", absorbs the slack.
        header.setStretchLastSection(True)
        self._bead_col = bead_col
        self._suspend_bead_signals = False

    def _update_residual_table_values(self):
        from PyQt6.QtGui import QBrush, QColor

        from minflux_viewer.core.bead_alignment import RMSE_WARN_NM, bead_residual_comment

        excluded_bg = QColor("#eeeeee")
        warn_bg = QColor("#fbeaea")
        normal_bg = QBrush(Qt.GlobalColor.transparent)
        grey_fg = QBrush(QColor("#999999"))
        black_fg = QBrush(QColor("#202020"))
        self._suspend_bead_signals = True
        for row, (ci, bi, gri) in enumerate(self._row_order):
            cur = self._current[ci]
            off = cur["ref_xyz"][bi] - cur["mov_xyz"][bi]
            res = cur["residual_all"][bi]
            rxy = float(np.hypot(res[0], res[1]))
            included = bool(cur["included"][bi])
            texts = [f"{off[0]:.0f}", f"{off[1]:.0f}", f"{off[2]:.0f}",
                     f"{res[0]:.0f}", f"{res[1]:.0f}", f"{res[2]:.0f}", f"{rxy:.0f}"]
            first_num = self._bead_col + 1
            for k, text in enumerate(texts):
                item = self.resid_table.item(row, first_num + k)
                if item is not None:
                    item.setText(text)
            # evaluation comment (helper for keep/exclude decisions)
            comment = bead_residual_comment(
                off, res, included=included, rmse_xy_nm=cur.get("rmse_xy_nm"))
            comment_item = self.resid_table.item(row, self._comment_col)
            if comment_item is not None:
                comment_item.setText(comment)
            # keep the Bead checkbox in sync (covers duplicate-gri rows on multi-channel)
            bead_item = self.resid_table.item(row, self._bead_col)
            if bead_item is not None:
                bead_item.setCheckState(
                    Qt.CheckState.Checked if included else Qt.CheckState.Unchecked)
            # row styling: greyed if excluded; light red when an included bead's
            # residual is large (helps spot the outliers to untick)
            highlight = included and rxy > RMSE_WARN_NM
            for c in range(self.resid_table.columnCount()):
                cell = self.resid_table.item(row, c)
                if cell is None:
                    continue
                cell.setForeground(grey_fg if not included else black_fg)
                cell.setBackground(excluded_bg if not included else (warn_bg if highlight else normal_bg))
        self._suspend_bead_signals = False

    def _on_bead_item_changed(self, item):
        if self._suspend_bead_signals:
            return
        gri = item.data(Qt.ItemDataRole.UserRole)
        if gri is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        if checked:
            self._excluded_gris.discard(int(gri))
        else:
            self._excluded_gris.add(int(gri))
        self._recompute_and_refresh()

    def closeEvent(self, event):
        # Stop the live-refit callback and disconnect the table signal before Qt tears
        # the widgets down; otherwise itemChanged emitted during destruction re-enters
        # the recompute/redraw on already-dying C++ objects and hangs.
        self._suspend_bead_signals = True
        try:
            self.resid_table.itemChanged.disconnect(self._on_bead_item_changed)
        except (TypeError, RuntimeError):
            pass
        from minflux_viewer.ui.common.qt_lifecycle import close_plot_widgets

        close_plot_widgets(self.plot)
        super().closeEvent(event)

    @staticmethod
    def _to_um(values_nm):
        return np.asarray(values_nm, dtype=np.float64) / 1000.0

    def _draw(self):
        import pyqtgraph as pg

        if self._view_mode == "3D":
            self._draw_3d()
            return
        self._stack.setCurrentWidget(self.plot)
        self.plot.clear()
        if getattr(self.plot.plotItem, "legend", None) is not None:
            self.plot.plotItem.legend.clear()
        self._items = {}
        x_idx, y_idx = self._axis_indices
        apply_spatial_y_direction(
            self.plot,
            vertical_coordinate="XYZ"[y_idx],
            prefs=self._prefs,
            preference_key="scatter_xy_origin",
        )
        for key, meta in self.datasets.items():
            xyz = self._to_um(meta["xyz_nm"])
            # Excluded beads stay visible but faint, so the user can see which beads
            # were taken out of the fit (per-point brush/pen alpha).
            excluded = self._excluded_bead_mask(meta["bead_ids"])
            inc_brush = pg.mkBrush(meta["color"] + "80")
            exc_brush = pg.mkBrush(meta["color"] + "1e")
            inc_pen = pg.mkPen(meta["color"], width=1)
            exc_pen = pg.mkPen(meta["color"] + "40", width=1)
            brushes = [exc_brush if ex else inc_brush for ex in excluded]
            pens = [exc_pen if ex else inc_pen for ex in excluded]
            item = pg.ScatterPlotItem(
                xyz[:, x_idx],
                xyz[:, y_idx],
                size=9,
                symbol=meta["symbol"],
                brush=brushes,
                pen=pens,
                name=meta["label"],
            )
            item.setVisible(self.visibility[key].isChecked())
            self.plot.addItem(item)
            self._items[key] = item
        self._draw_bead_id_labels()
        self._draw_data_range_box()
        labels = self._axis_labels()
        self.plot.setLabel("bottom", labels[0], units="µm")
        self.plot.setLabel("left", labels[1], units="µm")
        self.plot.setTitle(
            "Bead positions and the data region" if self._single_channel is not None
            else "Bead positions before and after channel alignment")
        self._set_equal_range()

    def _excluded_bead_mask(self, bead_ids) -> np.ndarray:
        """Boolean mask (per bead in *bead_ids*) of beads excluded from the fit."""
        gris = np.asarray(bead_ids, dtype=np.int64)
        if not self._excluded_gris:
            return np.zeros(gris.size, dtype=bool)
        return np.isin(gris, np.asarray(sorted(self._excluded_gris), dtype=np.int64))

    def _draw_bead_id_labels(self):
        """Draw the gri bead-ID next to each bead (reference position), like the
        ROI-ID labels in the ROI Manager. self.plot.clear() in _draw() removes the
        previous labels, so they are simply re-added each draw."""
        import pyqtgraph as pg

        x_idx, y_idx = self._axis_indices
        excluded = self._excluded_gris
        for gri, xyz_nm in self._bead_label_points().items():
            xyz = self._to_um(xyz_nm)
            if not (np.isfinite(xyz[x_idx]) and np.isfinite(xyz[y_idx])):
                continue
            # dim the label of an excluded bead to match its faint markers
            color = (170, 170, 170) if int(gri) in excluded else (40, 40, 40)
            text = pg.TextItem(str(gri), color=color, anchor=(-0.15, 1.15))
            text.setPos(float(xyz[x_idx]), float(xyz[y_idx]))
            self.plot.addItem(text)

    def _data_box_xy(self):
        """(x0, y0, x1, y1) of the data-range box for the current view, or None."""
        if self.data_bounds_nm is None:
            return None
        mins = self._to_um(self.data_bounds_nm[0])
        maxs = self._to_um(self.data_bounds_nm[1])
        x_idx, y_idx = self._axis_indices
        if x_idx >= len(mins) or y_idx >= len(mins):
            return None
        return (float(mins[x_idx]), float(mins[y_idx]),
                float(maxs[x_idx]), float(maxs[y_idx]))

    def _draw_data_range_box(self):
        import pyqtgraph as pg
        from PyQt6.QtCore import QRectF
        from PyQt6.QtWidgets import QGraphicsRectItem

        box = self._data_box_xy()
        if box is None:
            return
        x0, y0, x1, y1 = box
        # The MBM fiducial beads can be spread across the whole stage while the
        # measured data region is a small patch, so a thin outline is easy to miss.
        # Draw a translucent fill (behind the beads) + a label so the region reads
        # clearly as a region.
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        fill = QGraphicsRectItem(QRectF(lo_x, lo_y, hi_x - lo_x, hi_y - lo_y))
        fill.setPen(pg.mkPen(None))
        fill.setBrush(pg.mkBrush(230, 200, 0, 60))
        fill.setZValue(-10)                       # behind the bead markers
        self.plot.addItem(fill)
        self.plot.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen((230, 200, 0), width=2),
            name="data region",
        )
        label = pg.TextItem("data region", color=(150, 125, 0), anchor=(0.5, 1.25))
        label.setPos((lo_x + hi_x) / 2.0, hi_y)
        self.plot.addItem(label)

    def _update_visibility(self):
        for key, item in self._items.items():
            item.setVisible(self.visibility[key].isChecked())
        for key, item in self._items_3d.items():
            item.setVisible(self.visibility[key].isChecked())
        if self._view_mode == "3D":
            self._configure_3d_reference_items()

    def _axis_labels(self):
        mode = self._view_mode.upper()
        if mode == "XZ":
            return "x", "z"
        if mode == "YZ":
            return "y", "z"
        return "x", "y"

    def _set_view_mode(self, mode: str) -> None:
        mode = str(mode).upper()
        if mode == "XZ":
            self._axis_indices = (0, 2)
        elif mode == "YZ":
            self._axis_indices = (1, 2)
        elif mode == "3D":
            self._axis_indices = (0, 1)
        else:
            mode = "XY"
            self._axis_indices = (0, 1)
        self._view_mode = mode
        self._draw()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        view_menu = menu.addMenu("View")
        for mode in ("XY", "XZ", "YZ", "3D"):
            action = view_menu.addAction(mode)
            action.setCheckable(True)
            action.setChecked(mode == self._view_mode)
            action.triggered.connect(lambda _checked=False, value=mode: self._set_view_mode(value))
        if self._view_mode == "3D":
            menu.addSeparator()
            axis_action = menu.addAction("Axis")
            axis_action.setCheckable(True)
            axis_action.setChecked(self._show_3d_axis)
            axis_action.triggered.connect(self._set_3d_axis_visible)

            box_action = menu.addAction("Bounding Box")
            box_action.setCheckable(True)
            box_action.setChecked(self._show_3d_bounding_box)
            box_action.triggered.connect(self._set_3d_bounding_box_visible)
        menu.addSeparator()
        menu.addAction("Reset View", self._reset_view)
        sender = self.sender()
        if isinstance(sender, QWidget):
            menu.exec(sender.mapToGlobal(pos))
        else:
            menu.exec(self.mapToGlobal(pos))

    def _reset_view(self) -> None:
        if self._view_mode == "3D":
            self._reset_3d_camera()
        else:
            self._set_equal_range()

    def _set_equal_range(self):
        visible_xyz = [meta["xyz_nm"] for key, meta in self.datasets.items() if self.visibility[key].isChecked()]
        if not visible_xyz:
            visible_xyz = [meta["xyz_nm"] for meta in self.datasets.values()]
        xyz = self._to_um(np.vstack(visible_xyz))
        x_idx, y_idx = self._axis_indices
        xy = xyz[:, [x_idx, y_idx]]
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        box = self._data_box_xy()
        if box is not None:
            corners = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float64)
            xy = np.vstack([xy, corners]) if xy.size else corners
        if xy.size == 0:
            return
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        centers = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
        self.plot.setXRange(centers[0] - radius, centers[0] + radius, padding=0)
        self.plot.setYRange(centers[1] - radius, centers[1] + radius, padding=0)

    def _ensure_3d_built(self) -> bool:
        if self._view_3d is not None:
            return True
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:
            QMessageBox.warning(self, "3D view", f"3D view is unavailable:\n{exc}")
            return False
        view = gl.GLViewWidget()
        view.setBackgroundColor("w")
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(self._show_context_menu)
        grid = gl.GLGridItem()
        grid.setSize(1000, 1000)
        grid.setSpacing(100, 100)
        try:
            grid.setColor((160, 160, 160, 80))
        except Exception:
            pass
        view.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(500, 500, 500)
        view.addItem(axis)
        try:
            axis.setVisible(False)
        except Exception:
            pass
        self._view_3d = view
        self._grid_3d = grid
        self._axis_3d = axis
        self._items_3d = {}
        self._stack.addWidget(view)
        return True

    @staticmethod
    def _nice_grid_spacing_um(span_um: float, target_lines: int = 8) -> float:
        raw = float(span_um) / max(int(target_lines), 1)
        if not np.isfinite(raw) or raw <= 0:
            return 1.0
        mag = 10.0 ** np.floor(np.log10(raw))
        norm = raw / mag
        step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
        return float(max(step, 1e-6))

    def _plot_xyz_um(self, *, visible_only: bool, include_data_bounds: bool) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key, meta in self.datasets.items():
            if visible_only and not self.visibility[key].isChecked():
                continue
            xyz = self._to_um(meta["xyz_nm"])
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            if xyz.size:
                parts.append(xyz[:, :3])
        if include_data_bounds and self.data_bounds_nm is not None:
            bounds = np.vstack([
                self._to_um(self.data_bounds_nm[0]),
                self._to_um(self.data_bounds_nm[1]),
            ])
            bounds = bounds[np.all(np.isfinite(bounds), axis=1)]
            if bounds.size:
                parts.append(bounds[:, :3])
        if not parts and visible_only:
            return self._plot_xyz_um(visible_only=False, include_data_bounds=include_data_bounds)
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(parts).astype(np.float64, copy=False)

    @staticmethod
    def _visible_3d_rgba(color) -> tuple[float, float, float, float]:
        rgb = np.array([color.redF(), color.greenF(), color.blueF()], dtype=np.float64)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        if luminance > 0.58:
            rgb *= np.clip(0.58 / luminance, 0.55, 1.0)
        return float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0

    def _set_3d_axis_visible(self, checked: bool) -> None:
        self._show_3d_axis = bool(checked)
        if self._view_mode == "3D":
            self._configure_3d_reference_items()

    def _set_3d_bounding_box_visible(self, checked: bool) -> None:
        self._show_3d_bounding_box = bool(checked)
        if self._view_mode == "3D":
            self._configure_3d_reference_items()

    @staticmethod
    def _fmt_tick_nm(value_um: float) -> str:
        value_nm = float(value_um) * 1000.0
        if abs(value_nm) >= 1000 or float(value_nm).is_integer():
            return f"{value_nm:.0f}"
        return f"{value_nm:.3g}"

    @staticmethod
    def _tick_values_um(lo: float, hi: float, *, max_ticks: int = 5) -> list[float]:
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return []
        step = AlignmentPlotWindow._nice_grid_spacing_um(hi - lo, target_lines=max_ticks)
        start = np.ceil(lo / step) * step
        values: list[float] = []
        value = start
        while value <= hi + step * 1e-9 and len(values) < max_ticks + 2:
            values.append(float(value))
            value += step
        return values[:max_ticks]

    @staticmethod
    def _expanded_bounds_um(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
            return None
        xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1), :3]
        if xyz.size == 0:
            return None
        mins = np.nanmin(xyz, axis=0)
        maxs = np.nanmax(xyz, axis=0)
        spans = maxs - mins
        max_span = max(float(np.nanmax(spans)), 1.0)
        for dim in range(3):
            if spans[dim] <= 0:
                mins[dim] -= max_span * 0.05
                maxs[dim] += max_span * 0.05
        spans = np.maximum(maxs - mins, 1e-6)
        return mins, maxs, spans

    def _bounding_box_bounds_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.data_bounds_nm is not None:
            bounds = np.vstack([
                self._to_um(self.data_bounds_nm[0]),
                self._to_um(self.data_bounds_nm[1]),
            ])
            return self._expanded_bounds_um(bounds)
        return self._expanded_bounds_um(
            self._plot_xyz_um(visible_only=True, include_data_bounds=False)
        )

    @staticmethod
    def _reference_color() -> tuple[float, float, float, float]:
        return 0.35, 0.35, 0.35, 0.68

    @staticmethod
    def _text_color() -> tuple[int, int, int, int]:
        return 35, 35, 35, 230

    def _clear_3d_items(self, items: list) -> None:
        if self._view_3d is None:
            items.clear()
            return
        for item in list(items):
            try:
                self._view_3d.removeItem(item)
            except Exception:
                pass
        items.clear()

    def _add_3d_line(
        self,
        items: list,
        start: np.ndarray,
        end: np.ndarray,
        *,
        color: tuple[float, float, float, float],
        width: float = 1.2,
    ) -> None:
        if self._view_3d is None:
            return
        try:
            import pyqtgraph.opengl as gl

            item = gl.GLLinePlotItem(
                pos=np.vstack([start, end]).astype(np.float32),
                color=color,
                width=width,
                antialias=True,
            )
            self._view_3d.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _add_3d_text(self, items: list, pos: np.ndarray, text: str, *, label: bool = False) -> None:
        if self._view_3d is None:
            return
        try:
            import pyqtgraph.opengl as gl

            item = gl.GLTextItem(
                pos=np.asarray(pos, dtype=np.float64),
                text=str(text),
                color=self._text_color(),
                font=QFont("Helvetica", 11 if label else 8),
                glOptions="translucent",
            )
            self._view_3d.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _draw_3d_axis(self, mins: np.ndarray, maxs: np.ndarray, spans: np.ndarray) -> None:
        if not self._show_3d_axis:
            return
        origin = mins.copy()
        pad = max(float(np.max(spans)) * 0.04, 0.05)
        axis_defs = (
            (0, np.array([maxs[0], origin[1], origin[2]]), (0.9, 0.15, 0.15, 0.95), "X (nm)"),
            (1, np.array([origin[0], maxs[1], origin[2]]), (0.15, 0.7, 0.25, 0.95), "Y (nm)"),
            (2, np.array([origin[0], origin[1], maxs[2]]), (0.15, 0.35, 0.95, 0.95), "Z (nm)"),
        )
        for dim, end, color, label in axis_defs:
            self._add_3d_line(self._axis_3d_items, origin, end, color=color, width=2.0)
            label_pos = end.copy()
            label_pos[dim] += pad
            self._add_3d_text(self._axis_3d_items, label_pos, label, label=True)
            for tick in self._tick_values_um(float(mins[dim]), float(maxs[dim])):
                tick_pos = origin.copy()
                tick_pos[dim] = tick
                tick_end = tick_pos.copy()
                side_dim = 1 if dim != 1 else 0
                tick_end[side_dim] += pad * 0.35
                self._add_3d_line(self._axis_3d_items, tick_pos, tick_end, color=color, width=1.0)
                text_pos = tick_end.copy()
                text_pos[side_dim] += pad * 0.12
                self._add_3d_text(self._axis_3d_items, text_pos, self._fmt_tick_nm(tick))

    def _draw_3d_bounding_box(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        if not self._show_3d_bounding_box:
            return
        x0, y0, z0 = mins
        x1, y1, z1 = maxs
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float64)
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        color = self._reference_color()
        for a, b in edges:
            self._add_3d_line(self._box_3d_items, corners[a], corners[b], color=color, width=1.4)

    def _configure_3d_reference_items(self) -> None:
        if self._view_3d is None:
            return
        self._clear_3d_items(self._axis_3d_items)
        self._clear_3d_items(self._box_3d_items)
        xyz = self._plot_xyz_um(visible_only=True, include_data_bounds=True)
        bounds = self._expanded_bounds_um(xyz)
        if bounds is None:
            return
        mins, maxs, spans = bounds
        centre = (mins + maxs) / 2.0
        xy_size = max(float(np.max(spans[:2])) * 1.15, 1.0)
        spacing = self._nice_grid_spacing_um(xy_size)
        z_floor = float(min(mins[2], 0.0)) if np.isfinite(mins[2]) else 0.0

        if self._grid_3d is not None:
            try:
                self._grid_3d.setSize(xy_size, xy_size)
                self._grid_3d.setSpacing(spacing, spacing)
                self._grid_3d.resetTransform()
                self._grid_3d.translate(float(centre[0]), float(centre[1]), z_floor)
            except Exception:
                pass
        if self._axis_3d is not None:
            try:
                axis_size = max(min(xy_size * 0.25, xy_size), 0.5, float(spans[2]) * 1.1)
                self._axis_3d.setSize(axis_size, axis_size, axis_size)
                self._axis_3d.resetTransform()
                self._axis_3d.translate(float(mins[0]), float(mins[1]), z_floor)
                self._axis_3d.setVisible(False)
            except Exception:
                pass
        self._draw_3d_axis(mins, maxs, spans)
        box_bounds = self._bounding_box_bounds_um()
        if box_bounds is not None:
            box_mins, box_maxs, _box_spans = box_bounds
            self._draw_3d_bounding_box(box_mins, box_maxs)

    def _draw_3d(self) -> None:
        if not self._ensure_3d_built() or self._view_3d is None:
            self._view_mode = "XY"
            self._axis_indices = (0, 1)
            self._draw()
            return
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl

        self._stack.setCurrentWidget(self._view_3d)
        for item in list(self._items_3d.values()):
            try:
                self._view_3d.removeItem(item)
            except Exception:
                pass
        self._items_3d = {}
        self._clear_3d_items(self._axis_3d_items)
        self._clear_3d_items(self._box_3d_items)
        self._clear_3d_items(self._label_items_3d)
        for key, meta in self.datasets.items():
            xyz = self._to_um(meta["xyz_nm"])
            finite = np.all(np.isfinite(xyz), axis=1)
            xyz = xyz[finite]
            if xyz.size == 0:
                continue
            color = pg.mkColor(meta["color"])
            rgba = np.tile(self._visible_3d_rgba(color), (xyz.shape[0], 1)).astype(np.float32)
            # excluded beads stay visible but faint (low alpha)
            excluded = self._excluded_bead_mask(np.asarray(meta["bead_ids"])[finite])
            rgba[excluded, 3] = 0.12
            point_size = 10.5 if meta.get("symbol") == "star" else 8.5
            item = gl.GLScatterPlotItem(pxMode=True, size=point_size)
            try:
                item.setGLOptions("translucent")
            except Exception:
                pass
            item.setData(
                pos=xyz.astype(np.float32, copy=False),
                color=rgba,
                size=point_size,
                pxMode=True,
            )
            item.setVisible(self.visibility[key].isChecked())
            self._view_3d.addItem(item)
            self._items_3d[key] = item
        for gri, xyz_nm in self._bead_label_points().items():
            xyz = self._to_um(xyz_nm)
            if np.all(np.isfinite(xyz[:3])):
                self._add_3d_text(self._label_items_3d, xyz[:3], str(gri))
        self._configure_3d_reference_items()
        self._reset_3d_camera()
        self.hover_label.setText("3D view: use mouse to rotate, pan, and zoom. Coordinates are displayed in µm.")

    def _reset_3d_camera(self) -> None:
        if self._view_3d is None:
            return
        xyz = self._plot_xyz_um(visible_only=True, include_data_bounds=True)
        if xyz.size == 0:
            return
        centre = xyz.mean(axis=0)
        span = xyz.max(axis=0) - xyz.min(axis=0)
        extent = float(np.linalg.norm(span))
        if not np.isfinite(extent) or extent <= 0:
            extent = 10.0
        import pyqtgraph as pg
        self._view_3d.opts["center"] = pg.Vector(*centre)
        self._view_3d.setCameraPosition(distance=max(extent * 1.5, 5.0))

    def _best_hover_target(self, scene_pos):
        if self._view_mode == "3D":
            return None
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        point = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        mouse_xy = np.array([point.x(), point.y()], dtype=np.float64)
        view_range = self.plot.plotItem.vb.viewRange()
        sx = max(abs(view_range[0][1] - view_range[0][0]), 1.0)
        sy = max(abs(view_range[1][1] - view_range[1][0]), 1.0)
        best = None
        x_idx, y_idx = self._axis_indices
        for key in self.datasets:
            if not self.visibility[key].isChecked():
                continue
            xyz = self._to_um(self.datasets[key]["xyz_nm"])[:, [x_idx, y_idx]]
            scaled = np.column_stack([(xyz[:, 0] - mouse_xy[0]) / sx, (xyz[:, 1] - mouse_xy[1]) / sy])
            dist2 = np.sum(scaled ** 2, axis=1)
            idx = int(np.argmin(dist2))
            best_dist2 = float(dist2[idx])
            if best is None or best_dist2 < best["dist2"]:
                best = {"key": key, "index": idx, "dist2": best_dist2}
        return best

    def _on_hover(self, scene_pos):
        target = self._best_hover_target(scene_pos)
        if target is None or target["dist2"] > 0.0004:
            if self._view_mode != "3D":
                self.hover_label.setText("Hover near a point to see channel and coordinates (µm).")
            return
        key = target["key"]
        idx = target["index"]
        xyz = self._to_um(self.datasets[key]["xyz_nm"][idx])
        bead_id = int(self.datasets[key]["bead_ids"][idx])
        self.hover_label.setText(
            f"bead ID {bead_id} | {self.datasets[key]['label']} | "
            f"x {xyz[0]:.4f} µm, y {xyz[1]:.4f} µm, z {xyz[2]:.4f} µm"
        )
