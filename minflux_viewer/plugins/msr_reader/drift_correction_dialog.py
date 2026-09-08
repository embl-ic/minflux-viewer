"""
Preview and confirm a post-hoc MBM drift correction.

The correction itself is :mod:`minflux_viewer.msr.mbm_drift` — reads each row's
uncorrected ``lnc``, subtracts the mean drift of the **selected** fiducial beads,
and writes ``loc``.  This dialog is the decision point in front of it: it plots
the drift that would be removed, reports how far that moves the data from where
it sits now, and only rewrites ``loc`` if the user says so.

It is **modal** by the project convention — the caller needs an answer before
continuing — and shows one tab per dataset that has bead data.  The X/Y/Z curves
are the correction to be applied; the faint curves behind them are the individual
beads it is the mean of, so a fiducial that disagrees with the rest is visible
*before* it is baked into the coordinates.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...msr.mbm_drift import (
    DriftCorrectionError,
    apply_correction,
    corrected_loc,
    drift_curve,
    per_bead_curves,
    usable_bead_gris,
)

#: X / Y / Z curve colours for the drift that will be applied.
_AXIS_PENS = ((220, 40, 40), (30, 140, 40), (40, 80, 220))
_AXIS_NAMES = ("X", "Y", "Z")
#: Faint grey for the individual bead traces the mean is taken over.
_BEAD_PEN = (170, 170, 170, 110)
_MAX_CURVE_PTS = 3000


class DriftCorrectionDialog(QDialog):
    """Show the drift the selected beads define, then optionally apply it.

    *entries* is one ``{"name", "mfx", "points"}`` per dataset; *gris* the bead
    selection.  On accept, every entry's ``mfx["loc"]`` is rewritten **in place**
    — the parse result and the reader's ``mfx_map`` hold the same array objects,
    so "Open in MINFLUX viewer" then loads the corrected coordinates.
    """

    def __init__(self, entries: list[dict], gris, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Drift correction with selected beads")
        self.setMinimumSize(820, 560)
        self.resize(980, 700)

        self._entries = list(entries or [])
        self._gris = sorted(int(g) for g in gris)
        self.applied: list[dict] = []

        root = QVBoxLayout(self)
        root.addWidget(self._intro_label())

        self._tabs = QTabWidget()
        self._plots: list = []
        any_ok = False
        for entry in self._entries:
            page, ok = self._build_page(entry)
            self._tabs.addTab(page, str(entry.get("name", "dataset")))
            any_ok = any_ok or ok
        root.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox()
        apply_btn = buttons.addButton("Apply to 'loc'", QDialogButtonBox.ButtonRole.AcceptRole)
        apply_btn.setToolTip(
            "Overwrite the 'loc' coordinates of every dataset above with the "
            "drift-corrected positions. The uncorrected 'lnc' is left untouched, "
            "so this can be re-applied with a different bead selection.")
        apply_btn.setEnabled(any_ok)
        cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        cancel_btn.setToolTip("Close without changing any coordinates.")
        apply_btn.clicked.connect(self._apply)
        cancel_btn.clicked.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    def _intro_label(self) -> QLabel:
        ids = ", ".join(str(g) for g in self._gris) or "(none)"
        lbl = QLabel(
            f"Drift computed from the <b>{len(self._gris)} selected bead(s)</b>: {ids}."
            "<br>Each dataset's corrected position is its <b>uncorrected</b> "
            "<tt>lnc</tt> minus the mean drift of these beads. Coloured curves are "
            "the correction to be applied; faint grey curves are the individual "
            "beads it averages."
            "<br><span style='color:#666'>The absolute offset is a convention — "
            "the whole dataset may shift by a few nm relative to an Imspector "
            "export, which changes no measured distance.</span>")
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        return lbl

    def _build_page(self, entry: dict) -> tuple[QWidget, bool]:
        """One dataset's plot + summary; ``ok`` is False when it cannot be corrected."""
        import pyqtgraph as pg

        page = QWidget()
        box = QVBoxLayout(page)
        points, mfx = entry.get("points"), entry.get("mfx")

        usable = usable_bead_gris(points, self._gris)
        if not usable:
            box.addWidget(self._problem_label(
                "None of the selected beads has a usable trace in this dataset, "
                "so no drift can be computed for it. Select beads that were "
                "actually tracked during this acquisition."))
            return page, False
        names = getattr(getattr(mfx, "dtype", None), "names", None) or ()
        if "lnc" not in names:
            box.addWidget(self._problem_label(
                "This dataset carries no 'lnc' field, so its uncorrected "
                "positions are not available and it cannot be re-corrected. "
                "Drift correction reads 'lnc' and writes 'loc'."))
            return page, False

        try:
            times, drift_nm = drift_curve(points, usable)
            beads = per_bead_curves(points, usable, times)
        except DriftCorrectionError as exc:
            box.addWidget(self._problem_label(str(exc)))
            return page, False

        plot = pg.PlotWidget(background="w")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "time (sec)")
        plot.setLabel("left", "drift (nm)")
        plot.addLegend(offset=(-10, 10))
        step = max(1, len(times) // _MAX_CURVE_PTS)
        t_plot = times[::step]
        for trace in beads.values():
            for axis in range(3):
                plot.plot(t_plot, trace[::step, axis], pen=pg.mkPen(_BEAD_PEN, width=1))
        for axis in range(3):
            plot.plot(t_plot, drift_nm[::step, axis],
                      pen=pg.mkPen(_AXIS_PENS[axis], width=2),
                      name=f"{_AXIS_NAMES[axis]} drift")
        box.addWidget(plot, 1)
        self._plots.append(plot)

        box.addWidget(self._summary_label(entry, points, usable, drift_nm))
        return page, True

    def _summary_label(self, entry, points, usable, drift_nm) -> QLabel:
        """Amplitude of the correction, and how far it moves the current ``loc``.

        The second number is what the user actually cares about: the data has
        already been corrected online, so this reports the *change*, not the
        total drift removed.
        """
        ptp = np.ptp(drift_nm, axis=0)
        skipped = sorted(set(self._gris) - set(usable))
        try:
            new_loc = corrected_loc(entry["mfx"], points, usable)
            old_loc = np.asarray(entry["mfx"]["loc"], dtype=float)
            moved = np.linalg.norm(new_loc - old_loc, axis=1)
            finite = np.isfinite(moved)
            shift = (f"moves the current <tt>loc</tt> by a median of "
                     f"<b>{np.median(moved[finite]) * 1e9:.2f} nm</b> "
                     f"(max {moved[finite].max() * 1e9:.2f} nm) over "
                     f"{int(finite.sum()):,} localizations")
        except Exception as exc:                      # pragma: no cover - defensive
            shift = f"<span style='color:#a40'>could not be previewed: {exc}</span>"
        note = (f"<br><span style='color:#a40'>Beads with no usable trace here, "
                f"skipped: {', '.join(str(g) for g in skipped)}</span>") if skipped else ""
        lbl = QLabel(
            f"Drift amplitude removed: <b>X {ptp[0]:.1f} · Y {ptp[1]:.1f} · "
            f"Z {ptp[2]:.1f} nm</b> from {len(usable)} bead(s); {shift}.{note}")
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        return lbl

    @staticmethod
    def _problem_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #a40;")
        return lbl

    # ------------------------------------------------------------------
    def _apply(self) -> None:
        """Rewrite ``loc`` for every correctable dataset, then accept.

        A dataset that cannot be corrected is skipped rather than aborting the
        rest — its tab already says why.
        """
        self.applied = []
        for entry in self._entries:
            points, mfx = entry.get("points"), entry.get("mfx")
            usable = usable_bead_gris(points, self._gris)
            if not usable or mfx is None:
                continue
            try:
                result = apply_correction(mfx, points, usable)
            except (DriftCorrectionError, ValueError) as exc:
                self.applied.append({"name": entry.get("name"), "error": str(exc)})
                continue
            result["name"] = entry.get("name")
            self.applied.append(result)
        self.accept()

    def _dispose_plots(self) -> None:
        """Drop the process-wide pyqtgraph registrations this dialog's plots hold.

        The *safe* tier (`dispose_plot_widgets`, not `close_plot_widgets`): this
        is a QDialog that keeps its plots on an attribute and the caller reads
        :attr:`applied` after ``exec()`` returns, so the plots must stay valid.
        Idempotent, so being reached twice is fine.
        """
        from ...ui.qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)

    def done(self, result):  # noqa: N802 - Qt API
        """The single funnel for accept / reject / ``exec()`` returning.

        ⚠ ``closeEvent`` alone is **not** enough: ``accept()`` and ``reject()``
        do not raise a close event, so in the normal modal flow the plots would
        never be retired and their ViewBox registrations would leak for the life
        of the process — the documented cause of pyqtgraph aborts much later, in
        unrelated windows. A window-manager close routes through ``reject()``
        and so through here too.
        """
        self._dispose_plots()
        super().done(result)

    def closeEvent(self, event):  # noqa: N802 - Qt API
        """Belt-and-braces for a close that bypasses :meth:`done`."""
        self._dispose_plots()
        super().closeEvent(event)
