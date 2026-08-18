"""Everything one dataset's beam-monitoring (MBM) beads say, in one window.

Two views existed already, both inside the MSR reader and both reachable only
while a ``.msr`` was parsed:

* **Beads drift** (:class:`~minflux_viewer.plugins.msr_reader.beads_drift_dialog.BeadsDriftDialog`)
  — per-bead drift traces over the acquisition, colored by time;
* **Beads and data region** (``AlignmentPlotWindow`` in its single-channel mode)
  — absolute bead positions against the localization extent, plus a per-bead
  table of total drift.

They answer different questions about the same beads (*how much did each bead
move* vs *where were the beads relative to the data*), so they are combined here
as two tabs of one window rather than shown as two floating dialogs.  Both are
embedded in a **read-only** form: the drift dialog's alignment controls
(``info_mode``) and the alignment window's fit machinery have nothing to act on
for a single already-loaded dataset.

The data comes from the arrays the dataset itself carries
(``ds.mbm`` / ``metadata["mbm_points"]``), so no source file is re-parsed.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..plugins.msr_reader.beads_drift import (
    dataset_bead_drift,
    single_channel_bead_summary,
)


def dataset_loc_bounds_nm(ds):
    """``(min_xyz, max_xyz)`` of the dataset's localizations in nm, or ``None``.

    Drawn as the "data region" box the beads are shown against.  Uses the
    displayed coordinate view (``loc_nm``), so the box matches what the render
    view shows for the same dataset.
    """
    import numpy as np

    try:
        loc = np.asarray(ds.loc_nm, dtype=float)
    except Exception:
        return None
    if loc.ndim != 2 or loc.shape[0] == 0:
        return None
    xyz = np.zeros((loc.shape[0], 3), dtype=float)
    cols = min(3, loc.shape[1])
    xyz[:, :cols] = loc[:, :cols]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if xyz.size == 0:
        return None
    return xyz.min(axis=0), xyz.max(axis=0)


class MbmInfoWindow(QWidget):
    """Tabbed MBM view for one loaded dataset.

    Built as an unparented top-level widget per the window convention; show it
    through ``ui/modeless.py::show_modeless`` so it is retained and closed with
    the main window.
    """

    def __init__(self, dataset_name: str, beads: list[dict], *,
                 data_bounds_nm=None) -> None:
        super().__init__(None)
        self.setWindowTitle(f"MBM info — {dataset_name}")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1150, 820)

        self._beads = beads
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        summary = QLabel(
            f"<b>{len(beads)} beam-monitoring bead(s)</b> recorded with "
            f"'{dataset_name}'. Beads are the fiducials the microscope tracked "
            "during the acquisition; their motion is the stage/sample drift the "
            "measurement was corrected against."
        )
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        root.addWidget(summary)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._build_drift_tab(dataset_name, beads), "Drift")
        region = self._build_region_tab(dataset_name, beads, data_bounds_nm)
        if region is not None:
            self._tabs.addTab(region, "Beads && data region")
        root.addWidget(self._tabs, 1)

    # -- tabs ----------------------------------------------------------------

    def _build_drift_tab(self, name: str, beads: list[dict]) -> QWidget:
        from ..plugins.msr_reader.beads_drift_dialog import BeadsDriftDialog

        self._drift = BeadsDriftDialog(
            [{"name": name, "beads": beads}], info_mode=True)
        return _as_page(self._drift)

    def _build_region_tab(self, name: str, beads: list[dict],
                          data_bounds_nm) -> QWidget | None:
        from ..plugins.msr_reader.msr_reader_dialog import AlignmentPlotWindow

        payload = single_channel_bead_summary(name, beads)
        if payload is None:
            return None
        # results=[] + single_channel → the window's no-alignment mode: it plots
        # the beads at their absolute positions with the data-region box and
        # fills its table with drift instead of fit residuals.
        self._region = AlignmentPlotWindow(
            [], None, data_bounds_nm=data_bounds_nm, single_channel=payload)
        return _as_page(self._region)

    # -- lifetime ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # The embedded dialogs are children of the page widgets, so Qt destroys
        # them with this window; give each its own teardown first (the alignment
        # window disconnects its live-refit signal there).
        for child in (getattr(self, "_region", None), getattr(self, "_drift", None)):
            if child is not None:
                try:
                    child.close()
                except RuntimeError:
                    pass
        super().closeEvent(event)


def _as_page(dialog) -> QWidget:
    """Wrap a top-level dialog so it can live inside a tab.

    Clearing ``Qt.Window`` is what turns it from a window into an ordinary child
    widget; without it the "page" would be an empty area and the dialog would
    still float on its own.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    dialog.setWindowFlags(Qt.WindowType.Widget)
    if hasattr(dialog, "setSizeGripEnabled"):
        dialog.setSizeGripEnabled(False)
    layout.addWidget(dialog)
    return page


def open_mbm_info(owner, ds, dataset_name: str | None = None):
    """Open the MBM info window for *ds*, or return ``None`` with a reason.

    Returns ``(window, None)`` on success and ``(None, reason)`` when the
    dataset has nothing to show, so the caller decides how to report it.
    """
    beads = dataset_bead_drift(ds)
    name = dataset_name or getattr(ds, "name", "dataset")
    if not beads:
        from ..core.overlay import mbm_points_array

        points = mbm_points_array(ds)
        if points is None or not getattr(points, "size", 0):
            return None, (
                f"'{name}' carries no beam-monitoring (MBM) bead data.\n\n"
                "MBM beads come with a dataset imported from an .msr file (or one "
                "saved back to .msr); other formats do not carry them."
            )
        return None, (
            f"'{name}' has an MBM points array, but it has no usable "
            "gri / xyz / tim columns to reconstruct bead traces from."
        )

    from .modeless import show_modeless

    win = MbmInfoWindow(name, beads, data_bounds_nm=dataset_loc_bounds_nm(ds))
    show_modeless(win, owner)
    return win, None
