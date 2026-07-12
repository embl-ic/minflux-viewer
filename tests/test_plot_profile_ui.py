"""Plot Profile UI: active-line resolution + live dialog compute / width band."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.plot_profile_dialog import PlotProfileDialog
from minflux_viewer.ui.roi_overlay import RoiOverlayController


@pytest.fixture(scope="module")
def _app():
    yield QApplication.instance() or QApplication([])


class _Owner(QWidget):
    def __init__(self):
        super().__init__()
        self._state = SimpleNamespace(prefs={"plot": {"roi_color": "Yellow"}}, datasets=[])

    def roi_view_plane(self):
        return "XY"

    def roi_depth_center(self):
        return 0.0

    def normalize_roi_record(self, record):
        return record


def _ctrl(_app):
    plot = pg.PlotWidget()
    return RoiOverlayController(RoiStore(), _Owner(), plot, plot.getPlotItem())


# --------------------------------------------- active_open_line_record

def test_active_open_line_prefers_draft_line(_app):
    ctrl = _ctrl(_app)
    ctrl._set_draft(RoiRecord.create("polyline", {"points": [[0, 0], [10, 0]], "closed": False}))
    rec = ctrl.active_open_line_record()
    assert rec is not None and rec.type == "polyline"


def test_active_open_line_ignores_non_line_draft(_app):
    ctrl = _ctrl(_app)
    ctrl._set_draft(RoiRecord.create("rectangle", {"bounds": [0, 0, 10, 10]}))
    assert ctrl.active_open_line_record() is None


def test_active_open_line_uses_single_selected_stored_line(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("line", {"points": [[0, 0], [5, 5]], "closed": False})
    ctrl.store.add(rec)
    ctrl.store.select([rec.id])
    got = ctrl.active_open_line_record()
    assert got is not None and got.id == rec.id


def test_active_open_line_points_tracks_live_item_before_commit(_app):
    """The Plot Profile reads the **live** pyqtgraph item vertices, so a drag
    updates the profile before the record geometry commits (on release)."""
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("line", {"points": [[0.0, 0.0], [10.0, 0.0]], "closed": False})
    ctrl.store.add(rec)
    ctrl.store.select([rec.id])
    ctrl.refresh()                                    # build the pyqtgraph item
    item = ctrl.items.get(rec.id)
    assert item is not None
    item.setPos(item.pos() + pg.Point(5.0, 3.0))      # a live whole-line drag
    live = np.asarray(ctrl.active_open_line_points(), dtype=float)
    assert live[0] == pytest.approx([5.0, 3.0], abs=1e-6)
    assert live[1] == pytest.approx([15.0, 3.0], abs=1e-6)
    # the stored record is unchanged — it only commits on sigRegionChangeFinished
    assert [list(p[:2]) for p in rec.geometry["points"]] == [[0.0, 0.0], [10.0, 0.0]]


# --------------------------------------------- dialog compute + band

def _stub_view(line, locs, version=1):
    """A minimal render/scatter stand-in exposing the Plot Profile interface.
    Keeps its PlotWidget alive so the ViewBox survives for the band overlay."""
    pw = pg.PlotWidget()

    def _pts():
        if line is None:
            return None
        return [[float(p[0]), float(p[1])] for p in line.geometry.get("points", [])]

    overlay = SimpleNamespace(active_open_line_record=lambda: line,
                              active_open_line_points=_pts, activate=lambda: None)
    return SimpleNamespace(
        _pw=pw,                                        # retain — vb dies with its widget
        _roi_overlay=overlay,
        roi_view_plane=lambda: "XY",
        coordinate_view_box=lambda: pw.getPlotItem().vb,
        profile_localizations=lambda: locs,
        profile_locs_version=lambda: version,
    )


def _cluster(x, y, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(x, 1.0, n), rng.normal(y, 1.0, n)])


def test_dialog_profiles_localizations_along_the_line(_app):
    line = RoiRecord.create("line", {"points": [[100.0, 240.0], [200.0, 240.0]], "closed": False})
    locs = _cluster(150.0, 240.0)                      # a cluster at distance 50 (x=150)
    dlg = PlotProfileDialog(_stub_view(line, locs))
    dlg._tick()
    x, y = dlg._curve.getData()
    assert x is not None and len(x) > 2
    assert dlg._dist[np.argmax(dlg._vals)] == pytest.approx(50.0, abs=5.0)   # peak at the cluster
    dlg.close()


def test_dialog_width_band_added_and_removed_on_close(_app):
    line = RoiRecord.create("line", {"points": [[100.0, 240.0], [200.0, 240.0]], "closed": False})
    locs = np.column_stack([np.linspace(100, 200, 50), np.full(50, 240.0)])
    dlg = PlotProfileDialog(_stub_view(line, locs))    # default width 20 → band shown
    assert dlg._band_item is not None and dlg._band_item.isVisible()
    dlg.close()
    assert dlg._band_item is None                       # band removed on close


def test_dialog_without_line_shows_prompt_and_no_band(_app):
    dlg = PlotProfileDialog(_stub_view(None, np.zeros((0, 2))))
    dlg._tick()
    assert "line" in dlg._info.text().lower()
    assert dlg._band_item is None
    dlg.close()


def test_dialog_refetches_locs_only_on_data_version_change(_app):
    """Zoom/pan don't change the loc version → the profile isn't refetched/recomputed
    (the fix for the FOV-dependent disappear/hang). Only a data change refetches."""
    line = RoiRecord.create("line", {"points": [[100.0, 240.0], [200.0, 240.0]], "closed": False})
    view = _stub_view(line, _cluster(150.0, 240.0))
    calls = {"n": 0}
    base = view.profile_localizations
    view.profile_localizations = lambda: (calls.__setitem__("n", calls["n"] + 1), base())[1]
    ver = {"v": 1}
    view.profile_locs_version = lambda: ver["v"]
    dlg = PlotProfileDialog(view)                       # first tick → 1 fetch
    dlg._tick()                                         # nothing changed → no fetch
    dlg._tick()
    assert calls["n"] == 1
    ver["v"] = 2                                        # data changed → refetch
    dlg._recompute_now()
    assert calls["n"] == 2
    dlg.close()


def test_hover_readout_snaps_to_nearest_profile_point(_app):
    line = RoiRecord.create("line", {"points": [[100.0, 240.0], [200.0, 240.0]], "closed": False})
    locs = _cluster(170.0, 240.0, seed=1)              # a density peak at distance 70 (x=170)
    dlg = PlotProfileDialog(_stub_view(line, locs))
    dlg._tick()
    peak_d = float(dlg._dist[np.argmax(dlg._vals)])
    assert peak_d == pytest.approx(70.0, abs=5.0)
    d, rho = dlg._nearest_profile_point(peak_d)
    assert d == pytest.approx(peak_d) and rho == pytest.approx(float(np.max(dlg._vals)))
    # no profile → nothing to read, hover hidden
    dlg._clear("none")
    assert dlg._nearest_profile_point(70.0) is None
    assert not dlg._hover_dot.isVisible()
    dlg.close()


def test_hover_label_anchors_below_a_peak_to_avoid_top_clip(_app):
    xr, yr = (0.0, 100.0), (0.0, 50.0)
    # a point high in the view (a peak) → label goes BELOW (ay=0), not off the top
    assert PlotProfileDialog._label_anchor(50.0, 48.0, xr, yr)[1] == 0
    # a low point → label goes ABOVE (ay=1)
    assert PlotProfileDialog._label_anchor(50.0, 2.0, xr, yr)[1] == 1
    # near the right edge → anchor to the left (ax=1) so the text stays on-screen
    assert PlotProfileDialog._label_anchor(90.0, 10.0, xr, yr)[0] == 1
    assert PlotProfileDialog._label_anchor(10.0, 10.0, xr, yr)[0] == 0
