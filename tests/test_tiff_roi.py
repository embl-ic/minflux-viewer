"""The image viewer's single active ROI, stored ImageJ-style inside the TIFF.

ImageJ keeps an image's active ROI in the file (``IJMetadata`` tag 50839) and
restores it on open; these cover our half of that contract — encode, embed,
read back, display in nm, and the ``.msr`` acquisition rectangle that seeds it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("roifile")

from minflux_viewer.core.tiff_export import export_image_series_to_tiff, write_ome_tiff
from minflux_viewer.core.tiff_roi import (
    decode_roi,
    rectangle_roi,
    rectangle_roi_from_nm,
    roi_from_shape,
)
from minflux_viewer.core.tiff_source import TiffImageSource


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


# --- pure encode / decode -------------------------------------------------

def test_rectangle_roi_rounds_outward():
    """ImageJ bounds are integers; rounding inward would drop a touched pixel."""
    roi = rectangle_roi(10.4, 20.6, 5.3, 4.1, name="acq")
    assert roi.roi_type == "rectangle"
    assert roi.name == "acq"
    # left=floor(10.4)=10, right=ceil(15.7)=16, top=floor(20.6)=20, bottom=ceil(24.7)=25
    assert roi.bounds == (10.0, 20.0, 6.0, 5.0)


def test_rectangle_from_nm_needs_a_calibration():
    """Without a pixel size the box would land somewhere arbitrary, so the
    caller gets nothing rather than a 1 nm/px guess."""
    assert rectangle_roi_from_nm(100, 200, 50, 60,
                                 pixel_size_x_nm=None, pixel_size_y_nm=None) is None
    assert rectangle_roi_from_nm(100, 200, 50, 60,
                                 pixel_size_x_nm=0.0, pixel_size_y_nm=10.0) is None
    roi = rectangle_roi_from_nm(100, 200, 50, 60,
                                pixel_size_x_nm=10.0, pixel_size_y_nm=20.0)
    assert roi.bounds == (10.0, 10.0, 5.0, 3.0)


def test_roi_from_shape_covers_boxes_and_vertex_shapes():
    assert roi_from_shape("oval", bounds=(2, 3, 8, 6)).roi_type == "oval"
    poly = roi_from_shape("polygon", points=np.array([[0, 0], [10, 0], [10, 8]]))
    assert poly.roi_type == "polygon"
    assert poly.points is not None and len(poly.points) == 3
    # Degenerate shapes are refused rather than written as a zero-size ROI.
    assert roi_from_shape("rectangle", bounds=(1, 1, 0, 5)) is None
    assert roi_from_shape("polygon", points=np.empty((0, 2))) is None
    assert decode_roi(None) is None
    assert decode_roi(b"not a roi") is None


# --- embedding in a TIFF --------------------------------------------------

def _write(tmp_path, name, roi=None, shape=(40, 60)):
    path = tmp_path / name
    write_ome_tiff(path, np.zeros(shape, dtype=np.uint16), axes="YX", shape=shape,
                   dtype=np.uint16, pixel_size_x_nm=25.0, pixel_size_y_nm=25.0, roi=roi)
    return path


def test_active_roi_survives_a_tiff_round_trip(tmp_path):
    roi = rectangle_roi(3, 4, 12, 9, name="acq")
    src = TiffImageSource(_write(tmp_path, "with_roi.tif", roi))
    try:
        back = src.active_roi()
        assert back is not None
        assert (back.roi_type, back.name, back.bounds) == ("rectangle", "acq", (3.0, 4.0, 12.0, 9.0))
        # It is surfaced in the Info window too.
        assert any(k == "Active ROI" and "rectangle" in v
                   for k, v in src.metadata.raw_summary)
    finally:
        src.close()


def test_a_tiff_without_a_roi_reports_none(tmp_path):
    src = TiffImageSource(_write(tmp_path, "plain.tif"))
    try:
        assert src.active_roi() is None
        assert ("Active ROI", "none") in src.metadata.raw_summary
    finally:
        src.close()


def test_the_roi_tag_does_not_disturb_the_ome_calibration(tmp_path):
    """The ROI rides in an extra tag; OME pixel sizes must be unaffected."""
    src = TiffImageSource(_write(tmp_path, "cal.tif", rectangle_roi(0, 0, 5, 5)))
    try:
        assert src.metadata.pixel_size_x.nm == pytest.approx(25.0)
        assert src.metadata.is_ome is True
    finally:
        src.close()


class _Source:
    """Minimal reader-interface stand-in with its own active ROI."""

    def __init__(self, roi):
        self._roi = roi
        self.plane = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)

        class _PS:
            def __init__(self, nm): self.nm = nm

        class _Meta:
            axes = "YX"
            shape = (6, 4)
            channel_names = ()
            pixel_size_x = _PS(30.0)
            pixel_size_y = _PS(30.0)
            pixel_size_z = _PS(None)

            def axis_size(self, axis):
                return dict(zip(self.axes, self.shape)).get(axis, 1)

        self.metadata = _Meta()

    def active_roi(self):
        return self._roi

    def read_plane(self, *, t=0, c=0, z=0):
        return self.plane


def test_export_inherits_the_source_roi_but_none_means_none(tmp_path):
    """Deleting the ROI and saving must write a file with no ROI — the earlier
    ``roi=None`` default silently rewrote the source's own ROI back in."""
    source = _Source(rectangle_roi(1, 2, 3, 4, name="from source"))

    export_image_series_to_tiff(source, tmp_path / "inherited.tif")
    inherited = TiffImageSource(tmp_path / "inherited.tif")
    try:
        assert inherited.active_roi().name == "from source"
    finally:
        inherited.close()

    export_image_series_to_tiff(source, tmp_path / "dropped.tif", roi=None)
    dropped = TiffImageSource(tmp_path / "dropped.tif")
    try:
        assert dropped.active_roi() is None
    finally:
        dropped.close()

    export_image_series_to_tiff(source, tmp_path / "replaced.tif",
                                roi=rectangle_roi(0, 0, 2, 2, name="mine"))
    replaced = TiffImageSource(tmp_path / "replaced.tif")
    try:
        assert replaced.active_roi().name == "mine"
    finally:
        replaced.close()


# --- .msr acquisition ROIs ------------------------------------------------

def _zattrs(*boxes, did="d1"):
    """A ``.zattrs`` block like the one MFXDTA stores at its zarr root."""
    rois = ",".join(
        '{"corners": [[%r, %r], [%r, %r]], "dimensionality": 0, '
        '"linked_cfg": "acquiring mfx", "linked_dta": "%s", "name": "", '
        '"type": "ROI"}' % (x0, y0, x1, y1, did)
        for x0, y0, x1, y1 in boxes)
    return ('{"rois": [' + rois + '], "version": "2.1"}').encode()


def test_acquisition_rois_are_read_out_of_the_raw_file(tmp_path):
    """They are scanned from the file bytes: decoding the MFXDTA container to
    reach them would mean reading every localization blob."""
    from minflux_viewer.msr.acquisition_roi import read_acquisition_rois

    blob = (b"\x00\x01binary junk" + _zattrs((1e-6, 2e-6, 2e-6, 4e-6), did="run-a")
            + b"more junk\xff" + _zattrs((5e-6, 5e-6, 6e-6, 7e-6), did="run-b"))
    path = tmp_path / "fake.msr"
    path.write_bytes(blob)

    rois = read_acquisition_rois(path)
    assert [r.did for r in rois] == ["run-a", "run-b"]
    assert rois[0].bounds == pytest.approx((1e-6, 2e-6, 1e-6, 2e-6))


def test_repeated_roi_blocks_are_not_counted_twice(tmp_path):
    """A run and its aggregated companion share a did and repeat the ROI list."""
    from minflux_viewer.msr.acquisition_roi import read_acquisition_rois

    block = _zattrs((1e-6, 2e-6, 2e-6, 4e-6), did="run-a")
    path = tmp_path / "dup.msr"
    path.write_bytes(block + b"pad" + block)
    assert len(read_acquisition_rois(path)) == 1


def test_rois_group_by_the_run_that_drew_them():
    """Merging rectangles across runs would produce one box spanning the empty
    field between two unrelated acquisitions, marking nothing — which is what
    the wide overview used to get."""
    from minflux_viewer.msr.acquisition_roi import AcquisitionRoi, group_by_dataset

    rois = [AcquisitionRoi("run-a", 1e-6, 1e-6, 2e-6, 2e-6),
            AcquisitionRoi("run-b", 20e-6, 20e-6, 21e-6, 21e-6),
            AcquisitionRoi("run-a", 1.5e-6, 1.5e-6, 2.5e-6, 2.5e-6)]
    grouped = group_by_dataset(rois)
    assert set(grouped) == {"run-a", "run-b"}
    assert len(grouped["run-a"]) == 2 and len(grouped["run-b"]) == 1


def test_only_fully_contained_rois_belong_to_an_image():
    """A rectangle half outside the image would draw clipped and read as the
    wrong acquisition area."""
    from minflux_viewer.msr.acquisition_roi import (
        AcquisitionRoi, rois_within, union_bounds,
    )

    inside = AcquisitionRoi("d", 2e-6, 2e-6, 3e-6, 3e-6)
    straddling = AcquisitionRoi("d", 9e-6, 2e-6, 12e-6, 3e-6)
    extent = ((0.0, 10e-6), (0.0, 10e-6))

    assert rois_within([inside, straddling], extent) == [inside]
    assert union_bounds([]) is None
    # A tiled run collapses to the box its rectangles span.
    tiled = [AcquisitionRoi("d", 1e-6, 1e-6, 2e-6, 2e-6),
             AcquisitionRoi("d", 1.5e-6, 1.5e-6, 3e-6, 4e-6)]
    assert union_bounds(tiled) == pytest.approx((1e-6, 1e-6, 2e-6, 3e-6))


def test_an_image_spanning_several_runs_gets_no_roi(monkeypatch):
    """The overview covers three unrelated MINFLUX runs. Merging them gave one
    box spanning mostly empty field — verified against Imspector, that box
    marked nothing. Such an image now gets no ROI at all."""
    from minflux_viewer.core import obf_image_source as mod
    from minflux_viewer.msr.acquisition_roi import AcquisitionRoi

    src = mod.ObfImageSource.__new__(mod.ObfImageSource)     # no file needed
    src.path = "fake.msr"
    src._did_labels = {"run-a": "45pM", "run-b": "75pM"}
    src._acq_rois = [
        AcquisitionRoi("run-a", 1e-6, 1e-6, 2e-6, 2e-6),
        AcquisitionRoi("run-a", 1.5e-6, 1.5e-6, 2.5e-6, 2.5e-6),
        AcquisitionRoi("run-b", 20e-6, 20e-6, 21e-6, 21e-6),
    ]
    monkeypatch.setattr(mod, "extract_did_label_map", lambda _p: src._did_labels,
                        raising=False)

    def _series(extent):
        src.series_index = 0
        src._stacks = [{"extent_m": extent}]

    # A zoom covering only run-a: its two tiles merge into one named ROI.
    _series(((0.0, 10e-6), (0.0, 10e-6)))
    found = src._single_run_rois()
    assert found is not None and found[0] == "45pM" and len(found[1]) == 2

    # An overview covering both runs: no single acquisition area to mark.
    _series(((0.0, 30e-6), (0.0, 30e-6)))
    assert src._single_run_rois() is None
    assert src.active_roi() is None


# --- viewer overlay -------------------------------------------------------

def test_overlay_converts_between_pixels_and_display_nm(_app):
    """The viewer draws in nm and ImageJ stores pixels; the overlay owns that
    conversion, so a ROI must survive set → read unchanged."""
    import pyqtgraph as pg

    from minflux_viewer.ui.image_roi_overlay import ImageRoiOverlay

    view = pg.PlotWidget()
    overlay = ImageRoiOverlay(view.getPlotItem().vb, pixel_size=(20.0, 50.0))
    try:
        overlay.set_roi(rectangle_roi(4, 6, 10, 3, name="r"))
        assert overlay.has_roi()
        # Drawn at pixel * pixel-size nm ...
        assert overlay._item.pos().x() == pytest.approx(80.0)
        assert overlay._item.pos().y() == pytest.approx(300.0)
        # ... and read back in pixels.
        assert overlay.current_roi().bounds == (4.0, 6.0, 10.0, 3.0)

        overlay.clear()
        assert overlay.current_roi() is None
        assert not overlay.has_roi()
    finally:
        overlay.detach()
        view.close()


def test_overlay_holds_only_one_roi(_app):
    """The image format stores one active ROI, so drawing replaces."""
    import pyqtgraph as pg

    from minflux_viewer.ui.image_roi_overlay import ImageRoiOverlay

    view = pg.PlotWidget()
    vb = view.getPlotItem().vb
    overlay = ImageRoiOverlay(vb, pixel_size=(1.0, 1.0))
    try:
        overlay.set_roi(rectangle_roi(0, 0, 5, 5))
        overlay.set_roi(roi_from_shape("oval", bounds=(2, 2, 6, 6)))
        assert overlay.current_roi().roi_type == "oval"
        assert sum(1 for it in vb.addedItems if hasattr(it, "setFillColor")) == 1
    finally:
        overlay.detach()
        view.close()


def test_live_msr_acquisition_roi_is_toggleable_and_read_only(tmp_path, _app):
    """Only a live MSR source advertises a metadata-controlled acquisition ROI.

    The checkbox is display state, not data state: hiding the graphics keeps the
    exact source ROI available for TIFF export.
    """
    from PyQt6.QtCore import Qt

    from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

    source = TiffImageSource(_write(tmp_path, "live_source.tif"))
    source_roi = rectangle_roi(3, 4, 12, 9, name="MINFLUX acquisition ROI")
    source.active_roi_role = "acquisition"
    source.active_roi_label = "acquisition ROI"
    source.active_roi_read_only = True
    source.active_roi = lambda: source_roi

    window = TiffViewerWindow(source)
    try:
        layout = window._control_row.layout()
        assert not window._acquisition_roi_check.isHidden()
        assert window._acquisition_roi_check.isChecked()
        assert layout.indexOf(window._series_combo) < layout.indexOf(
            window._acquisition_roi_check
        )

        overlay = window._roi_overlay
        assert overlay.current_roi() is source_roi
        assert not overlay.editable
        assert overlay.visible
        assert overlay._item.translatable is False
        assert overlay._item.acceptedMouseButtons() == Qt.MouseButton.NoButton
        assert not overlay._item.handles

        window._acquisition_roi_check.setChecked(False)
        assert not overlay.visible
        assert not overlay._item.isVisible()
        assert overlay.current_roi() is source_roi
        assert "  |  ROI: " not in window._info_label.text()

        window._acquisition_roi_check.setChecked(True)
        assert overlay.visible
        assert overlay._item.isVisible()
    finally:
        window.close()
        _app.processEvents()


def test_roi_reopened_from_tiff_remains_an_ordinary_editable_roi(tmp_path, _app):
    from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

    source = TiffImageSource(
        _write(tmp_path, "exported.tif", rectangle_roi(3, 4, 12, 9, name="acq"))
    )
    window = TiffViewerWindow(source)
    try:
        assert window._acquisition_roi_check.isHidden()
        assert window._roi_overlay.editable
        assert window._roi_overlay._item.handles
    finally:
        window.close()
        _app.processEvents()


class _Drag:
    """Stand-in for a pyqtgraph mouse-drag event in scene coordinates."""

    def __init__(self, down, now, finish):
        from PyQt6.QtCore import QPointF
        self._down, self._now, self._finish = QPointF(*down), QPointF(*now), finish

    def button(self):
        from PyQt6.QtCore import Qt
        return Qt.MouseButton.LeftButton

    def buttonDownScenePos(self):
        return self._down

    def scenePos(self):
        return self._now

    def isFinish(self):
        return self._finish

    def accept(self):
        pass


def test_dragging_with_a_tool_armed_draws_and_replaces_the_roi(_app):
    """Drawing must pre-empt the ViewBox pan, commit only on release, and — the
    image format holding one ROI — replace whatever was there."""
    import pyqtgraph as pg

    from minflux_viewer.ui.image_roi_overlay import ImageRoiOverlay

    view = pg.PlotWidget()
    view.resize(400, 400)
    view.show()
    vb = view.getPlotItem().vb
    vb.setRange(xRange=(0, 1000), yRange=(0, 1000), padding=0)
    _app.processEvents()

    overlay = ImageRoiOverlay(vb, pixel_size=(10.0, 10.0))
    try:
        scene = lambda x, y: vb.mapViewToScene(pg.Point(x, y))          # noqa: E731
        down = scene(100, 200)
        moved = scene(300, 500)
        up = scene(400, 600)
        pt = lambda p: (p.x(), p.y())                                    # noqa: E731

        overlay.set_tool("rectangle")
        vb.mouseDragEvent(_Drag(pt(down), pt(moved), False))
        assert not overlay.has_roi()                    # preview only, not committed
        vb.mouseDragEvent(_Drag(pt(down), pt(up), True))
        assert overlay.has_roi()
        x, y, w, h = overlay.current_roi().bounds       # 300x400 nm at 10 nm/px
        assert (x, y, w, h) == pytest.approx((10, 20, 30, 40), abs=1.5)

        overlay.set_tool("oval")
        vb.mouseDragEvent(_Drag(pt(scene(50, 50)), pt(scene(150, 250)), True))
        assert overlay.current_roi().roi_type == "oval"
        assert sum(1 for it in vb.addedItems if hasattr(it, "setFillColor")) == 1

        # Disarmed, the drag falls through to the ViewBox's own handler (pan)
        # instead of drawing.
        overlay.set_tool(None)
        assert overlay.tool is None
        before = overlay.current_roi().bounds
        delegated = []
        overlay._original_drag = lambda ev, axis=None: delegated.append(ev)
        vb.mouseDragEvent(_Drag(pt(scene(700, 700)), pt(scene(900, 900)), True))
        assert len(delegated) == 1
        assert overlay.current_roi().bounds == before
    finally:
        overlay.detach()
        view.close()
def test_overlay_restores_the_viewbox_drag_on_detach(_app):
    """The overlay wraps ViewBox.mouseDragEvent to draw; leaving it wrapped
    after the window closes would keep a dead window's handler alive."""
    import pyqtgraph as pg

    from minflux_viewer.ui.image_roi_overlay import ImageRoiOverlay

    view = pg.PlotWidget()
    vb = view.getPlotItem().vb
    original = vb.mouseDragEvent
    overlay = ImageRoiOverlay(vb, pixel_size=(1.0, 1.0))
    assert "mouseDragEvent" in vars(vb)                  # wrapped on the instance
    overlay.detach()
    # Back to the class handler — not merely a re-bound copy shadowing it.
    assert "mouseDragEvent" not in vars(vb)
    assert vb.mouseDragEvent.__func__ is original.__func__
    view.close()
