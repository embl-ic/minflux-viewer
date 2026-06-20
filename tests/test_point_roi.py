"""Point ROI: fixed-size marker item + geometry round-trip + ImageJ mapping."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.roi import RoiRecord, record_from_imagej, record_to_imagej
from minflux_viewer.ui.roi_overlay import PointMarkerItem, item_to_geometry


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_point_marker_construction_and_geometry(_app):
    item = PointMarkerItem((12.5, -3.0))
    assert abs(float(item.pos().x()) - 12.5) < 1e-9
    assert abs(float(item.pos().y()) - -3.0) < 1e-9
    # item_to_geometry reads the centre back out (TargetItem has no size()).
    geo = item_to_geometry("point", item)
    assert geo["point"] == [12.5, -3.0]


def test_point_marker_setfillcolor_is_noop(_app):
    # generic overlay styling must not change our concentric-ring look
    item = PointMarkerItem((0.0, 0.0))
    item.setFillColor("#ff0000", alpha=200)  # should simply do nothing


def test_point_imagej_requires_pixel_space():
    plot_pt = RoiRecord.create("point", {"point": [4.0, 5.0]}, coordinate_space="plot")
    with pytest.raises(ValueError):
        record_to_imagej(plot_pt)


def test_point_imagej_roundtrip_pixel_space():
    roifile = pytest.importorskip("roifile")
    pix_pt = RoiRecord.create("point", {"point": [4.0, 5.0]}, coordinate_space="pixel")
    ij = record_to_imagej(pix_pt)
    assert ij.roitype == roifile.ROI_TYPE.POINT
    back = record_from_imagej(ij)
    assert back.type == "point"
    assert np.allclose(back.geometry["point"], [4.0, 5.0])
