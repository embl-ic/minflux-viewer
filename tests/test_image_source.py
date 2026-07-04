"""Series-aware image sources (TIFF + OBF/.msr) and the series chooser."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


# ---- pure helpers ---------------------------------------------------------

def test_obf_axes_for_sizes():
    from minflux_viewer.core.obf_image_source import axes_for_sizes
    assert axes_for_sizes((375, 375)) == "YX"
    assert axes_for_sizes((30, 143, 149)) == "ZYX"
    assert axes_for_sizes((8, 8, 3)) == "YXS"          # trailing RGB samples
    assert axes_for_sizes((90,)) == "X"                # 1-D (non-image)


def test_obf_classify_image_only():
    from minflux_viewer.core.obf_image_source import classify_image_only
    images_plus_hist = [{"ndim": 2, "dtype": "int16"}, {"ndim": 1, "dtype": "float64"}]
    with_minflux = [{"ndim": 2, "dtype": "int16"}, {"ndim": 1, "dtype": "uint8"}]
    assert classify_image_only(images_plus_hist) is True     # no 1-D uint8 ⇒ image-only
    assert classify_image_only(with_minflux) is False        # 1-D uint8 MFXDTA present
    assert classify_image_only([{"ndim": 1, "dtype": "float64"}]) is False   # no image


# ---- TIFF series switching -----------------------------------------------

def _write_two_series(path) -> None:
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8))      # ZYX
        tw.write(np.arange(6 * 6, dtype=np.uint16).reshape(6, 6))            # YX


def test_tiff_series_count_and_switch(tmp_path, _app):
    from minflux_viewer.core.tiff_source import TiffImageSource
    p = tmp_path / "two.ome.tif"
    _write_two_series(p)
    src = TiffImageSource(str(p))
    try:
        assert src.metadata.series_count == 2
        assert len(src.series_names()) == 2
        summaries = src.series_summaries()
        assert [s["index"] for s in summaries] == [0, 1]

        assert src.metadata.shape == (4, 8, 8)       # series 0 is a 4-plane stack
        # the 4-deep axis is exposed as one of T/C/Z (tifffile calls it C here)
        assert max(src.axis_size("T"), src.axis_size("C"), src.axis_size("Z")) == 4
        src.set_series(1)
        assert src.metadata.series_index == 1
        assert src.metadata.shape == (6, 6)
        plane = src.read_plane()
        assert plane.shape == (6, 6)                  # series 1 is a single YX plane
        with pytest.raises(IndexError):
            src.set_series(5)
    finally:
        src.close()


# ---- OBF / .msr (gated on the sample file) --------------------------------

_SAMPLE = (r"D:\Workspace\Microscopes\MINFLUX\sample data"
           r"\20260410 - legacy msr file as bioformat OBF\19_7469_60_HU_GFP.msr")


@pytest.mark.skipif(not os.path.exists(_SAMPLE), reason="OBF sample .msr not present")
def test_obf_source_and_viewer_from_sample(_app):
    from minflux_viewer.core.obf_image_source import (
        ObfImageSource,
        list_obf_image_series,
        msr_is_image_only,
    )
    from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

    assert msr_is_image_only(_SAMPLE) is True
    series = list_obf_image_series(_SAMPLE)
    assert len(series) >= 2
    # a 3-D series exists (Z-stack)
    three_d = [i for i, s in enumerate(series) if s["shape_str"].count("x") == 2]
    assert three_d

    src = ObfImageSource(_SAMPLE, series_index=0)
    win = TiffViewerWindow(src)
    try:
        assert src.metadata.axes == "YX"
        assert src.read_plane().ndim == 2
        # the in-window Series combo drives set_series
        assert win._series_combo.count() == len(series)
        win._series_combo.setCurrentIndex(three_d[0])
        assert src.metadata.axes == "ZYX"
        assert src.axis_size("Z") > 1
        assert src.read_plane(z=1).ndim == 2
    finally:
        win.close()
        _app.processEvents()
