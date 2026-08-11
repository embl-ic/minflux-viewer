"""Series-aware image sources (TIFF + OBF/.msr) and the series chooser."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def test_a_two_dimensional_mfxdta_blob_is_not_an_image():
    """Real .msr files declare the MINFLUX payload as a near-square 2-D uint8
    stack, so shape alone read it as an image: it was listed in the tree and
    exported as a multi-MB TIFF of raw container bytes."""
    from minflux_viewer.core.obf_image_source import (
        classify_image_only,
        is_image_stack,
        is_minflux_data_stack,
    )

    payload = {"ndim": 2, "dtype": "uint8", "sizes": (7301, 7301),
               "minflux_type": "data"}
    confocal = {"ndim": 2, "dtype": "int16", "minflux_type": ""}
    density = {"ndim": 3, "dtype": "uint16", "minflux_type": "density",
               "source_did": "abc"}

    assert is_minflux_data_stack(payload) is True
    assert is_image_stack(payload) is False
    # A density/trace render is MINFLUX-tagged but genuinely an image.
    assert is_minflux_data_stack(density) is False
    assert is_image_stack(density) is True
    assert is_image_stack(confocal) is True
    # ... and such a file is not "image-only" just because the blob is 2-D.
    assert classify_image_only([confocal, density, payload]) is False


def test_minflux_image_association_accepts_m2410_source_and_m2205_did():
    from minflux_viewer.core.obf_image_source import source_did_from_minflux_tag

    assert source_did_from_minflux_tag({"type": "density", "source": "new-did"}) == "new-did"
    assert source_did_from_minflux_tag({"type": "trace", "did": "old-did"}) == "old-did"
    # If both occur, the explicit modern source link is authoritative.
    assert source_did_from_minflux_tag({"source": "new", "did": "old"}) == "new"


def test_extract_ome_xml_is_scoped_to_one_image():
    from xml.etree import ElementTree as ET

    from minflux_viewer.core.tiff_source import extract_ome_image_xml

    xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image ID="Image:0" Name="Ch1"><Pixels ID="Pixels:0" DimensionOrder="XYZCT"
        Type="uint16" SizeX="8" SizeY="7" SizeZ="1" SizeC="1" SizeT="1"/></Image>
      <Image ID="Image:1" Name="Ch2"><Pixels ID="Pixels:1" DimensionOrder="XYZCT"
        Type="uint16" SizeX="6" SizeY="5" SizeZ="1" SizeC="1" SizeT="1"/></Image>
    </OME>"""
    selected = extract_ome_image_xml(xml, 1)
    assert selected is not None
    assert 'Name="Ch2"' in selected
    assert 'Name="Ch1"' not in selected
    assert sum(1 for element in ET.fromstring(selected).iter()
               if element.tag.rsplit("}", 1)[-1] == "Image") == 1


def test_z_range_sum_uses_a_wide_integer_accumulator():
    from minflux_viewer.ui.tiff_viewer_window import _sum_z_planes

    class Source:
        def __init__(self, stack):
            self.stack = stack

        def read_plane(self, *, t=0, c=0, z=0):
            return self.stack[z]

    signed = Source(np.full((3, 2, 2), 30_000, dtype=np.int16))
    signed_sum = _sum_z_planes(signed, t=0, c=0, z_start=0, z_stop=2)
    assert signed_sum.dtype == np.int64
    assert np.all(signed_sum == 90_000)

    unsigned = Source(np.full((2, 2, 2), 65_535, dtype=np.uint16))
    unsigned_sum = _sum_z_planes(unsigned, t=0, c=0, z_start=0, z_stop=1)
    assert unsigned_sum.dtype == np.uint64
    assert np.all(unsigned_sum == 131_070)


def test_imagej_tiff_resolution_and_spacing_are_calibration(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from minflux_viewer.core.tiff_source import TiffImageSource

    path = tmp_path / "imagej.tif"
    tifffile.imwrite(
        path,
        np.zeros((3, 8, 9), dtype=np.uint16),
        imagej=True,
        resolution=(20.0, 20.0),
        metadata={"axes": "ZYX", "unit": "micron", "spacing": 2.0},
    )
    source = TiffImageSource(path)
    try:
        assert source.metadata.pixel_size_x.nm == pytest.approx(50.0)
        assert source.metadata.pixel_size_y.nm == pytest.approx(50.0)
        assert source.metadata.pixel_size_z.nm == pytest.approx(2000.0)
        assert {doc.name for doc in source.metadata.documents} >= {
            "ImageJ metadata", "TIFF header"
        }
    finally:
        source.close()


def test_metadata_viewer_formats_xml_as_table_and_pretty_raw(_app):
    from minflux_viewer.core.tiff_source import MetadataDocument
    from minflux_viewer.ui.metadata_viewer import MetadataDocumentView

    view = MetadataDocumentView([
        MetadataDocument(
            "OME-XML",
            "xml",
            "<OME Creator='Abberior Instruments GmbH, imspector, 16.3-m2205' "
            "UUID='urn:uuid:a0'><Pixels BigEndian='false' DimensionOrder='XYZTC' "
            "PhysicalSizeX='5.000000198e-08'/></OME>",
        )
    ])
    try:
        assert view.tabs.tabText(0) == "XML table"
        assert view.tabs.tabText(1) == "Raw XML"
        assert view.tree.topLevelItemCount() == 1
        ome = view.tree.topLevelItem(0)
        assert [ome.child(i).text(0) for i in range(ome.childCount())] == [
            "Creator", "UUID", "Pixels"
        ]
        assert ome.child(0).text(1) == (
            "Abberior Instruments GmbH, imspector, 16.3-m2205"
        )
        assert view.tree.columnCount() == 2
        assert view.tree.headerItem().text(0) == "Name"
        assert view.tree.headerItem().text(1) == "Value"
        pixels = ome.child(2)
        assert [pixels.child(i).text(0) for i in range(pixels.childCount())] == [
            "BigEndian", "DimensionOrder", "PhysicalSizeX"
        ]
        assert pixels.child(2).text(1) == "5.000000198e-08"
        assert "\n" in view.raw.toPlainText()
        assert "PhysicalSizeX=\"5.000000198e-08\"" in view.raw.toPlainText()
    finally:
        view.close()
        view.deleteLater()
        _app.processEvents()


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


def test_tiff_viewer_z_range_controls_sum_selected_planes(tmp_path, _app):
    tifffile = pytest.importorskip("tifffile")
    from minflux_viewer.core.tiff_source import TiffImageSource
    from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

    stack = np.full((5, 4, 6), 30_000, dtype=np.int16)
    path = tmp_path / "z_stack.tif"
    tifffile.imwrite(path, stack, imagej=True, metadata={"axes": "ZYX"})
    source = TiffImageSource(path)
    window = TiffViewerWindow(source)
    try:
        assert source.metadata.axes == "ZYX"
        assert not hasattr(window, "_z_min_spin")
        assert not hasattr(window, "_z_max_spin")
        assert window._z_value.text() == "1-1 / 5"

        window._set_z_range(1, 5, reload=True)
        assert window._z_value.text() == "1-5 / 5"
        assert window._plane.dtype == np.int64
        assert np.all(window._plane == 150_000)

        window._z_slider.set_range(2, 4, emit=True)
        assert window._z_range_values == (2, 4)
        assert window._z_value.text() == "2-4 / 5"
        assert np.all(window._plane == 90_000)
    finally:
        window.close()
        _app.processEvents()


# ---- OBF / .msr (gated on the sample file) --------------------------------

_SAMPLE = (r"D:\Workspace\Microscopes\MINFLUX\sample data"
           r"\20260410 - legacy msr file as bioformat OBF\19_7469_60_HU_GFP.msr")

_OVERLAY_DIR = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data\rotar_project_confocal_overlay")
_OVERLAY_FILES = list(_OVERLAY_DIR.glob("*.msr")) if _OVERLAY_DIR.exists() else []


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


@pytest.mark.skipif(not _OVERLAY_FILES, reason="rotar overlay sample .msr not present")
def test_overlay_channels_have_per_image_ome_and_imspector_metadata():
    from xml.etree import ElementTree as ET

    from minflux_viewer.core.obf_image_source import ObfImageSource, list_obf_image_series

    path = _OVERLAY_FILES[0]
    series = list_obf_image_series(path)
    selected = [entry for entry in series if entry["name"] in {"Ch1 {12}", "Ch2 {12}"}]
    assert len(selected) == 2
    seen_names = set()
    for entry in selected:
        source = ObfImageSource(path, raw_stack_index=entry["raw_index"])
        try:
            meta = source.metadata
            images = [el for el in ET.fromstring(meta.ome_xml).iter()
                      if el.tag.rsplit("}", 1)[-1] == "Image"]
            assert len(images) == 1
            assert images[0].attrib["Name"] == entry["name"]
            assert any(doc.name == "Imspector XML" and len(doc.content) > 100_000
                       for doc in meta.documents)
            seen_names.add(images[0].attrib["Name"])
        finally:
            source.close()
    assert seen_names == {"Ch1 {12}", "Ch2 {12}"}
