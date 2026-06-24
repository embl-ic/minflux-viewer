import numpy as np
import pytest

from minflux_viewer.core.tiff_export import (
    TiffExportChannel,
    build_edges,
    choose_dtype,
    export_render_to_tiff,
)

tifffile = pytest.importorskip("tifffile")


def test_build_edges_matlab_convention():
    # 0..10 step 5 -> floor(10/5)+1 = 3 bins, last edge 15 keeps max strictly inside.
    edges = build_edges(0.0, 10.0, 5.0)
    assert np.allclose(edges, [0.0, 5.0, 10.0, 15.0])
    # Degenerate range -> single bin.
    assert np.allclose(build_edges(7.0, 7.0, 2.0), [7.0, 9.0])


def test_build_edges_rejects_bad_step():
    with pytest.raises(ValueError):
        build_edges(0.0, 1.0, 0.0)


def test_choose_dtype_thresholds():
    assert choose_dtype(255) is np.uint8
    assert choose_dtype(256) is np.uint16
    assert choose_dtype(65535) is np.uint16
    assert choose_dtype(65536) is np.uint32


def test_export_2d_single_channel(tmp_path):
    # 5 localizations stacked in one XY voxel -> max count 5, 8-bit, single page.
    xyz = np.zeros((5, 3), dtype=np.float64)
    xyz[:, 0] = 2.0
    xyz[:, 1] = 3.0
    out = tmp_path / "render.ome.tif"
    result = export_render_to_tiff(
        [TiffExportChannel("ch", xyz)],
        out,
        pixel_size_nm=10.0,
        voxel_depth_nm=10.0,
        is_3d=False,
    )
    assert result.axes == "YX"
    assert result.dtype == "uint8"
    assert result.max_count == 5
    assert result.n_slices == 1
    arr = tifffile.imread(str(out))
    assert arr.ndim == 2
    assert int(arr.max()) == 5
    assert int(arr.sum()) == 5  # all 5 in one pixel


def test_export_3d_stack_and_calibration(tmp_path):
    rng = np.random.default_rng(0)
    xyz = np.zeros((300, 3), dtype=np.float64)
    xyz[:, 0] = rng.uniform(0, 100, 300)
    xyz[:, 1] = rng.uniform(0, 100, 300)
    xyz[:, 2] = rng.uniform(0, 80, 300)
    out = tmp_path / "vol.ome.tif"
    result = export_render_to_tiff(
        [TiffExportChannel("ch", xyz)],
        out,
        pixel_size_nm=10.0,
        voxel_depth_nm=20.0,
        is_3d=True,
    )
    assert result.axes == "ZYX"
    assert result.n_slices >= 4
    # All localizations preserved across the stack.
    arr = tifffile.imread(str(out))
    assert int(arr.sum()) == 300

    from minflux_viewer.core.tiff_source import TiffImageSource

    src = TiffImageSource(out)
    try:
        assert src.metadata.pixel_size_x.nm == pytest.approx(10.0)
        assert src.metadata.pixel_size_z.nm == pytest.approx(20.0)
    finally:
        src.close()


def test_export_multichannel_ome(tmp_path):
    a = np.zeros((10, 3))
    a[:, 0] = 5.0
    a[:, 1] = 5.0
    b = np.zeros((4, 3))
    b[:, 0] = 50.0
    b[:, 1] = 50.0
    out = tmp_path / "multi.ome.tif"
    result = export_render_to_tiff(
        [TiffExportChannel("red", a), TiffExportChannel("green", b)],
        out,
        pixel_size_nm=10.0,
        voxel_depth_nm=10.0,
        is_3d=False,
    )
    assert result.axes == "CYX"
    assert result.n_channels == 2

    from minflux_viewer.core.tiff_source import TiffImageSource

    src = TiffImageSource(out)
    try:
        assert src.metadata.channel_names == ("red", "green")
    finally:
        src.close()


def test_progress_callback_reports_every_page(tmp_path):
    xyz = np.zeros((6, 3))
    xyz[:, 2] = np.array([0, 0, 25, 25, 50, 50], dtype=float)
    calls = []
    export_render_to_tiff(
        [TiffExportChannel("ch", xyz)],
        tmp_path / "p.ome.tif",
        pixel_size_nm=10.0,
        voxel_depth_nm=20.0,
        is_3d=True,
        progress=lambda done, total, msg: calls.append((done, total)),
    )
    assert calls
    assert calls[-1][0] == calls[-1][1]  # finished on the last page
    assert all(0 < d <= t for d, t in calls)


def test_export_xy_range_clips_to_roi(tmp_path):
    # 100 locs spread over 0..1000 nm; restrict to a 0..100 nm XY box.
    rng = np.random.default_rng(2)
    xyz = np.zeros((100, 3))
    xyz[:, 0] = rng.uniform(0, 1000, 100)
    xyz[:, 1] = rng.uniform(0, 1000, 100)
    expected = int(np.count_nonzero((xyz[:, 0] <= 100) & (xyz[:, 1] <= 100)))
    out = tmp_path / "roi.ome.tif"
    export_render_to_tiff(
        [TiffExportChannel("ch", xyz)],
        out,
        pixel_size_nm=10.0,
        voxel_depth_nm=10.0,
        is_3d=False,
        x_range=(0.0, 100.0),
        y_range=(0.0, 100.0),
    )
    arr = tifffile.imread(str(out))
    assert int(arr.sum()) == expected
    assert arr.shape[0] <= 12 and arr.shape[1] <= 12  # ~100/10 px + boundary bin


def test_export_z_range_clips_stack(tmp_path):
    xyz = np.zeros((9, 3))
    xyz[:, 2] = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
    out = tmp_path / "zr.ome.tif"
    export_render_to_tiff(
        [TiffExportChannel("ch", xyz)],
        out,
        pixel_size_nm=10.0,
        voxel_depth_nm=10.0,
        is_3d=True,
        z_range=(20.0, 50.0),  # keeps z in {20,30,40,50} -> 4 locs
    )
    arr = tifffile.imread(str(out))
    assert int(arr.sum()) == 4


def test_export_range_outside_data_raises(tmp_path):
    xyz = np.zeros((5, 3))
    with pytest.raises(ValueError, match="inside the selected export range"):
        export_render_to_tiff(
            [TiffExportChannel("ch", xyz)],
            tmp_path / "none.ome.tif",
            pixel_size_nm=10.0,
            voxel_depth_nm=10.0,
            is_3d=False,
            x_range=(1000.0, 2000.0),
        )


def test_export_rejects_tiny_pixel_size(tmp_path):
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.linspace(0, 1_000_000, 10)  # 1 mm extent
    with pytest.raises(ValueError, match="too small"):
        export_render_to_tiff(
            [TiffExportChannel("ch", xyz)],
            tmp_path / "huge.ome.tif",
            pixel_size_nm=0.1,  # 1 mm / 0.1 nm = 10M px wide
            voxel_depth_nm=10.0,
            is_3d=False,
        )


def test_export_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        export_render_to_tiff(
            [TiffExportChannel("ch", np.empty((0, 3)))],
            tmp_path / "empty.ome.tif",
            pixel_size_nm=10.0,
            voxel_depth_nm=10.0,
            is_3d=False,
        )
