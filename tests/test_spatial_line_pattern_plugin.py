"""UI integration for the spatial line-pattern plugin."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.plugins.spatial_line_pattern import spatial_line_pattern_window
from minflux_viewer.plugins.spatial_line_pattern.spatial_line_pattern_window import (
    SpatialLinePatternWindow,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication(sys.argv)


class _LineOverlay:
    def __init__(self, record):
        self.record = record

    def active_open_line_record(self):
        return self.record

    def active_open_line_points(self):
        return self.record.geometry["points"]


class _CoordinateView:
    def __init__(self, record):
        self._roi_overlay = _LineOverlay(record)
        self._view_box = pg.ViewBox()

    def coordinate_view_box(self):
        return self._view_box

    @staticmethod
    def roi_view_plane():
        return "XY"


def _periodic_dataset():
    rng = np.random.default_rng(12)
    centers = 30.0 + 60.0 * np.arange(10)
    x = np.concatenate([rng.normal(center, 2.0, 25) for center in centers])
    y = np.concatenate(
        [
            rng.normal(8.0 if index % 2 == 0 else -8.0, 1.0, 25)
            for index in range(centers.size)
        ]
    )
    return build_localization_dataset(name="periodic", x_nm=x, y_nm=y)


def test_plugin_is_registered(_app):
    from minflux_viewer import plugins

    plugins.ensure_loaded()
    assert "Spatial Pattern Analysis along Line Profile" in [
        entry.name for entry in plugins.available()
    ]


def test_window_analyzes_active_dataset_and_directed_line(_app):
    state = AppState()
    state.add_dataset(_periodic_dataset())
    record = RoiRecord.create(
        "line",
        {"points": [[0.0, 0.0], [600.0, 0.0]]},
        name="directed axis",
        target_hint=state.active_dataset.name,
    )
    record.context["dataset_idx"] = 0
    record.context["view_plane"] = "XY"
    view = _CoordinateView(record)

    window = SpatialLinePatternWindow(state, 0, view)
    assert window.parent() is None
    assert window._result is not None
    assert window._result.n_used == state.active_dataset.prop.num_loc
    assert window._result.centerline.arc_nm[0] == 0.0
    assert window._result.centerline.points_nm[0, 0] == 0.0
    assert window._result.density_fft_period_nm == pytest.approx(60.0, abs=6.0)
    assert window._map_image.image.shape == window._result.straightened_counts.T.shape
    window.close()


def test_window_exports_profiles_and_complete_analysis(_app, tmp_path, monkeypatch):
    state = AppState()
    state.add_dataset(_periodic_dataset())
    record = RoiRecord.create(
        "line",
        {"points": [[0.0, 0.0], [600.0, 0.0]]},
        target_hint=state.active_dataset.name,
    )
    record.context.update(dataset_idx=0, view_plane="XY")
    window = SpatialLinePatternWindow(state, 0, _CoordinateView(record))

    csv_base = tmp_path / "profiles"
    monkeypatch.setattr(
        spatial_line_pattern_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(csv_base), "CSV (*.csv)"),
    )
    window._save_csv()
    csv_path = Path(f"{csv_base}.csv")
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        "distance_nm,total_count,positive_side_count"
    )

    npz_base = tmp_path / "complete"
    monkeypatch.setattr(
        spatial_line_pattern_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(npz_base), "NumPy archive (*.npz)"),
    )
    window._save_npz()
    with np.load(f"{npz_base}.npz", allow_pickle=False) as archive:
        assert (
            archive["straightened_counts"].shape
            == window._result.straightened_counts.shape
        )
        assert np.array_equal(
            archive["source_roi_points_nm"],
            window._result.centerline.source_points_nm,
        )
        assert "filtered_input_row_indices" in archive
        assert "peak_spacing_order_1_nm" in archive
    window.close()
