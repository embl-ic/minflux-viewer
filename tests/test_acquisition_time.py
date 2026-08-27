"""Instrument acquisition timestamp: parsing, display, and metadata round trips.

The fact under test is ``mfx/.zattrs["acquisition_date"]`` (m2410) / ``tms``
(m2205) — when the microscope recorded the run, as opposed to when the file was
written. See :mod:`minflux_viewer.core.acquisition_time`.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core.acquisition_time import (
    ACQUISITION_DATE_KEY,
    ACQUISITION_SPAN_KEY,
    acquisition_date_from_zattrs,
    acquisition_metadata,
    dataset_acquisition_date,
    dataset_acquisition_span,
    dataset_acquisition_text,
    format_acquisition_range,
    format_span,
    parse_acquisition_date,
    span_seconds_from_tim,
    stamp_acquisition,
    stamp_dataset_acquisition,
)
from minflux_viewer.core.save import build_metadata

# A real value read out of 2_3C_measurement.msr, and the m2205 `tms` of
# 1_sample_A_1-100_seq_3D_ori_exc_5.msr (whose stack label is 220601-132142).
REAL_ISO = "2026-04-21T13:09:33+02:00"
REAL_TMS = 1654082503.02894


def _make_mfx(n=40, tim_max=3240.4):
    dt = np.dtype([
        ("vld", np.bool_), ("tid", np.int32), ("tim", np.float64),
        ("itr", np.int32), ("efo", np.float64),
        ("dcr", np.float64, (2,)), ("loc", np.float64, (3,)),
    ])
    rng = np.random.default_rng(0)
    a = np.zeros(n, dt)
    a["vld"] = True
    a["tid"] = np.repeat(np.arange(n // 4), 4)[:n]
    a["tim"] = np.linspace(0.0, tim_max, n)
    a["itr"] = 3
    a["efo"] = rng.uniform(10, 100, n)
    a["dcr"] = rng.uniform(0, 1, (n, 2))
    a["loc"] = np.column_stack([rng.uniform(0, 1e-6, n),
                                rng.uniform(0, 1e-6, n),
                                rng.uniform(0, 5e-7, n)])
    return a


def _make_ds(tim_max=3240.4, stamp=REAL_ISO):
    ds = L.load_from_mfx_array(_make_mfx(tim_max=tim_max), name="t", prefs={"data": {}})
    if stamp:
        stamp_dataset_acquisition(ds, stamp)
    return ds


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_iso_with_offset_keeps_its_timezone():
    when = parse_acquisition_date(REAL_ISO)
    assert when.utcoffset() == timedelta(hours=2)
    assert (when.year, when.month, when.day) == (2026, 4, 21)
    assert (when.hour, when.minute, when.second) == (13, 9, 33)


def test_m2205_tms_epoch_reproduces_the_imspector_label():
    # The stack description of that file is '220601-132142_minflux'; the epoch
    # must land on the same wall clock in local time.
    when = parse_acquisition_date(REAL_TMS)
    assert when is not None
    assert when.strftime("%y%m%d-%H%M%S")[:11] == "220601-1321"


@pytest.mark.parametrize("bad", [None, "", "   ", "not a date", True, False, 0, -1, float("nan")])
def test_unparseable_values_are_none_not_an_error(bad):
    assert parse_acquisition_date(bad) is None


def test_zattrs_prefers_m2410_verbatim_over_tms():
    # Verbatim matters: re-formatting would drop the instrument's own offset.
    attrs = {"acquisition_date": REAL_ISO, "tms": REAL_TMS}
    assert acquisition_date_from_zattrs(attrs) == REAL_ISO


def test_zattrs_falls_back_to_tms_for_m2205():
    got = acquisition_date_from_zattrs({"tms": REAL_TMS})
    assert got is not None
    assert parse_acquisition_date(got) == parse_acquisition_date(REAL_TMS)


@pytest.mark.parametrize("attrs", [{}, None, {"acquisition_date": ""}, {"tms": 0}])
def test_zattrs_without_a_date_returns_none(attrs):
    assert acquisition_date_from_zattrs(attrs) is None


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------
def test_span_is_max_tim_not_the_peak_to_peak():
    # tim counts from the acquisition start, so the first localization's delay
    # (here 40 s, the search phase) is part of the run, not an offset to remove.
    assert span_seconds_from_tim(np.array([40.0, 100.0, 3240.4])) == pytest.approx(3240.4)


def test_span_ignores_nan_and_empty():
    assert span_seconds_from_tim(np.array([np.nan, 5.0])) == pytest.approx(5.0)
    assert span_seconds_from_tim(np.array([])) is None
    assert span_seconds_from_tim(np.array([np.nan])) is None
    assert span_seconds_from_tim(None) is None


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seconds,expected", [
    (20 * 3600, "20 hours"),
    (3600, "1 hour"),
    (6840, "1 hour 54 min"),
    (3240.4, "54 min"),
    (540, "9 min"),
    (45, "45 s"),
    (0, "0 s"),
])
def test_format_span(seconds, expected):
    assert format_span(seconds) == expected


def test_rounding_never_reports_sixty_minutes():
    assert format_span(3599.9) == "1 hour"


def test_same_day_range_omits_the_repeated_date():
    got = format_acquisition_range(parse_acquisition_date(REAL_ISO), 6840)
    assert got == "2026-Apr-21,13:09:33 ~ 15:03:33 (span 1 hour 54 min)"


def test_range_crossing_midnight_repeats_the_date():
    start = parse_acquisition_date("2026-06-26T13:00:00+02:00")
    got = format_acquisition_range(start, 20 * 3600)
    assert got == "2026-Jun-26,13:00:00 ~ 2026-Jun-27,09:00:00 (span 20 hours)"


def test_month_name_is_locale_independent():
    # %b would render 'Jun' as 'Juni' under a German locale; the table must not.
    for month, name in ((1, "Jan"), (6, "Jun"), (12, "Dec")):
        start = datetime(2026, month, 5, 8, 0, 0, tzinfo=timezone.utc)
        assert format_acquisition_range(start, 60).startswith(f"2026-{name}-05,")


def test_start_without_a_span_still_reports_the_start():
    assert format_acquisition_range(parse_acquisition_date(REAL_ISO), None) == "2026-Apr-21,13:09:33"


def test_no_date_renders_the_placeholder():
    assert format_acquisition_range(None, 100) == "—"


# ---------------------------------------------------------------------------
# Stamping onto a dataset
# ---------------------------------------------------------------------------
def test_stamping_records_date_and_span_from_the_datasets_own_tim():
    ds = _make_ds(tim_max=3240.4)
    assert ds.metadata[ACQUISITION_DATE_KEY] == REAL_ISO
    assert ds.metadata[ACQUISITION_SPAN_KEY] == pytest.approx(3240.4)
    assert dataset_acquisition_text(ds) == "2026-Apr-21,13:09:33 ~ 14:03:33 (span 54 min)"


def test_stamping_a_missing_date_is_a_no_op():
    meta: dict = {}
    stamp_acquisition(meta, None)
    stamp_acquisition(meta, "")
    assert meta == {}


def test_a_dataset_without_an_acquisition_date_reports_unknown():
    ds = _make_ds(stamp=None)
    assert dataset_acquisition_date(ds) is None
    assert dataset_acquisition_text(ds) == "—"


def test_recorded_span_survives_losing_rows():
    """The span is stored, not recomputed — a crop must not shrink the run."""
    ds = _make_ds(tim_max=3240.4)
    # Simulate a crop/subset: the live tim column no longer covers the run.
    ds.attr["tim"] = np.asarray(ds.attr["tim"])[:5]
    if getattr(ds, "mfx_raw", None) is not None and ds.mfx_raw.get("tim") is not None:
        ds.mfx_raw["tim"] = np.asarray(ds.mfx_raw["tim"])[:5]
    assert dataset_acquisition_span(ds) == pytest.approx(3240.4)


# ---------------------------------------------------------------------------
# Metadata sidecar round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content", ["raw", "snapshot"])
def test_sidecar_carries_the_acquisition_for_both_contents(content):
    """Nothing about a date is 'baked', so unlike z_scaling_factor it survives a snapshot."""
    ds = _make_ds()
    meta = build_metadata(ds, content=content)
    assert meta["acquisition"]["date"] == REAL_ISO
    assert meta["acquisition"]["span_s"] == pytest.approx(3240.4)
    assert meta["acquisition"]["end"].startswith("2026-04-21T14:03:33")
    json.dumps(meta)          # must stay JSON-serializable


def test_sidecar_restores_the_acquisition_onto_a_fresh_dataset():
    source = _make_ds()
    payload = json.loads(json.dumps(build_metadata(source)))

    fresh = _make_ds(stamp=None)
    assert dataset_acquisition_date(fresh) is None

    applied = L.apply_metadata_recipe(fresh, payload)
    assert "acquisition date" in applied
    assert fresh.metadata[ACQUISITION_DATE_KEY] == REAL_ISO
    assert dataset_acquisition_text(fresh) == dataset_acquisition_text(source)


def test_sidecar_without_an_acquisition_block_is_harmless():
    ds = _make_ds(stamp=None)
    applied = L.apply_metadata_recipe(ds, {"calibration": {}})
    assert "acquisition date" not in applied
    assert ACQUISITION_DATE_KEY not in ds.metadata


def test_acquisition_metadata_is_empty_without_a_date():
    assert acquisition_metadata(_make_ds(stamp=None)) == {}


# ---------------------------------------------------------------------------
# .msr writer round trip
# ---------------------------------------------------------------------------
def test_msr_writer_round_trips_the_acquisition_date(tmp_path):
    from minflux_viewer.msr.io import read_zarr_attrs
    from minflux_viewer.msr.mfxdta import extract_zarr_store, read_obf_mfxdta_stacks
    from minflux_viewer.msr.writer import write_datasets_msr

    ds = _make_ds()
    out = write_datasets_msr(tmp_path / "rt.msr", [ds])

    (_, _, blob), = read_obf_mfxdta_stacks(out)
    attrs = read_zarr_attrs(extract_zarr_store(blob), "mfx")
    assert attrs.get("acquisition_date") == REAL_ISO
    # ...and the reader-side resolver picks it back up unchanged.
    assert acquisition_date_from_zattrs(attrs) == REAL_ISO


def test_msr_writer_omits_the_attr_when_there_is_no_date(tmp_path):
    from minflux_viewer.msr.io import read_zarr_attrs
    from minflux_viewer.msr.mfxdta import extract_zarr_store, read_obf_mfxdta_stacks
    from minflux_viewer.msr.writer import write_datasets_msr

    out = write_datasets_msr(tmp_path / "bare.msr", [_make_ds(stamp=None)])
    (_, _, blob), = read_obf_mfxdta_stacks(out)
    attrs = read_zarr_attrs(extract_zarr_store(blob), "mfx")
    assert "acquisition_date" not in attrs


# ---------------------------------------------------------------------------
# Dataset Information rows
# ---------------------------------------------------------------------------
def test_data_window_row_order_and_labels():
    from minflux_viewer.ui.data_window import _info_rows

    rows = _info_rows(_make_ds())
    labels = [label for label, _ in rows]
    assert labels.index("Acquisition") == labels.index("Folder") + 1
    assert labels.index("File created") == labels.index("Acquisition") + 1
    # 'Created' was renamed so it cannot be mistaken for the acquisition time.
    assert "Created" not in labels

    value = dict(rows)["Acquisition"]
    assert value == "2026-Apr-21,13:09:33 ~ 14:03:33 (span 54 min)"


def test_dims_row_double_click_sets_z_scaling_factor(qtbot, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import data_window as data_window_module
    from minflux_viewer.ui.data_window import DataWindow

    ds = _make_ds()
    state = AppState()
    state.add_dataset(ds)
    changed: list[int] = []
    state.calibration_changed.connect(changed.append)
    monkeypatch.setattr(
        data_window_module.ZScalingFactorDialog,
        "ask",
        staticmethod(lambda *_args, **_kwargs: (0.73, True)),
    )

    window = DataWindow(ds, 0, state)
    qtbot.addWidget(window)
    window.show()
    QTest.mouseDClick(
        window._value_labels["Dims"], Qt.MouseButton.LeftButton
    )

    assert ds.cali.z_scaling_factor == pytest.approx(0.73)
    assert ds.z_scaling_factor_provenance["source"] == "manual (Dataset Information)"
    assert "Z scaling factor = 0.73 (manual)" in window._value_labels["Dims"].text()
    assert changed == [0]
    assert state.active_idx == 0
    assert "(manual, Dataset Information)" in state.log_history[-1]["message"]


def test_transformed_row_double_click_edits_canonical_overlay_matrix(qtbot, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.overlay import matrix4_to_xy3
    from minflux_viewer.ui import transform_dialog as transform_module
    from minflux_viewer.ui.data_window import DataWindow

    ds = _make_ds()
    original_matrix = np.eye(4)
    original_matrix[0, 3] = 12.0
    original = {
        "overlay_id": "overlay-1",
        "matrix_4x4": original_matrix.tolist(),
        "matrix_3x3": matrix4_to_xy3(original_matrix).tolist(),
    }
    ds.state["overlay_transform"] = original
    ds.state["render_transform_2d"] = original

    edited_matrix = original_matrix.copy()
    edited_matrix[:3, 3] = (25.0, -8.0, 3.5)

    class FakeTransformDialog:
        manual_alignment_requested = False

        def __init__(self, transform, **_kwargs):
            self._transform = transform

        @staticmethod
        def exec():
            return QDialog.DialogCode.Accepted

        def updated_record(self):
            return transform_module.updated_transform_record(
                self._transform, edited_matrix
            )

    monkeypatch.setattr(transform_module, "TransformDialog", FakeTransformDialog)

    state = AppState()
    state.add_dataset(ds)
    changed: list[int] = []
    state.overlay_transform_changed.connect(changed.append)
    window = DataWindow(ds, 0, state)
    qtbot.addWidget(window)
    window.show()

    assert window._value_labels["Transformed"].text() == "yes"
    QTest.mouseDClick(
        window._value_labels["Transformed"], Qt.MouseButton.LeftButton
    )

    record = ds.state["overlay_transform"]
    assert np.asarray(record["matrix_4x4"])[:3, 3] == pytest.approx(
        (25.0, -8.0, 3.5)
    )
    assert ds.state["render_transform_2d"] is record
    assert record["overlay_id"] == "overlay-1"
    assert "overlay_transform" not in ds.metadata
    assert changed == [0]
    assert "edited in Dataset Information" in state.log_history[-1]["message"]


def test_transform_dialog_manual_align_button_uses_state_request(qtbot, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import transform_dialog as transform_module
    from minflux_viewer.ui.data_window import DataWindow

    ds = _make_ds()
    matrix = np.eye(4)
    matrix[1, 3] = 4.0
    ds.state.update(
        {
            "overlay_id": "overlay-1",
            "overlay_transform": {"matrix_4x4": matrix.tolist()},
        }
    )
    peer = _make_ds()
    peer.file.name = "peer.mfx"
    peer.state["overlay_id"] = "overlay-1"

    class FakeTransformDialog:
        manual_alignment_requested = True

        def __init__(self, *_args, **kwargs):
            assert kwargs["manual_align_enabled"] is True

        @staticmethod
        def exec():
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(transform_module, "TransformDialog", FakeTransformDialog)

    state = AppState()
    state.add_dataset(ds)
    state.add_dataset(peer)
    requested: list[int] = []
    state.overlay_manual_alignment_requested.connect(requested.append)
    window = DataWindow(ds, 0, state)
    qtbot.addWidget(window)
    window.show()
    QTest.mouseDClick(
        window._value_labels["Transformed"], Qt.MouseButton.LeftButton
    )

    assert requested == [0]


def test_transform_description_reports_xyz_direction_and_screen_rotation():
    from minflux_viewer.ui.transform_dialog import transform_description

    matrix = np.eye(4)
    matrix[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
    matrix[:3, 3] = (1.0, -2.0, 3.0)

    text = transform_description(matrix, xy_origin_top_left=True)

    assert "X  +1 nm — toward +X (right)" in text
    assert "Y  -2 nm — toward −Y (up in a top-left-origin XY view)" in text
    assert "Z  +3 nm — toward +Z (positive axial direction)" in text
    assert "Z  +90° — counter-clockwise" in text
    assert "appears clockwise in a top-left-origin XY view" in text


def test_transform_dialog_edits_full_xyz_matrix_and_preserves_record(qtbot):
    from minflux_viewer.ui.transform_dialog import TransformDialog

    matrix = np.eye(4)
    matrix[0, 3] = 5.0
    dialog = TransformDialog(
        {"overlay_id": "overlay-7", "matrix_4x4": matrix.tolist()},
        dataset_name="channel 2",
        xy_origin_top_left=True,
        manual_align_enabled=True,
    )
    qtbot.addWidget(dialog)

    dialog._matrix_spins[0][3].setValue(11.0)
    dialog._matrix_spins[2][3].setValue(-4.0)
    record = dialog.updated_record()

    assert record["overlay_id"] == "overlay-7"
    assert np.asarray(record["matrix_4x4"])[:3, 3] == pytest.approx(
        (11.0, 0.0, -4.0)
    )
    assert np.asarray(record["matrix_3x3"])[:, 2] == pytest.approx(
        (11.0, 0.0, 1.0)
    )
    assert record["alignment_mode"] == "manual matrix"
    assert "Z  -4 nm" in dialog._description.text()
