"""Standalone MBM (bead) companion files open as bead info, not as a dataset.

``.mat`` / ``.npy`` / ``.json`` / ``.csv`` cannot hold localizations and their
beam-monitoring reference in one file, so the MSR reader writes the beads as a
``<stem>_mbm.<ext>`` companion. Those files used to be unopenable: the ``.mat``
and ``.json`` were refused as "not a MINFLUX dataset" and the ``.npy`` loaded as
an empty dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.formats import OpenAction, resolve_open
from minflux_viewer.core.mbm_file import (
    is_mbm_points_fields,
    is_mbm_points_file,
    read_mbm_points,
)
from minflux_viewer.core.save import write_raw_array
from minflux_viewer.msr.legacy_mbm import POINTS_DTYPE

_FORMATS = ("mat", "npy", "json", "csv")


def _bead_points(n_beads: int = 3, per_bead: int = 10) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = n_beads * per_bead
    centres = np.array([[1e-5, 2e-5, 0.0], [3e-5, 1e-5, 0.0], [2e-5, 3e-5, 0.0]])
    points = np.zeros(n, dtype=POINTS_DTYPE)
    points["gri"] = np.repeat(np.arange(101, 101 + n_beads), per_bead)
    points["xyz"] = (np.repeat(centres[:n_beads], per_bead, axis=0)
                     + rng.normal(0.0, 1e-8, (n, 3)))
    points["tim"] = np.tile(np.arange(per_bead, dtype=float), n_beads)
    points["str"] = rng.random(n)
    return points


def _mfx_dataset(tmp_path):
    from minflux_viewer.msr.export import canonical_dataset

    rng = np.random.default_rng(11)
    dtype = np.dtype([("itr", "i4"), ("tid", "u4"), ("tim", "f8"), ("vld", "?"),
                      ("loc", ("f8", 3)), ("gri", "i8")])
    arr = np.zeros(24, dtype=dtype)
    arr["vld"] = True
    arr["loc"] = rng.random((24, 3)) * 1e-6
    return canonical_dataset(arr, name="run.msr", folder=str(tmp_path),
                             mbm=None, mbm_meta=None)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_companion_round_trips_through_every_written_format(tmp_path, fmt):
    points = _bead_points()
    path = write_raw_array(tmp_path / "run_mbm", fmt, points, root="mbm")

    assert is_mbm_points_file(path) is True
    got = read_mbm_points(path)
    assert got.size == points.size
    assert np.array_equal(got["gri"], points["gri"])
    assert np.allclose(got["xyz"], points["xyz"])
    assert np.allclose(got["tim"], points["tim"])
    assert np.allclose(got["str"], points["str"])


@pytest.mark.parametrize("fmt", _FORMATS)
def test_companion_routes_to_the_mbm_info_window(tmp_path, fmt):
    path = write_raw_array(tmp_path / "run_mbm", fmt, _bead_points(), root="mbm")
    spec = resolve_open(path)
    assert spec is not None
    assert spec.action is OpenAction.MBM_INFO


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_localization_file_is_never_mistaken_for_a_bead_table(tmp_path, fmt):
    """``gri`` appears in an mfx table too, so the localization markers decide."""
    from minflux_viewer.core.save import save_processed

    paths = save_processed(
        _mfx_dataset(tmp_path), data_path=tmp_path / f"mfx_{fmt}", fmt=fmt,
        content="raw", include={"attrs": True, "derived": False, "recipe": False})
    assert is_mbm_points_file(paths[0]) is False
    # .csv keeps its own route (the spreadsheet importer, which recognises a
    # canonical MINFLUX header and loads it directly); the rest load as data.
    expected = (OpenAction.SPREADSHEET_DIALOG if fmt == "csv"
                else OpenAction.DATASET)
    assert resolve_open(paths[0]).action is expected


@pytest.mark.parametrize("fmt", _FORMATS)
def test_companion_reconstructs_bead_traces_without_any_localizations(tmp_path, fmt):
    """The whole point: bead drift is readable with no dataset in hand."""
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift

    path = write_raw_array(tmp_path / "run_mbm", fmt, _bead_points(), root="mbm")
    beads = extract_bead_drift(read_mbm_points(path), {}, [])

    assert [bead["rid"] for bead in beads] == ["101", "102", "103"]
    assert all(bead["xyz_nm"].shape == (10, 3) for bead in beads)
    # Re-zeroed to the per-bead median, so the trace is drift, not position.
    assert all(abs(np.median(bead["xyz_nm"], axis=0)).max() < 1e-6 for bead in beads)


def test_field_predicate_needs_position_and_time_and_rejects_localizations():
    assert is_mbm_points_fields(["gri", "xyz", "tim", "str"]) is True
    assert is_mbm_points_fields(["gri", "xyz_0", "xyz_1", "xyz_2", "tim"]) is True
    assert is_mbm_points_fields(["GRI", "XYZ", "TIM"]) is True
    # Missing the position column.
    assert is_mbm_points_fields(["gri", "tim", "str"]) is False
    # A localization marker overrides everything else.
    assert is_mbm_points_fields(["gri", "xyz", "tim", "vld"]) is False
    assert is_mbm_points_fields(["gri", "xyz", "tim", "loc_x"]) is False


def test_classification_reads_only_a_prefix_of_a_json(tmp_path, monkeypatch):
    """The predicate runs on every .json the router sees, including huge ones."""
    import minflux_viewer.core.mbm_file as mod

    path = write_raw_array(tmp_path / "run_mbm", "json", _bead_points(), root="mbm")

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the classifier must not read the whole file")

    monkeypatch.setattr(mod.Path, "read_text", _refuse)
    assert is_mbm_points_file(path) is True


def test_reading_a_non_bead_file_says_so_instead_of_returning_nothing(tmp_path):
    path = tmp_path / "not_mbm.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an MBM bead table"):
        read_mbm_points(path)


# --- the MBM info window opened from a companion -------------------------

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(qtbot, *, data_bounds_nm=None):
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift
    from minflux_viewer.ui.mbm_info_window import MbmInfoWindow

    beads = extract_bead_drift(_bead_points(), {}, [])
    window = MbmInfoWindow("run_mbm.npy", beads, data_bounds_nm=data_bounds_nm)
    qtbot.addWidget(window)
    return window


def test_the_window_asks_for_a_size_its_content_can_actually_take(_app, qtbot):
    """Requesting less than the content needs made Qt enlarge it on show.

    Two full dialogs are embedded as tab pages; the nominal 820 was below their
    combined ``sizeHint``, so the window silently grew — and on Windows the move
    that places it then printed ``QWindowsWindow::setGeometry: Unable to set
    geometry ...``, because the size being re-sent was one the window could not
    take.
    """
    window = _window(qtbot)
    hint = window.sizeHint()
    assert window.width() >= hint.width()
    assert window.height() >= hint.height()
    assert window.width() >= window.PREFERRED_SIZE[0]
    assert window.height() >= window.PREFERRED_SIZE[1]

    # The size must survive being shown and then moved — that is the sequence
    # (``show_modeless`` -> ``ensure_on_screen``) that produced the warning.
    before = (window.width(), window.height())
    window.show()
    qtbot.waitExposed(window)
    assert (window.width(), window.height()) == before
    window.move(120, 90)
    _app.processEvents()
    assert (window.width(), window.height()) == before


def test_a_companion_says_why_there_is_no_data_region_box(_app, qtbot):
    """The box is the localization extent, which a bead file does not carry."""
    from PyQt6.QtWidgets import QLabel

    from minflux_viewer.ui.mbm_info_window import NO_DATA_REGION_NOTE

    window = _window(qtbot)
    page = window._tabs.widget(1)
    notes = [label.text() for label in page.findChildren(QLabel)
             if label.text() == NO_DATA_REGION_NOTE]
    assert notes, "an unexplained missing box reads as a bug"
    assert "_mfx" in NO_DATA_REGION_NOTE       # names where the extent lives


def test_a_loaded_dataset_still_gets_its_data_region_and_no_note(_app, qtbot):
    import numpy as np
    from PyQt6.QtWidgets import QLabel

    from minflux_viewer.ui.mbm_info_window import NO_DATA_REGION_NOTE

    bounds = (np.array([1e4, 1e4, -50.0]), np.array([3.2e4, 3.2e4, 50.0]))
    window = _window(qtbot, data_bounds_nm=bounds)
    page = window._tabs.widget(1)
    assert not [label for label in page.findChildren(QLabel)
                if label.text() == NO_DATA_REGION_NOTE]


def test_the_bead_companion_formats_cannot_carry_a_data_region():
    """Why the box is missing, pinned as a fact rather than a claim.

    ``write_raw_array`` writes one structured component. ``.npy`` is a bare
    array with no free-form header, and ``.csv`` is flat columns — neither has
    anywhere to put a bounding box without changing the format. (``.mat`` could
    take a second variable and ``.json`` could be wrapped in an object, but a
    field only two of the four formats can hold is not a uniform one.)
    """
    import numpy as np

    from minflux_viewer.core.save import _EXT, _write_npy

    tmp = pytest.importorskip("tempfile")
    import pathlib

    folder = pathlib.Path(tmp.mkdtemp())
    points = _bead_points()
    path = folder / "probe.npy"
    _write_npy(path, points)

    from numpy.lib import format as npy_format

    with open(path, "rb") as handle:
        version = npy_format.read_magic(handle)
        shape, _order, dtype = (npy_format.read_array_header_1_0(handle)
                                if version[0] == 1 else
                                npy_format.read_array_header_2_0(handle))
    # The whole .npy header: a dtype, a shape and a byte order. Nowhere else.
    assert shape == (points.size,)
    assert set(dtype.names) == {"gri", "xyz", "tim", "str"}
    assert ".npy" == _EXT["npy"]
