"""The single registry of formats this application reads and writes.

Its whole purpose is that the format tables scattered across the writers, the
save dialog, Preferences, the MSR reader and the drag-and-drop router stop
disagreeing. So the tests pin it against each of those tables: if one drifts,
this fails rather than a user discovering it.
"""

import inspect
import json

import pytest

from minflux_viewer.core import formats as F
from minflux_viewer.core.formats import OpenAction, resolve_open


# --------------------------------------------------------------------------- #
# Agreement with the tables it replaces
# --------------------------------------------------------------------------- #
def test_matches_the_writer_table():
    from minflux_viewer.core.save import DATA_FORMATS, _RAW_ONLY_FORMATS

    assert {spec.key for spec in F.save_formats()} == set(DATA_FORMATS)
    assert F.raw_only_formats() == _RAW_ONLY_FORMATS


def test_matches_the_fresh_install_preference():
    from minflux_viewer.core.app_state import DEFAULT_PREFS

    assert set(F.default_save_formats()) == set(DEFAULT_PREFS["data"]["export_formats"])


def test_matches_the_router_tables():
    from minflux_viewer.ui.main_window import _SUPPORTED_EXTS, MainWindow

    assert set(F.supported_extensions()) == set(_SUPPORTED_EXTS)
    # A Dataset-Manager row accepts two different verbs: the apply-to-this-
    # dataset kinds, and every openable data format (which the row cannot
    # receive, so it is opened as a new dataset instead of being refused).
    assert set(MainWindow.DROP_ON_DATASET_EXTS) == (
        set(F.drop_on_dataset_extensions()) | set(F.supported_extensions()))
    assert set(F.drop_on_dataset_extensions()) <= set(MainWindow.DROP_ON_DATASET_EXTS)


def test_msr_export_is_available_but_not_ticked():
    """The writer is reverse-engineered, so it is opt-in."""
    msr = next(spec for spec in F.FORMATS if spec.key == "msr")
    assert msr.writable and not msr.default_offered
    assert "msr" not in F.default_save_formats()


def test_npz_reads_but_no_longer_writes():
    npz = next(spec for spec in F.FORMATS if spec.key == "npz")
    assert npz.readable and not npz.writable
    assert ".npz" in F.supported_extensions()


# --------------------------------------------------------------------------- #
# Routing: opening a file is not always "load a dataset"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,action", [
    ("acq.msr", OpenAction.MSR_READER),
    ("data.mat", OpenAction.DATASET),
    ("data.npy", OpenAction.DATASET),
    ("store.zarr", OpenAction.DATASET),
    ("old.npz", OpenAction.DATASET),
    ("image.tif", OpenAction.IMAGE_VIEWER),
    ("table.csv", OpenAction.SPREADSHEET_DIALOG),
    ("book.xlsx", OpenAction.SPREADSHEET_DIALOG),
    ("set.roi", OpenAction.ROI_MANAGER),
    ("set.zip", OpenAction.ROI_MANAGER),
])
def test_extension_routes_to_its_action(tmp_path, name, action):
    path = tmp_path / name
    path.write_bytes(b"")
    spec = resolve_open(path)
    assert spec is not None and spec.action is action


def test_unsupported_extension_resolves_to_nothing(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"")
    assert resolve_open(path) is None


# --------------------------------------------------------------------------- #
# The .json fork -- the reason content detection exists
# --------------------------------------------------------------------------- #
def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_roi_json_goes_to_the_roi_manager(tmp_path):
    path = _write(tmp_path / "rois.json", {"version": 1, "rois": []})
    assert resolve_open(path).action is OpenAction.ROI_MANAGER


def test_filter_json_goes_to_the_filter_dialog(tmp_path):
    path = _write(tmp_path / "preset.json", [
        {"apply": True, "attribute": "efo", "value_as": "per loc",
         "min": 1.0, "max": 10.0},
    ])
    assert resolve_open(path).action is OpenAction.FILTER_DIALOG


def test_metadata_sidecar_is_a_recipe_not_data(tmp_path):
    from minflux_viewer.core.save import METADATA_JSON_MARKER

    path = _write(tmp_path / "x_metadata.json",
                  {METADATA_JSON_MARKER: 1, "content": "raw"})
    assert resolve_open(path).action is OpenAction.METADATA_RECIPE


def test_plain_localization_json_is_still_data(tmp_path):
    """Data is the fallback: it is the only .json with no positive marker.

    The old router tried to load data *first* and looked for a filter preset
    only if that raised, so a malformed data file was indistinguishable from a
    filter file. Detection is now explicit and ordered.
    """
    path = _write(tmp_path / "locs.json", [
        {"vld": True, "tid": 1, "itr": 3, "loc": [1e-9, 2e-9, 0.0], "efo": 10.0},
    ])
    spec = resolve_open(path)
    assert spec is not None and spec.action is OpenAction.DATASET
    assert spec.key == "json"


def test_json_detection_order_is_the_strongest_claim_first():
    """Self-identifying first, shape guesses next, plain data last.

    ⚠ The metadata sidecar is the only ``.json`` kind that names itself, with a
    dedicated top-level marker. The other two are guesses about shape — and a
    sidecar legitimately *contains* both a ``"rois"`` list and filter specs, so
    on shape alone it looks like either of them. Probing the ROI test first made
    every sidecar route to the ROI Manager.
    """
    order = {spec.key: spec.detect_order
             for spec in F.FORMATS if ".json" in spec.extensions}
    assert order["metadata_sidecar"] < order["roi_set"] < order["filter_preset"]
    assert order["json"] == max(order.values()), "plain data must be probed last"


def test_a_metadata_sidecar_is_not_mistaken_for_the_roi_set_it_contains(tmp_path):
    """The reported bug: saving in any format then re-opening the sidecar."""
    import json

    from minflux_viewer.core.roi import is_roi_json_file

    sidecar = tmp_path / "run_metadata.json"
    sidecar.write_text(json.dumps({
        "minflux_viewer_metadata": 1,
        "content": "raw",
        "name": "run.msr | channel",
        "filters": [{"attribute": "efo", "lo": 1.0, "hi": 2.0}],
        # A sidecar carries the dataset's whole ROI set — the shape a ROI-set
        # file has, which is why the shape test alone is not enough.
        "rois": [{"id": "a", "name": "rectangle-1", "type": "rectangle",
                  "geometry": {"bounds": [0, 0, 1, 1]}, "dataset_id": "d0"}],
    }), encoding="utf-8")

    assert is_roi_json_file(sidecar) is False
    spec = resolve_open(sidecar)
    assert spec.key == "metadata_sidecar"
    assert spec.action is F.OpenAction.METADATA_RECIPE


def test_a_real_roi_set_still_routes_to_the_manager(tmp_path):
    import json

    from minflux_viewer.core.roi import is_roi_json_file

    roi_set = tmp_path / "regions.json"
    roi_set.write_text(json.dumps({"version": 1, "rois": [
        {"id": "a", "name": "rectangle-1", "type": "rectangle",
         "geometry": {"bounds": [0, 0, 1, 1]}}]}), encoding="utf-8")
    assert is_roi_json_file(roi_set) is True
    assert resolve_open(roi_set).action is F.OpenAction.ROI_MANAGER


def test_an_unreadable_json_falls_through_rather_than_aborting(tmp_path):
    """A predicate that raises means "not this kind", not "stop routing"."""
    path = tmp_path / "broken.json"
    path.write_text("{ this is not json", encoding="utf-8")
    spec = resolve_open(path)
    assert spec is not None and spec.key == "json"


# --------------------------------------------------------------------------- #
# Compound extensions -- several unrelated formats share the zip container
# --------------------------------------------------------------------------- #
def test_sealed_zarr_package_is_not_mistaken_for_a_roiset(tmp_path):
    """``.zarr.zip`` must beat ``.zip``.

    ``Path.suffix`` is ``.zip``, so a sealed store resolved to the RoiSet format
    and a dropped package went to the ROI Manager.
    """
    package = tmp_path / "acquisition.zarr.zip"
    package.write_bytes(b"")
    spec = resolve_open(package)
    assert spec is not None
    assert spec.key == "zarr_zip"
    assert spec.action is OpenAction.DATASET

    roiset = tmp_path / "rois.zip"
    roiset.write_bytes(b"")
    assert resolve_open(roiset).action is OpenAction.ROI_MANAGER


def test_sealed_package_is_offered_for_saving():
    """It saves the same content as the directory store, in one file."""
    spec = next(s for s in F.FORMATS if s.key == "zarr_zip")
    assert spec.readable and spec.writable and spec.default_offered
    assert "zarr_zip" in {s.key for s in F.save_formats()}
    # Raw-canonical plus separate processing state, never a baked snapshot.
    assert spec.raw_only and "zarr_zip" in F.raw_only_formats()


def test_only_the_directory_store_takes_a_processing_only_update():
    """A zip appends rather than replaces a member, so it is always rewritten.

    This is the one behavioural difference between the two Zarr formats, and it
    is why both exist.
    """
    from minflux_viewer.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._save_as_format)
    assert 'self._zarr_overwrite_mode(path) if fmt == "zarr" else "replace"' in source


@pytest.mark.parametrize("stem,expected", [
    ("run", "run.zarr.zip"),
    ("run.zarr", "run.zarr.zip"),
    ("run.zarr.zip", "run.zarr.zip"),
])
def test_compound_extension_is_applied_without_doubling(stem, expected):
    """``with_suffix`` replaces only the last suffix.

    It turns ``run.zarr`` into ``run.zarr.zip`` correctly but also turns an
    already correct ``run.zarr.zip`` into ``run.zarr.zarr.zip``.
    """
    assert F.normalize_path("zarr_zip", stem).name == expected
    assert F.normalize_path("zarr", "run.zarr.zip").name == "run.zarr"


def test_compound_extension_outranks_the_container_magic_number(tmp_path):
    """xlsx, a RoiSet and a sealed store are all zips.

    The magic number therefore only says "this is a zip", and must not override
    an explicit ``.zarr.zip``. Before this, sniffing reported
    "extension '.zarr.zip' but the content is xlsx" and routed a sealed store
    into the spreadsheet column-mapping dialog.
    """
    import zipfile

    from minflux_viewer.core.format_sniff import resolve_format

    package = tmp_path / "sealed.zarr.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(".zgroup", '{"zarr_format": 2}')

    fmt, note = resolve_format(package)
    assert fmt == "zarr"
    assert note == "", f"a declared compound extension needs no override: {note}"


def test_a_real_xlsx_is_still_a_spreadsheet(tmp_path):
    """The compound rule must not swallow ordinary single-suffix files."""
    from minflux_viewer.core.format_sniff import resolve_format

    book = tmp_path / "table.xlsx"
    book.write_bytes(b"")
    assert resolve_format(book)[0] == "spreadsheet"


# --------------------------------------------------------------------------- #
# The GUI route: what a dropped file actually does
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _minflux_raw(n_loc=8, n_itr=2):
    import numpy as np

    dtype = np.dtype([
        ("vld", np.bool_), ("tid", np.int64), ("tim", np.float64),
        ("itr", np.int32), ("efo", np.float32),
        ("dcr", np.float32, (2,)), ("loc", np.float64, (3,)),
    ])
    raw = np.zeros(n_loc * n_itr, dtype=dtype)
    raw["vld"] = True
    raw["tid"] = np.repeat(np.arange(n_loc), n_itr)
    raw["tim"] = np.repeat(np.linspace(0.0, 3.0, n_loc), n_itr)
    raw["itr"] = np.tile(np.arange(n_itr), n_loc)
    raw["efo"] = np.arange(n_loc * n_itr, dtype=np.float32) + 5.0
    raw["loc"][:, 0] = np.repeat(np.linspace(0, 4e-7, n_loc), n_itr)
    raw["loc"][:, 1] = np.repeat(np.linspace(0, 3e-7, n_loc), n_itr)
    return raw


def test_dropping_a_sealed_package_opens_it_as_a_dataset(_app, tmp_path):
    """The whole point of the sealed package: drop it and it opens.

    It resolved to the RoiSet format before (``Path.suffix`` is ``.zip``), so a
    dropped package opened the ROI Manager and no dataset appeared.
    """
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.minflux_zarr import (
        pack_minflux_zarr,
        write_minflux_zarr,
    )
    from minflux_viewer.ui.main_window import MainWindow

    ds = load_from_mfx_array(_minflux_raw(), name="sealed", folder=str(tmp_path))
    ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                 "itr": "last", "lo": 1.0, "hi": 500.0,
                                 "lo_inc": True, "hi_inc": True}]
    package = pack_minflux_zarr(write_minflux_zarr(ds, tmp_path / "run.zarr"))

    window = MainWindow(AppState())
    try:
        before = len(window._state.datasets)
        window._route_file(str(package))
        # Opening a store runs on a worker thread now.
        from PyQt6.QtWidgets import QApplication
        import time

        deadline = time.time() + 60.0
        while time.time() < deadline and getattr(window, "_zarr_io_tasks", []):
            QApplication.instance().processEvents()
            time.sleep(0.02)
        QApplication.instance().processEvents()
        assert len(window._state.datasets) == before + 1, (
            "a sealed .zarr.zip must open as a dataset, not go to the ROI Manager")
        opened = window._state.datasets[-1]
        assert opened.prop.num_loc == ds.prop.num_loc
        assert opened.state["filter_specs"] == ds.state["filter_specs"]
    finally:
        window.close()


def test_saving_a_sealed_package_matches_the_directory_store(tmp_path):
    """Both Zarr forms carry identical content; only the container differs."""
    import numpy as np

    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.minflux_zarr import load_minflux_zarr_project
    from minflux_viewer.core.save import save_processed

    ds = load_from_mfx_array(_minflux_raw(), name="both", folder=str(tmp_path))
    ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                 "itr": "last", "lo": 1.0, "hi": 500.0,
                                 "lo_inc": True, "hi_inc": True}]

    written = {}
    for fmt, stem in (("zarr", "as_dir"), ("zarr_zip", "as_file")):
        paths = save_processed(
            ds, data_path=tmp_path / stem, fmt=fmt, content="raw",
            include={"attrs": True, "derived": False, "recipe": True},
        )
        # Both are self-contained: the processing lives inside, so neither
        # writes the sidecar the other formats rely on.
        assert not any(p.name.endswith("_metadata.json") for p in paths)
        written[fmt] = paths[0]

    assert written["zarr"].is_dir()
    assert written["zarr_zip"].is_file()
    assert written["zarr_zip"].name == "as_file.zarr.zip"

    directory = load_minflux_zarr_project(written["zarr"]).datasets[0]
    sealed = load_minflux_zarr_project(written["zarr_zip"]).datasets[0]
    for key in directory.components.mfx_raw.keys():
        np.testing.assert_array_equal(
            np.asarray(directory.components.mfx_raw.get(key)),
            np.asarray(sealed.components.mfx_raw.get(key)),
        )
    assert sealed.state["filter_specs"] == ds.state["filter_specs"]


def test_a_sealed_package_refuses_a_baked_snapshot(tmp_path):
    """Raw-canonical plus separate state, like the directory store."""
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.save import save_processed

    ds = load_from_mfx_array(_minflux_raw(), name="snap", folder=str(tmp_path))
    with pytest.raises(ValueError, match=r"\.zarr\.zip stores canonical raw"):
        save_processed(ds, data_path=tmp_path / "s", fmt="zarr_zip",
                       content="snapshot", include={"recipe": True})
