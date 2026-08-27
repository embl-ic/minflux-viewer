"""MSR reader batch export: file discovery, output layout, drop targets.

The reported failure was a batch run over a folder whose ``.msr`` files all sit
in sub-folders: the search was top-level only, found nothing, logged one line
and returned, so the export button appeared to do nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QMimeData, QUrl
from PyQt6.QtWidgets import QApplication

from minflux_viewer.plugins.msr_reader.msr_reader_dialog import (
    MsrReaderDialog,
    PathDropLineEdit,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tree(tmp_path):
    """A folder whose .msr files live only in sub-folders — the reported case."""
    (tmp_path / "20260624").mkdir()
    (tmp_path / "20260625" / "nested").mkdir(parents=True)
    made = [
        tmp_path / "20260624" / "a.msr",
        tmp_path / "20260625" / "b.msr",
        tmp_path / "20260625" / "nested" / "c.MSR",
    ]
    for path in made:
        path.write_bytes(b"")
    (tmp_path / "20260624" / "notes.txt").write_text("ignore me")
    return tmp_path, made


def test_recursive_search_finds_files_in_subfolders(_app, tree):
    root, made = tree
    dlg = MsrReaderDialog(state=None)
    try:
        assert dlg._find_msr_files(str(root), recursive=False) == []
        found = dlg._find_msr_files(str(root), recursive=True)
        assert sorted(p.name for p in found) == sorted(p.name for p in made)
    finally:
        dlg.close()


def test_search_is_case_insensitive_without_duplicating(_app, tmp_path):
    (tmp_path / "upper.MSR").write_bytes(b"")
    (tmp_path / "lower.msr").write_bytes(b"")
    dlg = MsrReaderDialog(state=None)
    try:
        found = dlg._find_msr_files(str(tmp_path), recursive=False)
        assert sorted(p.name for p in found) == ["lower.msr", "upper.MSR"]
    finally:
        dlg.close()


def test_search_follows_the_checkbox_when_not_told_otherwise(_app, tree):
    root, made = tree
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.recursive_check.setChecked(False)
        assert dlg._find_msr_files(str(root)) == []
        dlg.recursive_check.setChecked(True)
        assert len(dlg._find_msr_files(str(root))) == len(made)
    finally:
        dlg.close()


def test_a_file_path_searches_its_parent_folder(_app, tree):
    root, made = tree
    dlg = MsrReaderDialog(state=None)
    try:
        found = dlg._find_msr_files(str(made[0]), recursive=False)
        assert [p.name for p in found] == ["a.msr"]
    finally:
        dlg.close()


def test_every_input_file_gets_a_folder_named_after_it(_app):
    """Exports are named after the datasets *inside* the .msr, so each input
    needs its own folder for its outputs to be attributable."""
    dlg = MsrReaderDialog(state=None)
    try:
        in_root = Path("root")
        msr = in_root / "1st_dir" / "2nd_dir" / "test.msr"
        out = Path("out")

        # Mirror on: the relative path, then the file's own folder.
        assert (dlg._batch_output_dir(out, msr, in_root, True)
                == out / "1st_dir" / "2nd_dir" / "test")

        # Mirror off: the file's own folder directly under the output folder.
        assert dlg._batch_output_dir(out, msr, in_root, False) == out / "test"

        # A file at the batch root still gets its own folder.
        assert (dlg._batch_output_dir(out, in_root / "top.msr", in_root, True)
                == out / "top")
    finally:
        dlg.close()


def test_output_of_a_file_outside_the_root_stays_addressable(_app):
    dlg = MsrReaderDialog(state=None)
    try:
        target = dlg._batch_output_dir(
            Path("out"), Path("elsewhere") / "x.msr", Path("root"), True)
        assert target == Path("out") / "x"
    finally:
        dlg.close()


def test_msr_is_no_longer_an_export_format(_app):
    dlg = MsrReaderDialog(state=None)
    try:
        assert not hasattr(dlg, "fmt_msr")
        # The remaining formats are unchanged.
        for name in ("fmt_mat", "fmt_npy", "fmt_json", "fmt_csv", "fmt_zarr"):
            assert hasattr(dlg, name)
        dlg._save_settings()
        assert "msr" not in dlg._load_settings("")["formats"]
    finally:
        dlg.close()


def test_npz_still_opens_but_is_no_longer_offered_for_saving(_app):
    """.npz was retired as a save format; reading it must keep working.

    It is a NumPy zip of the same flat columns `.npy` already carries, so it
    added a fourth thing to keep in sync for no MINFLUX-specific gain. Files
    already written this way must still open, so the loader stays.
    """
    from minflux_viewer.core.save import DATA_FORMATS
    from minflux_viewer.ui.main_window import _FMT_LOADERS, _SUPPORTED_EXTS

    assert "npz" not in DATA_FORMATS
    assert ".npz" in _SUPPORTED_EXTS
    assert _FMT_LOADERS["npz"] == "_load_npz"
    dlg = MsrReaderDialog(state=None)
    try:
        assert not hasattr(dlg, "fmt_npz")
    finally:
        dlg.close()


def test_batch_options_are_disabled_for_a_single_file(_app):
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.mode_folder.setChecked(True)
        assert dlg.recursive_check.isEnabled()
        assert dlg.reproduce_tree_check.isEnabled()

        dlg.mode_file.setChecked(True)
        assert not dlg.recursive_check.isEnabled()
        assert not dlg.reproduce_tree_check.isEnabled()
    finally:
        dlg.close()


def test_both_options_default_to_off(_app, tmp_path, monkeypatch):
    """Read the defaults from a settings path that has no file yet — otherwise
    an earlier dialog's saved state answers instead."""
    fresh = tmp_path / "unwritten.json"
    monkeypatch.setattr(MsrReaderDialog, "_settings_path", lambda self: fresh)
    dlg = MsrReaderDialog(state=None)
    try:
        assert not dlg.recursive_check.isChecked()
        assert not dlg.reproduce_tree_check.isChecked()
        defaults = dlg._load_settings("")
        assert defaults["recursive"] is False
        assert defaults["reproduce_tree"] is False
    finally:
        dlg.close()


def test_both_options_persist(_app, tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(MsrReaderDialog, "_settings_path", lambda self: settings)
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.recursive_check.setChecked(True)
        dlg.reproduce_tree_check.setChecked(True)
        dlg._save_settings()

        reopened = MsrReaderDialog(state=None)
        try:
            assert reopened.recursive_check.isChecked()
            assert reopened.reproduce_tree_check.isChecked()
        finally:
            reopened.close()
    finally:
        dlg.close()


# --- drag and drop --------------------------------------------------------

def _drop(widget, *paths):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(1.0, 1.0), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    widget.dropEvent(event)
    return event


def test_input_field_accepts_a_dropped_msr_file_or_folder(_app, tmp_path):
    msr = tmp_path / "x.msr"
    msr.write_bytes(b"")
    field = PathDropLineEdit(accept="msr_or_dir")

    # QUrl.toLocalFile() yields '/' separators on Windows, exactly as
    # QFileDialog already does for these fields — compare as paths.
    _drop(field, msr)
    assert Path(field.text()) == msr

    _drop(field, tmp_path)
    assert Path(field.text()) == tmp_path


def test_input_field_ignores_an_unrelated_file(_app, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("no")
    field = PathDropLineEdit(accept="msr_or_dir")
    field.setText("keep me")

    _drop(field, other)

    assert field.text() == "keep me"


def test_input_field_takes_the_first_acceptable_of_several(_app, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("no")
    msr = tmp_path / "y.msr"
    msr.write_bytes(b"")
    field = PathDropLineEdit(accept="msr_or_dir")

    _drop(field, other, msr)

    assert Path(field.text()) == msr


def test_output_field_accepts_only_a_folder(_app, tmp_path):
    msr = tmp_path / "z.msr"
    msr.write_bytes(b"")
    field = PathDropLineEdit(accept="dir")
    field.setText("keep me")

    _drop(field, msr)
    assert field.text() == "keep me"

    _drop(field, tmp_path)
    assert Path(field.text()) == tmp_path


def test_export_output_preflight_requires_absolute_writable_folder(tmp_path):
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import (
        prepare_export_output_dir,
    )

    output = prepare_export_output_dir(tmp_path / "exports")
    assert output == (tmp_path / "exports").resolve()
    assert output.is_dir()

    with pytest.raises(ValueError, match="absolute"):
        prepare_export_output_dir("=====")

    existing_file = tmp_path / "not-a-folder"
    existing_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a folder"):
        prepare_export_output_dir(existing_file)


def test_invalid_output_stops_before_export(_app, monkeypatch):
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod

    dlg = mod.MsrReaderDialog(state=None)
    messages = []
    called = []
    saved = []
    try:
        dlg.out_dir_edit.setText("=====")
        dlg._save_settings = lambda: saved.append(True)
        dlg.log = messages.append
        monkeypatch.setattr(mod.QMessageBox, "critical", lambda *args: None)
        monkeypatch.setattr(
            dlg, "_export_current_parsed",
            lambda *args, **kwargs: called.append("file"),
        )
        monkeypatch.setattr(
            dlg, "_run_batch_export",
            lambda *args, **kwargs: called.append("batch"),
        )

        dlg.on_ok()

        assert called == []
        assert saved == []
        assert any("export preflight failed" in message for message in messages)
    finally:
        dlg.close()


def test_dropping_a_folder_switches_the_dialog_to_batch_mode(_app, tmp_path):
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.mode_file.setChecked(True)
        dlg._on_input_path_dropped(str(tmp_path))
        assert dlg.mode_folder.isChecked()
        assert dlg.recursive_check.isEnabled()
    finally:
        dlg.close()


def test_the_dialog_fields_are_drop_targets(_app):
    dlg = MsrReaderDialog(state=None)
    try:
        assert isinstance(dlg.input_path_edit, PathDropLineEdit)
        assert isinstance(dlg.out_dir_edit, PathDropLineEdit)
        assert dlg.input_path_edit.acceptDrops()
        assert dlg.out_dir_edit.acceptDrops()
    finally:
        dlg.close()


def test_same_named_files_in_different_folders_would_collide_without_mirroring(
        _app, tree):
    """Recursive + no mirroring can map two inputs onto one folder — the run
    must refuse rather than let the second overwrite the first."""
    root, _made = tree
    (root / "20260624" / "dup.msr").write_bytes(b"")
    (root / "20260625" / "dup.msr").write_bytes(b"")
    dlg = MsrReaderDialog(state=None)
    try:
        files = dlg._find_msr_files(str(root), recursive=True)
        flat = [dlg._batch_output_dir(Path("out"), f, root, False) for f in files]
        assert len(set(flat)) < len(flat)               # collides

        mirrored = [dlg._batch_output_dir(Path("out"), f, root, True) for f in files]
        assert len(set(mirrored)) == len(mirrored)      # mirroring separates them
    finally:
        dlg.close()


# --- image series export --------------------------------------------------

def test_image_series_selection_is_carried_by_name_not_index(_app, monkeypatch):
    """Raw stack indices address one file's stack table; batch export has to
    match series across files, so it carries names."""
    dlg = MsrReaderDialog(state=None)
    try:
        series = [
            {"raw_index": 0, "name": "Ch1 {1}", "shape_str": "", "dtype": "int16"},
            {"raw_index": 4, "name": "Ch2 {59}", "shape_str": "", "dtype": "int16"},
            {"raw_index": 9, "name": "Series 10", "shape_str": "", "dtype": "uint8"},
        ]
        monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: series)

        dlg.image_field_selection = set()
        assert dlg._selected_image_names() is None
        assert dlg._image_series_to_export(None) == []

        dlg.image_field_selection = {0, 9}
        assert dlg._selected_image_names() == {"Ch1 {1}", "Series 10"}

        # In another file the same names sit at different raw indices.
        other = [
            {"raw_index": 3, "name": "Ch1 {1}", "shape_str": "", "dtype": "int16"},
            {"raw_index": 7, "name": "Series 10", "shape_str": "", "dtype": "uint8"},
            {"raw_index": 8, "name": "Ch9 {2}", "shape_str": "", "dtype": "int16"},
        ]
        monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: other)
        picked = dlg._image_series_to_export({"Ch1 {1}", "Series 10"})
        assert [s["raw_index"] for s in picked] == [3, 7]
    finally:
        dlg.close()


def test_include_all_images_covers_files_the_selection_never_saw(_app, monkeypatch):
    """"Include all images" cannot be carried as indices or names — a batch hits
    files whose stack tables and channel labels differ from the previewed one —
    so it is its own instruction (``image_field_selection is None``)."""
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import select_image_series

    dlg = MsrReaderDialog(state=None)
    try:
        other_file = [
            {"raw_index": 2, "name": "Ch7 {4}", "shape_str": "", "dtype": "int16"},
            {"raw_index": 5, "name": "MF(run)/density/loc", "shape_str": "", "dtype": "uint16"},
        ]
        monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: other_file)

        dlg.image_field_selection = None                     # all
        assert dlg._selected_image_names() is None           # no name filter
        assert dlg._image_series_to_export(None) == other_file
        assert select_image_series(other_file, indices=None) == other_file

        dlg.image_field_selection = set()                    # none
        assert dlg._image_series_to_export(None) == []
    finally:
        dlg.close()


def test_all_images_survives_the_settings_round_trip(_app, tmp_path, monkeypatch):
    """Persisting ``None`` as an empty list would silently turn "all images"
    into "no images" on the next launch."""
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(MsrReaderDialog, "_settings_path", lambda self: settings)
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.image_field_selection = None
        dlg._save_settings()
        reopened = MsrReaderDialog(state=None)
        try:
            assert reopened.image_field_selection is None
        finally:
            reopened.close()

        dlg.image_field_selection = {0, 3}
        dlg._save_settings()
        reopened = MsrReaderDialog(state=None)
        try:
            assert reopened.image_field_selection == {0, 3}
        finally:
            reopened.close()
    finally:
        dlg.close()


def test_field_dialog_has_an_include_all_images_master(_app):
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import FieldDialog

    series = [
        {"raw_index": 0, "name": "Ch1 {1}", "shape_str": "375 x 375", "dtype": "int16"},
        {"raw_index": 4, "name": "Ch2 {59}", "shape_str": "400 x 404", "dtype": "int16"},
    ]
    datasets = [{"key": "a", "name": "run", "mfx_fields": ["tid"], "mbm_fields": []}]

    # Default: nothing ticked.
    dlg = FieldDialog(datasets, None, None, image_series=series)
    assert dlg.image_payload() == set()
    # The master ticks every individual box and reports "all", not the indices.
    dlg._image_all.setChecked(True)
    assert all(cb.isChecked() for cb in dlg._image_checks.values())
    assert dlg.image_payload() is None
    dlg.close()

    # Individual picks still work for a single-file export.
    dlg = FieldDialog(datasets, None, None, image_series=series,
                      prechecked_images={4})
    assert dlg._image_all.isChecked() is False
    assert dlg.image_payload() == {4}
    dlg.close()

    # Reopening on an "all" selection shows the master ticked.
    dlg = FieldDialog(datasets, None, None, image_series=series,
                      prechecked_images=None)
    assert dlg._image_all.isChecked() is True
    assert dlg.image_payload() is None
    dlg.close()


def test_exported_image_keeps_the_name_the_msr_uses(_app):
    """Imspector names its stacks ``Ch1 {1}``; ``Ch1__1_img.tif`` was neither
    findable nor matchable back to the source."""
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import image_export_stem

    assert image_export_stem("Ch1 {1}") == "Ch1 {1}"
    assert image_export_stem("Ch4 {59}") == "Ch4 {59}"
    # Only genuinely illegal characters go; a run of them collapses to one "_".
    assert (image_export_stem("MF(260624-150437_minflux)/density/lnc")
            == "MF(260624-150437_minflux)_density_lnc")
    assert image_export_stem('a<b>c:d"e|f?g*h') == "a_b_c_d_e_f_g_h"
    # Windows strips these silently, which would break an exact-name lookup.
    assert image_export_stem("trailing dot. ") == "trailing dot"
    assert image_export_stem("CON") == "CON_"
    assert image_export_stem("") == "image"


def test_unselected_image_series_are_not_exported(_app, monkeypatch):
    dlg = MsrReaderDialog(state=None)
    try:
        series = [{"raw_index": 0, "name": "Ch1", "shape_str": "", "dtype": "int16"}]
        monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: series)
        dlg.image_field_selection = set()
        assert dlg._image_series_to_export(None) == []
        assert dlg._image_series_to_export(set()) == []
    finally:
        dlg.close()


def test_selected_obf_series_reuse_one_viewer_and_show_first(monkeypatch, _app, tmp_path):
    """Several selected stacks share one viewer; its dropdown starts at the first."""
    from types import SimpleNamespace

    from minflux_viewer.core import obf_image_source

    calls = []

    class _Source:
        def __init__(self, path, *, raw_stack_index):
            self.path = Path(path)
            self.metadata = SimpleNamespace(series_index=4, image_name="first")
            self.closed = False

        def close(self):
            self.closed = True

    class _Owner:
        def _open_image_viewer(self, source, key, *, initial_series_index=None):
            calls.append((source, key, initial_series_index))

    monkeypatch.setattr(obf_image_source, "ObfImageSource", _Source)
    dlg = MsrReaderDialog(state=None)
    try:
        dlg.parsed = {"msr": str(tmp_path / "sample.msr")}
        dlg._owner = _Owner()
        dlg.log = lambda *_args, **_kwargs: None

        assert dlg._open_obf_series_group([17, 23, 17]) is True
        assert len(calls) == 1
        source, key, initial = calls[0]
        assert source.path.name == "sample.msr"
        assert key.endswith("#obf-images")
        assert initial == 4
        assert source.closed is False
    finally:
        dlg.close()


def test_image_viewer_registry_switches_existing_file_window(_app):
    """A second selected series changes the existing window, not its count."""
    from minflux_viewer.ui.main_window import MainWindow

    class _Existing:
        _source = object()

        def __init__(self):
            self.selected = None
            self.shown = 0

        def set_series_index(self, index):
            self.selected = index

        def show(self):
            self.shown += 1

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    class _Replacement:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    key = "sample.msr#obf-images"
    existing = _Existing()
    replacement = _Replacement()
    owner = MainWindow.__new__(MainWindow)
    owner._tiff_windows = {key: existing}

    owner._open_image_viewer(replacement, key, initial_series_index=3)

    assert owner._tiff_windows[key] is existing
    assert existing.selected == 3
    assert existing.shown == 1
    assert replacement.closed is True


def test_image_export_uses_the_shared_ome_tiff_writer(tmp_path):
    """The image export must go through core/tiff_export, so its output carries
    the same OME calibration the render export writes and the viewer reads."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    from minflux_viewer.core.tiff_export import export_image_series_to_tiff
    from minflux_viewer.core.tiff_source import TiffImageSource

    class _Meta:
        axes = "ZYX"
        shape = (3, 5, 7)
        channel_names = ()

        class _PS:
            def __init__(self, nm):
                self.nm = nm
        pixel_size_x = _PS(50.0)
        pixel_size_y = _PS(50.0)
        pixel_size_z = _PS(40.0)

        def axis_size(self, axis):
            return dict(zip(self.axes, self.shape)).get(axis, 1)

    volume = np.arange(3 * 5 * 7, dtype=np.int16).reshape(3, 5, 7)

    class _Source:
        metadata = _Meta()

        def read_plane(self, *, t=0, c=0, z=0):
            return volume[z]

    target = tmp_path / "series.tif"
    result = export_image_series_to_tiff(_Source(), target)

    assert result.axes == "ZYX"
    assert result.shape == (3, 5, 7)
    assert result.n_slices == 3

    back = TiffImageSource(target)
    try:
        assert back.metadata.shape == (3, 5, 7)
        assert back.metadata.pixel_size_x.nm == pytest.approx(50.0)
        assert back.metadata.pixel_size_z.nm == pytest.approx(40.0)
        for z in range(3):
            assert np.array_equal(back.read_plane(z=z), volume[z])
    finally:
        back.close()


def test_single_plane_image_series_round_trips(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    from minflux_viewer.core.tiff_export import export_image_series_to_tiff
    from minflux_viewer.core.tiff_source import TiffImageSource

    plane = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)

    class _Meta:
        axes = "YX"
        shape = (6, 4)
        channel_names = ()

        class _PS:
            def __init__(self, nm):
                self.nm = nm
        pixel_size_x = _PS(200.0)
        pixel_size_y = _PS(200.0)
        pixel_size_z = _PS(None)

        def axis_size(self, axis):
            return dict(zip(self.axes, self.shape)).get(axis, 1)

    class _Source:
        metadata = _Meta()

        def read_plane(self, *, t=0, c=0, z=0):
            return plane

    target = tmp_path / "plane.tif"
    export_image_series_to_tiff(_Source(), target)

    back = TiffImageSource(target)
    try:
        assert np.array_equal(back.read_plane(), plane)
        assert back.metadata.pixel_size_x.nm == pytest.approx(200.0)
    finally:
        back.close()


# --- image series in the parsed-contents tree -----------------------------

def _image_nodes(dlg, item):
    """(label, expanded, [child labels]) for every image-series group below *item*."""
    found = []
    if (dlg.nodeinfo.get(dlg._item_id(item)) or {}).get("type") == "image_series_group":
        found.append((item.text(0), item.isExpanded(),
                      [item.child(i).text(0) for i in range(item.childCount())]))
    for i in range(item.childCount()):
        found.extend(_image_nodes(dlg, item.child(i)))
    return found


def test_mfxdta_companion_export_is_retired():
    """Zarr is now one enriched dataset, not mfx + mbm + mfxdta companions."""
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod

    assert not hasattr(mod, "export_source_store")


def test_image_series_are_shown_in_the_tree_beside_and_under_datasets(_app, monkeypatch):
    """Unlinked images (confocal channels) form one group parallel to the
    datasets; a density render names the dataset it came from, so it nests
    under that dataset.  Both start collapsed."""
    dlg = MsrReaderDialog(state=None)
    try:
        parsed = {
            "mode": "modern", "msr": "sample.msr",
            "datasets": [
                {"display_name": "run a", "did": "did-a", "zroot": {}, "fields": []},
                {"display_name": "run b", "did": "did-b", "zroot": {}, "fields": []},
            ],
        }
        monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: [
            {"raw_index": 0, "name": "Ch1 {1}", "shape_str": "375 x 375",
             "dtype": "int16", "source_did": ""},
            {"raw_index": 1, "name": "Ch2 {1}", "shape_str": "375 x 375",
             "dtype": "int16", "source_did": ""},
            {"raw_index": 5, "name": "MF(b)/density/loc", "shape_str": "1 x 2 x 3",
             "dtype": "uint16", "source_did": "did-b"},
        ])
        dlg._build_tree_from_result(parsed)

        root = dlg.tree.topLevelItem(0)
        groups = _image_nodes(dlg, root)
        assert len(groups) == 2
        assert all(expanded is False for _label, expanded, _kids in groups)

        # The density render sits under its own dataset, not in the shared group.
        run_b = next(root.child(i) for i in range(root.childCount())
                     if root.child(i).text(0) == "run b")
        assert _image_nodes(dlg, run_b) == [
            ("Image series (1)", False, ["MF(b)/density/loc"])]

        # The unlinked ones form the last child of the file node.
        last = root.child(root.childCount() - 1)
        assert (last.text(0), [last.child(i).text(0) for i in range(last.childCount())]) \
            == ("Image series (2)", ["Ch1 {1}", "Ch2 {1}"])

        # Right-click "Open as image" reaches them via their raw stack index.
        leaf = last.child(0)
        dlg.tree.setCurrentItem(leaf)
        leaf.setSelected(True)
        assert dlg._selected_image_series_indices() == [0]
    finally:
        dlg.close()


# --- duplicate dataset labels --------------------------------------------

def test_unique_export_stem_disambiguates_instead_of_aborting(_app):
    dlg = MsrReaderDialog(state=None)
    try:
        used: set[str] = set()
        assert dlg._unique_export_stem("run_a", used) == "run_a"
        assert dlg._unique_export_stem("run_a", used) == "run_a_2"
        assert dlg._unique_export_stem("run_a", used) == "run_a_3"
        assert dlg._unique_export_stem("run_b", used) == "run_b"
        # Labels that only collide after sanitization are handled the same way.
        assert dlg._unique_export_stem("run/a", used) == "run_a_4"
    finally:
        dlg.close()


def test_datasets_sharing_a_label_both_export_with_their_own_data(_app, tmp_path,
                                                                 monkeypatch):
    """A real .msr lists one label twice with different arrays. Keying by name
    alone dropped one of them and aborted the rest of the file."""
    np = pytest.importorskip("numpy")

    big = np.zeros(500, dtype=[("tid", "u4")])
    small = np.zeros(3, dtype=[("tid", "u4")])
    third = np.zeros(7, dtype=[("tid", "u4")])

    dlg = MsrReaderDialog(state=None)
    try:
        dlg.parsed = {
            "mode": "modern",
            "datasets": [
                {"display_name": "run", "did": "d1", "_mfx": big, "_mbm": None},
                {"display_name": "run", "did": "d1", "_mfx": small, "_mbm": None},
                {"display_name": "later", "did": "d2", "_mfx": third, "_mbm": None},
            ],
        }
        # The name-keyed map can only hold one "run" — the per-entry arrays win.
        dlg._mfx_map = {"run": small, "later": third}
        dlg._mbm_map = {}
        dlg._mbm_meta_map = {}
        dlg.field_selection = {}
        monkeypatch.setattr(dlg, "_image_series_to_export", lambda _n: [])

        written = []

        def _fake_export(out_dir, stem, formats, mfx, mbm, log=None,
                         mbm_meta=None, overwrite=False, **kwargs):
            written.append((stem, None if mfx is None else int(mfx.size)))

        import minflux_viewer.msr.export as export_mod
        monkeypatch.setattr(export_mod, "export_arrays", _fake_export)

        dlg._export_current_parsed(str(tmp_path), ["mat"])

        assert written == [("run", 500), ("run_2", 3), ("later", 7)]
    finally:
        dlg.close()


# --- progress + background execution -------------------------------------

def test_export_reports_one_progress_unit_per_dataset_and_image(tmp_path,
                                                                monkeypatch):
    np = pytest.importorskip("numpy")
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod

    arr = np.zeros(4, dtype=[("tid", "u4")])
    parsed = {
        "mode": "modern", "msr": "x.msr",
        "datasets": [
            {"display_name": "a", "_mfx": arr, "_mbm": None},
            {"display_name": "b", "_mfx": arr, "_mbm": None},
        ],
    }
    monkeypatch.setattr(mod, "image_series_of", lambda *a, **k: [
        {"raw_index": 0, "name": "img"}])
    monkeypatch.setattr(mod, "export_image_series", lambda *a, **k: None)
    import minflux_viewer.msr.export as export_mod
    def _fake_export(*_a, on_prepared=None, on_format=None, **_k):
        if on_prepared is not None:
            on_prepared()
        if on_format is not None:
            on_format("mat")

    monkeypatch.setattr(export_mod, "export_arrays", _fake_export)

    seen = []
    mod.export_parsed_result(
        parsed, mod.state_namespace_for(parsed), str(tmp_path), ["mat"],
        image_indices={0},
        progress=lambda d, t, label="": seen.append((d, t, label)))

    # Progress is reported per written file, not per dataset: each dataset
    # reports its canonical preparation and then its format write, and the image
    # series reports itself. The last call must land exactly on the total.
    assert [label for _d, _t, label in seen] == [
        "a", "a · mat", "b", "b · mat", "img"]
    totals = {t for _d, t, _l in seen}
    assert len(totals) == 1
    assert seen[-1][0] == pytest.approx(seen[-1][1])
    # Monotonically non-decreasing, never past the total.
    assert all(a[0] <= b[0] for a, b in zip(seen, seen[1:]))


def test_progress_weights_track_the_measured_cost_of_each_format():
    """A CSV/JSON write dominates a run; the budget must say so.

    Counting one unit per dataset is what made the bar sit at a single value:
    on a one-dataset file exported to every format, the whole slow part was a
    single tick.
    """
    from minflux_viewer.msr.export import (
        dataset_prepare_weight, dataset_work_weight, format_work_weight,
    )

    assert format_work_weight("json") > format_work_weight("csv")
    assert format_work_weight("csv") > format_work_weight("mat")
    assert format_work_weight("mat") > format_work_weight("zarr")
    assert format_work_weight("zarr") > format_work_weight("npy")
    assert format_work_weight(".JSON") == format_work_weight("json")

    # Writing the same dataset as JSON is most of a mixed-format run.
    rows = 1_000_000
    every = ["npy", "mat", "csv", "json"]
    total = dataset_work_weight(rows, every)
    json_share = format_work_weight("json") * rows / 1_000_000
    assert json_share / total > 0.5

    # Twice the rows is (essentially) twice the work.
    assert (dataset_work_weight(2 * rows, every)
            > 1.9 * dataset_work_weight(rows, every))

    # Building the canonical dataset is charged separately and is part of the
    # same budget, so the ticks can never overshoot the total.
    assert dataset_prepare_weight(rows) < dataset_work_weight(rows, every)
    assert (dataset_prepare_weight(rows)
            + sum(format_work_weight(f) for f in every) * rows / 1_000_000
            == pytest.approx(dataset_work_weight(rows, every)))
    # A dataset with no rows still costs something, so it moves the bar.
    assert dataset_prepare_weight(0) > 0.0


def test_batch_task_runs_files_and_reports_progress(tmp_path, monkeypatch):
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod
    import minflux_viewer.msr as msr_mod

    files = [tmp_path / f"{i}.msr" for i in range(3)]
    for f in files:
        f.write_bytes(b"")
    targets = {f: tmp_path / "out" / f.stem for f in files}

    monkeypatch.setattr(msr_mod, "parse_msr_general",
                        lambda *a, **k: {"mode": "modern", "datasets": []})
    monkeypatch.setattr(mod, "export_parsed_result", lambda *a, **k: None)

    task = mod._BatchExportTask(files, targets, ["mat"], str(tmp_path),
                                mfx_sel=None, mbm_sel=None, image_names=None,
                                field_selection={})
    fractions, results = [], []
    task.signals.progress.connect(lambda f, n: fractions.append(f))
    task.signals.finished.connect(
        lambda e, fails, c: results.append((e, list(fails), c)))
    task.run()                                   # synchronous for the test

    assert results == [(3, [], False)]
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[-1] == pytest.approx(1.0)
    assert fractions == sorted(fractions)        # monotonic


def test_batch_task_cancel_stops_at_the_next_file(tmp_path, monkeypatch):
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod
    import minflux_viewer.msr as msr_mod

    files = [tmp_path / f"{i}.msr" for i in range(4)]
    for f in files:
        f.write_bytes(b"")
    targets = {f: tmp_path / "out" / f.stem for f in files}

    monkeypatch.setattr(msr_mod, "parse_msr_general",
                        lambda *a, **k: {"mode": "modern", "datasets": []})

    task = mod._BatchExportTask(files, targets, ["mat"], str(tmp_path),
                                mfx_sel=None, mbm_sel=None, image_names=None,
                                field_selection={})
    # Cancel while the first file is "exporting".
    monkeypatch.setattr(mod, "export_parsed_result",
                        lambda *a, **k: task.cancel())
    results = []
    task.signals.finished.connect(
        lambda e, fails, c: results.append((e, list(fails), c)))
    task.run()

    exported, failures, cancelled = results[0]
    assert cancelled is True
    assert exported == 1                          # the in-flight file finished
    assert failures == []


def test_batch_task_records_a_failure_without_stopping(tmp_path, monkeypatch):
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod
    import minflux_viewer.msr as msr_mod

    files = [tmp_path / f"{i}.msr" for i in range(3)]
    for f in files:
        f.write_bytes(b"")
    targets = {f: tmp_path / "out" / f.stem for f in files}
    monkeypatch.setattr(msr_mod, "parse_msr_general",
                        lambda *a, **k: {"mode": "modern", "datasets": []})

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bad file")

    monkeypatch.setattr(mod, "export_parsed_result", _boom)
    results = []
    task = mod._BatchExportTask(files, targets, ["mat"], str(tmp_path),
                                mfx_sel=None, mbm_sel=None, image_names=None,
                                field_selection={})
    task.signals.finished.connect(
        lambda e, fails, c: results.append((e, list(fails), c)))
    task.run()

    exported, failures, cancelled = results[0]
    assert (exported, cancelled) == (2, False)
    assert len(failures) == 1 and "bad file" in failures[0][1]


def test_dialog_shows_percentage_in_the_title_and_restores_it(_app, monkeypatch):
    # The completion path opens a modal QMessageBox, which would block a
    # headless run forever.
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    dlg = MsrReaderDialog(state=None)
    try:
        base = dlg.windowTitle()
        assert base == "MINFLUX .msr Reader & Converter"

        dlg._batch_context = {"out_dir": "out", "total": 11}
        dlg._set_batch_running(True, 11)
        # isVisibleTo(): the dialog itself is never shown in the test, and
        # isVisible() is False for any child of a hidden parent.
        assert dlg._progress_row.isVisibleTo(dlg) is True
        assert not dlg._export_button.isEnabled()

        dlg._on_batch_progress(0.115, "a.msr")
        assert dlg.windowTitle() == f"{base} (batch export 11.5%…)"
        assert dlg._progress_bar.value() == 115
        assert "a.msr" in dlg._progress_label.text()

        dlg._on_batch_finished(11, [], False)
        assert dlg.windowTitle() == base
        assert dlg._progress_row.isVisibleTo(dlg) is False
        assert dlg._export_button.isEnabled()
    finally:
        dlg.close()


def test_closing_the_dialog_cancels_a_running_batch(_app, tmp_path):
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mod

    dlg = MsrReaderDialog(state=None)
    task = mod._BatchExportTask([], {}, ["mat"], str(tmp_path), mfx_sel=None,
                                mbm_sel=None, image_names=None,
                                field_selection={})
    dlg._batch_task = task
    dlg.close()

    assert task._cancelled is True
    assert dlg._batch_task is None


def test_export_conflict_modes_error_skip_and_overwrite(tmp_path, monkeypatch):
    """Re-exporting into a used folder must be a choice, not a dead end.

    Every writer refuses to clobber, so before this the only route was to read
    a FileExistsError out of the log and delete the files by hand.
    """
    import numpy as np

    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import (
        existing_export_targets,
        export_parsed_result,
    )

    dtype = np.dtype([
        ("vld", np.bool_), ("tid", np.int64), ("tim", np.float64),
        ("itr", np.int32), ("loc", np.float64, (3,)),
    ])
    mfx = np.zeros(6, dtype=dtype)
    mfx["vld"] = True
    mfx["tid"] = np.repeat(np.arange(3), 2)
    mfx["itr"] = np.tile([0, 1], 3)
    mfx["loc"][:, 0] = np.linspace(0, 3e-7, 6)
    parsed = {
        "mode": "modern",
        "msr": str(tmp_path / "acq.msr"),
        "datasets": [{"display_name": "run", "did": "d1", "_mfx": mfx, "_mbm": None}],
    }

    def run(mode=None):
        kwargs = {} if mode is None else {"on_conflict": mode}
        export_parsed_result(parsed, None, str(tmp_path), ["mat"],
                             image_indices=set(), log=lambda _m: None, **kwargs)

    run()
    produced = sorted(p.name for p in tmp_path.glob("*.mat"))
    assert produced, "first export must write something"
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.mat")}

    # Default refuses, and changes nothing.
    with pytest.raises(FileExistsError):
        run()
    assert {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.mat")} == stamps

    # The preflight sees exactly what would be clobbered.
    existing = existing_export_targets(parsed, None, str(tmp_path), ["mat"])
    assert sorted(p.name for p in existing) == produced

    # Skip leaves every existing file untouched.
    run("skip")
    assert {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.mat")} == stamps

    # Overwrite actually rewrites them.
    run("overwrite")
    assert {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.mat")} != stamps
    assert sorted(p.name for p in tmp_path.glob("*.mat")) == produced


def test_unknown_conflict_mode_is_refused(tmp_path):
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import export_parsed_result

    with pytest.raises(ValueError, match="conflict mode"):
        export_parsed_result({"mode": "modern", "datasets": []}, None,
                             str(tmp_path), ["mat"], image_indices=set(),
                             log=lambda _m: None, on_conflict="clobber")


def test_batch_conflict_dropdown_sits_in_the_output_row_and_persists(_app, tmp_path):
    """Batch needs a standing policy, not a prompt.

    One dialog cannot sensibly answer for many files with different conflicts,
    so the choice lives beside the output folder and is remembered.
    """
    dlg = MsrReaderDialog(state=None)
    try:
        assert [dlg.conflict_combo.itemData(i)
                for i in range(dlg.conflict_combo.count())] == ["skip", "overwrite"]
        # Skip is the default: a re-run should complete an interrupted batch,
        # not rewrite hours of finished output.
        assert dlg.conflict_combo.currentData() == "skip"

        dlg.mode_folder.setChecked(True)
        dlg._on_mode_changed()
        assert dlg.conflict_combo.isEnabled()
        assert dlg._ask_export_conflict(str(tmp_path), ["mat"]) == "skip"

        dlg.conflict_combo.setCurrentIndex(1)
        assert dlg._ask_export_conflict(str(tmp_path), ["mat"]) == "overwrite"
        assert dlg._load_settings("")["batch_conflict"] == "overwrite"

        # A single file asks instead, so the fixed choice does not apply.
        dlg.mode_file.setChecked(True)
        dlg._on_mode_changed()
        assert not dlg.conflict_combo.isEnabled()
    finally:
        dlg.close()


def test_output_field_leaves_room_for_the_row_widgets(_app):
    """The field is the only stretching item, so trailing widgets shrink it."""
    dlg = MsrReaderDialog(state=None)
    try:
        assert dlg.out_dir_edit.minimumWidth() <= 120
    finally:
        dlg.close()
