"""Data files named on the command line.

``minflux-viewer <path>`` previously matched only ``.mat/.npy/.csv/.msr`` and
handed each to ``_route_file``, so every other supported format — and any
folder — was silently ignored.
"""

from __future__ import annotations

from minflux_viewer.ui.main_window import startup_paths_from_argv


def test_accepts_every_supported_extension(tmp_path):
    names = ["a.mat", "b.npy", "c.csv", "d.msr", "e.json", "f.tif", "g.tiff",
             "h.xlsx", "i.tsv", "j.roi", "k.zip", "l.txt", "m.xlsm"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    args = [str(tmp_path / name) for name in names]

    assert startup_paths_from_argv(args) == args


def test_accepts_a_folder_so_it_can_be_scanned_like_a_drop(tmp_path):
    folder = tmp_path / "acquisition"
    folder.mkdir()
    assert startup_paths_from_argv([str(folder)]) == [str(folder)]


def test_accepts_a_zarr_store_which_is_itself_a_directory(tmp_path):
    store = tmp_path / "export.zarr"
    store.mkdir()
    assert startup_paths_from_argv([str(store)]) == [str(store)]


def test_drops_option_like_arguments():
    """``-psn_0_…`` is passed to a bundled macOS .app; the rest are Qt options.

    Probing any of them as a filesystem path is wrong.
    """
    assert startup_paths_from_argv(["-psn_0_774321"]) == []
    assert startup_paths_from_argv(["-platform", "offscreen"]) == []
    assert startup_paths_from_argv(["--style", "fusion"]) == []


def test_ignores_unsupported_types(tmp_path):
    (tmp_path / "notes.docx").write_bytes(b"")
    assert startup_paths_from_argv([str(tmp_path / "notes.docx")]) == []


def test_keeps_a_missing_but_supported_path(tmp_path):
    """The loader reports a missing file properly; dropping it here would make
    a typo look like it opened successfully."""
    missing = str(tmp_path / "absent.mat")
    assert startup_paths_from_argv([missing]) == [missing]


def test_preserves_order(tmp_path):
    first, second = tmp_path / "1.mat", tmp_path / "2.msr"
    for path in (first, second):
        path.write_bytes(b"")
    assert startup_paths_from_argv([str(second), str(first)]) == [
        str(second), str(first)]


def test_empty_arguments_are_skipped():
    assert startup_paths_from_argv(["", None]) == []


def test_startup_loop_routes_through_route_path(qtbot, tmp_path):
    """A folder argument must be scanned, which _route_file cannot do."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    window = MainWindow(state)
    qtbot.addWidget(window)

    routed: list[str] = []
    window._route_path = routed.append

    folder = tmp_path / "acq"
    folder.mkdir()
    sample = tmp_path / "x.json"
    sample.write_bytes(b"")
    for path in startup_paths_from_argv([str(folder), str(sample)]):
        window._route_path(path)

    assert routed == [str(folder), str(sample)]
