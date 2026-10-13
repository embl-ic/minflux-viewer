"""Opening parsed .msr datasets in the viewer.

- The field copy out of the structured ``mfx`` record buffer is done in
  cache-sized blocks and the flat (m2410) layout is normalised once, not twice;
  the result must be identical to the field-by-field copy it replaced.
- The MSR reader builds the datasets OFF the UI thread (like opening a .mat /
  .npy), reports ``Opening dataset: '<name>'…  NN%`` to the status bar and the
  Log, and only adds them to the viewer when the build delivers.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from minflux_viewer.core import loader


def _flat_mfx(n: int = 1000, n_itr: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = np.dtype([
        ("vld", "?"), ("fnl", "?"), ("tid", "<i4"), ("itr", "<i4"),
        ("tim", "<f8"), ("loc", "<f8", (3,)), ("lnc", "<f8", (3,)),
        ("efo", "<f4"), ("cfr", "<f4"), ("dcr", "<f4", (2,)), ("eco", "<i4"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = np.tile(np.arange(n_itr), n // n_itr + 1)[:n]
    mfx["tid"] = np.arange(n) // (n_itr * 5)
    mfx["vld"] = rng.random(n) > 0.2
    mfx["tim"] = np.arange(n) * 1e-3
    mfx["loc"] = rng.normal(0, 1e-7, (n, 3))
    mfx["lnc"] = mfx["loc"] + 1e-9
    mfx["loc"][~mfx["vld"]] = np.nan
    mfx["efo"] = rng.random(n) * 1e5
    mfx["cfr"] = rng.random(n)
    mfx["dcr"] = rng.random((n, 2))
    mfx["eco"] = rng.integers(0, 300, n)
    return mfx


def test_block_copy_matches_the_field_by_field_copy(monkeypatch):
    mfx = _flat_mfx(1003)                           # not a multiple of the block
    monkeypatch.setattr(loader, "_FIELD_COPY_BLOCK_ROWS", 64)
    seen: list[float] = []
    cols = loader._mfx_columns(mfx, seen.append)
    for name in mfx.dtype.names:
        expected = np.array(mfx[name])
        assert cols[name].dtype == expected.dtype
        assert cols[name].shape == expected.shape
        assert cols[name].flags.c_contiguous
        assert np.array_equal(cols[name], expected, equal_nan=expected.dtype.kind == "f")
    assert seen and seen == sorted(seen) and 0.0 <= seen[0] and seen[-1] < 1.0


def test_flat_load_reports_progress_and_keeps_raw_and_view_separate():
    mfx = _flat_mfx()
    seen: list[float] = []
    ds = loader.load_from_mfx_array(mfx, name="x", progress=seen.append)
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    # The all-iteration store and the last-valid view no longer come from two
    # separate passes; they must still be independent arrays.
    raw_tim = ds.mfx_raw["tim"]
    assert raw_tim.shape == (mfx.shape[0],)
    assert not np.shares_memory(raw_tim, ds.attr["tim"])
    before = raw_tim.copy()
    ds.attr["tim"][:] = -1.0
    assert np.array_equal(ds.mfx_raw["tim"], before)
    # The view is the last valid iteration.
    last = (mfx["itr"] == mfx["itr"][mfx["vld"]].max()) & mfx["vld"]
    assert ds.prop.num_loc == int(last.sum())


# ---------------------------------------------------------------------------
# The reader's off-thread open
# ---------------------------------------------------------------------------

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


def test_reader_opens_datasets_off_the_ui_thread_with_progress(_app, monkeypatch):
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mrd

    state = AppState()
    state.prefs.setdefault("data", {}).update(
        compute_local_density=False, compute_loc_prec=False)
    status: list[str] = []
    progress_lines: list[str] = []
    state.status_message.connect(status.append)
    state.progress_log.connect(lambda text, _final=False: progress_lines.append(text))

    dlg = mrd.MsrReaderDialog(state=state)
    mfx = {"red": _flat_mfx(seed=1), "green": _flat_mfx(seed=2)}
    dlg._store_parse_result({"mode": "modern", "mfx_map": mfx,
                             "mbm_map": {}, "mbm_meta_map": {}})
    dlg.parsed = {"msr": "", "datasets": [{"display_name": k} for k in mfx]}
    monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: [])
    monkeypatch.setattr(dlg, "_open_obf_series_group", lambda raws: False)
    monkeypatch.setattr(dlg, "_viewer_import_plan", lambda: {k: None for k in mfx})
    monkeypatch.setattr(dlg, "_ask_viewer_channel_alignment",
                        lambda plan: mrd._OPEN_INDIVIDUAL)

    dlg._on_open_in_viewer()
    # Returned at once: nothing is added until the worker delivers.
    assert dlg._viewer_open_task is not None
    assert len(state.datasets) == 0

    deadline = time.monotonic() + 30
    while len(state.datasets) < 2 and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert [d.metadata["msr_dataset_key"] for d in state.datasets] == ["red", "green"]
    assert all(d.prop.num_loc > 0 for d in state.datasets)

    assert any(line.startswith("Opening dataset: 'red'…  ") for line in progress_lines)
    assert any(line.startswith("Opening dataset: 'green'…  ") and line.endswith("%")
               for line in status)
    assert any(s.startswith("Opened 2 dataset(s)") for s in status)
    # The reader closed itself after a successful open.
    QApplication.processEvents()


def test_adding_a_dataset_does_not_stat_the_recent_files(_app, monkeypatch, tmp_path):
    """The Open Recent menu checks every remembered path for existence, which
    on an unreachable drive measured 2.2 s on the GUI thread per opened
    dataset. It is rebuilt when shown, not on every add."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    # MainWindow.close() saves prefs to the (test-isolated) QSettings, and the
    # next AppState would load these edits -- keep them out of later tests.
    monkeypatch.setattr(state, "save_prefs", lambda: None)
    state.prefs.setdefault("data", {}).update(
        show_render=False, show_data_info=False, show_dataset_manager=False,
        compute_local_density=False, compute_loc_prec=False)
    existing = tmp_path / "seen.npy"
    existing.write_bytes(b"")
    state.prefs["file"]["recent_files"] = [str(existing)]
    win = MainWindow(state)
    try:
        calls: list[int] = []
        original = win._populate_recent_menu
        monkeypatch.setattr(win, "_populate_recent_menu",
                            lambda: (calls.append(1), original())[1])
        state.add_dataset(loader.load_from_mfx_array(_flat_mfx(), name="x"))
        assert calls == []
        assert win._recent_menu_stale
        win._recent_menu.aboutToShow.emit()
        assert calls == [1] and not win._recent_menu_stale
        assert any(a.data() == str(existing) for a in win._recent_menu.actions())
        win._recent_menu.aboutToShow.emit()            # nothing changed since
        assert calls == [1]
    finally:
        win.close()


def test_closing_the_reader_mid_open_abandons_it(_app, monkeypatch):
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mrd

    state = AppState()
    dlg = mrd.MsrReaderDialog(state=state)
    mfx = {"red": _flat_mfx(seed=3)}
    dlg._store_parse_result({"mode": "modern", "mfx_map": mfx,
                             "mbm_map": {}, "mbm_meta_map": {}})
    dlg.parsed = {"msr": "", "datasets": [{"display_name": "red"}]}
    monkeypatch.setattr(dlg, "_gather_image_series_for_dialog", lambda: [])
    monkeypatch.setattr(dlg, "_open_obf_series_group", lambda raws: False)
    monkeypatch.setattr(dlg, "_viewer_import_plan", lambda: {"red": None})
    monkeypatch.setattr(dlg, "_ask_viewer_channel_alignment",
                        lambda plan: mrd._OPEN_INDIVIDUAL)

    dlg._on_open_in_viewer()
    task = dlg._viewer_open_task
    dlg.close()
    assert dlg._viewer_open_task is None and task.is_cancelled
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert len(state.datasets) == 0
