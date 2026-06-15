import numpy as np

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _dataset(file: FileInfo | None = None) -> MinfluxDataset:
    attrs = AttrStore({
        "loc_x": np.array([0.0, 1e-9, 2e-9]),
        "loc_y": np.array([0.0, 0.0, 0.0]),
        "loc_z": np.zeros(3),
        "tid": np.array([1, 1, 2]),
        "tim": np.array([0.0, 0.1, 0.2]),
        "ftr": np.ones(3, dtype=bool),
    })
    return MinfluxDataset(
        file=file or FileInfo(name="component-test", folder=""),
        prop=DataProp(num_loc=3, num_traces=2, attr_names=attrs.keys()),
        attr=attrs,
    )


def test_mfx_component_wraps_existing_attr_store():
    ds = _dataset()

    assert ds.mfx.get_attr("tid").tolist() == [1, 1, 2]
    assert ds.mfx.get_role("track_id").tolist() == [1, 1, 2]
    assert ds.mfx.get_meta("loc_x")["unit"] == "m"

    ds.mfx.set_attr("efo", np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(ds.attr["efo"], [10.0, 20.0, 30.0])


def test_filter_mask_has_dataset_state_but_preserves_attr_compatibility():
    ds = _dataset()
    mask = np.array([True, False, True])

    ds.filter_mask = mask

    np.testing.assert_array_equal(ds.state["filter_mask"], mask)
    np.testing.assert_array_equal(ds.attr["ftr"], mask)
    np.testing.assert_array_equal(ds.filter_mask, mask)


def test_scripting_facade_exposes_active_dataset():
    state = AppState()
    ds = _dataset()
    state.add_dataset(ds)

    assert state.mfv.get_active_dataset() is ds
    assert state.mfv.get_datasets() == [ds]
    assert state.mfv.get_dataset(0) is ds


def test_synthetic_dataset_is_not_recorded_as_recent():
    state = AppState()
    state.prefs["file"]["recent_files"] = []

    state.add_dataset(_dataset())

    assert state.prefs["file"]["recent_files"] == []


def test_existing_dataset_file_is_recorded_as_recent(tmp_path, monkeypatch):
    path = tmp_path / "sample.mat"
    path.write_bytes(b"")
    state = AppState()
    state.prefs["file"]["recent_files"] = []
    monkeypatch.setattr(state, "save_prefs", lambda: None)

    state.add_dataset(_dataset(FileInfo(name=path.name, folder=str(path.parent))))

    assert state.prefs["file"]["recent_files"] == [str(path)]
