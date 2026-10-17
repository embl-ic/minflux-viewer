"""Exports retain one consistent submitted version without copying coordinates."""

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttributeComponent, build_localization_dataset
from minflux_viewer.core.save import save_processed
from minflux_viewer.core.save_snapshot import save_snapshot


def _dataset():
    return build_localization_dataset(name="snapshot", x_nm=np.arange(4.),
                                      y_nm=np.arange(4.), attrs={"efo": np.arange(4.)})


def test_all_export_components_are_captured_and_shared_arrays_are_deduplicated():
    ds = _dataset()
    ds.mbm = AttributeComponent({"points": np.arange(4.)})
    ds.metadata["nested"] = {"values": ds.mbm.attrs["points"]}
    ds.state["filter_specs"] = [{"attribute": "efo", "lo": 1}]
    ds.filter_mask = [True, False, True, False]
    captured, options = save_snapshot(ds, {"related_datasets": [ds], "include": {"recipe": True}})
    assert options["related_datasets"][0] is captured
    assert captured.mfx.attrs is captured.attr
    assert captured.metadata["nested"]["values"] is captured.mbm.attrs["points"]
    assert np.shares_memory(captured.attr["loc_x"], ds.attr["loc_x"])
    assert not np.shares_memory(captured.attr["efo"], ds.attr["efo"])
    ds.attr["efo"][:] = 99
    ds.mbm.attrs["points"][:] = 88
    ds.state["filter_specs"][0]["lo"] = 3
    ds.filter_mask = np.zeros(4, dtype=bool)
    ds.set_z_scaling_factor(.5)
    ds.attr.replace_coordinates(loc_x=np.ones(4))
    np.testing.assert_array_equal(captured.attr["efo"], np.arange(4.))
    np.testing.assert_array_equal(captured.mbm.attrs["points"], np.arange(4.))
    np.testing.assert_array_equal(captured.filter_mask, [True, False, True, False])
    assert captured.state["filter_specs"][0]["lo"] == 1
    assert captured.cali.z_scaling_factor == 1


def test_readonly_view_of_writable_buffer_is_not_mistaken_for_immutable_data():
    ds = _dataset()
    source = np.arange(4.)
    view = source.view()
    view.flags.writeable = False
    ds.attr["extra"] = view
    captured, _ = save_snapshot(ds)
    source[:] = 9
    np.testing.assert_array_equal(captured.attr["extra"], np.arange(4.))


def test_captured_snapshot_can_be_written_after_source_edits(tmp_path):
    from minflux_viewer.core.loader import load_npy

    ds = _dataset()
    captured, _ = save_snapshot(ds)
    ds.filter_mask = np.zeros(4, dtype=bool)
    ds.attr.replace_coordinates(loc_x=np.ones(4))
    path = tmp_path / "snapshot.npy"
    save_processed(captured, data_path=path, fmt="npy", content="snapshot",
                   filter_mode="apply", include={"recipe": False})
    loaded = load_npy(path)
    np.testing.assert_allclose(loaded.loc_nm[:, 0], np.arange(4.))


@pytest.mark.parametrize("fmt", ["mat", "msr", "zarr"])
def test_save_submission_captures_primary_related_data_and_roi_options(fmt, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from minflux_viewer.core import save
    from minflux_viewer.ui.controllers import save_export_controller

    ds, other = _dataset(), _dataset()
    members = [ds, other]
    rois = [{"geometry": {"bounds": [1, 2, 3, 4]}}]
    submitted, written, leases = [], [], []
    stub = SimpleNamespace(
        _active_dataset_for_save=lambda: ds,
        save_roi_records=lambda dataset: rois,
        _overlay_members_for=lambda dataset: members,
        _zarr_save_context=lambda dataset: {"related_datasets": members, "roi_records": rois},
        _zarr_overwrite_mode=lambda path: "replace",
        _state=SimpleNamespace(log=lambda *args: None),
        _start_file_save=lambda work, **kwargs: (submitted.append(work), leases.append(kwargs["resource_lease"])),
    )
    monkeypatch.setattr(save, "save_processed", lambda dataset, **kwargs: written.append((dataset, kwargs)))
    save_export_controller._save_as_format(stub, fmt, "data", path=str(tmp_path / f"out.{fmt}"))
    ds.attr["efo"][:] = 12
    other.attr["efo"][:] = 14
    rois[0]["geometry"]["bounds"][0] = 99
    members.clear()  # removal/reopen cannot retarget or change captured group
    submitted[0](lambda text: None)
    leases[0].release()
    captured, options = written[0]
    np.testing.assert_array_equal(captured.attr["efo"], np.arange(4.))
    assert options["roi_records"][0]["geometry"]["bounds"][0] == 1
    if fmt in {"msr", "zarr"}:
        assert options["related_datasets"][0] is captured
        np.testing.assert_array_equal(options["related_datasets"][1].attr["efo"], np.arange(4.))
