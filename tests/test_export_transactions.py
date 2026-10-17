"""Failure during export must not destroy an already usable dataset/recipe."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from minflux_viewer.core import save
from minflux_viewer.core.dataset import build_localization_dataset


def _dataset():
    return build_localization_dataset(name="example", x_nm=np.arange(4.),
                                      y_nm=np.arange(4.))


def test_csv_writer_failure_preserves_destination(tmp_path, monkeypatch):
    path = tmp_path / "old.csv"
    path.write_bytes(b"previous data")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(save.np, "savetxt", fail)
    with pytest.raises(OSError, match="disk full"):
        save._write_csv(path, {"id": np.arange(2), "x": np.arange(2.)})
    assert path.read_bytes() == b"previous data"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("exists", [True, False])
def test_recipe_failure_does_not_publish_data(tmp_path, monkeypatch, exists):
    path = tmp_path / "out.npy"
    recipe = save.metadata_sidecar_path(path)
    if exists:
        path.write_bytes(b"old data")
        recipe.write_bytes(b"old recipe")
    def fail(*args, **kwargs):
        raise ValueError("bad recipe")
    monkeypatch.setattr(save, "build_metadata", fail)
    with pytest.raises(ValueError, match="bad recipe"):
        save.save_processed(_dataset(), data_path=path, fmt="npy")
    assert path.exists() == exists and recipe.exists() == exists
    if exists:
        assert path.read_bytes() == b"old data"
        assert recipe.read_bytes() == b"old recipe"


def test_transaction_rolls_back_replacement_failure(tmp_path, monkeypatch):
    from minflux_viewer.core import file_transactions as ft

    a, b = tmp_path / "a.csv", tmp_path / "b.json"
    a.write_bytes(b"old a")
    b.write_bytes(b"old b")
    real_replace = ft.os.replace
    def fail(source, target):
        if source.name == "b.json" and source.parent != tmp_path:
            raise PermissionError("cannot publish recipe")
        return real_replace(source, target)
    monkeypatch.setattr(ft.os, "replace", fail)
    with pytest.raises(PermissionError, match="cannot publish"):
        with ft.file_transaction([a, b]) as staged:
            staged[a].write_bytes(b"new a")
            staged[b].write_bytes(b"new b")
    assert a.read_bytes() == b"old a" and b.read_bytes() == b"old b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "b.json"]


def test_same_target_writers_serialize(tmp_path):
    from minflux_viewer.core.file_transactions import file_transaction

    target = tmp_path / "out.csv"
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    def first():
        with file_transaction([target]) as staged:
            entered.set()
            assert release.wait(5)
            staged[target].write_bytes(b"first")
    def other():
        assert entered.wait(5)
        with file_transaction([target.parent / "." / target.name]) as staged:
            second.set()
            staged[target].write_bytes(b"second")
    with ThreadPoolExecutor(2) as pool:
        one, two = pool.submit(first), pool.submit(other)
        try:
            assert entered.wait(5)
            assert not second.wait(.1)
        finally:
            release.set()
        one.result(timeout=5)
        two.result(timeout=5)
    assert target.read_bytes() == b"second"


def test_rollback_failure_retains_recoverable_backups(tmp_path, monkeypatch):
    from minflux_viewer.core import file_transactions as ft

    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"original a")
    b.write_bytes(b"original b")
    replace = ft.os.replace
    def fail(source, target):
        if source.name == "b" and source.parent != tmp_path:
            raise PermissionError("publish failed")
        if source.suffix == ".backup" and target == a:
            raise PermissionError("restore failed")
        replace(source, target)
    monkeypatch.setattr(ft.os, "replace", fail)
    with pytest.raises(OSError, match="recovery files retained"):
        with ft.file_transaction([a, b]) as staged:
            staged[a].write_bytes(b"new a")
            staged[b].write_bytes(b"new b")
    assert b.read_bytes() == b"original b"
    assert [p.read_bytes() for p in tmp_path.glob(".mfv-save-*/*.backup")] == [b"original a"]


def test_overlay_group_prepares_every_channel_before_publication(tmp_path, monkeypatch):
    from minflux_viewer.core.overlay_save import save_overlay_group

    datasets = [_dataset(), _dataset()]
    paths = [tmp_path / name for name in ("a.npy", "a_viewer_metadata.json", "b.npy", "b_viewer_metadata.json")]
    for path in paths:
        path.write_bytes(b"previous " + path.name.encode())
    original = save.save_processed
    def fail_second(ds, **kwargs):
        if ds is datasets[1]:
            raise OSError("second channel failed")
        return original(ds, **kwargs)
    monkeypatch.setattr(save, "save_processed", fail_second)
    with pytest.raises(OSError, match="second channel"):
        save_overlay_group(datasets, tmp_path, fmt="npy", names=["a", "b"])
    for path in paths:
        assert path.read_bytes() == b"previous " + path.name.encode()
    assert set(tmp_path.iterdir()) == set(paths)


def test_data_only_overwrite_cannot_leave_a_stale_recipe(tmp_path):
    path = tmp_path / "out.npy"
    sidecar = save.metadata_sidecar_path(path)
    path.write_bytes(b"old data")
    sidecar.write_bytes(b"old recipe")
    with pytest.raises(ValueError, match="recipe would become stale"):
        save.save_processed(_dataset(), data_path=path, fmt="npy", include={"recipe": False})
    assert path.read_bytes() == b"old data" and sidecar.read_bytes() == b"old recipe"


def test_data_only_overlay_cannot_leave_stale_recipes(tmp_path):
    from minflux_viewer.core.overlay_save import save_overlay_group

    sidecar = tmp_path / "b_viewer_metadata.json"
    sidecar.write_bytes(b"old recipe")
    with pytest.raises(ValueError, match="recipe would become stale"):
        save_overlay_group([_dataset(), _dataset()], tmp_path, fmt="npy", names=["a", "b"],
                            include={"recipe": False})
    assert list(tmp_path.iterdir()) == [sidecar]


def test_custom_csv_cannot_inherit_an_old_canonical_recipe(tmp_path):
    path = tmp_path / "out.csv"
    sidecar = save.metadata_sidecar_path(path)
    path.write_bytes(b"old csv")
    sidecar.write_bytes(b"old recipe")
    with pytest.raises(ValueError, match="recipe would become stale"):
        save.write_spreadsheet_csv(_dataset(), path, column_headers=[("xnm", "x")])
    assert path.read_bytes() == b"old csv" and sidecar.read_bytes() == b"old recipe"
