"""Ownership, exact invalidation, byte bounds and concurrent coordinate reads."""

import copy
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from minflux_viewer.core import coordinates
from minflux_viewer.core.coordinates import (
    CoordinateService,
    apply_display_transform_nm,
    cache_budget_bytes,
    coordinate_view,
    display_coordinate_view,
)
from minflux_viewer.core.dataset import AttrStore, build_localization_dataset
from minflux_viewer.core.dataset_key import DatasetKey


def dataset(n=100):
    rng = np.random.default_rng(41)
    return build_localization_dataset(
        name="coordinates", x_nm=rng.normal(size=n), y_nm=rng.normal(size=n),
        z_nm=rng.normal(size=n), source_version="simulation")


def test_cached_views_are_shared_but_public_arrays_stay_writable():
    ds = dataset()
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    a = coordinate_view(ds)
    assert coordinate_view(ds) is a
    assert ds.coordinate_view_nm is a
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0] = 123
    public = ds.loc_nm
    np.testing.assert_array_equal(public, a)
    public[:] = 123
    assert not np.any(coordinate_view(ds) == 123)
    assert service.retained_bytes == a.nbytes
    assert display_coordinate_view(ds) is a  # no duplicate identity-transform array


def test_exact_invalidation_and_filter_independence():
    ds = dataset()
    service = CoordinateService(1_000_000)
    service.attach(ds, DatasetKey(1))
    original = coordinate_view(ds)
    ds.filter_mask = np.zeros(100, bool)
    assert coordinate_view(ds) is original
    ds.attr.replace_coordinates(loc_x=np.zeros(100))
    replaced = coordinate_view(ds)
    assert replaced is not original
    assert np.all(replaced[:, 0] == 0)
    ds.cali.z_scaling_factor = 0.67  # direct assignment is also observed
    scaled = coordinate_view(ds)
    np.testing.assert_array_equal(scaled, ds.loc_nm)
    assert scaled is not replaced
    # A new store starts at revision zero but must never collide with the old one.
    ds.attr = AttrStore({"loc_x": np.ones(100), "loc_y": np.zeros(100), "loc_z": np.zeros(100)})
    ds.attr.seal_coordinates()
    assert np.all(coordinate_view(ds)[:, 0] == 1e9)


@pytest.mark.parametrize("size", [3, 4])
def test_transform_snapshot_is_exact_and_detects_in_place_edits(size):
    ds = dataset()
    service = CoordinateService(1_000_000)
    service.attach(ds, DatasetKey(1))
    matrix = np.eye(size)
    matrix[0, -1] = 2.0
    record = {f"matrix_{size}x{size}": matrix}
    ds.state["overlay_transform"] = record
    first = display_coordinate_view(ds)
    np.testing.assert_array_equal(first, apply_display_transform_nm(ds.loc_nm, record))
    assert display_coordinate_view(ds) is first
    matrix[0, -1] += 1e-8  # below the old rounded overlay-key resolution
    second = display_coordinate_view(ds)
    assert second is not first
    np.testing.assert_array_equal(second, apply_display_transform_nm(ds.loc_nm, record))


def test_eviction_resize_oversized_and_disabled_cache():
    first, second, third = dataset(), dataset(), dataset()
    service = CoordinateService(4800)
    for number, ds in enumerate((first, second, third), 1):
        service.attach(ds, DatasetKey(number))
    a, b = coordinate_view(first), coordinate_view(second)
    assert coordinate_view(first) is a  # refresh LRU
    c = coordinate_view(third)
    assert service.retained_bytes == a.nbytes + c.nbytes
    assert coordinate_view(first) is a
    assert coordinate_view(second) is not b
    np.testing.assert_array_equal(b, second.loc_nm)  # eviction doesn't invalidate consumers
    service.set_budget_bytes(1)
    assert service.retained_bytes == 0
    assert coordinate_view(first) is not coordinate_view(first)
    service.set_budget_bytes(0)
    assert coordinate_view(first) is not coordinate_view(first)
    assert service.retained_bytes == 0


def test_close_reopen_clone_and_collection_do_not_retain_or_reuse_old_views():
    ds = dataset()
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    before = coordinate_view(ds)
    clone = copy.deepcopy(ds)
    assert coordinate_view(clone) is not before
    assert coordinate_view(clone) is not coordinate_view(clone)
    service.detach(DatasetKey(1))
    assert service.retained_bytes == 0
    coordinate_view(ds)
    assert service.retained_bytes == 0  # a closed dataset cannot refill the cache
    service.attach(ds, DatasetKey(2))
    assert coordinate_view(ds) is not before
    ref = weakref.ref(ds)
    del ds
    gc.collect()
    assert ref() is None
    assert service.retained_bytes == 0


def test_sessions_with_equal_keys_are_independent():
    first, second = dataset(), dataset()
    a, b = CoordinateService(100_000), CoordinateService(100_000)
    a.attach(first, DatasetKey(1))
    b.attach(second, DatasetKey(1))
    assert coordinate_view(first) is not coordinate_view(second)
    a.detach(DatasetKey(1))
    assert b.retained_bytes == 2400


@pytest.mark.parametrize("action", ["detach", "clear", "resize"])
def test_single_flight_and_late_completion_cannot_refill_closed_cache(monkeypatch, action):
    ds = dataset()
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    entered, release = threading.Event(), threading.Event()
    original = coordinates.calibrated_coordinates
    calls = []

    def paused(*args):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(coordinates, "calibrated_coordinates", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(coordinate_view, ds)
        assert entered.wait(5)
        # Prove the second caller has joined, without sleeps or timing guesses.
        pending = next(iter(service._pending.values()))
        joined = threading.Event()
        result = pending.result

        def waiting(*args, **kwargs):
            joined.set()
            return result(*args, **kwargs)

        monkeypatch.setattr(pending, "result", waiting)
        second = pool.submit(coordinate_view, ds)
        assert joined.wait(5)
        if action == "detach":
            service.detach(DatasetKey(1))
        elif action == "clear":
            service.clear()
        else:
            service.set_budget_bytes(0)
        release.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert len(calls) == 1
    assert service.retained_bytes == 0


def test_budget_is_physical_ram_and_rejects_invalid_values(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(coordinates.psutil, "virtual_memory", lambda: SimpleNamespace(total=8000))
    assert cache_budget_bytes() == 2000
    assert cache_budget_bytes(0) == 0
    assert cache_budget_bytes(10) == 800
    for value in (-1, 101, np.nan, np.inf):
        with pytest.raises(ValueError):
            cache_budget_bytes(value)


def test_warm_display_does_not_rebuild_evicted_calibrated_array(monkeypatch):
    ds = dataset()
    service = CoordinateService(2400)  # exactly one Nx3 view
    service.attach(ds, DatasetKey(1))
    ds.state["overlay_transform"] = {"matrix_3x3": np.eye(3)}
    view = display_coordinate_view(ds)

    def forbidden(*args):
        pytest.fail("A warm transformed view rebuilt its evicted source")

    monkeypatch.setattr(coordinates, "calibrated_coordinates", forbidden)
    assert display_coordinate_view(ds) is view
    assert service.retained_bytes == 2400


def test_failed_build_is_not_cached_and_can_be_retried(monkeypatch):
    ds = dataset()
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    original = coordinates.calibrated_coordinates

    def broken(*args):
        raise ValueError("bad input")

    monkeypatch.setattr(coordinates, "calibrated_coordinates", broken)
    with pytest.raises(ValueError, match="bad input"):
        coordinate_view(ds)
    assert not service._pending
    assert service.retained_bytes == 0
    monkeypatch.setattr(coordinates, "calibrated_coordinates", original)
    np.testing.assert_array_equal(coordinate_view(ds), ds.loc_nm)


def test_input_aliases_cannot_bypass_coordinate_revision():
    ds = dataset()
    base = np.ones((100, 3))
    mapping = {"loc_x": base[:, 0], "loc_y": base[:, 1], "loc_z": base[:, 2]}
    store = AttrStore(mapping)
    store.seal_coordinates()
    ds.attr = store
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    view = coordinate_view(ds)
    base[:] = 5
    mapping["loc_x"] = np.zeros(100)
    np.testing.assert_array_equal(ds.loc_nm, view)
    assert coordinate_view(ds) is view


@pytest.mark.parametrize("dimensions", [2, 3])
def test_chunked_transform_is_bit_identical_to_whole_array_math(dimensions):
    rng = np.random.default_rng(63)
    points = rng.normal(size=(coordinates._TRANSFORM_CHUNK_ROWS + 17, 3))
    matrix = rng.normal(size=(dimensions + 1, dimensions + 1))
    expected = points.copy()
    homogeneous = np.column_stack([points[:, :dimensions], np.ones(len(points))])
    expected[:, :dimensions] = (matrix @ homogeneous.T).T[:, :dimensions]
    result = apply_display_transform_nm(points, {f"matrix_{dimensions+1}x{dimensions+1}": matrix})
    np.testing.assert_array_equal(result, expected)


def test_dataset_like_adapter_does_not_require_weak_reference_support():
    from types import SimpleNamespace

    ds = SimpleNamespace(loc_nm=np.ones((4, 3)), state={})
    service = CoordinateService(1000)
    service.attach(ds, DatasetKey(1))
    result = coordinate_view(ds)
    assert not result.flags.writeable
    assert not np.shares_memory(result, ds.loc_nm)
    assert service.retained_bytes == 0


def test_clear_also_retires_a_display_requests_not_yet_started_dependency(monkeypatch):
    ds = dataset()
    ds.state["overlay_transform"] = {"matrix_3x3": np.eye(3)}
    service = CoordinateService(100_000)
    service.attach(ds, DatasetKey(1))
    entered, release = threading.Event(), threading.Event()
    original = service._get

    def gate(key, build, **kwargs):
        if len(key) == 3:  # pause before the nested calibrated lookup starts
            entered.set()
            assert release.wait(5)
        return original(key, build, **kwargs)

    monkeypatch.setattr(service, "_get", gate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(display_coordinate_view, ds)
        assert entered.wait(5)
        service.clear()
        release.set()
        np.testing.assert_array_equal(pending.result(timeout=5), ds.loc_nm)
    assert service.retained_bytes == 0
