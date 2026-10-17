"""Extracted images belong to a package version, not just its pathname."""

import gc
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from minflux_viewer.core.minflux_zarr import materialize_image


def _package(path, data):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images/image.tif", data)
    return {"package_path": str(path), "path": "images/image.tif"}


def test_same_path_same_size_replacement_returns_new_pixels(tmp_path):
    path = tmp_path / "images.zip"
    record = _package(path, b"old image")
    old = materialize_image(record)
    assert old.read_bytes() == b"old image"
    replacement = tmp_path / "replacement.zip"
    _package(replacement, b"new image")
    replacement.replace(path)
    new = materialize_image(record)
    assert new.read_bytes() == b"new image"
    assert old.read_bytes() == b"old image"  # a live consumer retains its version
    assert old != new


def test_cache_streams_serializes_and_evicts_only_unleased_files(tmp_path, monkeypatch):
    from minflux_viewer.core.image_materialization import ImageMaterializationCache

    cache = ImageMaterializationCache(tmp_path / "cache", budget_bytes=8, max_member_bytes=32)
    first = _package(tmp_path / "a.zip", b"first image")
    second = _package(tmp_path / "b.zip", b"other image")
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *a, **kw: pytest.fail("must stream"))
    with ThreadPoolExecutor(3) as pool:
        paths = list(pool.map(lambda _: cache.materialize(first), range(3)))
    assert len(set(paths)) == 1
    active = cache.materialize(second)
    old_path = str(paths[0])
    assert paths[0].is_file() and active.is_file()
    paths.clear()
    gc.collect()
    from pathlib import Path
    assert not Path(old_path).exists()
    assert active.read_bytes() == b"other image"
    assert cache.retained_bytes <= 8


def test_limit_duplicates_corruption_and_source_change(tmp_path, monkeypatch):
    from minflux_viewer.core.image_materialization import ImageMaterializationCache

    cache = ImageMaterializationCache(tmp_path / "cache", budget_bytes=32, max_member_bytes=8)
    record = _package(tmp_path / "large.zip", b"too many bytes")
    with pytest.raises(ValueError, match="limit"):
        cache.materialize(record)
    record = _package(tmp_path / "dupe.zip", b"image")
    with pytest.warns(UserWarning), zipfile.ZipFile(record["package_path"], "a") as archive:
        archive.writestr(record["path"], b"other")
    with pytest.raises(ValueError, match="duplicate"):
        cache.materialize(record)
    assert materialize_image({"package_path": str(tmp_path / "none"), "path": "absent"}) is None
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    assert materialize_image({"package_path": str(broken), "path": "x"}) is None
    record = _package(tmp_path / "changed.zip", b"image")
    original = cache._copy_member
    def change(archive, info, output):
        original(archive, info, output)
        replacement = tmp_path / "new.zip"
        _package(replacement, b"other")
        replacement.replace(record["package_path"])
    monkeypatch.setattr(cache, "_copy_member", change)
    with pytest.raises(OSError, match="changed"):
        cache.materialize(record)
    assert not list((tmp_path / "cache").glob("*"))


def test_tiff_consumer_holds_and_releases_extraction_lease(tmp_path):
    import io

    import tifffile

    from minflux_viewer.core.image_materialization import ImageMaterializationCache
    from minflux_viewer.core.tiff_source import TiffImageSource

    buffer = io.BytesIO()
    import numpy as np
    tifffile.imwrite(buffer, np.arange(16, dtype=np.uint8).reshape(4, 4))
    cache = ImageMaterializationCache(tmp_path / "cache", budget_bytes=0)
    record = _package(tmp_path / "pixels.zip", buffer.getvalue())
    source = TiffImageSource(cache.materialize(record))
    path = str(source.path)
    assert source.path.exists()
    np.testing.assert_array_equal(source.read_plane(), np.arange(16).reshape(4, 4))
    source.close()
    gc.collect()
    from pathlib import Path
    assert not Path(path).exists()


def test_count_age_and_close_never_evict_a_live_lease(tmp_path, monkeypatch):
    from pathlib import Path

    from minflux_viewer.core import image_materialization

    cache = image_materialization.ImageMaterializationCache(
        tmp_path / "cache", budget_bytes=100, max_unleased_files=1, max_idle_seconds=10)
    first = cache.materialize(_package(tmp_path / "first.zip", b""))
    second = cache.materialize(_package(tmp_path / "second.zip", b""))
    old, recent = str(first), str(second)
    del first, second
    gc.collect()
    assert cache.retained_files == 1 and not Path(old).exists()
    entry = next(iter(cache._entries.values()))
    entry.last_used -= 20
    live = cache.materialize(_package(tmp_path / "live.zip", b"live"))
    assert not Path(recent).exists()
    cache.close()
    assert live.read_bytes() == b"live"
    live_path = str(live)
    del live
    gc.collect()
    assert not Path(live_path).exists()


def test_failed_eviction_retains_accounting_for_retry(tmp_path, monkeypatch):
    from pathlib import Path

    from minflux_viewer.core.image_materialization import ImageMaterializationCache

    cache = ImageMaterializationCache(tmp_path / "cache", budget_bytes=0)
    live = cache.materialize(_package(tmp_path / "locked.zip", b"locked"))
    original = Path.unlink

    def refuse(path, **kwargs):
        raise PermissionError("locked by an external reader")

    monkeypatch.setattr(Path, "unlink", refuse)
    del live
    gc.collect()
    assert cache.retained_files == 1 and cache.retained_bytes == 6
    monkeypatch.setattr(Path, "unlink", original)
    cache.close()
    assert cache.retained_files == cache.retained_bytes == 0
