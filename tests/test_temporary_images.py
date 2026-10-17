"""Abandoned cache cleanup requires format, dead owner, age, lock and flat paths."""

import os
import stat
from types import SimpleNamespace

from minflux_viewer.core import temporary_images as temporary


def test_old_but_live_or_locked_cache_is_never_deleted(tmp_path, monkeypatch):
    owner = temporary.TemporaryImages(tmp_path)
    image = owner.path / ("a" * 24 + "-image.tif")
    image.write_bytes(b"pixels")
    try:
        assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 0
        monkeypatch.setattr(temporary, "_owner_alive", lambda *args: False)
        assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 0
        assert image.read_bytes() == b"pixels"
    finally:
        image.unlink()
        owner.close()


def test_dead_unlocked_cache_is_reclaimed_but_unknown_contents_and_legacy_are_not(tmp_path, monkeypatch):
    owner = temporary.TemporaryImages(tmp_path)
    image = owner.path / ("b" * 24 + "-image.tif")
    image.write_bytes(b"recoverable from source package")
    os.close(owner._fd)
    owner._fd = None
    monkeypatch.setattr(temporary, "_owner_alive", lambda *args: False)
    assert temporary.cleanup_abandoned_images(tmp_path) == 0  # age also matters
    unknown = owner.path / "unrelated.txt"
    unknown.write_text("leave this alone", encoding="utf-8")
    assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 0
    assert image.exists()
    unknown.unlink()
    legacy = tmp_path / "minflux-viewer-images-legacy"
    legacy.mkdir()
    assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 1
    assert not owner.path.exists() and legacy.exists()


def test_path_gate_rejects_links_and_windows_reparse_points():
    for info in (SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0),
                 SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)):
        path = SimpleNamespace(lstat=lambda: info)
        assert not temporary._plain(path)


def test_real_child_process_ownership_and_exit(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; from pathlib import Path; "
        "from minflux_viewer.core.temporary_images import TemporaryImages; "
        "owner=TemporaryImages(Path(sys.argv[1])); "
        "(owner.path / ('c'*24+'-pixels.tif')).write_bytes(b'pixels'); "
        "print(owner.path, flush=True); sys.stdin.readline(); owner.close()"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        path = Path(child.stdout.readline().strip())
        assert path.parent == tmp_path
        assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 0
    finally:
        _out, error = child.communicate("exit\n", timeout=10)
    assert child.returncode == 0, error
    assert temporary.cleanup_abandoned_images(tmp_path, minimum_age=0) == 1
    assert not path.exists()
