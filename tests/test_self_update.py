"""One-click self-update: download + staging + helper generation.

The pure/testable pieces of the frozen-Windows updater. The real file swap +
relaunch (``launch_helper``) is stubbed here — it can't run in CI without
replacing an install and restarting.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from minflux_viewer.core import self_update as su
from minflux_viewer.core.updater import ReleaseAsset, download_asset, download_url


@pytest.fixture(autouse=True)
def _stage_inside_tmp_path(tmp_path, monkeypatch):
    """``stage_update`` stages in the system temp folder. Redirect it into
    ``tmp_path`` so the tests' staging folders do not pile up there (484 had)
    and anything a hostile archive member would write beside one stays here."""
    import tempfile

    root = tmp_path / "staging"
    root.mkdir()
    real = tempfile.mkdtemp
    monkeypatch.setattr(su.tempfile, "mkdtemp",
                        lambda *a, **k: real(*a, **{**k, "dir": str(root)}))


# --- download (via file:// so no network) -----------------------------------
def test_download_url_writes_and_reports_progress(tmp_path):
    src = tmp_path / "payload.bin"
    data = b"x" * (256 * 1024 + 7)
    src.write_bytes(data)
    dest = tmp_path / "out" / "copy.bin"

    seen = []
    out = download_url(src.as_uri(), dest, progress=lambda d, t: seen.append((d, t)),
                       expected_size=len(data))
    assert out == dest
    assert dest.read_bytes() == data
    assert seen and seen[-1][0] == len(data)          # final tick == full size


def test_download_url_rejects_truncated(tmp_path):
    src = tmp_path / "p.bin"
    src.write_bytes(b"abc")
    with pytest.raises(ValueError):
        download_url(src.as_uri(), tmp_path / "o.bin", expected_size=999)


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_download_asset_verifies_size_and_checksum(tmp_path):
    src = tmp_path / "a.zip"
    src.write_bytes(b"zipbytes-here")
    asset = ReleaseAsset(name="a.zip", url=src.as_uri(), size=src.stat().st_size,
                         digest="sha256:" + _sha256(b"zipbytes-here"))
    out = download_asset(asset, tmp_path / "got.zip")
    assert out.read_bytes() == b"zipbytes-here"


def test_a_download_whose_checksum_differs_is_refused_and_removed(tmp_path):
    """GitHub publishes a SHA-256 per release asset; a self-updater replaces
    the running executable, so bytes that do not match it are never kept."""
    src = tmp_path / "a.zip"
    src.write_bytes(b"tampered")
    asset = ReleaseAsset(name="a.zip", url=src.as_uri(), size=8,
                         digest="sha256:" + _sha256(b"original"))
    dest = tmp_path / "got.zip"
    with pytest.raises(ValueError, match="checksum"):
        download_asset(asset, dest)
    assert not dest.exists()


@pytest.mark.parametrize("digest", ["", "md5:0123", "sha256:"])
def test_an_asset_without_a_sha256_is_not_installed(tmp_path, digest):
    src = tmp_path / "a.zip"
    src.write_bytes(b"zipbytes-here")
    asset = ReleaseAsset(name="a.zip", url=src.as_uri(), size=13, digest=digest)
    with pytest.raises(ValueError, match="cannot be verified"):
        download_asset(asset, tmp_path / "got.zip")
    assert not (tmp_path / "got.zip").exists()


# --- find_app_root / stage_update -------------------------------------------
def test_find_app_root_exe_at_root(tmp_path):
    (tmp_path / "minflux_viewer.exe").write_bytes(b"")
    assert su.find_app_root(tmp_path) == tmp_path


def test_find_app_root_wrapped_folder(tmp_path):
    inner = tmp_path / "minflux_viewer"
    inner.mkdir()
    (inner / "minflux_viewer.exe").write_bytes(b"")
    (inner / "_internal").mkdir()
    assert su.find_app_root(tmp_path) == inner


def test_find_app_root_missing_raises(tmp_path):
    (tmp_path / "readme.txt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        su.find_app_root(tmp_path)


#: Safe tar extraction (the ``data`` filter) arrived in Python 3.10.12 and
#: 3.11.4. The last Windows build of 3.10 is 3.10.11, so a Windows Python 3.10
#: never has it; the app refuses .tar updates there instead, which
#: ``test_a_python_without_the_tar_filter_refuses_a_tarball`` checks.
needs_tar_filter = pytest.mark.skipif(
    not hasattr(tarfile, "data_filter"),
    reason="this Python has no tarfile data filter, so it refuses .tar updates")


def _make_release_zip(tmp_path) -> Path:
    """A zip shaped like `Compress-Archive dist\\minflux_viewer`."""
    build = tmp_path / "build" / "minflux_viewer"
    (build / "_internal").mkdir(parents=True)
    (build / "minflux_viewer.exe").write_bytes(b"NEWEXE")
    (build / "_internal" / "lib.dll").write_bytes(b"NEWDLL")
    zpath = tmp_path / "MINFLUX-Viewer-9.9.9-win.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in build.rglob("*"):
            zf.write(p, p.relative_to(build.parent))
    return zpath


def test_stage_update_extracts_and_finds_root(tmp_path):
    zpath = _make_release_zip(tmp_path)
    app_root, extract_dir = su.stage_update(zpath)
    try:
        assert (app_root / "minflux_viewer.exe").read_bytes() == b"NEWEXE"
        assert (app_root / "_internal" / "lib.dll").exists()
        assert extract_dir in app_root.parents or extract_dir == app_root.parent
    finally:
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


# --- helper .bat generation --------------------------------------------------
def test_build_helper_bat_has_wait_swap_rollback_relaunch():
    bat = su.build_helper_bat(
        pid=4321, staged_dir=r"C:\tmp\stage\minflux_viewer",
        install_dir=r"C:\Apps\minflux_viewer", exe_path=r"C:\Apps\minflux_viewer\minflux_viewer.exe",
        extract_dir=r"C:\tmp\stage")
    assert 'PID eq 4321' in bat                                   # waits for our process
    assert 'move "C:\\Apps\\minflux_viewer" "C:\\Apps\\minflux_viewer.bak"' in bat
    assert "robocopy" in bat and "/e" in bat
    assert ':rollback' in bat and 'move "C:\\Apps\\minflux_viewer.bak"' in bat
    assert 'start "" "C:\\Apps\\minflux_viewer\\minflux_viewer.exe"' in bat  # relaunch
    assert 'del "%~f0"' in bat                                    # self-delete
    assert bat.endswith("\r\n") and "\r\n" in bat                # CRLF line endings


# --- gating ------------------------------------------------------------------
def test_can_self_update_frozen_windows_or_linux(monkeypatch):
    monkeypatch.setattr(su, "is_windows", lambda: False)
    monkeypatch.setattr(su, "is_linux", lambda: False)
    monkeypatch.setattr(su, "is_frozen", lambda: False)
    assert su.can_self_update() is False              # source run
    monkeypatch.setattr(su, "is_frozen", lambda: True)
    assert su.can_self_update() is False              # frozen macOS → excluded
    monkeypatch.setattr(su, "is_windows", lambda: True)
    assert su.can_self_update() is True               # frozen Windows
    monkeypatch.setattr(su, "is_windows", lambda: False)
    monkeypatch.setattr(su, "is_linux", lambda: True)
    assert su.can_self_update() is True               # frozen Linux


def test_build_helper_sh_has_wait_swap_rollback_relaunch():
    sh = su.build_helper_sh(
        pid=4321, staged_dir="/tmp/stage/minflux_viewer",
        install_dir="/opt/minflux_viewer", exe_path="/opt/minflux_viewer/minflux_viewer",
        extract_dir="/tmp/stage")
    assert sh.startswith("#!/bin/sh")
    assert "kill -0 4321" in sh                                   # waits for our process
    assert 'mv "/opt/minflux_viewer" "/opt/minflux_viewer.bak"' in sh
    assert 'cp -a "/tmp/stage/minflux_viewer" "/opt/minflux_viewer"' in sh
    assert 'mv "/opt/minflux_viewer.bak" "/opt/minflux_viewer"' in sh  # rollback
    assert '"/opt/minflux_viewer/minflux_viewer" >/dev/null 2>&1 &' in sh  # relaunch
    assert 'rm -- "$0"' in sh                                     # self-delete


@needs_tar_filter
def test_stage_update_handles_tarball(tmp_path):
    import tarfile
    build = tmp_path / "build" / "minflux_viewer"
    (build / "_internal").mkdir(parents=True)
    (build / "minflux_viewer").write_bytes(b"NEWEXE")            # Linux exe (no .exe)
    tgz = tmp_path / "MINFLUX-Viewer-9.9-linux.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(build, arcname="minflux_viewer")
    app_root, extract_dir = su.stage_update(tgz, exe_name="minflux_viewer")
    try:
        assert (app_root / "minflux_viewer").read_bytes() == b"NEWEXE"
        assert (app_root / "_internal").exists()
    finally:
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


def _evil_tarball(tmp_path, member):
    import io
    import tarfile

    tgz = tmp_path / "MINFLUX-Viewer-9.9-linux.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        info = tarfile.TarInfo(member)
        info.size = 4
        tf.addfile(info, io.BytesIO(b"evil"))
    return tgz


@needs_tar_filter
def test_a_tarball_member_climbing_out_of_the_staging_folder_is_refused(tmp_path):
    """``tarfile.extractall`` without a filter writes wherever a member's name
    points; a release archive must only ever fill its own staging folder.
    (Staging sits in ``tmp_path``, see the autouse fixture, so a regression
    writes there rather than next to the real temp folder.)"""
    import tarfile

    tgz = _evil_tarball(tmp_path, "../../escaped.txt")
    with pytest.raises(tarfile.FilterError):
        su.stage_update(tgz, exe_name="minflux_viewer")
    assert not (tmp_path / "escaped.txt").exists()


@needs_tar_filter
def test_a_tarball_member_with_an_absolute_path_cannot_land_there(tmp_path):
    """Unfiltered, an absolute name is joined onto the drive root on Windows
    (it was: a red run of this test wrote ``C:\\abs\\escaped.txt``). The
    target here is inside ``tmp_path`` so a regression stays harmless."""
    import tarfile

    target = tmp_path / "absolute_escape.txt"
    tgz = _evil_tarball(tmp_path, target.as_posix())
    # Refused outright on Windows (a drive path); on POSIX the leading slash
    # is stripped, the member lands inside staging, and there is no exe.
    with pytest.raises((tarfile.FilterError, FileNotFoundError)):
        su.stage_update(tgz, exe_name="minflux_viewer")
    assert not target.exists()


def test_is_writable(tmp_path):
    assert su.is_writable(tmp_path) is True
    assert su.is_writable(tmp_path / "does-not-exist") is False


# --- apply_update orchestration (launch stubbed) ----------------------------
def test_apply_update_stages_and_launches(tmp_path, monkeypatch):
    zpath = _make_release_zip(tmp_path)
    install = tmp_path / "install" / "minflux_viewer"
    install.mkdir(parents=True)
    exe = install / "minflux_viewer.exe"
    exe.write_bytes(b"OLD")

    calls = {}
    monkeypatch.setattr(su, "is_windows", lambda: True)       # take the Windows branch
    monkeypatch.setattr(su, "launch_helper",
                        lambda bat, target: calls.update(bat=Path(bat), target=Path(target)))

    bat_path = su.apply_update(zpath, pid=4321, exe_path=exe, target_dir=install)
    assert bat_path.exists() and bat_path.suffix == ".bat"
    text = bat_path.read_text()
    assert str(install) in text and str(exe) in text and "PID eq 4321" in text
    assert calls["bat"] == bat_path and calls["target"] == install
    # the staged new files were extracted and referenced in the script
    assert "minflux_viewer.exe" in text
    bat_path.unlink(missing_ok=True)


@needs_tar_filter
def test_apply_update_linux_writes_sh(tmp_path, monkeypatch):
    import tarfile
    monkeypatch.setattr(su, "is_windows", lambda: False)      # take the POSIX branch
    build = tmp_path / "build" / "minflux_viewer"
    (build / "_internal").mkdir(parents=True)
    (build / "minflux_viewer").write_bytes(b"NEW")
    tgz = tmp_path / "app-linux.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(build, arcname="minflux_viewer")

    install = tmp_path / "install" / "minflux_viewer"
    install.mkdir(parents=True)
    exe = install / "minflux_viewer"
    exe.write_bytes(b"OLD")
    calls = {}
    monkeypatch.setattr(su, "launch_helper", lambda s, t: calls.update(s=Path(s), t=Path(t)))

    sh_path = su.apply_update(tgz, pid=999, exe_path=exe, target_dir=install)
    assert sh_path.suffix == ".sh" and sh_path.exists()
    text = sh_path.read_text()
    assert "kill -0 999" in text and install.as_posix() in text
    assert calls["s"] == sh_path
    sh_path.unlink(missing_ok=True)


def test_a_python_without_the_tar_filter_refuses_a_tarball(tmp_path, monkeypatch):
    """Without the ``data`` filter, extraction would trust member paths, so the
    update is refused before anything is unpacked."""
    build = tmp_path / "build" / "minflux_viewer"
    build.mkdir(parents=True)
    (build / "minflux_viewer").write_bytes(b"NEW")
    tgz = tmp_path / "app-linux.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(build, arcname="minflux_viewer")
    monkeypatch.delattr(tarfile, "data_filter", raising=False)

    with pytest.raises(RuntimeError, match="cannot extract a .tar archive safely"):
        su.stage_update(tgz, exe_name="minflux_viewer")
    assert not list((tmp_path / "staging").iterdir())    # nothing was staged


def test_failed_staging_removes_its_owned_directory(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        su.stage_update(archive)
    assert not list((tmp_path / "staging").iterdir())


def test_failed_helper_launch_keeps_install_and_cleans_staging(tmp_path, monkeypatch):
    archive = _make_release_zip(tmp_path)
    exe = tmp_path / "install" / "minflux_viewer.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"OLD")
    monkeypatch.setattr(su, "is_windows", lambda: True)

    def fail(*args):
        raise OSError("launch refused")

    monkeypatch.setattr(su, "launch_helper", fail)
    with pytest.raises(OSError, match="launch refused"):
        su.apply_update(archive, exe_path=exe)
    assert exe.read_bytes() == b"OLD"
    assert not list((tmp_path / "staging").iterdir())


def test_an_abandoned_queued_preparation_cleans_itself(tmp_path, monkeypatch):
    import gc

    monkeypatch.setattr(su, "is_windows", lambda: True)
    prepared = su.prepare_update(_make_release_zip(tmp_path), exe_path=tmp_path / "minflux_viewer.exe")
    owned = prepared.extract_dir
    assert owned.is_dir()
    del prepared
    gc.collect()
    assert not owned.exists()


def test_handoff_owns_staging_until_helper_finishes(tmp_path, monkeypatch):
    import gc

    monkeypatch.setattr(su, "is_windows", lambda: True)
    prepared = su.prepare_update(_make_release_zip(tmp_path), exe_path=tmp_path / "minflux_viewer.exe")
    owned = prepared.extract_dir
    monkeypatch.setattr(su, "launch_helper", lambda *args: None)
    su.launch_prepared_update(prepared)
    with pytest.raises(ValueError, match="already"):
        su.launch_prepared_update(prepared)
    del prepared
    gc.collect()
    assert owned.is_dir()
    # No real helper ran in this test; explicitly finish its cleanup contract.
    import shutil
    shutil.rmtree(owned)
