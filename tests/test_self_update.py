"""One-click self-update: download + staging + helper generation.

The pure/testable pieces of the frozen-Windows updater. The real file swap +
relaunch (``launch_helper``) is stubbed here — it can't run in CI without
replacing an install and restarting.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from minflux_viewer.core import self_update as su
from minflux_viewer.core.updater import ReleaseAsset, download_asset, download_url


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


def test_download_asset_verifies_size(tmp_path):
    src = tmp_path / "a.zip"
    src.write_bytes(b"zipbytes-here")
    asset = ReleaseAsset(name="a.zip", url=src.as_uri(), size=src.stat().st_size)
    out = download_asset(asset, tmp_path / "got.zip")
    assert out.read_bytes() == b"zipbytes-here"


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
