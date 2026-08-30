from __future__ import annotations

import hashlib
import io
import logging
import sys
import sysconfig
import threading
import uuid
import zipfile
from pathlib import Path

import pytest

from minflux_viewer.core import user_libs


def _write_dist_info(
    root: Path,
    name: str,
    version: str,
    *,
    tag: str = "py3-none-any",
    package_file: Path | None = None,
) -> Path:
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        f"Wheel-Version: 1.0\nTag: {tag}\n",
        encoding="utf-8",
    )
    record_lines = [
        f"{dist_info.name}/METADATA,,",
        f"{dist_info.name}/WHEEL,,",
        f"{dist_info.name}/RECORD,,",
    ]
    if package_file is not None:
        record_lines.insert(0, f"{package_file.relative_to(root).as_posix()},,")
    (dist_info / "RECORD").write_text("\n".join(record_lines), encoding="utf-8")
    return dist_info


def test_install_paths_appends_without_disturbing_existing_order(tmp_path: Path) -> None:
    original = list(sys.path)
    prefix = str(tmp_path / "already-first")
    sys.path.insert(0, prefix)
    try:
        added = user_libs.install_paths(
            {"plugin": {"user_lib_enabled": True, "user_lib_paths": [str(tmp_path)]}}
        )
        resolved = str(tmp_path.resolve())
        assert added == [resolved]
        assert sys.path[0] == prefix
        assert sys.path[-1] == resolved
    finally:
        sys.path[:] = original


def test_external_site_packages_becomes_importable(tmp_path: Path) -> None:
    module_name = f"external_{uuid.uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text("ANSWER = 42\n", encoding="utf-8")
    original = list(sys.path)
    try:
        assert user_libs.install_paths(
            {"plugin": {"user_lib_enabled": True, "user_lib_paths": [str(tmp_path)]}}
        )
        module = __import__(module_name)
        assert module.ANSWER == 42
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = original


def test_install_paths_respects_disabled_preference(tmp_path: Path) -> None:
    original = list(sys.path)
    try:
        assert user_libs.install_paths(
            {"plugin": {"user_lib_enabled": False, "user_lib_paths": [str(tmp_path)]}}
        ) == []
        assert str(tmp_path.resolve()) not in sys.path
    finally:
        sys.path[:] = original


def test_abi_mismatch_is_refused_with_clear_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    wrong_minor = 10 if sys.version_info.minor != 10 else 11
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    wrong_tag = f"cp3{wrong_minor}-cp3{wrong_minor}-{platform_tag}"
    _write_dist_info(tmp_path, "compiled-demo", "1.0", tag=wrong_tag)

    inspection = user_libs.inspect_path(tmp_path)
    assert not inspection.ok
    assert "ABI mismatch" in inspection.message
    assert f"CPython {sys.version_info.major}.{sys.version_info.minor}" in inspection.message

    original = list(sys.path)
    try:
        with caplog.at_level(logging.WARNING):
            added = user_libs.install_paths(
                {"plugin": {"user_lib_enabled": True, "user_lib_paths": [str(tmp_path)]}}
            )
        assert added == []
        assert "ABI mismatch" in caplog.text
        assert str(tmp_path.resolve()) not in sys.path
    finally:
        sys.path[:] = original


def test_pure_python_wheel_metadata_is_abi_compatible(tmp_path: Path) -> None:
    _write_dist_info(tmp_path, "pure-demo", "1.0")
    inspection = user_libs.inspect_path(tmp_path)
    assert inspection.ok
    assert inspection.message.startswith("ABI OK")


def test_blocklisted_install_is_rejected_before_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(user_libs, "_run_pip", fail_if_called)
    for spec in ("numpy", "SciPy>=1.11", "PyQt6[extras]", "tifffile==2026.1"):
        with pytest.raises(user_libs.UserLibraryError, match="bundled runtime"):
            user_libs.install_package(spec)
    assert not called


def test_external_blocklisted_package_warns_but_path_is_appended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "numpy").mkdir()
    original = list(sys.path)
    try:
        with caplog.at_level(logging.WARNING):
            added = user_libs.install_paths(
                {"plugin": {"user_lib_enabled": True, "user_lib_paths": [str(tmp_path)]}}
            )
        assert added == [str(tmp_path.resolve())]
        assert "contains bundled packages (numpy)" in caplog.text
    finally:
        sys.path[:] = original


def test_managed_install_uses_wheel_only_pip_and_records_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_pip(arguments: list[str]) -> int:
        captured.extend(arguments)
        target = Path(arguments[arguments.index("--target") + 1])
        package = target / "demo_package"
        package.mkdir(parents=True)
        package_file = package / "__init__.py"
        package_file.write_text("", encoding="utf-8")
        _write_dist_info(target, "demo-package", "1.2.3", package_file=package_file)
        return 0

    monkeypatch.setattr(user_libs, "_run_pip", fake_pip)
    prefs = {"plugin": {"installed_packages": {}, "user_lib_paths": []}}
    stages: list[str] = []
    result = user_libs.install_package(
        "demo-package>=1", prefs, target=tmp_path, report=stages.append
    )

    assert result.name == "demo-package"
    assert result.version == "1.2.3"
    assert prefs["plugin"]["installed_packages"] == {"demo-package": "1.2.3"}
    assert prefs["plugin"]["user_lib_paths"] == [str(tmp_path.resolve())]
    assert captured[0:2] == ["install", "demo-package>=1"]
    for option in (
        "--target",
        "--only-binary=:all:",
        "--no-deps",
        "--no-input",
        "--disable-pip-version-check",
    ):
        assert option in captured
    assert stages[-1] == "Installed demo-package 1.2.3"


def test_remove_package_uses_record_and_updates_manifest(tmp_path: Path) -> None:
    package = tmp_path / "demo_package"
    package.mkdir()
    package_file = package / "__init__.py"
    package_file.write_text("VALUE = 1\n", encoding="utf-8")
    dist_info = _write_dist_info(
        tmp_path, "demo-package", "1.2.3", package_file=package_file
    )
    prefs = {"plugin": {"installed_packages": {"demo-package": "1.2.3"}}}

    assert user_libs.remove_package("demo-package", prefs, target=tmp_path)
    assert not package_file.exists()
    assert not dist_info.exists()
    assert prefs["plugin"]["installed_packages"] == {}


def test_python_packages_group_shows_abi_status_and_saves(qtbot, tmp_path: Path) -> None:
    from minflux_viewer.ui.prefs_python_packages import PythonPackagesGroup

    prefs = {
        "plugin": {
            "user_lib_enabled": True,
            "user_lib_paths": [str(tmp_path)],
            "installed_packages": {"demo-package": "1.2.3"},
        }
    }
    group = PythonPackagesGroup()
    qtbot.addWidget(group)
    group.load(prefs)

    assert group.package_list.count() == 1
    assert "demo-package" in group.package_list.item(0).text()
    assert group.path_list.count() == 1
    assert "ABI OK" in group.path_list.item(0).text()

    group.external_enabled.setChecked(False)
    saved = {"plugin": {}}
    group.save(saved)
    assert saved["plugin"]["user_lib_enabled"] is False
    assert saved["plugin"]["user_lib_paths"] == [str(tmp_path)]


def test_python_packages_group_installs_off_gui_thread_and_records_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6.QtCore import Qt

    from minflux_viewer.ui.prefs_python_packages import PythonPackagesGroup

    worker_threads: list[int] = []

    def fake_install(
        spec: str,
        prefs: dict | None = None,
        *,
        target: str | Path | None = None,
        report=None,
    ) -> user_libs.PackageInstallResult:
        assert prefs is None
        assert target is None
        worker_threads.append(threading.get_ident())
        report(f"Installing {spec}…")
        return user_libs.PackageInstallResult(
            "demo-package", "2.0", tmp_path.resolve(), spec
        )

    monkeypatch.setattr(user_libs, "install_package", fake_install)
    prefs = {
        "plugin": {
            "user_lib_enabled": True,
            "user_lib_paths": [],
            "installed_packages": {},
        }
    }
    group = PythonPackagesGroup()
    qtbot.addWidget(group)
    group.load(prefs)
    group.package_edit.setText("demo-package==2.0")
    qtbot.mouseClick(group.install_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not group._tasks, timeout=3000)

    assert worker_threads and worker_threads[0] != threading.get_ident()
    assert prefs["plugin"]["installed_packages"] == {"demo-package": "2.0"}
    assert prefs["plugin"]["user_lib_paths"] == [str(tmp_path.resolve())]
    assert group.package_list.count() == 1


def test_vendor_pip_verifies_and_unpacks_official_wheel_shape(tmp_path: Path) -> None:
    from tools import vendor_pip

    wheel_buffer = io.BytesIO()
    with zipfile.ZipFile(wheel_buffer, "w") as archive:
        archive.writestr("pip/__init__.py", "__version__ = '1.2.3'\n")
        archive.writestr("pip/_vendor/distlib/__init__.py", "")
        archive.writestr("pip-1.2.3.dist-info/METADATA", "Name: pip\nVersion: 1.2.3\n")
    wheel_bytes = wheel_buffer.getvalue()
    wheel_url = "https://files.pythonhosted.org/pip-1.2.3-py3-none-any.whl"
    metadata = {
        "urls": [
            {
                "packagetype": "bdist_wheel",
                "filename": "pip-1.2.3-py3-none-any.whl",
                "url": wheel_url,
                "digests": {"sha256": hashlib.sha256(wheel_bytes).hexdigest()},
            }
        ]
    }

    def opener(url: str, timeout: int = 0) -> io.BytesIO:
        assert timeout == 60
        if url == wheel_url:
            return io.BytesIO(wheel_bytes)
        return io.BytesIO(__import__("json").dumps(metadata).encode("utf-8"))

    target = tmp_path / "pip"
    assert vendor_pip.vendor_pip("1.2.3", target, opener=opener) == target.resolve()
    assert (target / "pip" / "__init__.py").is_file()
    assert (target / "pip" / "_vendor" / "distlib").is_dir()
