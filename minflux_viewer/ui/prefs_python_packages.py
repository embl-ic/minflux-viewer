"""Preferences controls for managed and external Python packages."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import user_libs
from .background_tasks import (
    BackgroundTask,
    retire_background_tasks,
    shared_thread_pool,
)

_PATH_ROLE = int(Qt.ItemDataRole.UserRole)
_PACKAGE_ROLE = _PATH_ROLE + 1


class PythonPackagesGroup(QGroupBox):
    """Managed wheels and ABI-checked external site-packages roots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Python packages", parent)
        self._prefs: dict | None = None
        self._tasks: set[BackgroundTask] = set()
        self._pool = shared_thread_pool("python-packages", max_threads=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        managed_label = QLabel(
            "Managed packages (wheel-only; dependencies must be added explicitly):"
        )
        managed_label.setWordWrap(True)
        layout.addWidget(managed_label)

        managed_path = QLabel(str(user_libs.managed_pylibs_dir()))
        managed_path.setObjectName("managedPythonPath")
        managed_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        managed_path.setStyleSheet("color: palette(mid);")
        layout.addWidget(managed_path)

        self.package_list = QListWidget()
        self.package_list.setObjectName("installedPythonPackages")
        self.package_list.setMinimumHeight(76)
        self.package_list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.package_list)

        install_row = QHBoxLayout()
        self.package_edit = QLineEdit()
        self.package_edit.setObjectName("pythonPackageSpec")
        self.package_edit.setPlaceholderText("Package name, for example pandas==2.3.2")
        self.package_edit.returnPressed.connect(self._install_requested)
        install_row.addWidget(self.package_edit, stretch=1)
        self.install_button = QPushButton("Add")
        self.install_button.setObjectName("installPythonPackage")
        self.install_button.clicked.connect(self._install_requested)
        install_row.addWidget(self.install_button)
        self.remove_package_button = QPushButton("Remove")
        self.remove_package_button.setObjectName("removePythonPackage")
        self.remove_package_button.clicked.connect(self._remove_package_requested)
        install_row.addWidget(self.remove_package_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelPythonPackageTask")
        self.cancel_button.clicked.connect(self._cancel_tasks)
        install_row.addWidget(self.cancel_button)
        layout.addLayout(install_row)

        self.status_label = QLabel("")
        self.status_label.setObjectName("pythonPackageStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addSpacing(4)
        self.external_enabled = QCheckBox("Enable external Python-library paths")
        self.external_enabled.setObjectName("externalPythonPathsEnabled")
        layout.addWidget(self.external_enabled)

        external_label = QLabel(
            "External site-packages (appended after bundled libraries at next startup):"
        )
        external_label.setWordWrap(True)
        layout.addWidget(external_label)

        self.path_list = QListWidget()
        self.path_list.setObjectName("externalPythonPaths")
        self.path_list.setMinimumHeight(76)
        self.path_list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.path_list)

        path_buttons = QHBoxLayout()
        self.add_path_button = QPushButton("Add folder…")
        self.add_path_button.setObjectName("addExternalPythonPath")
        self.add_path_button.clicked.connect(self._add_path_requested)
        path_buttons.addWidget(self.add_path_button)
        self.remove_path_button = QPushButton("Remove folder")
        self.remove_path_button.setObjectName("removeExternalPythonPath")
        self.remove_path_button.clicked.connect(self._remove_path_requested)
        path_buttons.addWidget(self.remove_path_button)
        path_buttons.addStretch()
        layout.addLayout(path_buttons)

        self._selection_changed()

    def _plugin_prefs(self) -> dict:
        if self._prefs is None:
            return {}
        return self._prefs.setdefault("plugin", {})

    @staticmethod
    def _package_version(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("version", "unknown"))
        return str(value)

    def _refresh_packages(self) -> None:
        self.package_list.clear()
        installed = self._plugin_prefs().get("installed_packages", {}) or {}
        for name in sorted(installed, key=str.casefold):
            version = self._package_version(installed[name])
            item = QListWidgetItem(f"{name}  {version}")
            item.setData(_PACKAGE_ROLE, str(name))
            self.package_list.addItem(item)
        self._selection_changed()

    def _add_path_item(self, path: str | Path) -> None:
        text = str(path)
        inspection = user_libs.inspect_path(text)
        item = QListWidgetItem(f"{text}  —  {inspection.message}")
        item.setData(_PATH_ROLE, text)
        item.setToolTip(inspection.message)
        if not inspection.ok:
            item.setForeground(Qt.GlobalColor.red)
        elif inspection.blocked_packages:
            item.setForeground(Qt.GlobalColor.darkYellow)
        self.path_list.addItem(item)

    def load(self, prefs: dict) -> None:
        """Populate the widgets from the Preferences dialog's draft."""
        self._prefs = prefs
        plugin = (prefs or {}).get("plugin", {}) or {}
        self.external_enabled.setChecked(plugin.get("user_lib_enabled", True))
        self.path_list.clear()
        for path in plugin.get("user_lib_paths", []) or []:
            text = str(path).strip()
            if text:
                self._add_path_item(text)
        self._refresh_packages()

    def save(self, prefs: dict) -> None:
        """Write widget values into the Preferences dialog's draft in place."""
        plugin = prefs.setdefault("plugin", {})
        plugin["user_lib_enabled"] = self.external_enabled.isChecked()
        plugin["user_lib_paths"] = [
            str(self.path_list.item(row).data(_PATH_ROLE))
            for row in range(self.path_list.count())
        ]

    def _selection_changed(self, *_args: object) -> None:
        busy = bool(self._tasks)
        self.install_button.setEnabled(not busy)
        self.remove_package_button.setEnabled(
            not busy and self.package_list.currentItem() is not None
        )
        self.cancel_button.setEnabled(busy)
        self.remove_path_button.setEnabled(self.path_list.currentItem() is not None)

    def _add_path_requested(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose external site-packages folder",
            str(Path.home()),
        )
        if not folder:
            return
        key = str(Path(folder).resolve())
        existing = {
            str(Path(str(self.path_list.item(row).data(_PATH_ROLE))).resolve())
            for row in range(self.path_list.count())
        }
        if key not in existing:
            self._add_path_item(key)

    def _remove_path_requested(self) -> None:
        row = self.path_list.currentRow()
        if row >= 0:
            self.path_list.takeItem(row)
        self._selection_changed()

    def _start_task(self, task: BackgroundTask) -> None:
        self._tasks.add(task)
        task.signals.stage.connect(self.status_label.setText)
        task.signals.failed.connect(self._task_failed)
        task.signals.cancelled.connect(self._task_cancelled)
        task.signals.finished.connect(partial(self._task_finished, task))
        self._selection_changed()
        self._pool.start(task)

    def _install_requested(self) -> None:
        spec = self.package_edit.text().strip()
        if not spec or self._tasks:
            return

        def work(report: Any) -> user_libs.PackageInstallResult:
            return user_libs.install_package(spec, report=report)

        task = BackgroundTask(
            work,
            description=f"Install Python package {spec}",
            category="Network",
            discard_result=user_libs.discard_install,
        )
        task.signals.done.connect(self._install_done)
        self._start_task(task)

    def _install_done(self, result: object) -> None:
        if not isinstance(result, user_libs.PackageInstallResult) or self._prefs is None:
            return
        user_libs.record_install(self._prefs, result)
        self.package_edit.clear()
        self.status_label.setText(f"Installed {result.name} {result.version}")
        self._refresh_packages()
        if all(
            str(self.path_list.item(row).data(_PATH_ROLE)) != str(result.target)
            for row in range(self.path_list.count())
        ):
            self._add_path_item(result.target)

    def _remove_package_requested(self) -> None:
        item = self.package_list.currentItem()
        if item is None or self._tasks:
            return
        name = str(item.data(_PACKAGE_ROLE))

        def work(report: Any) -> tuple[str, bool]:
            return name, user_libs.remove_package(name, report=report)

        task = BackgroundTask(
            work,
            description=f"Remove Python package {name}",
            category="I/O",
        )
        task.signals.done.connect(self._remove_done)
        self._start_task(task)

    def _remove_done(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2 or self._prefs is None:
            return
        name, removed = result
        user_libs.record_removal(self._prefs, str(name))
        message = f"Removed {name}" if removed else f"{name} was not present on disk"
        self.status_label.setText(message)
        self._refresh_packages()

    def _task_failed(self, message: str) -> None:
        self.status_label.setText(f"Package operation failed: {message}")

    def _task_cancelled(self) -> None:
        self.status_label.setText("Package operation cancelled")

    def _task_finished(self, task: BackgroundTask) -> None:
        self._tasks.discard(task)
        self._selection_changed()

    def _cancel_tasks(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            self.status_label.setText("Cancelling after the current pip step…")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        retire_background_tasks(self._tasks)
        self._tasks.clear()
        super().closeEvent(event)


def build_python_packages_group(
    parent: QWidget | None = None,
) -> PythonPackagesGroup:
    """Construct the group used by the Preferences Plugin page."""
    return PythonPackagesGroup(parent)
