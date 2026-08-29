"""Preferences > Plugin > *Python packages* extension-layer seam.

The Preferences dialog constructs this group and calls :meth:`load` and
:meth:`save` against its draft preferences. Extension-layer Track D owns the
eventual package/path controls, so the foundation keeps the interface stable
without implementing any package-management behaviour.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class PythonPackagesGroup(QGroupBox):
    """Placeholder for managed packages and external library paths."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Python packages", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        placeholder = QLabel(
            "External Python-library management is not enabled in this build."
        )
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)

    def load(self, prefs: dict) -> None:
        """Populate the widgets from the Preferences dialog's draft."""
        del prefs

    def save(self, prefs: dict) -> None:
        """Write widget values into the Preferences dialog's draft in place."""
        del prefs


def build_python_packages_group(
    parent: QWidget | None = None,
) -> PythonPackagesGroup:
    """Construct the group used by the Preferences Plugin page."""
    return PythonPackagesGroup(parent)
