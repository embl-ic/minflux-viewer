"""Asking before a processing recipe is applied.

A ``<stem>_viewer_metadata.json`` file changes Z scaling, filters and ROIs, so it is
never restored silently — not even the one sitting next to the data file being
opened. Two questions, two dialogs:

* :class:`SidecarPromptDialog` — "there is a recipe beside this file; load it?"
  Asked **before** the data is read, so the answer can be passed to the loader
  and nothing has to be undone.
* :class:`MetadataApplyDialog` — a recipe handed over on its own; which dataset
  should it act on? A recipe is portable by design (see
  ``core/metadata_match``), so the dropdown offers **every** loaded dataset and
  the match only picks the default.

Both are modal: the caller needs the decision before it can continue.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core.metadata_match import (
    best_match,
    describe_match,
    is_snapshot_recipe,
    recipe_summary,
)

__all__ = [
    "APPLY",
    "KEEP",
    "CANCEL",
    "SKIP",
    "LOAD",
    "SidecarPromptDialog",
    "MetadataApplyDialog",
    "ask_apply_metadata",
    "ask_load_sidecar",
]

#: :class:`MetadataApplyDialog` outcomes.
APPLY, KEEP, CANCEL = "apply", "keep", "cancel"
#: :class:`SidecarPromptDialog` outcomes.
LOAD, SKIP = "load", "skip"

#: Shown when a snapshot recipe would be applied to a dataset other than the
#: one it was baked from. By the baked-XOR-recipe rule such a recipe pins the Z
#: scaling factor to 1.0 and carries no transform or filters, so elsewhere it
#: only flattens Z -- silently, which is why it is said out loud.
SNAPSHOT_WARNING = (
    "This recipe was written for data that already has the processing baked in, "
    "so it carries no transform or filters and pins the Z scaling factor to "
    "1.00. Applying it to other data will flatten that dataset's Z scaling."
)


def _dataset_label(index: int, ds) -> str:
    name = str(getattr(ds, "name", "") or f"dataset {index + 1}")
    return f"{index + 1}: {name}"


class SidecarPromptDialog(QDialog):
    """"A processing recipe sits beside this file — load it?" (point 1)."""

    def __init__(self, data_name: str, sidecar_name: str, meta: dict,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing metadata found")
        self._choice = SKIP

        root = QVBoxLayout(self)
        headline = QLabel(f"'{data_name}' has saved processing beside it:")
        headline.setWordWrap(True)
        root.addWidget(headline)

        detail = QLabel(f"<b>{sidecar_name}</b><br>{recipe_summary(meta)}")
        detail.setWordWrap(True)
        detail.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(detail)

        row = QHBoxLayout()
        row.addStretch()
        load = QPushButton("Load")
        load.setDefault(True)
        load.clicked.connect(lambda: self._done(LOAD))
        row.addWidget(load)
        skip = QPushButton("Skip")
        skip.clicked.connect(lambda: self._done(SKIP))
        row.addWidget(skip)
        root.addLayout(row)

    def _done(self, choice: str) -> None:
        self._choice = choice
        self.accept()

    def choice(self) -> str:
        return self._choice


class MetadataApplyDialog(QDialog):
    """"Apply this recipe on which dataset?" (point 2).

    *datasets* is the full loaded list; the best match only selects the default
    entry, so any dataset can be chosen. With no datasets loaded the Apply row
    is disabled and only Keep / Cancel remain.
    """

    def __init__(self, sidecar_name: str, meta: dict, datasets,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing metadata loading")
        self._choice = CANCEL
        self._datasets = list(datasets or [])
        index, kind = best_match(meta, self._datasets)

        root = QVBoxLayout(self)

        if not self._datasets:
            headline = "No dataset loaded — keep it, then load the dataset"
        elif index is None:
            headline = "No matching dataset — keep it, then load the dataset"
        else:
            headline = f"Matching dataset found ({describe_match(kind)})"
        label = QLabel(f"<b>{headline}</b>")
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        root.addWidget(label)

        detail = QLabel(f"{sidecar_name}<br>{recipe_summary(meta)}")
        detail.setWordWrap(True)
        detail.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(detail)

        if is_snapshot_recipe(meta):
            warning = QLabel(f"⚠ {SNAPSHOT_WARNING}")
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b45309;")
            root.addWidget(warning)

        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply on")
        self._apply_btn.clicked.connect(lambda: self._done(APPLY))
        apply_row.addWidget(self._apply_btn)
        self._combo = QComboBox()
        for position, ds in enumerate(self._datasets):
            self._combo.addItem(_dataset_label(position, ds), position)
        apply_row.addWidget(self._combo, 1)
        root.addLayout(apply_row)

        if not self._datasets:
            self._apply_btn.setEnabled(False)
            self._combo.setEnabled(False)
            self._combo.addItem("(no dataset loaded)")
        elif index is not None:
            self._combo.setCurrentIndex(index)
            self._apply_btn.setDefault(True)

        buttons = QHBoxLayout()
        buttons.addStretch()
        keep = QPushButton("Keep without apply")
        keep.setToolTip(
            "Hold the recipe until the next dataset is loaded. It is discarded "
            "if that dataset does not match, and never survives a restart.")
        keep.clicked.connect(lambda: self._done(KEEP))
        buttons.addWidget(keep)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(lambda: self._done(CANCEL))
        buttons.addWidget(cancel)
        if not self._datasets:
            keep.setDefault(True)
        root.addLayout(buttons)

    def _done(self, choice: str) -> None:
        self._choice = choice
        self.accept() if choice != CANCEL else self.reject()

    def choice(self) -> str:
        return self._choice

    def dataset_index(self) -> int | None:
        """The chosen dataset's index in the list passed in, or ``None``."""
        if not self._datasets:
            return None
        value = self._combo.currentData()
        return int(value) if isinstance(value, int) else None


def ask_load_sidecar(parent, data_name: str, sidecar_name: str, meta: dict) -> str:
    """Run :class:`SidecarPromptDialog`; returns ``LOAD`` or ``SKIP``."""
    dialog = SidecarPromptDialog(data_name, sidecar_name, meta, parent)
    dialog.exec()
    return dialog.choice()


def ask_apply_metadata(parent, sidecar_name: str, meta: dict,
                       datasets) -> tuple[str, int | None]:
    """Run :class:`MetadataApplyDialog`; returns ``(choice, dataset index)``."""
    dialog = MetadataApplyDialog(sidecar_name, meta, datasets, parent)
    dialog.exec()
    return dialog.choice(), dialog.dataset_index()
