'Field dialog'

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
)


class FieldDialog(QDialog):
    """Pick which datasets/fields — and which image series — an export includes.

    ``prechecked_images`` mirrors the mfx/mbm "all" convention: ``None`` means
    *every* image series (the master checkbox), a set means exactly those raw
    stack indices, and the default empty set means none.

    **"Include dataset" hides and shows that dataset's field ticks.** Unticking
    it remembers every tick below it (both "All …" boxes included), clears them
    and disables the two field boxes; ticking it again brings the remembered
    ticks back exactly, so a field that was off stays off. It changes nothing
    else: an excluded dataset is skipped wherever the selection is used, and
    :meth:`result_payload` reports its *remembered* fields, so they are still
    there the next time the dialog opens and the dataset is included again.
    """

    def __init__(self, datasets: list[dict], prechecked: dict[str, dict] | None = None,
                 parent=None, image_series: list[dict] | None = None,
                 prechecked_images: set | None = frozenset()):
        super().__init__(parent)
        self.setWindowTitle("Datasets / Fields included…")
        self.resize(max(560, 320 * max(1, len(datasets))), 560)
        self._vars: dict[str, dict[str, Any]] = {}
        # Field ticks of an excluded dataset, remembered while they are hidden.
        self._hidden: dict[str, dict[str, Any]] = {}
        self._image_checks: dict[int, QCheckBox] = {}
        self._image_all: QCheckBox | None = None

        root = QVBoxLayout(self)
        outer = QHBoxLayout()
        root.addLayout(outer)
        column = QVBoxLayout()

        for ds in datasets:
            prev = prechecked.get(ds["key"], {}) if prechecked else {}
            panel = QGroupBox(ds["name"])
            panel_layout = QVBoxLayout(panel)
            include = QCheckBox("Include dataset")
            include.setChecked(prev.get("checked", True))
            panel_layout.addWidget(include)

            mfx_all = QCheckBox("All mfx fields")
            mfx_all.setChecked(prev.get("mfx") is None if prechecked else True)
            mfx_box = QGroupBox("mfx fields")
            mfx_layout = QVBoxLayout(mfx_box)
            mfx_layout.addWidget(mfx_all)
            mfx_checks = {}
            pre_mfx = set(prev.get("mfx") or []) if prechecked else set()
            for name in ds.get("mfx_fields", []):
                cb = QCheckBox(name)
                cb.setChecked(True if (prev.get("mfx") is None if prechecked else True) else name in pre_mfx)
                mfx_layout.addWidget(cb)
                mfx_checks[name] = cb
            mfx_all.toggled.connect(lambda checked, checks=mfx_checks: [cb.setChecked(checked) for cb in checks.values()])

            mbm_all = QCheckBox("All mbm fields")
            mbm_all.setChecked(prev.get("mbm") is None if prechecked else True)
            mbm_box = QGroupBox("mbm fields")
            mbm_layout = QVBoxLayout(mbm_box)
            mbm_layout.addWidget(mbm_all)
            mbm_checks = {}
            pre_mbm = set(prev.get("mbm") or []) if prechecked else set()
            for name in ds.get("mbm_fields", []):
                cb = QCheckBox(name)
                cb.setChecked(True if (prev.get("mbm") is None if prechecked else True) else name in pre_mbm)
                mbm_layout.addWidget(cb)
                mbm_checks[name] = cb
            mbm_all.toggled.connect(lambda checked, checks=mbm_checks: [cb.setChecked(checked) for cb in checks.values()])

            panel_layout.addWidget(mfx_box)
            panel_layout.addWidget(mbm_box)
            panel_layout.addStretch(1)
            outer.addWidget(panel)
            self._vars[ds["key"]] = {
                "ds": include,
                "mfx_all": mfx_all,
                "mfx": mfx_checks,
                "mfx_box": mfx_box,
                "mbm_all": mbm_all,
                "mbm": mbm_checks,
                "mbm_box": mbm_box,
            }
            include.toggled.connect(
                lambda checked, key=ds["key"]: self._show_dataset_fields(key, checked))
            if not include.isChecked():
                # Built with the remembered ticks above; hide them now.
                self._show_dataset_fields(ds["key"], False)

        # Image series (OBF stacks) → opened in the standalone image viewer, not
        # as MINFLUX datasets. Checked ones open on "Open in MINFLUX viewer".
        # Laid out like a dataset panel: an "all" master above a box of the
        # individual series, because the usual choice is all-or-nothing and only
        # a single-file export cares about picking one channel.
        if image_series:
            include_all = prechecked_images is None
            pre_img = set() if include_all else set(prechecked_images or set())
            panel = QGroupBox("Image series (open in image viewer)")
            panel_layout = QVBoxLayout(panel)
            self._image_all = QCheckBox("Include all images")
            self._image_all.setToolTip(
                "Export/open every image series of every processed file. "
                "Unlike the individual ticks below (which address one file's "
                "stack list) this also covers files with different channels.")
            self._image_all.setChecked(include_all)
            panel_layout.addWidget(self._image_all)

            img_box = QGroupBox("images")
            img_layout = QVBoxLayout(img_box)
            for s in image_series:
                raw = int(s.get("raw_index"))
                extra = "  ·  ".join(p for p in (s.get("shape_str", ""), s.get("dtype", "")) if p)
                cb = QCheckBox(f"{s.get('name', f'Series {raw}')}"
                               + (f"    [{extra}]" if extra else ""))
                cb.setChecked(include_all or raw in pre_img)
                img_layout.addWidget(cb)
                self._image_checks[raw] = cb
            self._image_all.toggled.connect(
                lambda checked, checks=self._image_checks:
                    [cb.setChecked(checked) for cb in checks.values()])
            img_layout.addStretch(1)
            panel_layout.addWidget(img_box)
            panel_layout.addStretch(1)
            column.addWidget(panel)

        if column.count():
            outer.addLayout(column)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _field_ticks(self, key: str) -> dict[str, Any]:
        """The current ticks of one dataset's field boxes."""
        vv = self._vars[key]
        return {
            "mfx_all": vv["mfx_all"].isChecked(),
            "mfx": {name: cb.isChecked() for name, cb in vv["mfx"].items()},
            "mbm_all": vv["mbm_all"].isChecked(),
            "mbm": {name: cb.isChecked() for name, cb in vv["mbm"].items()},
        }

    def _show_dataset_fields(self, key: str, shown: bool) -> None:
        """Hide (remember, clear, disable) or show (restore, enable) the field
        ticks of dataset *key* -- what "Include dataset" does."""
        vv = self._vars[key]
        ticks: dict[str, Any] | None
        if not shown:
            if key not in self._hidden:
                self._hidden[key] = self._field_ticks(key)
            ticks = {"mfx_all": False, "mbm_all": False,
                     "mfx": dict.fromkeys(vv["mfx"], False),
                     "mbm": dict.fromkeys(vv["mbm"], False)}
        else:
            ticks = self._hidden.pop(key, None)
        if ticks is not None:
            for kind in ("mfx", "mbm"):
                # The "All" box first: its toggle sets every field, and each
                # field then gets its own remembered tick.
                vv[f"{kind}_all"].setChecked(bool(ticks[f"{kind}_all"]))
                for name, cb in vv[kind].items():
                    cb.setChecked(bool(ticks[kind].get(name, False)))
        vv["mfx_box"].setEnabled(shown)
        vv["mbm_box"].setEnabled(shown)

    def result_payload(self):
        out = {}
        for key, vv in self._vars.items():
            # An excluded dataset reports the ticks it will get back, not the
            # cleared boxes, so they survive until it is included again.
            ticks = self._hidden.get(key) or self._field_ticks(key)
            out[key] = {
                "checked": vv["ds"].isChecked(),
                "mfx": None if ticks["mfx_all"] else {name for name, on in ticks["mfx"].items() if on},
                "mbm": None if ticks["mbm_all"] else {name for name, on in ticks["mbm"].items() if on},
            }
        return out

    def image_payload(self) -> set | None:
        """The chosen image series: ``None`` for *all*, else raw stack indices.

        ``None`` is not the same as "every box happens to be ticked" — it is the
        instruction to take whatever image series each processed file has, which
        is what a batch over files with differing channels needs."""
        if self._image_all is not None and self._image_all.isChecked():
            return None
        return {raw for raw, cb in self._image_checks.items() if cb.isChecked()}
