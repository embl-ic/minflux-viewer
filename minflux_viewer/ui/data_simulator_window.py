"""
Data Simulator (Plugins › Data Simulator).

The full simulation UI: configure a structure + sampling, **Generate** a dataset
on demand, and manage the named presets that appear under *File › Open Sample
Data* (add / update / rename / remove / reorder, plus save/load to a JSON file).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.sample_presets import (
    normalize_preset,
    read_presets_file,
    write_presets_file,
)
from ..core.simulate import param_specs, sim_kind, structure_labels


class SimulateConfigWidget(QWidget):
    """Structure + sampling controls; ``get_config``/``set_config`` a preset dict."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._structure = QComboBox()
        for key, label in structure_labels():
            self._structure.addItem(label, key)
        self._structure.currentIndexChanged.connect(self._rebuild_params)
        top = QFormLayout()
        top.addRow("Structure:", self._structure)
        root.addLayout(top)

        sampling = QGroupBox("Sampling")
        sform = QFormLayout(sampling)
        self._n_points = QSpinBox(); self._n_points.setRange(1, 5_000_000); self._n_points.setValue(2000)
        self._locs = QDoubleSpinBox(); self._locs.setRange(1.0, 100.0); self._locs.setValue(4.0); self._locs.setDecimals(1)
        self._prec = QDoubleSpinBox(); self._prec.setRange(0.0, 100.0); self._prec.setValue(5.0); self._prec.setSuffix(" nm")
        self._channels = QSpinBox(); self._channels.setRange(1, 6); self._channels.setValue(1)
        self._channels.setToolTip("Number of overlaid colour channels; >1 makes a multi-channel overlay.")
        self._is_3d = QCheckBox("3-D (uncheck for planar / 2-D)"); self._is_3d.setChecked(True)
        self._beads = QCheckBox("Include MBM fiducial beads")
        self._beads.setToolTip(
            "Generate synthetic drift-tracking fiducial beads (grd/mbm/points) and "
            "attach them to the dataset, so they are written out when you Export to .msr.")
        self._seed = QSpinBox(); self._seed.setRange(-1, 2_000_000_000); self._seed.setValue(-1)
        self._seed.setSpecialValueText("random"); self._seed.setToolTip("Fixed seed for reproducibility; 'random' = new each generate.")
        sform.addRow("Molecules:", self._n_points)
        sform.addRow("Localizations / molecule:", self._locs)
        sform.addRow("Localization precision:", self._prec)
        sform.addRow("Channels:", self._channels)
        sform.addRow("", self._is_3d)
        sform.addRow("", self._beads)
        sform.addRow("Seed:", self._seed)
        root.addWidget(sampling)

        self._param_box = QGroupBox("Structure parameters")
        self._param_form = QFormLayout(self._param_box)
        self._param_spins: dict = {}
        root.addWidget(self._param_box)
        self._rebuild_params()

    def include_beads(self) -> bool:
        """Whether to attach synthetic MBM fiducial beads to generated datasets."""
        return self._beads.isChecked()

    def _structure_key(self) -> str:
        return self._structure.currentData()

    def _rebuild_params(self, *_a) -> None:
        # Multi-channel sims (NPC overlay / DCR) define their own channels/dims.
        multi = sim_kind(self._structure_key()) != "single"
        self._channels.setEnabled(not multi)
        self._is_3d.setEnabled(not multi)
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._param_spins = {}
        for ps in param_specs(self._structure_key()):
            if ps.integer:
                spin = QSpinBox(); spin.setRange(int(ps.lo), int(ps.hi)); spin.setValue(int(ps.default))
            else:
                spin = QDoubleSpinBox(); spin.setRange(float(ps.lo), float(ps.hi))
                spin.setDecimals(2 if ps.hi <= 1.0 else 1); spin.setValue(float(ps.default))
            if ps.suffix:
                spin.setSuffix(ps.suffix)
            if ps.desc:                                    # hover tooltip
                spin.setToolTip(ps.desc)
                label = QLabel(f"{ps.label}:"); label.setToolTip(ps.desc)
                self._param_form.addRow(label, spin)
            else:
                self._param_form.addRow(f"{ps.label}:", spin)
            self._param_spins[ps.name] = spin

    def get_config(self) -> dict:
        key = self._structure_key()
        seed = self._seed.value()
        return normalize_preset({
            "name": dict(structure_labels()).get(key, key),
            "structure": key,
            "n_points": int(self._n_points.value()),
            "locs_per_trace": float(self._locs.value()),
            "precision_nm": float(self._prec.value()),
            "dim": 3 if self._is_3d.isChecked() else 2,
            "seed": None if seed < 0 else int(seed),
            "channels": int(self._channels.value()),
            "params": {name: spin.value() for name, spin in self._param_spins.items()},
        })

    def set_config(self, preset: dict) -> None:
        p = normalize_preset(preset)
        i = self._structure.findData(p["structure"])
        if i >= 0:
            self._structure.setCurrentIndex(i)           # triggers _rebuild_params
        self._n_points.setValue(int(p["n_points"]))
        self._locs.setValue(float(p["locs_per_trace"]))
        self._prec.setValue(float(p["precision_nm"]))
        self._channels.setValue(int(p["channels"]))
        self._is_3d.setChecked(p["dim"] == 3)
        self._seed.setValue(-1 if p["seed"] is None else int(p["seed"]))
        for name, val in (p["params"] or {}).items():
            spin = self._param_spins.get(name)
            if spin is not None:
                spin.setValue(val)


class DataSimulatorWindow(QWidget):
    """Modeless Data Simulator: generate on demand + manage the Sample Data menu."""

    def __init__(self, state, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._owner = owner
        self.setWindowTitle("Data Simulator")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        # Content lives inside a QScrollArea so the window's minimum height is
        # bounded: tall presets (the NPC-by-DCR type has the most parameter
        # rows) otherwise push the packed layout past the screen height, which
        # on Windows triggers repeated "QWindowsWindow::setGeometry: Unable to
        # set geometry" warnings as the WM cannot grant the demanded size.
        content = QWidget()
        root = QVBoxLayout(content)
        self._config = SimulateConfigWidget()
        root.addWidget(self._config)

        gen = QPushButton("Generate now")
        gen.clicked.connect(self._generate)
        root.addWidget(gen)

        presets = QGroupBox("Sample Data menu presets (File › Open Sample Data)")
        pl = QVBoxLayout(presets)
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _=None: self._load_selected())
        pl.addWidget(self._list)

        row1 = QHBoxLayout()
        for text, slot in (("Add current", self._add_current), ("Update", self._update_selected),
                           ("Rename…", self._rename_selected), ("Remove", self._remove_selected)):
            b = QPushButton(text); b.clicked.connect(slot); row1.addWidget(b)
        pl.addLayout(row1)
        row2 = QHBoxLayout()
        for text, slot in (("Move up", lambda: self._move(-1)), ("Move down", lambda: self._move(1)),
                           ("Load into editor", self._load_selected)):
            b = QPushButton(text); b.clicked.connect(slot); row2.addWidget(b)
        pl.addLayout(row2)
        row3 = QHBoxLayout()
        save_b = QPushButton("Save presets to file…"); save_b.clicked.connect(self._save_file)
        open_b = QPushButton("Open presets from file…"); open_b.clicked.connect(self._open_file)
        row3.addWidget(save_b); row3.addWidget(open_b)
        pl.addLayout(row3)
        root.addWidget(presets)

        hint = QLabel("Generate builds datasets now (version = 'simulation'). Presets add "
                      "editable entries to File › Open Sample Data that regenerate silently.")
        hint.setStyleSheet("color:#888;"); hint.setWordWrap(True)
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Open at the natural content size, but never taller than the screen.
        screen = QApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen is not None else 720
        self.resize(480, min(760, max(400, avail_h - 80)))

        self._presets: list[dict] = []
        self._reload_presets()

    # -- presets -------------------------------------------------------------
    def _reload_presets(self) -> None:
        if self._owner is not None and hasattr(self._owner, "sample_presets"):
            self._presets = [normalize_preset(p) for p in self._owner.sample_presets()]
        self._refresh_list()

    def _refresh_list(self) -> None:
        self._list.clear()
        for p in self._presets:
            self._list.addItem(p["name"])

    def _commit(self) -> None:
        if self._owner is not None and hasattr(self._owner, "set_sample_presets"):
            self._owner.set_sample_presets(self._presets)
        self._refresh_list()

    def _selected(self) -> int:
        return self._list.currentRow()

    def _add_current(self) -> None:
        cfg = self._config.get_config()
        name, ok = QInputDialog.getText(self, "Add preset", "Menu entry name:", text=cfg["name"])
        if not ok or not name.strip():
            return
        cfg["name"] = name.strip()
        self._presets.append(cfg)
        self._commit()
        self._list.setCurrentRow(len(self._presets) - 1)

    def _update_selected(self) -> None:
        i = self._selected()
        if 0 <= i < len(self._presets):
            cfg = self._config.get_config()
            cfg["name"] = self._presets[i]["name"]        # keep the menu name
            self._presets[i] = cfg
            self._commit()
            self._list.setCurrentRow(i)

    def _rename_selected(self) -> None:
        i = self._selected()
        if 0 <= i < len(self._presets):
            name, ok = QInputDialog.getText(self, "Rename preset", "Menu entry name:",
                                            text=self._presets[i]["name"])
            if ok and name.strip():
                self._presets[i]["name"] = name.strip()
                self._commit()
                self._list.setCurrentRow(i)

    def _remove_selected(self) -> None:
        i = self._selected()
        if 0 <= i < len(self._presets):
            del self._presets[i]
            self._commit()

    def _move(self, delta: int) -> None:
        i = self._selected()
        j = i + delta
        if 0 <= i < len(self._presets) and 0 <= j < len(self._presets):
            self._presets[i], self._presets[j] = self._presets[j], self._presets[i]
            self._commit()
            self._list.setCurrentRow(j)

    def _load_selected(self) -> None:
        i = self._selected()
        if 0 <= i < len(self._presets):
            self._config.set_config(self._presets[i])

    def _save_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save sample presets", "sample_presets.json",
                                              "JSON (*.json);;All files (*)")
        if path:
            try:
                write_presets_file(path, self._presets)
            except Exception as exc:
                QMessageBox.warning(self, "Save presets", str(exc))

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open sample presets", "",
                                              "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            loaded = read_presets_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open presets", str(exc))
            return
        if loaded:
            self._presets = loaded
            self._commit()

    # -- generate ------------------------------------------------------------
    def _generate(self) -> None:
        if self._owner is not None and hasattr(self._owner, "generate_simulated"):
            self._owner.generate_simulated(
                self._config.get_config(), include_beads=self._config.include_beads())
