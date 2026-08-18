"""User controls and manual preview for confocal-to-localization mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..colors import solid_color_names, solid_color_rgba
from ..core.confocal_mapping import (
    ConfocalCandidate,
    ConfocalMappingTransform,
    attach_confocal_signal,
    candidate_attribute_name,
    load_confocal_candidate_array,
    localization_pixel_coordinates,
    mapping_image,
)
from .overlay_alignment import alignment_help_label


@dataclass(frozen=True)
class ConfocalMappingChoice:
    candidate: ConfocalCandidate
    attribute_name: str


@dataclass(frozen=True)
class ConfocalMappingOptions:
    choices: tuple[ConfocalMappingChoice, ...]
    dimension: str
    method: str
    alignment: str


class ConfocalMappingOptionsWidget(QGroupBox):
    """Reusable candidate selection block for reader and Dataset Manager."""

    def __init__(
        self,
        candidates: list[ConfocalCandidate],
        parent: QWidget | None = None,
        *,
        reserved_names=(),
    ) -> None:
        super().__init__("Map fluorescent image signal to localizations", parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._candidates = list(candidates)
        self._reserved_names = {str(name) for name in reserved_names}
        self._rows: list[tuple[QCheckBox, QLineEdit, ConfocalCandidate]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        explanation = QLabel(
            "This MSR file contains potential fluorescent image channels whose "
            "calibrated X/Y bounds match the MINFLUX acquisition ROI within 1%. "
            "Select only the channels to map as additional attributes."
        )
        explanation.setWordWrap(True)
        explanation.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        root.addWidget(explanation)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Image stack"), 0, 0)
        grid.addWidget(QLabel("Attribute name"), 0, 1)
        used_defaults: set[str] = set()
        for row, candidate in enumerate(self._candidates, start=1):
            check = QCheckBox(candidate.name, self)
            match_text = ", ".join(
                f"{match.dataset_key} (X {100 * match.x_error_fraction:.3g}%, "
                f"Y {100 * match.y_error_fraction:.3g}%)"
                for match in candidate.matches
            )
            check.setToolTip(
                f"Shape: {' × '.join(str(v) for v in candidate.shape)} ({candidate.axes})\n"
                f"Matches: {match_text}"
            )
            default = candidate_attribute_name(candidate.name)
            if default in used_defaults:
                suffix = 2
                while f"{default}_{suffix}" in used_defaults:
                    suffix += 1
                default = f"{default}_{suffix}"
            used_defaults.add(default)
            edit = QLineEdit(default, self)
            edit.setEnabled(False)
            check.toggled.connect(edit.setEnabled)
            check.toggled.connect(self._refresh_dimension_availability)
            grid.addWidget(check, row, 0)
            grid.addWidget(edit, row, 1)
            self._rows.append((check, edit, candidate))
        root.addLayout(grid)

        controls = QGridLayout()
        controls.addWidget(QLabel("Signal mapping dimension:"), 0, 0)
        self.dimension_combo = QComboBox(self)
        self.dimension_combo.addItems(["2D", "3D"])
        self.dimension_combo.currentTextChanged.connect(self._on_dimension_changed)
        controls.addWidget(self.dimension_combo, 0, 1)

        controls.addWidget(QLabel("Interpolation method:"), 1, 0)
        self.method_combo = QComboBox(self)
        controls.addWidget(self.method_combo, 1, 1)

        controls.addWidget(QLabel("Image alignment:"), 2, 0)
        self.alignment_combo = QComboBox(self)
        self.alignment_combo.addItems(["automatic", "manual"])
        self.alignment_combo.setToolTip(
            "Automatic uses the calibrated stack geometry. Manual opens an overlay "
            "preview with editable pixel-step and degree-step controls."
        )
        controls.addWidget(self.alignment_combo, 2, 1)
        controls.setColumnStretch(1, 1)
        root.addLayout(controls)
        self._on_dimension_changed("2D")
        self._refresh_dimension_availability()

    def _selected_candidates(self) -> list[ConfocalCandidate]:
        return [candidate for check, _edit, candidate in self._rows if check.isChecked()]

    def _refresh_dimension_availability(self, *_args) -> None:
        selected = self._selected_candidates()
        usable_3d = bool(selected) and all(
            c.has_z and c.z_start_m is not None and c.z_step_m not in (None, 0.0) for c in selected
        )
        model_item = getattr(self.dimension_combo.model(), "item", lambda *_: None)(1)
        if model_item is not None:
            model_item.setEnabled(usable_3d)
        if self.dimension_combo.currentText() == "3D" and not usable_3d:
            self.dimension_combo.setCurrentText("2D")
        self.dimension_combo.setToolTip(
            "3D is available only when every selected stack is ZYX and has a calibrated Z origin and spacing."
            if not usable_3d
            else ""
        )

    def _on_dimension_changed(self, dimension: str) -> None:
        old = self.method_combo.currentText()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        if str(dimension).upper() == "3D":
            methods = ["nearest neighbour", "trilinear"]
            default = "trilinear"
        else:
            methods = ["nearest neighbour", "bilinear", "bicubic"]
            default = "bilinear"
        self.method_combo.addItems(methods)
        self.method_combo.setCurrentText(old if old in methods else default)
        self.method_combo.blockSignals(False)

    def validate(self, *, require_selection: bool = False) -> tuple[bool, str]:
        choices = self.selected_choices()
        if require_selection and not choices:
            return False, "Select at least one fluorescent image channel to map."
        names = [choice.attribute_name for choice in choices]
        if any(not name for name in names):
            return False, "Every selected channel needs a non-empty attribute name."
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            return False, f"Attribute names must be unique: {', '.join(duplicates)}"
        conflicts = sorted(set(names) & self._reserved_names)
        if conflicts:
            return False, f"These attributes already exist: {', '.join(conflicts)}"
        if self.dimension_combo.currentText() == "3D":
            invalid = [
                choice.candidate.name
                for choice in choices
                if not choice.candidate.has_z
                or choice.candidate.z_start_m is None
                or choice.candidate.z_step_m in (None, 0.0)
            ]
            if invalid:
                return False, "3D mapping is not calibrated for: " + ", ".join(invalid)
        return True, ""

    def selected_choices(self) -> tuple[ConfocalMappingChoice, ...]:
        return tuple(
            ConfocalMappingChoice(candidate, edit.text().strip())
            for check, edit, candidate in self._rows
            if check.isChecked()
        )

    def options(self) -> ConfocalMappingOptions:
        return ConfocalMappingOptions(
            choices=self.selected_choices(),
            dimension=self.dimension_combo.currentText(),
            method=self.method_combo.currentText(),
            alignment=self.alignment_combo.currentText(),
        )


class ConfocalMappingOptionsDialog(QDialog):
    """Standalone mapping chooser used from Dataset Manager."""

    def __init__(self, candidates, parent=None, *, reserved_names=()) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map confocal signal to localizations")
        self.resize(650, 360)
        root = QVBoxLayout(self)
        self.options_widget = ConfocalMappingOptionsWidget(
            list(candidates),
            self,
            reserved_names=reserved_names,
        )
        root.addWidget(self.options_widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _accept_if_valid(self) -> None:
        valid, reason = self.options_widget.validate(require_selection=True)
        if not valid:
            QMessageBox.warning(self, "Confocal mapping", reason)
            return
        self.accept()

    def options(self) -> ConfocalMappingOptions:
        return self.options_widget.options()


class ConfocalManualAlignmentDialog(QDialog):
    """Shared image/localization overlay adjusted with keyboard shortcuts."""

    _DEFAULT_IMAGE_COLORS = ("Green", "Magenta", "Yellow", "Orange", "Red", "Blue")
    _DEFAULT_LOCALIZATION_COLOR = "Cyan"
    _DEFAULT_TRANSLATION_STEP = 0.5
    _DEFAULT_ROTATION_STEP = 0.1

    def __init__(
        self,
        candidate: ConfocalCandidate,
        image,
        x_m,
        y_m,
        parent=None,
        *,
        dataset_name: str = "",
        channels: list[tuple[ConfocalCandidate, np.ndarray]] | None = None,
    ) -> None:
        super().__init__(parent)
        source_channels = list(channels) if channels is not None else [(candidate, image)]
        if not source_channels:
            raise ValueError("Manual confocal alignment needs at least one image channel")
        self._channels = [
            (channel, mapping_image(np.asarray(channel_image), "2D"))
            for channel, channel_image in source_channels
        ]
        image_shapes = {channel_image.shape for _channel, channel_image in self._channels}
        if len(image_shapes) != 1:
            raise ValueError("Shared manual alignment channels must have the same Y/X shape")
        self._candidate = self._channels[0][0]
        self._image = self._channels[0][1]
        self._normalized_images = [
            self._normalize_preview_image(channel_image)
            for _channel, channel_image in self._channels
        ]
        self._x_m = np.asarray(x_m, dtype=np.float64).ravel()
        self._y_m = np.asarray(y_m, dtype=np.float64).ravel()
        self._transform = ConfocalMappingTransform()
        self._state = getattr(parent, "_state", None)
        plot_prefs = (
            self._state.prefs.setdefault("plot", {})
            if self._state is not None
            else {}
        )
        translation_step = float(
            plot_prefs.get("confocal_alignment_translation_px", self._DEFAULT_TRANSLATION_STEP)
        )
        rotation_step = float(
            plot_prefs.get("confocal_alignment_rotation_deg", self._DEFAULT_ROTATION_STEP)
        )
        channel_names = ", ".join(channel.name for channel, _image in self._channels)
        self.setWindowTitle(f"Manual fluorescent channel alignment — {channel_names}")
        self.resize(850, 760)

        import pyqtgraph as pg

        self._pg = pg
        root = QVBoxLayout(self)
        title = QLabel(
            f"Overlay: <b>{channel_names}</b>"
            + (f" on <b>{dataset_name}</b>" if dataset_name else "")
        )
        root.addWidget(title)

        self._plot = pg.PlotWidget(background="k")
        self._plot.setAspectLocked(True)
        self._plot.getViewBox().invertY(True)
        self._plot.getPlotItem().setMenuEnabled(False)
        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)
        self._scatter = pg.ScatterPlotItem(
            size=4,
            pen=pg.mkPen(0, 255, 255, 210, width=1),
            brush=pg.mkBrush(0, 255, 255, 80),
        )
        self._plot.addItem(self._scatter)
        ny, nx = self._image.shape
        centre = pg.ScatterPlotItem(
            x=[(nx - 1) / 2],
            y=[(ny - 1) / 2],
            symbol="+",
            size=12,
            pen=pg.mkPen(255, 255, 255, 180, width=1),
            brush=None,
        )
        self._plot.addItem(centre)
        self._plot.setXRange(-0.5, nx - 0.5, padding=0.01)
        self._plot.setYRange(-0.5, ny - 0.5, padding=0.01)
        root.addWidget(self._plot, 1)

        self._channel_controls = QHBoxLayout()
        self._localization_check = QCheckBox("mfx.loc", self)
        self._localization_check.setChecked(True)
        self._localization_check.setToolTip("Show or hide MINFLUX localizations")
        self._channel_controls.addWidget(self._localization_check)
        self._localization_color_combo = self._make_color_combo(
            self._DEFAULT_LOCALIZATION_COLOR
        )
        self._localization_color_combo.setToolTip("Overlay color for MINFLUX localizations")
        self._channel_controls.addWidget(self._localization_color_combo)
        self._localization_check.toggled.connect(self._refresh_localization_overlay)
        self._localization_color_combo.currentTextChanged.connect(
            self._refresh_localization_overlay
        )
        self._image_checks: list[QCheckBox] = []
        self._image_color_combos: list[QComboBox] = []
        for channel, _channel_image in self._channels:
            check = QCheckBox(channel.name, self)
            check.setChecked(True)
            check.setToolTip(
                "Show or hide this image in the alignment preview; mapping selection is unchanged"
            )
            check.toggled.connect(self._refresh_image_preview)
            self._channel_controls.addWidget(check)
            color_combo = self._make_color_combo(
                self._DEFAULT_IMAGE_COLORS[
                    len(self._image_color_combos) % len(self._DEFAULT_IMAGE_COLORS)
                ]
            )
            self._channel_controls.addWidget(color_combo)
            self._image_checks.append(check)
            self._image_color_combos.append(color_combo)
            color_combo.setToolTip(f"Overlay color for {channel.name}")
            color_combo.currentTextChanged.connect(self._refresh_image_preview)
        self._channel_controls.addStretch(1)
        self._reset_button = QPushButton("Reset", self)
        self._reset_button.clicked.connect(self._reset)
        self._channel_controls.addWidget(self._reset_button)
        root.addLayout(self._channel_controls)

        self._help_label = alignment_help_label(self)
        root.addWidget(self._help_label)

        self._transform_report_row = QHBoxLayout()
        self._transform_report_row.addWidget(QLabel("Each key press moves"))
        self._translation_step_spin = self._make_step_spin(translation_step)
        self._transform_report_row.addWidget(self._translation_step_spin)
        self._transform_report_row.addWidget(QLabel("pixel,"))
        self._rotation_step_spin = self._make_step_spin(rotation_step)
        self._transform_report_row.addWidget(self._rotation_step_spin)
        self._transform_report_row.addWidget(QLabel("degree"))
        self._transform_report_row.addSpacing(12)
        self._transform_report_row.addWidget(QLabel("|"))
        self._status = QLabel("")
        self._transform_report_row.addWidget(self._status)
        self._transform_report_row.addStretch(1)
        root.addLayout(self._transform_report_row)
        self._translation_step_spin.valueChanged.connect(self._save_steps)
        self._rotation_step_spin.valueChanged.connect(self._save_steps)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._shortcuts: list[QShortcut] = []
        for sequence, delta in (
            ("Left", (-1.0, 0.0, 0.0)),
            ("Right", (1.0, 0.0, 0.0)),
            ("Up", (0.0, -1.0, 0.0)),
            ("Down", (0.0, 1.0, 0.0)),
            (",", (0.0, 0.0, 1.0)),
            (".", (0.0, 0.0, -1.0)),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda dx=delta[0], dy=delta[1], dr=delta[2]: self._handle_shortcut(
                    dx, dy, dr
                )
            )
            self._shortcuts.append(shortcut)
        self._drag_view = self._plot.viewport()
        self._drag_last: QPointF | None = None
        self._drag_view.installEventFilter(self)
        self._drag_view.setMouseTracking(True)
        self._drag_view.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_view.setToolTip("Drag to move the fluorescent-channel alignment")
        self._refresh_localization_overlay()
        self._refresh_image_preview()
        self._redraw()

    @property
    def transform(self) -> ConfocalMappingTransform:
        return self._transform

    def _make_color_combo(self, color: str) -> QComboBox:
        combo = QComboBox(self)
        combo.addItems(solid_color_names())
        combo.setCurrentText(color)
        combo.setFixedWidth(101)
        return combo

    @staticmethod
    def _make_step_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setFixedWidth(108)
        return spin

    def _handle_shortcut(self, dx: float, dy: float, rotation: float) -> None:
        self._nudge(
            dx * self._translation_step_spin.value(),
            dy * self._translation_step_spin.value(),
            rotation * self._rotation_step_spin.value(),
        )

    def _save_steps(self, _value: float) -> None:
        if self._state is None:
            return
        plot = self._state.prefs.setdefault("plot", {})
        plot["confocal_alignment_translation_px"] = self._translation_step_spin.value()
        plot["confocal_alignment_rotation_deg"] = self._rotation_step_spin.value()
        self._state.save_prefs()

    def eventFilter(self, watched, event) -> bool:
        if watched is not self._drag_view:
            return super().eventFilter(watched, event)
        event_type = event.type()
        if (
            event_type == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag_last = QPointF(event.position())
            watched.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return True
        if event_type == QEvent.Type.MouseMove and self._drag_last is not None:
            current = QPointF(event.position())
            view_box = self._plot.getViewBox()
            start_view = view_box.mapSceneToView(
                self._plot.mapToScene(self._drag_last.toPoint())
            )
            current_view = view_box.mapSceneToView(self._plot.mapToScene(current.toPoint()))
            dx = float(current_view.x() - start_view.x())
            dy = float(current_view.y() - start_view.y())
            if dx or dy:
                self._nudge(dx, dy, 0.0)
            self._drag_last = current
            event.accept()
            return True
        if event_type == QEvent.Type.MouseButtonRelease and self._drag_last is not None:
            self._drag_last = None
            watched.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _nudge(self, dx: float, dy: float, rotation: float) -> None:
        current = self._transform
        self._transform = ConfocalMappingTransform(
            current.dx_pixels + dx,
            current.dy_pixels + dy,
            current.rotation_deg + rotation,
        )
        self._redraw()

    def _reset(self) -> None:
        self._transform = ConfocalMappingTransform()
        self._redraw()

    @staticmethod
    def _normalize_preview_image(image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.float64)
        lo, hi = np.nanpercentile(finite, [1.0, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return np.zeros(values.shape, dtype=np.float64)
        normalized = (values - float(lo)) / (float(hi) - float(lo))
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, 0.0, 1.0)

    def _refresh_image_preview(self, *_args) -> None:
        selected = [
            index for index, check in enumerate(self._image_checks) if check.isChecked()
        ]
        preview = np.zeros((*self._image.shape, 3), dtype=np.float64)
        for index in selected:
            rgba = solid_color_rgba(self._image_color_combos[index].currentText())
            color = np.asarray(rgba[:3], dtype=np.float64) / 255.0
            color *= rgba[3] / 255.0
            preview += self._normalized_images[index][..., None] * color
        preview = np.clip(preview, 0.0, 1.0)
        self._image_item.setImage(
            preview,
            autoLevels=False,
            levels=(0.0, 1.0),
            axisOrder="row-major",
        )

    def _refresh_localization_overlay(self, *_args) -> None:
        red, green, blue, alpha = solid_color_rgba(
            self._localization_color_combo.currentText()
        )
        self._scatter.setPen(self._pg.mkPen(
            red, green, blue, round(220 * alpha / 255), width=1
        ))
        self._scatter.setBrush(self._pg.mkBrush(
            red, green, blue, round(80 * alpha / 255)
        ))
        self._scatter.setVisible(self._localization_check.isChecked())

    def _redraw(self) -> None:
        x, y = localization_pixel_coordinates(
            self._candidate,
            self._x_m,
            self._y_m,
            transform=self._transform,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        indices = np.flatnonzero(finite)
        if indices.size > 50_000:
            indices = indices[np.linspace(0, indices.size - 1, 50_000, dtype=int)]
        self._scatter.setData(x=x[indices], y=y[indices])
        tr = self._transform
        self._status.setText(
            f"X {tr.dx_pixels:+.1f} px | Y {tr.dy_pixels:+.1f} px | "
            f"rotation {tr.rotation_deg:+.1f}°"
        )


def apply_confocal_mapping_options(
    dataset,
    msr_path,
    options: ConfocalMappingOptions,
    parent=None,
    *,
    transform_cache: dict | None = None,
):
    """Apply validated choices to one dataset, opening manual previews as needed.

    *msr_path* is the **source image path**: the ``.msr`` an OBF candidate came
    from, or the ``.tif`` of a standalone candidate built by
    ``confocal_mapping.candidates_from_tiff``. ``load_confocal_candidate_array``
    dispatches on the candidate, so both work here unchanged.
    """
    from ..core.loader import attr_values_1d

    applicable = [
        choice
        for choice in options.choices
        if not choice.candidate.matched_dataset_keys
        or str(dataset.metadata.get("msr_dataset_key", "")) in choice.candidate.matched_dataset_keys
    ]
    if not applicable:
        return []

    x = attr_values_1d(dataset, "loc_x")
    y = attr_values_1d(dataset, "loc_y")
    if x is None or y is None:
        raise ValueError("Dataset has no row-aligned X/Y localization coordinates")

    cache = transform_cache if transform_cache is not None else {}
    loaded_choices = [
        (choice, load_confocal_candidate_array(msr_path, choice.candidate))
        for choice in applicable
    ]
    transforms = [ConfocalMappingTransform() for _choice, _image in loaded_choices]
    geometry_groups: dict[tuple, list[int]] = {}
    dataset_key = str(dataset.metadata.get("msr_dataset_key", dataset.name))
    for index, (choice, _image) in enumerate(loaded_choices):
        candidate = choice.candidate
        geometry_key = (
            candidate.shape,
            candidate.x_start_m,
            candidate.y_start_m,
            candidate.x_step_m,
            candidate.y_step_m,
            dataset_key,
        )
        geometry_groups.setdefault(geometry_key, []).append(index)

    if options.alignment.lower() == "manual":
        for geometry_key, indices in geometry_groups.items():
            transform = cache.get(geometry_key)
            if transform is None:
                first_choice, first_image = loaded_choices[indices[0]]
                channels = [
                    (loaded_choices[index][0].candidate, loaded_choices[index][1])
                    for index in indices
                ]
                dlg = ConfocalManualAlignmentDialog(
                    first_choice.candidate,
                    first_image,
                    x,
                    y,
                    parent,
                    dataset_name=dataset.name,
                    channels=channels,
                )
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    raise RuntimeError("Manual confocal alignment was cancelled")
                transform = dlg.transform
                cache[geometry_key] = transform
            for index in indices:
                transforms[index] = transform

    prepared: list[tuple[ConfocalMappingChoice, np.ndarray, ConfocalMappingTransform]] = []
    for (choice, image), transform in zip(loaded_choices, transforms, strict=True):
        prepared.append((choice, image, transform))

    results = []
    for choice, image, transform in prepared:
        results.append(
            attach_confocal_signal(
                dataset,
                msr_path,
                choice.candidate,
                choice.attribute_name,
                dimension=options.dimension,
                method=options.method,
                alignment=options.alignment,
                transform=transform,
                image=image,
            )
        )
    return results
