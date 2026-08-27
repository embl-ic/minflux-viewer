"""Modeless, application-wide COLOR settings dialog."""

from __future__ import annotations

import copy

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionFrame,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..colors import (
    DEFAULT_COLOR_PREFS,
    normalize_color_preferences,
    normalize_rgba,
    parse_hex_rgba,
)
from .color_button import ColorSwatchButton, qcolor_from_rgba, rgba_from_qcolor

_VIEWER_ROWS = (
    ("Attribute Plot", (("data", ("viewer", "attribute_data")),
                        ("background", ("viewer", "attribute_background")))),
    ("Histogram", (("data", ("viewer", "histogram_data")),
                   ("background", ("viewer", "histogram_background")))),
    ("Filter", (("range", ("viewer", "filter_range")),
                ("bounds", ("viewer", "filter_bounds")),
                ("text", ("viewer", "filter_text")))),
    ("Overlay", tuple(
        (ordinal, ("viewer", "overlay", index))
        for index, ordinal in enumerate(("1st", "2nd", "3rd", "4th", "5th", "6th"))
    )),
    ("ROI", (("face", ("viewer", "roi_face")),
             ("edge", ("viewer", "roi_edge")),
             ("corner", ("viewer", "roi_corner")),
             ("highlight data in ROI", ("viewer", "roi_highlight")))),
)

#: Viewer rows laid out in one spanning cell rather than shared grid columns.
_VIEWER_SPAN_ROWS = {"ROI"}

#: Most entries wanted on one row.  Narrowing the dialog closes the gaps first
#: and only then moves an entry down, so a section never clips out of view.
_SOLID_WRAP_COLUMNS = 5
_SOLID_MAX_SPACING = 32
_SOLID_MIN_SPACING = 8
_COMPONENT_WRAP_COLUMNS = 5
_COMPONENT_MAX_SPACING = 12
_COMPONENT_MIN_SPACING = 6
_VIEWER_MAX_SPACING = 20
_VIEWER_MIN_SPACING = 6
#: Group-box frame, group margins and grid margins between the scroll viewport
#: and a section's grid; subtracted before deciding how many items fit.
_SECTION_MARGIN_ALLOWANCE = 30
#: Qt lays the basic-color swatches out 8 to a row; used to size a one-swatch inset.
_BASIC_GRID_COLUMNS = 8
#: Qt's own spacing between the picker's stacked blocks, tightened to close the
#: gap between the gradient and the custom-color row beneath it.
_PICKER_LAYOUT_SPACING = 2
#: Blank row above the solid list, so its entries are not jammed under the tab.
_SOLID_TOP_ROWS = 1
#: Height of that blank row; matches one name-plus-swatch entry.
_SOLID_TOP_ROW_HEIGHT = 57
#: Spare row under the tallest tab page, so the section does not end flush.
_TAB_SPARE_ROW = 30
#: 'Viewer / Plots' is the height standard every other tab page is sized against.
_VIEWER_TAB_INDEX = 1
#: Blank leading row in the Viewer grid — 1.5 rows.  Reset is excluded because
#: it sits beside the grid rather than inside it.
_VIEWER_TOP_ROW_HEIGHT = 45
#: Always reserved for a page's vertical scrollbar, so the wrapping grids do not
#: reflow the moment one appears.
_SCROLLBAR_RESERVE = 18


def wrap_columns(available, item_width, *, preferred, max_spacing, min_spacing):
    """Columns and horizontal spacing for a left-packed, wrapping grid.

    Spacing gives way before the row does: the widest arrangement is
    ``preferred`` columns at ``max_spacing``, and as the width drops the gaps
    close to ``min_spacing`` before a column is moved to the next row.  That
    keeps section content — and the Reset button beside it — inside the dialog
    instead of being clipped away.
    """
    item_width = max(1, int(item_width))
    if available <= 0:
        return preferred, max_spacing
    for columns in range(max(1, preferred), 1, -1):
        spacing = (available - columns * item_width) / (columns - 1)
        if spacing >= min_spacing:
            return columns, int(min(spacing, max_spacing))
    return 1, max_spacing
#: The former fixed entry width and the requested 105 % of it.  105 % (53 px) is
#: treated as a floor rather than the answer: ``Magenta`` measures 55 px here and
#: the shortfall depends on the platform font, so the runtime width is measured.
_SOLID_BASE_ENTRY_WIDTH = 50
_SOLID_ENTRY_WIDTH_SCALE = 1.05
#: Breathing room past the measured text.  Sizing exactly to the longest name
#: left 'Magenta' flush against the frame, which still reads as clipped.
_SOLID_ENTRY_PADDING = 3
_MISC_PREVIOUS_HEIGHT = 146
_MISC_TARGET_HEIGHT = int(_MISC_PREVIOUS_HEIGHT * 1.3 + 0.5)
#: 105 % of the content height, so the last row of swatches clears the frame.
_MISC_HEIGHT_HEADROOM = 1.05
#: One tab page plus the palette, rather than every section stacked: the whole
#: point of the tabs is that this no longer grows with the number of sections.
_DEFAULT_DIALOG_HEIGHT = 760
#: Kept clear of the screen edge; the dialog is clamped to whatever fits.
_SCREEN_HEIGHT_MARGIN = 60
#: Width was 1120, then 784.  The floor is set by the two things that cannot
#: wrap — the color-picker panel and the Viewer/Plots rows at their minimum
#: gap (measured ~597 px of content) — and the default leaves room above it so
#: the dialog opens without any section already at its tightest spacing.
_DEFAULT_DIALOG_WIDTH = 660
_MINIMUM_DIALOG_WIDTH = 645


def measure_solid_entry_width(names) -> int:
    """Entry width for the solid list: 105 % of the old width, never clipping.

    The requested 105 % still cut ``Magenta`` off by a couple of pixels, and the
    shortfall is font/DPI dependent, so the longest name is measured instead of
    hard-coding a second magic number.
    """
    scaled = int(_SOLID_BASE_ENTRY_WIDTH * _SOLID_ENTRY_WIDTH_SCALE + 0.5)
    probe = QLineEdit()
    probe.resize(200, probe.sizeHint().height())
    try:
        option = QStyleOptionFrame()
        probe.initStyleOption(option)
        content = probe.style().subElementRect(
            QStyle.SubElement.SE_LineEditContents, option, probe
        )
        overhead = max(0, probe.width() - content.width())
    except (AttributeError, TypeError):  # pragma: no cover - style dependent
        overhead = 8
    metrics = probe.fontMetrics()
    longest = max(
        (metrics.horizontalAdvance(str(name)) for name in names), default=0
    )
    return max(scaled, longest + overhead + _SOLID_ENTRY_PADDING)


class _SolidNameEdit(QLineEdit):
    selected = pyqtSignal()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if not getattr(self, "_restoring_selection_focus", False):
            self.selected.emit()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        return super().event(event)

    def restore_selection_focus(self) -> None:
        self._restoring_selection_focus = True
        try:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        finally:
            self._restoring_selection_focus = False


class _ColorPreview(QWidget):
    """Tall RGBA preview over a checkerboard transparency background."""

    _WIDTH = 64

    def __init__(self) -> None:
        super().__init__()
        self._rgba = (0, 0, 0, 255)
        self._size = QSize(self._WIDTH, 148)
        self.setFixedSize(self._size)

    def set_block_height(self, height: int) -> None:
        """Stretch the block; the panel matches it to the gradient's height."""
        height = max(40, int(height))
        if height == self._size.height():
            return
        self._size = QSize(self._WIDTH, height)
        self.setFixedSize(self._size)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        # setFixedSize does not set the size hint; without this the panel that
        # positions the preview reads width -1 and clips it off its right edge.
        return QSize(self._size)

    def set_rgba(self, rgba) -> None:
        self._rgba = normalize_rgba(rgba)
        self.update()

    def rgba(self) -> tuple[int, int, int, int]:
        return tuple(self._rgba)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        square = 8
        for y in range(0, self.height(), square):
            for x in range(0, self.width(), square):
                shade = 232 if (x // square + y // square) % 2 == 0 else 190
                painter.fillRect(x, y, square, square, QColor(shade, shade, shade))
        painter.fillRect(self.rect(), QColor(*self._rgba))
        painter.setPen(QPen(self.palette().color(self.foregroundRole()), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class _PickerPanel(QWidget):
    """Lay out QColorDialog with the app's own preview / value fields.

    Qt's color dialog is used whole (its palette grids, gradient and alpha bar
    are worth keeping), but its own value form is hidden and replaced.  The
    replacements are positioned rather than laid out, because the pieces have to
    line up with widgets *inside* the picker that Qt does not expose.
    """

    _RAINBOW_BOTTOM_FRACTION = 0.70
    _VERTICAL_GAP = 6
    _PREVIEW_GAP = 12

    def __init__(self, picker: QColorDialog, preview: QWidget,
                 fields: QWidget, update_button: QPushButton,
                 baseline: QWidget | None = None) -> None:
        super().__init__()
        self._picker = picker
        self._preview = preview
        self._fields = fields
        self._update_button = update_button
        # Row the button's foot is aligned to (the HEX field); the fields widget
        # itself ends lower because it carries a trailing stretch.
        self._baseline = baseline
        self._preview_label = QLabel("Preview")
        for child in (picker, preview, fields, update_button, self._preview_label):
            child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.refresh_size()

    def _picker_child_rects(self) -> dict:
        """Geometry of the QColorDialog pieces this panel has to line up with.

        Qt exposes none of them, so they are found by shape and label text:
        the gradient is the one large framed child, and each swatch grid is the
        widget directly under its label.  Every value is optional and the
        callers fall back to proportions, so a Qt layout change degrades to the
        old approximate placement rather than putting things in the wrong spot.
        """
        picker = self._picker
        rects: dict = {"gradient": None, "basic_label": None,
                       "basic_grid": None, "custom_grid": None,
                       "add_button": None}

        for button in picker.findChildren(QPushButton):
            if "add to custom" in button.text().replace("&", "").casefold():
                rects["add_button"] = button.geometry()
                break

        frames = [
            child for child in picker.findChildren(QFrame)
            if child.parentWidget() is picker
            and child.width() > 80 and child.height() > 80
        ]
        if frames:
            rects["gradient"] = max(
                frames, key=lambda f: f.width() * f.height()
            ).geometry()

        labels = {}
        for label in picker.findChildren(QLabel):
            text = label.text().replace("&", "").strip().casefold()
            for key in ("basic", "custom"):
                if text.startswith(key) and key not in labels:
                    labels[key] = label.geometry()
        rects["basic_label"] = labels.get("basic")

        for key in ("basic", "custom"):
            anchor = labels.get(key)
            if anchor is None:
                continue
            below = [
                child.geometry()
                for child in picker.findChildren(QWidget)
                if child.parentWidget() is picker
                and child.geometry().y() >= anchor.bottom()
                and child.geometry().width() > 60
                and child.geometry().height() > 24
                and abs(child.geometry().x() - anchor.x()) <= 8
            ]
            if below:
                rects[f"{key}_grid"] = min(below, key=lambda g: g.y())
        return rects

    def _geometry_values(self):
        picker_size = self._picker.sizeHint()
        fields_size = self._fields.sizeHint()
        button_size = self._update_button.sizeHint()
        label_size = self._preview_label.sizeHint()

        rects = self._picker_child_rects()
        gradient = rects["gradient"]
        basic_label = rects["basic_label"]
        basic_grid = rects["basic_grid"]
        custom_grid = rects["custom_grid"]

        gradient_x = gradient.x() if gradient else picker_size.width() // 2
        column_x = basic_grid.x() if basic_grid else 9

        # Preview: right of the gradient and alpha bar.  Its caption lines up
        # with 'Basic colors' and the block spans down to the gradient's foot.
        preview_x = picker_size.width() + self._PREVIEW_GAP
        label_y = basic_label.y() if basic_label else 9
        preview_y = basic_grid.y() if basic_grid else label_y + label_size.height()
        preview_bottom = gradient.bottom() if gradient else picker_size.height()
        preview_height = max(40, preview_bottom - preview_y)
        preview_size = QSize(self._preview.sizeHint().width(), preview_height)

        # Value fields: first row level with the first row of custom swatches,
        # inset one swatch past the gradient so they clear the left column.
        swatch = (basic_grid.width() // _BASIC_GRID_COLUMNS) if basic_grid else 28
        fields_x = gradient_x + swatch
        fields_y = (
            custom_grid.y() if custom_grid
            else int(picker_size.height() * self._RAINBOW_BOTTOM_FRACTION + 0.5)
        )

        # Update button: matched to 'Add to Custom Colors' in the column above
        # it, and sitting on the same baseline as the last field row (HEX/Alpha)
        # so the two halves of the block end together.
        add_button = rects["add_button"]
        if add_button is not None:
            button_x = add_button.x()
            button_size = QSize(add_button.width(), button_size.height())
        else:
            button_x = column_x
        baseline_bottom = fields_size.height()
        if self._baseline is not None and self._baseline.height() > 0:
            baseline_bottom = self._baseline.geometry().bottom() + 1
        button_y = max(
            fields_y + baseline_bottom - button_size.height(),
            (add_button.bottom() + self._VERTICAL_GAP) if add_button else 0,
        )

        width = max(
            picker_size.width(),
            preview_x + preview_size.width(),
            fields_x + fields_size.width(),
            button_x + button_size.width(),
        )
        height = button_y + button_size.height()
        return (
            picker_size, preview_size, fields_size, button_size, label_size,
            preview_x, preview_y, label_y, fields_x, fields_y,
            button_x, button_y, width, height,
        )

    def sizeHint(self) -> QSize:
        *_, width, height = self._geometry_values()
        return QSize(width, height)

    def refresh_size(self) -> None:
        self.setFixedSize(self.sizeHint())
        self._place_children()

    def _place_children(self) -> None:
        (
            picker_size, preview_size, fields_size, button_size, label_size,
            preview_x, preview_y, label_y, fields_x, fields_y,
            button_x, button_y, _width, _height,
        ) = self._geometry_values()
        self._picker.setGeometry(0, 0, picker_size.width(), picker_size.height())
        self._preview.set_block_height(preview_size.height())
        self._preview_label.setGeometry(
            preview_x, label_y, max(label_size.width(), preview_size.width()),
            label_size.height(),
        )
        self._preview.setGeometry(
            preview_x, preview_y, preview_size.width(), preview_size.height()
        )
        self._fields.setGeometry(
            fields_x, fields_y, fields_size.width(), fields_size.height()
        )
        self._update_button.setGeometry(
            button_x, button_y, button_size.width(), button_size.height()
        )


class GlobalColorDialog(QDialog):
    """Edit the global RGBA registry; Apply keeps this modeless window open."""

    def __init__(self, state) -> None:
        super().__init__(None)
        self._state = state
        self._draft = normalize_color_preferences(state.prefs.get("colors", {}))
        self._applied = copy.deepcopy(self._draft)
        self._buttons: dict[tuple, ColorSwatchButton] = {}
        self._active_path: tuple | None = None
        self._picker_guard = False
        self._apply_in_progress = False
        self._misc_buttons: list[QWidget] = []
        self._solid_origins: dict[str, str | None] = {
            name: name for name in self._draft["solid"]
        }
        self._solid_name_edits: dict[str, _SolidNameEdit] = {}
        self._solid_entry_width = measure_solid_entry_width(self._draft["solid"])
        self._solid_columns_used = _SOLID_WRAP_COLUMNS
        self._solid_relayout_guard = False
        self._solid_reset: QPushButton | None = None

        self.setWindowTitle("COLOR")
        icon_path = resource_path("icons", "minflux_viewer_logo.png")
        self.setWindowIcon(QIcon(str(icon_path)))
        height = self._preferred_height()
        self.resize(_DEFAULT_DIALOG_WIDTH, height)
        self.setMinimumSize(_MINIMUM_DIALOG_WIDTH, height)
        self._build_ui()
        self._load_custom_palette()
        first_name = next(iter(self._draft["solid"]), None)
        first = ("solid", first_name) if first_name else ("viewer", "attribute_data")
        self._select_path(first)
        self._state.colors_changed.connect(self._colors_changed)

    @staticmethod
    def _preferred_height() -> int:
        """Full content height, capped to the screen the dialog opens on.

        The sections need ~1090 px to show everything without an inner
        scrollbar, which is taller than a 1080p desktop; clamping keeps the
        buttons reachable there and simply lets the settings area scroll.
        """
        screen = QApplication.primaryScreen()
        if screen is None:  # pragma: no cover - headless safety
            return _DEFAULT_DIALOG_HEIGHT
        available = screen.availableGeometry().height() - _SCREEN_HEIGHT_MARGIN
        return max(600, min(_DEFAULT_DIALOG_HEIGHT, available))

    @staticmethod
    def _page(content: QWidget) -> QScrollArea:
        """Wrap a section as a scrollable tab page.

        The tab area is capped at the Viewer / Plots height, so a section with
        more entries than fit — a long solid list, a component with many
        features — scrolls instead of squeezing its rows shorter.
        """
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QScrollArea.Shape.NoFrame)
        # The dialog is a fixed width and the grids wrap, so only Y can overflow.
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(content)
        layout.addStretch(1)
        page.setWidget(host)
        return page

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # One section per tab: only the section being edited is on screen, which
        # is what keeps the dialog short instead of stacking every section.
        self._tabs = QTabWidget()
        self._tabs.addTab(self._page(self._solid_section()), "Solid Color List")
        self._tabs.addTab(self._page(self._viewer_section()), "Viewer / Plots")
        self._tabs.addTab(self._page(self._components_section()), "Components")
        self._tabs.addTab(self._page(self._plugins_section()), "Plugins")
        self._tabs.currentChanged.connect(lambda _i: self._refresh_wrapping())
        root.addWidget(self._tabs, 1)

        palette_group = QGroupBox("Custom Color Palette")
        self._palette_group = palette_group
        palette_layout = QVBoxLayout(palette_group)
        palette_layout.setContentsMargins(6, 4, 6, 6)
        palette_layout.setSpacing(4)
        self._picker = QColorDialog()
        self._picker.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        self._picker.setOption(QColorDialog.ColorDialogOption.NoButtons, True)
        self._picker.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self._picker.setWindowFlags(Qt.WindowType.Widget)
        picker_layout = self._picker.layout()
        if picker_layout is not None:
            picker_layout.setSpacing(_PICKER_LAYOUT_SPACING)
        self._hide_picker_native_fields()
        QTimer.singleShot(0, self._hide_picker_native_fields)
        self._picker.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )
        # Preview and value fields are placed independently: the preview sits
        # beside the gradient at the top, the fields below it.
        self._current_preview = _ColorPreview()
        fields_widget = QWidget()
        fields = QVBoxLayout(fields_widget)
        fields.setContentsMargins(0, 0, 0, 0)
        self._build_picker_fields(fields)
        self._update_button = QPushButton("Update Selected Feature Color")
        self._update_button.clicked.connect(self._update_selected)
        self._picker_panel = _PickerPanel(
            self._picker, self._current_preview, fields_widget,
            self._update_button, baseline=self._html,
        )
        palette_layout.addWidget(
            self._picker_panel,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        QTimer.singleShot(0, self._picker_panel.refresh_size)
        palette_group.setMinimumHeight(palette_group.sizeHint().height())
        palette_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        root.addWidget(palette_group, 0)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._apply_button = QPushButton("Apply")
        self._ok_button = QPushButton("OK")
        self._cancel_button = QPushButton("Cancel")
        self._apply_button.clicked.connect(self._apply)
        self._ok_button.clicked.connect(self._accept)
        self._cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self._apply_button)
        buttons.addWidget(self._ok_button)
        buttons.addWidget(self._cancel_button)
        root.addLayout(buttons)

    def _hide_picker_native_fields(self) -> None:
        """Hide Qt's private value form; the safe app-owned form replaces it."""
        labels = self._picker.findChildren(QLabel)
        alpha_label = next(
            (
                label
                for label in labels
                if "alpha channel" in label.text().replace("&", "").casefold()
            ),
            None,
        )
        # Title-case Qt's own headings to match the rest of the dialog, keeping
        # their '&' mnemonics intact.
        for label in labels:
            text = label.text()
            plain = text.replace("&", "").strip()
            if plain in ("Basic colors", "Custom colors"):
                label.setText(text.replace("colors", "Colors"))

        if alpha_label is None:
            return
        alpha_label.parentWidget().hide()

    def _build_picker_fields(self, parent_layout: QVBoxLayout) -> None:
        self._field_guard = False
        form = QGridLayout()
        self._picker_form = form
        form.setContentsMargins(4, 0, 4, 0)
        form.setHorizontalSpacing(5)
        form.setVerticalSpacing(3)

        def spin(maximum: int) -> QSpinBox:
            widget = QSpinBox()
            widget.setRange(0, maximum)
            # Width is the style's own minimum for the range, so a 3-digit
            # value ('255') can never clip the way it did at a flat 76 px.
            widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            widget.setFixedWidth(widget.sizeHint().width())
            return widget

        self._hue = spin(359)
        self._saturation = spin(255)
        self._value_spin = spin(255)
        self._red = spin(255)
        self._green = spin(255)
        self._blue = spin(255)
        self._alpha = spin(255)
        self._html = QLineEdit()
        # Same width as the numeric fields, so the left column lines up.
        self._html.setFixedWidth(self._hue.width())
        self._html.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left = (("Hue:", self._hue), ("Sat:", self._saturation),
                ("Val:", self._value_spin), ("HEX:", self._html))
        right = (("Red:", self._red), ("Green:", self._green),
                 ("Blue:", self._blue), ("Alpha", self._alpha))
        for row, ((left_label, left_widget), (right_label, right_widget)) in enumerate(
            zip(left, right)
        ):
            form.addWidget(QLabel(left_label), row, 0)
            form.addWidget(left_widget, row, 1)
            form.addWidget(QLabel(right_label), row, 3)
            form.addWidget(right_widget, row, 4)
        form.setColumnMinimumWidth(2, 10)
        form.setColumnStretch(5, 1)
        parent_layout.addLayout(form)
        # Pin the value rows to the top of the editor block.  Centred (the
        # default) they sat ~17 px lower, which left no room for the Update
        # button once its bottom was aligned with the preview's bottom edge.
        parent_layout.addStretch(1)

        for widget in (self._hue, self._saturation, self._value_spin):
            widget.valueChanged.connect(self._hsv_fields_changed)
        for widget in (self._red, self._green, self._blue, self._alpha):
            widget.valueChanged.connect(self._rgba_fields_changed)
        # textEdited as well as editingFinished: a pasted hex code must show up
        # in the preview at once, not only when the field happens to lose focus.
        self._html.textEdited.connect(self._html_field_edited)
        self._html.editingFinished.connect(self._html_field_changed)
        self._picker.currentColorChanged.connect(self._sync_picker_fields)
        self._sync_picker_fields(self._picker.currentColor())

    def _sync_picker_fields(self, color) -> None:
        if self._field_guard or not color.isValid():
            return
        self._field_guard = True
        try:
            hue = color.hsvHue()
            self._hue.setValue(0 if hue < 0 else hue)
            self._saturation.setValue(color.hsvSaturation())
            self._value_spin.setValue(color.value())
            self._red.setValue(color.red())
            self._green.setValue(color.green())
            self._blue.setValue(color.blue())
            self._alpha.setValue(color.alpha())
            # Upper case: hex digits are case-insensitive, and '#FF8000' reads
            # more clearly than '#ff8000' beside the numeric fields.  Typed
            # lower case is normalized here when the edit completes.
            self._html.setText(color.name().upper())
            self._current_preview.set_rgba(rgba_from_qcolor(color))
        finally:
            self._field_guard = False

    def _hsv_fields_changed(self) -> None:
        if self._field_guard:
            return
        color = QColor.fromHsv(
            self._hue.value(), self._saturation.value(),
            self._value_spin.value(), self._alpha.value()
        )
        self._picker.setCurrentColor(color)

    def _rgba_fields_changed(self) -> None:
        if self._field_guard:
            return
        self._picker.setCurrentColor(QColor(
            self._red.value(), self._green.value(), self._blue.value(),
            self._alpha.value()
        ))

    def _html_field_edited(self, text: str) -> None:
        """Follow the field live, so a pasted code recolours the preview at once.

        Incomplete input is simply ignored -- typing the second digit of a
        six-digit code must not fight the user by rewriting the field.
        """
        if self._field_guard:
            return
        rgba = parse_hex_rgba(text, default_alpha=self._alpha.value())
        if rgba is None:
            return
        red, green, blue, alpha = rgba
        self._picker.setCurrentColor(QColor(red, green, blue, alpha))

    def _html_field_changed(self) -> None:
        if self._field_guard:
            return
        rgba = parse_hex_rgba(self._html.text(), default_alpha=self._alpha.value())
        if rgba is not None:
            red, green, blue, alpha = rgba
            self._picker.setCurrentColor(QColor(red, green, blue, alpha))
        else:
            # Not a colour at all: put the current one back rather than leaving
            # the field disagreeing with everything beside it.
            self._sync_picker_fields(self._picker.currentColor())

    @staticmethod
    def _reset_button(reset_slot) -> QPushButton:
        """Reset control for a section; the caller decides where it sits.

        It used to occupy a row of its own above the section content.  Each
        section now tucks it into the top-right corner of its first content row.
        """
        reset = QPushButton("Reset")
        reset.clicked.connect(reset_slot)
        return reset

    def _solid_section(self) -> QWidget:
        group = QWidget()          # the tab label names the section
        root = QVBoxLayout(group)
        root.setContentsMargins(4, 4, 4, 4)
        self._solid_host = QWidget()
        self._solid_layout = QGridLayout(self._solid_host)
        self._solid_layout.setContentsMargins(3, 2, 3, 2)
        self._solid_layout.setHorizontalSpacing(_SOLID_MAX_SPACING)
        self._solid_layout.setVerticalSpacing(4)
        self._solid_host.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._solid_host.customContextMenuRequested.connect(self._show_solid_add_menu)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._solid_host, 1)
        self._solid_reset = self._reset_button(self._reset_solid)
        top.addWidget(
            self._solid_reset,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        root.addLayout(top)
        group.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        group.customContextMenuRequested.connect(self._show_solid_add_menu)
        self._rebuild_solid_entries()
        return group

    def _rebuild_solid_entries(self) -> None:
        for path in list(self._buttons):
            if path and path[0] == "solid":
                self._buttons.pop(path, None)
        self._solid_name_edits.clear()
        while self._solid_layout.count():
            item = self._solid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index in range(self._solid_layout.columnCount() + 1):
            self._solid_layout.setColumnStretch(index, 0)
        columns, spacing = self._solid_columns()
        self._solid_columns_used = columns
        self._solid_layout.setHorizontalSpacing(spacing)
        # Blank leading row(s), so the entries sit clear of the tab bar.
        for row in range(_SOLID_TOP_ROWS):
            self._solid_layout.setRowMinimumHeight(row, _SOLID_TOP_ROW_HEIGHT)

        for index, name in enumerate(self._draft["solid"]):
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(3)
            cell.setFixedWidth(self._solid_entry_width)
            editor = _SolidNameEdit(name)
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
            editor.setFixedWidth(self._solid_entry_width)
            editor.selected.connect(lambda n=name: self._select_path(("solid", n)))
            editor.editingFinished.connect(
                lambda old=name, edit=editor: self._rename_solid(old, edit.text())
            )
            editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            editor.customContextMenuRequested.connect(
                lambda pos, n=name, edit=editor: self._show_solid_delete_menu(n, edit, pos)
            )
            self._solid_name_edits[name] = editor
            column.addWidget(editor)
            swatch = self._make_button(("solid", name))
            swatch.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            swatch.customContextMenuRequested.connect(
                lambda pos, n=name, button=swatch: self._show_solid_delete_menu(n, button, pos)
            )
            column.addWidget(swatch, alignment=Qt.AlignmentFlag.AlignCenter)
            self._solid_layout.addWidget(
                cell,
                _SOLID_TOP_ROWS + index // columns,
                index % columns,
                alignment=Qt.AlignmentFlag.AlignLeft,
            )
        # A trailing empty column absorbs the slack so a short final row stays
        # packed to the left instead of spreading across the section.
        self._solid_layout.setColumnStretch(columns, 1)
        self._solid_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

    def _section_content_width(self, reset_button=None) -> int:
        """Width a section's grid may use, measured from the visible viewport.

        A grid's own ``width()`` is driven by the items already in it, so using
        it to decide how many items fit is circular — it never reports that the
        dialog got narrower, which is why the sections used to slide under a
        scrollbar (taking Reset with them) instead of rewrapping.
        """
        viewport = self._tabs.width() - 16 - _SCROLLBAR_RESERVE
        if viewport <= 1:
            return 0
        reserve = _SECTION_MARGIN_ALLOWANCE
        if reset_button is not None:
            reserve += reset_button.sizeHint().width() + 8
        return max(1, viewport - reserve)

    def _solid_columns(self) -> tuple[int, int]:
        """Columns and gap for the solid list at the current width."""
        available = self._section_content_width(self._solid_reset)
        if available <= 1:
            return _SOLID_WRAP_COLUMNS, _SOLID_MAX_SPACING
        return wrap_columns(
            available,
            self._solid_entry_width,
            preferred=_SOLID_WRAP_COLUMNS,
            max_spacing=_SOLID_MAX_SPACING,
            min_spacing=_SOLID_MIN_SPACING,
        )

    def _refresh_palette_height(self) -> None:
        """Re-fit the dialog once the picker reports its real size.

        Its minimum is taken at build time, when QColorDialog still reports the
        un-laid-out 640x480 default — leaving the group hundreds of pixels too
        tall, which centred the panel under a blank band and pushed the bottom
        of the palette off the dialog.  The palette is also the widest thing
        here, so it sets the dialog's fixed width.
        """
        try:
            self._picker_panel.refresh_size()
            group = self._palette_group
            group.setMinimumHeight(0)
            group.layout().activate()
            group.setMinimumHeight(group.sizeHint().height())
            self._fit_tab_height()
            margins = self.layout().contentsMargins()
            self.setFixedWidth(
                group.sizeHint().width() + margins.left() + margins.right()
            )
            # Re-fit the wrapping grids to the final width: they were laid out
            # for the pre-fixed width, which left a Reset button past the edge.
            self.layout().activate()
            self._refresh_wrapping()
        except RuntimeError:  # pragma: no cover - dialog destroyed before timer
            return

    def _fit_tab_height(self) -> None:
        """Cap the tab area at the Viewer / Plots page plus one spare row.

        That page is the height standard.  A section holding more than fits
        (many added solid colors, a component with many features) scrolls
        inside its page rather than making the whole dialog grow.
        """
        standard = self._tabs.widget(_VIEWER_TAB_INDEX)
        inner = standard.widget() if isinstance(standard, QScrollArea) else standard
        bar = self._tabs.tabBar().sizeHint().height()
        self._tabs.setFixedHeight(
            inner.sizeHint().height() + _TAB_SPARE_ROW + bar
        )

    def _refresh_wrapping(self) -> None:
        """Repack every wrapping grid after the dialog width changed."""
        if self._solid_relayout_guard:
            return
        self._solid_relayout_guard = True
        try:
            if self._solid_columns()[0] != self._solid_columns_used:
                self._rebuild_solid_entries()
                if self._active_path is not None:
                    self._select_path(self._active_path)
            self._rebuild_components(
                "functions",
                self._function_combo.currentText(),
                self._function_components,
            )
            self._rebuild_components(
                "plugins", self._plugin_combo.currentText(), self._plugin_components
            )
            self._refresh_viewer_spacing()
        except RuntimeError:  # pragma: no cover - dialog destroyed before timer
            return
        finally:
            self._solid_relayout_guard = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_wrapping()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The picker only reports its real size once laid out, and the panel is
        # fixed-size, so re-measure here or the preview is clipped off the edge.
        QTimer.singleShot(0, self._refresh_palette_height)
        QTimer.singleShot(0, self._refresh_wrapping)

    def _show_solid_delete_menu(self, name: str, source: QWidget, pos) -> None:
        if name not in self._draft["solid"]:
            return
        self._select_path(("solid", name))
        if isinstance(source, QLineEdit):
            menu = source.createStandardContextMenu()
            menu.addSeparator()
        else:
            menu = QMenu(self)
        menu.addAction("Delete", lambda: self._delete_solid(name))
        menu.exec(source.mapToGlobal(pos))

    def _show_solid_add_menu(self, pos) -> None:
        source = self.sender()
        if not isinstance(source, QWidget):
            return
        menu = QMenu(self)
        menu.addAction("Add", self._add_solid)
        menu.exec(source.mapToGlobal(pos))

    def _rename_solid(self, old_name: str, requested_name: str) -> None:
        if old_name not in self._draft["solid"]:
            return
        from ..colormaps import named_colormap_names

        new_name = " ".join(str(requested_name).strip().split())
        if new_name == old_name:
            return
        duplicate = any(
            name != old_name and name.casefold() == new_name.casefold()
            for name in self._draft["solid"]
        )
        reserved = {
            name.casefold()
            for name in named_colormap_names(include_custom=True, include_legacy=True)
        }
        if (
            not new_name
            or duplicate
            or new_name.casefold().startswith("custom:")
            or new_name.casefold() in reserved
        ):
            editor = self._solid_name_edits.get(old_name)
            if editor is not None:
                editor.setText(old_name)
            QMessageBox.warning(
                self,
                "Solid color name",
                "Use a non-empty, unique name that is not already used by a colormap.",
            )
            return
        renamed: dict[str, list[int]] = {}
        for name, color in self._draft["solid"].items():
            renamed[new_name if name == old_name else name] = color
        self._draft["solid"] = renamed
        origin = self._solid_origins.pop(old_name, None)
        self._solid_origins[new_name] = origin
        self._active_path = ("solid", new_name)
        self._rebuild_solid_entries()
        self._select_path(self._active_path)

    def _delete_solid(self, name: str) -> None:
        if name not in self._draft["solid"]:
            return
        names = list(self._draft["solid"])
        index = names.index(name)
        self._draft["solid"].pop(name)
        self._solid_origins.pop(name, None)
        self._rebuild_solid_entries()
        remaining = list(self._draft["solid"])
        if remaining:
            next_name = remaining[min(index, len(remaining) - 1)]
            self._select_path(("solid", next_name))
        else:
            self._select_path(("viewer", "attribute_data"))

    def _add_solid(self) -> None:
        existing = {name.casefold() for name in self._draft["solid"]}
        number = 1
        while f"color {number}" in existing:
            number += 1
        name = f"Color {number}"
        color = self._picker.currentColor()
        rgba = rgba_from_qcolor(color) if color.isValid() else (0, 0, 0, 255)
        self._draft["solid"][name] = list(rgba)
        self._solid_origins[name] = None
        self._rebuild_solid_entries()
        self._select_path(("solid", name))

    def _viewer_section(self) -> QWidget:
        group = QWidget()
        root = QVBoxLayout(group)
        root.setContentsMargins(4, 4, 4, 4)
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(2)
        # Blank leading row: the rows sit lower without moving Reset, which is
        # placed beside the grid rather than inside it.
        grid.setRowMinimumHeight(0, _VIEWER_TOP_ROW_HEIGHT)
        for index, (name, components) in enumerate(_VIEWER_ROWS):
            row_index = index + 1
            label = QLabel(f"{name}:")
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            grid.addWidget(label, row_index, 0)
            if name in _VIEWER_SPAN_ROWS:
                # This row's labels are descriptive rather than one-word, and a
                # shared grid column would size every other row to the longest
                # of them.  Lay it out in one spanning cell instead, so 'ROI'
                # cannot widen the Overlay row above it.
                span = QHBoxLayout()
                span.setContentsMargins(0, 0, 0, 0)
                span.setSpacing(14)
                for component, path in components:
                    span.addWidget(QLabel(component))
                    span.addWidget(self._make_button(path))
                span.addStretch(1)
                grid.addLayout(span, row_index, 1, 1, 7)
                continue
            for col, (component, path) in enumerate(components, start=1):
                pair = QHBoxLayout()
                pair.setContentsMargins(0, 0, 0, 0)
                pair.setSpacing(4)
                pair.addWidget(QLabel(component))
                pair.addWidget(self._make_button(path))
                grid.addLayout(pair, row_index, col)
        grid.setColumnStretch(7, 1)
        self._viewer_grid = grid
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addLayout(grid, 1)
        self._viewer_reset = self._reset_button(self._reset_viewer)
        top.addWidget(
            self._viewer_reset,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        root.addLayout(top)
        return group

    def _refresh_viewer_spacing(self) -> None:
        """Close the Viewer/Plots gaps as the dialog narrows.

        Its rows are meaningful groupings and deliberately never wrap, so the
        only thing that can give is the space between the pairs.
        """
        grid = self._viewer_grid
        widths: dict[int, int] = {}
        for index in range(grid.count()):
            _row, column, _row_span, column_span = grid.getItemPosition(index)
            if column_span > 1:
                continue          # a spanning row does not size a single column
            widths[column] = max(
                widths.get(column, 0), grid.itemAt(index).sizeHint().width()
            )
        gaps = max(1, len(widths) - 1)
        available = self._section_content_width(self._viewer_reset)
        if available <= 1:
            return
        spacing = (available - sum(widths.values())) / gaps
        grid.setHorizontalSpacing(
            int(min(_VIEWER_MAX_SPACING, max(_VIEWER_MIN_SPACING, spacing)))
        )

    def _components_section(self) -> QWidget:
        self._function_combo = QComboBox()
        self._function_combo.addItems(list(self._draft.get("functions", {})))
        page, self._function_row, self._function_components, self._function_label = (
            self._selector_section(self._function_combo, self._reset_functions)
        )
        self._function_combo.currentTextChanged.connect(
            lambda name: self._rebuild_components(
                "functions", name, self._function_components
            )
        )
        self._rebuild_components(
            "functions", self._function_combo.currentText(), self._function_components
        )
        return page

    def _plugins_section(self) -> QWidget:
        self._plugin_combo = QComboBox()
        self._plugin_combo.addItems(list(self._draft.get("plugins", {})))
        page, self._plugin_row, self._plugin_components, self._plugin_label = (
            self._selector_section(self._plugin_combo, self._reset_plugins)
        )
        self._plugin_combo.currentTextChanged.connect(
            lambda name: self._rebuild_components(
                "plugins", name, self._plugin_components
            )
        )
        self._rebuild_components(
            "plugins", self._plugin_combo.currentText(), self._plugin_components
        )
        return page

    def _selector_section(self, combo: QComboBox, reset_slot):
        """A dropdown-driven section: Reset on top, then the combo and items."""
        group = QWidget()
        root = QVBoxLayout(group)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(5)
        label = QLabel("")           # kept for API/back-compat; the tab names it
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(label)
        title_row.addStretch(1)
        title_row.addWidget(
            self._reset_button(reset_slot),
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        root.addLayout(title_row)

        combo.setFixedWidth(210)
        content_row = QVBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(5)
        content_row.addWidget(combo, 0, Qt.AlignmentFlag.AlignLeft)
        host = QWidget()
        components = QGridLayout(host)
        components.setContentsMargins(0, 0, 0, 0)
        components.setHorizontalSpacing(_COMPONENT_MAX_SPACING)
        components.setVerticalSpacing(5)
        content_row.addWidget(host)
        root.addLayout(content_row)
        return group, content_row, components, label

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_components(self, section: str, group: str, layout: QGridLayout) -> None:
        for path in list(self._buttons):
            if path and path[0] == section:
                self._buttons.pop(path, None)
        for index in range(layout.columnCount() + 1):
            layout.setColumnStretch(index, 0)
        self._clear_layout(layout)
        components = self._draft.get(section, {}).get(group, {})

        def _pair(path: tuple, text: str) -> QWidget:
            pair = QWidget()
            pair.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
            )
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(4)
            label = QLabel(text)
            label.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
            )
            pair_layout.addWidget(label)
            pair_layout.addWidget(self._make_button(path))
            return pair

        # A group may nest one level: ``{row-name: {item: color}}`` renders as
        # labelled rows (ROI Manager's entries vs selected), while a flat group
        # keeps the wrapping grid.  Both go through the same builder.
        rows: list[tuple[str | None, list[QWidget]]] = []
        if components and all(
            isinstance(value, dict) for value in components.values()
        ):
            for row_name, items in components.items():
                rows.append((
                    row_name,
                    [
                        _pair((section, group, row_name, item), item)
                        for item in items
                    ],
                ))
        else:
            rows.append((
                None,
                [_pair((section, group, name), name) for name in components],
            ))

        # Wrap on the widest built item, so a long name ('Autocorrelation') can
        # never be the one that gets cut off at the right edge.
        widest = max(
            (pair.sizeHint().width() for _name, pairs in rows for pair in pairs),
            default=1,
        )
        columns, spacing = wrap_columns(
            self._section_content_width(),
            widest,
            preferred=_COMPONENT_WRAP_COLUMNS,
            max_spacing=_COMPONENT_MAX_SPACING,
            min_spacing=_COMPONENT_MIN_SPACING,
        )
        layout.setHorizontalSpacing(spacing)

        grid_row = 0
        heading_column = 0
        for row_name, pairs in rows:
            offset = 0
            if row_name is not None:
                heading = QLabel(f"{row_name}:")
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
                layout.addWidget(
                    heading, grid_row, 0, alignment=Qt.AlignmentFlag.AlignLeft
                )
                heading_column = 1
                offset = 1
            for index, pair in enumerate(pairs):
                position = index + offset
                layout.addWidget(
                    pair,
                    grid_row + position // max(1, columns),
                    position % max(1, columns),
                    alignment=Qt.AlignmentFlag.AlignLeft,
                )
            grid_row += max(1, -(-(len(pairs) + offset) // max(1, columns)))
        # Trailing empty column takes the slack so items stay packed left.
        layout.setColumnStretch(max(columns, heading_column), 1)

    def _value(self, path: tuple):
        value = self._draft
        for key in path:
            value = value[key]
        return value

    def _set_value(self, path: tuple, value) -> None:
        target = self._draft
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = list(normalize_rgba(value))
        button = self._buttons.get(path)
        if button is not None:
            button.set_rgba(value)

    def _make_button(self, path: tuple) -> ColorSwatchButton:
        button = ColorSwatchButton(self._value(path))
        button.setProperty("color_path", path)
        button.clicked.connect(lambda _checked=False, p=path: self._select_path(p))
        self._buttons[path] = button
        return button

    def _select_path(self, path: tuple) -> None:
        try:
            value = self._value(path)
        except (KeyError, IndexError):
            return
        focus_widget = QApplication.focusWidget()
        self._active_path = path
        self._picker_guard = True
        try:
            self._picker.setCurrentColor(qcolor_from_rgba(value))
        finally:
            self._picker_guard = False
        for candidate, button in self._buttons.items():
            button.setStyleSheet(
                "QPushButton { border: 2px solid palette(highlight); }"
                if candidate == path else ""
            )
        selected_solid = path[1] if len(path) == 2 and path[0] == "solid" else None
        for name, editor in self._solid_name_edits.items():
            editor.setStyleSheet(
                "QLineEdit { border: 2px solid palette(highlight); }"
                if name == selected_solid else ""
            )
        if isinstance(focus_widget, _SolidNameEdit):
            focus_widget.restore_selection_focus()

    def _update_selected(self) -> None:
        color = self._picker.currentColor()
        if self._active_path is None or not color.isValid():
            return
        self._set_value(self._active_path, rgba_from_qcolor(color))

    def _reset_section(self, key: str) -> None:
        self._draft[key] = copy.deepcopy(DEFAULT_COLOR_PREFS[key])
        for path, button in list(self._buttons.items()):
            if path and path[0] == key:
                button.set_rgba(self._value(path))
        if key in ("functions", "plugins"):
            self._rebuild_components("functions", self._function_combo.currentText(), self._function_components)
            self._rebuild_components("plugins", self._plugin_combo.currentText(), self._plugin_components)
        if self._active_path and self._active_path[0] == key:
            self._select_path(self._active_path)

    def _reset_solid(self) -> None:
        applied_names = set(self._applied.get("solid", {}))
        self._draft["solid"] = copy.deepcopy(DEFAULT_COLOR_PREFS["solid"])
        self._solid_origins = {
            name: name if name in applied_names else None
            for name in self._draft["solid"]
        }
        self._rebuild_solid_entries()
        first = next(iter(self._draft["solid"]), None)
        if first is not None:
            self._select_path(("solid", first))

    def _reset_viewer(self) -> None:
        self._reset_section("viewer")

    def _reset_functions(self) -> None:
        # Components and Plugins are separate tabs, so each resets only its own.
        self._reset_section("functions")

    def _reset_plugins(self) -> None:
        self._reset_section("plugins")

    def _load_custom_palette(self) -> None:
        colors = list(self._draft.get("custom_palette", []))
        defaults = DEFAULT_COLOR_PREFS["custom_palette"]
        for index in range(QColorDialog.customCount()):
            value = colors[index] if index < len(colors) else defaults[index % len(defaults)]
            QColorDialog.setCustomColor(index, qcolor_from_rgba(value))

    def _capture_custom_palette(self) -> None:
        self._draft["custom_palette"] = [
            list(rgba_from_qcolor(QColorDialog.customColor(index)))
            for index in range(QColorDialog.customCount())
        ]

    def _apply(self) -> set[str]:
        self._capture_custom_palette()
        solid_renames = {
            origin: name
            for name, origin in self._solid_origins.items()
            if origin is not None and origin != name
        }
        self._apply_in_progress = True
        try:
            changed = self._state.apply_color_preferences(
                self._draft, solid_renames=solid_renames
            )
            self._applied = copy.deepcopy(self._draft)
            self._solid_origins = {name: name for name in self._draft["solid"]}
        finally:
            self._apply_in_progress = False
        return changed

    def _colors_changed(self, _payload=None) -> None:
        """Synchronize an already-open editor after Preferences applies."""
        if self._apply_in_progress:
            return
        self._draft = normalize_color_preferences(self._state.prefs.get("colors", {}))
        self._applied = copy.deepcopy(self._draft)
        self._solid_origins = {name: name for name in self._draft["solid"]}
        self._rebuild_solid_entries()
        for path, button in list(self._buttons.items()):
            try:
                button.set_rgba(self._value(path))
            except (KeyError, IndexError):
                pass
        self._rebuild_components(
            "functions", self._function_combo.currentText(), self._function_components
        )
        self._rebuild_components(
            "plugins", self._plugin_combo.currentText(), self._plugin_components
        )
        self._load_custom_palette()
        if self._active_path in self._buttons:
            self._select_path(self._active_path)
        else:
            first = next(iter(self._draft["solid"]), None)
            self._select_path(
                ("solid", first) if first else ("viewer", "attribute_data")
            )

    def _accept(self) -> None:
        self._apply()
        self.accept()
