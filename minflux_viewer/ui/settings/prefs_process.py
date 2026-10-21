"""
Preferences > Process page: *Filter*, *Channel* and *ROI* sections.

The page mirrors the main window's **Process** menu. Only the Filter section
has settings today: the ordered rules the Filter dialog's **Auto** button
applies (``core/auto_filter.py``). Channel and ROI are present so the page
already has the menu's shape; each says plainly that it has no settings yet.

Seam contract, the same as ``prefs_plugin_paths.py`` -- the dialog calls
exactly three things:

* ``build_process_page(parent)`` at page-build time,
* ``page.load(prefs)`` when the dialog populates itself,
* ``page.save(prefs)`` when the user accepts.

``load``/``save`` take the dialog's **draft** dict and mutate it in place,
because the modeless dialog writes back only the leaves that changed
(``core/app_state.py::merge_pref_changes``). ``process.filter.auto_rules`` is
a list, which that merge treats as one value: it is replaced whole, which is
what an ordered rule list wants.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.core import auto_filter as _af
from minflux_viewer.ui.widgets.attribute_help import apply_attribute_tooltips

#: Offered in the attribute box; it is editable, so any attribute can be typed.
_RULE_ATTRIBUTES = (
    "efo", "cfr", "siz", "dcr", "eco", "ecc", "efc", "fbg",
    "tim", "dur", "len", "spd", "dt", "den",
)

_RULES_HELP = (
    "Rules the Filter dialog's <b>Auto</b> button applies, top to bottom. Each "
    "rule's bounds are computed from the localizations that pass the filters "
    "already switched on and the rules above it, then entered as an ordinary "
    "filter row, which can be edited like any other. A trace-wise attribute "
    "(siz, dur, len) is evaluated once per trace. Every suggestion, and every "
    "rule that finds nothing to suggest, is reported in the dialog and the Log."
)

_METHOD_TIPS = {
    _af.METHOD_SINGLE_EMITTER: (
        "Fits the rates in log space as single emitters at f1 plus double emitters "
        "held at exactly 2·f1, and bounds where a value becomes more likely double "
        "than single. Uses every value, not a histogram, so no bin size enters. "
        "f1 is fitted unless stated. Without a double population the bound is the "
        "stated multiple of f1 (reported as a fallback), or none."),
    _af.METHOD_ROBUST_SPREAD: (
        "median ± k·σ, with σ = 1.4826 × the median absolute deviation (Leys et al. "
        "2013 recommend k = 2.5). Insensitive to the outliers it is meant to remove."),
    _af.METHOD_PERCENTILE: (
        "Keep the values between two percentiles, optionally with an absolute floor "
        "(for siz: at least that many localizations per trace)."),
}


def _spin(lo: float, hi: float, decimals: int, step: float, *, width: int = 86) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setMinimumWidth(width)
    spin.setKeyboardTracking(False)
    return spin


class AutoRuleRow(QWidget):
    """One editable rule: on, attribute, method, the method's parameters."""

    def __init__(self, page: ProcessPage, rule: dict) -> None:
        super().__init__()
        self._page = page
        # Two lines -- the rule, then its parameters indented beneath it -- so a
        # rule fits the page without a horizontal scroll bar.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 4)
        outer.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(6)
        outer.addLayout(row)
        below = QHBoxLayout()
        below.setContentsMargins(26, 0, 0, 0)
        outer.addLayout(below)

        self.enabled = QCheckBox()
        self.enabled.setToolTip("Apply this rule when Auto is pressed.")
        row.addWidget(self.enabled)

        self.attribute = QComboBox()
        self.attribute.setEditable(True)
        self.attribute.addItems(_RULE_ATTRIBUTES)
        apply_attribute_tooltips(self.attribute)
        self.attribute.setMinimumWidth(78)
        self.attribute.setToolTip("Attribute the rule bounds. Any attribute name may be typed.")
        row.addWidget(self.attribute)

        self.method = QComboBox()
        for key, label in _af.METHOD_LABELS.items():
            self.method.addItem(label, key)
            self.method.setItemData(self.method.count() - 1, _METHOD_TIPS[key],
                                    Qt.ItemDataRole.ToolTipRole)
        self.method.currentIndexChanged.connect(self._on_method_changed)
        row.addWidget(self.method)
        row.addStretch(1)

        self.params = QStackedWidget()
        self.params.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._build_params()
        below.addWidget(self.params, stretch=1)

        for text, tip, slot in (("▲", "Move up", lambda: page.move_row(self, -1)),
                                ("▼", "Move down", lambda: page.move_row(self, +1)),
                                ("✕", "Remove this rule", lambda: page.remove_row(self))):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(slot)
            row.addWidget(button)

        self.set_rule(rule)

    # -- parameter pages ------------------------------------------------
    def _build_params(self) -> None:
        valley = QWidget()
        v = QHBoxLayout(valley)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("single-emitter rate f1"))
        self.expected = _spin(0.0, 1.0e9, 0, 1000.0, width=110)
        self.expected.setMaximumWidth(160)
        self.expected.setSpecialValueText("fitted")
        self.expected.setToolTip(
            "The single-emitter rate in the attribute's own units (efo: Hz). "
            "0 = fit it from the data.")
        v.addWidget(self.expected)
        v.addWidget(QLabel("no doubles found → cut at"))
        self.fallback = _spin(0.0, 10.0, 2, 0.25, width=64)
        self.fallback.setMaximumWidth(90)
        self.fallback.setSuffix("·f1")
        self.fallback.setSpecialValueText("no bound")
        self.fallback.setToolTip(
            "When the fit finds no double-emitter population, the bound is this "
            "multiple of f1 (2 = the rate of two emitters of that brightness), "
            "reported as a fallback. 0 = suggest no bound.")
        v.addWidget(self.fallback)
        v.addStretch()

        robust = QWidget()
        r = QHBoxLayout(robust)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(QLabel("k"))
        self.k = _spin(0.1, 20.0, 2, 0.1, width=64)
        self.k.setToolTip("Multiple of σ = 1.4826 × MAD from the median.")
        r.addWidget(self.k)
        r.addWidget(QLabel("bound"))
        self.side = QComboBox()
        for side in _af.ROBUST_SIDES:
            self.side.addItem(side, side)
        r.addWidget(self.side)
        r.addStretch()

        percentile = QWidget()
        p = QHBoxLayout(percentile)
        p.setContentsMargins(0, 0, 0, 0)
        p.addWidget(QLabel("P"))
        self.lower_pct = _spin(0.0, 100.0, 1, 0.5, width=58)
        p.addWidget(self.lower_pct)
        p.addWidget(QLabel("–  P"))
        self.upper_pct = _spin(0.0, 100.0, 1, 0.5, width=58)
        p.addWidget(self.upper_pct)
        self.use_floor = QCheckBox("at least")
        self.use_floor.setToolTip("Also require at least this value (e.g. localizations per trace).")
        p.addWidget(self.use_floor)
        self.floor = _spin(-1.0e9, 1.0e9, 2, 1.0, width=70)
        self.floor.setMaximumWidth(120)
        self.use_floor.toggled.connect(self.floor.setEnabled)
        p.addWidget(self.floor)
        p.addStretch()

        self._pages = {
            _af.METHOD_SINGLE_EMITTER: self.params.addWidget(valley),
            _af.METHOD_ROBUST_SPREAD: self.params.addWidget(robust),
            _af.METHOD_PERCENTILE: self.params.addWidget(percentile),
        }

    def _on_method_changed(self, _index: int) -> None:
        key = self.method.currentData()
        self.params.setCurrentIndex(self._pages.get(key, 0))
        self.method.setToolTip(_METHOD_TIPS.get(key, ""))

    # -- value exchange -------------------------------------------------
    def set_rule(self, rule: dict) -> None:
        defaults = {key: dict(value) for key, value in _af.METHOD_DEFAULTS.items()}
        method = str(rule.get("method") or _af.METHOD_PERCENTILE)
        params = dict(defaults.get(method, {}))
        params.update(rule.get("params") or {})
        self.enabled.setChecked(bool(rule.get("enabled", True)))
        self.attribute.setCurrentText(str(rule.get("attribute") or ""))
        index = self.method.findData(method)
        self.method.setCurrentIndex(max(0, index))
        self._on_method_changed(self.method.currentIndex())
        # Every page keeps its own values, so switching method and back loses nothing.
        valley = {**defaults[_af.METHOD_SINGLE_EMITTER],
                  **(params if method == _af.METHOD_SINGLE_EMITTER else {})}
        robust = {**defaults[_af.METHOD_ROBUST_SPREAD],
                  **(params if method == _af.METHOD_ROBUST_SPREAD else {})}
        pct = {**defaults[_af.METHOD_PERCENTILE],
               **(params if method == _af.METHOD_PERCENTILE else {})}
        self.expected.setValue(float(valley["expected"] or 0.0))
        self.fallback.setValue(float(valley["fallback"] or 0.0))
        self.k.setValue(float(robust["k"]))
        self.side.setCurrentIndex(max(0, self.side.findData(str(robust["side"]))))
        self.lower_pct.setValue(float(pct["lower_pct"]))
        self.upper_pct.setValue(float(pct["upper_pct"]))
        floor = pct.get("min_value")
        self.use_floor.setChecked(floor is not None)
        self.floor.setValue(float(floor) if floor is not None else 0.0)
        self.floor.setEnabled(floor is not None)

    def rule(self) -> dict:
        method = str(self.method.currentData())
        if method == _af.METHOD_SINGLE_EMITTER:
            params: dict = {"expected": float(self.expected.value()),
                            "fallback": float(self.fallback.value())}
        elif method == _af.METHOD_ROBUST_SPREAD:
            params = {"k": float(self.k.value()), "side": str(self.side.currentData())}
        else:
            params = {"lower_pct": float(self.lower_pct.value()),
                      "upper_pct": float(self.upper_pct.value()),
                      "min_value": float(self.floor.value()) if self.use_floor.isChecked() else None}
        return {"enabled": self.enabled.isChecked(),
                "attribute": self.attribute.currentText().strip(),
                "method": method, "params": params}


class ProcessPage(QWidget):
    """Preferences > Process."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        self.filter_group = QGroupBox("Filter")
        filt = QVBoxLayout(self.filter_group)
        filt.setSpacing(6)
        title = QLabel("<b>Auto filtering</b>")
        filt.addWidget(title)
        help_label = QLabel(_RULES_HELP)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray;")
        help_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        filt.addWidget(help_label)

        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(4)
        filt.addLayout(self._rows_box)
        self._rows: list[AutoRuleRow] = []

        buttons = QHBoxLayout()
        self.add_button = QPushButton("+")
        self.add_button.setFixedWidth(36)
        self.add_button.setToolTip("Add a rule (any attribute, any method).")
        self.add_button.clicked.connect(self._add_default_row)
        buttons.addWidget(self.add_button)
        self.restore_button = QPushButton("Restore default rules")
        self.restore_button.setToolTip("Replace the list with the three default rules (efo, cfr, siz).")
        self.restore_button.clicked.connect(lambda _checked=False: self._set_rules(
            _af.default_auto_rules()))
        buttons.addWidget(self.restore_button)
        buttons.addStretch()
        filt.addLayout(buttons)

        refs = QLabel(
            "<span style='color:gray'>Robust spread: Leys <i>et al.</i>, "
            "<i>J. Exp. Soc. Psychol.</i> 49:764 (2013).</span>")
        refs.setWordWrap(True)
        filt.addWidget(refs)
        root.addWidget(self.filter_group)

        for name in ("Channel", "ROI"):
            group = QGroupBox(name)
            box = QVBoxLayout(group)
            note = QLabel("No settings yet.")
            note.setStyleSheet("color: gray;")
            box.addWidget(note)
            root.addWidget(group)
        root.addStretch()

    # -- rows -------------------------------------------------------------
    def rows(self) -> list[AutoRuleRow]:
        return list(self._rows)

    def _add_default_row(self, _checked: bool = False) -> None:
        self._append_row({"enabled": True, "attribute": "dcr",
                          "method": _af.METHOD_PERCENTILE,
                          "params": dict(_af.METHOD_DEFAULTS[_af.METHOD_PERCENTILE])})

    def _append_row(self, rule: dict) -> AutoRuleRow:
        row = AutoRuleRow(self, rule)
        self._rows.append(row)
        self._rows_box.addWidget(row)
        return row

    def remove_row(self, row: AutoRuleRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            self._rows_box.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

    def move_row(self, row: AutoRuleRow, step: int) -> None:
        if row not in self._rows:
            return
        old = self._rows.index(row)
        new = max(0, min(len(self._rows) - 1, old + step))
        if new == old:
            return
        self._rows.insert(new, self._rows.pop(old))
        self._rows_box.removeWidget(row)
        self._rows_box.insertWidget(new, row)

    def _set_rules(self, rules: list[dict]) -> None:
        for row in list(self._rows):
            self.remove_row(row)
        for rule in rules:
            self._append_row(rule)

    # -- seam -------------------------------------------------------------
    def load(self, prefs: dict) -> None:
        stored = ((prefs.get("process") or {}).get("filter") or {}).get("auto_rules")
        rules = _af.default_auto_rules() if stored is None else list(stored)
        self._set_rules([rule for rule in rules if isinstance(rule, dict)])

    def save(self, prefs: dict) -> None:
        process = prefs.setdefault("process", {})
        filt = process.setdefault("filter", {})
        filt["auto_rules"] = [row.rule() for row in self._rows]


def build_process_page(parent: QWidget | None = None) -> ProcessPage:
    return ProcessPage(parent)
