"""
minflux_viewer.scripting
========================
Small script-facing facade for ImageJ/Fiji-like automation.

Scripts should use the runtime ``mfv`` module instead of reaching into Qt
window internals directly.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core.app_state import AppState
    from .core.dataset import MinfluxDataset


class ScriptError(RuntimeError):
    """Raised for script-facing API misuse with a clear user message."""


class _AdHocPlotWindow:
    """Lazy Qt wrapper for simple script-created plots."""

    def __init__(self, *, title: str = "Script Plot") -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        from .ui.plot_format import plot_widget

        self.widget = QWidget()
        self.widget.setWindowTitle(title)
        self.widget.setWindowFlags(Qt.WindowType.Window)
        self.widget.resize(760, 560)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        self.plot = plot_widget(background="w")
        layout.addWidget(self.plot)

    def show(self):
        self.widget.show()
        self.widget.raise_()
        self.widget.activateWindow()
        return self

    def close(self) -> None:
        self.widget.close()


class ViewerFacade:
    """Viewer-window helpers exposed as ``mfv.viewer``."""

    def __init__(self, owner: "MinfluxViewerFacade") -> None:
        self._owner = owner

    def render(self, dataset=None):
        main = self._owner.require_main_window()
        idx = self._owner.dataset_index(dataset) if dataset is not None else None
        return main._show_render(idx)

    def scatter(
        self,
        x=None,
        y=None,
        z=None,
        *,
        dataset=None,
        color_by: str | None = None,
        title: str | None = None,
    ):
        main = self._owner.require_main_window()
        if x is None and y is None:
            idx = self._owner.dataset_index(dataset) if dataset is not None else None
            return main._show_scatter(idx)
        if x is None or y is None:
            raise ScriptError("viewer.scatter requires both x and y, or neither for dataset scatter.")
        win = _AdHocPlotWindow(title=title or "Script Scatter")
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.shape != y_arr.shape:
            raise ScriptError("viewer.scatter x and y must have the same length.")
        win.plot.plot(x_arr, y_arr, pen=None, symbol="o", symbolSize=5, symbolBrush=(30, 120, 220, 160))
        win.plot.setLabel("bottom", "x")
        win.plot.setLabel("left", "y")
        if title:
            win.plot.setTitle(title)
        self._owner._keep_window(win)
        return win

    def histogram(self, values=None, *, dataset=None, attr: str | None = None, bins: int | None = None):
        main = self._owner.require_main_window()
        if values is None and attr is None:
            idx = self._owner.dataset_index(dataset) if dataset is not None else None
            return main._show_histogram(idx)
        if values is None:
            values = self._owner.get_attr(attr, dataset=dataset, filtered=True)
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ScriptError("viewer.histogram received no finite values.")
        counts, edges = np.histogram(arr, bins=int(bins or 100))
        centers = 0.5 * (edges[:-1] + edges[1:])
        win = _AdHocPlotWindow(title=f"Script Histogram: {attr or 'values'}")
        win.plot.plot(centers, counts, stepMode=False, fillLevel=0, brush=(220, 80, 40, 120), pen=(170, 50, 30))
        win.plot.setLabel("bottom", attr or "value")
        win.plot.setLabel("left", "count")
        self._owner._keep_window(win)
        return win

    def attribute_plot(self, *, dataset=None, x: str = "idx", y: str = "efo"):
        main = self._owner.require_main_window()
        idx = self._owner.dataset_index(dataset) if dataset is not None else None
        win = main._show_attr_plot(idx)
        if win is not None:
            for name, combo in (("x", getattr(win, "_x_combo", None)), ("y", getattr(win, "_y_combo", None))):
                value = x if name == "x" else y
                if combo is not None and combo.findText(value) >= 0:
                    combo.setCurrentText(value)
        return win


class MinfluxViewerFacade:
    """Stable, compact API intended for interactive scripts."""

    def __init__(self, state: "AppState") -> None:
        self._state = state
        self._main_window = None
        self._windows: list[_AdHocPlotWindow] = []
        self.viewer = ViewerFacade(self)

    def bind_main_window(self, main_window) -> None:
        self._main_window = main_window

    def require_main_window(self):
        if self._main_window is None:
            raise ScriptError("No main viewer window is bound to the scripting API yet.")
        return self._main_window

    def _keep_window(self, win: _AdHocPlotWindow) -> None:
        self._windows.append(win)
        try:
            win.widget.destroyed.connect(lambda _=None, w=win: self._forget_window(w))
        except Exception:
            pass

    def _forget_window(self, win: _AdHocPlotWindow) -> None:
        try:
            self._windows.remove(win)
        except ValueError:
            pass

    def close_windows(self) -> None:
        for win in list(self._windows):
            try:
                win.close()
            except Exception:
                pass
        self._windows.clear()

    def get_active_dataset(self):
        return self._state.active_dataset

    def get_datasets(self) -> list:
        return list(self._state.datasets)

    def get_dataset(self, index_or_name=None):
        if index_or_name is None:
            ds = self.get_active_dataset()
            if ds is None:
                raise ScriptError("No active dataset is loaded.")
            return ds
        for ds in self._state.datasets:
            if index_or_name is ds:
                return ds
        if isinstance(index_or_name, int):
            try:
                return self._state.datasets[index_or_name]
            except IndexError:
                raise ScriptError(f"Dataset index out of range: {index_or_name}") from None
        wanted = str(index_or_name)
        for ds in self._state.datasets:
            if ds.name == wanted or getattr(ds.file, "name", "") == wanted:
                return ds
        raise ScriptError(f"Dataset not found: {wanted}")

    def dataset_index(self, dataset=None) -> int:
        ds = self.get_dataset(dataset)
        for idx, candidate in enumerate(self._state.datasets):
            if candidate is ds:
                return idx
        raise ScriptError("Dataset is not loaded in the current viewer.")

    def get_attr(
        self,
        name: str,
        *,
        dataset=None,
        source: str = "mfx",
        filtered: bool = True,
    ) -> np.ndarray:
        if not name:
            raise ScriptError("Attribute name is required.")
        ds = self.get_dataset(dataset)
        source_key = str(source).lower()
        value = None
        if source_key == "mfx":
            value = ds.mfx.get_attr(name, None)
        elif source_key == "mbm":
            if ds.mbm is None:
                raise ScriptError(f"Dataset '{ds.name}' has no mbm component.")
            value = ds.mbm.get_attr(name, None)
        elif source_key == "derived":
            value = ds.derived.get(name, None)
        elif source_key == "attr":
            value = ds.attr.get(name, None)
        else:
            raise ScriptError("source must be one of: mfx, mbm, derived, attr.")
        if value is None:
            raise ScriptError(f"Attribute '{name}' not found in {source_key}.")
        arr = np.asarray(value)
        return self._apply_filter(ds, arr) if filtered else arr

    def get_loc(self, dataset=None, *, unit: str = "nm", filtered: bool = True) -> np.ndarray:
        ds = self.get_dataset(dataset)
        loc = np.asarray(ds.loc_nm, dtype=float)
        if unit == "m":
            loc = loc / 1e9
        elif unit != "nm":
            raise ScriptError("unit must be 'nm' or 'm'.")
        return self._apply_filter(ds, loc) if filtered else loc

    def _apply_filter(self, ds: "MinfluxDataset", arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 0 or arr.shape[0] != ds.prop.num_loc:
            return arr
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] != arr.shape[0]:
            return arr
        return arr[mask]

    def log(self, message: str, level: str = "INFO") -> None:
        self._state.log(str(message), str(level).upper())

    def show_console(self):
        return self.require_main_window()._show_console()


def create_facade(state: "AppState") -> MinfluxViewerFacade:
    return MinfluxViewerFacade(state)


def install_runtime_module(facade: MinfluxViewerFacade) -> types.ModuleType:
    """Install/update the runtime ``mfv`` module used by in-app scripts."""
    module = sys.modules.get("mfv")
    if module is None:
        module = types.ModuleType("mfv")
        sys.modules["mfv"] = module
    module.__doc__ = "Runtime scripting facade for the active MINFLUX Viewer session."
    module.viewer = facade.viewer
    module.ScriptError = ScriptError
    for name in (
        "get_active_dataset",
        "get_datasets",
        "get_dataset",
        "get_attr",
        "get_loc",
        "log",
        "show_console",
    ):
        setattr(module, name, getattr(facade, name))
    module.__all__ = [
        "get_active_dataset",
        "get_datasets",
        "get_dataset",
        "get_attr",
        "get_loc",
        "log",
        "show_console",
        "viewer",
        "ScriptError",
    ]
    return module
