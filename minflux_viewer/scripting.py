"""
minflux_viewer.scripting
========================
Binds the published :mod:`minflux_viewer.api` namespaces to the live
application and installs them as the runtime ``mfv`` module.

This module is the *binding*; the contract itself lives in
``minflux_viewer/api/``. Scripts and plugins use ``mfv``, never Qt window
internals.

Backwards compatibility
-----------------------
Every name the pre-namespace facade published still works and is tested:
``mfv.get_active_dataset``, ``get_datasets``, ``get_dataset``, ``get_attr``,
``get_loc``, ``log``, ``show_console``, ``mfv.viewer.*`` and ``ScriptError``.
They are thin aliases onto the namespaces, so there is one implementation of
each behaviour rather than two that can drift.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import numpy as np

from .api import NAMESPACES, __api_version__
from .api._base import ApiError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core.app_state import AppState
    from .core.dataset import MinfluxDataset


#: Retained for compatibility: scripts catch ``mfv.ScriptError``.
ScriptError = ApiError


class _AdHocPlotWindow:
    """
    Lazy Qt wrapper for the simple script plots the original facade offered.

    Kept as-is for compatibility (``mfv.viewer.scatter(x, y)`` is in the
    published examples and in the Script Editor's default script). Richer
    plotting is ``mfv.plot``.
    """

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
    """
    ``mfv.viewer`` -- the original window helpers.

    Superseded by ``mfv.view`` (windows) and ``mfv.plot`` (ad-hoc plots), and
    kept because published scripts use it.
    """

    def __init__(self, owner: MinfluxViewerFacade) -> None:
        self._owner = owner

    def render(self, dataset=None):
        return self._owner.view.render(dataset)

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
        if x is None and y is None:
            return self._owner.view.scatter(dataset, color_by=color_by)
        if x is None or y is None:
            raise ScriptError(
                "viewer.scatter requires both x and y, or neither for dataset scatter."
            )
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.shape != y_arr.shape:
            raise ScriptError("viewer.scatter x and y must have the same length.")
        win = _AdHocPlotWindow(title=title or "Script Scatter")
        win.plot.plot(x_arr, y_arr, pen=None, symbol="o", symbolSize=5,
                      symbolBrush=(30, 120, 220, 160))
        win.plot.setLabel("bottom", "x")
        win.plot.setLabel("left", "y")
        if title:
            win.plot.setTitle(title)
        self._owner._keep_window(win)
        return win

    def histogram(self, values=None, *, dataset=None, attr: str | None = None,
                  bins: int | None = None):
        if values is None and attr is None:
            return self._owner.view.histogram(dataset)
        if values is None:
            values = self._owner.get_attr(attr, dataset=dataset, filtered=True)
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ScriptError("viewer.histogram received no finite values.")
        counts, edges = np.histogram(arr, bins=int(bins or 100))
        centers = 0.5 * (edges[:-1] + edges[1:])
        win = _AdHocPlotWindow(title=f"Script Histogram: {attr or 'values'}")
        win.plot.plot(centers, counts, stepMode=False, fillLevel=0,
                      brush=(220, 80, 40, 120), pen=(170, 50, 30))
        win.plot.setLabel("bottom", attr or "value")
        win.plot.setLabel("left", "count")
        self._owner._keep_window(win)
        return win

    def attribute_plot(self, *, dataset=None, x: str = "idx", y: str = "efo"):
        return self._owner.view.attribute_plot(dataset, x=x, y=y)


class MinfluxViewerFacade:
    """The object a script sees as ``mfv`` and a plugin receives as ``ctx``."""

    api_version = __api_version__

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._main_window = None
        self._windows: list[_AdHocPlotWindow] = []
        self._tasks: list = []

        # One bound instance of each published namespace.
        for name in NAMESPACES:
            module = __import__(f"minflux_viewer.api.{name}", fromlist=[name])
            cls = getattr(module, name.capitalize())
            setattr(self, name, cls(self))

        self.viewer = ViewerFacade(self)

    # -- application access --------------------------------------------------

    @property
    def state(self):
        """
        Refused. ``AppState`` is not part of the published API.

        The nine namespaces are the contract. Reaching past them into
        application state would couple external code to internals this project
        refactors freely -- exactly what the extension layer exists to prevent.
        If something you need is genuinely missing from the namespaces, that is
        an API gap worth reporting rather than routing around.
        """
        raise ApiError(
            "mfv does not expose AppState. Use the published namespaces "
            "(data, roi, results, plot, view, ui, run, journal, record); if what you "
            "need is missing from them, that is an API gap to report."
        )

    def bind_main_window(self, main_window) -> None:
        self._main_window = main_window

    def require_main_window(self):
        if self._main_window is None:
            raise ScriptError("No main viewer window is bound to the scripting API yet.")
        return self._main_window

    def resolve_dataset(self, dataset: Any = None) -> MinfluxDataset:
        """``None`` / index / name / dataset -> dataset, or a readable error."""
        datasets = list(self._state.datasets)
        if dataset is None:
            active = self._state.active_dataset
            if active is None:
                raise ScriptError("No dataset is loaded.")
            return active
        for candidate in datasets:
            if candidate is dataset:
                return candidate
        if isinstance(dataset, bool):
            raise ScriptError("A dataset reference cannot be a boolean.")
        if hasattr(dataset, "__index__"):
            idx = int(dataset.__index__())
            try:
                return datasets[idx]
            except IndexError:
                raise ScriptError(
                    f"Dataset index {idx} is out of range; {len(datasets)} loaded."
                ) from None
        wanted = str(dataset)
        for candidate in datasets:
            if candidate.name == wanted or getattr(candidate.file, "name", "") == wanted:
                return candidate
        names = ", ".join(repr(d.name) for d in datasets) or "(none loaded)"
        raise ScriptError(f"No dataset named {wanted!r}. Loaded: {names}")

    def resolve_dataset_index(self, dataset: Any = None) -> int:
        ds = self.resolve_dataset(dataset)
        for idx, candidate in enumerate(self._state.datasets):
            if candidate is ds:
                return idx
        raise ScriptError("That dataset is not loaded in the current viewer.")

    # -- lifetime ------------------------------------------------------------

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

    def retain_task(self, task) -> None:
        """Hold a submitted runnable so it is not collected mid-flight."""
        self._tasks.append(task)

    def release_task(self, task) -> None:
        try:
            self._tasks.remove(task)
        except ValueError:
            pass

    def last_task(self):
        return self._tasks[-1] if self._tasks else None

    def close_windows(self) -> None:
        """
        Close every window this facade opened, and stop its tasks.

        Called from the main window's shutdown path. Cancellation is requested,
        never waited on -- the project's rule is that closing must not block on
        a worker.
        """
        for win in list(self._windows):
            try:
                win.close()
            except Exception:
                pass
        self._windows.clear()
        for task in list(self._tasks):
            try:
                from .ui.background_tasks import request_task_cancel

                request_task_cancel(task)
            except Exception:
                pass
        self._tasks.clear()
        for name in ("results", "plot"):
            closer = getattr(getattr(self, name, None), "close_all", None)
            if callable(closer):
                try:
                    closer()
                except NotImplementedError:
                    pass          # namespace not implemented yet
                except Exception:
                    pass

    # -- compatibility aliases -----------------------------------------------

    def get_active_dataset(self):
        return self.data.active()

    def get_datasets(self) -> list:
        return self.data.datasets()

    def get_dataset(self, index_or_name=None):
        return self.data.get(index_or_name)

    def dataset_index(self, dataset=None) -> int:
        return self.resolve_dataset_index(dataset)

    def get_attr(
        self,
        name: str,
        *,
        dataset=None,
        source: str = "mfx",
        filtered: bool = True,
    ) -> np.ndarray:
        """
        Original attribute reader.

        ``source`` selected a storage component. ``"mfx"`` (the default) now
        routes through :meth:`mfv.data.attr`, which resolves cfr/efc to their
        effective iteration -- so it agrees with the Filter dialog where the
        old direct component read did not. The other components are read
        directly, as before.
        """
        if not name:
            raise ScriptError("Attribute name is required.")
        key = str(source).lower()
        if key == "mfx":
            return self.data.attr(name, dataset=dataset, filtered=filtered)

        ds = self.resolve_dataset(dataset)
        if key == "mbm":
            if ds.mbm is None:
                raise ScriptError(f"Dataset '{ds.name}' has no mbm component.")
            value = ds.mbm.get_attr(name, None)
        elif key == "derived":
            value = ds.derived.get(name, None)
        elif key == "attr":
            value = ds.attr.get(name, None)
        else:
            raise ScriptError("source must be one of: mfx, mbm, derived, attr.")
        if value is None:
            raise ScriptError(f"Attribute '{name}' not found in {key}.")
        arr = np.asarray(value)
        if not filtered:
            return arr
        from .api.data import Data

        return Data._apply_filter(ds, arr)

    def get_loc(self, dataset=None, *, unit: str = "nm", filtered: bool = True) -> np.ndarray:
        return self.data.loc(dataset=dataset, unit=unit, filtered=filtered)

    def log(self, message: str, level: str = "INFO") -> None:
        self.ui.log(message, level)

    def show_console(self):
        return self.view.console()


def create_facade(state: AppState) -> MinfluxViewerFacade:
    return MinfluxViewerFacade(state)


def install_runtime_module(facade: MinfluxViewerFacade) -> types.ModuleType:
    """Install/update the runtime ``mfv`` module used by in-app scripts."""
    module = sys.modules.get("mfv")
    if module is None:
        module = types.ModuleType("mfv")
        sys.modules["mfv"] = module
    module.__doc__ = "Runtime scripting facade for the active MINFLUX Viewer session."
    module.__api_version__ = facade.api_version
    module.ScriptError = ScriptError
    module.ApiError = ApiError

    for name in NAMESPACES:
        setattr(module, name, getattr(facade, name))
    module.viewer = facade.viewer

    legacy = (
        "get_active_dataset", "get_datasets", "get_dataset",
        "get_attr", "get_loc", "log", "show_console",
    )
    for name in legacy:
        setattr(module, name, getattr(facade, name))

    module.__all__ = [*NAMESPACES, "viewer", *legacy, "ScriptError", "__api_version__"]
    return module
