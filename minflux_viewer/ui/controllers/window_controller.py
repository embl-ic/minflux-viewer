"""Window controller delegated by MainWindow."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from minflux_viewer.ui.common.dataset_binding import (
    DatasetWindowMap,
    event_index,
    window_dataset_index,
    window_draws_dataset,
)
from minflux_viewer.ui.controllers.window_support import _raise_or_create

if TYPE_CHECKING:
    from minflux_viewer.ui.main_window import MainWindow


def _place_new_window(self: MainWindow, win, dataset_idx: int | None = None) -> None:
    """Open *win* clear of the windows it would otherwise hide.

        A render view and an image viewer are both 880x920 and **neither
        positions itself**, so Qt opens them at the same spot and the second
        covers the first completely -- which is exactly what opening a ``.msr``
        carrying image series looked like: the render view appeared and was gone
        again before it could be read.

        What it avoids: every open coordinate view and image viewer (one view of
        the data must never hide another), plus this dataset's own plots. Other
        datasets' histograms and attribute plots are deliberately left out -- in
        a busy session avoiding everything would push each new window into a
        corner, and ``modeless.open_position`` would be choosing between bad
        options rather than good ones.
        """
    from minflux_viewer.ui.common.modeless import place_clear_of
    # Insertion order == opening order, and open_position prefers a side of
    # the most recent window, so the newest view is what a new one sits
    # beside. Registries are dicts, which preserve it.
    occupied = list(self._tiff_windows.values())
    occupied += list(self._render_windows.values())
    if dataset_idx is not None:
        for registry in (
            self._scatter_windows,
            self._tracking_windows,
            self._msd_windows,
            self._histogram_windows,
            self._attr_windows,
            self._data_windows,
        ):
            other = registry.get(dataset_idx)
            if other is not None:
                occupied.append(other)
    place_clear_of(win, occupied)


def _mbm_windows(self: MainWindow) -> list:
    """Every visible bead-info window, whichever module it came from."""
    found = []
    for widget in QApplication.topLevelWidgets():
        try:
            if type(widget).__name__ not in self.MBM_WINDOW_CLASSES:
                continue
            if not widget.isVisible():
                continue
        except RuntimeError:        # its C++ object is already gone
            continue
        found.append(widget)
    return found


def _dataset_windows(self: MainWindow, dataset_indices) -> list:
    """The plot/info windows of *dataset_indices*, render view last.

        Last because the caller raises in order, so the render view -- the view
        of the localizations themselves -- ends up in front of the plots derived
        from them.
        """
    windows: list = []
    registries = (
        self._data_windows,
        self._attr_windows,
        self._histogram_windows,
        self._scatter_windows,
        self._tracking_windows,
        self._msd_windows,
        self._render_windows,
    )
    for registry in registries:
        for idx in dataset_indices or ():
            win = registry.get(idx)
            if win is not None and win not in windows:
                windows.append(win)
    return windows


def restack_import_windows(self: MainWindow, dataset_indices=()) -> None:
    """Stack an import's windows the way the MSR reader lists its contents.

        The reader shows a file as mfx, then mbm, then image series, so that is
        the order they are given on screen: the MINFLUX data views in front, the
        bead info behind them, the image viewer behind that. Raising runs
        **bottom-up**, because ``raise_()`` puts a window on top of its siblings,
        and only the topmost is activated so focus lands on the data.

        ⚠ Call it *after* the reader has closed. Closing a window hands
        activation to whatever the window manager picks next -- on Windows often
        the image viewer, which is why the render view appeared and then
        vanished behind it without anything in this application asking for that.
        """
    tiers = {
        "image": list(self._tiff_windows.values()),
        "mbm": self._mbm_windows(),
        "data": self._dataset_windows(dataset_indices),
    }
    topmost = None
    for tier in self.IMPORT_STACK_ORDER:
        for win in tiers.get(tier, ()):
            try:
                if not win.isVisible():
                    continue
                win.raise_()
            except RuntimeError:
                continue
            topmost = win
    if topmost is not None:
        try:
            topmost.activateWindow()
        except RuntimeError:
            pass


def _dataset_idx_arg(idx) -> int | None:
    """
        Return the only two argument shapes the private openers accept.

        ``bool`` maps to ``None`` deliberately: a plugin may well connect a
        public opener straight to a ``QAction.triggered``, which supplies
        ``checked=False``, and index 0 is the wrong reading of that.

        Anything integer-*like* is accepted and narrowed — ``numpy`` integers
        above all, because a script index is very often ``np.argmax(...)`` or a
        value pulled out of an array. A plain ``type(idx) is int`` test is
        right for the private methods, which only ever see Qt's own arguments,
        but as the public contract it would silently retarget such a call to
        the active dataset. Anything else raises rather than guessing: a
        silent fallback that masks a caller error is exactly what this project
        does not do.
        """
    if idx is None or isinstance(idx, bool):
        return None
    try:
        return int(idx.__index__())
    except AttributeError:
        raise TypeError(
            f"dataset index must be an integer or None, not "
                f"{type(idx).__name__}."
        ) from None


def show_render(self: MainWindow, idx: int | None = None):
    """Open (or raise) the Render View for a dataset (default: active)."""
    return self._show_render(self._dataset_idx_arg(idx))


def show_scatter(self: MainWindow, idx: int | None = None):
    """Open (or raise) the Localization Scatter Plot for a dataset."""
    return self._show_scatter(self._dataset_idx_arg(idx))


def show_histogram(self: MainWindow, idx: int | None = None):
    """Open (or raise) the Attribute Histogram for a dataset."""
    return self._show_histogram(self._dataset_idx_arg(idx))


def show_attribute_plot(self: MainWindow, idx: int | None = None):
    """Open (or raise) the Attribute Plot for a dataset."""
    return self._show_attr_plot(self._dataset_idx_arg(idx))


def show_console(self: MainWindow) -> None:
    """Open (or raise) the Console window."""
    return self._show_console()


def show_script_editor(self):
    """Open (or raise) the Script Editor."""
    return self._show_script_editor()


def show_macro_recorder(self):
    """Open (or raise) the modeless Macro Recorder plugin window."""
    from minflux_viewer.ui.common.modeless import show_modeless
    from minflux_viewer.ui.tools.macro_recorder_window import MacroRecorderWindow
    window = self._macro_recorder_win
    if window is not None:
        try:
            window.show()
            window.raise_()
            window.activateWindow()
            return window
        except RuntimeError:
            self._macro_recorder_win = None
    window = MacroRecorderWindow(self._state, owner=self)
    window.destroyed.connect(self._on_macro_recorder_destroyed)
    self._macro_recorder_win = window
    return show_modeless(window, self)


def show_hlyb_cell_collection(self, defaults=None):
    """Open or raise the persistent HlyB/D multi-dataset ROI pool."""
    from minflux_viewer.ui.analysis.hlyb_collection_dialog import HlyBCollectionWindow
    from minflux_viewer.ui.common.modeless import show_modeless
    if defaults is not None:
        self._state._hlyb_staged_cfg = defaults
    window = getattr(self, "_hlyb_collection_win", None)
    if window is not None:
        try:
            window.show()
            window.raise_()
            window.activateWindow()
            return window
        except RuntimeError:
            self._hlyb_collection_win = None
    window = HlyBCollectionWindow(self._state, owner=self)
    window.destroyed.connect(
        lambda *_args: setattr(self, "_hlyb_collection_win", None))
    self._hlyb_collection_win = window
    return show_modeless(window, self)


def _on_macro_recorder_destroyed(self: MainWindow, *_args) -> None:
    self._macro_recorder_win = None


def _bind_scripting_api(self: MainWindow) -> None:
    """Expose this viewer instance through the runtime ``mfv`` module."""
    try:
        from minflux_viewer.scripting import install_runtime_module
        self._state.mfv.bind_main_window(self)
        install_runtime_module(self._state.mfv)
    except Exception as exc:
        self._state.log(f"Could not initialize scripting API: {exc}", "WARN")


def _show_script_editor(self):
    from minflux_viewer.ui.tools.script_editor_window import ScriptEditorWindow
    if self._script_editor_win is None:
        self._script_editor_win = ScriptEditorWindow(self._state, parent=self)
    self._script_editor_win.show()
    self._script_editor_win.raise_()
    self._script_editor_win.activateWindow()
    return self._script_editor_win


def _show_scatter(self: MainWindow, dataset_idx: int | None = None):
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.plots.scatter_window import ScatterWindow
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return None
    win = self._scatter_windows.get(idx)
    fresh = win is None
    if fresh:
        win = ScatterWindow(self._state, dataset_idx=idx)
        self._unregister_when_destroyed(self._scatter_windows, win)
        self._scatter_windows[idx] = win
    self._scatter_win = win
    self._install_window_shortcuts(win)
    win.show()
    if fresh:
        self._place_new_window(win, idx)
    win.raise_(); win.activateWindow()
    self._notify_view_state_changed()
    return win


def _show_tracking_view(self: MainWindow, dataset_idx: int | None = None):
    """Open (or raise) this dataset's tracking view."""
    if self._state.active_dataset is None:
        self._no_data_warning()
        return None
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return None
    win = self._tracking_windows.get(idx)
    fresh = win is None
    if fresh:
        from minflux_viewer.ui.plots.tracking_window import TrackingWindow
        win = TrackingWindow(self._state, dataset_idx=idx)
        self._unregister_when_destroyed(self._tracking_windows, win)
        self._tracking_windows[idx] = win
    self._install_window_shortcuts(win)
    win.show()
    if fresh:
        self._place_new_window(win, idx)
    win.raise_(); win.activateWindow()
    self._notify_view_state_changed()
    return win


def _show_msd_analysis(self: MainWindow, dataset_idx: int | None = None):
    """Open (or raise) the active dataset's modeless MSD workbench."""
    if self._state.active_dataset is None:
        self._no_data_warning()
        return None
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return None
    win = self._msd_windows.get(idx)
    if win is None:
        from minflux_viewer.ui.common.modeless import show_modeless
        from minflux_viewer.ui.plots.tracking_analysis_window import MsdAnalysisWindow
        win = MsdAnalysisWindow(self._state, idx, owner=self)
        self._unregister_when_destroyed(self._msd_windows, win)
        self._msd_windows[idx] = win
        show_modeless(win, self)
    else:
        win.show()
    win.raise_()
    win.activateWindow()
    return win


def _show_histogram(self: MainWindow, dataset_idx: int | None = None):
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return None
    win = self._histogram_windows.get(idx)
    fresh = win is None
    if fresh:
        win = HistogramWindow(self._state, dataset_idx=idx)
        self._unregister_when_destroyed(self._histogram_windows, win)
        self._histogram_windows[idx] = win
    self._histogram_win = win
    self._install_window_shortcuts(win)
    win.show()
    if fresh:
        self._place_new_window(win, idx)
    win.raise_(); win.activateWindow()
    self._notify_view_state_changed()
    return win


def _show_attr_plot(self: MainWindow, dataset_idx: int | None = None):
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.plots.attribute_window import AttributeWindow
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return None
    win = self._attr_windows.get(idx)
    fresh = win is None
    if fresh:
        win = AttributeWindow(self._state, dataset_idx=idx)
        self._unregister_when_destroyed(self._attr_windows, win)
        self._attr_windows[idx] = win
    self._attr_win = win
    self._install_window_shortcuts(win)
    win.show()
    if fresh:
        self._place_new_window(win, idx)
    win.raise_(); win.activateWindow()
    self._notify_view_state_changed()
    return win


def _show_filter(self: MainWindow) -> None:
    from minflux_viewer.ui.common.modeless import show_modeless
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog
    idx = self._state.active_idx
    win = self._filter_dlgs.get(idx)
    if win is not None:
        try:
            if not win.isHidden():
                win.show()
                win.raise_()
                win.activateWindow()
                self._filter_dlg = win
                self._notify_view_state_changed()
                return win
        except RuntimeError:
            self._filter_dlgs.pop(idx, None)
            win = None
    # Filter dialogs are top-level modeless windows, like the normal
    # viewer windows.  Giving a top-level Qt.Window the main window as a
    # QWidget parent makes Windows keep it above that owner, so it can
    # obscure the main window whenever the two overlap.
    win = FilterDialog(self._state, parent=None, dataset_idx=idx)
    self._unregister_when_destroyed(self._filter_dlgs, win)
    self._filter_dlgs[idx] = win
    self._filter_dlg = win
    show_modeless(win, self)
    self._notify_view_state_changed()
    return win


def _load_filter_json(self: MainWindow, path: str) -> None:
    """Open (or raise) the filter dialog and append filters from a JSON file."""
    self._show_filter()
    self._filter_dlg.load_filter_json(path)


def _show_dataset_manager(self: MainWindow) -> None:
    manager = self._get_dataset_manager()
    manager.show()
    manager.raise_()
    manager.activateWindow()
    self._notify_view_state_changed()


def _get_dataset_manager(self):
    """Return the existing Dataset Manager, creating it if necessary."""
    from minflux_viewer.ui.tools.dataset_manager import DatasetManager
    if self._ds_manager is None:
        self._ds_manager = DatasetManager(self._state, parent=self)
        self._ds_manager.destroyed.connect(
            lambda _=None: setattr(self, "_ds_manager", None)
        )
    return self._ds_manager


def _ensure_dataset_manager_visible(self: MainWindow) -> None:
    """Create/show the Dataset Manager without changing window focus."""
    manager = self._get_dataset_manager()
    try:
        if not manager.isVisible():
            manager.show()
    except RuntimeError:
        # The QObject may have been deleted but the destroyed signal has
        # not run yet; retry through the normal creation path.
        self._ds_manager = None
        manager = self._get_dataset_manager()
        manager.show()
    self._notify_view_state_changed()


def _show_roi_manager(self: MainWindow) -> None:
    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow
    self._roi_manager_win = _raise_or_create(self._roi_manager_win, RoiManagerWindow, self._state)






def _show_particle_average(self: MainWindow) -> None:
    """Analyze › Segmentation › Particle Average… — open the particle-averaging
        tool. No active dataset is required: particles can be collected from the
        active dataset + ROI Manager box ROIs, or loaded from saved particle sets /
        data+ROI pairs."""
    from minflux_viewer.ui.analysis.particle_average_dialog import ParticleAverageWindow
    from minflux_viewer.ui.common.modeless import show_modeless
    win = ParticleAverageWindow(self._state, self._state.active_idx, owner=self)
    show_modeless(win, self)


def _notify_view_state_changed(self: MainWindow) -> None:
    mgr = getattr(self, "_ds_manager", None)
    if mgr is not None and hasattr(mgr, "refresh_views"):
        try:
            mgr.refresh_views()
        except RuntimeError:
            self._ds_manager = None


def dataset_view_status(self: MainWindow, idx: int) -> str:
    if not (0 <= idx < len(self._state.datasets)):
        return "None"
    ds = self._state.datasets[idx]
    group_id = ds.state.get("overlay_id") or ds.state.get("render_group_id")
    from minflux_viewer.core.overlay import is_multichannel_overlay
    is_overlay = bool(group_id and is_multichannel_overlay(self._state, idx))
    if is_overlay:
        overlay_idx = ds.state.get("overlay_index")
        visible_overlay = False
        for win in (
            list(self._render_windows.values())
            + list(self._scatter_windows.values())
        ):
            if window_draws_dataset(win, self._state, idx, visible_only=True):
                visible_overlay = True
                break
        if visible_overlay or is_overlay:
            return f"Overlay {overlay_idx}" if overlay_idx else "Overlay"
    own_maps = tuple(
        getattr(self, name, {})
        for name in (
            "_render_windows",
            "_scatter_windows",
            "_histogram_windows",
            "_attr_windows",
            "_filter_dlgs",
        )
    )
    for mapping in own_maps:
        win = mapping.get(idx)
        if win is not None:
            try:
                if win.isVisible():
                    return "Own"
            except RuntimeError:
                pass
    return "None"


def _show_log(self: MainWindow) -> None:
    """Open (or raise) the Log window — structured application events."""
    self._ensure_log_window(show=True, raise_window=True)


def _ensure_log_window(self: MainWindow, *, show: bool = True, raise_window: bool = False):
    from minflux_viewer.ui.tools.log_window import LogWindow
    if self._log_win is None or not self._log_win.isVisible():
        if self._log_win is None:
            self._log_win = LogWindow(self._state)
        if show:
            self._log_win.show()
    elif show and not self._log_win.isVisible():
        self._log_win.show()
    if raise_window and self._log_win is not None:
        self._log_win.raise_()
        self._log_win.activateWindow()
    return self._log_win


def _on_log_message(self: MainWindow, message: str, level: str) -> None:
    """Auto-open the Log window the first time anything is logged — but placed
        so it does NOT cover the main window's menu, and without stealing focus.

        Only fires once (subsequent messages append via the window's own signal
        connection). If the user has closed the window, it is not forced back."""
    if self._log_win is not None:
        return
    from minflux_viewer.ui.common.modeless import place_beside
    from minflux_viewer.ui.tools.log_window import LogWindow
    # LogWindow.__init__ replays state.log_history, so the just-logged message
    # (already in history before the signal fired) appears without a manual add.
    self._log_win = LogWindow(self._state)
    # Dock below the main window (falling back to right/above/left, then
    # clamped) so the menu bar at the top stays uncovered on any screen size.
    place_beside(self._log_win, self, prefer=("below", "right", "above", "left"))
    self._log_win.show()
    # Keep the main window in front/active so its menus remain accessible; the
    # log sits beside/below it, visible but not on top of the menu.
    self.raise_()
    self.activateWindow()


def _show_console(self: MainWindow) -> None:
    """Open (or raise) the Console window — raw stdout / stderr stream."""
    from minflux_viewer.ui.tools.console_window import ConsoleWindow
    if self._console_win is None or not self._console_win.isVisible():
        if self._console_win is None:
            self._console_win = ConsoleWindow()
        self._console_win.show()
    self._console_win.raise_()
    self._console_win.activateWindow()


def _render_window_for_dataset(self: MainWindow, idx: int | None):
    """The open render window showing dataset *idx* — either keyed by it, or
        (in an **overlay**) one whose channels include it. None if none is open.

        An overlay render window is stored in ``_render_windows`` under only its
        anchor/primary dataset index, but displays every channel. Keying strictly
        by the active index therefore misses the window when a non-primary channel
        is active — the cause of a duplicate overlay view being spawned."""
    if idx is None:
        return None
    win = self._render_windows.get(idx)
    if win is not None:
        return win
    for win in self._render_windows.values():
        if window_draws_dataset(win, self._state, idx):
            return win
    return None


def _show_brightness_contrast(self: MainWindow) -> None:
    """
        Forward to the render window of the currently active dataset.
        The active dataset follows whichever render window is focused (#2).
        """
    # The Shift+C shortcut is application-wide. When a window that owns its own
    # brightness/contrast is focused (a render view, the standalone TIFF/image
    # viewer), route to it rather than the active dataset's render — so the key
    # means "adjust what I'm looking at". This is why those windows no longer
    # need their own Shift+C QShortcut (which would be an ambiguous overload).
    active = QApplication.activeWindow()
    own_bc = getattr(active, "_show_brightness_contrast", None) if (
        active is not None and active is not self) else None
    if active is not None and callable(own_bc):
        try:
            own_bc()
            active.raise_()
            return
        except Exception as exc:
            self._state.log(f"Brightness/Contrast failed for focused window: {exc}", "ERROR")
    if self._state.active_dataset is None:
        self._no_data_warning()
        return
    idx = self._state.active_idx
    # Find the window that already displays this dataset (incl. as an overlay
    # channel) before opening a new one — otherwise pressing Shift+C on a
    # non-primary overlay channel spawns a duplicate overlay view.
    active = QApplication.activeWindow()
    if (
        getattr(active, "TAG", None) == "render_window"
        and window_draws_dataset(active, self._state, idx)
    ):
        rwin = active
    else:
        rwin = self._render_window_for_dataset(idx)
    if rwin is None:
        self._show_render(idx)
        rwin = self._render_window_for_dataset(idx)
    if rwin is not None:
        rwin._show_brightness_contrast()
        rwin.raise_()


def _focus_main_window(self: MainWindow) -> None:
    """Restore, show, and raise the main data-viewer window."""
    if self.isMinimized():
        self.showNormal()
    else:
        self.show()
    self.raise_()
    self.activateWindow()


def _viewer_windows(self: MainWindow) -> list[QWidget]:
    app = QApplication.instance()
    if app is None:
        return []
    windows = []
    for win in app.topLevelWidgets():
        if win.isVisible() and win.windowTitle():
            windows.append(win)
    return windows


def _cycle_windows(self: MainWindow, direction: int) -> None:
    windows = self._viewer_windows()
    if not windows:
        return
    active = QApplication.activeWindow()
    try:
        idx = windows.index(active)
    except ValueError:
        idx = self._window_cycle_index if 0 <= self._window_cycle_index < len(windows) else 0
    next_idx = (idx + direction) % len(windows)
    self._window_cycle_index = next_idx
    win = windows[next_idx]
    win.show()
    win.raise_()
    win.activateWindow()


def _cycle_dataset(self: MainWindow, direction: int) -> None:
    if not self._state.datasets:
        self._show_dataset_manager()
        return
    current = self._state.active_idx if self._state.active_idx is not None else 0
    idx = (current + direction) % len(self._state.datasets)
    self._state.set_active(idx)
    self._raise_dataset_window(idx)


def _raise_dataset_window(self: MainWindow, idx: int) -> None:
    win = self._render_windows.get(idx) or self._data_windows.get(idx)
    if win is None or win.isHidden():
        self._show_dataset_manager()
        return
    win.show()
    win.raise_()
    win.activateWindow()


def _close_current_child_window(self: MainWindow) -> None:
    active = QApplication.activeWindow()
    if active is not None and active is not self:
        active.close()
        return
    # Fallback: use the top-level ancestor of the current focus widget.
    # QApplication.activeWindow() can lag on Windows when a window is shown
    # programmatically without an explicit user click.
    focus = QApplication.focusWidget()
    if focus is not None:
        top = focus.window()
        if top is not None and top is not self:
            top.close()


def _show_set_measurements(self: MainWindow) -> None:
    from minflux_viewer.ui.analysis.set_measurements_dialog import SetMeasurementsDialog
    SetMeasurementsDialog(self._state, parent=self).exec()


def _show_task_monitor(self: MainWindow) -> None:
    """Open (or raise) the Task Monitor (Help → Task Monitor).

        Tasks, threads and memory are its three tabs; there is no separate
        *Monitor Memory* entry any more.
        """
    if getattr(self, "_task_monitor_win", None) is None:
        from minflux_viewer.ui.tools.task_monitor import TaskMonitor
        self._task_monitor_win = TaskMonitor(self._state, parent=self)
    self._task_monitor_win.show()
    self._task_monitor_win.raise_()
    self._task_monitor_win.activateWindow()


def _close_active_dataset(self: MainWindow) -> None:
    # For multi-channel overlay views (render/scatter with multiple channels),
    # honour the active dataset that the user selected in the channel panel.
    # For standalone single-dataset windows, resolve active from the focused window.
    active_win = QApplication.activeWindow()
    is_multichannel = len(getattr(active_win, "_channels", None) or []) > 1
    if not is_multichannel:
        self._set_active_from_focused_dataset_window()
    idx = self._state.active_idx
    if idx is None:
        return
    self.close_datasets([idx])


def close_datasets(self: MainWindow, indices) -> None:
    """Close (remove) every dataset in *indices*, and its windows.

        Shared by ``Ctrl+W`` (the active dataset) and the Dataset Manager's
        *Close* button / multi-selection *Close all*.  Removal runs highest
        index first so the remaining indices stay valid while iterating.
        """
    from minflux_viewer.core.overlay import dataset_group_id
    wanted = sorted(
        {int(i) for i in indices if 0 <= int(i) < len(self._state.datasets)},
        reverse=True,
    )
    if not wanted:
        return
    touched_group = False
    for idx in wanted:
        group_id = dataset_group_id(self._state.datasets[idx])
        if group_id:
            touched_group = True
            remaining = [
                i for i, d in enumerate(self._state.datasets)
                if i != idx and dataset_group_id(d) == group_id
            ]
            # Last surviving member: strip overlay state so it renders standalone
            if len(remaining) == 1:
                survivor = self._state.datasets[remaining[0]]
                for key in (
                    "overlay_id", "render_group_id", "overlay_index",
                    "overlay_order", "overlay_lut", "overlay_transform",
                    "render_transform_2d",
                ):
                    survivor.state.pop(key, None)
        self._state.remove_dataset(idx)
    if self._state.active_idx is None:
        self.setWindowTitle(self.APP_NAME)
    # A view drops a closed member by itself; a member left alone has just
    # lost its overlay state, so its view has to be rebuilt as standalone.
    if touched_group:
        self._refresh_overlay_windows()


def duplicate_datasets(self: MainWindow, datasets) -> None:
    """Plain-duplicate every dataset in *datasets* (Dataset Manager
        multi-selection *Duplicate all*).

        Whole-dataset copies: the ROI-crop duplicate is a single-dataset command
        (``Shift+D``), since a crop is defined by one dataset's active ROI.
        Datasets are passed by identity, not index — each ``add_dataset`` grows
        the list, so an index captured before the loop would not survive it.
        """
    for ds in list(datasets):
        self._plain_duplicate(ds)


def combine_datasets_as_overlay(self: MainWindow, indices) -> None:
    """Open *Process › Channel… › Combine* restricted to *indices*."""
    self._show_channel_combine(list(indices))


def reset_dataset(self: MainWindow, idx: int) -> None:
    """Dataset Manager *Reset* — put the dataset back to its as-loaded state.

        Filters, ROI selection masks, Z scaling factor and the live view layer (LUT /
        transform) all revert; overlay *membership* is kept, because a per-dataset
        reset must not dissolve a group of channels.  See
        ``core/dataset_reset.py`` for the rule.
        """
    if not (0 <= idx < len(self._state.datasets)):
        return
    from minflux_viewer.core.dataset_reset import reset_dataset as _reset
    ds = self._state.datasets[idx]
    try:
        changes = _reset(ds)
    except Exception as exc:
        self._state.log(f"Reset '{ds.name}' failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Reset dataset", str(exc))
        return
    if not changes:
        self._state.log(f"Reset '{ds.name}': already in its as-loaded state.",
                        dataset_idx=idx)
        return
    self._state.log(f"Reset '{ds.name}': " + "; ".join(changes) + ".", dataset_idx=idx)
    # Filters / ROI / Z scaling factor each drive a different set of views.
    self._state.notify_filter_changed(idx)
    self._state.notify_calibration_changed(idx)
    self._state.notify_roi_selection_changed(idx)
    self._refresh_overlay_windows()


def view_dataset_mbm(self: MainWindow, idx: int) -> None:
    """Dataset Manager *View mbm info…* — the dataset's beam-monitoring beads.

        Opens the combined MBM window (drift traces + beads vs the data region).
        The beads travel with the dataset (``ds.mbm`` / ``metadata["mbm_points"]``
        from the MSR import or an ``.msr`` round trip), so no source file is
        re-parsed.
        """
    if not (0 <= idx < len(self._state.datasets)):
        return
    from minflux_viewer.ui.tools.mbm_info_window import open_mbm_info
    ds = self._state.datasets[idx]
    win, reason = open_mbm_info(self, ds)
    if win is None:
        self._state.log(f"View mbm info: {reason.splitlines()[0]}", "WARN")
        QMessageBox.information(self, "MBM info", reason)
        return
    self._state.log(
        f"MBM info for '{ds.name}': {len(win._beads)} bead(s).", dataset_idx=idx)


def view_dataset_image_series(self: MainWindow, idx: int) -> None:
    """Dataset Manager *View image series* — open this dataset's source ``.msr``
        images in the shared image viewer.

        One viewer per file (every series is in its Series dropdown); the series
        rendered *from this dataset* — matched on the OBF footer's ``source`` did
        — is the one selected on open.
        """
    if not (0 <= idx < len(self._state.datasets)):
        return
    ds = self._state.datasets[idx]
    # A sealed .zarr.zip holds its images inside the archive, so the record's
    # absolute_path does not exist on disk; materialize_image extracts the
    # member on demand. Without it the empty list fell through to reopening
    # the source .msr, which looked like it worked while that file was still
    # where it was imported from -- and showed nothing once it moved.
    from minflux_viewer.core.minflux_zarr import materialize_image
    embedded: list[Path] = []
    labels: list[str] = []
    did = str(ds.metadata.get("msr_dataset_did") or "")
    selected = 0
    for item in (ds.metadata.get("minflux_viewer_images") or []):
        if not isinstance(item, dict):
            continue
        resolved = materialize_image(item)
        if resolved is None:
            continue
        # Prefer the series rendered from THIS dataset, exactly as the .msr
        # path does; it becomes the entry shown when the viewer opens.
        if did and str(item.get("source_did") or "") == did:
            selected = len(embedded)
        embedded.append(resolved)
        labels.append(str(item.get("name") or resolved.stem))
    if embedded:
        from minflux_viewer.core.tiff_source import EmbeddedImageSource
        # ONE viewer listing every embedded image in its Series dropdown --
        # the same shape as the .msr path. Opening a bare TiffImageSource on
        # embedded[0] showed only the first image, and named it by the temp
        # file it had been extracted to.
        try:
            source = EmbeddedImageSource(embedded, labels, series_index=selected)
            store = str(ds.metadata.get("minflux_viewer_zarr_path")
                        or ds.metadata.get("minflux_viewer_project_path")
                        or embedded[0].parent)
            self._open_image_viewer(
                source, f"{store}#embedded-images",
                initial_series_index=selected,
            )
        except Exception as exc:
            self._state.log(f"View embedded image failed for '{ds.name}': {exc}", "ERROR")
            QMessageBox.critical(self, "View image series", str(exc))
            return
        self._state.log(
            f"Embedded images of '{ds.name}': {len(embedded)} image(s), "
                f"showing '{labels[selected]}'.", dataset_idx=idx,
        )
        return
    # A dataset restored from a .zarr store carries its own images as
    # OME-TIFFs inside that store. It must NOT reach for the source .msr:
    # the store is meant to stand alone, and the .msr may have moved, been
    # renamed, or never travelled with it. Reporting the empty store is
    # honest; borrowing images from a neighbouring file is not.
    store_path = str(ds.metadata.get("minflux_viewer_zarr_path")
                     or ds.metadata.get("minflux_viewer_project_path") or "")
    if store_path:
        missing = len(ds.metadata.get("minflux_viewer_images") or [])
        detail = (
            f"Its {missing} image record(s) could not be read from the store."
            if missing else "It contains no image series."
        )
        self._state.log(
            f"View image series: '{ds.name}' came from {Path(store_path).name}; "
                f"{detail.lower()}", "WARN", dataset_idx=idx)
        QMessageBox.information(
            self, "View image series",
            f"'{ds.name}' was opened from\n{store_path}\n\n{detail}\n\n"
                "A MINFLUX Viewer Zarr store carries its own images, so no "
                "source .msr is consulted.",
        )
        return
    source_path = Path(str(ds.metadata.get("msr_source_path", "") or ""))
    if not source_path.is_file():
        self._state.log(
            f"View image series: no source .msr for '{ds.name}'.", "WARN")
        QMessageBox.information(
            self, "View image series",
            f"'{ds.name}' has no available source .msr file, so it has no "
                "image series.\n\nImage series come from the .msr the dataset was "
                "imported from.",
        )
        return
    from minflux_viewer.core.obf_image_source import ObfImageSource, list_obf_image_series
    try:
        series = list_obf_image_series(source_path)
    except Exception as exc:
        self._state.log(f"View image series failed for '{ds.name}': {exc}", "ERROR")
        QMessageBox.critical(self, "View image series", str(exc))
        return
    if not series:
        self._state.log(
            f"View image series: '{source_path.name}' has no image series.", "WARN")
        QMessageBox.information(
            self, "View image series",
            f"'{source_path.name}' contains no viewable image series.",
        )
        return
    did = str(ds.metadata.get("msr_dataset_did", "") or "")
    wanted = next(
        (s for s in series if did and str(s.get("source_did") or "") == did),
        series[0],
    )
    try:
        src = ObfImageSource(source_path, raw_stack_index=int(wanted["raw_index"]))
    except Exception as exc:
        self._state.log(f"View image series failed for '{ds.name}': {exc}", "ERROR")
        QMessageBox.critical(self, "View image series", str(exc))
        return
    self._open_image_viewer(
        src,
        f"{source_path.resolve()}#obf-images",
        initial_series_index=int(src.metadata.series_index),
    )
    self._state.log(
        f"Image series of '{source_path.name}': {len(series)} series, showing "
            f"'{wanted.get('name')}'.", dataset_idx=idx)


def _refresh_overlay_windows(self: MainWindow) -> None:
    """Rebuild channel lists and re-render open render/scatter overlay windows."""
    for win in list(self._render_windows.values()):
        try:
            win._build_channels()
            win._rebuild_channel_ui()
            win._rebuild_all_grids()
            win._schedule_render()
        except Exception:
            pass
    for win in list(self._scatter_windows.values()):
        try:
            win._refresh()
        except Exception:
            pass


def _close_all_datasets(self: MainWindow) -> None:
    n = len(self._state.datasets)
    if n == 0:
        return
    for idx in range(n - 1, -1, -1):
        self._state.remove_dataset(idx)
    self.setWindowTitle(self.APP_NAME)


def _close_all_windows(self: MainWindow) -> None:
    """``Ctrl+Shift+W`` — close **everything**: all datasets, their viewer
        windows, and every plugin / analysis / tool dialog (particle average, MSR
        reader, ROI manager, dataset manager, …). Only the **Log** and **Console**
        windows are kept."""
    self._close_all_datasets()
    self._close_all_child_windows(keep_log_console=True)
    self.setWindowTitle(self.APP_NAME)
    self._status_label.setText("Closed all windows (kept Log / Console).")


def _show_render(self: MainWindow, dataset_idx: int | None = None):
    """
        Open (or raise) the render window for a dataset.
        Defaults to the active dataset.
        """
    if self._state.active_dataset is None:
        self._no_data_warning()
        return
    idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
    if idx is None:
        return
    # An overlay render window is keyed only by its anchor index but displays
    # every channel; reuse it (raise, don't spawn a duplicate) when idx is one
    # of its non-anchor channels — e.g. segmentation adding ROIs on channel 2.
    win = self._render_windows.get(idx) or self._render_window_for_dataset(idx)
    if win is not None:
        self._install_window_shortcuts(win)
        # _refresh_from_dataset CANCELS an open manual alignment, so merely
        # raising the window (Ctrl+R) would silently discard it.
        if getattr(win, "_overlay_alignment_panel", None) is None:
            try:
                win._refresh_from_dataset()
            except Exception:
                pass
        win.show()
        win.raise_()
        win.activateWindow()
        self._notify_view_state_changed()
        return win
    from minflux_viewer.ui.rendering.precision_render_window import PrecisionRenderWindow
    win = PrecisionRenderWindow(self._state, dataset_idx=idx)
    self._install_window_shortcuts(win)
    self._unregister_when_destroyed(self._render_windows, win)
    self._render_windows[idx] = win
    win.show()
    self._place_new_window(win, idx)
    self._notify_view_state_changed()
    return win


def _on_dataset_removed(self: MainWindow, event) -> None:
    """Retire what belonged to a dataset that has closed (typed removal).

        Its background work and its windows are found by key. Every other
        dataset's windows stay registered under their own key and draw their
        datasets by key, so nothing is re-keyed and nothing reloads: an overlay
        view drops a closed member itself (on the same event).
        """
    self._retire_dataset_tasks(event)
    for mapping in (
        self._data_windows,
        self._render_windows,
        self._scatter_windows,
        self._tracking_windows,
        self._msd_windows,
        self._histogram_windows,
        self._attr_windows,
        self._filter_dlgs,
    ):
        win = mapping.pop_key(event.key)
        if win is not None:
            try:
                delete_on_close = win.testAttribute(
                    Qt.WidgetAttribute.WA_DeleteOnClose
                )
                win.close()
                if not delete_on_close:
                    win.deleteLater()
            except Exception:
                pass
    # ROI scopes are held by key, so the other datasets' ROIs need nothing;
    # this dataset's ROIs stay listed, now resolving to "closed dataset".
    from minflux_viewer.core.roi_scope import references_dataset
    if any(references_dataset(r, event.key) for r in self._state.rois.records):
        self._state.rois.changed.emit()
    self._roi_last_active.pop_key(event.key)
    self._notify_view_state_changed()


def _unregister_when_destroyed(
        self: MainWindow, registry: dict | DatasetWindowMap, window) -> None:
    """Remove *window* from *registry* when Qt destroys it -- by identity.

        The index a window was opened at goes stale as soon as an earlier
        dataset closes, so removing by that index took out whichever window had
        moved into it. The registry is also what keeps a parentless window's Python
        wrapper alive: a window dropped from it while still on screen was
        destroyed by the next garbage collection, without a close event, and
        its pyqtgraph views took the process down with it.
        """
    from PyQt6 import sip
    ref = weakref.ref(window)
    def deleted(value) -> bool:
        try:
            return sip.isdeleted(value)
        except TypeError:               # not a Qt object (a test double)
            return False
    def forget(_obj=None) -> None:
        target = ref()
        def gone(value) -> bool:
            return value is target or deleted(value)
        discard = getattr(registry, "discard_where", None)
        if callable(discard):           # a DatasetWindowMap: every entry
            discard(gone)
            return
        for key, value in list(registry.items()):
            if gone(value):
                registry.pop(key, None)
    window.destroyed.connect(forget)


def _on_active_changed(self: MainWindow, event) -> None:
    idx = event_index(self._state, event)
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        return
    ds = self._state.datasets[idx]
    self.setWindowTitle(f"{self.APP_NAME}  —  {ds.name}")
    # Report the active dataset to the Log (skip consecutive duplicates from
    # window-focus re-emits). Compared by identity, not index: closing an
    # earlier dataset shifts the active one's index without changing it.
    # A weak reference, so a closed dataset is not kept alive by the log.
    last = getattr(self, "_last_active_log_ds", None)
    if last is None or last() is not ds:
        self._last_active_log_ds = weakref.ref(ds)
        self._state.log(f"Active dataset: '{ds.name}'")


def _record_file_drop(self: MainWindow, paths) -> None:
    """Record the OS drop attempt before parsing/routing can fail.

        A literal drop is interactive rather than a semantic data load. The
        generated call replays the same router and is therefore GUI_ONLY; a
        successful loader separately records the normalized load outcome.
        """
    recorder = getattr(self._state, "recorder", None)
    if recorder is None or not recorder.enabled:
        return
    normalized = [str(path) for path in paths if str(path or "").strip()]
    if not normalized:
        return
    noun = "path" if len(normalized) == 1 else "paths"
    recorder.append(
        f"Dropped {len(normalized)} local {noun} on the main window",
        code=f"mfv.ui.drop_files({normalized!r})",
        gui_class="gui_only",
        command="fileDrop",
        category="other",
        paths=normalized,
        origin="os_drag_drop",
    )


def route_paths(self: MainWindow, paths) -> None:
    """Open a whole batch of paths — a multi-file drop, or the command line.

        The batch is classified **before** anything is routed, so a data file
        dropped together with its ``_viewer_metadata.json`` applies the recipe
        asking: dropping both files is the answer to the question point 1 would
        otherwise pose, and it must not be asked whichever of the two the OS
        happens to list first. Routing them one at a time could not know a
        sidecar was coming.

        Data files are routed **first** for the same reason from the other
        side: a recipe named in the batch then finds its dataset already loaded
        instead of going through the pending-recipe path.
        """
    from minflux_viewer.core.save import is_metadata_json_file
    paths = [str(path) for path in (paths or []) if str(path or "").strip()]
    sidecars: dict = {}
    for path in paths:
        try:
            if Path(path).is_file() and is_metadata_json_file(path):
                sidecars[str(Path(path).resolve()).casefold()] = path
        except Exception:                                   # noqa: BLE001
            continue
    previous, previous_used = self._batch_sidecars, self._batch_sidecars_used
    self._batch_sidecars, self._batch_sidecars_used = sidecars, set()
    try:
        data_first = [path for path in paths
                      if str(Path(path).resolve()).casefold() not in sidecars]
        for path in data_first:
            self._route_path(path)
        for key, path in sidecars.items():
            # A recipe already consumed by its data file in this same batch
            # is finished: routing it again would re-apply it and ask the
            # very question dropping both files answered.
            if key in self._batch_sidecars_used:
                continue
            self._route_path(path)
    finally:
        self._batch_sidecars = previous
        self._batch_sidecars_used = previous_used


def _on_focus_changed(self: MainWindow, _old, new) -> None:
    """Record the last render/scatter window the user focused (for the LUT
        button). Guarded — focus can shift to a window being torn down."""
    if new is None:
        return
    try:
        win = new.window()
        if win is not None and win is not self and hasattr(win, "open_lut_dialog"):
            self._last_active_plot_window = win
            # Focusing a plot window refreshes its (already-open) LUT dialog,
            # so the dialog reflects whichever view the user is working in.
            sync = getattr(win, "sync_lut_dialog", None)
            if callable(sync):
                sync()
            self._retarget_lut_dialog(win)
    except (RuntimeError, AttributeError):
        pass


def _lut_view_shows_dataset(view, idx: int | None) -> bool:
    """True when *view* displays dataset *idx* (directly or as an overlay
        channel), so the LUT button targets the view of the active dataset."""
    if view is None or idx is None:
        return view is not None
    try:
        if window_dataset_index(view) == idx:
            return True
        return window_draws_dataset(view, view._state, idx)
    except (RuntimeError, TypeError, AttributeError):
        return False


def _retarget_lut_dialog(self: MainWindow, view) -> None:
    """Move the one open LUT dialog onto the newly focused *view*.

        The dialog is per-view, so 'following focus' means closing the other
        view's and opening this one's; the geometry is carried over so it stays
        put instead of jumping.  Does nothing when no LUT dialog is open — this
        must never *summon* one just because the user clicked a plot.
        """
    own = getattr(view, "_lut_dialog", None)
    try:
        if own is not None and own.isVisible():
            return                      # already showing this view's
    except RuntimeError:
        own = None
    dialog = getattr(self._state, "_shared_lut_dialog", None)
    try:
        if dialog is None or not dialog.isVisible():
            return                      # nothing open; never summon one
        geometry = dialog.geometry()
    except RuntimeError:
        return
    # Rebinding keeps the same window, so it neither blinks nor moves; the
    # geometry is restored anyway in case the view re-lays it out.
    try:
        view.open_lut_dialog()
    except Exception as exc:
        self._state.log(f"LUT retarget failed: {exc}", "ERROR")
        return
    try:
        dialog.setGeometry(geometry)
    except RuntimeError:
        pass


def _lut_capable_views(self: MainWindow) -> list:
    """Every view that can own a LUT dialog."""
    views: list = []
    registries: list = [
        self._render_windows,
        self._scatter_windows,
        self._attr_windows,
    ]
    # Present only when the advanced renderer has been opened.
    advanced = getattr(self, "_advanced_render_windows", None)
    if isinstance(advanced, dict):
        registries.insert(1, advanced)
    for registry in registries:
        for view in registry.values():
            views.append(view)
            volume = getattr(view, "_volume_window", None)
            if volume is not None:
                views.append(volume)
    return views


def _close_other_lut_dialogs(self: MainWindow, keep) -> None:
    """No-op kept for callers: there is only ever one LUT dialog now.

        ``ui/dialogs/lut_dialog.py::shared_lut_dialog`` hands every view the same
        instance and rebinds its callbacks, so nothing else can be on screen to
        close.  This used to hide the other views' dialogs, which could not
        actually guarantee the invariant — ``close()`` may be refused, and every
        view that had ever opened one left a hidden instance behind.
        """
    del keep


def _open_lut_on_view(self: MainWindow, view, idx: int | None) -> bool:
    """Open *view*'s LUT dialog if it is a live, visible LUT-capable view of
        dataset *idx*. Returns True once handled (or on failure, to stop trying)."""
    if view is None:
        return False
    try:
        opener = getattr(view, "open_lut_dialog", None)
    except RuntimeError:
        return False
    if not callable(opener):
        return False
    try:
        if not view.isVisible():
            return False
    except (RuntimeError, AttributeError):
        return False
    if not self._lut_view_shows_dataset(view, idx):
        return False
    try:
        # Open first, then retire the others: a view whose data is not ready
        # declines to show (scatter does), and closing first would leave the
        # user with no LUT dialog at all.
        opener()
        self._close_other_lut_dialogs(view)
        self._last_active_plot_window = view
        return True
    except Exception as exc:
        self._state.log(f"LUT failed for active view: {exc}", "ERROR")
        return True


def _image_viewer_for_lut(self: MainWindow, active) -> object | None:
    """The image viewer the LUT button should act on, or ``None``.

        The focused window wins; failing that the last plot window, if it is one;
        failing that a lone open image viewer, so the button works right after an
        image is opened and the main window has focus. With several open and none
        focused there is no defensible target, so the dataset path runs instead.
        """
    windows = [win for win in self._tiff_windows.values() if win is not None]
    for candidate in (active, getattr(self, "_last_active_plot_window", None)):
        if candidate is not None and any(candidate is win for win in windows):
            return candidate
    visible = []
    for win in windows:
        try:
            if win.isVisible():
                visible.append(win)
        except RuntimeError:
            continue
    if len(visible) == 1 and self._state.active_dataset is None:
        return visible[0]
    return None


def _show_lut(self: MainWindow) -> None:
    """Toolbar LUT button — open the LUT / colormap editor for the active
        view: render, scatter, Attribute Plot C dimension, or 3-D volume.

        Clicking the toolbar activates the main window, so ``activeWindow()`` no
        longer identifies the plot window; try the focused view and the last-used
        view, then scan the plot registries (and each render's 3-D volume
        window) for one showing the active dataset.
        """
    active = QApplication.activeWindow()
    # An image viewer is resolved first and WITHOUT a dataset: a TIFF /
    # OME-TIFF is not a dataset (images stay outside the dataset model), so
    # the dataset gate below would refuse the button for the very window the
    # user is looking at.
    image_view = self._image_viewer_for_lut(active)
    if image_view is not None:
        if image_view.open_lut_dialog():
            self._close_other_lut_dialogs(image_view)
            self._last_active_plot_window = image_view
        else:
            self._state.log(
                "LUT: this image plane is RGB, so it has no single channel "
                    "to map. Convert it to a single channel first.", "WARN")
            QMessageBox.information(
                self, "LUT",
                "This image plane is RGB — there is no single channel for a "
                    "colormap to map.")
        return
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    idx = self._state.active_idx
    candidates = [active, getattr(self, "_last_active_plot_window", None)]
    for registry in (
        self._render_windows,
        self._scatter_windows,
        self._attr_windows,
    ):
        candidates.extend(registry.values())
        for view in registry.values():
            volume = getattr(view, "_volume_window", None)
            if volume is not None:
                candidates.append(volume)
    seen: set[int] = set()
    for view in candidates:
        if id(view) in seen:
            continue
        seen.add(id(view))
        if self._open_lut_on_view(view, idx):
            return
    # No suitable view yet — open the active dataset's render and target it.
    rwin = self._render_window_for_dataset(idx)
    if rwin is None:
        self._show_render(idx)
        rwin = self._render_window_for_dataset(idx)
    if rwin is not None and self._open_lut_on_view(rwin, idx):
        return
    QMessageBox.information(
        self, "LUT",
        "Open a render, scatter, or four-dimensional Attribute Plot first, "
            "then use the LUT button.",
    )


def _show_color_picker(self: MainWindow) -> None:
    """Open or raise the one modeless application-wide COLOR editor."""
    from minflux_viewer.ui.common.modeless import show_modeless
    from minflux_viewer.ui.dialogs.global_color_dialog import GlobalColorDialog
    dialog = self._color_dialog
    if dialog is not None:
        try:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        except RuntimeError:
            self._color_dialog = None
    dialog = GlobalColorDialog(self._state)
    dialog.destroyed.connect(lambda _=None: setattr(self, "_color_dialog", None))
    self._color_dialog = dialog
    show_modeless(dialog, self)


def _show_info_for_active(self: MainWindow) -> None:
    """View → Show Info — open data-info window(s) for the active dataset."""
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    active_idx = self._state.active_idx
    if active_idx is None:
        return
    indices = self._info_indices_for_active(active_idx)
    active_win = None
    for offset, idx in enumerate(indices):
        win = self._data_windows.get(idx)
        if win is None:
            from minflux_viewer.ui.tools.data_window import DataWindow
            win = DataWindow(self._state.datasets[idx], idx, self._state)
            self._data_windows[idx] = win
        if offset:
            win.move(self.x() + 120 + offset * 28, self.y() + 160 + offset * 28)
        win.show()
        win.raise_()
        if idx == active_idx:
            active_win = win
    if active_win is not None:
        active_win.raise_()
        active_win.activateWindow()


def _info_indices_for_active(self: MainWindow, active_idx: int) -> list[int]:
    """Show all members of an imported multi-channel MSR group together."""
    try:
        active = self._state.datasets[active_idx]
    except Exception:
        return []
    group_id = active.state.get("overlay_id") or active.state.get("render_group_id")
    msr_source = active.metadata.get("msr_source_path")
    if not group_id and not msr_source:
        return [active_idx]
    indices: list[int] = []
    for idx, ds in enumerate(self._state.datasets):
        if group_id and (ds.state.get("overlay_id") or ds.state.get("render_group_id")) == group_id:
            indices.append(idx)
        elif msr_source and ds.metadata.get("msr_source_path") == msr_source:
            indices.append(idx)
    return indices or [active_idx]


def _loc_precision_frc(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.precision_analysis import run_frc
    run_frc(self, self._state)


def _loc_precision_crlb(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.precision_analysis import run_crlb
    run_crlb(self, self._state)


def _loc_precision_stddev(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.precision_analysis import run_stddev_per_trace
    run_stddev_per_trace(self, self._state)


def _run_local_density(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.density_analysis import run_local_density
    run_local_density(self, self._state)


def _run_trace_size(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.trace_analysis_dialogs import show_trace_size_dialog
    show_trace_size_dialog(self, self._state.active_dataset, self._state)


def _run_trace_anisotropy(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._no_data_warning(); return
    from minflux_viewer.ui.analysis.trace_analysis_dialogs import show_anisotropy_dialog
    show_anisotropy_dialog(self, self._state.active_dataset, self._state)


def _drain_background_tasks(self: MainWindow) -> None:
    """Detach background work without ever blocking the GUI close path."""
    from minflux_viewer.ui.common.background_tasks import (
        clear_pool_nonblocking,
        retire_background_tasks,
    )
    retire_background_tasks(
        getattr(self, "_update_tasks", set()), signal_names=("done",)
    )
    self._update_tasks.clear()
    # File I/O and filter restoration in flight must not deliver into a
    # closing window. Native reads/writes finish on their process-owned pool.
    retire_background_tasks(
        getattr(self, "_file_io_tasks", []),
        signal_names=(
            "stage", "done", "failed", "cancelled", "finished"
        ),
    )
    if hasattr(self, "_file_io_tasks"):
        self._file_io_tasks.clear()
    retire_background_tasks(
        getattr(self, "_post_load_tasks", {}).values(),
        signal_names=(
            "stage", "done", "failed", "cancelled", "finished"
        ),
    )
    if hasattr(self, "_post_load_tasks"):
        self._post_load_tasks.clear()
    try:
        pool = QThreadPool.globalInstance()
        clear_pool_nonblocking(pool)
    except Exception:
        pass
    # The named process-owned pools outlive this window, so work already
    # queued on them would still start after it closed. Retirement above
    # cancelled those tasks, but dropping what has not begun is cheaper
    # than starting each one only to have it exit at its first checkpoint.
    # Still non-blocking: clear() never touches work already running.
    try:
        from minflux_viewer.ui.common.background_tasks import shared_thread_pools
        for pool in shared_thread_pools():
            clear_pool_nonblocking(pool)
    except Exception:
        pass


def _close_all_child_windows(self: MainWindow, *, keep_log_console: bool = False) -> None:
    """
        Close every floating / tool window this main window has created.

        Safe to call repeatedly. Uses ``close()`` (not ``deleteLater``) so
        each child runs its own ``closeEvent`` (saves its settings, etc.)
        before being torn down by Qt on the main window's destruction.

        The shared LUT dialog is parentless (so it survives the view that
        opened it) and is therefore closed here explicitly — otherwise it
        outlives the main window.

        With ``keep_log_console=True`` the Log and Console windows are left open
        (used by the *Close All Windows* command, which clears everything else).
        """
    from minflux_viewer.ui.dialogs.lut_dialog import close_shared_lut_dialog
    close_shared_lut_dialog(self._state)
    # Per-dataset windows
    for win in list(self._data_windows.values()):
        try: win.close()
        except RuntimeError: pass
    for win in list(self._render_windows.values()):
        try: win.close()
        except RuntimeError: pass
    for win in list(self._tiff_windows.values()):
        try: win.close()
        except RuntimeError: pass
    for mapping in (
        self._scatter_windows,
        self._tracking_windows,
        self._msd_windows,
        self._histogram_windows,
        self._attr_windows,
    ):
        for win in list(mapping.values()):
            try: win.close()
            except RuntimeError: pass
    for win in list(self._filter_dlgs.values()):
        try: win.close()
        except RuntimeError: pass
    try:
        self._state.mfv.close_windows()
    except Exception:
        pass
    # Singleton tool windows
    singletons = [
        "_filter_dlg",  "_ds_manager",
        "_log_win",     "_console_win", "_task_monitor_win",
        "_roi_manager_win",
        "_script_editor_win",
    ]
    if keep_log_console:
        singletons = [a for a in singletons if a not in ("_log_win", "_console_win")]
    for attr in singletons:
        w = getattr(self, attr, None)
        if w is not None:
            try:
                if hasattr(w, "force_close"):
                    w.force_close()
                else:
                    w.close()
            except RuntimeError: pass
    # Plugin dialogs that stashed themselves on the main window. Several MSR
    # reader dialogs can be open at once — close every one in the registry
    # (plus the legacy single-attr alias, for safety).
    msr_dialogs = list(getattr(self, "_plugin_msr_reader_dialogs", None) or [])
    legacy = getattr(self, "_plugin_msr_reader_dialog", None)
    if legacy is not None and legacy not in msr_dialogs:
        msr_dialogs.append(legacy)
    for w in msr_dialogs:
        try: w.close()
        except RuntimeError: pass
    # Modeless analysis/plugin windows retained via ui/common/modeless.py
    try:
        from minflux_viewer.ui.common.modeless import close_modeless
        close_modeless(self)
    except RuntimeError:  # deleted C++ peers during shutdown
        pass


def _terminate_paraview_processes(self: MainWindow) -> None:
    """
        Politely stop every ParaView subprocess spawned during this session.

        ParaView is launched in *detached* mode so it keeps running after
        the viewer normally. When the user wants the viewer to act as the
        owner of its children, we clean those subprocesses up here.
        """
    import subprocess  # stdlib, no deps
    import threading
    if not self._paraview_procs:
        return
    still_alive = [p for p in self._paraview_procs if p.poll() is None]
    if not still_alive:
        return
    self._state.log(
        f"Terminating {len(still_alive)} ParaView process(es)…",
        "INFO",
    )
    # Graceful shutdown first. terminate() only posts the request, so this
    # part is instant and every process has been asked before we return.
    for p in still_alive:
        try: p.terminate()
        except Exception: pass
    # ⚠ The escalation must NOT run here. This is called from closeEvent,
    # and `p.wait(timeout=2)` per process made application close hang for up
    # to 2 s × N on ParaView instances that ignore the request — the same
    # class of freeze the Qt background-work rules exist to prevent, just
    # via a subprocess rather than a QThread. A daemon thread cannot hold
    # interpreter exit open, so the kill is best-effort by design; the
    # terminate above is the part that is guaranteed to have been delivered.
    def _escalate(procs) -> None:
        for proc in procs:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
    threading.Thread(
        target=_escalate, args=(list(still_alive),),
        name="paraview-terminate", daemon=True,
    ).start()
    self._paraview_procs.clear()
