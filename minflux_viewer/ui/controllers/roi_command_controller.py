"""Roi command controller delegated by MainWindow."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QTransform
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QMessageBox, QWidget

from minflux_viewer.core.coordinates import coordinate_view
from minflux_viewer.core.dataset_key import DatasetKey
from minflux_viewer.ui.common.dataset_binding import window_dataset_index, window_dataset_key
from minflux_viewer.ui.controllers.window_support import (
    _LASSO_FAMILY,
    _LINE_FAMILY,
    _OVAL_FAMILY,
    _POINT_FAMILY,
    _POLY_FAMILY,
    _RECT_FAMILY,
    _ROI_TOOL_DEFS,
    _adaptive_toolbar_icon,
    _adaptive_toolbar_pixmap,
    roi_tool_status,
)

if TYPE_CHECKING:
    from minflux_viewer.ui.main_window import MainWindow


def _active_roi(self):
    """The active ROI as ``(record, kind, adapter)``: the single selected
        stored ROI, else the active overlay's draft. ``kind`` is
        ``"stored"`` / ``"draft"`` / ``None``."""
    store = self._state.rois
    selected = store.selected_records()
    if len(selected) == 1:
        return selected[0], "stored", store.active_adapter
    adapter = store.active_adapter
    if adapter is not None:
        try:
            draft = adapter.current_record()
        except Exception:
            draft = None
        if draft is not None:
            return draft, "draft", adapter
    return None, None, None


def _clear_roi_cached_mask(self: MainWindow, record) -> None:
    """Drop a stored ROI's cached selection mask (state + derived) so a
        converted/edited ROI doesn't leave a stale highlight behind."""
    from minflux_viewer.core.roi_scope import roi_owner_dataset
    from minflux_viewer.core.roi_selection import ROI_MASKS_STATE_KEY
    idx = roi_owner_dataset(record, self._state)
    datasets = self._state.datasets
    ds = datasets[idx] if isinstance(idx, int) and 0 <= idx < len(datasets) else None
    if ds is None:
        ds = self._state.active_dataset
    if ds is None:
        return
    masks = ds.state.get(ROI_MASKS_STATE_KEY, {})
    meta = masks.pop(record.id, None)
    key = getattr(record, "mask_key", "") or (meta or {}).get("key")
    if key:
        ds.derived.pop(key, None)


def _commit_roi_replacement(self: MainWindow, record, kind, adapter, new_record) -> None:
    """Replace a stored ROI in place; a draft stays a draft of the new type
        in the view (not filed into the Manager)."""
    store = self._state.rois
    if kind == "stored":
        self._clear_roi_cached_mask(record)  # drop the old shape's stale highlight
        new_record.name = (
            f"{new_record.type}-{store.next_type_index(new_record.type)}"
            if store._is_auto_name(record) else record.name)
        store.update(record.id, new_record)
        store.select([record.id])
    else:  # draft — keep it editable in the view, do not open/file the Manager
        replace_draft = getattr(adapter, "replace_draft", None)
        if callable(replace_draft):
            replace_draft(new_record)
        else:
            if adapter is not None:
                try:
                    adapter.consume_draft()
                except Exception:
                    pass
            store.add(new_record)
    from minflux_viewer.core.roi_scope import roi_owner_dataset
    idx = roi_owner_dataset(record, self._state)
    self._state.notify_roi_selection_changed(
        idx if isinstance(idx, int) and idx >= 0 else None)


def _convert_active_roi(self: MainWindow, target: str) -> None:
    from minflux_viewer.core.roi_convert import available_conversions, convert_roi
    record, kind, adapter = self._active_roi()
    if record is None:
        QMessageBox.information(self, "Convert ROI", "Select an ROI (or draw one) first.")
        return
    if target not in available_conversions(record):
        QMessageBox.information(
            self, "Convert ROI",
            f"Cannot convert a {record.type} ROI ('{record.name}') to {target}.")
        return
    width = height = None
    from minflux_viewer.core.roi_convert import _SHAPE_TARGETS
    from minflux_viewer.ui.roi.roi_convert_dialog import RoiSizeDialog
    if record.type == "point" and target in _SHAPE_TARGETS:
        dlg = RoiSizeDialog(self, title=f"Point → {target}", need_height=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        width, height = dlg.values()
    elif target == "region":
        dlg = RoiSizeDialog(self, title="Line → region", need_height=False, width_label="Width")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        width, _ = dlg.values()
    try:
        new_record = convert_roi(record, target, width=width, height=height)
    except Exception as exc:
        QMessageBox.warning(self, "Convert ROI", str(exc))
        return
    old_type = record.type
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(f"Converted ROI '{record.name}' ({old_type}) → {new_record.type}.")


def _resize_active_roi(self: MainWindow) -> None:
    from minflux_viewer.core.roi_convert import LINE_TYPES, can_resize, enlarge_shrink_roi
    from minflux_viewer.ui.roi.roi_convert_dialog import RoiResizeDialog
    record, kind, adapter = self._active_roi()
    if record is None:
        QMessageBox.information(self, "Enlarge / Shrink ROI", "Select an ROI (or draw one) first.")
        return
    if not can_resize(record):
        QMessageBox.information(
            self, "Enlarge / Shrink ROI",
            f"A {record.type} ROI cannot be enlarged or shrunk.")
        return
    allow_shrink = record.type not in ({"point"} | LINE_TYPES)
    dlg = RoiResizeDialog(self, allow_shrink=allow_shrink)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    mode, value = dlg.values()
    try:
        new_record = enlarge_shrink_roi(record, value, mode=mode)
    except Exception as exc:
        QMessageBox.warning(self, "Enlarge / Shrink ROI", str(exc))
        return
    old_type = record.type
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(
        f"{mode.capitalize()}d ROI '{record.name}' ({old_type}) by {value:g} nm → {new_record.type}.")


def _skeletonize_active_roi(self: MainWindow) -> None:
    from minflux_viewer.core.roi_convert import REGION_TYPES, skeletonize_roi
    record, kind, adapter = self._active_roi()
    if record is None:
        QMessageBox.information(self, "Skeletonize ROI", "Select a region ROI (or draw one) first.")
        return
    if record.type not in REGION_TYPES:
        QMessageBox.information(
            self, "Skeletonize ROI",
            "Skeletonize applies to region ROIs (rectangle / oval / polygon / freehand).")
        return
    try:
        new_record = skeletonize_roi(record)
    except Exception as exc:
        QMessageBox.warning(self, "Skeletonize ROI", str(exc))
        return
    old_type = record.type
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(f"Skeletonized ROI '{record.name}' ({old_type}) → line.")


def _convex_hull_active_roi(self: MainWindow) -> None:
    """Process › ROI › Convex Hull — convert the active ROI to its convex
        hull when that conversion is available (polygon / freehand). Any other
        case (incompatible type, no active ROI, no active dataset) is ignored."""
    from minflux_viewer.core.roi_convert import available_conversions, convert_roi
    record, kind, adapter = self._active_roi()
    if record is None or "convex_hull" not in available_conversions(record):
        return  # silently ignore
    try:
        new_record = convert_roi(record, "convex_hull")
    except Exception:
        return
    old_type = record.type
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(f"Converted ROI '{record.name}' ({old_type}) → convex hull.")


def _roi_highlighted_points(self, record, ds):
    """(M, 2) localizations the region *record* highlights, in the ROI's plane
        (display nm, filter applied)."""
    import numpy as np

    from minflux_viewer.core.roi_crop import display_xyz_filtered
    from minflux_viewer.core.roi_selection import roi_region_mask
    plane = "XY"
    if isinstance(record.context, dict):
        plane = record.context.get("view_plane") or "XY"
    axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}.get(plane, (0, 1))
    xyz = display_xyz_filtered(ds)
    if xyz.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    x, y = xyz[:, axes[0]], xyz[:, axes[1]]
    mask = np.asarray(roi_region_mask(x, y, record), dtype=bool)
    return np.column_stack([x[mask], y[mask]])


def _fit_active_roi(self: MainWindow, target: str) -> None:
    """Process › ROI › Fit ▸ Rectangle/Circle/Ellipse/Polygon/Convex Hull —
        fit the shape to the localizations the active region ROI highlights."""
    from minflux_viewer.core.roi_fit import data_fit
    record, kind, adapter = self._active_roi()
    if record is None or record.type not in self._REGION_ROI_TYPES:
        return  # region ROI only; silently ignore otherwise
    ds = self._state.active_dataset
    if ds is None:
        self._no_data_warning()
        return
    pts = self._roi_highlighted_points(record, ds)
    if pts.shape[0] < 3:
        self._state.log(
            "Fit: fewer than 3 localizations highlighted by the ROI.", "WARN")
        return
    try:
        new_record = data_fit(record, pts, target)
    except Exception as exc:
        self._state.log(f"Fit {target} failed: {exc}", "ERROR")
        return
    old_type = record.type
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(
        f"Fit {target.replace('_', ' ')} on ROI '{record.name}' ({old_type}) "
            f"using {pts.shape[0]} localization(s).")


def _spline_fit_active_roi(self: MainWindow) -> None:
    """Process › ROI › Fit ▸ Fit Spline — smooth the active ROI outline."""
    from minflux_viewer.core.roi_fit import spline_fit
    record, kind, adapter = self._active_roi()
    if record is None:
        return
    try:
        new_record = spline_fit(record)
    except Exception as exc:
        self._state.log(f"Spline fit failed: {exc}", "ERROR")
        return
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(f"Spline-fit ROI '{record.name}'.")


def _interpolate_active_roi(self: MainWindow) -> None:
    """Process › ROI › Fit ▸ Interpolate — resample the active ROI outline at
        an even arc-length interval (nm), ImageJ-style."""
    from PyQt6.QtWidgets import QInputDialog

    from minflux_viewer.core.roi_fit import interpolate_outline
    record, kind, adapter = self._active_roi()
    if record is None:
        return
    interval, ok = QInputDialog.getDouble(
        self, "Interpolate ROI", "Interval (nm):", 20.0, 0.1, 1e6, 1)
    if not ok:
        return
    try:
        new_record = interpolate_outline(record, interval_nm=float(interval))
    except Exception as exc:
        self._state.log(f"Interpolate failed: {exc}", "ERROR")
        return
    self._commit_roi_replacement(record, kind, adapter, new_record)
    self._state.log(f"Interpolated ROI '{record.name}' at {interval:g} nm.")


def _coordinate_views_for_dataset(self, idx):
    """Open render / scatter windows for dataset *idx*."""
    wins = []
    for reg in (self._render_windows, self._scatter_windows):
        w = reg.get(idx)
        if w is not None:
            wins.append(w)
    return wins


def _remember_roi_for_restore(self: MainWindow, window, record) -> None:
    """Cache the active ROI draft (called from the overlay controller) so
        Restore ROI can bring it back after a delete."""
    import copy
    if record is None:
        return
    # By key when the window is bound to one (its dataset may move before
    # the draft is restored); the render view reads its dataset as
    # ``_idx`` and the scatter view as ``_dataset_idx``, so neither name
    # can be the one asked for.
    slot: DatasetKey | int | None = window_dataset_key(window)
    if slot is None:
        slot = window_dataset_index(window)
    if slot is not None:
        self._roi_last_active[slot] = copy.deepcopy(record)


def _restore_roi(self: MainWindow) -> None:
    """Process › ROI › Restore ROI — put the active ROI draft onto the other
        open coordinate views of the dataset (render ↔ scatter), or, if none is
        active, restore the last active draft (e.g. after an accidental delete)."""
    import copy
    idx = self._state.active_idx
    ds = self._state.active_dataset
    if idx is None or ds is None:
        self._no_data_warning()
        return
    views = self._coordinate_views_for_dataset(idx)
    source = None
    source_ctrl = None
    for win in views:
        ctrl = getattr(win, "_roi_overlay", None)
        if ctrl is None:
            continue
        try:
            rec = ctrl.current_record()
        except Exception:
            rec = None
        if rec is not None:
            source, source_ctrl = rec, ctrl
            break
    if source is None:
        source = self._roi_last_active.get(idx)   # deleted / last → restore
    if source is None:
        self._state.log(
            "Restore ROI: no active or remembered ROI for this dataset.", "WARN")
        return
    self._roi_last_active[idx] = copy.deepcopy(source)
    applied = 0
    for win in views:
        ctrl = getattr(win, "_roi_overlay", None)
        if ctrl is None or ctrl is source_ctrl:
            continue   # skip the view the source draft already lives on
        try:
            ctrl.replace_draft(copy.deepcopy(source))
            applied += 1
        except Exception:
            pass
    if applied:
        self._state.log(f"Restored ROI '{source.name}' to {applied} view(s).")
    else:
        self._state.log(
            "Restore ROI: no other coordinate view open to restore onto.", "WARN")


def _show_conv_segmentation(self: MainWindow) -> None:
    """Analyze › Segmentation › Convolution… — open the interactive
        geometry-kernel convolution segmentation tool for the active dataset."""
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    from minflux_viewer.ui.analysis.conv_segmentation_dialog import ConvSegmentationWindow
    from minflux_viewer.ui.common.modeless import show_modeless
    win = ConvSegmentationWindow(self._state, idx, owner=self)
    show_modeless(win, self)


def _show_shape_segmentation(self: MainWindow) -> None:
    """Analyze › Segmentation › Shape Model… — open the known-geometry
        (shape-model) segmentation tool for the active dataset."""
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    from minflux_viewer.ui.analysis.shape_segmentation_dialog import ShapeSegmentationWindow
    from minflux_viewer.ui.common.modeless import show_modeless
    win = ShapeSegmentationWindow(self._state, idx, owner=self)
    show_modeless(win, self)


def _show_curvilinear_segmentation(self: MainWindow) -> None:
    """Analyze › Segmentation › Curvilinear Structures… — open the
        interactive Hessian/Frangi filament-tracing tool for the active dataset."""
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    from minflux_viewer.ui.analysis.curvilinear_segmentation_dialog import (
        CurvilinearSegmentationWindow,
    )
    from minflux_viewer.ui.common.modeless import show_modeless
    win = CurvilinearSegmentationWindow(self._state, idx, owner=self)
    show_modeless(win, self)


def _show_straightened_volume_skeleton(self: MainWindow) -> None:
    """Analyze › Segmentation › Straightened Volume along Skeleton… —
        experimental centerline-straightened projection of a 3-D point cloud.

        The skeleton is supplied by selected 3-D point/polyline ROI records in
        the ROI Manager.
        """
    import numpy as np
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    ds = self._state.datasets[idx]
    try:
        loc = coordinate_view(ds)
    except Exception:
        loc = np.empty((0, 2))
    z = loc[:, 2] if loc.ndim == 2 and loc.shape[1] >= 3 else np.empty(0)
    z = z[np.isfinite(z)]
    if loc.ndim != 2 or loc.shape[1] < 3 or z.size == 0 or float(np.ptp(z)) <= 0:
        QMessageBox.information(
            self, "Straightened Volume along Skeleton",
            "The active dataset does not provide a non-degenerate Z coordinate. "
                "This experimental tool needs a 3-D localization dataset.")
        return
    self._show_render(idx)
    self._show_roi_manager()
    from minflux_viewer.ui.analysis.straightened_volume_dialog import (
        StraightenedVolumeAlongSkeletonWindow,
    )
    from minflux_viewer.ui.common.modeless import show_modeless
    win = StraightenedVolumeAlongSkeletonWindow(self._state, idx, owner=self)
    show_modeless(win, self)


def _system_roi_color(self: MainWindow) -> str:
    """The user's configured default ROI color (Preferences ▸ Appearance ▸ ROI,
        default ``Yellow``). Auto-generated ROIs (segmentation / detection) use this
        as their base stroke so they match manually drawn ROIs; the bright cyan
        ``RoiOverlayController.MANAGER_SELECT_COLOR`` is reserved for the ROI-Manager
        selection highlight only (so a selected ROI stays distinguishable)."""
    # rgba_hex (#RRGGBBAA), not rgba_qt_hex: this string is handed to
    # pyqtgraph, which reads Qt's #AARRGGBB as RGBA and turns an opaque
    # color into a fully transparent one.
    from minflux_viewer.colors import pg_safe_hex, viewer_color
    return pg_safe_hex(viewer_color(self._state.prefs, "roi_edge"))


def add_segmentation_rois(self: MainWindow, idx: int, centers, *, side_nm: float,
                          name_prefix: str, source: str, stroke_color: str | None = None,
                          names: list[str] | None = None,
                          log_message: str | None = None) -> int:
    """Add a rectangle ROI of side ``side_nm`` centred on each ``(x, y)`` in
        *centers* to the ROI Manager, reveal them, and show render + manager.

        ``names`` (if given, one per centre) sets each ROI's name; otherwise they
        are auto-numbered ``"<name_prefix> <i>"``. ``stroke_color`` defaults to the
        system ROI color. Shared by the convolution segmentation tool; returns the
        count added."""
    import numpy as np

    from minflux_viewer.core.roi import RoiRecord
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
        return 0
    stroke = stroke_color or self._system_roi_color()
    side = max(float(side_nm), 1.0)
    for i, (cx, cy) in enumerate(centers, start=1):
        name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
        rec = RoiRecord.create(
            "rectangle",
            {"bounds": [float(cx) - side / 2.0, float(cy) - side / 2.0, side, side]},
            name=name, coordinate_space="plot",
            stroke_color=stroke)
        # source_view scopes it to the coordinate views (render/scatter):
        # a detected region is spatial and must not land on a histogram.
        rec.context = {"dataset_idx": idx, "source": source,
                       "source_view": "render"}
        self._state.rois.add(rec)
    self._state.rois.set_show_all(True)
    self._show_render(idx)
    self._show_roi_manager()
    if log_message:
        self._state.log(log_message)
    return int(centers.shape[0])


def add_point_rois_3d(self: MainWindow, idx: int, centers, *, name_prefix: str, source: str,
                      stroke_color: str | None = None, names: list[str] | None = None,
                      log_message: str | None = None) -> int:
    """Add a 3-D point ROI (full ``[x, y, z]`` nm geometry) for each centre in
        *centers* ``(K, 3)`` to the ROI Manager, reveal them, and show render +
        manager. ``stroke_color`` defaults to the system ROI color. Used by the 3-D
        convolution segmentation tool; returns the count added."""
    import numpy as np

    from minflux_viewer.core.roi import RoiRecord
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
        return 0
    stroke = stroke_color or self._system_roi_color()
    for i, (cx, cy, cz) in enumerate(centers, start=1):
        name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
        rec = RoiRecord.create(
            "point", {"point": [float(cx), float(cy), float(cz)]},
            name=name, coordinate_space="plot", stroke_color=stroke)
        # source_view scopes it to the coordinate views (render/scatter):
        # a detected region is spatial and must not land on a histogram.
        rec.context = {"dataset_idx": idx, "source": source,
                       "source_view": "render"}
        self._state.rois.add(rec)
    self._state.rois.set_show_all(True)
    self._show_render(idx)
    self._show_roi_manager()
    if log_message:
        self._state.log(log_message)
    return int(centers.shape[0])


def add_polyline_rois(self: MainWindow, idx: int, paths, *, name_prefix: str, source: str,
                      stroke_color: str | None = None, names: list[str] | None = None,
                      log_message: str | None = None,
                      roi_type: str = "freehand_line") -> int:
    """Add an open poly-line (``freehand_line``) ROI tracing each path in
        *paths* (each an ``(M, 2)`` array of ``(x, y)`` nm vertices) to the ROI
        Manager, reveal them, and show render + manager. ``stroke_color`` defaults to
        the system ROI color. Returns the count added.

        Used by the curvilinear segmentation tool for traced centre lines."""
    import numpy as np

    from minflux_viewer.core.roi import RoiRecord
    if not (0 <= idx < len(self._state.datasets)):
        return 0
    stroke = stroke_color or self._system_roi_color()
    requested = str(roi_type or "freehand_line").strip().lower()
    if requested not in {"freehand_line", "polyline", "line"}:
        requested = "freehand_line"
    added = 0
    for i, path in enumerate(paths, start=1):
        pts = np.asarray(path, dtype=float).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
        record_type = requested
        if requested == "line":
            record_type = "line" if pts.shape[0] == 2 else "polyline"
        rec = RoiRecord.create(
            record_type,
            {"points": pts.tolist(), "closed": False},
            name=name, coordinate_space="plot", stroke_color=stroke)
        # source_view scopes it to the coordinate views (render/scatter):
        # a detected region is spatial and must not land on a histogram.
        rec.context = {"dataset_idx": idx, "source": source,
                       "source_view": "render"}
        self._state.rois.add(rec)
        added += 1
    if added:
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
    return added


def add_polygon_rois(self: MainWindow, idx: int, polygons, *, name_prefix: str, source: str,
                     stroke_color: str | None = None,
                     names: list[str] | None = None,
                     log_message: str | None = None) -> int:
    """Add a closed ``polygon`` ROI tracing each contour in *polygons* (each an
        ``(M, 2)`` array of ``(x, y)`` nm vertices) to the ROI Manager, reveal them,
        and show render + manager. ``stroke_color`` defaults to the system ROI
        color. Returns the count added.

        Used by the shape-model segmentation tool for fitted object contours. A
        ``polygon`` is deliberately the output type rather than the parametric
        shape: it is both a region (so masks/crop/highlighting work) and a
        vertex-editable ROI, so the fitted contour can be corrected by hand and
        then behaves like any hand-drawn ROI."""
    import numpy as np

    from minflux_viewer.core.roi import RoiRecord
    if not (0 <= idx < len(self._state.datasets)):
        return 0
    stroke = stroke_color or self._system_roi_color()
    added = 0
    for i, polygon in enumerate(polygons, start=1):
        pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
        # A closed polygon repeats its first vertex for drawing; the record
        # stores the ring once and is marked closed.
        if pts.shape[0] > 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if pts.shape[0] < 3:
            continue
        name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
        rec = RoiRecord.create(
            "polygon", {"points": pts.tolist(), "closed": True},
            name=name, coordinate_space="plot", stroke_color=stroke)
        # source_view scopes it to the coordinate views (render/scatter):
        # a detected region is spatial and must not land on a histogram.
        rec.context = {"dataset_idx": idx, "source": source,
                       "source_view": "render"}
        self._state.rois.add(rec)
        added += 1
    if added:
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
    return added


def add_particle_average_dataset(self: MainWindow, points_xyz_nm, *, name: str,
                                 log_message: str | None = None) -> int | None:
    """Build a dataset from an averaged particle's pooled localizations
        (``(N, 3)`` nm, already centred/aligned), register it, and open its render
        view (3-D-ready). Returns the new dataset index."""
    import uuid

    import numpy as np

    from minflux_viewer.core.dataset import build_localization_dataset
    pts = np.asarray(points_xyz_nm, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return None
    # Re-zero to the averaged/template frame: subtract the (robust) median so
    # the averaged particle is centred on the origin regardless of the source
    # coordinates. For the symmetric NPC ring the median is the ring centre,
    # so this is ~a no-op when the aligner already centred — and corrective if
    # any residual offset remains.
    pts = pts - np.median(pts, axis=0)
    ds = build_localization_dataset(
        name=name, x_nm=pts[:, 0], y_nm=pts[:, 1], z_nm=pts[:, 2],
        source_version="particle_average", prefs=self._state.prefs)
    # Unique synthetic path so repeated averages don't dedup onto each other.
    ds.file.folder = f"<particle-average>/{uuid.uuid4().hex}"
    ds.metadata["particle_average"] = True
    # Coordinates are final/aligned — pin Z scaling factor to 1.0 and suppress auto-z.
    try:
        ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
        ds.derived["z_scaling_factor"] = 1.0
    except Exception:
        pass
    idx = self._state.add_dataset(ds)
    if log_message:
        self._state.log(log_message)
    # Deselect the (source) detection ROIs: otherwise the selected box ROIs
    # would highlight in the averaged particle's fresh render/scatter at their
    # original — now meaningless — coordinates, which is confusing.
    try:
        self._state.rois.deselect()
    except Exception:
        pass
    # A freshly-appended dataset gets a new index, so its render is created
    # fresh (stale windows from closed datasets are handled by the re-index
    # path). Do NOT close/recreate the auto-opened render here — its async tile
    # workers would deliver into a deleted window and crash. Only the render
    # view is opened by default; the user can open a scatter plot on demand.
    self._show_render(idx)
    return idx


def add_particle_average_overlay(self: MainWindow, channels: dict, *, luts: dict, name: str,
                                 log_message: str | None = None) -> int | None:
    """Add a **multi-channel** averaged overlay: ``channels`` = ``{name: (N,3) nm}``
        (reference channel first). The reference is centred on its median and the
        **same** offset is applied to every channel so they stay aligned; each
        channel becomes an overlay dataset (shared group, per-channel LUT)."""
    import uuid

    import numpy as np

    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.overlay import overlay_color_cycle
    items = []
    for cname, pts in channels.items():
        arr = np.asarray(pts, dtype=float).reshape(-1, 3)
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.shape[0]:
            items.append((str(cname), arr))
    if not items:
        return None
    ref_off = np.median(items[0][1], axis=0)          # centre by the reference channel
    overlay_id = f"pa:{uuid.uuid4().hex}"
    overlay_index = int(getattr(self, "_next_overlay_index", 1))
    self._next_overlay_index = overlay_index + 1
    cycle = overlay_color_cycle(self._state.prefs)
    prev_suspend = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    made: list[int] = []
    try:
        for order, (cname, pts) in enumerate(items, start=1):
            p = pts - ref_off
            ds = build_localization_dataset(
                name=f"{name} · {cname}", x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
                source_version="particle_average", prefs=self._state.prefs)
            ds.file.folder = f"<particle-average>/{uuid.uuid4().hex}"
            ds.metadata["particle_average"] = True
            try:
                ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
                ds.derived["z_scaling_factor"] = 1.0
            except Exception:
                pass
            lut = (luts or {}).get(cname) or cycle[(order - 1) % len(cycle)]
            ds.state.update({
                "overlay_id": overlay_id, "render_group_id": overlay_id,
                "overlay_index": overlay_index, "overlay_order": order,
                "overlay_lut": lut, "render_channel_lut": lut})
            ds.metadata["overlay_id"] = overlay_id
            made.append(self._state.add_dataset(ds))
    finally:
        self._state.suspend_auto_render = prev_suspend
    if log_message:
        self._state.log(log_message)
    try:
        self._state.rois.deselect()
    except Exception:
        pass
    if made:
        self._show_render(made[0])
    return made[0] if made else None


def _on_roi_tool(self: MainWindow, tool: str, checked: bool) -> None:
    if checked:
        self._activate_roi_tool(tool)
    elif self._state.rois.active_tool == tool:
        self._state.rois.set_tool(None)


def _activate_roi_tool(self: MainWindow, tool: str) -> None:
    if not self._volume_tool_allowed(tool):
        return
    self._enter_ortho_for_volume_tool(tool)
    self._state.rois.set_tool(tool)
    self._sync_roi_tool_actions(tool)
    if self._state.rois.active_adapter is None:
        self._state.log("Select a render, histogram, scatter, or attribute plot window before drawing ROIs.", "WARN")


def _enter_ortho_for_volume_tool(self: MainWindow, tool: str) -> None:
    """Selecting a 3-D ROI tool turns the orthogonal view on.

        A volume shape is drawn in one plane and *bounded* in the other two, so a
        single projection cannot show what is being made: the seeded depth would
        be invisible until the ROI was filed. The mode is turned on, never off --
        leaving it when the tool is deselected would yank the layout out from
        under someone who had arranged it.

        Preference order is the focused coordinate view, then this dataset's
        render or scatter; with neither open a scatter is opened, because it is
        the cheaper surface (its ortho mode is an embedded 2x2 grid, where
        render's opens two more top-level windows).
        """
    from minflux_viewer.core.roi_selection import VOLUME_ROI_TYPES
    if tool not in VOLUME_ROI_TYPES:
        return
    idx = self._state.active_idx
    if not isinstance(idx, int) or not (0 <= idx < len(self._state.datasets)):
        return
    view = self._active_coordinate_view()
    if view is None:
        view = self._render_windows.get(idx) or self._scatter_windows.get(idx)
    if view is None:
        self._show_scatter(idx)
        view = self._scatter_windows.get(idx)
    entered = False
    try:
        if view is not None and hasattr(view, "enter_ortho_mode"):
            entered = bool(view.enter_ortho_mode())
    except Exception:
        entered = False
    if entered:
        self._state.status_message.emit(
            f"{tool}: orthogonal view on — draw in any pane; "
                "the remaining axis is taken from the data under the shape")


def _volume_tool_allowed(self: MainWindow, tool: str) -> bool:
    """Refuse a 3-D drawing tool on a 2-D dataset, before anything is drawn.

        Refusing the *gesture* after the fact -- letting the user draw and then
        reverting it -- is the worse experience, and the project's convention is
        to say why rather than to fail silently. The tool simply does not arm,
        and the reason goes to the status bar and the Log.
        """
    from minflux_viewer.core.roi_selection import VOLUME_ROI_TYPES
    if tool not in VOLUME_ROI_TYPES:
        return True
    idx = self._state.active_idx
    ds = (self._state.datasets[idx]
          if isinstance(idx, int) and 0 <= idx < len(self._state.datasets) else None)
    from minflux_viewer.core.dataset_kind import is_3d
    if ds is not None and is_3d(ds):
        return True
    reason = (f"{tool} needs a 3-D dataset"
              if ds is not None else f"{tool} needs an active 3-D dataset")
    self._state.status_message.emit(reason)
    self._state.log(reason, "WARN")
    self._sync_roi_tool_actions(self._state.rois.active_tool or "")
    return False


def _announce_roi_tool(self: MainWindow, tool: str) -> None:
    """Report in the status bar that *tool* is armed and waiting for a draw.

        ⚠ The line is cleared on deactivation only while it is still OURS. A
        finished draw emits its geometry read-out and *then* releases the tool,
        so a blanket clear would wipe the read-out the user just earned -- the
        status bar is shared, and the last writer is not always this method.
        """
    has_view = self._state.rois.active_adapter is not None
    text = roi_tool_status(tool, has_view=has_view)
    if text:
        self._roi_tool_status_text = text
        self._state.status_message.emit(text)
        return
    previous = getattr(self, "_roi_tool_status_text", "")
    self._roi_tool_status_text = ""
    if previous and self._status_label.text() == previous:
        self._state.status_message.emit("")


def _sync_roi_tool_actions(self: MainWindow, tool: str = "") -> None:
    active_tool = tool or self._state.rois.active_tool or ""
    rect_family = {t for _l, t, _ic, _r in _RECT_FAMILY}
    oval_family = {t for _l, t, _ic, _r in _OVAL_FAMILY}
    poly_family = {t for _l, t, _ic, _r in _POLY_FAMILY}
    point_family = {t for _l, t, _ic, _r in _POINT_FAMILY}
    lasso_family = {t for _l, t, _ic, _r in _LASSO_FAMILY}
    line_family = {t for _l, t, _ic in _LINE_FAMILY}
    # Track the active family member + refresh that button's variant icon.
    if active_tool in line_family:
        self._line_variant = active_tool
        self._update_line_button_icon()
    elif active_tool in rect_family:
        self._rect_variant = active_tool
        self._update_shape_button_icon("rectangle")
    elif active_tool in oval_family:
        self._oval_variant = active_tool
        self._update_shape_button_icon("oval")
    elif active_tool in poly_family:
        self._poly_variant = active_tool
        self._update_shape_button_icon("polygon")
    elif active_tool in point_family:
        self._point_variant = active_tool
        self._update_shape_button_icon("point")
    elif active_tool in lasso_family:
        self._lasso_variant = active_tool
        self._update_shape_button_icon("lasso")
    for name, action in self._roi_tool_actions.items():
        blocked = action.blockSignals(True)
        # A family button stays checked for any of its variants.
        if name == "line":
            checked = active_tool in line_family
        elif name == "rectangle":
            checked = active_tool in rect_family
        elif name == "oval":
            checked = active_tool in oval_family
        elif name == "polygon":
            checked = active_tool in poly_family
        elif name == "point":
            checked = active_tool in point_family
        elif name == "magnetic_lasso":
            # The action key is the tool, not the family, so name it here.
            checked = active_tool in lasso_family
        else:
            checked = name == active_tool
        action.setChecked(checked)
        action.blockSignals(blocked)


def _update_line_button_icon(self: MainWindow) -> None:
    from minflux_viewer import resource_path
    icon_file = {t: ic for _l, t, ic in _LINE_FAMILY}.get(self._line_variant, "line.png")
    path = resource_path("icons", icon_file)
    if path.exists():
        self._ui.toolLine.setIcon(_adaptive_toolbar_icon(str(path)))
        self._ui.toolLine.setToolTip(
            {t: lbl for lbl, t, _ic in _LINE_FAMILY}.get(self._line_variant, "Line")
            + " — right-click to switch (straight / poly / freehand line)")


def _install_roi_tool_menus(self: MainWindow) -> None:
    # The Line / Rectangle / Oval / Polygon buttons get a right-click family
    # switcher (Fiji-style). A QToolButton in a QToolBar swallows
    # ``customContextMenuRequested``, so we intercept the right mouse press
    # with an event filter instead (reliable).
    self._roi_tool_buttons = {}
    for attr, key in (
        ("toolLine", "line"),
        ("toolRect", "rectangle"),
        ("toolOval", "oval"),
        ("toolPolygon", "polygon"),
        ("toolPoint", "point"),
        ("toolMagneticLasso", "lasso"),
    ):
        action = getattr(self._ui, attr, None) or getattr(self, attr, None)
        button = self._ui.toolbar.widgetForAction(action) if action is not None else None
        if button is not None:
            self._roi_tool_buttons[button] = key
            button.installEventFilter(self)


def _show_line_family_menu(self: MainWindow, widget: QWidget, pos) -> None:
    """Fiji-style right-click switcher on the Line button."""
    menu = QMenu(widget)
    active = self._state.rois.active_tool
    for label, tool, _ic in _LINE_FAMILY:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(tool == active or (active is None and tool == self._line_variant))
        action.triggered.connect(lambda _checked=False, t=tool: self._select_line_variant(t))
    menu.exec(widget.mapToGlobal(pos))


def _select_line_variant(self: MainWindow, tool: str) -> None:
    self._line_variant = tool
    self._update_line_button_icon()
    self._activate_roi_tool(tool)


def _shape_family(self: MainWindow, kind: str):
    if kind == "rectangle":
        return _RECT_FAMILY
    if kind == "oval":
        return _OVAL_FAMILY
    if kind == "polygon":
        return _POLY_FAMILY
    if kind == "point":
        return _POINT_FAMILY
    if kind == "lasso":
        return _LASSO_FAMILY
    return ()


def _shape_variant(self: MainWindow, kind: str) -> str:
    if kind == "rectangle":
        return self._rect_variant
    if kind == "oval":
        return self._oval_variant
    if kind == "polygon":
        return self._poly_variant
    if kind == "point":
        return self._point_variant
    if kind == "lasso":
        return self._lasso_variant
    return kind


def _show_shape_family_menu(self: MainWindow, widget: QWidget, pos, kind: str) -> None:
    """Fiji-style right-click switcher on a shape-family toolbar button.

        A member whose tool is not registered in ``core.roi.ROI_TOOLS`` is shown
        **disabled, with the reason in its tooltip**, rather than omitted or
        offered: ``RoiStore.set_tool`` ignores an unknown tool, so an enabled
        entry would switch the button icon and then draw nothing -- the project
        convention is disabled-with-a-reason over a silently dead control."""
    from minflux_viewer.core.roi import ROI_TOOLS
    menu = QMenu(widget)
    menu.setToolTipsVisible(True)
    active = self._state.rois.active_tool
    current = self._shape_variant(kind)
    for label, tool, _ic, _rot in self._shape_family(kind):
        action = menu.addAction(label)
        action.setCheckable(True)
        if tool not in ROI_TOOLS:
            action.setEnabled(False)
            action.setToolTip(f"{label} — the drawing tool is not implemented yet")
            continue
        action.setChecked(tool == active or (active is None and tool == current))
        action.triggered.connect(
            lambda _checked=False, k=kind, t=tool: self._select_shape_variant(k, t))
    menu.exec(widget.mapToGlobal(pos))


def _select_shape_variant(self: MainWindow, kind: str, tool: str) -> None:
    if kind == "rectangle":
        self._rect_variant = tool
    elif kind == "oval":
        self._oval_variant = tool
    elif kind == "polygon":
        self._poly_variant = tool
    elif kind == "point":
        self._point_variant = tool
    elif kind == "lasso":
        self._lasso_variant = tool
    self._update_shape_button_icon(kind)
    self._activate_roi_tool(tool)


def _show_wand_dialog(self: MainWindow) -> None:
    """Open the Magic Wand Tool options, arming the wand with them.

        Double-clicking a checkable toolbar button toggles it on the first press,
        so a wand that was already armed would be switched *off* by the very
        gesture that opens its options. Re-selecting the variant here makes the
        double-click mean "show me the options for this tool" in both directions.
        """
    from minflux_viewer.ui.common import modeless
    from minflux_viewer.ui.roi.wand_dialog import MagicWandDialog
    self._select_shape_variant("lasso", "magic_wand")
    dialog = getattr(self, "_wand_dialog", None)
    if dialog is not None:
        try:
            dialog.raise_()
            dialog.activateWindow()
            return
        except RuntimeError:           # the C++ side is gone; build a new one
            self._wand_dialog = None
    dialog = MagicWandDialog(self._state.prefs, self)
    dialog.parameters_changed.connect(self._on_wand_parameters_changed)
    dialog.destroyed.connect(self._on_wand_dialog_destroyed)
    self._wand_dialog = dialog
    modeless.show_modeless(dialog, self)


def _on_wand_dialog_destroyed(self: MainWindow, *_args) -> None:
    self._wand_dialog = None


def _on_wand_parameters_changed(self: MainWindow) -> None:
    """Re-apply the new tolerances to the last wand click.

        Two tolerances are hard to set blind, so moving a slider re-grows the
        region that is already on screen instead of asking for another click.
        The controller that served the last click remembers its seed; with no
        previous click there is simply nothing to re-run.
        """
    controller = getattr(self._state, "_wand_last_controller", None)
    if controller is None:
        return
    try:
        controller.redo_wand()
    except RuntimeError:               # its view has closed since
        self._state._wand_last_controller = None


def _rotated_icon(self: MainWindow, path: str, rotate_deg: float) -> QIcon:
    """A QIcon from *path*, optionally turned *rotate_deg*° (the rotated rectangle
        / ellipse variants reuse the base icon at 45°, no separate asset).

        The source artwork has a baked-in white matte. Normalize it (matte →
        transparent, monochrome linework tinted for the palette) before rotating
        so the toolbar background remains visible, including on dark palettes."""
    if abs(rotate_deg) < 1e-6:
        return _adaptive_toolbar_icon(path)
    pix = _adaptive_toolbar_pixmap(path)
    if pix is None:
        return QIcon(path)
    rotated = pix.transformed(
        QTransform().rotate(rotate_deg), Qt.TransformationMode.SmoothTransformation)
    canvas = QPixmap(rotated.size())
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.drawPixmap(0, 0, rotated)
    painter.end()
    return QIcon(canvas)


def _update_shape_button_icon(self: MainWindow, kind: str) -> None:
    from minflux_viewer import resource_path
    action_name = {
        "rectangle": "toolRect",
        "oval": "toolOval",
        "polygon": "toolPolygon",
        "point": "toolPoint",
        "lasso": "toolMagneticLasso",
    }.get(kind)
    action = None
    if action_name:
        # The .ui buttons hang off ``self._ui``; the ones built in code
        # (Magnetic Lasso, Angle) hang off ``self``.
        #
        # ⚠ A missing attribute on a QObject whose C++ peer is gone raises
        # RuntimeError, not AttributeError, so getattr's default does not
        # absorb it -- the same stale-wrapper tolerance the rest of the
        # QObject lookups in this file are written with.
        try:
            action = (getattr(self._ui, action_name, None)
                      or getattr(self, action_name, None))
        except RuntimeError:
            return
    if action is None:
        return
    current = self._shape_variant(kind)
    entry = {t: (ic, rot) for _l, t, ic, rot in self._shape_family(kind)}.get(current)
    label = {t: lbl for lbl, t, _ic, _rot in self._shape_family(kind)}.get(current, kind.title())
    if entry is None:
        return
    icon_file, rot = entry
    path = resource_path("icons", icon_file)
    if not path.exists():
        return
    action.setIcon(self._rotated_icon(str(path), rot))
    switch = {
        "rectangle": "rectangle / rotated / cuboid",
        "oval": "oval / ellipse / sphere / cylinder",
        "polygon": "polygon / polyhedron",
        "point": "point / multi-point",
        "lasso": "magnetic lasso / magic wand",
    }.get(kind, kind)
    action.setToolTip(f"{label} — right-click to switch ({switch})")


def _show_roi_tool_menu(self: MainWindow, widget: QWidget, pos) -> None:
    menu = QMenu(widget)
    active_tool = self._state.rois.active_tool
    for label, tool, _attr in _ROI_TOOL_DEFS:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(tool == active_tool)
        action.triggered.connect(lambda _checked=False, t=tool: self._activate_roi_tool(t))
    menu.addSeparator()
    clear_action = menu.addAction("No ROI tool")
    clear_action.triggered.connect(lambda: self._state.rois.set_tool(None))
    menu.exec(widget.mapToGlobal(pos))


def _active_coordinate_view(self):
    """A 2-D coordinate view (render/scatter) for the active dataset, or None.

        Prefers the currently focused window when it is the active dataset's
        render/scatter and is in a 2-D (XY/XZ/YZ) view."""
    idx = self._state.active_idx
    if idx is None:
        return None
    rw = self._render_windows.get(idx)
    sw = self._scatter_windows.get(idx)
    def is_2d(win):
        try:
            return win is not None and win.coordinate_view_box() is not None
        except Exception:
            return False
    active = QApplication.activeWindow()
    for win in (active, rw, sw):
        if win in (rw, sw) and is_2d(win):
            return win
    return None


def _window_active_rectangle_bounds(self, win):
    """Bounds (x0,x1,y0,y1) of the active rectangle ROI in *win*'s current
        view plane (draft or selected), or None."""
    from minflux_viewer.core.roi_selection import rectangle_bounds
    plane = win.roi_view_plane()
    candidates = []
    overlay = getattr(win, "_roi_overlay", None)
    if overlay is not None:
        try:
            rec = overlay.current_record()
        except Exception:
            rec = None
        if rec is not None:
            candidates.append(rec)
    try:
        sel = set(self._state.rois.selected_ids)
        candidates.extend(r for r in self._state.rois.records if r.id in sel)
    except Exception:
        pass
    for rec in candidates:
        if getattr(rec, "type", None) != "rectangle":
            continue
        rec_plane = (getattr(rec, "context", {}) or {}).get("view_plane") or plane
        if rec_plane != plane:
            continue
        b = rectangle_bounds(rec)
        if b is not None:
            return b
    return None


def _nice_scalebar_width(span: float) -> float:
    if span <= 0:
        return 100.0
    target = span / 5.0
    best = 1.0
    for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
              10000, 20000, 50000, 100000, 200000, 500000):
        if n <= target:
            best = float(n)
        else:
            break
    return best


def _show_plot_profile(self: MainWindow) -> None:
    """ImageJ-style Plot Profile: localization density along a line / polyline /
        freehand-line ROI in a **render or scatter** view, live-updating with the ROI
        and a tunable width band. Samples the localizations directly, so it is
        independent of the render viewport/zoom."""
    idx = self._state.active_idx
    if self._state.active_dataset is None or idx is None:
        QMessageBox.information(self, "Plot Profile", "Open a dataset first.")
        return
    # Prefer the focused 2-D coordinate view (render or scatter); fall back to the
    # active dataset's render window.
    win = self._active_coordinate_view()
    if win is None or not hasattr(win, "profile_localizations"):
        win = self._render_windows.get(idx) or self._render_window_for_dataset(idx)
    if win is None or win.coordinate_view_box() is None or not hasattr(win, "profile_localizations"):
        self._state.log("Plot Profile: no 2-D localization view for the active dataset.", "WARN")
        QMessageBox.information(
            self, "Plot Profile",
            "Open a render or scatter view of the active dataset in a 2-D "
                "orientation (XY / XZ / YZ) first.")
        return
    # No open-line ROI yet → activate the line tool so the user can draw one and
    # watch the profile appear live.
    overlay = getattr(win, "_roi_overlay", None)
    if overlay is not None and overlay.active_open_line_record() is None:
        try:
            overlay.activate()
            self._state.rois.set_tool("line")
            self._state.log("Plot Profile: draw a line on the view to profile it.", "INFO")
        except Exception:
            pass
    from minflux_viewer.ui.common import modeless
    from minflux_viewer.ui.plots.plot_profile_dialog import PlotProfileDialog
    modeless.show_modeless(PlotProfileDialog(win, owner=self), self)


def _show_scale_bar(self: MainWindow) -> None:
    if self._state.active_dataset is None:
        self._state.log("Scale bar: open a dataset first.", "WARN")
        QMessageBox.information(self, "Scale Bar", "Open a dataset first.")
        return
    win = self._active_coordinate_view()
    if win is None:
        self._state.log("Scale bar: no 2-D coordinate view (render/scatter) for the "
                            "active dataset.", "WARN")
        QMessageBox.information(
            self, "Scale Bar",
            "Open a render or scatter view of the active dataset in a 2-D "
                "orientation (XY / XZ / YZ) first.")
        return
    vb = win.coordinate_view_box()
    try:
        (xmin, xmax), _ = vb.viewRange()
        span = abs(xmax - xmin)
    except Exception:
        span = 0.0
    default_w = self._nice_scalebar_width(span)
    default_h = max(default_w / 20.0, (span / 400.0) if span else 1.0)
    from minflux_viewer.ui.dialogs.scale_bar_dialog import ScaleBarDialog
    dlg = ScaleBarDialog(self, default_width_nm=default_w, default_height_nm=default_h,
                         has_selection=self._window_active_rectangle_bounds(win) is not None)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    params = dlg.values()
    if params["location"] != "At Selection":
        self._add_scale_bar(win, params, roi_center=None)
        return
    bounds = self._window_active_rectangle_bounds(win)
    if bounds is not None:
        self._add_scale_bar(win, params, roi_center=self._rect_center(bounds))
        return
    # No rectangle ROI yet → let the user draw one, then place at its centre.
    overlay = getattr(win, "_roi_overlay", None)
    if overlay is None or not hasattr(overlay, "request_rectangle"):
        self._add_scale_bar(win, params, roi_center=None)  # fall back to a corner
        return
    self._state.log("Scale bar (At Selection): draw a rectangle on the view to "
                        "position the scale bar.", "INFO")
    def _on_rect(record, _win=win, _params=params):
        from minflux_viewer.core.roi_selection import rectangle_bounds
        b = rectangle_bounds(record)
        if b is not None:
            self._add_scale_bar(_win, _params, roi_center=self._rect_center(b))
    overlay.request_rectangle(_on_rect)


def _rect_center(bounds) -> tuple[float, float]:
    x0, x1, y0, y1 = bounds
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _add_scale_bar(self: MainWindow, win, params, *, roi_center) -> None:
    vb = win.coordinate_view_box()
    if vb is None:
        self._state.log("Scale bar: the view is no longer a 2-D coordinate view.", "WARN")
        return
    from minflux_viewer.ui.widgets.scale_bar import (
        ScaleBarItem,
        format_nm,
        initial_scale_bar_center,
    )
    center = initial_scale_bar_center(vb, params, roi_center=roi_center)
    try:
        sb = ScaleBarItem(
            vb, width_nm=params["width_nm"], height_nm=params["height_nm"],
            font_size=params["font_size"], color=params["color"],
            bg_color=params["bg_color"], horizontal=params["horizontal"],
            center=center)
    except Exception as exc:
        self._state.log(f"Scale bar failed: {exc}", "ERROR")
        return
    sb.sigEditRequested.connect(lambda item, _w=win: self._edit_scale_bar(_w, item))
    sb.sigDeleteRequested.connect(lambda item, _w=win: self._delete_scale_bar(_w, item))
    bars = getattr(win, "_scale_bars", None)
    if bars is None:
        bars = []
        win._scale_bars = bars
    bars.append(sb)
    where = "at selection" if roi_center is not None else params["location"].lower()
    self._state.log(f"Added {format_nm(params['width_nm'])} scale bar ({where}). "
                        "Drag to move; right-click for Property / Delete.")


def _edit_scale_bar(self: MainWindow, win, item) -> None:
    from minflux_viewer.ui.dialogs.scale_bar_dialog import ScaleBarDialog
    dlg = ScaleBarDialog(self, initial=item.params(), edit_mode=True)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    item.set_params(dlg.values())
    self._state.log("Updated scale bar properties.")


def _delete_scale_bar(self: MainWindow, win, item) -> None:
    try:
        item.remove()
    except Exception:
        pass
    bars = getattr(win, "_scale_bars", None)
    if bars and item in bars:
        bars.remove(item)
    self._state.log("Deleted scale bar.")


def _duplicate_active_dataset(self: MainWindow) -> None:
    """Edit → Duplicate / ``Shift+D``.

        With an active rectangle ROI on the dataset, opens the crop options
        dialog (first time per session, or whenever "stop asking" is off / a
        specific Z range is involved) and produces a cropped duplicate;
        otherwise a plain duplicate.
        """
    src = self._state.active_dataset
    if src is None:
        self._no_data_warning()
        return
    src_idx = self._state.active_idx
    record = self._active_region_record(src, src_idx)
    if record is None:
        self._plain_duplicate(src)     # non-region ROI or no ROI → whole dataset
        return
    from minflux_viewer.core import roi_crop as RC
    from minflux_viewer.core.overlay import overlay_members
    try:
        members = overlay_members(self._state, src_idx) or [(src_idx, src)]
    except Exception:
        members = [(src_idx, src)]
    channel_names = [ds.name for _i, ds in members]
    setup = self._roi_crop_setup.get(id(src))
    # Silent reuse only when stop-asking AND All-Z (a specific Z range can't
    # be reused — it varies by XY region); otherwise (re)show the dialog.
    show_dialog = setup is None or not setup.stop_asking or not setup.z_all
    opts = setup
    if show_dialog:
        from minflux_viewer.ui.roi.crop_dialog import CropDialog
        z_vals = None
        if int(getattr(src.prop, "num_dim", 2) or 2) >= 3:
            coords = RC.display_coords(src)
            if coords.size:
                xy_inside = RC.compute_crop_mask(src, record, trace_complete=False)
                z_vals = coords[:, 2][xy_inside]
        dlg = CropDialog(
            src.name, has_roi=True,
            channels=channel_names if len(channel_names) > 1 else None,
            z_values=z_vals, initial=setup, parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        if opts.stop_asking:
            self._roi_crop_setup[id(src)] = opts
        else:
            self._roi_crop_setup.pop(id(src), None)
    if opts is None:
        self._plain_duplicate(src)
        return
    self._execute_crop(src_idx, src, record, members, opts)


def _add_cross_section_to_roi(self: MainWindow) -> None:
    """Process › ROI › Add Cross-Section… — make a polyhedron multi-slice.

        The gesture is: select the polyhedron, draw the outline again where it
        should be at another Z, then invoke this. The drawn polygon becomes a
        level and everything between the levels is *interpolated* (angular
        resampling), so the shape follows the structure instead of extruding one
        outline through it.

        ⚠ The Z is asked for rather than taken from the view. Render has a
        crosshair to read, scatter does not, and a level placed at a depth the
        user never chose is worse than one typed in.
        """
    from PyQt6.QtWidgets import QInputDialog

    from minflux_viewer.core.roi_volume import add_cross_section
    idx = self._state.active_idx
    if not isinstance(idx, int) or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    selected = [r for r in self._state.rois.records
                if r.id in set(self._state.rois.selected_ids or [])
                and r.type == "polyhedron"]
    if len(selected) != 1:
        QMessageBox.information(
            self, "Add Cross-Section",
            "Select exactly one polyhedron ROI in the ROI Manager first.")
        return
    target = selected[0]
    if str((target.geometry or {}).get("representation") or "") == "projection_hull":
        QMessageBox.information(
            self, "Add Cross-Section",
            "This projection-hull ROI is edited directly in its XY, XZ, "
                "and YZ panes. Add Cross-Section applies only to legacy "
                "contour-stack polyhedra.")
        return
    polygon = None
    view = self._active_coordinate_view()
    for win in (view, self._render_windows.get(idx), self._scatter_windows.get(idx)):
        controller = getattr(win, "_roi_overlay", None)
        draft = getattr(controller, "draft", None) if controller else None
        if draft is not None and draft.type in {"polygon", "freehand"}:
            polygon = (draft.geometry or {}).get("points")
            break
    if not polygon or len(polygon) < 3:
        QMessageBox.information(
            self, "Add Cross-Section",
            "Draw a polygon where the new cross-section should be, then run "
                "this again. It becomes a level of the selected polyhedron.")
        return
    levels = (target.geometry or {}).get("levels") or []
    default = float(levels[-1]["at"]) if levels else 0.0
    at, ok = QInputDialog.getDouble(
        self, "Add Cross-Section", "Place this cross-section at Z (nm):",
        default, -1e9, 1e9, 1)
    if not ok:
        return
    geometry = add_cross_section(target, at, [[p[0], p[1]] for p in polygon])
    if geometry is None:
        QMessageBox.information(self, "Add Cross-Section",
                                "That outline cannot be used as a cross-section.")
        return
    import copy as _copy
    updated = _copy.deepcopy(target)
    updated.geometry = geometry
    updated.selection_dirty = True
    self._state.rois.update(target.id, updated)
    self._state.log(
        f"Add cross-section at Z={at:g} nm: '{target.name}' now has "
            f"{len(geometry['levels'])} level(s).")


def _active_region_record(self, ds, ds_idx):
    """The active **region** ROI (rectangle/oval/polygon/freehand) for *ds*.

        A freshly drawn ROI is a *draft* on the render/scatter overlay controller
        (not yet in ``state.rois.records``), so we look there as well as among the
        selected persistent ROIs. Non-region ROIs (line/point/angle) → ``None``
        (the duplicate then ignores the ROI and copies the whole dataset).
        """
    from minflux_viewer.core.roi_convert import REGION_TYPES
    from minflux_viewer.core.roi_selection import VOLUME_ROI_TYPES
    # A volume ROI encloses a region too, so Shift+D crops to a cuboid /
    # sphere / polyhedron exactly as it does to a rectangle -- and it is the
    # 3-D case that makes the crop dialog's separate Z slab redundant
    # (compute_crop_mask intersects the two rather than picking one).
    region_types = REGION_TYPES | VOLUME_ROI_TYPES
    # 1) a selected, persisted region ROI
    try:
        wanted = set(self._state.rois.selected_ids or [])
    except Exception:
        wanted = set()
    try:
        for record in self._state.rois.records:
            if record.id in wanted and record.type in region_types:
                return record
    except Exception:
        pass
    # 2) a freshly drawn draft on this dataset's render/scatter overlay.
    for win in (
        self._render_windows.get(ds_idx),
        self._scatter_windows.get(ds_idx),
    ):
        ctrl = getattr(win, "_roi_overlay", None)
        if ctrl is None:
            continue
        try:
            draft = ctrl.current_record()
        except Exception:
            draft = getattr(ctrl, "draft", None)
        if draft is not None and getattr(draft, "type", None) in region_types:
            return draft
    return None


def _copy_dataset(self: MainWindow, src, new_name: str):
    """Deep-copy a dataset as a fresh standalone (no overlay/source links)."""
    import copy

    from minflux_viewer.core.dataset import FileInfo, MinfluxDataset
    timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
    new_file = FileInfo(
        name=new_name, folder=src.file.folder, datetime=timestamp,
        raw_data=None, recent_path=src.file.recent_path,
    )
    kwargs = dict(
        file=new_file,
        prop=copy.deepcopy(src.prop),
        attr=copy.deepcopy(src.attr),
    )
    cali = copy.deepcopy(getattr(src, "cali", None))
    channel = copy.deepcopy(getattr(src, "channel", None))
    if cali is not None:
        kwargs["cali"] = cali
    if channel is not None:
        kwargs["channel"] = channel
    dup = MinfluxDataset(**kwargs)
    dup.metadata.update(copy.deepcopy(src.metadata))
    dup.state.update(copy.deepcopy(src.state))
    for key in ("msr_source_path", "msr_dataset_key", "msr_dataset_name"):
        dup.metadata.pop(key, None)
    for key in ("overlay_id", "overlay_index", "overlay_order",
                "overlay_transform", "render_group_id", "render_transform_2d"):
        dup.state.pop(key, None)
        dup.metadata.pop(key, None)
    dup.metadata["source_dataset_name"] = src.file.name
    dup.metadata["created_note"] = f"{timestamp}, derived from dataset: {src.file.name}"
    return dup


def _plain_duplicate(self: MainWindow, src) -> None:
    try:
        dup = self._copy_dataset(src, self._next_duplicate_name(src.file.name))
        dup.metadata["duplicated_from_dataset"] = src.file.name
        try:
            dup.filter_mask = src.filter_mask.copy()
        except Exception:
            pass
    except Exception as exc:
        self._state.log(f"Duplicate failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Duplicate", str(exc))
        return
    self._state.add_dataset(dup)
    self._state.log(f"Duplicated dataset as '{dup.file.name}'.")
    try:
        self._state.journal.add(
            "transform",
            f"Duplicated dataset '{src.file.name}' as '{dup.file.name}' "
                f"(filter mask preserved).",
        )
    except Exception:
        pass


def _execute_crop(self: MainWindow, src_idx, src, record, members, opts) -> None:
    """Produce cropped duplicate(s) per the chosen options (Model A or B).

        A multi-channel crop is kept grouped as a new overlay (each channel's
        LUT + transform carried over); a single-channel crop stays standalone,
        so it renders with the default LUT (hot) like any lone dataset.
        """
    import copy
    import uuid

    import numpy as np

    from minflux_viewer.core import roi_crop as RC
    if len(members) > 1 and opts.channels:
        sel = [(i, ds) for pos, (i, ds) in enumerate(members, start=1)
               if pos in opts.channels]
    else:
        sel = [(src_idx, src)]
    if not sel:
        sel = [(src_idx, src)]
    z_range = None if opts.z_all else opts.z_range
    trace_complete = not opts.clip
    multi = len(sel) > 1
    overlay_id = overlay_index = None
    if multi:
        overlay_index = self._next_overlay_index
        self._next_overlay_index += 1
        overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
    made: list[int] = []
    prev_suspend = getattr(self._state, "suspend_auto_render", False)
    if multi:
        self._state.suspend_auto_render = True   # group before rendering
    try:
        exact = bool(getattr(opts, "exact_shape", False))
        for order, (_idx, ds) in enumerate(sel, start=1):
            try:
                mask = RC.compute_crop_mask(
                    ds, record, z_range=z_range, trace_complete=trace_complete,
                    exact_shape=exact)
                name = self._unique_name("CROP_", ds.file.name)
                if opts.spatial_filter:                       # Model A
                    dup = self._copy_dataset(ds, name)
                    base = np.asarray(ds.filter_mask, dtype=bool)
                    if base.shape[0] != mask.shape[0]:
                        base = np.ones(mask.shape[0], dtype=bool)
                    dup.filter_mask = base & mask
                    if RC.crop_is_axis_aligned(ds, record, exact_shape=exact):
                        dup.state["filter_specs"] = list(ds.state.get("filter_specs") or []) \
                            + RC.crop_filter_specs(record, trace_complete=trace_complete,
                                                   z_range=z_range, exact_shape=exact)
                    else:
                        dup.metadata["crop_mask_only"] = True   # opaque mask (not re-evaluable)
                    dup.metadata["cropped_from_dataset"] = ds.file.name
                    kept = int(np.asarray(dup.filter_mask, dtype=bool).sum())
                else:                                          # Model B (subset)
                    dup = RC.subset_dataset(ds, mask, name=name, prefs=self._state.prefs)
                    kept = int(dup.prop.num_loc)
                if multi:
                    dup.state["overlay_id"] = overlay_id
                    dup.state["render_group_id"] = overlay_id
                    dup.state["overlay_index"] = overlay_index
                    dup.state["overlay_order"] = order
                    for key in ("overlay_lut", "render_channel_lut",
                                "overlay_transform", "render_transform_2d"):
                        val = ds.state.get(key)
                        if val is not None:
                            dup.state[key] = copy.deepcopy(val)
                made.append(self._state.add_dataset(dup))
                model = "filter" if opts.spatial_filter else "subset"
                self._state.log(
                    f"Cropped '{ds.file.name}' → '{name}' ({model}, {kept:,} locs).")
            except Exception as exc:
                self._state.log(f"Crop of '{ds.file.name}' failed: {exc}", "ERROR")
    finally:
        self._state.suspend_auto_render = prev_suspend
    if not made:
        QMessageBox.warning(self, "Duplicate / crop",
                            "No cropped dataset was produced (check the ROI).")
        return
    # Move the ROI itself into the cropped view (active there), per the request.
    self._carry_roi_to_cropped(record, made[0])
    self._show_render(made[0])                  # open the cropped (grouped) view


def _carry_roi_to_cropped(self: MainWindow, record, new_idx: int) -> None:
    """Add a copy of the source ROI to the store, retargeted to the cropped
        dataset and selected, so it shows as the active ROI in the new view."""
    import copy
    import uuid
    try:
        clone = copy.deepcopy(record)
    except Exception:
        return
    clone.id = uuid.uuid4().hex
    from minflux_viewer.core.roi_scope import (
        SHARED_DATASET_KEYS_FIELD,
        SHARED_DATASETS_KEY,
        with_owner,
    )
    # The source's scope is held by key: retarget the key, not a position.
    ctx = with_owner(clone.context, new_idx, self._state)
    # The copy belongs to the crop. Inheriting the source's manual sharing
    # list would put it on datasets the user never chose it for.
    ctx.pop(SHARED_DATASETS_KEY, None)
    ctx.pop(SHARED_DATASET_KEYS_FIELD, None)
    clone.context = ctx
    clone.mask_key = ""
    clone.selected_count = None
    clone.selection_dirty = True
    self._state.rois.add(clone)
    self._state.rois.select([clone.id])


def _unique_name(self: MainWindow, prefix: str, base: str) -> str:
    """``<prefix><base>`` with numbering only when needed."""
    existing = {ds.file.name for ds in self._state.datasets}
    name = f"{prefix}{base}"
    if name not in existing:
        return name
    n = 2
    while f"{prefix}{n}_{base}" in existing:
        n += 1
    return f"{prefix}{n}_{base}"


def _next_duplicate_name(self: MainWindow, source_name: str) -> str:
    """Return DUP_<source_name>, with numbering only when needed."""
    return self._unique_name("DUP_", source_name)
