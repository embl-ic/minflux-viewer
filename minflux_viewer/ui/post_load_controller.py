"""Post load controller delegated by MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer

from .background_tasks import BackgroundTask
from .data_window import DataWindow
from .dataset_binding import (
    event_index,
)

if TYPE_CHECKING:
    from .main_window import MainWindow


def _on_dataset_added(self: MainWindow, event) -> None:
    idx = event_index(self._state, event)
    if idx is None:
        return
    ds  = self._state.datasets[idx]
    # A file saved as part of an overlay says so in its recipe; offering to
    # rebuild the set waits for the rest of the batch to arrive.
    self._maybe_offer_regroup(idx)
    # A recipe loaded while its data was not yet open applies here, or is
    # discarded -- its lifetime is exactly one dataset load.
    pending = self._take_pending_metadata(ds)
    if pending:
        path, meta, kind = pending
        from ..core.metadata_match import describe_match
        self._state.log(
            f"Pending processing metadata '{path.name}' matches "
                f"'{ds.name}' ({describe_match(kind)}) — applying.",
            dataset_idx=idx)
        self.apply_metadata_to_dataset(idx, meta, path.name)
    win = DataWindow(ds, idx, self._state)
    offset = idx * 22
    win.move(120 + offset, 600 - offset)
    self._data_windows[idx] = win
    self._mark_recent_menu_stale()
    data_prefs = self._state.prefs.get("data", {})
    if data_prefs.get("show_dataset_manager", False):
        self._ensure_dataset_manager_visible()
    # Open the render view first (when requested, unless a batch importer will
    # open a grouped render view) so the data-info window can be placed *beside*
    # it rather than on top of it.
    if data_prefs.get("show_render", True) and not getattr(self._state, "suspend_auto_render", False):
        self._show_render(idx)
    if data_prefs.get("show_data_info", True):
        from .modeless import place_beside
        # Anchor to this dataset's render view if it opened, else the main
        # window; try right → below → left → above, staying on-screen.
        anchor = self._render_windows.get(idx) or self
        win.adjustSize()                              # realise its fixed size
        place_beside(win, anchor, prefer=("right", "below", "left", "above"))
        win.show()
    if data_prefs.get("show_attr_plot", False):
        self._show_attr_plot(idx)
    if data_prefs.get("show_histogram", False):
        self._show_histogram(idx)
    if data_prefs.get("show_scatter", False):
        self._show_scatter(idx)
    self._schedule_post_load_computations(idx)


def _schedule_post_load_computations(self: MainWindow, idx: int) -> None:
    """Run optional one-time computed attributes after initial windows show.

        Scheduled with the dataset's key, not its index: a dataset closed before
        the timer fires shifts every later one, and the index would then start
        post-load for whichever dataset had moved into that position.
        """
    key = self._state.key_at(idx)
    if key is None:
        return
    delay_ms = 450 + max(0, idx) * 100
    QTimer.singleShot(delay_ms, lambda k=key: self._run_post_load_for_key(k))


def _run_post_load_for_key(self: MainWindow, key) -> None:
    """Start post-load for the dataset with *key*, wherever it is now."""
    if self._is_shutting_down:
        return
    idx = self._state.index_of(key)
    if idx is not None:
        self._run_post_load_computations(idx)


def _run_post_load_computations(self: MainWindow, idx: int) -> None:
    """Compute derived attributes in a worker and apply them on Qt's thread."""
    if self._is_shutting_down:
        return
    if not (0 <= idx < len(self._state.datasets)):
        return
    ds = self._state.datasets[idx]
    data_prefs = self._state.prefs.get("data", {})
    plot_prefs = self._state.prefs.get("plot", {})
    z_scaling_factor_requested = (
        data_prefs.get("estimate_z_scaling_factor", False)
        or plot_prefs.get("use_fixed_z_scaling_factor", False)
    )
    needs = (
        (z_scaling_factor_requested and "z_scaling_factor" not in ds.derived)
        or (data_prefs.get("compute_loc_prec", True) and "sigma_per_trace_nm" not in ds.derived)
        or (data_prefs.get("compute_local_density", True) and "den" not in ds.attr)
        or bool(ds.state.get("filter_specs"))
    )
    if needs:
        self._state.log(f"Post-load processing of '{ds.name}' (Z scaling factor, precision, density)…")
        self._state.status_progress(f"Processing '{ds.name}'")
    else:
        self._post_load_finalize(ds)
        return
    import copy

    from ..analysis.post_load import POST_LOAD_ALGORITHMS
    from ..core.post_load import compute_post_load_results
    tasks = getattr(self, "_post_load_tasks", None)
    if tasks is None:
        tasks = self._post_load_tasks = {}
    # Keyed by the dataset's key, which is never given to another dataset,
    # so closing one retires exactly its own work (_retire_dataset_tasks).
    key = self._state.key_at(idx)
    if key is None or key in tasks:
        return
    prefs = copy.deepcopy(self._state.prefs)
    def work(report):
        return compute_post_load_results(ds, prefs, report, algorithms=POST_LOAD_ALGORITHMS)
    task = BackgroundTask(
        work,
        description=f"Post-load processing of {ds.name}",
        category="analysis",
    )
    tasks[key] = task
    task.signals.stage.connect(
        lambda text: self._state.status_progress(text)
    )
    task.signals.done.connect(
        lambda result, _key=key, _ds=ds: self._on_post_load_done(_key, _ds, result)
    )
    task.signals.failed.connect(
        lambda message, _key=key, _ds=ds: self._on_post_load_error(_key, _ds, message)
    )
    task.signals.finished.connect(
        lambda _key=key, _task=task: self._finish_post_load_task(_key, _task)
    )
    from .background_tasks import shared_thread_pool
    shared_thread_pool("post-load").start(task)


def _finish_post_load_task(self: MainWindow, key, task) -> None:
    tasks = getattr(self, "_post_load_tasks", {})
    if tasks.get(key) is task:
        tasks.pop(key, None)


def _retire_dataset_tasks(self: MainWindow, event) -> None:
    """Cancel and detach the post-load work of a dataset that has closed.

        Only that dataset's: closing one dataset used to retire every running
        post-load task, so a dataset opened just before another was closed never
        received its density or precision. Native work already running finishes
        on its pool; nothing it returns is delivered.
        """
    from .background_tasks import retire_background_tasks
    task = getattr(self, "_post_load_tasks", {}).pop(event.key, None)
    if task is not None:
        retire_background_tasks([task])


def _on_post_load_done(self: MainWindow, key, ds, result: dict) -> None:
    """Deliver a post-load result only while the dataset it started for is open.

        Checked by the key the task started under: a dataset closed and opened
        again is a new registration, and a result for the old one is not its.
        """
    if self._state.index_of(key) is None:
        return
    self._apply_post_load_results(ds, result)


def _on_post_load_error(self: MainWindow, key, ds, message: str) -> None:
    if self._state.index_of(key) is None:
        return
    self._on_post_load_failed(ds, message)


def _on_post_load_failed(self: MainWindow, ds, message: str) -> None:
    if self._post_load_index(ds) is None:
        return
    self._state.log(
        f"Post-load processing failed for '{ds.name}': {message}", "WARN"
    )
    self._start_saved_filter_restore(ds)


def _apply_post_load_results(self: MainWindow, ds, result: dict) -> None:
    idx = self._post_load_index(ds)
    if idx is None or self._is_shutting_down:
        return
    import numpy as np
    attributes_changed = False
    z_update = result.get("z_update")
    if z_update is not None and "z_scaling_factor" not in ds.derived:
        value = float(z_update["value"])
        ds.set_z_scaling_factor(value, source=str(z_update["source"]))
        ds.derived["z_scaling_factor"] = np.asarray([value], dtype=float)
        details = z_update.get("details")
        if details is not None:
            ds.derived["z_scaling_factor_sizes_x"] = details["x"].sizes
            ds.derived["z_scaling_factor_sizes_y"] = details["y"].sizes
            ds.derived["z_scaling_factor_sizes_z"] = details["z"].sizes
        self._state.notify_calibration_changed(idx)
    precision = result.get("loc_precision")
    if precision is not None and "sigma_per_trace_nm" not in ds.derived:
        ds.attr["sigma_per_trace_nm"] = precision["per_trace_sigma_xyz"]
        ds.attr["sigma_trace_ids"] = precision["trace_ids"]
        ds.derived["sigma_per_trace_nm"] = precision["per_trace_sigma_xyz"]
        ds.derived["sigma_trace_ids"] = precision["trace_ids"]
        ds.cali.loc_precision = np.asarray(
            precision["median_sigma_xyz"], dtype=float
        )
        attributes_changed = True
    density = result.get("density")
    if density is not None and "den" not in ds.attr:
        ds.attr["den"] = density
        ds.attr["local_density"] = density
        ds.derived["den"] = density
        ds.derived["local_density"] = density
        for name in ("den", "local_density"):
            if name not in ds.prop.attr_names:
                ds.prop.attr_names.append(name)
        attributes_changed = True
    last_density = result.get("last_density")
    if last_density is not None and "den" not in ds.components.derived_last:
        ds.components.derived_last["den"] = last_density
    for message, level in result.get("logs", []):
        self._state.log(message, level)
    if attributes_changed:
        self._state.notify_attributes_changed(idx)
    self._start_saved_filter_restore(ds)


def _start_saved_filter_restore(self: MainWindow, ds) -> None:
    key = self._state.key_of(ds)
    if key is None:
        return
    if not ds.state.get("filter_specs"):
        self._post_load_finalize(ds)
        return
    from ..core.loader import evaluate_saved_filters
    def work(report):
        report(f"Restoring saved filters for '{ds.name}'")
        return evaluate_saved_filters(ds)
    task = BackgroundTask(
        work,
        description=f"Restoring filters for {ds.name}",
        category="analysis",
    )
    task.signals.done.connect(
        lambda mask, _key=key, _ds=ds: self._on_saved_filters_done(_key, _ds, mask)
    )
    task.signals.failed.connect(
        lambda message, _key=key, _ds=ds: self._on_saved_filters_error(_key, _ds, message)
    )
    task.signals.finished.connect(
        lambda _key=key, _t=task: self._finish_post_load_task(_key, _t)
    )
    # Registered with the dataset's post-load work, so closing the dataset
    # retires it too (_retire_dataset_tasks).
    tasks = getattr(self, "_post_load_tasks", None)
    if tasks is None:
        tasks = self._post_load_tasks = {}
    tasks[key] = task
    from .background_tasks import shared_thread_pool
    shared_thread_pool("post-load").start(task)


def _on_saved_filters_done(self: MainWindow, key, ds, mask) -> None:
    if self._state.index_of(key) is not None:
        self._on_saved_filters_evaluated(ds, mask)


def _on_saved_filters_error(self: MainWindow, key, ds, message: str) -> None:
    if self._state.index_of(key) is not None:
        self._on_saved_filter_failed(ds, message)


def _on_saved_filters_evaluated(self: MainWindow, ds, mask) -> None:
    idx = self._post_load_index(ds)
    if idx is None:
        return
    if mask is not None:
        ds.filter_mask = mask
        self._state.notify_filter_changed(idx)
    self._post_load_finalize(ds)


def _on_saved_filter_failed(self: MainWindow, ds, message: str) -> None:
    if self._post_load_index(ds) is None:
        return
    self._state.log(
        f"Restoring filters for '{ds.name}' failed: {message}", "WARN"
    )
    self._post_load_finalize(ds)


def _post_load_index(self: MainWindow, ds) -> int | None:
    """Current index of *ds* (through its key), or None if it was removed."""
    return self._state.index_of_dataset(ds)


def _post_load_next(self: MainWindow, ds, step) -> None:
    """Run the next post-load step from the event loop (keeps the UI live)."""
    QTimer.singleShot(0, lambda: step(ds))


def _restart_post_load_in_background(self: MainWindow, ds) -> None:
    """Compatibility entry point for extensions calling the former steps."""
    idx = self._post_load_index(ds)
    if idx is not None:
        self._run_post_load_computations(idx)


def _post_load_z_scaling_factor(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _legacy_post_load_z_scaling_factor(self: MainWindow, ds) -> None:
    idx = self._post_load_index(ds)
    if idx is None:
        return
    data_prefs = self._state.prefs.get("data", {})
    plot_prefs = self._state.prefs.get("plot", {})
    estimate_z_scaling_factor = data_prefs.get("estimate_z_scaling_factor", False)
    use_fixed_z_scaling_factor = plot_prefs.get("use_fixed_z_scaling_factor", False)
    if (estimate_z_scaling_factor or use_fixed_z_scaling_factor) and "z_scaling_factor" not in ds.derived:
        import numpy as np
        if ds.prop.num_dim < 3:
            # 2D data: Z is all zero, so anisotropy estimation is moot.
            ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
            ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
            self._state.log(f"Z scaling factor for '{ds.name}': 1.0 (2D dataset, computation skipped).")
            self._state.notify_calibration_changed(idx)
        elif use_fixed_z_scaling_factor:
            fixed = float(plot_prefs.get("z_scaling_factor", 0.67))
            ds.set_z_scaling_factor(fixed, source="fixed (preference)")
            ds.derived["z_scaling_factor"] = np.asarray([fixed], dtype=float)
            self._state.log(f"Z scaling factor for '{ds.name}': {fixed:.4g} (fixed preference value).")
            self._state.notify_calibration_changed(idx)
        elif estimate_z_scaling_factor:
            # Heavy estimate: announce now and run it on the next event-loop
            # turn so this line is painted before the (sub-second) compute.
            self._state.log(
                f"Estimating Z scaling factor from trace anisotropy for '{ds.name}'…"
            )
            self._state.status_progress(
                f"Estimating Z scaling factor for '{ds.name}'"
            )
            self._post_load_next(ds, self._post_load_z_scaling_factor_estimate)
            return
    self._post_load_next(ds, self._post_load_loc_prec)


def _post_load_z_scaling_factor_estimate(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _legacy_post_load_z_scaling_factor_estimate(self: MainWindow, ds) -> None:
    idx = self._post_load_index(ds)
    if idx is None:
        return
    import time

    import numpy as np
    value = None
    try:
        t0 = time.perf_counter()
        from ..analysis.trace_analysis import estimate_anisotropy_for_dataset
        res = estimate_anisotropy_for_dataset(ds)
        dt = time.perf_counter() - t0
        if res is not None and np.isfinite(res["z_scaling_factor"]):
            value = float(res["z_scaling_factor"])
            if 0.5 <= value <= 1.0:
                ds.set_z_scaling_factor(
                    value, source="estimated (trace anisotropy)"
                )
                ds.derived["z_scaling_factor"] = np.asarray([value], dtype=float)
                ds.derived["z_scaling_factor_sizes_x"] = res["x"].sizes
                ds.derived["z_scaling_factor_sizes_y"] = res["y"].sizes
                ds.derived["z_scaling_factor_sizes_z"] = res["z"].sizes
                self._state.log(
                    f"Estimated Z scaling factor for '{ds.name}': {value:.4g} "
                        f"(trace anisotropy, raw last-valid z, {dt:.1f}s)"
                )
    except Exception as exc:
        self._state.log(f"Z scaling factor estimation failed for '{ds.name}': {exc}", "WARN")
    if "z_scaling_factor" not in ds.derived:
        ds.set_z_scaling_factor(
            1.0, source="estimate out of range (reset to 1.0)"
        )
        ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
        shown = f"{value:.4g}" if value is not None else "failed"
        self._state.log(
            f"Z scaling factor for '{ds.name}': estimate {shown} outside [0.5, 1.0] — reset to 1.0.",
            "WARN",
        )
    self._state.notify_calibration_changed(idx)
    self._post_load_next(ds, self._post_load_loc_prec)


def _post_load_loc_prec(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _legacy_post_load_loc_prec(self: MainWindow, ds) -> None:
    if self._post_load_index(ds) is None:
        return
    # A pooled averaged particle has no real traces — its ``tid`` is a per-loc
    # placeholder (each localization its own "trace"), so per-trace precision is
    # meaningless and iterating the N single-point traces is pure wasted work.
    if ds.metadata.get("particle_average"):
        self._post_load_next(ds, self._post_load_density)
        return
    data_prefs = self._state.prefs.get("data", {})
    if data_prefs.get("compute_loc_prec", True) and "sigma_per_trace_nm" not in ds.derived:
        try:
            import time

            import numpy as np
            self._state.log(f"Computing localization precision of '{ds.name}'…")
            self._state.status_progress(f"Computing localization precision of '{ds.name}'")
            t0 = time.perf_counter()
            from ..analysis.localization_precision import stddev_per_trace
            from ..core.loader import attr_values_1d
            loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
            loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
            _z = attr_values_1d(ds, "loc_z")
            loc_z = np.zeros_like(loc_x) if _z is None else np.asarray(_z)
            _tid = attr_values_1d(ds, "tid")
            tid = np.arange(loc_x.size) if _tid is None else np.asarray(_tid)
            result = stddev_per_trace(np.column_stack([loc_x, loc_y, loc_z]), tid)
            dt = time.perf_counter() - t0
            ds.attr["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
            ds.attr["sigma_trace_ids"] = result["trace_ids"]
            ds.derived["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
            ds.derived["sigma_trace_ids"] = result["trace_ids"]
            ds.cali.loc_precision = np.asarray(result["median_sigma_xyz"], dtype=float)
            self._state.log(
                f"Computed localization precision for '{ds.name}' using StdDev per trace "
                    f"({dt:.1f}s): median sigma=({result['median_sigma_xyz'][0]:.3g}, "
                    f"{result['median_sigma_xyz'][1]:.3g}, {result['median_sigma_xyz'][2]:.3g}) nm."
            )
        except Exception as exc:
            self._state.log(f"Localization precision computation skipped for '{ds.name}': {exc}", "WARN")
    self._post_load_next(ds, self._post_load_density)


def _post_load_density(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _legacy_post_load_density(self: MainWindow, ds) -> None:
    if self._post_load_index(ds) is None:
        return
    # A pooled averaged particle is a dense superposition centred at the origin.
    # The KD-tree range-count auto→voxel heuristic estimates work from the
    # bounding-box *volume* assuming uniform density; the averaged cloud is the
    # opposite (dense core, box-sized extent), so the estimate can undershoot and
    # run the exact all-core KD-tree at ≈O(N²) for minutes (the reported freeze).
    # Local density of a synthetic super-particle is not a standard measurement
    # anyway — skip on load; the user can compute it on demand.
    if ds.metadata.get("particle_average"):
        self._post_load_next(ds, self._post_load_finalize)
        return
    data_prefs = self._state.prefs.get("data", {})
    if data_prefs.get("compute_local_density", True) and "den" not in ds.attr:
        try:
            import time
            self._state.log(f"Computing local density of '{ds.name}'…")
            self._state.status_progress(f"Computing local density of '{ds.name}'")
            t0 = time.perf_counter()
            from ..analysis.local_density import compute_local_density_for_dataset
            density, method, detail = compute_local_density_for_dataset(ds, self._state.prefs)
            dt = time.perf_counter() - t0
            if "auto fallback" in detail:
                fallback_msg = f"Local density auto fallback for '{ds.name}': {detail}."
                print(fallback_msg)
                self._state.log(fallback_msg, "WARN")
            ds.attr["den"] = density
            ds.attr["local_density"] = density
            ds.derived["den"] = density
            ds.derived["local_density"] = density
            for name in ("den", "local_density"):
                if name not in ds.prop.attr_names:
                    ds.prop.attr_names.append(name)
            self._state.log(
                f"Computed local density for '{ds.name}' using {method} ({detail}, {dt:.1f}s)."
            )
        except Exception as exc:
            self._state.log(f"Local density computation skipped for '{ds.name}': {exc}", "WARN")
    # For all-iteration loads ds.attr is not the last-valid materialization;
    # compute density at the last-valid selection too so mfx_get can
    # broadcast `den` across iteration views (matches a last-valid
    # load of the same file).
    if (
        data_prefs.get("compute_local_density", True)
        and len(ds.components.derived_last)
        and "den" not in ds.components.derived_last
    ):
        try:
            import numpy as np

            from ..analysis.local_density import compute_local_density_for_points
            from ..core.loader import mfx_get
            x = mfx_get(ds, "loc_x", itr="last")
            y = mfx_get(ds, "loc_y", itr="last")
            z = mfx_get(ds, "loc_z", itr="last")
            if x is not None and y is not None:
                if z is None:
                    z = np.zeros_like(np.asarray(x, dtype=float))
                points_nm = np.column_stack([
                    np.asarray(x, dtype=float) * 1e9,
                    np.asarray(y, dtype=float) * 1e9,
                    np.asarray(z, dtype=float) * 1e9 * ds.cali.z_scaling_factor,
                ])
                dims = 3 if ds.prop.num_dim == 3 else 2
                density, method, detail = compute_local_density_for_points(
                    points_nm, self._state.prefs, dimensions=dims,
                )
                if "auto fallback" in detail:
                    fallback_msg = (
                        f"Last-valid local density auto fallback for '{ds.name}': {detail}."
                    )
                    print(fallback_msg)
                    self._state.log(fallback_msg, "WARN")
                ds.components.derived_last["den"] = density
                self._state.log(
                    f"Computed last-valid local density for '{ds.name}' using {method} ({detail})."
                )
        except Exception as exc:
            self._state.log(
                f"Last-valid local density computation skipped for '{ds.name}': {exc}", "WARN",
            )
    self._post_load_next(ds, self._post_load_finalize)


def _post_load_finalize(self: MainWindow, ds) -> None:
    idx = self._post_load_index(ds)
    if idx is None:
        return
    # Persisted filters were evaluated by ``_start_saved_filter_restore``
    # after derived arrays were applied, so finalization is Qt-only UI work.
    self._restore_saved_rois(ds, idx)
    self._show_saved_filters(ds, idx)
    self._state.status_message.emit(f"Finished processing '{ds.name}'.")


def _restore_saved_rois(self: MainWindow, ds, idx: int) -> None:
    """Put a reopened dataset's saved ROIs into the ROI Manager.

        They are *concatenated*: an already-open Manager keeps what it holds and
        gains these, so opening a second processed dataset does not discard the
        first one's regions. Records already present by id are skipped, which is
        what makes re-opening the same file idempotent.
        """
    records = list(ds.metadata.get("minflux_viewer_roi_records") or [])
    if not records:
        return
    from ..core.roi import RoiRecord
    existing = {record.id for record in self._state.rois.records}
    added = 0
    for payload in records:
        if not isinstance(payload, dict):
            continue
        values = dict(payload)
        values.pop("dataset_id", None)
        context = dict(values.get("context") or {})
        context["dataset_idx"] = idx
        values["context"] = context
        try:
            record = RoiRecord(**values)
        except (TypeError, ValueError) as exc:
            self._state.log(
                f"Skipped a saved ROI of '{ds.name}': {exc}", "WARN")
            continue
        if record.id in existing:
            continue
        self._state.rois.add(record)
        existing.add(record.id)
        added += 1
    if not added:
        return
    self._show_roi_manager()
    self._state.log(
        f"Restored {added} ROI(s) of '{ds.name}' into the ROI Manager.",
        dataset_idx=idx)


def _show_saved_filters(self: MainWindow, ds, idx: int) -> None:
    """Open the Filter dialog on a dataset that was saved with filters.

        Showing them is the point: a reopened dataset can be filtered without
        anything on screen saying so. Each row keeps the enabled/disabled state
        it was saved with, so this displays the filter rather than changing it.
        """
    specs = list(ds.state.get("filter_specs") or [])
    if not specs:
        return
    previous = self._state.active_idx
    try:
        self._state.set_active(idx)
        self._show_filter()
    except Exception as exc:                                # noqa: BLE001
        self._state.log(
            f"Could not show filters for '{ds.name}': {exc}", "WARN")
        return
    finally:
        if previous != idx and 0 <= previous < len(self._state.datasets):
            self._state.set_active(previous)
    enabled = sum(1 for spec in specs
                  if isinstance(spec, dict) and spec.get("enabled", True))
    self._state.log(
        f"'{ds.name}' was saved with {len(specs)} filter row(s), "
            f"{enabled} enabled.", dataset_idx=idx)
