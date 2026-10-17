"""Post load controller delegated by MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer

from minflux_viewer.ui.common.background_tasks import BackgroundTask
from minflux_viewer.ui.common.dataset_binding import event_index
from minflux_viewer.ui.tools.data_window import DataWindow

if TYPE_CHECKING:
    from minflux_viewer.ui.main_window import MainWindow


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
        from minflux_viewer.core.metadata_match import describe_match
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
        from minflux_viewer.ui.common.modeless import place_beside
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

    from minflux_viewer.analysis.post_load import POST_LOAD_ALGORITHMS
    from minflux_viewer.core.computation_snapshot import computation_snapshot, input_signature
    from minflux_viewer.core.post_load import compute_post_load_results
    tasks = getattr(self, "_post_load_tasks", None)
    if tasks is None:
        tasks = self._post_load_tasks = {}
    # Keyed by the dataset's key, which is never given to another dataset,
    # so closing one retires exactly its own work (_retire_dataset_tasks).
    key = self._state.key_at(idx)
    if key is None or key in tasks:
        return
    prefs = copy.deepcopy(self._state.prefs)
    signature, revision = input_signature(ds), self._state.data_revision(key)
    snapshot = computation_snapshot(ds)
    def work(report):
        return compute_post_load_results(snapshot, prefs, report, algorithms=POST_LOAD_ALGORITHMS)
    task = BackgroundTask(
        work,
        description=f"Post-load processing of {ds.name}",
        category="analysis",
        estimated_bytes=512 * int(ds.prop.num_loc) + 8 * 1024**2,
    )
    tasks[key] = task
    task.signals.stage.connect(
        lambda text: self._state.status_progress(text)
    )
    task.signals.done.connect(
        lambda result: self._on_post_load_done(key, ds, result, task, signature, revision)
    )
    task.signals.failed.connect(
        lambda message: self._on_post_load_error(key, ds, message, task, signature, revision)
    )
    task.signals.finished.connect(
        lambda _key=key, _task=task: self._finish_post_load_task(_key, _task)
    )
    from minflux_viewer.ui.common.background_tasks import shared_thread_pool
    shared_thread_pool("post-load").start(task)


def _finish_post_load_task(self: MainWindow, key, task) -> None:
    from minflux_viewer.ui.common.background_tasks import detach_task_signals
    detach_task_signals(task)
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
    from minflux_viewer.ui.common.background_tasks import retire_background_tasks
    task = getattr(self, "_post_load_tasks", {}).pop(event.key, None)
    if task is not None:
        retire_background_tasks([task])


class _Delivery:
    """Controller-private validation, not another MainWindow capability."""

    @staticmethod
    def current(self, key, ds, task: BackgroundTask | None, signature, revision) -> bool:
        from minflux_viewer.core.computation_snapshot import input_signature
        if getattr(self, "_is_shutting_down", False) or self._state.dataset_for(key) is not ds:
            return False
        if task is None:
            return True  # synchronous/internal caller, not a queued task delivery
        return (not task.is_cancelled and self._state.data_revision(key) == revision
                and input_signature(ds) == signature
                and getattr(self, "_post_load_tasks", {}).get(key) is task)

    @staticmethod
    def report_outdated(self, key, ds, task) -> None:
        if (task is not None and not getattr(self, "_is_shutting_down", False)
                and self._state.dataset_for(key) is ds
                and getattr(self, "_post_load_tasks", {}).get(key) is task):
            message = f"Discarded outdated processing result for '{ds.name}'; newer dataset state was kept."
            self._state.log(message, "INFO")
            self._state.status_message.emit(message)


def _on_post_load_done(self: MainWindow, key, ds, result: dict,
                       task=None, signature=None, revision=None) -> None:
    """Deliver a post-load result only while the dataset it started for is open.

        Checked by the key the task started under: a dataset closed and opened
        again is a new registration, and a result for the old one is not its.
        """
    if not _Delivery.current(self, key, ds, task, signature, revision):
        _Delivery.report_outdated(self, key, ds, task)
        return
    self._apply_post_load_results(ds, result)


def _on_post_load_error(self: MainWindow, key, ds, message: str,
                        task=None, signature=None, revision=None) -> None:
    if not _Delivery.current(self, key, ds, task, signature, revision):
        _Delivery.report_outdated(self, key, ds, task)
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
    from minflux_viewer.core.post_load_publication import publish_post_load
    changes = publish_post_load(ds, result)
    if changes.calibration:
        self._state.notify_calibration_changed(idx)
    for message, level in changes.logs:
        self._state.log(message, level)
    if changes.attributes:
        self._state.notify_attributes_changed(idx)
    self._start_saved_filter_restore(ds)


def _start_saved_filter_restore(self: MainWindow, ds) -> None:
    key = self._state.key_of(ds)
    if key is None or self._is_shutting_down:
        return
    if not ds.state.get("filter_specs"):
        self._post_load_finalize(ds)
        return
    from minflux_viewer.core.computation_snapshot import computation_snapshot, input_signature
    from minflux_viewer.core.loader_raw_access import evaluate_saved_filters
    signature, revision = input_signature(ds), self._state.data_revision(key)
    snapshot = computation_snapshot(ds)
    def work(report):
        report(f"Restoring saved filters for '{snapshot.name}'")
        return evaluate_saved_filters(snapshot)
    task = BackgroundTask(
        work,
        description=f"Restoring filters for {ds.name}",
        category="analysis",
    )
    task.signals.done.connect(
        lambda mask: self._on_saved_filters_done(key, ds, mask, task, signature, revision)
    )
    task.signals.failed.connect(
        lambda message: self._on_saved_filters_error(key, ds, message, task, signature, revision)
    )
    task.signals.finished.connect(
        lambda _key=key, _t=task: self._finish_post_load_task(_key, _t)
    )
    # Registered with the dataset's post-load work, so closing the dataset
    # retires it too (_retire_dataset_tasks).
    tasks = getattr(self, "_post_load_tasks", None)
    if tasks is None:
        tasks = self._post_load_tasks = {}
    previous = tasks.get(key)
    if previous is not None:
        from minflux_viewer.ui.common.background_tasks import retire_background_tasks
        retire_background_tasks([previous])
    tasks[key] = task
    from minflux_viewer.ui.common.background_tasks import shared_thread_pool
    shared_thread_pool("post-load").start(task)


def _on_saved_filters_done(self: MainWindow, key, ds, mask,
                           task=None, signature=None, revision=None) -> None:
    if _Delivery.current(self, key, ds, task, signature, revision):
        self._on_saved_filters_evaluated(ds, mask)
    else:
        _Delivery.report_outdated(self, key, ds, task)


def _on_saved_filters_error(self: MainWindow, key, ds, message: str,
                            task=None, signature=None, revision=None) -> None:
    if _Delivery.current(self, key, ds, task, signature, revision):
        self._on_saved_filter_failed(ds, message)
    else:
        _Delivery.report_outdated(self, key, ds, task)


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


def _restart_post_load_in_background(self: MainWindow, ds) -> None:
    """Compatibility entry point for extensions calling the former steps."""
    idx = self._post_load_index(ds)
    if idx is not None:
        self._run_post_load_computations(idx)


def _post_load_z_scaling_factor(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _post_load_z_scaling_factor_estimate(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _post_load_loc_prec(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


def _post_load_density(self: MainWindow, ds) -> None:
    self._restart_post_load_in_background(ds)


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
    from minflux_viewer.core.roi import RoiRecord
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
