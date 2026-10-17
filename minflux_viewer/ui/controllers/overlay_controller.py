"""Overlay controller delegated by MainWindow."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QDialog, QMessageBox

from minflux_viewer.ui.common.dataset_binding import event_index, window_draws_dataset

if TYPE_CHECKING:
    from minflux_viewer.ui.main_window import MainWindow


def _on_overlay_manual_alignment_requested(self: MainWindow, dataset_idx) -> None:
    """Enter the existing channel-row Manual align mode for a dataset."""
    from minflux_viewer.core.overlay import is_multichannel_overlay
    dataset_idx = event_index(self._state, dataset_idx)
    if dataset_idx is None or not (0 <= dataset_idx < len(self._state.datasets)):
        return
    if not is_multichannel_overlay(self._state, dataset_idx):
        self._state.log(
            "Manual alignment requires at least two datasets in the same overlay.",
            "WARN",
            dataset_idx=dataset_idx,
        )
        return
    self._state.set_active(dataset_idx)
    def contains_dataset(window) -> bool:
        return window_draws_dataset(window, self._state, dataset_idx)
    candidates = []
    focused = getattr(self, "_last_active_plot_window", None)
    if focused is not None and contains_dataset(focused):
        candidates.append(focused)
    candidates.extend(
        window
        for window in self._render_windows.values()
        if contains_dataset(window)
    )
    candidates.extend(
        window
        for window in self._scatter_windows.values()
        if contains_dataset(window)
    )
    seen: set[int] = set()
    for window in candidates:
        if id(window) in seen:
            continue
        seen.add(id(window))
        start = getattr(window, "start_manual_alignment_for_dataset", None)
        if callable(start) and start(dataset_idx):
            window.show()
            window.raise_()
            window.activateWindow()
            return
    # No overlay coordinate view is open. Open the standard render view,
    # then enter the same handler used by channel-row right-click.
    window = self._show_render(dataset_idx)
    start = getattr(window, "start_manual_alignment_for_dataset", None)
    if callable(start) and start(dataset_idx):
        window.raise_()
        window.activateWindow()
        return
    self._state.log(
        "Could not start manual alignment for this overlay.",
        "WARN",
        dataset_idx=dataset_idx,
    )


def _split_active_channel_group(self: MainWindow) -> None:
    """
        Split the active multi-channel render overlay into one render per dataset.

        The dataset objects, transforms, metadata, dataset manager entries, and
        data-info behaviour are left intact. Only the shared render-group key is
        removed so future Render windows no longer auto-compose the channels.
        """
    active_idx = self._state.active_idx
    if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    active = self._state.datasets[active_idx]
    group_id = active.state.get("overlay_id") or active.state.get("render_group_id")
    if not group_id:
        QMessageBox.information(
            self,
            "Split channels",
            "The active dataset is not part of a multi-channel render overlay.",
        )
        return
    group_indices = [
        idx for idx, ds in enumerate(self._state.datasets)
        if (ds.state.get("overlay_id") or ds.state.get("render_group_id")) == group_id
    ]
    if len(group_indices) < 2:
        QMessageBox.information(
            self,
            "Split channels",
            "The active render group contains only one dataset.",
        )
        return
    lut_by_idx = self._current_group_luts(group_indices)
    from minflux_viewer.colors import solid_color_names
    fallback_luts = list(solid_color_names())
    for pos, idx in enumerate(group_indices):
        ds = self._state.datasets[idx]
        ds.state["render_channel_lut"] = lut_by_idx.get(idx, fallback_luts[pos % len(fallback_luts)])
        ds.state.pop("overlay_id", None)
        ds.state.pop("overlay_index", None)
        ds.state.pop("overlay_order", None)
        ds.state.pop("overlay_transform", None)
        ds.state.pop("render_group_id", None)
        ds.state.pop("render_transform_2d", None)
        ds.metadata["render_split_from_group"] = group_id
    for idx in group_indices:
        win = self._render_windows.get(idx)
        if win is not None:
            try:
                win._refresh_from_dataset()
            except Exception:
                pass
    for idx in group_indices:
        self._show_render(idx)
    self._state.set_active(active_idx)
    self._state.log(
        f"Split channel overlay into {len(group_indices)} separate render view(s)."
    )


def _flatten_active_channel_group(self: MainWindow) -> None:
    """Process › Channel › Flatten — combine the active multi-channel overlay
        into a single non-overlay dataset.

        Each channel's localizations are concatenated **in place** (overlay
        transform baked into display nm), trace ids are remapped to stay distinct,
        and combinable per-loc attributes are merged; conflicting / stale ones are
        dropped and reported to the Log. The result renders with the ``hot`` LUT
        and can feed convolution / particle analysis on the combined cloud."""
    import uuid

    from minflux_viewer.core.channel_flatten import flatten_overlay
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.overlay import dataset_group_id, overlay_members
    active_idx = self._state.active_idx
    if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    active = self._state.datasets[active_idx]
    if not dataset_group_id(active):
        QMessageBox.information(
            self, "Flatten channels",
            "The active dataset is not part of a multi-channel overlay.")
        return
    members = overlay_members(self._state, active_idx)
    if len(members) < 2:
        QMessageBox.information(
            self, "Flatten channels",
            "The active overlay contains only one dataset.")
        return
    datasets = [ds for _, ds in members]
    res = flatten_overlay(datasets)
    if res.coords_nm.shape[0] == 0:
        QMessageBox.information(
            self, "Flatten channels",
            "No localizations to flatten (all filtered out?).")
        return
    base = getattr(active, "name", "overlay")
    ds = build_localization_dataset(
        name=f"{base} (flattened {len(members)}ch)",
        x_nm=res.coords_nm[:, 0], y_nm=res.coords_nm[:, 1], z_nm=res.coords_nm[:, 2],
        tid=res.tid, attrs=res.attrs, source_version="flattened",
        prefs=self._state.prefs)
    ds.file.folder = f"<flattened>/{uuid.uuid4().hex}"
    ds.metadata["flattened_overlay"] = True
    ds.metadata["flattened_channels"] = [getattr(d, "name", "") for d in datasets]
    ds.state["render_channel_lut"] = "hot"     # single-channel render LUT
    # Coordinates are final display nm (each channel's Z scaling factor already baked) —
    # pin Z scaling factor to 1.0 and suppress the post-load auto-z estimate.
    try:
        ds.set_z_scaling_factor(1.0, source="flattened (z baked)")
        ds.derived["z_scaling_factor"] = 1.0
    except Exception:
        pass
    r = res.report
    self._state.log(
        f"Flatten channels: combined {r.n_channels} channel(s) → {r.n_kept:,} of "
            f"{r.n_total:,} localization(s), {r.n_traces:,} trace(s). "
            f"Kept attributes: {', '.join(r.kept_attrs) or 'none'}.")
    for note in r.notes:
        self._state.log(f"Flatten: {note}.")
    for attr, reason in r.dropped:
        self._state.log(f"Flatten: dropped '{attr}' — {reason}.", "WARN")
    idx = self._state.add_dataset(ds)
    self._show_render(idx)


def _revert_overlay_to_original(self: MainWindow) -> None:
    """Process › Channel › Convert Dataset to Multi-Channel Overlay › Revert
        Overlay to Original Dataset — the inverse of a channel separation.

        Removes the active overlay's channel datasets and restores the single
        source dataset. When the source is still open (separation keeps it) it is
        re-activated exactly; otherwise it is reconstructed by concatenating the
        channels (:func:`analysis.attribute_channels.reconstruct_from_channels`).
        Only offered for separation overlays (DCR / time / by-attribute); an MSR
        multi-channel overlay has no single original — use Flatten instead."""
    from minflux_viewer.core.overlay import dataset_group_id, overlay_members
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    active = self._state.datasets[idx]
    group_id = dataset_group_id(active)
    if not group_id:
        QMessageBox.information(
            self, "Revert Overlay",
            "The active dataset is not part of a multi-channel overlay.")
        return
    members = overlay_members(self._state, idx)
    member_objs = [ds for _, ds in members]
    if len(member_objs) < 2:
        QMessageBox.information(
            self, "Revert Overlay", "The active overlay contains only one dataset.")
        return
    if not all(m.metadata.get("separated_from") for m in member_objs):
        QMessageBox.information(
            self, "Revert Overlay",
            "This overlay was not created by a channel separation, so it has no "
                "single original dataset to revert to. Use Flatten to combine it.")
        return
    base_name = member_objs[0].metadata.get("separated_from") or "dataset"
    index_of = self._state.index_of_dataset
    member_keys = {self._state.key_of(m) for m in member_objs}
    def is_member(obj) -> bool:
        return self._state.key_of(obj) in member_keys
    # Prefer the still-present source (exact restore); else reconstruct.
    source = None
    for d in self._state.datasets:
        if is_member(d):
            continue
        if group_id in (d.metadata.get("produced_overlays") or []):
            source = d
            break
    if source is None:                          # name-match fallback
        for d in self._state.datasets:
            if is_member(d) or dataset_group_id(d):
                continue
            if getattr(d, "name", None) == base_name:
                source = d
                break
    reconstructed = None
    if source is None:
        from minflux_viewer.analysis.attribute_channels import reconstruct_from_channels
        from minflux_viewer.core.dataset import build_localization_dataset
        rec = reconstruct_from_channels(member_objs)
        if not rec or rec.get("x_nm") is None or rec["x_nm"].size == 0:
            QMessageBox.warning(
                self, "Revert Overlay",
                "Could not reconstruct the original dataset from the channels.")
            return
        reconstructed = build_localization_dataset(
            name=base_name, x_nm=rec["x_nm"], y_nm=rec["y_nm"], z_nm=rec["z_nm"],
            tid=rec["tid"], attrs=rec["attrs"], source_version="reverted",
            prefs=self._state.prefs)
        try:
            reconstructed.set_z_scaling_factor(1.0, source="reverted (z baked)")
            reconstructed.derived["z_scaling_factor"] = 1.0
        except Exception:
            pass
        reconstructed.metadata["reverted_from_overlay"] = group_id
    # Remove the channel datasets (descending index so lower ones stay valid).
    member_idxs = sorted(
        (i for i in (index_of(m) for m in member_objs) if i is not None), reverse=True)
    previous = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    try:
        for i in member_idxs:
            self._state.remove_dataset(i)
    finally:
        self._state.suspend_auto_render = previous
    if source is not None and index_of(source) is not None:
        try:
            (source.metadata.get("produced_overlays") or []).remove(group_id)
        except ValueError:
            pass
        si = index_of(source)
        if si is not None:
            self._state.set_active(si)
            self._show_render(si)
        self._state.log(
            f"Reverted overlay to original dataset '{source.name}' "
                f"({len(member_objs)} channel(s) removed).")
    else:
        new_idx = self._state.add_dataset(reconstructed)
        self._state.set_active(new_idx)
        self._show_render(new_idx)
        self._state.log(
            f"Reverted overlay by reconstructing '{base_name}' from "
                f"{len(member_objs)} channel(s).")
    self._notify_view_state_changed()


def _aggregate_active_dataset(self: MainWindow) -> None:
    """Process › Aggregate Localizations — Imspector-style per-trace photon
        binning of the active dataset into a new aggregated dataset.

        Within each trace, valid final localizations are accumulated in time order
        until their combined photon count (``eco`` over the final-scale iterations)
        reaches the threshold, then emitted as one averaged localization. The
        original dataset is kept; the result is a new ``(aggregated)`` dataset.
        Reproduces Abberior Imspector's *Aggregation* to ≈99 %."""
    import uuid

    from minflux_viewer.analysis.aggregation import raw_dict_from_dataset
    from minflux_viewer.core.dataset_kind import is_minflux
    from minflux_viewer.core.mfx_sequence import photon_iterations_for_dataset
    from minflux_viewer.core.overlay import dataset_group_id, overlay_members
    active_idx = self._state.active_idx
    if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    ds = self._state.datasets[active_idx]
    if not is_minflux(ds):
        QMessageBox.information(
            self, "Aggregate Localizations",
            "The active dataset is not a MINFLUX dataset (no trace ids).")
        return
    if raw_dict_from_dataset(ds) is None:
        QMessageBox.information(
            self, "Aggregate Localizations",
            "Aggregation needs raw all-iteration MINFLUX data with trace, "
                "time, photon and coordinate fields. Load the original .msr / "
                "raw .mat and try again.")
        return
    # Overlay → aggregate every channel with the same threshold and keep them
    # linked as a new overlay; single dataset → just that one.
    group_id = dataset_group_id(ds)
    if group_id:
        members = [m for _, m in overlay_members(self._state, active_idx)]
    else:
        members = [ds]
    aggregatable = [m for m in members
                    if is_minflux(m) and raw_dict_from_dataset(m) is not None]
    if not aggregatable:
        QMessageBox.information(
            self, "Aggregate Localizations",
            "No channel in this overlay has the raw data aggregation needs.")
        return
    pit = photon_iterations_for_dataset(ds)
    thr = self._ask_aggregation_threshold(ds, pit, n_channels=len(aggregatable))
    if thr is None:
        return
    self._state.status_progress("Aggregating localizations")
    if not group_id:
        new = self._build_aggregated_dataset(ds, thr)
        if new is None:
            self._state.status_progress("Aggregation produced no localizations")
            QMessageBox.information(self, "Aggregate Localizations",
                                    "No valid localizations to aggregate.")
            return
        idx = self._state.add_dataset(new)
        self._log_aggregated_dataset(new, idx)
        self._state.status_progress("Aggregation done", 1.0)
        self._show_render(idx)
        return
    # Multi-channel overlay: aggregate each channel, re-link as a new overlay
    # (preserving each channel's LUT + alignment transform).
    new_overlay_id = f"agg:{uuid.uuid4().hex}"
    overlay_index = int(getattr(self, "_next_overlay_index", 1))
    self._next_overlay_index = overlay_index + 1
    prev_suspend = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    made: list[int] = []
    skipped: list[str] = []
    try:
        for order, m in enumerate(members, start=1):
            if m not in aggregatable:
                skipped.append(getattr(m, "name", "?"))
                continue
            new = self._build_aggregated_dataset(m, thr)
            if new is None:
                skipped.append(getattr(m, "name", "?"))
                continue
            lut = m.state.get("overlay_lut") or m.state.get("render_channel_lut")
            new.state.update({
                "overlay_id": new_overlay_id, "render_group_id": new_overlay_id,
                "overlay_index": overlay_index,
                "overlay_order": int(m.state.get("overlay_order", order)),
            })
            if lut:
                new.state["overlay_lut"] = lut
                new.state["render_channel_lut"] = lut
            # Carry the channel's alignment (applied at display time) so the
            # aggregated channels stay registered exactly as before.
            for tkey in ("overlay_transform", "render_transform_2d"):
                tval = m.state.get(tkey)
                if tval is not None:
                    new.state[tkey] = tval
                    new.metadata[tkey] = tval
            new.metadata["overlay_id"] = new_overlay_id
            new_idx = self._state.add_dataset(new)
            made.append(new_idx)
            self._log_aggregated_dataset(new, new_idx)
    finally:
        self._state.suspend_auto_render = prev_suspend
    if not made:
        self._state.status_progress("Aggregation produced no localizations")
        QMessageBox.information(self, "Aggregate Localizations",
                                "No valid localizations to aggregate.")
        return
    if skipped:
        self._state.log(
            f"Aggregation: skipped {len(skipped)} channel(s) without usable "
                f"raw data: {', '.join(skipped)}.", "WARN")
    self._state.log(
        f"Aggregated overlay: {len(made)} channel(s) at photon threshold "
            f"{int(thr)} → new overlay.")
    self._state.status_progress("Aggregation done", 1.0)
    self._show_render(made[0])


def _build_aggregated_dataset(self, ds, thr):
    """Build (not add) an aggregated dataset from *ds* at photon threshold
        *thr*, with the Z scaling factor + aggregation provenance but no overlay state.
        Returns the new dataset, or ``None`` when nothing aggregates."""
    import uuid

    import numpy as np

    from minflux_viewer.analysis.aggregation import aggregate_dataset, aggregation_time_mode
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.mfx_sequence import photon_iterations_for_dataset
    pit = photon_iterations_for_dataset(ds)
    time_mode = aggregation_time_mode(ds)
    res = aggregate_dataset(ds, photon_threshold=float(thr),
                            photon_iters=pit.photon_iters,
                            time_mode=time_mode)
    if res is None or res["tid"].size == 0:
        return None
    loc = res["loc"]                       # metres, (M, 3) — raw (un-Z scaling factor) z
    n_contributing = int(np.asarray(res["n"], dtype=np.int64).sum())
    attrs = {"eco": res["eco"], "n_agg": res["n"].astype(float)}
    for key in ("ecc", "efo", "efc", "fbg", "dcr"):
        if key in res:
            attrs[key] = res[key]
    source_z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
    new = build_localization_dataset(
        name=f"{ds.name} (aggregated {int(thr)})",
        x_nm=loc[:, 0] * 1e9, y_nm=loc[:, 1] * 1e9, z_nm=loc[:, 2] * 1e9,
        tid=res["tid"], tim=res["tim"], attrs=attrs,
        source_version="aggregated", prefs=self._state.prefs)
    try:
        new.set_z_scaling_factor(source_z_scaling_factor, source="aggregated (from raw z)")
        new.derived["z_scaling_factor"] = np.asarray([source_z_scaling_factor], dtype=float)
    except Exception:
        pass
    new.file.folder = f"<aggregated>/{uuid.uuid4().hex}"
    new.metadata["aggregated"] = True
    new.metadata["aggregation"] = {
        "photon_threshold": float(thr),
        "photon_iterations": list(pit.photon_iters),
        "photon_iter_source": pit.source,
        "photon_attribute": "eco",
        "coordinate_mode": "photon_weighted_centroid",
        "timestamp_mode": time_mode,
        "grouping": "per_trace_time_order",
        "valid_final_only": True,
        "trailing_remainder": "retained",
        "source": ds.name,
        "n_in": n_contributing,
        "n_out": int(res["tid"].size),
    }
    return new


def _log_aggregated_dataset(self: MainWindow, ds, dataset_idx: int) -> None:
    """Record a parseable, result-tagged aggregation event in the Log."""
    agg = ds.metadata.get("aggregation", {})
    threshold = float(agg.get("photon_threshold", 0.0))
    threshold_text = f"{threshold:g}"
    self._state.log(
        f"Aggregated '{agg.get('source', 'the source dataset')}' into '{ds.name}': "
            f"{int(agg.get('n_in', 0)):,} -> {int(agg.get('n_out', 0)):,} "
            f"localizations; photon threshold = {threshold_text} photons per "
            f"aggregated localization; photon iterations = "
            f"{list(agg.get('photon_iterations', []))} "
            f"({agg.get('photon_iter_source', 'unknown')}); position = "
            f"photon-weighted centroid; timestamp mode = "
            f"{agg.get('timestamp_mode', 'photon_weighted')}; valid final "
            f"localizations grouped per trace in time order; trailing remainder retained.",
        dataset_idx=dataset_idx,
    )


def _ask_aggregation_threshold(self: MainWindow, ds, pit, n_channels: int = 1) -> int | None:
    """Modal photon-threshold picker for aggregation (default 3000)."""
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QSpinBox, QVBoxLayout

    from minflux_viewer.analysis.aggregation import aggregation_time_mode
    dlg = QDialog(self)
    dlg.setWindowTitle("Aggregate Localizations")
    dlg.setMinimumWidth(620)
    lay = QVBoxLayout(dlg)
    ax = "" if pit.axial_iter is None else f", axial itr {pit.axial_iter}"
    time_mode = aggregation_time_mode(ds)
    if time_mode == "first":
        time_text = (
            "The aggregate timestamp is taken from the first contributing "
                "localization (legacy nested-record convention)."
        )
    else:
        time_text = (
            "The aggregate timestamp is the photon-weighted mean of the "
                "contributing localization timestamps (modern flat-record convention)."
        )
    overlay_note = (f"<i>Overlay: all {n_channels} channels will be aggregated "
                        f"and kept as an overlay.</i><br>" if n_channels > 1 else "")
    info = QLabel(
        f"<b>{ds.name}</b><br>{overlay_note}{ds.prop.num_loc:,} localizations · "
            f"{ds.prop.num_traces:,} traces · {ds.prop.num_dim}-D<br>"
            f"Per-localization photon count <i>P</i> is the sum of background-corrected "
            f"effective counts (<code>eco</code>) over final-scale iterations "
            f"{pit.photon_iters} (lateral itr {pit.lateral_iter}{ax}; "
            f"{pit.source}; 0-based raw iteration indices).<br><br>"
            "Valid final localizations are processed separately for each trace and "
            "ordered by time. Consecutive localizations are accumulated until their "
            "combined photon count reaches or exceeds the threshold below. Complete "
            "localizations are not split, so a bin can exceed the threshold; the final "
            "sub-threshold remainder of each trace is retained.<br><br>"
            "The resulting coordinate is the photon-weighted centroid "
            "Σ(<i>P</i><sub>i</sub> · <i>r</i><sub>i</sub>) / "
            "Σ<i>P</i><sub>i</sub>, and its photon count is Σ<i>P</i><sub>i</sub>. "
            f"{time_text} The source dataset remains unchanged.")
    info.setWordWrap(True)
    lay.addWidget(info)
    row = QLabel("Photon threshold per aggregated localization:")
    lay.addWidget(row)
    spin = QSpinBox()
    spin.setRange(1, 10_000_000)
    spin.setSingleStep(500)
    spin.setSuffix(" photons")
    # Remember the last-used threshold across sessions (default 3000).
    try:
        last = int(self._state.prefs.get("aggregation", {}).get(
            "photon_threshold", 3000))
    except Exception:
        last = 3000
    spin.setValue(max(1, min(10_000_000, last)))
    lay.addWidget(spin)
    bb = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    value = int(spin.value())
    try:
        self._state.prefs.setdefault("aggregation", {})["photon_threshold"] = value
        self._state.save_prefs()
    except Exception:
        pass
    return value


def _current_group_luts(self: MainWindow, group_indices: list[int]) -> dict[int, str]:
    """Return current render LUTs for group members, if a render is open."""
    luts: dict[int, str] = {}
    wanted = set(group_indices)
    for win in list(self._render_windows.values()):
        try:
            channels = getattr(win, "_channels", [])
        except RuntimeError:
            continue
        for ch in channels:
            key = ch.get("dataset_key") if isinstance(ch, dict) else None
            idx = None if key is None else self._state.index_of(key)
            if idx in wanted and idx not in luts:
                luts[idx] = str(ch.get("lut") or "")
    return {idx: lut for idx, lut in luts.items() if lut}


def _duplicate_dataset_for_overlay(self: MainWindow, src_idx: int, timestamp: str) -> int:
    import copy

    from minflux_viewer.core.dataset import FileInfo, MinfluxDataset
    src = self._state.datasets[src_idx]
    new_attrs = copy.deepcopy(src.attr)
    new_components = copy.deepcopy(src.components)
    if getattr(new_components, "mfx", None) is not None:
        new_components.mfx.attrs = new_attrs
    dup = MinfluxDataset(
        file=FileInfo(
            name=self._next_duplicate_name(src.file.name),
            folder=src.file.folder,
            datetime=timestamp,
            raw_data=None,
            recent_path=src.file.recent_path,
        ),
        prop=copy.deepcopy(src.prop),
        attr=new_attrs,
        cali=copy.deepcopy(src.cali),
        channel=copy.deepcopy(src.channel),
        components=new_components,
    )
    dup.file.datetime = timestamp
    dup.metadata.update(copy.deepcopy(src.metadata))
    dup.state.update(copy.deepcopy(src.state))
    for key in (
        "overlay_id", "overlay_index", "overlay_order", "overlay_transform",
        "render_group_id", "render_transform_2d",
    ):
        dup.state.pop(key, None)
        dup.metadata.pop(key, None)
    dup.metadata["duplicated_from_dataset"] = src.file.name
    dup.metadata["created_note"] = f"{timestamp}, duplicated for channel overlay."
    return self._state.add_dataset(dup)


def _show_channel_combine(self: MainWindow, dataset_indices=None) -> None:
    # QAction.triggered passes checked=False; only a real sequence restricts
    # the dialog to a subset (the Dataset Manager's multi-selection).
    restrict = (
        list(dataset_indices)
        if isinstance(dataset_indices, (list, tuple, set))
        else None
    )
    if restrict is not None and len(restrict) < 2:
        QMessageBox.information(self, "Combine datasets", "Select at least two datasets to combine.")
        return
    if len(self._state.datasets) < 2:
        QMessageBox.information(self, "Combine datasets", "Load at least two datasets before combining channels.")
        return
    import uuid

    from minflux_viewer.core.overlay import OverlayMemberSpec, build_overlay_transforms
    from minflux_viewer.ui.analysis.channel_combine_dialog import ChannelCombineDialog
    dlg = ChannelCombineDialog(
        self._state,
        previous=self._last_channel_combine_settings,
        dataset_indices=restrict,
        parent=self,
    )
    result = dlg.exec()
    # A restricted run sees only part of the list, so its per-dataset
    # order/LUT choices would be a partial memory — don't overwrite the
    # settings the full dialog remembers.
    if restrict is None:
        self._last_channel_combine_settings = dlg.session_state()
    if result != QDialog.DialogCode.Accepted:
        return
    rows = dlg.selected_rows()
    if len(rows) < 2:
        QMessageBox.warning(self, "Combine datasets", "Select at least two datasets to combine.")
        return
    timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
    keep_source = dlg.keep_source_check.isChecked()
    previous_suspend = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    try:
        overlay_rows = []
        for row in rows:
            idx = int(row["dataset_idx"])
            if keep_source:
                idx = self._duplicate_dataset_for_overlay(idx, timestamp)
            overlay_rows.append({"dataset_idx": idx, "order": int(row["order"]), "lut": row["lut"]})
    finally:
        self._state.suspend_auto_render = previous_suspend
    overlay_index = self._next_overlay_index
    self._next_overlay_index += 1
    overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
    specs = [
        OverlayMemberSpec(dataset_idx=row["dataset_idx"], order=pos + 1, lut=row["lut"])
        for pos, row in enumerate(sorted(overlay_rows, key=lambda item: (item["order"], item["dataset_idx"])))
    ]
    transforms = build_overlay_transforms(
        state=self._state,
        members=specs,
        overlay_id=overlay_id,
        overlay_index=overlay_index,
        alignment_mode=dlg.align_combo.currentText(),
    )
    for spec in specs:
        ds = self._state.datasets[spec.dataset_idx]
        transform = transforms[spec.dataset_idx]
        ds.state["overlay_id"] = overlay_id
        ds.state["render_group_id"] = overlay_id
        ds.state["overlay_index"] = overlay_index
        ds.state["overlay_order"] = spec.order
        ds.state["overlay_lut"] = spec.lut
        ds.state["render_channel_lut"] = spec.lut
        ds.state["overlay_transform"] = transform
        ds.state["render_transform_2d"] = transform
        ds.metadata["overlay_id"] = overlay_id
        ds.metadata["overlay_index"] = overlay_index
        ds.metadata["overlay_alignment_mode"] = dlg.align_combo.currentText()
    anchor_idx = specs[0].dataset_idx
    had_scatter = self._collapse_member_coordinate_views(
        [spec.dataset_idx for spec in specs], anchor_idx)
    self._state.set_active(anchor_idx)
    self._show_render(anchor_idx)
    if had_scatter:
        self._show_scatter(anchor_idx)
    # Any other open overlay view (e.g. one keyed by a dataset that was
    # already a channel) re-reads its channel list.
    self._refresh_overlay_windows()
    self._notify_view_state_changed()
    self._state.log(f"Created overlay {overlay_index} with {len(specs)} dataset(s).")


def _collapse_member_coordinate_views(self: MainWindow, member_indices, anchor_idx: int) -> bool:
    """Fold the members' standalone render/scatter windows into the anchor's.

        Render and scatter are **overlay-aware**: one window, keyed by the anchor,
        draws every channel. So once datasets are combined in place (*keep source
        dataset* unchecked) the other members' own windows are stale duplicates —
        each still titled and drawn as a standalone dataset, i.e. exactly the
        "Own" views the Dataset Manager no longer lists. They are closed here;
        nothing folds them by itself, which is why one was always left behind.

        Only the coordinate views are folded. A histogram, attribute plot or data
        window of a single channel stays meaningful and is left open, and **no
        dataset is closed** — combining changes how they are viewed, not what
        exists. Returns whether any member had a scatter window, so the caller
        can reopen it on the anchor rather than leaving the user with none.
        """
    wanted_scatter = anchor_idx in self._scatter_windows
    for idx in member_indices:
        if idx == anchor_idx:
            continue
        for registry in (self._render_windows, self._scatter_windows):
            win = registry.pop(idx, None)
            if win is None:
                continue
            if registry is self._scatter_windows:
                wanted_scatter = True
            try:
                win.close()
            except Exception:
                pass
    return wanted_scatter


def _show_channel_separation(self: MainWindow) -> None:
    """Process › Channel… › Separate Channel by DCR — open the DCR two-color
        separation tool for the active dataset."""
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    ds = self._state.datasets[idx]
    from minflux_viewer.core.loader_raw_access import attr_values_1d
    if attr_values_1d(ds, "dcr") is None:
        QMessageBox.information(
            self, "Separate Channel by DCR",
            "The active dataset has no DCR attribute, so it cannot be separated by DCR.")
        return
    from minflux_viewer.ui.analysis.attribute_separation_dialog import AttributeSeparationDialog
    from minflux_viewer.ui.common.modeless import show_modeless
    win = AttributeSeparationDialog(
        self._state, idx, attribute="dcr",
        title="Separate Channel by DCR", owner=self)
    show_modeless(win, self)


def _show_attribute_separation(self: MainWindow) -> None:
    """Process › Channel › Convert Dataset to Multi-Channel Overlay › by MINFLUX
        data attribute — the generic attribute-based channel separation. Same
        dialog as by-DCR, but with an attribute picker (DCR is one instance)."""
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    ds = self._state.datasets[idx]
    from minflux_viewer.core.loader_raw_access import attr_values_1d
    default = "dcr" if attr_values_1d(ds, "dcr") is not None else "efo"
    from minflux_viewer.ui.analysis.attribute_separation_dialog import AttributeSeparationDialog
    from minflux_viewer.ui.common.modeless import show_modeless
    win = AttributeSeparationDialog(
        self._state, idx, attribute=default,
        title="Convert to Multi-Channel Overlay (by attribute)",
        pick_attribute=True, owner=self)
    show_modeless(win, self)


def _show_time_channel_separation(self: MainWindow) -> None:
    """Open the time-window editor for the active dataset."""
    import numpy as np

    from minflux_viewer.core.loader_raw_access import attr_values_1d
    from minflux_viewer.core.overlay import overlay_color_cycle
    from minflux_viewer.ui.analysis.time_channel_dialog import TimeChannelDialog
    idx = self._state.active_idx
    if idx is None or not (0 <= idx < len(self._state.datasets)):
        self._no_data_warning()
        return
    source = self._state.datasets[idx]
    tim = attr_values_1d(source, "tim")
    if tim is None:
        QMessageBox.information(
            self,
            "Separate Channels from Time Windows",
            "The active dataset has no tim attribute.",
        )
        return
    tim = np.asarray(tim, dtype=float).ravel()
    base_mask = np.asarray(source.filter_mask, dtype=bool).ravel()
    if tim.size != int(source.prop.num_loc) or base_mask.size != tim.size:
        QMessageBox.warning(
            self,
            "Separate Channels from Time Windows",
            "The tim attribute does not align with the active localization table.",
        )
        return
    active_tim = tim[base_mask & np.isfinite(tim)]
    if active_tim.size < 2 or float(np.max(active_tim)) <= float(np.min(active_tim)):
        QMessageBox.information(
            self,
            "Separate Channels from Time Windows",
            "The active localizations do not span a usable acquisition-time range.",
        )
        return
    try:
        dialog = TimeChannelDialog(
            tim,
            source_name=source.name,
            base_mask=base_mask,
            color_cycle=overlay_color_cycle(self._state.prefs),
            parent=self,
        )
    except ValueError as exc:
        QMessageBox.warning(self, "Separate Channels from Time Windows", str(exc))
        return
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    self.apply_time_channel_separation(idx, dialog.windows())


def apply_time_channel_separation(self: MainWindow, source_idx: int, windows) -> bool:
    """Create filtered full-data clones and group them as a channel overlay."""
    import uuid

    import numpy as np

    from minflux_viewer.core.loader_raw_access import attr_values_1d
    from minflux_viewer.core.overlay import display_transform_record, identity_matrix4
    from minflux_viewer.core.time_channels import (
        clone_time_channel_dataset,
        time_channel_selections,
    )
    if not (0 <= source_idx < len(self._state.datasets)):
        return False
    source = self._state.datasets[source_idx]
    tim = attr_values_1d(source, "tim")
    if tim is None:
        return False
    try:
        selections = time_channel_selections(
            tim,
            windows,
            base_mask=source.filter_mask,
        )
    except ValueError as exc:
        self._state.log(f"Time-window channel separation failed: {exc}", "ERROR")
        return False
    if any(not np.any(selection.mask) for selection in selections):
        self._state.log(
            "Time-window channel separation failed: at least one channel is empty.",
            "ERROR",
        )
        return False
    reserved = {dataset.name.casefold() for dataset in self._state.datasets}
    def unique_name(requested: str) -> str:
        base = requested.strip()
        candidate = base
        number = 2
        while candidate.casefold() in reserved:
            candidate = f"{base} ({number})"
            number += 1
        reserved.add(candidate.casefold())
        return candidate
    timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
    try:
        new_datasets = [
            clone_time_channel_dataset(
                source,
                selection,
                name=unique_name(selection.window.name),
                timestamp=timestamp,
            )
            for selection in selections
        ]
    except Exception as exc:
        self._state.log(f"Time-window channel separation failed: {exc}", "ERROR")
        QMessageBox.critical(
            self, "Separate Channels from Time Windows", str(exc)
        )
        return False
    overlay_index = self._next_overlay_index
    self._next_overlay_index += 1
    overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
    new_indices: list[int] = []
    previous = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    try:
        for order, (selection, dataset) in enumerate(
            zip(selections, new_datasets), start=1
        ):
            idx = self._state.add_dataset(dataset)
            channel = self._state.datasets[idx]
            lut = selection.window.lut
            transform = display_transform_record(
                overlay_id=overlay_id,
                overlay_index=overlay_index,
                order=order,
                lut=lut,
                source_dataset_idx=idx,
                alignment_mode="stage origin",
                matrix_4x4=identity_matrix4(),
                provenance={
                    "method": "time-window channel separation",
                    "source_dataset": source.name,
                },
            )
            channel.state["overlay_id"] = overlay_id
            channel.state["render_group_id"] = overlay_id
            channel.state["overlay_index"] = overlay_index
            channel.state["overlay_order"] = order
            channel.state["overlay_lut"] = lut
            channel.state["render_channel_lut"] = lut
            channel.state["overlay_transform"] = transform
            channel.state["render_transform_2d"] = transform
            channel.metadata["overlay_id"] = overlay_id
            channel.metadata["overlay_index"] = overlay_index
            channel.metadata["overlay_alignment_mode"] = "stage origin"
            channel.metadata["separated_from"] = source.name
            channel.metadata["separated_by"] = "tim"
            new_indices.append(idx)
    finally:
        self._state.suspend_auto_render = previous
    if not new_indices:
        return False
    # Breadcrumb so "Revert Overlay to Original Dataset" can find this source.
    source.metadata.setdefault("produced_overlays", []).append(overlay_id)
    anchor = new_indices[0]
    self._state.set_active(anchor)
    self._show_render(anchor)
    self._notify_view_state_changed()
    summary = ", ".join(
        f"{selection.window.name}: {int(selection.mask.sum()):,}"
        for selection in selections
    )
    self._state.log(
        f"Separated '{source.name}' into {len(new_indices)} time-window "
            f"channels (overlay {overlay_index}): {summary}."
    )
    try:
        self._state.journal.add(
            "transform",
            f"Separated dataset '{source.name}' into {len(new_indices)} "
                "channels using acquisition-time windows.",
            windows=[
                {
                    "name": selection.window.name,
                    "start_s": float(selection.window.start_s),
                    "end_s": float(selection.window.end_s),
                }
                for selection in selections
            ],
        )
    except Exception:
        pass
    return True


def apply_channel_separation(self: MainWindow, src_idx: int, labels, channels, *,
                             attribute: str = "", method_label: str = "channel separation",
                             masks=None, include_unassigned: bool = True,
                             basis: str = "") -> bool:
    """Build one truncated dataset per channel (+ a hidden *unassigned*) and
        combine them as a render overlay. *channels* is a list of carriers exposing
        ``.name`` and ``.lut`` (see :class:`analysis.attribute_channels.Channel`).
        Attribute-agnostic — DCR is one instance. Returns True on success.

        Membership comes either from *labels* — the exclusive per-localization
        label array (0..N-1, -1 = unassigned) — or from *masks*, one boolean mask
        per channel. Masks are the general form: they can **overlap**, which is
        how "keep a contested localization in both channels" is expressed, and a
        label array cannot say that. ``include_unassigned=False`` drops the rows
        no channel claims instead of keeping them as a hidden channel. ``basis``
        names the per-localization value the channel windows were drawn on (e.g.
        "final iterations, photon-weighted"); it is recorded and logged.
        """
    import uuid

    import numpy as np

    from minflux_viewer.core.overlay import display_transform_record, identity_matrix4
    from minflux_viewer.core.roi_crop import subset_dataset
    if not (0 <= src_idx < len(self._state.datasets)):
        return False
    src = self._state.datasets[src_idx]
    n = int(getattr(src.prop, "num_loc", 0) or 0)
    if masks is not None:
        channel_masks = [np.asarray(m, dtype=bool).ravel() for m in masks]
        if len(channel_masks) != len(channels) or any(m.size != n for m in channel_masks):
            return False
    else:
        labels = np.asarray(labels).ravel()
        if labels.size != n:
            return False
        channel_masks = [labels == k for k in range(len(channels))]
    base_name = src.name
    # (mask, LUT, dataset name, hidden-by-default), one per channel + unassigned.
    plan = [
        (channel_masks[k], getattr(ch, "lut", "Gray") or "Gray",
         getattr(ch, "name", None) or f"{base_name} [ch {k + 1}]", False)
        for k, ch in enumerate(channels)
    ]
    if include_unassigned:
        claimed = np.zeros(n, dtype=bool)
        for mask in channel_masks:
            claimed |= mask
        plan.append((~claimed, "Gray", f"{base_name} [unassigned]", True))
    overlay_index = self._next_overlay_index
    self._next_overlay_index += 1
    overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
    prefs = self._state.prefs
    new_indices: list[int] = []
    order = 0
    previous = getattr(self._state, "suspend_auto_render", False)
    self._state.suspend_auto_render = True
    try:
        for mask, lut, name, hidden in plan:
            keep = np.asarray(mask, dtype=bool)
            if not keep.any():
                continue
            order += 1
            new_ds = subset_dataset(src, keep, name=name, prefs=prefs)
            idx = self._state.add_dataset(new_ds)
            ds = self._state.datasets[idx]
            transform = display_transform_record(
                overlay_id=overlay_id, overlay_index=overlay_index, order=order,
                lut=lut, source_dataset_idx=idx, alignment_mode="stage origin",
                matrix_4x4=identity_matrix4(),
                provenance={"method": method_label, "source_dataset": base_name,
                            "attribute": attribute, "basis": basis})
            ds.state["overlay_id"] = overlay_id
            ds.state["render_group_id"] = overlay_id
            ds.state["overlay_index"] = overlay_index
            ds.state["overlay_order"] = order
            ds.state["overlay_lut"] = lut
            ds.state["render_channel_lut"] = lut
            ds.state["overlay_transform"] = transform
            ds.state["render_transform_2d"] = transform
            if hidden:
                ds.state["overlay_default_hidden"] = True
            ds.metadata["overlay_id"] = overlay_id
            ds.metadata["overlay_index"] = overlay_index
            ds.metadata["separated_from"] = base_name
            ds.metadata["separated_by"] = attribute
            new_indices.append(idx)
    finally:
        self._state.suspend_auto_render = previous
    if not new_indices:
        QMessageBox.warning(self, "Separate Channels",
                            "No localizations were assigned to a channel.")
        return False
    # Breadcrumb so "Revert Overlay to Original Dataset" can find this source.
    src.metadata.setdefault("produced_overlays", []).append(overlay_id)
    anchor = new_indices[0]
    self._state.set_active(anchor)
    self._show_render(anchor)
    self._notify_view_state_changed()
    by = attribute or method_label
    if basis:
        by += f" [{basis}]"
    self._state.log(
        f"Separated '{base_name}' into {len(new_indices)} channel(s) by {by} (overlay {overlay_index}).")
    return True
