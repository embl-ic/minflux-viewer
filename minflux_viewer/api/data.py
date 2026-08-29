"""
``mfv.data`` -- datasets, attributes, coordinates and filters.

Reading rule that matters
-------------------------
:meth:`Data.attr` resolves values through
``core/loader.py::attr_values_for_selection``, never by indexing ``ds.attr``
directly. That accessor is what makes a script agree with the Filter dialog and
the Histogram window: it resolves ``cfr``/``efc`` to their *effective*
measurement iteration (their value at the global-max iteration is 0/NaN), and
it understands the pooled selectors. Indexing ``ds.attr`` bypasses all of it and
silently returns different numbers than the UI shows for the same attribute.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._base import ApiError, Namespace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.dataset import MinfluxDataset


class Data(Namespace):
    """Read and write the loaded localization data."""

    name = "data"

    # -- finding datasets ----------------------------------------------------

    def datasets(self) -> list[MinfluxDataset]:
        """Every loaded dataset, in viewer order."""
        return list(self._state.datasets)

    def active(self) -> MinfluxDataset | None:
        """The active dataset, or ``None`` when nothing is loaded."""
        return self._state.active_dataset

    def get(self, ref: Any = None) -> MinfluxDataset:
        """
        Resolve *ref* to a dataset: ``None`` = active, an ``int`` = index, a
        ``str`` = name or file name, or a dataset object (returned as-is).
        """
        return self._dataset(ref)

    def index(self, dataset: Any = None) -> int:
        """The viewer index of *dataset* -- the handle other namespaces take."""
        return self._dataset_index(dataset)

    def properties(self, *, dataset: Any = None) -> dict:
        """Summary facts about a dataset, as a plain dict."""
        ds = self._dataset(dataset)
        prop = ds.prop
        return {
            "name": ds.name,
            "index": self._dataset_index(ds),
            "num_loc": int(getattr(prop, "num_loc", 0)),
            "num_traces": int(getattr(prop, "num_traces", 0)),
            "num_itr": int(ds.metadata.get("raw_num_itr", 0) or 0),
            "dimensions": int(getattr(prop, "num_dim", 0)),
            "z_scaling_factor": float(getattr(ds.cali, "z_scaling_factor", 1.0)),
            "source_version": str(ds.metadata.get("source_version", "")),
            "num_filtered": int(np.count_nonzero(np.asarray(ds.filter_mask, dtype=bool))),
        }

    # -- reading -------------------------------------------------------------

    def attr_names(self, *, dataset: Any = None, include_derived: bool = True) -> list[str]:
        """
        Attribute names available for *dataset*.

        Goes through ``core/attributes.py::plot_attribute_names`` so the list
        respects Preferences and matches what the UI dropdowns offer.
        """
        from ..core.attributes import plot_attribute_names

        ds = self._dataset(dataset)
        return list(plot_attribute_names(
            ds, self._state.prefs, include_derived=include_derived,
        ))

    def attr(
        self,
        name: str,
        *,
        dataset: Any = None,
        itr: Any = "auto",
        vld_only: bool = True,
        filtered: bool = True,
    ) -> np.ndarray:
        """
        One attribute as a 1-D array, one value per localization.

        ``itr`` selects the iteration with the application's own tokens:
        ``"auto"`` (the default view -- cfr/efc at their effective iteration,
        everything else last-valid), ``"last"``, ``"effective"``, an ``int`` for
        a specific 0-based iteration, or ``"sum"``/``"average"`` to pool a
        localization's iterations into one value.

        ``vld_only=False`` reads through the raw store including invalid rows,
        which does **not** align to the materialized rows -- ``filtered`` is
        then ignored, and so is the one-value-per-localization guarantee.
        """
        from ..core.loader import attr_values_for_selection, mfx_get

        if not name:
            raise ApiError("An attribute name is required.")
        ds = self._dataset(dataset)

        if not vld_only:
            values = mfx_get(ds, name, itr=itr if itr != "auto" else "last",
                             vld_only=False)
            if values is None:
                raise ApiError(
                    f"Attribute {name!r} is not available in the raw store of "
                    f"{ds.name!r}."
                )
            return np.asarray(values).ravel()

        values = attr_values_for_selection(ds, name, itr=itr)
        if values is None:
            raise ApiError(
                f"Attribute {name!r} has no per-localization values in "
                f"{ds.name!r}. Available: {', '.join(self.attr_names(dataset=ds))}"
            )
        values = np.asarray(values).ravel()
        return self._apply_filter(ds, values) if filtered else values

    def loc(
        self,
        *,
        dataset: Any = None,
        unit: str = "nm",
        filtered: bool = True,
        transformed: bool = False,
    ) -> np.ndarray:
        """
        Localization coordinates as ``(N, 3)``.

        ``unit`` is ``"nm"`` (display units, Z scaling factor applied) or
        ``"m"`` (raw metres). ``transformed`` additionally applies the
        dataset's overlay transform, giving the coordinates the render and
        scatter views actually draw -- which is the frame ROIs live in.
        """
        ds = self._dataset(dataset)
        if transformed:
            from ..core.roi_crop import display_coords

            loc = np.asarray(display_coords(ds), dtype=float)
        else:
            loc = np.asarray(ds.loc_nm, dtype=float)
        if unit == "m":
            loc = loc / 1e9
        elif unit != "nm":
            raise ApiError("unit must be 'nm' or 'm'.")
        return self._apply_filter(ds, loc) if filtered else loc

    # -- writing -------------------------------------------------------------

    def add_attr(
        self,
        name: str,
        values: Any,
        *,
        dataset: Any = None,
        description: str = "",
    ) -> None:
        """
        Attach a computed per-localization attribute to a dataset.

        *values* must have one entry per localization of the **unfiltered**
        dataset. Afterwards the attribute is selectable in the histogram,
        filter, colour-by and attribute-plot lists.

        It is registered the way the application registers its own computed
        per-localization attributes (see
        ``core/confocal_mapping.py::attach_confocal_signal``): stored through
        ``mfx.set_attr`` with ``user_visible`` metadata, and appended to
        ``prop.attr_names``. That flag is what
        ``core/attributes.py::plot_attribute_names`` honours to list an
        attribute the Preferences enabled/computed tables have never heard of --
        which every script-created attribute is by definition. Writing only to
        ``derived`` would store the values correctly and leave them invisible
        in every dropdown.
        """
        ds = self._dataset(dataset)
        arr = np.asarray(values).ravel()
        expected = int(ds.prop.num_loc)
        if arr.size != expected:
            raise ApiError(
                f"add_attr('{name}') needs one value per localization: got "
                f"{arr.size}, dataset {ds.name!r} has {expected}. "
                "Values must cover the unfiltered dataset."
            )
        ds.mfx.set_attr(name, arr, meta={
            "component": "mfx",
            "source": "script",
            "description": str(description or f"{name}, added by a script."),
            "user_visible": True,
        })
        if name not in ds.prop.attr_names:
            ds.prop.attr_names.append(name)
        # Also kept in ``derived`` so code that reads computed values there --
        # and ``mfv.data.attr(source="derived")`` -- finds it.
        ds.derived[name] = arr
        ds.metadata.setdefault("script_attributes", {})[name] = str(description or "")
        self._state.notify_attributes_changed(self._dataset_index(ds))

    def create(
        self,
        xyz: Any,
        *,
        tid: Any = None,
        attrs: Mapping[str, Any] | None = None,
        name: str = "script dataset",
        unit: str = "nm",
        add: bool = True,
    ) -> MinfluxDataset:
        """
        Build a new dataset from computed coordinates and add it to the viewer.

        Goes through ``build_localization_dataset``, then pins the Z scaling
        factor to 1.0 and sets ``derived["z_scaling_factor"]`` so the post-load
        trace-anisotropy estimate is suppressed -- the coordinates are already
        final and must not be corrected a second time. This is the established
        pattern for computed datasets (drift correction, particle average,
        channel flatten).

        ``tid=None`` gives every localization its own synthetic trace.
        """
        from ..core.dataset import build_localization_dataset

        pts = np.asarray(xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ApiError("create() needs an (N, 2) or (N, 3) coordinate array.")
        if unit == "m":
            pts = pts * 1e9
        elif unit != "nm":
            raise ApiError("unit must be 'nm' or 'm'.")

        z = pts[:, 2] if pts.shape[1] == 3 else None
        ds = build_localization_dataset(
            name=str(name),
            x_nm=pts[:, 0],
            y_nm=pts[:, 1],
            z_nm=z,
            attrs={k: np.asarray(v).ravel() for k, v in (attrs or {}).items()},
            tid=None if tid is None else np.asarray(tid).ravel(),
            source_version="script",
            prefs=self._state.prefs,
        )
        # Coordinates are final: pin the factor and record it so the post-load
        # chain does not estimate one and scale z a second time.
        ds.set_z_scaling_factor(1.0, source="script (coordinates already final)")
        ds.derived["z_scaling_factor"] = 1.0
        ds.metadata["created_by_script"] = True
        if add:
            self._state.add_dataset(ds)
        return ds

    # -- filters -------------------------------------------------------------

    def filter_mask(self, *, dataset: Any = None) -> np.ndarray:
        """The dataset's current boolean filter mask, one entry per localization."""
        ds = self._dataset(dataset)
        return np.asarray(ds.filter_mask, dtype=bool)

    def filter_specs(self, *, dataset: Any = None) -> list[dict]:
        """The dataset's persisted filter definitions, as plain dicts."""
        ds = self._dataset(dataset)
        return [dict(spec) for spec in ds.state.get("filter_specs", [])]

    def set_filter(
        self,
        specs: Iterable[Mapping[str, Any]],
        *,
        dataset: Any = None,
        replace: bool = True,
    ) -> np.ndarray:
        """
        Apply filter definitions and return the resulting mask.

        A spec is ``{attribute, mode, lo, hi}`` plus optional ``itr``,
        ``lo_inc`` and ``hi_inc``. ``mode`` is ``"per loc"`` or one of the trace
        read-outs in ``utils/filters.py::TRACE_AGG_FUNCS``.

        ``replace=False`` appends to the dataset's existing filter rather than
        replacing it -- the same concatenating behaviour a metadata recipe has,
        and for the same reason: silently discarding the user's filters is the
        worse failure.
        """
        from ..core.loader import attr_values_for_selection
        from ..utils.filters import AGG_MODES, compute_filter_mask

        ds = self._dataset(dataset)
        normalized: list[dict] = []
        for raw_spec in specs:
            spec = dict(raw_spec)
            attribute = str(spec.get("attribute", "") or "")
            if not attribute:
                raise ApiError("Every filter spec needs an 'attribute'.")
            mode = str(spec.get("mode", "per loc") or "per loc")
            if mode not in AGG_MODES:
                raise ApiError(
                    f"Unknown filter mode {mode!r}. One of: {', '.join(AGG_MODES)}"
                )
            if "lo" not in spec or "hi" not in spec:
                raise ApiError(
                    f"Filter spec for {attribute!r} needs both 'lo' and 'hi'."
                )
            normalized.append({
                "attribute": attribute,
                "mode": mode,
                "lo": float(spec["lo"]),
                "hi": float(spec["hi"]),
                "lo_inc": bool(spec.get("lo_inc", True)),
                "hi_inc": bool(spec.get("hi_inc", True)),
                "itr": spec.get("itr", "last"),
            })

        existing = [] if replace else list(ds.state.get("filter_specs", []))
        combined = existing + normalized

        mask = np.ones(int(ds.prop.num_loc), dtype=bool)
        for spec in combined:
            values = attr_values_for_selection(ds, spec["attribute"], itr=spec.get("itr"))
            if values is None:
                # No aligned 1-D view. The persisted spec still applies in the
                # plot windows through the raw store; skip it here rather than
                # dropping the whole filter.
                continue
            mask &= compute_filter_mask(
                np.asarray(values).ravel().astype(float),
                spec["mode"], spec["lo"], spec["hi"],
                ds.prop.trace_idx, ds.prop.num_loc_per_trace, ds.prop.num_traces,
                spec["lo_inc"], spec["hi_inc"],
            )

        ds.state["filter_specs"] = combined
        ds.filter_mask = mask
        self._state.notify_filter_changed(self._dataset_index(ds))
        return mask

    def clear_filter(self, *, dataset: Any = None) -> None:
        """Drop every filter row and reset the mask to all-True."""
        ds = self._dataset(dataset)
        ds.state["filter_specs"] = []
        ds.filter_mask = np.ones(int(ds.prop.num_loc), dtype=bool)
        self._state.notify_filter_changed(self._dataset_index(ds))

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _apply_filter(ds: MinfluxDataset, arr: np.ndarray) -> np.ndarray:
        """Apply the filter mask when *arr* is one-row-per-localization."""
        if arr.ndim == 0 or arr.shape[0] != int(ds.prop.num_loc):
            return arr
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] != arr.shape[0]:
            return arr
        return arr[mask]
