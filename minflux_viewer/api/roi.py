"""
``mfv.roi`` -- regions of interest.

Scoping rule that matters
-------------------------
A ROI belongs to **one dataset and one family of axes**. Every record created
through this namespace is stamped at creation with a scope
(``context["source_view"]`` and ``context["dataset_idx"]``) via
``core/roi_scope.py`` -- an unscoped record is displayed in *every* view of
*every* dataset, so a spatial region made for one dataset would appear over
another's attribute histogram.

Coordinates here are **display nm** (Z scaling factor and overlay transform
applied), because that is the frame the ROI overlay and the render/scatter
views work in. That is also why :meth:`Roi.points_in` uses
``roi_crop.display_xyz_filtered`` rather than ``ds.loc_nm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ._base import ApiError, Namespace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.dataset import MinfluxDataset
    from ..core.roi import RoiRecord

#: ROI types this namespace accepts, matching ``core/roi.py::ROI_TYPES``.
ROI_TYPES = (
    "rectangle", "oval", "polygon", "freehand",
    "line", "polyline", "freehand_line", "magnetic_lasso",
    "point", "angle",
)

#: The subset that encloses an area, and so can produce a mask or a crop.
REGION_TYPES = ("rectangle", "oval", "polygon", "freehand")


class Roi(Namespace):
    """Inspect, create and measure regions of interest."""

    name = "roi"

    # -- finding -------------------------------------------------------------

    def list(self, *, dataset: Any = None, region_only: bool = False) -> list["RoiRecord"]:
        """
        Every stored ROI belonging to *dataset*, in ROI Manager order.

        ``region_only`` keeps just the area-enclosing types, which are the ones
        :meth:`mask`, :meth:`points_in` and :meth:`crop` accept.
        """
        from ..core.roi_scope import roi_dataset_indices

        idx = self._dataset_index(dataset)
        out = []
        for record in self._state.rois.records:
            owners = roi_dataset_indices(record)
            # ``None`` means an unscoped record — deliberately shown everywhere,
            # because hiding a user's saved work is worse than one view too many.
            if owners is not None and idx not in owners:
                continue
            if region_only and record.type not in REGION_TYPES:
                continue
            out.append(record)
        return out

    def active(self, *, dataset: Any = None) -> "RoiRecord | None":
        """
        The ROI a command would act on: the single selected stored ROI,
        otherwise the dataset's recorded active draft, otherwise ``None``.

        A draft still being drawn lives in the view's overlay controller rather
        than the store, so it is reachable here only once it has been recorded
        against the dataset.
        """
        selected = self._state.rois.selected_records()
        if len(selected) == 1:
            return selected[0]
        ds = self._dataset(dataset)
        draft_id = ds.state.get("active_roi_draft_id")
        if draft_id:
            for record in self._state.rois.records:
                if record.id == draft_id:
                    return record
        return None

    def selected(self, *, dataset: Any = None) -> list["RoiRecord"]:
        """Every ROI currently selected in the ROI Manager."""
        return list(self._state.rois.selected_records())

    # -- creating ------------------------------------------------------------

    def add(
        self,
        roi_type: str,
        geometry: "Mapping[str, Any]",
        *,
        dataset: Any = None,
        name: str = "",
        color: str | None = None,
        view: str = "render",
        select: bool = False,
    ) -> "RoiRecord":
        """
        Store a new ROI and show it in the ROI Manager.

        *geometry* uses the same keys the ROI store does -- ``{"bounds":
        [x, y, w, h], "angle": deg}`` for a rectangle or oval, ``{"points":
        [...]}`` for a polygon or line, ``{"point": [x, y, z]}`` for a point;
        coordinates in display nm. *view* is the axis family the ROI belongs to
        and is stamped into the record's scope.
        """
        from ..core.roi import RoiRecord

        if roi_type not in ROI_TYPES:
            raise ApiError(
                f"Unknown ROI type {roi_type!r}. One of: {', '.join(ROI_TYPES)}"
            )
        if not isinstance(geometry, Mapping) or not geometry:
            raise ApiError("A ROI needs a geometry dict, e.g. {'bounds': [x, y, w, h]}.")

        idx = self._dataset_index(dataset)
        record = RoiRecord.create(
            roi_type,
            dict(geometry),
            name=name or None,
            context={"source_view": str(view), "dataset_idx": int(idx)},
            **({"stroke_color": color} if color else {}),
        )
        self._state.rois.add(record)
        if select:
            self._state.rois.select([record.id])
        return record

    def remove(self, roi: Any, *, dataset: Any = None) -> bool:
        """Delete a stored ROI by record or id. Returns whether it existed."""
        roi_id = self._roi_id(roi)
        store = self._state.rois
        if not any(r.id == roi_id for r in store.records):
            return False
        store.select([roi_id])
        store.delete_selected()
        return True

    def select(self, rois: Any, *, dataset: Any = None) -> None:
        """Select one ROI or a list of them in the ROI Manager."""
        if rois is None:
            self._state.rois.deselect()
            return
        items = rois if isinstance(rois, (list, tuple)) else [rois]
        self._state.rois.select([self._roi_id(r) for r in items])

    # -- measuring -----------------------------------------------------------

    def mask(self, roi: Any, *, dataset: Any = None, filtered: bool = True) -> np.ndarray:
        """
        Boolean mask of the localizations inside a region ROI.

        Aligned to the coordinates :meth:`points_in` returns. Open-line and
        point ROIs enclose no area and yield an all-False mask rather than an
        error -- that is what ``roi_region_mask`` already promises.
        """
        from ..core.roi_selection import roi_region_mask

        record = self._record(roi)
        xyz = self._coords(dataset, filtered=filtered)
        return np.asarray(
            roi_region_mask(xyz[:, 0], xyz[:, 1], record), dtype=bool
        )

    def points_in(
        self,
        roi: Any,
        *,
        dataset: Any = None,
        filtered: bool = True,
        unit: str = "nm",
    ) -> np.ndarray:
        """The ``(N, 3)`` coordinates inside a region ROI, in display nm."""
        xyz = self._coords(dataset, filtered=filtered)
        inside = self.mask(roi, dataset=dataset, filtered=filtered)
        out = xyz[inside]
        if unit == "m":
            return out / 1e9
        if unit != "nm":
            raise ApiError("unit must be 'nm' or 'm'.")
        return out

    def attr_in(
        self,
        roi: Any,
        name: str,
        *,
        dataset: Any = None,
        filtered: bool = True,
    ) -> np.ndarray:
        """One attribute's values for the localizations inside a region ROI."""
        ds = self._dataset(dataset)
        values = self._facade.data.attr(name, dataset=ds, filtered=filtered)
        inside = self.mask(roi, dataset=ds, filtered=filtered)
        if values.shape[0] != inside.shape[0]:
            raise ApiError(
                f"Attribute {name!r} has {values.shape[0]} values but the ROI "
                f"mask covers {inside.shape[0]} localizations."
            )
        return values[inside]

    def crop(
        self,
        roi: Any,
        *,
        dataset: Any = None,
        exact_shape: bool = True,
        z_all: bool = True,
        name: str | None = None,
        add: bool = True,
    ) -> "MinfluxDataset":
        """
        Duplicate a dataset restricted to a region ROI.

        ``exact_shape=True`` follows the ROI outline; ``False`` takes its
        axis-aligned bounding box. Delegates to ``core/roi_crop.py``, so the
        result matches *Shift+D*.
        """
        from ..core.roi_crop import compute_crop_mask, subset_dataset

        record = self._record(roi)
        if record.type not in REGION_TYPES:
            raise ApiError(
                f"Only a region ROI can be cropped; {record.type!r} encloses no "
                f"area. Region types: {', '.join(REGION_TYPES)}"
            )
        ds = self._dataset(dataset)
        keep = compute_crop_mask(ds, record, exact_shape=exact_shape)
        keep = np.asarray(keep, dtype=bool)
        if not keep.any():
            raise ApiError("The ROI contains no localizations, so the crop is empty.")
        cropped = subset_dataset(
            ds, keep,
            name=name or f"{ds.name} (crop)",
            prefs=self._state.prefs,
        )
        if add:
            self._state.add_dataset(cropped)
        return cropped

    # -- geometry ------------------------------------------------------------

    def geometry(self, roi: Any) -> dict:
        """A ROI's geometry as a plain dict, safe to serialise."""
        return dict(self._record(roi).geometry)

    def bounds(self, roi: Any) -> tuple[float, float, float, float]:
        """A ROI's bounding box in display nm as ``(x, y, w, h)``."""
        from ..core.roi import _bounds

        record = self._record(roi)
        try:
            x, y, w, h = _bounds(record.geometry)
        except Exception as exc:
            raise ApiError(
                f"ROI {record.name or record.id!r} has no bounding box: {exc}"
            ) from None
        return (float(x), float(y), float(w), float(h))

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _roi_id(roi: Any) -> str:
        if isinstance(roi, str):
            return roi
        roi_id = getattr(roi, "id", None)
        if not roi_id:
            raise ApiError("Expected a ROI record or a ROI id.")
        return str(roi_id)

    def _record(self, roi: Any) -> "RoiRecord":
        if hasattr(roi, "geometry"):
            return roi
        roi_id = self._roi_id(roi)
        for record in self._state.rois.records:
            if record.id == roi_id:
                return record
        raise ApiError(f"No ROI with id {roi_id!r} is stored.")

    def _coords(self, dataset: Any, *, filtered: bool) -> np.ndarray:
        """
        Display-space XYZ -- the frame ROI geometry is expressed in.

        Deliberately **not** ``display_xyz_filtered``: that also drops
        non-finite rows, so its length is not ``mask.sum()`` and the result
        would not line up with ``mfv.data.attr(filtered=True)``. Row alignment
        between coordinates, masks and attributes is what makes
        :meth:`attr_in` meaningful, so the filter is applied the same way
        ``mfv.data`` applies it and NaN coordinates are left in place (a region
        test returns False for them anyway).
        """
        from ..core.roi_crop import display_coords

        ds = self._dataset(dataset)
        xyz = np.asarray(display_coords(ds), dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            return np.empty((0, 3), dtype=float)
        if filtered:
            mask = np.asarray(ds.filter_mask, dtype=bool)
            if mask.shape[0] == xyz.shape[0]:
                xyz = xyz[mask]
        return xyz[:, :3]
