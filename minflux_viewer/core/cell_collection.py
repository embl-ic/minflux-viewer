"""Accumulate ROI-delimited cells across several datasets for a pooled analysis.

One :class:`CellSample` is the localizations of a single object — for this
project an *E. coli* cell — cut out of one dataset by one region ROI, together
with the provenance needed to report where it came from. A
:class:`CellCollection` is an ordered pool of them, built up across as many
datasets (and sessions) as the operator needs.

The pooling rule that makes this sound lives in the analysis, not here: each
collected cell becomes its own spatial component, so pairs are only ever formed
*within* a cell. Coordinates are therefore kept exactly as recorded — cells are
never re-zeroed or moved to avoid collisions, because two cells from different
acquisitions occupying the same coordinates can still never interact.

Contrast with :mod:`minflux_viewer.core.particle_set`, which re-zeroes each
particle precisely because particle averaging must superimpose them.

Pure NumPy/h5py; no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "minflux_cell_collection"
FORMAT_VERSION = 1
#: Below this a cell cannot support trace consolidation, let alone a pair profile.
MIN_LOCS_PER_CELL = 20

__all__ = [
    "CellSample",
    "CellCollection",
    "extract_cells",
    "region_roi_records",
    "save_cell_collection",
    "load_cell_collection",
    "is_cell_collection_file",
    "FORMAT",
    "MIN_LOCS_PER_CELL",
]


@dataclass(frozen=True)
class CellSample:
    """One ROI-delimited cell's raw localizations, with provenance.

    ``loc_m`` is ``(N, 3)`` in **metres**, exactly as recorded — not Z-scaled
    and not re-zeroed. The analysis applies its own z scaling, so baking one in
    here would double-correct it.
    """

    loc_m: np.ndarray
    tid: np.ndarray
    tim: np.ndarray | None
    dataset: str
    roi: str
    source_path: str = ""

    @property
    def n_locs(self) -> int:
        return int(np.asarray(self.loc_m).shape[0])

    @property
    def n_traces(self) -> int:
        return int(np.unique(np.asarray(self.tid)).size)

    @property
    def label(self) -> str:
        return f"{self.dataset} · {self.roi}" if self.dataset else self.roi

    def as_cell(self) -> dict:
        """The mapping :func:`~minflux_viewer.analysis.hlyb_staged.
        analyze_hlyb_staged_pooled` consumes."""
        return {"loc_m": self.loc_m, "tid": self.tid, "tim": self.tim,
                "label": self.label, "dataset": self.dataset, "roi": self.roi}

    def summary(self) -> dict:
        return {"dataset": self.dataset, "roi": self.roi,
                "n_locs": self.n_locs, "n_traces": self.n_traces,
                "has_time": self.tim is not None,
                "source_path": self.source_path}


@dataclass
class CellCollection:
    """An ordered pool of collected cells."""

    cells: list[CellSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def add(self, cell: CellSample) -> None:
        self.cells.append(cell)

    def extend(self, cells) -> int:
        added = 0
        for cell in cells:
            self.add(cell)
            added += 1
        return added

    def remove(self, indices) -> int:
        drop = {int(i) for i in indices}
        keep = [c for i, c in enumerate(self.cells) if i not in drop]
        removed = len(self.cells) - len(keep)
        self.cells = keep
        return removed

    def clear(self) -> None:
        self.cells = []

    @property
    def datasets(self) -> list[str]:
        seen: list[str] = []
        for cell in self.cells:
            if cell.dataset and cell.dataset not in seen:
                seen.append(cell.dataset)
        return seen

    def has(self, dataset: str, roi: str) -> bool:
        """True when this exact dataset/ROI pair is already pooled.

        Collecting is a repeated manual step, so re-collecting the same dataset
        would otherwise silently double-count its cells.
        """
        return any(cell.dataset == dataset and cell.roi == roi
                   for cell in self.cells)

    def as_cells(self) -> list[dict]:
        return [cell.as_cell() for cell in self.cells]

    def summary(self) -> dict:
        return {
            "n_cells": len(self.cells),
            "n_datasets": len(self.datasets),
            "n_locs": int(sum(cell.n_locs for cell in self.cells)),
            "n_traces": int(sum(cell.n_traces for cell in self.cells)),
        }


# --------------------------------------------------------------------------- #
# Extraction from a dataset + ROI records
# --------------------------------------------------------------------------- #
def region_roi_records(store, dataset_idx: int) -> list:
    """Region ROIs in *store* belonging to *dataset_idx*, in Manager order.

    Open-line, point and angle ROIs enclose no area and are skipped: a cell has
    to be a region for its localizations to be selectable at all.
    """
    from .roi_convert import REGION_TYPES

    records = list(getattr(store, "records", []) or [])
    out = []
    for record in records:
        if getattr(record, "type", None) not in REGION_TYPES:
            continue
        context = getattr(record, "context", None) or {}
        owner = context.get("dataset_idx")
        if owner is not None and int(owner) != int(dataset_idx):
            continue
        out.append(record)
    return out


def extract_cells(ds, records, *, dataset_idx: int = 0,
                  min_locs: int = MIN_LOCS_PER_CELL,
                  trace_complete: bool = False) -> tuple[list[CellSample], list[str]]:
    """Cut one :class:`CellSample` per region ROI out of *ds*.

    Returns ``(cells, skipped)`` where *skipped* explains every ROI that yielded
    no usable cell, so the caller can report it rather than silently dropping it.
    """
    from .loader import mfx_get
    from .roi_crop import compute_crop_mask

    def column(attr):
        value = mfx_get(ds, attr, itr="last", vld_only=True)
        return None if value is None else np.asarray(value, dtype=float).ravel()

    lx, ly, lz, tid = (column("loc_x"), column("loc_y"),
                       column("loc_z"), column("tid"))
    if lx is None or ly is None or lz is None or tid is None:
        raise ValueError(
            "The dataset does not expose loc_x/loc_y/loc_z/tid at the last "
            "valid iteration.")
    loc = np.column_stack([lx, ly, lz])
    tim = column("tim")
    if tim is not None and tim.size != loc.shape[0]:
        tim = None

    name = str(getattr(ds, "name", "") or "")
    source = str((getattr(ds, "metadata", None) or {}).get("msr_source_path", "")
                 or "")
    cells: list[CellSample] = []
    skipped: list[str] = []
    for index, record in enumerate(records, start=1):
        roi_name = str(getattr(record, "name", "") or f"roi-{index}")
        mask = np.asarray(
            compute_crop_mask(ds, record, exact_shape=True,
                              trace_complete=trace_complete), dtype=bool)
        if mask.size != loc.shape[0]:
            # The ROI mask is built on ds.attr rows; if that is not the
            # last-valid materialization the two cannot be aligned, and
            # guessing would silently mis-assign localizations.
            raise ValueError(
                f"'{roi_name}': the ROI mask ({mask.size} rows) does not align "
                f"with the last-valid localizations ({loc.shape[0]} rows). "
                f"Load the dataset with the default last-iteration, valid-only "
                f"setting to pool it.")
        count = int(np.count_nonzero(mask))
        if count < int(min_locs):
            skipped.append(f"'{roi_name}': {count} localization(s), "
                           f"below the {int(min_locs)} minimum")
            continue
        cells.append(CellSample(
            loc_m=loc[mask], tid=tid[mask],
            tim=None if tim is None else tim[mask],
            dataset=name, roi=roi_name, source_path=source))
    return cells, skipped


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def is_cell_collection_file(path) -> bool:
    """True when *path* is an HDF5 file written by :func:`save_cell_collection`."""
    import h5py

    try:
        with h5py.File(str(path), "r") as handle:
            return str(handle.attrs.get("format", "")) == FORMAT
    except Exception:                                             # noqa: BLE001
        return False


def save_cell_collection(path, collection: CellCollection) -> Path:
    """Write the pool as one HDF5 file (pooled columns keyed by ``cell_id``)."""
    import h5py

    target = Path(path)
    cells = list(collection)
    if not cells:
        raise ValueError("Nothing to save: the collection is empty.")
    cell_id = np.concatenate(
        [np.full(cell.n_locs, i, dtype=np.int32) for i, cell in enumerate(cells)])
    loc = np.concatenate([np.asarray(cell.loc_m, dtype=np.float64)
                          for cell in cells], axis=0)
    tid = np.concatenate([np.asarray(cell.tid, dtype=np.float64)
                          for cell in cells])
    has_time = all(cell.tim is not None for cell in cells)
    with h5py.File(str(target), "w") as handle:
        handle.attrs["format"] = FORMAT
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["n_cells"] = len(cells)
        group = handle.create_group("localizations")
        group.create_dataset("cell_id", data=cell_id, compression="gzip")
        for index, axis in enumerate("xyz"):
            group.create_dataset(f"loc_{axis}", data=loc[:, index],
                                 compression="gzip")
        group.create_dataset("tid", data=tid, compression="gzip")
        if has_time:
            group.create_dataset(
                "tim", compression="gzip",
                data=np.concatenate([np.asarray(cell.tim, dtype=np.float64)
                                     for cell in cells]))
        meta = handle.create_group("cells")
        dt = h5py.special_dtype(vlen=str)
        for key in ("dataset", "roi", "source_path"):
            meta.create_dataset(
                key, dtype=dt,
                data=np.array([getattr(cell, key) for cell in cells], dtype=object))
        meta.create_dataset(
            "n_locs", data=np.array([cell.n_locs for cell in cells], dtype=np.int64))
    return target


def load_cell_collection(path) -> CellCollection:
    """Read a collection written by :func:`save_cell_collection`."""
    import h5py

    with h5py.File(str(path), "r") as handle:
        if str(handle.attrs.get("format", "")) != FORMAT:
            raise ValueError(f"'{Path(path).name}' is not a MINFLUX cell collection.")
        group = handle["localizations"]
        cell_id = np.asarray(group["cell_id"])
        loc = np.column_stack([np.asarray(group[f"loc_{axis}"]) for axis in "xyz"])
        tid = np.asarray(group["tid"])
        tim = np.asarray(group["tim"]) if "tim" in group else None
        meta = handle["cells"]

        def text(key, index):
            if key not in meta:
                return ""
            value = meta[key][index]
            return value.decode() if isinstance(value, bytes) else str(value)

        cells = []
        for index in range(int(handle.attrs.get("n_cells", 0))):
            sel = cell_id == index
            if not np.any(sel):
                continue
            cells.append(CellSample(
                loc_m=loc[sel], tid=tid[sel],
                tim=None if tim is None else tim[sel],
                dataset=text("dataset", index), roi=text("roi", index),
                source_path=text("source_path", index)))
    return CellCollection(cells)
