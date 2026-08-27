"""Reading a **standalone MBM (beam-monitoring bead) companion file**.

``.mat`` / ``.npy`` / ``.json`` / ``.csv`` cannot hold a MINFLUX localization
dataset and its bead reference in one file, so the MSR reader writes the beads
beside the localizations as ``<stem>_mbm.<ext>``. Those companions were
unopenable: the ``.mat`` and ``.json`` were rejected as "not a MINFLUX dataset",
and the ``.npy`` loaded as an *empty* dataset because a bead table has no
``loc`` field.

A bead table is not localization data and must not be loaded as a dataset. This
module identifies one from its column names and reads it back into the model
``grd/mbm/points`` array (``gri`` / ``xyz`` metres / ``tim`` seconds / ``str``),
which is exactly what ``beads_drift.extract_bead_drift`` consumes -- so a
companion file opens the same **MBM info** window as *View mbm info...* on a
loaded dataset, with no localizations needed.

Qt-free; the router and the window sit on top.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..msr.legacy_mbm import POINTS_DTYPE

__all__ = [
    "MBM_REQUIRED_FIELDS",
    "is_mbm_points_fields",
    "is_mbm_points_file",
    "read_mbm_points",
]

#: Fields ``extract_bead_drift`` needs; ``str`` (PMT signal) is optional.
MBM_REQUIRED_FIELDS: frozenset[str] = frozenset({"gri", "tim"})
#: Accepted spellings of the bead position column.
_XYZ_VECTOR = "xyz"
_XYZ_COMPONENTS = ("xyz_0", "xyz_1", "xyz_2")
#: Names that mean the file is localization data, whatever else it carries.
#: ``gri`` also appears in an ``mfx`` table, so the absence of these is what
#: makes the identification safe.
_LOCALIZATION_MARKERS: frozenset[str] = frozenset({
    "itr", "vld", "loc", "loc_x", "loc_y", "loc_0", "loc_1", "xnm", "ynm",
    "efo", "cfr", "dcr", "dcr_0", "eco", "tid",
})
#: Bytes of a ``.json`` inspected to classify it -- a bead table is small, but
#: the classifier runs on every ``.json`` the router sees, including a
#: multi-gigabyte localization array.
_JSON_PREFIX_BYTES = 4096


def is_mbm_points_fields(names) -> bool:
    """Whether a set of column names is a bead table and not localization data."""
    keys = {str(name).strip().lower() for name in names or ()}
    if not MBM_REQUIRED_FIELDS <= keys:
        return False
    has_xyz = _XYZ_VECTOR in keys or set(_XYZ_COMPONENTS) <= keys
    return has_xyz and not (keys & _LOCALIZATION_MARKERS)


# ---------------------------------------------------------------------------
# Bounded per-format classification
# ---------------------------------------------------------------------------

def _npy_field_names(path: Path) -> list[str]:
    """Field names from a ``.npy`` **header only** -- the array is not read."""
    from numpy.lib import format as npy_format

    with open(path, "rb") as handle:
        version = npy_format.read_magic(handle)
        if version[0] == 1:
            _shape, _order, dtype = npy_format.read_array_header_1_0(handle)
        else:
            _shape, _order, dtype = npy_format.read_array_header_2_0(handle)
    return list(dtype.names or ())


def _mat_struct_names(path: Path) -> list[str]:
    """Field names of a ``.mat`` bead struct, read without loading a big array.

    ``whosmat`` lists the variables from the header; only a variable actually
    named ``mbm`` is then loaded, and a bead struct is a few kilobytes.
    """
    from scipy.io import loadmat, whosmat

    variables = {str(name) for name, _shape, _cls in whosmat(str(path))}
    if "mbm" not in variables:
        return []
    value = loadmat(str(path), variable_names=["mbm"]).get("mbm")
    dtype = getattr(value, "dtype", None)
    return list(getattr(dtype, "names", None) or ())


def _json_prefix_names(path: Path) -> list[str]:
    """Keys of the first record of a JSON array, from a bounded prefix read."""
    with open(path, "rb") as handle:
        prefix = handle.read(_JSON_PREFIX_BYTES)
    text = prefix.decode("utf-8", errors="ignore").lstrip()
    if not text.startswith("["):
        return []
    start = text.find("{")
    if start < 0:
        return []
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return list(json.loads(text[start:index + 1]).keys())
                except (ValueError, AttributeError):
                    return []
    return []


def _csv_header_names(path: Path) -> list[str]:
    import csv as _csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
    if not sample.strip():
        return []
    try:
        delimiter = _csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except _csv.Error:
        delimiter = ","
    line = sample.splitlines()[0]
    return [cell.strip() for cell in next(_csv.reader([line], delimiter=delimiter))]


_FIELD_READERS = {
    ".npy": _npy_field_names,
    ".mat": _mat_struct_names,
    ".json": _json_prefix_names,
    ".csv": _csv_header_names,
    ".tsv": _csv_header_names,
}


def is_mbm_points_file(path) -> bool:
    """Whether *path* is a standalone MBM bead table.

    Every check reads only a header / prefix, because the router probes this
    predicate for every ``.mat``, ``.npy``, ``.json`` and ``.csv`` it opens.
    An unreadable file is simply "not one".
    """
    target = Path(path)
    reader = _FIELD_READERS.get(target.suffix.lower())
    if reader is None:
        return False
    try:
        names = reader(target)
    except Exception:                                       # noqa: BLE001
        return False
    return is_mbm_points_fields(names)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _points_from_columns(columns: dict) -> np.ndarray:
    """Flat columns -> the model ``points`` array, recomposing ``xyz``."""
    lowered = {str(key).strip().lower(): value for key, value in columns.items()}
    vector = lowered.get(_XYZ_VECTOR)
    if vector is not None:
        xyz = np.asarray(vector, dtype=float).reshape(-1, 3)
    else:
        missing = [name for name in _XYZ_COMPONENTS if name not in lowered]
        if missing:
            raise ValueError(
                "MBM bead table is missing the position column(s) "
                f"{', '.join(missing)}."
            )
        xyz = np.column_stack([
            np.asarray(lowered[name], dtype=float).ravel()
            for name in _XYZ_COMPONENTS
        ])
    n = int(xyz.shape[0])
    points = np.zeros(n, dtype=POINTS_DTYPE)
    points["xyz"] = xyz
    points["gri"] = np.asarray(lowered["gri"]).ravel()[:n].astype(np.int64)
    points["tim"] = np.asarray(lowered["tim"], dtype=float).ravel()[:n]
    if "str" in lowered:
        points["str"] = np.asarray(lowered["str"], dtype=float).ravel()[:n]
    return points


def read_mbm_points(path) -> np.ndarray:
    """Read a standalone MBM companion into the model ``points`` array.

    Raises ``ValueError`` when *path* is not a bead table, so a caller never has
    to guess whether an empty result means "no beads" or "wrong file".
    """
    target = Path(path)
    suffix = target.suffix.lower()

    if suffix == ".npy":
        array = np.load(str(target), allow_pickle=False)
        names = list(getattr(array.dtype, "names", None) or ())
        columns = {name: array[name] for name in names}
    elif suffix == ".mat":
        from scipy.io import loadmat

        value = loadmat(str(target), variable_names=["mbm"]).get("mbm")
        if value is None:
            raise ValueError(f"'{target.name}' has no 'mbm' bead struct.")
        # savemat wrote one struct with length-N array fields, so it comes back
        # as a 1x1 structured array whose scalar fields are (1, N) row vectors
        # while the position field keeps its (N, 3) shape.
        value = np.asarray(value)
        record = value.reshape(-1)[0]
        columns = {}
        for name in (getattr(value.dtype, "names", None) or ()):
            field = np.asarray(record[name])
            columns[name] = (field if field.ndim == 2 and field.shape[1] == 3
                             else field.ravel())
    elif suffix == ".json":
        records = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"'{target.name}' is not a JSON array of bead records.")
        columns = {key: np.array([record.get(key) for record in records])
                   for key in records[0]}
    elif suffix in {".csv", ".tsv"}:
        names = _csv_header_names(target)
        delimiter = "\t" if suffix == ".tsv" else ","
        values = np.loadtxt(str(target), delimiter=delimiter, skiprows=1, ndmin=2)
        columns = {name: values[:, index] for index, name in enumerate(names)}
    else:
        raise ValueError(f"'{target.name}' is not a readable MBM companion format.")

    if not is_mbm_points_fields(columns.keys()):
        found = ", ".join(str(key) for key in columns) or "none"
        raise ValueError(
            f"'{target.name}' is not an MBM bead table (found columns: {found})."
        )
    return _points_from_columns(columns)
