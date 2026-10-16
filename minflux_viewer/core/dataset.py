"""
minflux_viewer.core.dataset
===========================
Central data model for one loaded localisation or image dataset.

Historically this mirrored the MATLAB ``app.data(idx)`` MINFLUX struct:

    .file     — path, name, load datetime
    .prop     — num_loc, num_itr, num_dim, num_traces, trace_idx, attr_names
    .attr     — all attribute arrays (loc_x/y/z, tid, efo, cfr, ftr, …)
    .cali     — calibration scalars (Z scaling factor, pixel_size, loc_precision)
    .channel  — DCR / color channel info

The same container can now also hold generic localisation coordinates
already stored in nanometres, plus optional image payloads for rendering.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# AttrStore — dict-backed attribute container
# ---------------------------------------------------------------------------

class _Virtual:
    """An attribute read from its owner (a dataset property), never stored.

    Holds the property name only, so a copied store carries no reference to
    the dataset it was copied from; the dataset that receives the store binds
    it to itself.
    """

    __slots__ = ("prop",)

    def __init__(self, prop: str) -> None:
        self.prop = prop

    def __deepcopy__(self, memo) -> _Virtual:
        return _Virtual(self.prop)

    def __repr__(self) -> str:
        return f"<virtual: dataset.{self.prop}>"


#: The localization coordinate columns, in metres: the corrected ``loc`` and the
#: uncorrected ``lnc``. Once their dataset is built they are read-only and
#: change only through :meth:`AttrStore.replace_coordinates`.
COORDINATE_COLUMNS: tuple[str, ...] = (
    "loc_x", "loc_y", "loc_z", "lnc_x", "lnc_y", "lnc_z")


class AttrStore:
    """
    Thin wrapper around a plain dict of numpy arrays.

    Allows both styles::

        attrs["loc_x"]   # dict-style
        attrs.loc_x      # attribute-style  (mirrors MATLAB data.attr.loc_x)

    An entry may be **virtual** (:meth:`bind_virtual`): read from the owning
    dataset on every access, listed where it always was, and read-only -- the
    filter mask's ``ftr`` is one, so the mask has a single owner.

    Once the store belongs to a dataset its :data:`COORDINATE_COLUMNS` are
    **sealed** (:meth:`seal_coordinates`): their arrays are read-only and
    assigning or deleting one raises. :meth:`replace_coordinates` is the one
    way to change them, and :attr:`coordinate_revision` counts its calls.
    """

    def __init__(self, data: dict[str, np.ndarray] | None = None) -> None:
        object.__setattr__(self, "_d", data or {})
        object.__setattr__(self, "_owner", None)
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_coordinate_revision", 0)

    def _resolve(self, name: str, value: Any) -> Any:
        if not isinstance(value, _Virtual):
            return value
        owner_ref = object.__getattribute__(self, "_owner")
        owner = owner_ref() if owner_ref is not None else None
        if owner is None:
            raise RuntimeError(
                f"Attribute {name!r} is read from a dataset this store is not bound to.")
        return getattr(owner, value.prop)

    def _refuse_virtual(self, name: str) -> None:
        if isinstance(object.__getattribute__(self, "_d").get(name), _Virtual):
            raise TypeError(
                f"Attribute {name!r} is read-only: it is the dataset's "
                f"{object.__getattribute__(self, '_d')[name].prop}; set that instead.")
        if name in COORDINATE_COLUMNS and object.__getattribute__(self, "_sealed"):
            raise TypeError(
                f"Coordinate column {name!r} is read-only once its dataset is built; "
                f"change it with replace_coordinates({name}=...).")

    # -- coordinates -------------------------------------------------------
    def seal_coordinates(self) -> None:
        """Make the coordinate columns read-only from now on (idempotent).

        Their arrays are flagged non-writeable in place, so a reference taken
        earlier is guarded too; a column added later must come through
        :meth:`replace_coordinates`.
        """
        d = object.__getattribute__(self, "_d")
        for name in COORDINATE_COLUMNS:
            value = d.get(name)
            if value is None or isinstance(value, _Virtual):
                continue
            if not isinstance(value, np.ndarray):
                value = np.array(value)
                d[name] = value
            value.flags.writeable = False
        object.__setattr__(self, "_sealed", True)

    def replace_coordinates(self, **columns: Any) -> None:
        """The one way to change coordinate columns once they are sealed.

        Each value becomes a read-only float copy (the caller's array is not
        kept) with as many rows as the store already has; a refused call
        changes nothing. :attr:`coordinate_revision` goes up by one per call.
        """
        unknown = sorted(set(columns) - set(COORDINATE_COLUMNS))
        if unknown:
            raise ValueError(f"{unknown} is not a coordinate column; "
                             f"those are {COORDINATE_COLUMNS}")
        d = object.__getattribute__(self, "_d")
        rows = next((len(v) for v in d.values()
                     if isinstance(v, np.ndarray) and v.ndim >= 1), None)
        prepared: dict[str, np.ndarray] = {}
        for name, value in columns.items():
            arr = np.array(value, dtype=float, copy=True)
            if arr.ndim != 1 or (rows is not None and arr.shape[0] != rows):
                raise ValueError(f"{name} needs {rows} rows in one column, "
                                 f"got shape {arr.shape}")
            arr.flags.writeable = False
            prepared[name] = arr
        d.update(prepared)
        object.__setattr__(
            self, "_coordinate_revision",
            object.__getattribute__(self, "_coordinate_revision") + 1)

    @property
    def coordinate_revision(self) -> int:
        """How many times :meth:`replace_coordinates` has changed this store."""
        return object.__getattribute__(self, "_coordinate_revision")

    def bind_virtual(self, name: str, owner: Any, prop: str) -> None:
        """Make *name* read ``owner.<prop>`` (keeping its place in the order)."""
        object.__setattr__(self, "_owner", weakref.ref(owner))
        object.__getattribute__(self, "_d")[name] = _Virtual(prop)

    def __deepcopy__(self, memo) -> AttrStore:
        import copy as _copy
        new = AttrStore(_copy.deepcopy(object.__getattribute__(self, "_d"), memo))
        # Copied as part of its dataset: bind to the copy (deepcopy registers
        # the new dataset in *memo* before copying its fields). Copied alone:
        # unbound, and whichever dataset it is given to binds it.
        owner_ref = object.__getattribute__(self, "_owner")
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None and id(owner) in memo:
            object.__setattr__(new, "_owner", weakref.ref(memo[id(owner)]))
        # The copy's own coordinate arrays are as guarded as these.
        object.__setattr__(new, "_coordinate_revision",
                           object.__getattribute__(self, "_coordinate_revision"))
        if object.__getattribute__(self, "_sealed"):
            new.seal_coordinates()
        return new

    # -- attribute access --------------------------------------------------
    def __getattr__(self, name: str) -> np.ndarray:
        d = object.__getattribute__(self, "_d")
        try:
            value = d[name]
        except KeyError:
            raise AttributeError(f"AttrStore has no attribute '{name}'") from None
        return self._resolve(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        self._refuse_virtual(name)
        object.__getattribute__(self, "_d")[name] = value

    # -- dict-style access -------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_d")

    def __getitem__(self, name: str) -> np.ndarray:
        return self._resolve(name, object.__getattribute__(self, "_d")[name])

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        self._refuse_virtual(name)
        object.__getattribute__(self, "_d")[name] = value

    def __delitem__(self, name: str) -> None:
        self._refuse_virtual(name)
        del object.__getattribute__(self, "_d")[name]

    def keys(self) -> list[str]:
        return list(object.__getattribute__(self, "_d").keys())

    def items(self):
        return [(name, self._resolve(name, value))
                for name, value in object.__getattribute__(self, "_d").items()]

    def values(self):
        return [value for _name, value in self.items()]

    def get(self, name: str, default: Any = None) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            return default
        return self._resolve(name, d[name])

    def pop(self, name: str, *args) -> Any:
        self._refuse_virtual(name)
        return object.__getattribute__(self, "_d").pop(name, *args)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_d"))

    def __repr__(self) -> str:
        keys = list(object.__getattribute__(self, "_d").keys())
        return f"AttrStore({keys})"


class AttributeComponent:
    """
    Flexible component-level attribute table.

    This is the first compatibility layer toward the canonical data model. It
    wraps an :class:`AttrStore` so existing code can keep using
    ``dataset.attr`` while new code can use ``dataset.mfx`` or
    ``dataset.mbm`` with role and metadata lookups.
    """

    def __init__(
        self,
        attrs: AttrStore | dict[str, np.ndarray] | None = None,
        *,
        roles: dict[str, str] | None = None,
        meta: dict[str, dict] | None = None,
    ) -> None:
        self.attrs = attrs if isinstance(attrs, AttrStore) else AttrStore(attrs or {})
        self.roles: dict[str, str] = dict(roles or {})
        self.meta: dict[str, dict] = dict(meta or {})

    def get_attr(self, name: str, default: Any = None) -> Any:
        return self.attrs.get(name, default)

    def set_attr(self, name: str, value: Any, *, meta: dict | None = None) -> None:
        self.attrs[name] = np.asarray(value)
        if meta is not None:
            self.meta[name] = dict(meta)

    def attr_names(self) -> list[str]:
        return self.attrs.keys()

    def get_role(self, role: str, default: Any = None) -> Any:
        name = self.roles.get(role)
        if not name:
            return default
        return self.get_attr(name, default)

    def set_role(self, role: str, attr_name: str) -> None:
        self.roles[role] = attr_name

    def get_meta(self, name: str, default: Any = None) -> Any:
        return self.meta.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.attrs

    def __repr__(self) -> str:
        return f"AttributeComponent(attrs={self.attr_names()!r})"


@dataclass
class DatasetComponents:
    """Canonical component containers owned by a dataset."""

    mfx: AttributeComponent | None = None
    mbm: AttributeComponent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived: AttrStore = field(default_factory=AttrStore)
    state: dict[str, Any] = field(default_factory=dict)
    mfx_raw: AttrStore = field(default_factory=AttrStore)
    # Derived attributes (dt, dst, spd, siz, dur, len, tim_trace, den, …)
    # computed at the last-valid selection of mfx_raw, aligned to
    # mfx_row_mask(itr="last", vld_only=True) rows. Populated when ds.attr is
    # NOT the last-valid materialization (an all-iteration store), so mfx_get can
    # broadcast them onto any iteration selection.
    derived_last: AttrStore = field(default_factory=AttrStore)

    def __setattr__(self, name: str, value: Any) -> None:
        # Loaders attach the all-iteration store after the dataset is built;
        # its coordinates are sealed the moment it is attached.
        if name == "mfx_raw" and isinstance(value, AttrStore):
            value.seal_coordinates()
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Sub-structs
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """File-level metadata."""
    name: str                       # e.g. "my_data.mat"
    folder: str                     # absolute parent directory
    datetime: str = ""              # formatted load/modification time
    raw_data: dict | None = None    # original loaded dict (kept for save-with-filter)
    recent_path: str | None = None  # optional path to record in File > Open recent

    @property
    def path(self) -> str:
        return os.path.join(self.folder, self.name)


def dataset_source_file(ds) -> Path | None:
    """The file this dataset was loaded from, when it is still on disk.

    Follows the same order the Data window reports: the ``.msr`` an import came
    from, then the recorded recent path, then ``folder``/``name``. Returns None
    for anything without a file behind it — a simulation, or a dataset derived
    in the app (those are built with an empty ``folder``, which also stops a
    bare ``name`` from resolving against the working directory).
    """
    candidates = [
        ds.metadata.get("msr_source_path"),
        ds.file.recent_path,
    ]
    if ds.file.folder and ds.file.name:
        candidates.append(ds.file.path)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(str(candidate))
            # A .zarr store is a directory; revealing it is still right.
            if path.exists():
                return path
        except (OSError, ValueError):
            continue
    return None


@dataclass
class DataProp:
    """Dataset-level properties computed once on load."""
    num_loc: int = 0
    num_itr: int = 0
    num_dim: int = 2                # 2 or 3
    num_traces: int = 0
    trace_idx: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=int)
    )
    num_loc_per_trace: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    attr_names: list[str] = field(default_factory=list)


@dataclass
class Calibration:
    """Physical calibration parameters."""
    # Dimensionless multiplier in z_calibrated = z_raw * z_scaling_factor.
    z_scaling_factor: float = 1.0
    pixel_size: float = 4.0         # nm per pixel for rendering
    loc_precision: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )


@dataclass
class Channel:
    """Color-channel information (DCR-based 2-color separation)."""
    num_channels: int = 1
    cut1: float = 0.5
    cut2: float = 1.0
    do_trace: bool = False
    keep_ch3: bool = False
    dcr: np.ndarray = field(default_factory=lambda: np.empty(0))
    dcr_trace: np.ndarray = field(default_factory=lambda: np.empty(0))


def _default_mfx_roles(attrs: AttrStore) -> dict[str, str]:
    roles: dict[str, str] = {}
    candidates = {
        "position": ("loc", "loc_x"),
        "track_id": ("tid",),
        "iteration": ("itr",),
        "valid": ("vld", "ftr"),
        "confidence": ("cfr",),
        "time": ("tim",),
    }
    for role, names in candidates.items():
        for name in names:
            if name in attrs:
                roles[role] = name
                break
    return roles


def _default_mfx_meta(attrs: AttrStore) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for name in attrs.keys():
        if name.endswith("_nm"):
            meta[name] = {"unit": "nm", "component": "mfx"}
        elif name.startswith("loc_") or name in {"loc", "lnc", "ext"}:
            meta[name] = {"unit": "m", "component": "mfx"}
        elif name == "tim":
            meta[name] = {"unit": "s", "component": "mfx"}
        else:
            meta[name] = {"component": "mfx"}
    return meta


# ---------------------------------------------------------------------------
# MinfluxDataset — the main object
# ---------------------------------------------------------------------------

@dataclass
class MinfluxDataset:
    """
    One loaded MINFLUX dataset.

    Equivalent to a single element of the MATLAB ``app.data`` struct array.
    """

    file: FileInfo
    prop: DataProp = field(default_factory=DataProp)
    attr: AttrStore = field(default_factory=AttrStore)
    cali: Calibration = field(default_factory=Calibration)
    channel: Channel = field(default_factory=Channel)
    components: DatasetComponents = field(default_factory=DatasetComponents)

    def __post_init__(self) -> None:
        self._adopt_filter_attr()
        self.attr.seal_coordinates()
        if self.components.mfx is None:
            self.components.mfx = AttributeComponent(
                self.attr,
                roles=_default_mfx_roles(self.attr),
                meta=_default_mfx_meta(self.attr),
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short display name (filename without path)."""
        return self.file.name

    @property
    def mfx(self) -> AttributeComponent:
        """Localization component; currently backed by ``attr`` for compatibility."""
        if self.components.mfx is None:
            self.components.mfx = AttributeComponent(
                self.attr,
                roles=_default_mfx_roles(self.attr),
                meta=_default_mfx_meta(self.attr),
            )
        return self.components.mfx

    @property
    def mbm(self) -> AttributeComponent | None:
        """Reference/bead component when available."""
        return self.components.mbm

    @mbm.setter
    def mbm(self, component: AttributeComponent | None) -> None:
        self.components.mbm = component

    @property
    def metadata(self) -> dict[str, Any]:
        return self.components.metadata

    @property
    def derived(self) -> AttrStore:
        return self.components.derived

    @property
    def mfx_raw(self) -> AttrStore:
        """All-iteration, all-measurement raw localization store (no vld/itr filtering)."""
        return self.components.mfx_raw

    @property
    def derived_last(self) -> AttrStore:
        """Derived attributes at the last-valid selection of ``mfx_raw``."""
        return self.components.derived_last

    @property
    def state(self) -> dict[str, Any]:
        return self.components.state

    @property
    def loc_nm(self) -> np.ndarray:
        """
        (N, 3) array of localizations in **nanometres**, with calibrated Z.

        The Z scaling factor is applied here as a **view only**: the stored raw
        ``loc_z`` is never modified, so this property can be read any number of
        times and the factor can be reset via :meth:`set_z_scaling_factor`
        without double-applying it. Code that needs the **raw, unscaled** z
        (for example, trace-anisotropy estimation) must read ``loc_z`` (or
        ``mfx_get(ds, "loc_z", ...)``) and convert it to nanometres directly.
        """
        x = np.asarray(self.attr.get("loc_x", np.empty(0)), dtype=float) * 1e9
        y = np.asarray(self.attr.get("loc_y", np.empty(0)), dtype=float) * 1e9
        z = np.asarray(self.attr.get("loc_z", np.empty(0)), dtype=float) * 1e9 * self.cali.z_scaling_factor
        return np.column_stack([x, y, z])

    @property
    def xnm(self) -> np.ndarray:
        """Localization x in **nanometres** (canonical coordinate view)."""
        return np.asarray(self.attr.get("loc_x", np.empty(0)), dtype=float) * 1e9

    @property
    def ynm(self) -> np.ndarray:
        """Localization y in **nanometres**."""
        return np.asarray(self.attr.get("loc_y", np.empty(0)), dtype=float) * 1e9

    @property
    def znm(self) -> np.ndarray:
        """Localization z in **nanometres**, Z-scaled (view only)."""
        return (np.asarray(self.attr.get("loc_z", np.empty(0)), dtype=float)
                * 1e9 * self.cali.z_scaling_factor)

    @property
    def z_scaling_factor_provenance(self) -> dict[str, Any]:
        """How the current Z scaling factor value was set: ``{value, source, history}``.

        ``applied_as_view`` is always True: the factor only rescales the z view
        (:attr:`loc_nm`); it is never baked into raw ``loc_z``.
        """
        return self.metadata.get("z_scaling_factor_provenance", {
            "value": float(getattr(self.cali, "z_scaling_factor", 1.0) or 1.0),
            "source": "default",
            "applied_as_view": True,
            "history": [],
        })

    def set_z_scaling_factor(self, value: float, *, source: str = "manual") -> float:
        """Set the Z scaling factor and record its provenance.

        The factor is applied to z only as a display view (see :attr:`loc_nm`);
        raw ``loc_z`` in ``attr``/``mfx_raw`` is never modified, so it can be
        reset without double-scaling the raw data. ``source`` (e.g.
        ``"auto"``, ``"manual"``, ``"fixed"``, ``"2D"``) is appended to the
        change history in ``metadata["z_scaling_factor_provenance"]``.
        """
        value = float(value)
        prov = self.metadata.get("z_scaling_factor_provenance")
        if prov is None:
            prov = {
                "value": float(getattr(self.cali, "z_scaling_factor", 1.0) or 1.0),
                "source": "default",
                "applied_as_view": True,
                "history": [],
            }
            self.metadata["z_scaling_factor_provenance"] = prov
        self.cali.z_scaling_factor = value
        prov["value"] = value
        prov["source"] = str(source)
        prov["applied_as_view"] = True
        prov["history"].append({
            "value": value,
            "source": str(source),
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        return value

    @property
    def has_localizations(self) -> bool:
        return "loc_x" in self.attr or "loc_y" in self.attr

    @property
    def image_data(self) -> np.ndarray | None:
        value = self.attr.get("image_data", None)
        return None if value is None else np.asarray(value)

    @property
    def image_origin_nm(self) -> tuple[float, float]:
        return (
            float(self.attr.get("image_origin_x_nm", 0.0)),
            float(self.attr.get("image_origin_y_nm", 0.0)),
        )

    @property
    def image_pixel_size_nm(self) -> tuple[float, float]:
        sx = float(self.attr.get("image_pixel_size_x_nm", self.cali.pixel_size))
        sy = float(self.attr.get("image_pixel_size_y_nm", self.cali.pixel_size))
        return sx, sy

    def _adopt_filter_attr(self) -> None:
        """Make ``attr["ftr"]`` the virtual, read-only view of ``filter_mask``.

        A loader puts an ``ftr`` array among the attributes it builds (all
        True, or a saved filter). It is adopted here once: a mask that filters
        anything becomes the dataset's mask unless one is already set, and an
        all-True one is simply the default. ``ftr`` then stays where it was in
        the attribute order and is read from the mask from now on.
        """
        raw = object.__getattribute__(self.attr, "_d")
        value = raw.get("ftr")
        if value is not None and not isinstance(value, _Virtual):
            arr = np.asarray(value, dtype=bool)
            if "filter_mask" not in self.state and arr.size and not arr.all():
                self.state["filter_mask"] = np.array(arr, dtype=bool)
        self.attr.bind_virtual("ftr", self, "filter_mask")

    @property
    def filter_mask(self) -> np.ndarray:
        """Boolean mask; True = localization passes all active filters.

        The one place the mask lives is ``state["filter_mask"]`` (absent until
        a filter is set: everything passes). ``attr["ftr"]`` reads this. The
        returned array is a **read-only view**: change the mask by assigning
        ``filter_mask`` (and notifying), never by editing it in place.
        """
        stored = self.state.get("filter_mask", None)
        if stored is None:
            n = int(self.prop.num_loc)
            default = getattr(self, "_all_pass", None)
            if default is None or default.size != n or default.flags.writeable:
                default = np.ones(n, dtype=bool)
                default.flags.writeable = False
                object.__setattr__(self, "_all_pass", default)
            return default
        view = np.asarray(stored, dtype=bool).view()
        view.flags.writeable = False
        return view

    @filter_mask.setter
    def filter_mask(self, mask: np.ndarray) -> None:
        # A copy: the caller's array must not be able to change the mask later.
        self.state["filter_mask"] = np.array(mask, dtype=bool)

    def summary(self) -> str:
        """One-line string suitable for a status bar."""
        return (
            f"{self.name}  |  "
            f"{self.prop.num_loc:,} loc  |  "
            f"{self.prop.num_traces:,} traces  |  "
            f"{self.prop.num_dim}D"
        )

    def __repr__(self) -> str:
        return f"MinfluxDataset({self.name!r}, {self.prop.num_loc:,} loc)"


def build_localization_dataset(
    *,
    name: str,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    z_nm: np.ndarray | None = None,
    folder: str = "",
    datetime_str: str = "",
    attrs: dict[str, np.ndarray] | None = None,
    loc_precision_nm: np.ndarray | None = None,
    tid: np.ndarray | None = None,
    tim: np.ndarray | None = None,
    is_minflux: bool | None = None,
    source_version: str = "imported",
    prefs: dict | None = None,
) -> MinfluxDataset:
    """Canonical builder for generic localizations given in **nanometres**.

    Coordinates are stored as canonical ``loc_x``/``loc_y``/``loc_z`` in
    **metres** (z zero-filled when 2-D) and run through the same property +
    derived-attribute pipeline as the native loaders, so the result renders,
    filters, and runs the analyses like a native dataset.

    ``tid`` is the trace id: pass the real one (→ MINFLUX-eligible) or leave
    ``None`` to synthesise a 1..N index (→ non-MINFLUX). ``is_minflux`` defaults
    to "a real tid was supplied". Sets ``metadata`` flags accordingly.
    """
    from .loader import _add_derived_attributes, _compute_properties

    x_nm = np.asarray(x_nm, dtype=float).ravel()
    y_nm = np.asarray(y_nm, dtype=float).ravel()
    if x_nm.shape != y_nm.shape:
        raise ValueError("x_nm and y_nm must have the same shape")
    n = int(x_nm.size)
    if z_nm is None:
        z_nm = np.zeros(n, dtype=float)
    else:
        z_nm = np.asarray(z_nm, dtype=float).ravel()
        if z_nm.shape != x_nm.shape:
            raise ValueError("z_nm must have the same shape as x_nm/y_nm")

    has_real_tid = tid is not None
    tid_arr = (np.arange(1, n + 1, dtype=np.int64)
               if tid is None else np.asarray(tid).ravel())
    tim_arr = np.zeros(n, dtype=float) if tim is None else np.asarray(tim, dtype=float).ravel()

    data: dict[str, np.ndarray] = {
        "loc_x": x_nm * 1.0e-9,        # canonical metres — single coordinate store
        "loc_y": y_nm * 1.0e-9,
        "loc_z": z_nm * 1.0e-9,
        "tid": tid_arr,
        "tim": tim_arr,
    }
    if loc_precision_nm is not None:
        data["loc_precision_xy"] = np.asarray(loc_precision_nm, dtype=float).ravel()
    if attrs:
        for key, value in attrs.items():
            if key in data:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == n:
                data[key] = arr

    attr_store = AttrStore(data)
    prop = _compute_properties(attr_store, n, 1, prefs=prefs or {})
    attr_store = _add_derived_attributes(attr_store, prop, prefs=prefs or {})
    prop.attr_names = attr_store.keys()

    cali = Calibration()
    if loc_precision_nm is not None:
        cali.loc_precision = np.asarray(loc_precision_nm, dtype=float)

    ds = MinfluxDataset(
        file=FileInfo(
            name=name,
            folder=folder,
            datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        ),
        prop=prop,
        attr=attr_store,
        cali=cali,
        channel=Channel(),
    )
    ds.metadata["source_version"] = source_version
    ds.metadata["is_minflux"] = bool(has_real_tid if is_minflux is None else is_minflux)
    ds.metadata["has_real_tid"] = bool(has_real_tid)
    return ds


def build_image_dataset(
    *,
    name: str,
    image: np.ndarray,
    pixel_size_nm: float | tuple[float, float] = 1.0,
    origin_nm: tuple[float, float] = (0.0, 0.0),
    folder: str = "",
    datetime_str: str = "",
    attrs: dict[str, np.ndarray] | None = None,
) -> MinfluxDataset:
    """Create a dataset carrying an image payload for the render window."""
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError("image must be at least 2D")
    if isinstance(pixel_size_nm, tuple):
        sx, sy = pixel_size_nm
    else:
        sx = sy = float(pixel_size_nm)
    data = {
        "image_data": image,
        "image_origin_x_nm": float(origin_nm[0]),
        "image_origin_y_nm": float(origin_nm[1]),
        "image_pixel_size_x_nm": float(sx),
        "image_pixel_size_y_nm": float(sy),
    }
    if attrs:
        for key, value in attrs.items():
            data[key] = np.asarray(value)
    attr_store = AttrStore(data)
    arr = np.asarray(image)
    is_color_2d = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    is_stack = (
        (arr.ndim == 3 and not is_color_2d)
        or (arr.ndim == 4 and arr.shape[-1] in (3, 4))
    )
    stack_depth = int(arr.shape[0]) if is_stack else 1
    prop = DataProp(
        num_loc=0,
        num_itr=0,
        num_dim=3 if stack_depth > 1 else 2,
        num_traces=0,
        attr_names=attr_store.keys(),
    )
    return MinfluxDataset(
        file=FileInfo(
            name=name,
            folder=folder,
            datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        ),
        prop=prop,
        attr=attr_store,
        cali=Calibration(pixel_size=float(sx)),
        channel=Channel(),
    )
