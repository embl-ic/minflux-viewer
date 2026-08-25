"""Detection and segmentation of objects with a *known* geometry in 2-D.

The problem this solves is "I know what shape my objects are, find them" — for
example rod-shaped (obround / capsule) *E. coli* cells in a MINFLUX projection.
Objects may touch one another and may be clipped by the field border.

Method
------
The pipeline is deliberately *shape-prior first* rather than filter-bank first:

1. **Field** — an image (+ pixel size) or a point cloud is reduced to one scalar
   density raster at ``detection_pixel_nm``.  This is what lets the same code
   segment a rendered TIFF and a localization table.
2. **Foreground** — smooth, threshold (Otsu by default), fill holes, drop specks.
   This stage only has to *localize* candidate material, not delineate it, so a
   single global threshold is adequate and cheap.
3. **Per connected component, choose the instance count by explanation cost.**
   For ``k = 1, 2, …`` the component is explained by ``k`` shape instances and
   the winner minimises a description-length cost

       ``cost(k) = unexplained_area / nominal_object_area + instance_cost * k``

   where ``unexplained_area`` is the symmetric difference between the union of
   the fitted models and the component.  Overlap alone (IoU/Dice) cannot choose
   ``k``: it rises monotonically with model complexity, so three capsules always
   wrap a blob at least as well as two.  Expressing the residual in units of one
   *nominal object* makes the penalty scale-free — an extra instance has to earn
   its place by explaining a real fraction of an object.
4. **Fit** — each instance is refined by direct maximisation of the overlap
   between its rasterised model and the foreground pixels assigned to it,
   alternating assignment and refit (block coordinate ascent).

Why not a rotated template bank?  Two objects that touch *without a waist* — the
common failure case for rod-shaped bacteria — cannot be separated by image
concavity (erosion/watershed never splits them) nor by a greedy "keep the next
best-scoring template" rule, which spends instance slots on fragments.  Only the
shape prior separates them, and the cleanest way to apply it is to ask how many
instances of the known geometry explain the blob.

Clipped objects are first class.  Overlap is measured only over the *visible*
field, so an object whose centre lies outside the frame is fitted from the part
that is visible and is reported with ``clipped=True`` and a ``visible_fraction``.

Genericity
----------
A shape is one :class:`ShapeModel`, defined by a **signed distance function**
(negative inside, nm).  Masks, outlines, membership, the fit objective and the
instance assignment all derive from that single primitive, so registering a new
geometry is a few lines.  ``capsule``, ``ellipse``, ``rectangle`` and ``disk``
ship here.

This module is pure NumPy/SciPy and has no Qt dependency.

Note: :mod:`minflux_viewer.analysis.hlyb_clustering` carries its own private
Otsu/disk helpers that mirror specific MATLAB behaviour inside a validated
pipeline; they are intentionally left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, pi
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

__all__ = [
    "ShapeModel",
    "SizeSpec",
    "SHAPE_MODELS",
    "get_shape_model",
    "ShapePrior",
    "ShapeSegmentationConfig",
    "ShapeInstance",
    "ShapeSegmentationResult",
    "ScalarField",
    "field_from_image",
    "field_from_points",
    "otsu_threshold",
    "segment_shapes",
    "segment_shapes_in_image",
    "segment_shapes_in_points",
    "instance_mask",
    "instance_outline",
]


# --------------------------------------------------------------------------- #
# Shape registry — one signed distance function per geometry
# --------------------------------------------------------------------------- #
def _to_local(x, y, cx: float, cy: float, angle_deg: float):
    """World nm -> shape-local (along-axis, across-axis) nm."""
    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = np.asarray(x, dtype=float) - float(cx)
    dy = np.asarray(y, dtype=float) - float(cy)
    return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t


def _capsule_sdf(x, y, cx, cy, angle_deg, size) -> np.ndarray:
    length, width = float(size[0]), float(size[1])
    along, across = _to_local(x, y, cx, cy, angle_deg)
    radius = 0.5 * min(width, length)
    half_line = max(0.5 * length - radius, 0.0)
    return np.hypot(across, np.maximum(np.abs(along) - half_line, 0.0)) - radius


def _capsule_area(size) -> float:
    length, width = float(size[0]), float(size[1])
    width = min(width, length)
    return width * max(length - width, 0.0) + pi * (0.5 * width) ** 2


def _capsule_outline(cx, cy, angle_deg, size, n: int) -> np.ndarray:
    length, width = float(size[0]), float(size[1])
    theta = np.radians(float(angle_deg))
    axis = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    centre = np.array([float(cx), float(cy)])
    radius = 0.5 * min(width, length)
    half_line = max(0.5 * length - radius, 0.0)
    n_cap = max(int(n) // 2, 3)
    angles = np.linspace(-pi / 2.0, pi / 2.0, n_cap)
    end = centre + half_line * axis
    start = centre - half_line * axis
    cap_end = end + radius * (
        np.cos(angles)[:, None] * axis + np.sin(angles)[:, None] * normal)
    cap_start = start + radius * (
        -np.cos(angles)[:, None] * axis - np.sin(angles)[:, None] * normal)
    return np.vstack([cap_end, cap_start, cap_end[:1]])


def _arc_capsule_sdf(x, y, cx, cy, angle_deg, size) -> np.ndarray:
    """Capsule whose spine is a circular arc — a *bent* rod.

    ``size`` is ``(length_nm, width_nm, bend_deg)`` where ``bend_deg`` is the
    total turn along the spine; 0 reduces exactly to a straight capsule.  Real
    rod-shaped bacteria are commonly curved, and a straight model caps the
    achievable overlap on them at roughly 0.7 no matter how well it is fitted.
    """
    length, width = float(size[0]), float(size[1])
    bend = np.radians(float(size[2]))
    radius = 0.5 * min(width, length)
    half_spine = max(0.5 * length - radius, 0.0)
    along, across = _to_local(x, y, cx, cy, angle_deg)
    if abs(bend) < 1e-3 or half_spine <= 0.0:
        return np.hypot(across, np.maximum(np.abs(along) - half_spine, 0.0)) - radius
    # Spine: arc of curvature radius R through the local origin, centred at
    # (0, R), sweeping +-bend/2 so the tips bow toward +across.
    half_turn = 0.5 * abs(bend)
    curve_r = half_spine / half_turn
    sign = 1.0 if bend >= 0 else -1.0
    across_s = sign * across
    delta_y = across_s - curve_r
    rho = np.hypot(along, delta_y)
    theta = np.arctan2(along, -delta_y)
    on_arc = np.abs(theta) <= half_turn
    radial = np.abs(rho - curve_r)
    tip_along = curve_r * np.sin(half_turn)
    tip_across = curve_r * (1.0 - np.cos(half_turn))
    to_tip = np.minimum(
        np.hypot(along - tip_along, across_s - tip_across),
        np.hypot(along + tip_along, across_s - tip_across))
    return np.where(on_arc, radial, to_tip) - radius


def _arc_capsule_area(size) -> float:
    """Exact area of the arc-swept capsule (spine length x width + cap disc)."""
    length, width = float(size[0]), float(size[1])
    width = min(width, length)
    return width * max(length - width, 0.0) + pi * (0.5 * width) ** 2


def _arc_capsule_outline(cx, cy, angle_deg, size, n: int) -> np.ndarray:
    """Offset the arc spine by +-radius, closed by the two end caps."""
    length, width = float(size[0]), float(size[1])
    bend = np.radians(float(size[2]))
    radius = 0.5 * min(width, length)
    half_spine = max(0.5 * length - radius, 0.0)
    n_spine = max(int(n) // 2, 8)
    if abs(bend) < 1e-3 or half_spine <= 0.0:
        return _capsule_outline(cx, cy, angle_deg, (length, width), n)
    half_turn = 0.5 * abs(bend)
    curve_r = half_spine / half_turn
    sign = 1.0 if bend >= 0 else -1.0
    t = np.linspace(-half_turn, half_turn, n_spine)
    spine_u = curve_r * np.sin(t)
    spine_v = sign * curve_r * (1.0 - np.cos(t))
    # Unit normal of the arc in the local frame.
    nu = -np.sin(t)
    nv = sign * np.cos(t)
    left = np.column_stack([spine_u + radius * nu, spine_v + radius * nv])
    right = np.column_stack([spine_u - radius * nu, spine_v - radius * nv])[::-1]
    cap_t = np.linspace(0.0, pi, max(int(n) // 4, 4))
    end_n = np.array([-np.sin(half_turn), sign * np.cos(half_turn)])
    end_tan = np.array([np.cos(half_turn), sign * np.sin(half_turn)])
    end_c = np.array([curve_r * np.sin(half_turn),
                      sign * curve_r * (1.0 - np.cos(half_turn))])
    cap_end = end_c + radius * (np.cos(cap_t)[:, None] * end_n
                                + np.sin(cap_t)[:, None] * end_tan)
    start_n = np.array([np.sin(half_turn), sign * np.cos(half_turn)])
    start_tan = np.array([-np.cos(half_turn), sign * np.sin(half_turn)])
    start_c = np.array([-curve_r * np.sin(half_turn),
                        sign * curve_r * (1.0 - np.cos(half_turn))])
    cap_start = start_c + radius * (np.cos(cap_t)[:, None] * start_n
                                    + np.sin(cap_t)[:, None] * start_tan)
    local = np.vstack([left, cap_end, right, cap_start])
    local = np.vstack([local, local[:1]])
    theta = np.radians(float(angle_deg))
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return local @ rot.T + np.array([float(cx), float(cy)])


def _ellipse_sdf(x, y, cx, cy, angle_deg, size) -> np.ndarray:
    a = max(0.5 * float(size[0]), 1e-9)
    b = max(0.5 * float(size[1]), 1e-9)
    along, across = _to_local(x, y, cx, cy, angle_deg)
    # Normalised implicit rescaled to a length: zero on the boundary, negative
    # inside, monotonic outward — all the fit and the mask need.
    return (np.hypot(along / a, across / b) - 1.0) * min(a, b)


def _ellipse_area(size) -> float:
    return pi * 0.25 * float(size[0]) * float(size[1])


def _ellipse_outline(cx, cy, angle_deg, size, n: int) -> np.ndarray:
    a, b = 0.5 * float(size[0]), 0.5 * float(size[1])
    theta = np.radians(float(angle_deg))
    axis = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    t = np.linspace(0.0, 2.0 * pi, max(int(n), 8))
    pts = (np.array([float(cx), float(cy)])
           + (a * np.cos(t))[:, None] * axis + (b * np.sin(t))[:, None] * normal)
    return np.vstack([pts, pts[:1]])


def _rectangle_sdf(x, y, cx, cy, angle_deg, size) -> np.ndarray:
    along, across = _to_local(x, y, cx, cy, angle_deg)
    qx = np.abs(along) - 0.5 * float(size[0])
    qy = np.abs(across) - 0.5 * float(size[1])
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def _rectangle_area(size) -> float:
    return float(size[0]) * float(size[1])


def _rectangle_outline(cx, cy, angle_deg, size, n: int) -> np.ndarray:
    theta = np.radians(float(angle_deg))
    axis = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    hl, hw = 0.5 * float(size[0]), 0.5 * float(size[1])
    centre = np.array([float(cx), float(cy)])
    corners = [centre + sx * hl * axis + sy * hw * normal
               for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    return np.vstack(corners + corners[:1])


def _disk_sdf(x, y, cx, cy, angle_deg, size) -> np.ndarray:
    along, across = _to_local(x, y, cx, cy, 0.0)
    return np.hypot(along, across) - 0.5 * float(size[0])


def _disk_area(size) -> float:
    return pi * (0.5 * float(size[0])) ** 2


def _disk_outline(cx, cy, angle_deg, size, n: int) -> np.ndarray:
    return _ellipse_outline(cx, cy, 0.0, (size[0], size[0]), n)


@dataclass(frozen=True)
class SizeSpec:
    """One size parameter of a shape, with the range a UI should offer for it.

    Keeping this beside the geometry is what makes registering a new shape a
    single-file change: the segmentation dialog builds its min/max controls from
    these specs rather than hard-coding a widget per model.
    """

    name: str
    label: str
    default_lo: float
    default_hi: float
    limit_lo: float
    limit_hi: float
    decimals: int = 0
    step: float = 50.0
    suffix: str = " nm"
    tooltip: str = ""


@dataclass(frozen=True)
class ShapeModel:
    """A geometry defined by its signed distance function (nm, <0 inside).

    ``size`` is the tuple of shape parameters named by :attr:`size_names`; the
    full parameter vector used by the fitter is ``[cx, cy, angle_deg, *size]``.
    """

    key: str
    label: str
    description: str
    size_specs: tuple[SizeSpec, ...]
    sdf: Callable[..., np.ndarray]
    area: Callable[[Sequence[float]], float]
    outline: Callable[..., np.ndarray]
    #: ``size -> (extent_along_axis, extent_across_axis)`` in nm
    extent: Callable[[Sequence[float]], tuple[float, float]]
    #: ``(extent_along, extent_across) -> size``; inverse of :attr:`extent`
    size_from_extent: Callable[[float, float], tuple[float, ...]]
    #: True when the geometry is unchanged by a 180 degree rotation
    axial: bool = True
    #: True when the geometry has no orientation at all (angle is meaningless)
    isotropic: bool = False

    @property
    def size_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.size_specs)

    @property
    def n_size(self) -> int:
        return len(self.size_specs)


_LENGTH = SizeSpec("length_nm", "Length", 1400.0, 4000.0, 10.0, 100_000.0,
                   tooltip="Tip-to-tip length of the object.")
_WIDTH = SizeSpec("width_nm", "Width", 600.0, 1200.0, 10.0, 100_000.0,
                  tooltip="Width across the long axis.")
_BEND = SizeSpec("bend_deg", "Bend", -80.0, 80.0, -180.0, 180.0,
                 decimals=1, step=5.0, suffix=" deg",
                 tooltip="Total turn along the spine; 0 is a straight rod.")


SHAPE_MODELS: dict[str, ShapeModel] = {
    "capsule": ShapeModel(
        key="capsule",
        label="Capsule (obround / stadium)",
        description="A rectangle capped by two half-discs — the shape of a "
                    "rod-shaped bacterium such as E. coli. Parameters are the "
                    "total tip-to-tip length and the width (= cap diameter).",
        size_specs=(_LENGTH, _WIDTH),
        sdf=_capsule_sdf,
        area=_capsule_area,
        outline=_capsule_outline,
        extent=lambda s: (float(s[0]), float(s[1])),
        size_from_extent=lambda a, c: (a, c),
    ),
    "arc_capsule": ShapeModel(
        key="arc_capsule",
        label="Curved capsule (bent rod)",
        description="A capsule whose spine is a circular arc — the shape of a "
                    "real, slightly bent rod-shaped bacterium. Adds a bend "
                    "angle (total turn along the spine, degrees) to the "
                    "capsule's length and width; bend 0 is a straight capsule.",
        size_specs=(_LENGTH, _WIDTH, _BEND),
        sdf=_arc_capsule_sdf,
        area=_arc_capsule_area,
        outline=_arc_capsule_outline,
        extent=lambda s: (float(s[0]), float(s[1])),
        size_from_extent=lambda a, c: (a, c, 0.0),
    ),
    "ellipse": ShapeModel(
        key="ellipse",
        label="Ellipse",
        description="A filled ellipse of the given major and minor axis length.",
        size_specs=(SizeSpec("major_nm", "Major axis", 1400.0, 4000.0,
                             10.0, 100_000.0),
                    SizeSpec("minor_nm", "Minor axis", 600.0, 1200.0,
                             10.0, 100_000.0)),
        sdf=_ellipse_sdf,
        area=_ellipse_area,
        outline=_ellipse_outline,
        extent=lambda s: (float(s[0]), float(s[1])),
        size_from_extent=lambda a, c: (a, c),
    ),
    "rectangle": ShapeModel(
        key="rectangle",
        label="Rectangle",
        description="A filled, freely rotatable rectangle of the given length "
                    "and width.",
        size_specs=(_LENGTH, _WIDTH),
        sdf=_rectangle_sdf,
        area=_rectangle_area,
        outline=_rectangle_outline,
        extent=lambda s: (float(s[0]), float(s[1])),
        size_from_extent=lambda a, c: (a, c),
    ),
    "disk": ShapeModel(
        key="disk",
        label="Disk",
        description="A filled circle of the given diameter (orientation-free).",
        size_specs=(SizeSpec("diameter_nm", "Diameter", 600.0, 1200.0,
                             10.0, 100_000.0),),
        sdf=_disk_sdf,
        area=_disk_area,
        outline=_disk_outline,
        extent=lambda s: (float(s[0]), float(s[0])),
        size_from_extent=lambda a, c: (0.5 * (a + c),),
        isotropic=True,
    ),
}


def get_shape_model(key: str) -> ShapeModel:
    try:
        return SHAPE_MODELS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SHAPE_MODELS))
        raise ValueError(
            f"unknown shape model {key!r}; known models: {known}") from exc


# --------------------------------------------------------------------------- #
# Prior, configuration and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShapePrior:
    """The *known* geometry: which shape, and the plausible range of each size.

    The bounds are a hard constraint on the fit.  They are what separates two
    objects that touch without a visible waist, so they should describe the real
    spread of the population rather than being opened up "just in case".
    """

    model_key: str = "capsule"
    size_lo: tuple[float, ...] = (1400.0, 700.0)
    size_hi: tuple[float, ...] = (3400.0, 1150.0)

    @classmethod
    def defaults(cls, model_key: str) -> "ShapePrior":
        """The prior a model declares for itself, from its :class:`SizeSpec`s."""
        specs = get_shape_model(model_key).size_specs
        return cls(model_key,
                   tuple(spec.default_lo for spec in specs),
                   tuple(spec.default_hi for spec in specs))

    @classmethod
    def capsule(cls, *, length_nm: tuple[float, float],
                width_nm: tuple[float, float]) -> "ShapePrior":
        return cls("capsule", (length_nm[0], width_nm[0]),
                   (length_nm[1], width_nm[1]))

    @classmethod
    def arc_capsule(cls, *, length_nm: tuple[float, float],
                    width_nm: tuple[float, float],
                    bend_deg: tuple[float, float] = (-75.0, 75.0)) -> "ShapePrior":
        return cls("arc_capsule",
                   (length_nm[0], width_nm[0], bend_deg[0]),
                   (length_nm[1], width_nm[1], bend_deg[1]))

    @classmethod
    def disk(cls, *, diameter_nm: tuple[float, float]) -> "ShapePrior":
        return cls("disk", (diameter_nm[0],), (diameter_nm[1],))

    @property
    def model(self) -> ShapeModel:
        return get_shape_model(self.model_key)

    def nominal(self) -> tuple[float, ...]:
        """Mid-range size — the "typical" object used for area/count estimates."""
        return tuple(0.5 * (lo + hi) for lo, hi in zip(self.size_lo, self.size_hi))

    def nominal_area_nm2(self) -> float:
        return float(self.model.area(self.nominal()))

    def clip(self, size: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(np.clip(v, lo, hi))
                     for v, lo, hi in zip(size, self.size_lo, self.size_hi))

    def validate(self) -> None:
        model = self.model
        if len(self.size_lo) != model.n_size or len(self.size_hi) != model.n_size:
            raise ValueError(
                f"{model.key!r} takes {model.n_size} size parameter(s) "
                f"({', '.join(model.size_names)}); got {len(self.size_lo)}")
        for name, lo, hi in zip(model.size_names, self.size_lo, self.size_hi):
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"{name}: bounds must be finite")
            # A length must be positive; a signed parameter such as a bend
            # angle legitimately spans zero, so key the check on the unit.
            if name.endswith("_nm") and lo <= 0:
                raise ValueError(f"{name}: lower bound must be positive")
            if hi < lo:
                raise ValueError(
                    f"{name}: upper bound {hi} is below lower bound {lo}")


@dataclass(frozen=True)
class ShapeSegmentationConfig:
    """Tuning for :func:`segment_shapes`.  Physical values are nm."""

    detection_pixel_nm: float = 20.0
    #: Foreground smoothing.  ``None`` -> ``smoothing_width_ratio`` x the nominal
    #: object width, which closes gaps inside one object without merging
    #: neighbours.
    smoothing_nm: float | None = None
    smoothing_width_ratio: float = 1.0 / 6.0
    #: Absolute foreground threshold on the smoothed field.  ``None`` -> Otsu.
    threshold: float | None = None
    #: Alternative to ``threshold``: keep the brightest ``1 - q`` of pixels.
    threshold_quantile: float | None = None
    fill_holes: bool = True
    closing_nm: float = 0.0
    #: Components below this fraction of the nominal model area are discarded.
    min_component_area_frac: float = 0.25
    #: Instance-count search cap, per component and overall.
    max_instances_per_component: int = 6
    max_instances: int = 128
    #: Description-length price of one extra instance, in units of the nominal
    #: object area: an added instance must reduce the unexplained area by more
    #: than this fraction of one object to be kept.  Raise it if touching
    #: objects are over-split, lower it if they stay merged.
    instance_cost: float = 0.25
    #: Instances explaining their own territory worse than this are dropped.
    min_instance_iou: float = 0.30
    #: Alternating assignment/refit rounds.
    refine_rounds: int = 6
    #: Nelder-Mead evaluations per instance refit.
    max_fit_evals: int = 600
    max_detection_pixels: int = 4_000_000


@dataclass(frozen=True)
class ShapeInstance:
    """One fitted object in physical (nm) image coordinates.

    ``size_nm`` is named by ``model.size_names``.  For a ``clipped`` instance the
    size is an *extrapolation* from the visible part and is only as trustworthy
    as ``visible_fraction`` — a cell half outside the frame does not determine
    its own length.
    """

    model_key: str
    center_nm: tuple[float, float]
    angle_deg: float
    size_nm: tuple[float, ...]
    iou: float
    visible_fraction: float
    clipped: bool
    area_nm2: float
    component_id: int

    @property
    def model(self) -> ShapeModel:
        return get_shape_model(self.model_key)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.center_nm[0], self.center_nm[1], self.angle_deg,
                         *self.size_nm], dtype=float)

    def size(self) -> dict[str, float]:
        return {name: float(value)
                for name, value in zip(self.model.size_names, self.size_nm)}

    def as_dict(self) -> dict:
        return {
            "model": self.model_key,
            "center_x_nm": float(self.center_nm[0]),
            "center_y_nm": float(self.center_nm[1]),
            "angle_deg": float(self.angle_deg),
            **self.size(),
            "area_nm2": float(self.area_nm2),
            "iou": float(self.iou),
            "visible_fraction": float(self.visible_fraction),
            "clipped": bool(self.clipped),
            "component_id": int(self.component_id),
        }


@dataclass
class ShapeSegmentationResult:
    """Fitted instances plus their rasterisations on the source grid."""

    instances: list[ShapeInstance] = field(default_factory=list)
    #: ``(K, H, W)`` model masks.  These *may overlap*: in a 2-D projection the
    #: pixels of a true overlap cannot be assigned uniquely.
    masks: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=bool))
    #: Exclusive labelling for convenience — 0 = background, i = instance i-1,
    #: ties broken by the smaller signed distance.
    labels: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    union_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool))
    overlap_count: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.uint16))
    #: Thresholded foreground, on the *detection* grid (``detection_pixel_nm``
    #: and the field origin) — not the source grid that ``masks``/``labels`` use.
    foreground: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool))
    #: The resampled density the fit ran on, also on the detection grid.
    detection_field: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=float))
    #: nm coordinate of the corner of detection pixel (0, 0), so a caller can
    #: place ``detection_field`` in the same frame as the fitted instances.
    detection_origin_nm: tuple[float, float] = (0.0, 0.0)
    #: Per input point, 0 = unassigned, i = instance i-1 (point-cloud entry only).
    point_labels: np.ndarray | None = None
    detection_pixel_nm: float = 0.0
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)


# --------------------------------------------------------------------------- #
# Scalar field
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarField:
    """A 2-D density raster with a physical calibration.

    ``values[row, col]``; ``origin_nm`` is the nm coordinate of the *corner* of
    pixel ``(0, 0)``, so pixel centres are ``origin + (index + 0.5) * pixel_nm``.
    """

    values: np.ndarray
    pixel_nm: float
    origin_nm: tuple[float, float] = (0.0, 0.0)
    #: The frame an object can be *seen* in, ``(x0, y0, x1, y1)`` nm — e.g. a
    #: MINFLUX acquisition ROI.  ``None`` means the raster itself.  It is kept
    #: separate from the raster extent because the two genuinely differ: points
    #: scatter slightly outside the scanned box, so the raster must cover the
    #: data while clipping has to be judged against the scanned frame.
    visible_bounds_nm: tuple[float, float, float, float] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.indices(self.values.shape, dtype=float)
        return (self.origin_nm[0] + (cols + 0.5) * self.pixel_nm,
                self.origin_nm[1] + (rows + 0.5) * self.pixel_nm)

    def bounds_nm(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        return (self.origin_nm[0], self.origin_nm[1],
                self.origin_nm[0] + w * self.pixel_nm,
                self.origin_nm[1] + h * self.pixel_nm)

    def visible_bounds(self) -> tuple[float, float, float, float]:
        """Frame used to decide whether an object is clipped."""
        return self.visible_bounds_nm or self.bounds_nm()


def field_from_image(image: np.ndarray, pixel_size_nm: float,
                     origin_nm: tuple[float, float] = (0.0, 0.0)) -> ScalarField:
    """Wrap a 2-D scalar image as a :class:`ScalarField`."""
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(
            f"image must be a 2-D scalar array, got shape {values.shape}")
    if not np.isfinite(float(pixel_size_nm)) or float(pixel_size_nm) <= 0:
        raise ValueError("pixel_size_nm must be a positive, finite value")
    return ScalarField(np.asarray(values, dtype=float), float(pixel_size_nm),
                       origin_nm)


def field_from_points(x_nm: np.ndarray, y_nm: np.ndarray, pixel_size_nm: float, *,
                      margin_nm: float = 0.0,
                      bounds_nm: tuple[float, float, float, float] | None = None
                      ) -> ScalarField:
    """Render a point cloud as a counts raster at ``pixel_size_nm``.

    ``bounds_nm`` is ``(x0, y0, x1, y1)`` — the *real* field of view, e.g. the
    MINFLUX acquisition ROI.  Supply it whenever it is known: without it the
    raster spans the data itself, so no object can ever be found to run off the
    edge and ``clipped`` is meaningless.  ``margin_nm`` is ignored when explicit
    bounds are given.
    """
    x = np.asarray(x_nm, dtype=float).ravel()
    y = np.asarray(y_nm, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0:
        raise ValueError("no finite points to render")
    px = float(pixel_size_nm)
    if px <= 0:
        raise ValueError("pixel_size_nm must be positive")
    x0 = float(x.min()) - float(margin_nm)
    y0 = float(y.min()) - float(margin_nm)
    x1 = float(x.max()) + float(margin_nm)
    y1 = float(y.max()) + float(margin_nm)
    visible = None
    if bounds_nm is not None:
        bx0, by0, bx1, by1 = (float(v) for v in bounds_nm)
        if bx1 <= bx0 or by1 <= by0:
            raise ValueError(
                f"bounds_nm must be (x0, y0, x1, y1) with x1 > x0 and y1 > y0, "
                f"got {bounds_nm}")
        visible = (bx0, by0, bx1, by1)
        # Cover both: the scanned frame decides clipping, but localizations
        # falling just outside it are real and must not be dropped.
        x0, y0 = min(x0, bx0), min(y0, by0)
        x1, y1 = max(x1, bx1), max(y1, by1)
    n_x = max(int(ceil((x1 - x0) / px)), 1)
    n_y = max(int(ceil((y1 - y0) / px)), 1)
    counts, _, _ = np.histogram2d(
        y, x, bins=(n_y, n_x),
        range=((y0, y0 + n_y * px), (x0, x0 + n_x * px)))
    return ScalarField(counts, px, (x0, y0), visible)


# --------------------------------------------------------------------------- #
# Raster helpers
# --------------------------------------------------------------------------- #
def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's between-class-variance threshold of a scalar array."""
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0 or data.max() <= data.min():
        return float(data.min()) if data.size else 0.0
    hist, edges = np.histogram(data, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    weight_bg = np.cumsum(hist).astype(float)
    weight_fg = total - weight_bg
    csum = np.cumsum(hist * centers)
    mean_bg = csum / np.maximum(weight_bg, 1.0)
    mean_fg = (csum[-1] - csum) / np.maximum(weight_fg, 1.0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def _resample(values: np.ndarray, source_px: float, target_px: float) -> np.ndarray:
    """Rebin a density raster.  Integer ratios are exact area sums."""
    ratio = float(target_px) / float(source_px)
    if np.isclose(ratio, 1.0, rtol=0, atol=1e-9):
        return np.asarray(values, dtype=float)
    whole = int(round(ratio))
    if whole >= 1 and np.isclose(ratio, whole, rtol=0, atol=1e-8):
        h, w = values.shape
        out_h, out_w = int(ceil(h / whole)), int(ceil(w / whole))
        padded = np.pad(np.asarray(values, dtype=float),
                        ((0, out_h * whole - h), (0, out_w * whole - w)))
        return padded.reshape(out_h, whole, out_w, whole).sum((1, 3))
    scale = float(source_px) / float(target_px)
    src = np.asarray(values, dtype=float)
    if scale < 1.0:
        src = ndi.gaussian_filter(src, max(0.0, 0.5 * (1.0 / scale - 1.0)))
    # ``zoom`` preserves density, not mass; rescale so the total is conserved.
    out = ndi.zoom(src, zoom=scale, order=1, mode="nearest", prefilter=False)
    return out / max(scale * scale, 1e-12)


def _disk_struct(radius_px: int) -> np.ndarray:
    r = max(int(radius_px), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _angle_axial(angle_deg: float) -> float:
    """Fold a 180 degree-periodic orientation into ``(-90, 90]``."""
    value = float(angle_deg) % 180.0
    return value - 180.0 if value > 90.0 else value


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _model_mask(model: ShapeModel, x_nm, y_nm, params) -> np.ndarray:
    return model.sdf(x_nm, y_nm, params[0], params[1], params[2], params[3:]) < 0.0


def _clip_params(model: ShapeModel, prior: ShapePrior,
                 params: np.ndarray) -> np.ndarray:
    out = np.asarray(params, dtype=float).copy()
    out[3:] = prior.clip(out[3:])
    if model.key in ("capsule", "arc_capsule", "rectangle"):
        # Length is the long axis by definition; keep the parameterisation
        # unambiguous instead of letting the optimiser swap the two axes.
        out[4] = min(out[4], out[3])
    return out


def _fit_instance(model: ShapeModel, prior: ShapePrior,
                  cfg: ShapeSegmentationConfig, x_nm: np.ndarray, y_nm: np.ndarray,
                  target: np.ndarray,
                  params0: np.ndarray) -> tuple[np.ndarray, float]:
    """Maximise the Dice overlap of the rasterised model with ``target``."""
    target_sum = int(np.count_nonzero(target))
    if target_sum == 0:
        return _clip_params(model, prior, params0), 0.0

    def negative_dice(raw: np.ndarray) -> float:
        params = _clip_params(model, prior, raw)
        mask = _model_mask(model, x_nm, y_nm, params)
        model_sum = int(np.count_nonzero(mask))
        if model_sum == 0:
            return 0.0
        overlap = int(np.count_nonzero(mask & target))
        return -2.0 * overlap / float(model_sum + target_sum)

    params0 = _clip_params(model, prior, params0)
    steps = [2.0 * cfg.detection_pixel_nm, 2.0 * cfg.detection_pixel_nm, 8.0]
    steps += [max(0.12 * float(v), cfg.detection_pixel_nm)
              if name.endswith("_nm") else 10.0
              for name, v in zip(model.size_names, params0[3:])]
    n = params0.size
    simplex = np.repeat(params0[None, :], n + 1, axis=0)
    for i in range(n):
        simplex[i + 1, i] += steps[i]
    result = minimize(
        negative_dice, params0, method="Nelder-Mead",
        options=dict(initial_simplex=simplex, maxfev=int(cfg.max_fit_evals),
                     xatol=0.25, fatol=1e-4))
    best = _clip_params(model, prior, result.x)
    return best, float(-result.fun)


def _principal_frame(x_nm: np.ndarray, y_nm: np.ndarray,
                     weights: np.ndarray | None = None):
    """Weighted centroid, principal axis (unit), and extents along/across it."""
    pts = np.column_stack([np.asarray(x_nm, float), np.asarray(y_nm, float)])
    if weights is None:
        weights = np.ones(len(pts), dtype=float)
    w = np.asarray(weights, dtype=float)
    if float(w.sum()) <= 0:
        w = np.ones(len(pts), dtype=float)
    total = float(w.sum())
    centre = (pts * w[:, None]).sum(0) / total
    centred = pts - centre
    cov = (centred * w[:, None]).T @ centred / total
    _, vecs = np.linalg.eigh(cov)
    major, minor = vecs[:, -1], vecs[:, 0]
    along, across = centred @ major, centred @ minor
    span_along = float(along.max() - along.min()) if len(pts) else 0.0
    span_across = float(across.max() - across.min()) if len(pts) else 0.0
    return centre, major, minor, span_along, span_across


def _weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int,
                     seed: int = 0) -> np.ndarray:
    """Small weighted k-means; used only to propose seed layouts."""
    rng = np.random.default_rng(seed)
    probs = weights / max(float(weights.sum()), 1e-12)
    start = rng.choice(len(points), size=min(k, len(points)), replace=False, p=probs)
    centres = points[start].astype(float).copy()
    assign = np.zeros(len(points), dtype=int)
    for _ in range(50):
        dist = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        assign = dist.argmin(1)
        moved = False
        for j in range(len(centres)):
            sel = assign == j
            if not sel.any():
                continue
            new = (points[sel] * weights[sel, None]).sum(0) / max(
                float(weights[sel].sum()), 1e-12)
            if not np.allclose(new, centres[j]):
                centres[j], moved = new, True
        if not moved:
            break
    return assign


def _seed_layouts(model: ShapeModel, prior: ShapePrior, x_nm, y_nm,
                  weights: np.ndarray, k: int) -> list[list[np.ndarray]]:
    """Propose parameter seeds for ``k`` instances covering one component.

    Two objects that touch end-to-end make the blob *longer*; two that touch
    side-by-side make it *wider*.  Both layouts are proposed (plus a k-means
    split for irregular clumps) and the fit keeps whichever explains best, which
    removes the guess.
    """
    centre, major, minor, span_along, span_across = _principal_frame(
        x_nm, y_nm, weights)
    nom_along, nom_across = model.extent(prior.nominal())
    angle = (0.0 if model.isotropic
             else float(np.degrees(np.arctan2(major[1], major[0]))))

    def make(size_along: float, size_across: float, offsets, direction):
        size = prior.clip(model.size_from_extent(max(size_along, 1.0),
                                                 max(size_across, 1.0)))
        return [np.array([*(centre + off * direction), angle, *size], dtype=float)
                for off in offsets]

    spread = np.arange(k) - (k - 1) / 2.0
    layouts: list[list[np.ndarray]] = []
    if k == 1:
        return [make(min(span_along, nom_along * 1.5),
                     min(span_across, nom_across * 1.5), [0.0], major)]
    # (a) split end-to-end, (b) split side-by-side
    layouts.append(make(span_along / k, span_across,
                        spread * (span_along / k), major))
    layouts.append(make(span_along, span_across / k,
                        spread * (span_across / k), minor))
    # (c) k-means on the component pixels, for irregular clumps
    pts = np.column_stack([np.asarray(x_nm, float), np.asarray(y_nm, float)])
    w = np.asarray(weights, dtype=float)
    if len(pts) >= k and float(w.sum()) > 0:
        assign = _weighted_kmeans(pts, w, k)
        seeds: list[np.ndarray] = []
        for j in range(k):
            sel = assign == j
            if not sel.any():
                seeds = []
                break
            sub_c, sub_major, _, sub_along, sub_across = _principal_frame(
                pts[sel, 0], pts[sel, 1], w[sel])
            sub_angle = (0.0 if model.isotropic else
                         float(np.degrees(np.arctan2(sub_major[1], sub_major[0]))))
            size = prior.clip(model.size_from_extent(max(sub_along, 1.0),
                                                     max(sub_across, 1.0)))
            seeds.append(np.array([sub_c[0], sub_c[1], sub_angle, *size],
                                  dtype=float))
        if seeds:
            layouts.append(seeds)
    return layouts


def _explain_component(model: ShapeModel, prior: ShapePrior,
                       cfg: ShapeSegmentationConfig, x_nm: np.ndarray,
                       y_nm: np.ndarray, target: np.ndarray, weights: np.ndarray,
                       k: int) -> tuple[list[np.ndarray], float, int]:
    """Best ``k``-instance explanation of one component.

    Returns ``(params, IoU, unexplained_pixels)`` where the last item is the
    symmetric difference between the union of the models and the component —
    the quantity the instance-count selection is scored on.  All rasterising
    happens on the caller's crop, so cost scales with the component, not the
    field.
    """
    ys, xs = np.nonzero(target)
    if ys.size == 0:
        return [], 0.0, 0
    px_x, px_y, px_w = x_nm[ys, xs], y_nm[ys, xs], weights[ys, xs]

    best_params: list[np.ndarray] = []
    best_iou = -1.0
    best_unexplained = int(np.count_nonzero(target))
    for seeds in _seed_layouts(model, prior, px_x, px_y, px_w, k):
        params = [_clip_params(model, prior, s) for s in seeds]
        for _round in range(max(int(cfg.refine_rounds), 1)):
            if k == 1:
                territory = [target]
            else:
                dists = np.stack([
                    model.sdf(x_nm, y_nm, p[0], p[1], p[2], p[3:]) for p in params])
                owner = dists.argmin(0)
                territory = [target & (owner == j) for j in range(k)]
            updated = []
            for j, p in enumerate(params):
                if int(np.count_nonzero(territory[j])) < 4:
                    updated.append(p)
                    continue
                fitted, _ = _fit_instance(model, prior, cfg, x_nm, y_nm,
                                          territory[j], p)
                updated.append(fitted)
            settled = all(np.allclose(a, b, atol=0.5)
                          for a, b in zip(params, updated))
            params = updated
            if settled:
                break
        union = np.zeros_like(target)
        for p in params:
            union |= _model_mask(model, x_nm, y_nm, p)
        intersection = int(np.count_nonzero(union & target))
        denominator = int(np.count_nonzero(union | target))
        iou = intersection / denominator if denominator else 0.0
        if iou > best_iou:
            best_params = params
            best_iou = float(iou)
            best_unexplained = denominator - intersection
    return best_params, max(best_iou, 0.0), int(best_unexplained)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _foreground(detection: np.ndarray, cfg: ShapeSegmentationConfig,
                prior: ShapePrior) -> tuple[np.ndarray, float]:
    _, nominal_across = prior.model.extent(prior.nominal())
    smooth_nm = (cfg.smoothing_nm if cfg.smoothing_nm is not None
                 else cfg.smoothing_width_ratio * float(nominal_across))
    smoothed = detection
    if smooth_nm and smooth_nm > 0:
        smoothed = ndi.gaussian_filter(
            detection, float(smooth_nm) / cfg.detection_pixel_nm)
    if cfg.threshold is not None:
        level = float(cfg.threshold)
    elif cfg.threshold_quantile is not None:
        level = float(np.quantile(smoothed, float(cfg.threshold_quantile)))
    else:
        level = otsu_threshold(smoothed)
    mask = smoothed > level
    if cfg.closing_nm > 0:
        radius = max(int(round(cfg.closing_nm / cfg.detection_pixel_nm)), 1)
        mask = ndi.binary_closing(mask, structure=_disk_struct(radius))
    if cfg.fill_holes:
        mask = ndi.binary_fill_holes(mask)
    return mask, level


def _is_clipped(model: ShapeModel, params: np.ndarray,
                field_obj: ScalarField, *, n_points: int = 256) -> bool:
    """True when the fitted outline leaves the visible frame.

    Decided geometrically rather than by thresholding ``visible_fraction``: that
    ratio is rasterised, so its discretisation error (order one pixel along the
    boundary) is easily a percent, which no fixed threshold separates cleanly
    from a genuine sliver poking over the edge.
    """
    x0, y0, x1, y1 = field_obj.visible_bounds()
    outline = model.outline(params[0], params[1], params[2], params[3:], n_points)
    ox, oy = outline[:, 0], outline[:, 1]
    return bool(np.any((ox < x0) | (ox > x1) | (oy < y0) | (oy > y1)))


def _visible_fraction(model: ShapeModel, params: np.ndarray,
                      field_obj: ScalarField, pixel_nm: float) -> float:
    """Fraction of the fitted model's area that lies inside the field."""
    x0, y0, x1, y1 = field_obj.visible_bounds()
    along, across = model.extent(params[3:])
    reach = 0.5 * float(max(along, across)) + pixel_nm
    gx = np.arange(params[0] - reach, params[0] + reach + pixel_nm, pixel_nm)
    gy = np.arange(params[1] - reach, params[1] + reach + pixel_nm, pixel_nm)
    if gx.size == 0 or gy.size == 0:
        return 1.0
    mx, my = np.meshgrid(gx, gy)
    inside = _model_mask(model, mx, my, params)
    total = int(np.count_nonzero(inside))
    if total == 0:
        return 1.0
    in_field = inside & (mx >= x0) & (mx < x1) & (my >= y0) & (my < y1)
    return float(np.count_nonzero(in_field) / total)


def segment_shapes(field_obj: ScalarField, *, prior: ShapePrior | None = None,
                   cfg: ShapeSegmentationConfig | None = None
                   ) -> ShapeSegmentationResult:
    """Detect and segment objects of a known geometry in a scalar field.

    Instance masks are the fitted shape models and *may overlap*; ``labels``
    additionally gives an exclusive per-pixel assignment.
    """
    prior = prior or ShapePrior()
    cfg = cfg or ShapeSegmentationConfig()
    prior.validate()
    if cfg.detection_pixel_nm <= 0:
        raise ValueError("detection_pixel_nm must be positive")
    if cfg.max_instances_per_component < 1:
        raise ValueError("max_instances_per_component must be at least 1")

    model = prior.model
    source = np.nan_to_num(np.asarray(field_obj.values, dtype=float),
                           nan=0.0, posinf=0.0, neginf=0.0)
    shape = source.shape
    base_stats = {
        "n_instances": 0, "source_pixel_nm": float(field_obj.pixel_nm),
        "detection_pixel_nm": float(cfg.detection_pixel_nm),
        "image_shape": [int(v) for v in shape], "model": model.key,
    }
    if source.size == 0 or not np.any(source > 0):
        return ShapeSegmentationResult(
            masks=np.zeros((0, *shape), dtype=bool),
            labels=np.zeros(shape, dtype=np.int32),
            union_mask=np.zeros(shape, dtype=bool),
            overlap_count=np.zeros(shape, dtype=np.uint16),
            foreground=np.zeros(shape, dtype=bool),
            detection_pixel_nm=float(cfg.detection_pixel_nm),
            stats={**base_stats, "reason": "field has no positive signal"})

    detection = _resample(source, float(field_obj.pixel_nm), cfg.detection_pixel_nm)
    if detection.size > cfg.max_detection_pixels:
        raise ValueError(
            f"detection grid needs {detection.size:,} pixels, above the "
            f"{cfg.max_detection_pixels:,} limit; raise detection_pixel_nm "
            f"or max_detection_pixels")
    det_field = ScalarField(detection, cfg.detection_pixel_nm, field_obj.origin_nm,
                            field_obj.visible_bounds_nm)
    x_nm, y_nm = det_field.coords()

    foreground, level = _foreground(detection, cfg, prior)
    components, n_components = ndi.label(foreground)
    nominal_area = prior.nominal_area_nm2()
    pixel_area = cfg.detection_pixel_nm ** 2
    min_pixels = max(int(cfg.min_component_area_frac * nominal_area / pixel_area), 1)

    nominal_pixels = max(nominal_area / pixel_area, 1.0)
    objects = ndi.find_objects(components)
    margin = max(int(ceil(0.5 * max(model.extent(prior.nominal()))
                          / cfg.detection_pixel_nm)), 2)

    fitted: list[tuple[np.ndarray, int]] = []
    component_report: list[dict] = []
    for comp_id in range(1, int(n_components) + 1):
        window = objects[comp_id - 1]
        if window is None:
            continue
        if int(np.count_nonzero(components[window] == comp_id)) < min_pixels:
            continue
        # Fit on a crop around the component: cost then scales with the object,
        # not with the field.  The crop is clipped to the field on purpose, so a
        # model that runs off the edge is scored on its *visible* part only.
        rows = slice(max(window[0].start - margin, 0),
                     min(window[0].stop + margin, detection.shape[0]))
        cols = slice(max(window[1].start - margin, 0),
                     min(window[1].stop + margin, detection.shape[1]))
        target = components[rows, cols] == comp_id
        crop_x, crop_y = x_nm[rows, cols], y_nm[rows, cols]
        crop_w = detection[rows, cols]

        n_pixels = int(np.count_nonzero(target))
        area_nm2 = n_pixels * pixel_area
        estimate = int(round(area_nm2 / max(nominal_area, 1e-9)))
        k_max = int(np.clip(estimate + 1, 1, cfg.max_instances_per_component))
        best_k, best_params, best_cost = 0, [], np.inf
        report: dict[int, dict] = {}
        for k in range(1, k_max + 1):
            params, iou, unexplained = _explain_component(
                model, prior, cfg, crop_x, crop_y, target, crop_w, k)
            cost = unexplained / nominal_pixels + cfg.instance_cost * k
            report[k] = {"iou": round(float(iou), 4), "cost": round(float(cost), 4)}
            if params and cost < best_cost:
                best_k, best_params, best_cost = k, params, cost
        component_report.append({
            "component_id": comp_id, "area_nm2": float(area_nm2),
            "area_estimate_k": estimate, "chosen_k": best_k, "by_k": report,
        })
        for params in best_params:
            fitted.append((params, comp_id))
        if len(fitted) >= cfg.max_instances:
            break

    # Per-instance quality on its own territory, then drop poor explanations.
    instances: list[ShapeInstance] = []
    kept_params: list[np.ndarray] = []
    if fitted:
        dists = np.stack([
            model.sdf(x_nm, y_nm, p[0], p[1], p[2], p[3:]) for p, _ in fitted])
        owner = dists.argmin(0)
        for index, (params, comp_id) in enumerate(fitted):
            mask = _model_mask(model, x_nm, y_nm, params)
            territory = foreground & (owner == index)
            denominator = int(np.count_nonzero(mask | territory))
            iou = (int(np.count_nonzero(mask & territory)) / denominator
                   if denominator else 0.0)
            if iou < cfg.min_instance_iou:
                continue
            visible = _visible_fraction(model, params, det_field,
                                        cfg.detection_pixel_nm)
            clipped = _is_clipped(model, params, det_field)
            angle = (0.0 if model.isotropic else
                     _angle_axial(params[2]) if model.axial
                     else float(params[2]) % 360.0)
            kept_params.append(params)
            instances.append(ShapeInstance(
                model_key=model.key,
                center_nm=(float(params[0]), float(params[1])),
                angle_deg=float(angle),
                size_nm=tuple(float(v) for v in params[3:]),
                iou=float(iou), visible_fraction=float(visible),
                clipped=bool(clipped),
                area_nm2=float(model.area(params[3:])),
                component_id=int(comp_id)))

    # Rasterise each instance only inside its own bounding box: a full-field
    # signed-distance stack would cost K x the image in float64 for no gain.
    masks = np.zeros((len(kept_params), *shape), dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    nearest = np.full(shape, np.inf, dtype=float)
    src_px = float(field_obj.pixel_nm)
    origin_x, origin_y = field_obj.origin_nm
    for index, params in enumerate(kept_params):
        along, across = model.extent(params[3:])
        reach = 0.5 * float(max(along, across)) + src_px
        c0 = max(int(np.floor((params[0] - reach - origin_x) / src_px)), 0)
        c1 = min(int(np.ceil((params[0] + reach - origin_x) / src_px)) + 1, shape[1])
        r0 = max(int(np.floor((params[1] - reach - origin_y) / src_px)), 0)
        r1 = min(int(np.ceil((params[1] + reach - origin_y) / src_px)) + 1, shape[0])
        if c1 <= c0 or r1 <= r0:
            continue
        rows, cols = np.mgrid[r0:r1, c0:c1]
        sub_x = origin_x + (cols + 0.5) * src_px
        sub_y = origin_y + (rows + 0.5) * src_px
        dist = model.sdf(sub_x, sub_y, params[0], params[1], params[2], params[3:])
        inside = dist < 0.0
        masks[index, r0:r1, c0:c1] = inside
        closer = inside & (dist < nearest[r0:r1, c0:c1])
        nearest[r0:r1, c0:c1][closer] = dist[closer]
        labels[r0:r1, c0:c1][closer] = index + 1
    overlap = masks.sum(axis=0, dtype=np.uint16)

    stats = {
        **base_stats,
        "n_instances": len(instances),
        "n_clipped": int(sum(item.clipped for item in instances)),
        "n_components": int(n_components),
        "n_components_used": len(component_report),
        "threshold": float(level),
        "nominal_area_nm2": float(nominal_area),
        "min_component_area_nm2": float(min_pixels * pixel_area),
        "components": component_report,
    }
    if not instances:
        # Finding nothing is a result, but a bare zero is not diagnosable —
        # say which stage dropped everything so the fix is obvious.
        if n_components == 0:
            stats["reason"] = (
                f"nothing passed the foreground threshold ({level:.3g}); the "
                f"field may be empty, or the smoothing too small to connect it")
        elif not component_report:
            stats["reason"] = (
                f"{n_components} foreground blob(s) were found but every one is "
                f"smaller than {min_pixels * pixel_area / 1e6:.2f} um^2 "
                f"({cfg.min_component_area_frac:g} x the nominal object area). "
                f"Raise the detection pixel size or the smoothing so the "
                f"objects form solid blobs, or lower the expected size")
        else:
            stats["reason"] = (
                f"{len(component_report)} blob(s) were fitted but no instance "
                f"reached the minimum overlap of {cfg.min_instance_iou:g}")
    
    return ShapeSegmentationResult(
        instances=instances, masks=masks, labels=labels, union_mask=overlap > 0,
        overlap_count=overlap, foreground=foreground, detection_field=detection,
        detection_origin_nm=(float(field_obj.origin_nm[0]),
                             float(field_obj.origin_nm[1])),
        detection_pixel_nm=float(cfg.detection_pixel_nm), stats=stats)


def segment_shapes_in_image(image: np.ndarray, pixel_size_nm: float, *,
                            prior: ShapePrior | None = None,
                            cfg: ShapeSegmentationConfig | None = None,
                            origin_nm: tuple[float, float] = (0.0, 0.0)
                            ) -> ShapeSegmentationResult:
    """Segment known-geometry objects in a calibrated 2-D image."""
    return segment_shapes(field_from_image(image, pixel_size_nm, origin_nm),
                          prior=prior, cfg=cfg)


def segment_shapes_in_points(x_nm: np.ndarray, y_nm: np.ndarray, *,
                             prior: ShapePrior | None = None,
                             cfg: ShapeSegmentationConfig | None = None,
                             render_pixel_nm: float | None = None,
                             bounds_nm: tuple[float, float, float, float] | None = None
                             ) -> ShapeSegmentationResult:
    """Segment known-geometry objects in a 2-D point cloud (e.g. localizations).

    ``result.point_labels`` assigns every input point to an instance
    (0 = outside every fitted object), which is usually what a downstream
    per-object analysis needs.

    Pass ``bounds_nm`` (the acquisition field of view) when it is known.  By
    default the raster is spanned by the localizations themselves, and an object
    can then never be seen to leave the frame — so ``clipped`` stays False even
    for a cell the acquisition truly cut in half.
    """
    prior = prior or ShapePrior()
    cfg = cfg or ShapeSegmentationConfig()
    prior.validate()
    px = float(render_pixel_nm) if render_pixel_nm else float(cfg.detection_pixel_nm)
    _, nominal_across = prior.model.extent(prior.nominal())
    scene = field_from_points(x_nm, y_nm, px, margin_nm=0.5 * nominal_across,
                              bounds_nm=bounds_nm)
    result = segment_shapes(scene, prior=prior, cfg=cfg)

    x = np.asarray(x_nm, dtype=float).ravel()
    y = np.asarray(y_nm, dtype=float).ravel()
    labels = np.zeros(x.size, dtype=np.int32)
    if result.instances:
        model = prior.model
        dists = np.stack([
            model.sdf(x, y, item.center_nm[0], item.center_nm[1], item.angle_deg,
                      item.size_nm)
            for item in result.instances])
        labels = np.where(dists.min(0) < 0.0, dists.argmin(0) + 1, 0).astype(np.int32)
    result.point_labels = labels
    result.stats["n_points"] = int(x.size)
    result.stats["n_points_assigned"] = int(np.count_nonzero(labels))
    return result


def instance_mask(instance: ShapeInstance, shape: tuple[int, int],
                  pixel_size_nm: float,
                  origin_nm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Rasterise one instance on an image grid (pixel-centre convention)."""
    rows, cols = np.indices(shape, dtype=float)
    x_nm = origin_nm[0] + (cols + 0.5) * float(pixel_size_nm)
    y_nm = origin_nm[1] + (rows + 0.5) * float(pixel_size_nm)
    return _model_mask(instance.model, x_nm, y_nm, instance.params)


def instance_outline(instance: ShapeInstance, *, n_points: int = 64) -> np.ndarray:
    """Closed ``(N, 2)`` outline in physical (nm) image coordinates."""
    model = instance.model
    return model.outline(instance.center_nm[0], instance.center_nm[1],
                         instance.angle_deg, instance.size_nm, int(n_points))
