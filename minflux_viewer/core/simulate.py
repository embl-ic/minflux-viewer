"""
minflux_viewer.core.simulate
============================
Synthetic **MINFLUX-like sample datasets** (File › Open Sample Data).

A structure template produces molecule positions (in nm); each molecule is then
expanded into a short **trace** of localizations (Poisson blink count, Gaussian
localization jitter), and every localization is given the canonical MINFLUX
attributes drawn from their characteristic distributions (Gaussian / log-normal /
Poisson, range-clipped). The output feeds :func:`core.dataset.build_localization_dataset`.

Pure NumPy, Qt-free, deterministic under a ``seed`` — unit-tested.

Structure templates (``STRUCTURES``): homogeneous random, random clusters, NPC
(8-fold two-ring), microtubule (hollow filaments), sphere shell, cylinder, cube
(surface), pyramid (surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """One structure parameter (drives the dialog's spin boxes)."""
    name: str
    label: str
    default: float
    lo: float
    hi: float
    integer: bool = False
    suffix: str = " nm"
    desc: str = ""            # hover tooltip explaining the parameter


# --------------------------------------------------------------------------- #
# structure generators — each returns (n, 3) molecule positions in nm
# --------------------------------------------------------------------------- #
def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to *direction*."""
    d = direction / (np.linalg.norm(direction) or 1.0)
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a)
    u /= (np.linalg.norm(u) or 1.0)
    w = np.cross(d, u)
    return u, w


def place_nonoverlapping(n: int, min_dist: float, rng, *,
                         disk_radius: float | None = None,
                         box_half: tuple[float, float] | None = None,
                         tries: int = 48) -> np.ndarray:
    """Place up to *n* 2-D instance centres with pairwise distance ≥ ``min_dist``
    (real structures — NPC pores, clusters, microtubules — cannot spatially
    overlap). Rejection sampling with a grid hash for O(1) neighbour checks; if the
    region is too dense to fit *n*, **fewer** are returned (never overlapping).

    Sample in a centred disk (``disk_radius``) or box (``box_half``). This applies
    per **channel/scaffold** — different channels of the *same* overlay share one
    scaffold and are allowed to overlap (different labelling of one structure)."""
    n = max(int(n), 0)
    if n == 0:
        return np.empty((0, 2))

    def _sample() -> np.ndarray:
        if disk_radius is not None:
            rad = float(disk_radius) * np.sqrt(rng.uniform())
            ang = rng.uniform(0.0, 2 * np.pi)
            return np.array([rad * np.cos(ang), rad * np.sin(ang)])
        hx, hy = box_half
        return rng.uniform([-hx, -hy], [hx, hy])

    if min_dist <= 0:                                    # no constraint → uniform
        return np.array([_sample() for _ in range(n)])

    # cell = min_dist so any pair closer than min_dist differs by ≤1 grid cell per
    # axis → the 3×3 neighbourhood below catches every potential conflict.
    cell = float(min_dist)
    grid: dict[tuple[int, int], list[int]] = {}
    placed: list[np.ndarray] = []

    def _ok(pt: np.ndarray) -> bool:
        gi, gj = int(np.floor(pt[0] / cell)), int(np.floor(pt[1] / cell))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for k in grid.get((gi + di, gj + dj), ()):
                    if np.hypot(pt[0] - placed[k][0], pt[1] - placed[k][1]) < min_dist:
                        return False
        return True

    for _ in range(n):
        for _t in range(tries):
            pt = _sample()
            if _ok(pt):
                gi, gj = int(np.floor(pt[0] / cell)), int(np.floor(pt[1] / cell))
                grid.setdefault((gi, gj), []).append(len(placed))
                placed.append(pt)
                break
    return np.asarray(placed) if placed else np.empty((0, 2))


def gen_homogeneous(n, p, rng):
    s = float(p["size_nm"])
    return rng.uniform(-s / 2, s / 2, size=(n, 3))


def gen_clusters(n, p, rng):
    s = float(p["size_nm"])
    nc = max(int(p["n_clusters"]), 1)
    sig = float(p["cluster_sigma_nm"])
    # cluster centres non-overlapping (≥ ~4σ apart) so blobs stay distinct
    xy = place_nonoverlapping(nc, 4.0 * sig, rng, box_half=(s / 2, s / 2))
    nc = xy.shape[0]
    centres = np.column_stack([xy, rng.uniform(-s / 2, s / 2, nc)])
    idx = rng.integers(0, nc, size=n)
    return centres[idx] + rng.normal(0.0, sig, size=(n, 3))


def gen_sphere(n, p, rng):
    r = float(p["radius_nm"])
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
    return v * (r + rng.normal(0.0, r * 0.02, n)[:, None])


def gen_cylinder(n, p, rng):
    r = float(p["radius_nm"])
    h = float(p["height_nm"])
    theta = rng.uniform(0.0, 2 * np.pi, n)
    z = rng.uniform(-h / 2, h / 2, n)
    rr = r + rng.normal(0.0, r * 0.03, n)
    return np.column_stack([rr * np.cos(theta), rr * np.sin(theta), z])


def gen_cube(n, p, rng):
    a = float(p["side_nm"]) / 2.0
    face = rng.integers(0, 6, n)
    u = rng.uniform(-a, a, n)
    v = rng.uniform(-a, a, n)
    pts = np.zeros((n, 3))
    for f in range(6):
        m = face == f
        axis = f // 2
        sign = -a if f % 2 == 0 else a
        cols = [c for c in range(3) if c != axis]
        pts[m, axis] = sign
        pts[m, cols[0]] = u[m]
        pts[m, cols[1]] = v[m]
    return pts


def gen_pyramid(n, p, rng):
    base = float(p["base_nm"]) / 2.0
    height = float(p["height_nm"])
    apex = np.array([0.0, 0.0, height])
    c = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) * base
    faces = [(c[0], c[1], c[2]), (c[0], c[2], c[3])]                  # square base (2 tris)
    faces += [(c[i], c[(i + 1) % 4], apex) for i in range(4)]         # 4 sides
    faces = np.asarray(faces)                                         # (6, 3, 3)
    fi = rng.integers(0, faces.shape[0], n)
    r1 = np.sqrt(rng.uniform(0, 1, n))
    r2 = rng.uniform(0, 1, n)
    bary = np.column_stack([1 - r1, r1 * (1 - r2), r1 * r2])          # random-in-triangle
    tri = faces[fi]
    return (bary[:, :, None] * tri).sum(axis=1)


def _frames_from_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row orthonormal tangent vectors (u, v) perpendicular to each unit normal."""
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
    a = np.tile(np.array([1.0, 0.0, 0.0]), (n.shape[0], 1))
    a[np.abs(n[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-9)
    v = np.cross(n, u)
    return u, v


def gen_npc(n, p, rng):
    """NPCs on a (optionally curved) membrane. ``field_curvature`` 0→flat, 1→a
    hemisphere over the field of view, so pores near the edge tilt toward the
    equator (axial tilt = local surface angle); ``local_tilt_deg`` adds a random
    per-NPC tilt to mimic membrane roughness. Each pore is a 8-fold two-ring."""
    n_pores = max(int(p["n_pores"]), 1)
    diam = float(p["diameter_nm"])
    sep = float(p["ring_sep_nm"])
    fov = float(p["size_nm"])
    curvature = float(p.get("field_curvature", 0.0))
    local_tilt = float(p.get("local_tilt_deg", 0.0))
    sym = 8
    R = (fov / 2.0) or 1.0

    # pore centres in the FOV disk, **non-overlapping** (rings ≥ 1 diameter apart) —
    # NPCs are membrane-embedded and cannot overlap. (radial distance drives the tilt)
    pores_xy = place_nonoverlapping(n_pores, diam * 1.05, rng, disk_radius=R)
    px, py = pores_xy[:, 0], pores_xy[:, 1]
    n_pores = px.shape[0]
    rr = np.hypot(px, py)

    # dome: pores lie on a spherical cap; polar angle = axial tilt, apex at z=0
    theta_max = curvature * (np.pi / 2.0)
    if theta_max > 1e-6:
        Rs = R / np.sin(theta_max)
        theta = np.arcsin(np.clip(rr / Rs, 0.0, 1.0))
        pz = -Rs * (1.0 - np.cos(theta))
    else:
        theta = np.zeros(n_pores)
        pz = np.zeros(n_pores)
    phi = np.arctan2(py, px)
    normals = np.column_stack([np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi), np.cos(theta)])

    # per-NPC random tilt about a random tangent azimuth
    if local_tilt > 1e-6:
        u0, v0 = _frames_from_normals(normals)
        d = np.radians(rng.uniform(0.0, local_tilt, n_pores))[:, None]
        az = rng.uniform(0.0, 2 * np.pi, n_pores)[:, None]
        normals = (np.cos(d) * normals
                   + np.sin(d) * (np.cos(az) * u0 + np.sin(az) * v0))

    u, v = _frames_from_normals(normals)
    centres = np.column_stack([px, py, pz])                             # (n_pores, 3)

    ang = np.arange(sym) * 2 * np.pi / sym
    sub_local = np.vstack([                                             # (16, 3) in (u,v,n)
        np.column_stack([(diam / 2) * np.cos(ang), (diam / 2) * np.sin(ang), np.full(sym, zc)])
        for zc in (-sep / 2, sep / 2)])
    # world subunit positions (n_pores, 16, 3)
    subs = (centres[:, None, :]
            + sub_local[None, :, 0, None] * u[:, None, :]
            + sub_local[None, :, 1, None] * v[:, None, :]
            + sub_local[None, :, 2, None] * normals[:, None, :])

    pore_i = rng.integers(0, n_pores, n)
    sub_i = rng.integers(0, subs.shape[1], n)
    return subs[pore_i, sub_i] + rng.normal(0.0, 3.0, size=(n, 3))       # subunit jitter                      # subunit jitter


def gen_microtubule(n, p, rng):
    n_fil = max(int(p["n_filaments"]), 1)
    length = float(p["length_nm"])
    diam = float(p["diameter_nm"])
    fov = float(p["size_nm"])
    # filament seeds non-overlapping in XY so filaments don't originate on top of
    # each other (random 3-D directions then rarely intersect in 3-D).
    seeds = place_nonoverlapping(n_fil, max(diam * 4.0, 50.0), rng, box_half=(fov / 2, fov / 2))
    n_fil = seeds.shape[0]
    per = max(n // n_fil, 1)
    out = []
    for i in range(n_fil):
        start = np.array([seeds[i, 0], seeds[i, 1], rng.uniform(-fov / 2, fov / 2)])
        d = rng.normal(size=3)
        d /= (np.linalg.norm(d) or 1.0)
        u, w = _basis(d)
        t = rng.uniform(0.0, length, per)
        theta = rng.uniform(0.0, 2 * np.pi, per)
        rr = diam / 2 + rng.normal(0.0, diam * 0.05, per)
        radial = (rr * np.cos(theta))[:, None] * u + (rr * np.sin(theta))[:, None] * w
        out.append(start + t[:, None] * d + radial)
    pts = np.vstack(out)
    return pts[:n] if pts.shape[0] >= n else pts


#: template key → (generator, human label, structure params)
STRUCTURES: dict[str, tuple] = {
    "homogeneous": (gen_homogeneous, "Homogeneous random",
                    [ParamSpec("size_nm", "Field size", 5000, 100, 100000)]),
    "clusters": (gen_clusters, "Random clusters", [
        ParamSpec("size_nm", "Field size", 5000, 100, 100000),
        ParamSpec("n_clusters", "Number of clusters", 20, 1, 5000, True, ""),
        ParamSpec("cluster_sigma_nm", "Cluster spread", 40, 1, 5000)]),
    "npc": (gen_npc, "NPC (8-fold two-ring)", [
        ParamSpec("size_nm", "Field size", 5000, 100, 100000),
        ParamSpec("n_pores", "Number of pores", 30, 1, 5000, True, ""),
        ParamSpec("diameter_nm", "Ring diameter", 107, 10, 500),
        ParamSpec("ring_sep_nm", "Ring separation (Z)", 50, 0, 500),
        ParamSpec("field_curvature", "Field curvature", 0.0, 0.0, 1.0, False, "",
                  desc="Membrane curvature across the field of view. 0 = a flat "
                       "membrane (all NPCs axis-vertical); 1 = a full hemisphere over "
                       "the field, so pores near the edge tilt up to 90° (axial tilt = "
                       "local surface angle) — mimics NPCs on a curved nuclear envelope."),
        ParamSpec("local_tilt_deg", "Local tilt variation", 0.0, 0.0, 45.0, False, "°",
                  desc="Extra random axial tilt applied per NPC (0..this many degrees) "
                       "on top of the membrane curvature, mimicking local membrane "
                       "roughness. Useful for testing how particle-average algorithms "
                       "handle a spread of axial tilts.")]),
    "microtubule": (gen_microtubule, "Microtubule (hollow filaments)", [
        ParamSpec("size_nm", "Field size", 5000, 100, 100000),
        ParamSpec("n_filaments", "Number of filaments", 5, 1, 500, True, ""),
        ParamSpec("length_nm", "Filament length", 4000, 100, 50000),
        ParamSpec("diameter_nm", "Diameter", 25, 5, 200)]),
    "sphere": (gen_sphere, "Sphere shell",
               [ParamSpec("radius_nm", "Radius", 250, 10, 10000)]),
    "cylinder": (gen_cylinder, "Cylinder surface", [
        ParamSpec("radius_nm", "Radius", 100, 10, 10000),
        ParamSpec("height_nm", "Height", 1000, 10, 50000)]),
    "cube": (gen_cube, "Cube surface",
             [ParamSpec("side_nm", "Side length", 500, 10, 20000)]),
    "pyramid": (gen_pyramid, "Pyramid surface", [
        ParamSpec("base_nm", "Base side", 500, 10, 20000),
        ParamSpec("height_nm", "Height", 500, 10, 20000)]),
}


_SCAFFOLD_PARAMS = [
    ParamSpec("size_nm", "Field size", 5000, 100, 100000),
    ParamSpec("n_pores", "Number of pores", 30, 1, 5000, True, ""),
    ParamSpec("field_curvature", "Field curvature", 0.0, 0.0, 1.0, False, "",
              desc="Membrane curvature across the field (0 = flat, 1 = hemisphere) — "
                   "pores near the edge tilt, giving a spread of axial tilts."),
    ParamSpec("local_tilt_deg", "Local tilt variation", 0.0, 0.0, 45.0, False, "°",
              desc="Extra random per-NPC axial tilt on top of the field curvature."),
    ParamSpec("ring_sep_nm", "Ring separation (Z)", 50, 0, 500),
]

#: Multi-channel / spectral simulations (key → (generator-kind, label, ParamSpec list)).
#: ``kind`` = "overlay" (co-registered channels → a multi-dataset overlay) or
#: "dcr" (one dataset, two reporters distinguished only by a bimodal ``dcr``).
MULTI_SIMS: dict[str, tuple] = {
    "npc_overlay_3ch": ("overlay", "NPC 3-channel overlay (shared scaffold)",
                        _SCAFFOLD_PARAMS + [
                            ParamSpec("ch1_diameter_nm", "Ch1 diameter (scaffold)", 107, 10, 500),
                            ParamSpec("ch2_diameter_nm", "Ch2 diameter (inner ring)", 70, 10, 500),
                            ParamSpec("ch3_diameter_nm", "Ch3 diameter (central cap)", 40, 5, 500)]),
    "npc_dcr_2ch": ("dcr", "NPC 2-channel by DCR (spectral, 1 dataset)",
                    _SCAFFOLD_PARAMS + [
                        ParamSpec("outer_diameter_nm", "Reporter A diameter (outer)", 107, 10, 500),
                        ParamSpec("inner_diameter_nm", "Reporter B diameter (inner)", 70, 10, 500),
                        ParamSpec("dcr_low", "Reporter A mean DCR", 0.3, 0.0, 1.0, False, "",
                                  desc="Mean DCR of reporter A (outer ring) — the two "
                                       "reporters' DCR distributions should be separable."),
                        ParamSpec("dcr_high", "Reporter B mean DCR", 0.7, 0.0, 1.0, False, "",
                                  desc="Mean DCR of reporter B (inner ring).")]),
}


def param_specs(key: str) -> list:
    """The ``ParamSpec`` list for a structure or multi-sim key."""
    if key in STRUCTURES:
        return STRUCTURES[key][2]
    if key in MULTI_SIMS:
        return MULTI_SIMS[key][2]
    raise KeyError(key)


def sim_kind(key: str) -> str:
    """"single" (a plain structure), "overlay" or "dcr" (a multi-channel sim)."""
    return MULTI_SIMS[key][0] if key in MULTI_SIMS else "single"


def structure_labels() -> list[tuple[str, str]]:
    labels = [(key, val[1]) for key, val in STRUCTURES.items()]
    labels += [(key, val[1]) for key, val in MULTI_SIMS.items()]
    return labels


def default_params(structure: str) -> dict[str, float]:
    return {ps.name: ps.default for ps in param_specs(structure)}


# --------------------------------------------------------------------------- #
# attributes — canonical MINFLUX per-loc values from characteristic distributions
# --------------------------------------------------------------------------- #
def simulate_attributes(n: int, rng) -> dict[str, np.ndarray]:
    """Per-localization MINFLUX attributes (range-clipped):
    ``efo`` log-normal (Hz), ``cfr``/``dcr`` Gaussian in [0,1], ``eco``/``ecc``
    Poisson photon counts, ``fbg`` Gaussian background."""
    return {
        "efo": rng.lognormal(np.log(8.0e4), 0.4, n),               # ~80 kHz, positive
        "cfr": np.clip(rng.normal(0.45, 0.12, n), 0.0, 1.0),       # center freq ratio
        "dcr": np.clip(rng.normal(0.5, 0.15, n), 0.0, 1.0),        # detector channel ratio
        "eco": rng.poisson(60, n).astype(float),                   # effective counts
        "ecc": rng.poisson(600, n).astype(float),                  # collected counts
        "fbg": np.clip(rng.normal(15.0, 5.0, n), 0.0, None),       # background
    }


def simulate_localizations(
    structure: str,
    *,
    n_points: int = 2000,
    locs_per_trace: float = 4.0,
    precision_nm: float = 5.0,
    params: dict | None = None,
    dim: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate a MINFLUX-like point cloud on *structure*.

    Returns ``(coords_nm (N,3), tid (N,), attrs)``. Each of ``n_points`` molecules
    is placed on the structure and expanded into a Poisson(``locs_per_trace``, ≥1)
    trace of localizations jittered by ``precision_nm``; ``dim=2`` flattens Z.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure '{structure}'")
    rng = np.random.default_rng(seed)
    gen = STRUCTURES[structure][0]
    p = {**default_params(structure), **(params or {})}
    n_points = max(int(n_points), 1)

    centres = np.asarray(gen(n_points, p, rng), dtype=float).reshape(-1, 3)
    coords, tid = _expand_to_localizations(centres, locs_per_trace, precision_nm, rng)
    if int(dim) == 2:
        coords[:, 2] = 0.0
    return coords, tid, simulate_attributes(coords.shape[0], rng)


def _expand_to_localizations(centres: np.ndarray, locs_per_trace: float,
                             precision_nm: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Expand molecule *centres* ``(M,3)`` into a Poisson(``locs_per_trace``, ≥1)
    trace each, jittered by ``precision_nm``. Returns ``(coords (N,3), tid (N,))``."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 3)
    m = centres.shape[0]
    if m == 0:
        return np.empty((0, 3)), np.empty(0, np.int64)
    counts = np.maximum(rng.poisson(max(float(locs_per_trace), 0.1), m), 1)
    tid = np.repeat(np.arange(1, m + 1, dtype=np.int64), counts)
    n = int(counts.sum())
    coords = np.repeat(centres, counts, axis=0) + rng.normal(0.0, float(precision_nm), size=(n, 3))
    return coords, tid


# --------------------------------------------------------------------------- #
# Multi-channel NPC simulations — several labelling channels sharing ONE pore
# scaffold (co-registered), for testing multi-channel particle averaging + the
# DCR (spectral) channel-separation workflow.
# --------------------------------------------------------------------------- #
def _npc_scaffold(n_pores: int, fov: float, curvature: float, local_tilt: float, rng,
                  *, min_separation: float = 0.0):
    """Shared NPC scaffold: pore centres ``(n,3)`` + per-pore ring-plane frame
    ``(u, v, normal)`` (dome curvature + per-pore tilt applied). Every labelling
    channel is generated against this same scaffold so they are co-registered.

    Pores are placed **non-overlapping** (centres ≥ ``min_separation`` apart — the
    largest labelling diameter — so distinct pores never overlap, though the
    co-registered channels of one pore do)."""
    n_pores = max(int(n_pores), 1)
    R = (fov / 2.0) or 1.0
    pores_xy = place_nonoverlapping(n_pores, float(min_separation) * 1.05, rng, disk_radius=R)
    px, py = pores_xy[:, 0], pores_xy[:, 1]
    rr = np.hypot(px, py)
    n_pores = px.shape[0]
    theta_max = curvature * (np.pi / 2.0)
    if theta_max > 1e-6:
        Rs = R / np.sin(theta_max)
        theta = np.arcsin(np.clip(rr / Rs, 0.0, 1.0))
        pz = -Rs * (1.0 - np.cos(theta))
    else:
        theta = np.zeros(n_pores)
        pz = np.zeros(n_pores)
    phi = np.arctan2(py, px)
    normals = np.column_stack([np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi), np.cos(theta)])
    if local_tilt > 1e-6:
        u0, v0 = _frames_from_normals(normals)
        d = np.radians(rng.uniform(0.0, local_tilt, n_pores))[:, None]
        az = rng.uniform(0.0, 2 * np.pi, n_pores)[:, None]
        normals = np.cos(d) * normals + np.sin(d) * (np.cos(az) * u0 + np.sin(az) * v0)
    u, v = _frames_from_normals(normals)
    return np.column_stack([px, py, pz]), u, v, normals


def _npc_channel_subunits(scaffold, *, diameter, ring_sep, symmetry=8, z_offset=0.0,
                          radial_offset=0.0, jitter=3.0, rng=None):
    """Subunit (molecule) positions for one channel labelling the shared scaffold
    with its own ``diameter`` / inter-ring ``ring_sep`` / ``z_offset`` (relative
    location along the pore axis). ``ring_sep=0`` → a single ring."""
    centres, u, v, normals = scaffold
    ang = np.arange(int(symmetry)) * 2 * np.pi / int(symmetry)
    rings = [-ring_sep / 2.0, ring_sep / 2.0] if ring_sep > 1e-6 else [0.0]
    local = np.array([[(diameter / 2.0) * np.cos(a) + radial_offset,
                       (diameter / 2.0) * np.sin(a), zc + z_offset]
                      for zc in rings for a in ang])            # (K, 3) in (u,v,n)
    subs = (centres[:, None, :]
            + local[None, :, 0, None] * u[:, None, :]
            + local[None, :, 1, None] * v[:, None, :]
            + local[None, :, 2, None] * normals[:, None, :]).reshape(-1, 3)
    if rng is not None and jitter:
        subs = subs + rng.normal(0.0, jitter, subs.shape)
    return subs


#: 3 co-registered NPC channels: (name, LUT, diameter param, ring_sep, z_offset).
_NPC_OVERLAY_CHANNELS = (
    ("Nup-scaffold", "Red", "ch1_diameter_nm", "two", 0.0),
    ("Nup-inner", "Green", "ch2_diameter_nm", "two", 0.0),
    ("central-cap", "Blue", "ch3_diameter_nm", "single", 0.5),   # single ring, +½·sep along axis
)


def simulate_npc_overlay(params: dict, *, locs_per_trace: float = 4.0,
                         precision_nm: float = 5.0, seed: int | None = None) -> list[dict]:
    """3 co-registered NPC labelling channels on ONE shared pore scaffold, each with
    its own diameter / inter-ring / axial location. Returns a list of
    ``{name, lut, coords, tid, attrs}`` — build them as an overlay."""
    rng = np.random.default_rng(seed)
    p = {**default_params("npc_overlay_3ch"), **(params or {})}
    # non-overlap by the largest labelling diameter so distinct pores don't touch
    max_diam = max(float(p["ch1_diameter_nm"]), float(p["ch2_diameter_nm"]), float(p["ch3_diameter_nm"]))
    scaffold = _npc_scaffold(int(p["n_pores"]), float(p["size_nm"]),
                             float(p["field_curvature"]), float(p["local_tilt_deg"]), rng,
                             min_separation=max_diam)
    ring_sep = float(p["ring_sep_nm"])
    out = []
    for name, lut, diam_key, rings, zoff_frac in _NPC_OVERLAY_CHANNELS:
        subs = _npc_channel_subunits(
            scaffold, diameter=float(p[diam_key]),
            ring_sep=(ring_sep if rings == "two" else 0.0),
            z_offset=zoff_frac * ring_sep, rng=rng)
        coords, tid = _expand_to_localizations(subs, locs_per_trace, precision_nm, rng)
        out.append({"name": name, "lut": lut, "coords": coords, "tid": tid,
                    "attrs": simulate_attributes(coords.shape[0], rng)})
    return out


def simulate_npc_dcr(params: dict, *, locs_per_trace: float = 4.0,
                     precision_nm: float = 5.0, seed: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Two reporters (outer + inner ring) on ONE shared NPC scaffold, mixed into a
    **single** dataset whose ``dcr`` attribute is **bimodal** — the two channels are
    recoverable only via DCR (spectral) separation. Returns ``(coords, tid, attrs)``."""
    rng = np.random.default_rng(seed)
    p = {**default_params("npc_dcr_2ch"), **(params or {})}
    max_diam = max(float(p["outer_diameter_nm"]), float(p["inner_diameter_nm"]))
    scaffold = _npc_scaffold(int(p["n_pores"]), float(p["size_nm"]),
                             float(p["field_curvature"]), float(p["local_tilt_deg"]), rng,
                             min_separation=max_diam)
    ring_sep = float(p["ring_sep_nm"])
    reporters = (("outer", float(p["outer_diameter_nm"]), float(p["dcr_low"])),
                 ("inner", float(p["inner_diameter_nm"]), float(p["dcr_high"])))
    coords_all, tid_all, dcr_all = [], [], []
    offset = 0
    for _name, diam, dcr_mean in reporters:
        subs = _npc_channel_subunits(scaffold, diameter=diam, ring_sep=ring_sep, rng=rng)
        coords, tid = _expand_to_localizations(subs, locs_per_trace, precision_nm, rng)
        coords_all.append(coords)
        tid_all.append(tid + offset)
        offset += int(tid.max()) if tid.size else 0
        dcr_all.append(np.clip(rng.normal(dcr_mean, 0.05, coords.shape[0]), 0.0, 1.0))
    coords = np.vstack(coords_all)
    tid = np.concatenate(tid_all)
    attrs = simulate_attributes(coords.shape[0], rng)
    attrs["dcr"] = np.concatenate(dcr_all)               # bimodal → recoverable by DCR
    return coords, tid, attrs


# --------------------------------------------------------------------------- #
# MBM / fiducial-bead reference tracks (for .msr export round-trips)
# --------------------------------------------------------------------------- #
#: Structured dtype for a simulated ``grd/mbm/points`` bead array (metres / s).
#: Matches :data:`minflux_viewer.msr.writer.BEAD_DTYPE`.
BEAD_DTYPE = np.dtype([
    ("gri", "<i4"),                # bead group id
    ("xyz", "<f8", (3,)),          # position, metres
    ("tim", "<f8"),                # time, seconds
    ("str", "<f8"),                # signal strength (arbitrary)
])


def simulate_beads(n_beads: int = 4, *, n_time: int = 60, field_nm: float = 5000.0,
                   duration_s: float = 300.0, drift_nm: float = 30.0,
                   jitter_nm: float = 2.0, dim: int = 3, seed: int | None = None
                   ) -> tuple[np.ndarray, dict, list]:
    """Synthetic MBM fiducial-bead drift tracks for one acquisition.

    Places ``n_beads`` fiducials at random XY across a ``field_nm`` field and
    tracks each over ``n_time`` samples spanning ``duration_s`` seconds. A
    shared, smooth common-mode **stage drift** (≈ ``drift_nm`` total — the part
    real drift correction removes) is added to every bead, plus independent
    per-sample localization ``jitter_nm``. Returns ``(points, points_by_gri,
    used)`` where ``points`` is a ``grd/mbm/points`` structured array (``gri`` /
    ``xyz`` metres / ``tim`` seconds / ``str``), ready for the ``.msr`` writer.
    """
    rng = np.random.default_rng(seed)
    n_beads = max(1, int(n_beads))
    n_time = max(2, int(n_time))
    half = field_nm / 2.0
    homes = np.column_stack([
        rng.uniform(-half, half, n_beads),
        rng.uniform(-half, half, n_beads),
        np.zeros(n_beads) if dim < 3 else rng.uniform(-half / 10, half / 10, n_beads),
    ])
    times = np.linspace(0.0, duration_s, n_time)

    # Shared common-mode drift: a smooth random walk normalised to ≈ drift_nm.
    steps = rng.normal(0.0, 1.0, (n_time, 3))
    if dim < 3:
        steps[:, 2] = 0.0
    walk = np.cumsum(steps, axis=0)
    span = np.ptp(walk, axis=0)
    span[span == 0] = 1.0
    drift = walk / span * float(drift_nm)                # (n_time, 3) nm

    recs = []
    for b in range(n_beads):
        jit = rng.normal(0.0, float(jitter_nm), (n_time, 3))
        if dim < 3:
            jit[:, 2] = 0.0
        pos_nm = homes[b] + drift + jit                  # (n_time, 3) nm
        strength = rng.normal(120.0, 15.0, n_time)
        for k in range(n_time):
            recs.append((b + 1, pos_nm[k] * 1e-9, float(times[k]), float(strength[k])))
    points = np.array(recs, dtype=BEAD_DTYPE)
    points_by_gri = {str(b + 1): {"name": f"R{b + 1}", "gri": b + 1} for b in range(n_beads)}
    used = [f"R{b + 1}" for b in range(n_beads)]
    return points, points_by_gri, used
