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
_ECOLI_PARAMS = [
    # --- Measured from the 2026 reference set (19 hand-drawn E. coli, 10
    # 3-D MINFLUX acquisitions, 1.0 M localizations). Each default is the
    # per-cell median; the tooltip carries the observed p10-p90 spread, and
    # the parameters are listed most-variable first within each group.
    ParamSpec("acquisition_s", "Acquisition duration", 28800, 1, 200000,
              False, " s",
              desc="Total acquisition time. MEASURED median 28,800 s (8 h); "
                   "range 700-43,000 s. This is the most variable quantity in "
                   "the reference set (CV 54%) and the one that most affects "
                   "real data quality — long runs accumulate drift."),
    ParamSpec("subunit_density_per_um2", "Subunit density", 375, 1, 20000,
              False, " /um2",
              desc="Labelled subunits per um2 of cell surface, one trace each. "
                   "MEASURED median 373; range 155-537 (CV 45%), extremes 51 "
                   "and 706. Sets how far apart unrelated molecules sit, so it "
                   "governs how easily a planted pair distance stands out."),
    ParamSpec("cell_length_nm", "Cell length", 1900, 500, 20000, False, " nm",
              desc="Tip-to-tip length of the capsule. MEASURED median 1924 nm; "
                   "range 1594-2882 nm (CV 26%), extremes 1535 and 3252."),
    ParamSpec("cell_radius_nm", "Cell radius", 370, 50, 3000, False, " nm",
              desc="Capsule radius; the cell is 2x this wide. MEASURED median "
                   "369 nm; range 344-390 nm. The most tightly constrained "
                   "quantity in the set (CV 8%) — E. coli width is regulated."),
    ParamSpec("locs_per_trace_tail", "Locs/trace tail (lognormal sigma)", 1.24,
              0.0, 3.0, False, "",
              desc="Shape of the localizations-per-trace distribution, which is "
                   "strongly heavy-tailed. FITTED to the reference quantiles "
                   "(median 14, mean 29, p90 71); suggested range 1.1-1.4. The "
                   "dialog's 'Locs per trace' sets the median. Note the real "
                   "lower tail is fatter than lognormal — 10% of traces carry a "
                   "single localization — so this value is fitted to the mean "
                   "and p90 rather than to the 16-84 spread."),
    ParamSpec("precision_z_nm", "Per-loc precision Z", 3.3, 0.1, 100, False, " nm",
              desc="Per-localization scatter about the true subunit position "
                   "along z. MEASURED median 3.30 nm; range 2.92-3.77 (CV 11%). "
                   "The dialog's 'Precision' sets x and y, measured 5.96 nm per "
                   "axis. Both are debiased for the finite localization count "
                   "per trace, so they are per-localization sigmas."),
    ParamSpec("trace_burst_s", "Trace burst duration", 0.15, 0.0, 3600, False, " s",
              desc="How long one trace's localizations span. MEASURED median "
                   "0.147 s; range 0.133-0.172 s. Short relative to the run, "
                   "which is why a trace is a point in time rather than a "
                   "trajectory."),
    # --- NOT inferable from the data: these are the hypothesis being tested.
    # The reference measurements cannot constrain them, which is exactly why the
    # simulation is needed — it supplies the ground truth the analysis is
    # checked against.
    ParamSpec("dimer_distance_nm", "Dimer subunit distance", 14.0, 0.5, 200,
              False, " nm",
              desc="HYPOTHESIS, not measured: mean centre-to-centre distance "
                   "between the two subunits of a dimer — the quantity the pair "
                   "analysis should recover. The real data does not constrain "
                   "it; that is the open question."),
    ParamSpec("dimer_distance_sd_nm", "Dimer distance spread", 3.0, 0.0, 100,
              False, " nm",
              desc="HYPOTHESIS, not measured: standard deviation of that "
                   "distance, so the planted population is a distribution "
                   "rather than a single value."),
    ParamSpec("dimer_fraction", "Fraction of subunits in dimers", 1.0, 0.0, 1.0,
              False, "",
              desc="HYPOTHESIS, not measured. 1.0 = every subunit belongs to a "
                   "dimer; 0.0 = all subunits are isolated monomers (the "
                   "negative control). Lower values dilute the planted signal "
                   "with unpaired background."),
    ParamSpec("detection_probability", "Detection probability", 1.0, 0.05, 1.0,
              False, "",
              desc="HYPOTHESIS, not measured: probability that a subunit is "
                   "labelled and detected. Below 1 it breaks dimers into single "
                   "observed sites, which is the dominant way a real labelling "
                   "efficiency weakens the signal. Sweep it to find the "
                   "efficiency at which a planted distance stops being "
                   "detectable."),
    ParamSpec("pair_in_membrane", "Pair axis in membrane plane", 1.0, 0.0, 1.0,
              False, "",
              desc="Structural assumption, not measured. 1 = the two subunits "
                   "lie in the local membrane plane (tangential, the physical "
                   "case for a membrane complex); 0 = the pair axis is "
                   "isotropic."),
]


_TRACK_PARAMS = [
    ParamSpec("n_tracks", "Number of tracks", 60, 1, 5000, True, "",
              desc="How many separate molecules are tracked. Each becomes one "
                   "trace with its own start time."),
    ParamSpec("radius_nm", "Shell radius", 400, 20, 5000, False, " nm",
              desc="Radius of the spherical shell the molecules diffuse on."),
    ParamSpec("n_shells", "Number of shells", 8, 1, 500, True, "",
              desc="How many shells are in the field. Tracks are shared out "
                   "between them, as separate structures in one acquisition."),
    ParamSpec("field_nm", "Field size", 8000, 500, 100000, False, " nm",
              desc="Width of the simulated field the shells are placed in."),
    ParamSpec("locs_per_track", "Localizations per track", 120, 3, 20000, True, "",
              desc="Mean length of a trace, Poisson-distributed. The reference "
                   "file has a median of 77."),
    ParamSpec("dt_ms", "Sampling interval", 0.9, 0.001, 1000.0, False, " ms",
              desc="Time between consecutive localizations in a trace. The "
                   "reference file samples every 0.9 ms."),
    ParamSpec("step_nm", "Step size", 12.0, 0.1, 1000.0, False, " nm",
              desc="Root-mean-square displacement per sampling interval, along "
                   "the shell surface. The reference file steps 12.7 nm."),
    ParamSpec("acquisition_s", "Acquisition length", 600.0, 0.1, 100000.0, False, " s",
              desc="How long the run lasts. Traces start at random times inside "
                   "it, which is what makes absolute-time playback mostly empty "
                   "and trace-relative playback the useful one."),
]

_NPC_TRACK_PARAMS = _SCAFFOLD_PARAMS + [
    ParamSpec("scaffold_diameter_nm", "Scaffold ring diameter", 107, 10, 500, False, " nm",
              desc="Diameter of the labelled NPC scaffold ring."),
    ParamSpec("n_tracks", "Number of cargo tracks", 40, 1, 5000, True, "",
              desc="How many cargo molecules are tracked through the pores."),
    ParamSpec("channel_radius_nm", "Central channel radius", 25, 2, 300, False, " nm",
              desc="Lateral confinement of the cargo inside the pore."),
    ParamSpec("travel_nm", "Axial travel", 180, 10, 2000, False, " nm",
              desc="How far along the pore axis a translocating cargo travels."),
    ParamSpec("abort_fraction", "Aborted fraction", 0.35, 0.0, 1.0, False, "",
              desc="Fraction of cargo that enters, fails to translocate and "
                   "returns the way it came."),
    ParamSpec("locs_per_track", "Localizations per track", 120, 3, 20000, True, "",
              desc="Mean trace length, Poisson-distributed."),
    ParamSpec("dt_ms", "Sampling interval", 0.9, 0.001, 1000.0, False, " ms",
              desc="Time between consecutive localizations in a trace."),
    ParamSpec("acquisition_s", "Acquisition length", 600.0, 0.1, 100000.0, False, " s",
              desc="How long the run lasts; traces start at random times in it."),
]


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
    "ecoli_hlyb_dimer": ("ecoli", "E. coli HlyB dimers (rod cell, known distance)",
                         _ECOLI_PARAMS),
    "tracking_shells": ("tracking", "Tracking: molecules on spherical shells",
                        _TRACK_PARAMS),
    "npc_tracking_2ch": ("track_overlay",
                         "Tracking: NPC scaffold + cargo (2 channels)",
                         _NPC_TRACK_PARAMS),
}


def param_specs(key: str) -> list:
    """The ``ParamSpec`` list for a structure or multi-sim key."""
    if key in STRUCTURES:
        return STRUCTURES[key][2]
    if key in MULTI_SIMS:
        return MULTI_SIMS[key][2]
    raise KeyError(key)


def sim_kind(key: str) -> str:
    """The shape of a simulation's output.

    ``single`` a plain structure; ``overlay`` / ``dcr`` the multi-channel
    labelling sims; ``tracking`` one dataset whose molecules move and which
    therefore carries ``tim``; ``track_overlay`` a structure channel plus a
    tracking channel, the two-colour trafficking design.
    """
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


# --------------------------------------------------------------------------- #
# E. coli HlyB dimer simulation — a known-distance control for the staged
# subunit pair analysis, matched to the measured statistics of real 3-D MINFLUX
# acquisitions (see ``_ECOLI_PARAMS`` for where each default comes from).
# --------------------------------------------------------------------------- #
def capsule_surface_points(n: int, length_nm: float, radius_nm: float, rng):
    """*n* points sampled uniformly **by area** on a capsule surface.

    Returns ``(points (n,3), normals (n,3))`` with the long axis along x. Uniform
    by area — not by angle — so the caps are not over-sampled relative to the
    cylinder, which would put a spurious density gradient along the cell.
    """
    n = max(int(n), 0)
    if n == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    radius = max(float(radius_nm), 1e-6)
    barrel = max(float(length_nm) - 2.0 * radius, 0.0)
    area_barrel = 2.0 * np.pi * radius * barrel
    area_caps = 4.0 * np.pi * radius * radius
    total = area_barrel + area_caps
    on_barrel = ((rng.random(n) < (area_barrel / total)) if total > 0
                 else np.ones(n, bool))

    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    along = rng.uniform(-0.5 * barrel, 0.5 * barrel, n)
    barrel_pts = np.column_stack([along, radius * np.cos(phi), radius * np.sin(phi)])
    barrel_nrm = np.column_stack([np.zeros(n), np.cos(phi), np.sin(phi)])

    sphere = rng.normal(size=(n, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    cap_shift = np.where(sphere[:, 0] >= 0.0, 0.5 * barrel, -0.5 * barrel)
    cap_pts = sphere * radius + np.column_stack([cap_shift, np.zeros(n), np.zeros(n)])

    points = np.where(on_barrel[:, None], barrel_pts, cap_pts)
    normals = np.where(on_barrel[:, None], barrel_nrm, sphere)
    return points, normals


def _tangent_frame(normals: np.ndarray):
    """Two orthonormal vectors spanning the tangent plane of each normal."""
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (normals.shape[0], 1))
    parallel = np.abs(normals[:, 2]) > 0.9
    reference[parallel] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(normals, reference)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(normals, t1)
    t2 /= np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12)
    return t1, t2




def simulate_ecoli_hlyb(params: dict, *, locs_per_trace: float = 14.0,
                        precision_nm: float = 5.7,
                        seed: int | None = None):
    """One rod-shaped cell carrying labelled HlyB **dimers**, as a control.

    Every trace is one labelled subunit observed exactly once, so a trace is
    never a repeat visit to a molecule already seen. That is deliberate: it is
    the idealisation the staged pair analysis assumes, and it makes the
    simulation a clean test of whether the workflow recovers a planted
    inter-subunit distance when that assumption actually holds.

    Trimers are not modelled — pairs only.

    Returns ``(coords_nm (N,3), tid (N,), attrs)``; ``attrs`` carries ``tim``.
    """
    rng = np.random.default_rng(seed)
    p = {**default_params("ecoli_hlyb_dimer"), **(params or {})}
    length = float(p["cell_length_nm"])
    radius = float(p["cell_radius_nm"])
    if length < 2.0 * radius:
        raise ValueError(
            f"cell_length_nm ({length:g}) must be at least twice "
            f"cell_radius_nm ({radius:g}) — a capsule cannot be shorter than "
            f"its own end caps")

    barrel = max(length - 2.0 * radius, 0.0)
    area_nm2 = 2.0 * np.pi * radius * barrel + 4.0 * np.pi * radius * radius
    n_subunits = max(int(round(float(p["subunit_density_per_um2"])
                               * area_nm2 / 1e6)), 2)

    fraction = float(np.clip(p["dimer_fraction"], 0.0, 1.0))
    n_dimers = int(round(0.5 * n_subunits * fraction))
    n_monomers = max(n_subunits - 2 * n_dimers, 0)

    sites = []
    if n_dimers:
        centres, normals = capsule_surface_points(n_dimers, length, radius, rng)
        # Distances are drawn per dimer, so the planted population is a
        # distribution; the fold guards the rare negative draw of a wide SD.
        separation = np.abs(rng.normal(float(p["dimer_distance_nm"]),
                                       float(p["dimer_distance_sd_nm"]), n_dimers))
        t1, t2 = _tangent_frame(normals)
        angle = rng.uniform(0.0, 2.0 * np.pi, n_dimers)
        tangential = (np.cos(angle)[:, None] * t1 + np.sin(angle)[:, None] * t2)
        if float(p["pair_in_membrane"]) >= 0.5:
            axis = tangential
        else:
            axis = rng.normal(size=(n_dimers, 3))
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
        half = 0.5 * separation[:, None] * axis
        sites.append(centres + half)
        sites.append(centres - half)
    if n_monomers:
        sites.append(capsule_surface_points(n_monomers, length, radius, rng)[0])
    subunits = np.vstack(sites) if sites else np.empty((0, 3))

    detected = rng.random(subunits.shape[0]) < float(p["detection_probability"])
    subunits = subunits[detected]
    if subunits.shape[0] == 0:
        raise ValueError("no subunit was detected — raise the density or the "
                         "detection probability")

    # Heavy-tailed localizations per trace: the dialog's value is the median.
    median = max(float(locs_per_trace), 1.0)
    counts = np.maximum(np.round(rng.lognormal(
        np.log(median), max(float(p["locs_per_trace_tail"]), 0.0),
        subunits.shape[0])).astype(np.int64), 1)
    tid = np.repeat(np.arange(1, subunits.shape[0] + 1, dtype=np.int64), counts)
    total = int(counts.sum())

    sigma_xy = max(float(precision_nm), 0.0)
    sigma_z = max(float(p["precision_z_nm"]), 0.0)
    jitter = np.column_stack([rng.normal(0.0, sigma_xy, total),
                              rng.normal(0.0, sigma_xy, total),
                              rng.normal(0.0, sigma_z, total)])
    coords = np.repeat(subunits, counts, axis=0) + jitter

    # One short observation per trace at a random time — no revisits.
    duration = float(p["acquisition_s"])
    starts = rng.uniform(0.0, duration, subunits.shape[0])
    span = max(float(p["trace_burst_s"]), 0.0)
    burst = np.concatenate([np.sort(rng.uniform(0.0, span, int(c)))
                            for c in counts])
    tim = np.repeat(starts, counts) + burst

    attrs = simulate_attributes(total, rng)
    attrs["tim"] = tim
    return coords, tid, attrs


# --------------------------------------------------------------------------- #
# Tracking simulations — molecules that MOVE, so the Tracking View has something
# to play. Two shapes, matching the two experiments it has to serve: a pure
# tracking run, and a structure channel with a tracking channel referred to it
# (the two-colour NPC-trafficking design).
#
# The step sizes and sampling below are taken from the reference tracking file
# (1_sample_A_1-100_seqTrk-3D-Seb_Octahedron): 0.9 ms between localizations,
# 12.7 nm median step, ~77 localizations and 104 nm end-to-end per trace.
# --------------------------------------------------------------------------- #

def _track_lengths(n_tracks: int, mean_locs: float, rng) -> np.ndarray:
    """Poisson trace lengths with a floor of 3 -- two points is a line, not a track."""
    return np.maximum(rng.poisson(max(float(mean_locs), 3.0), int(n_tracks)), 3)


def _track_times(counts: np.ndarray, dt_s: float, acquisition_s: float, rng
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absolute times, and each trace's onset.

    Traces start at random points in the acquisition, which is what a real run
    looks like -- and is exactly why the Tracking View zeroes them by default.
    """
    onsets = np.sort(rng.uniform(0.0, max(float(acquisition_s), dt_s), counts.size))
    within = np.concatenate([np.arange(c, dtype=float) * dt_s for c in counts])
    return np.repeat(onsets, counts) + within, onsets


def _walk_on_sphere(n_steps: int, radius: float, step: float, rng) -> np.ndarray:
    """A random walk constrained to the surface of a sphere of *radius*.

    Each step is taken in the tangent plane and then projected back onto the
    shell, so the track stays on the surface instead of drifting off it -- which
    is what "a molecule moving on the surface of a spherical shell" means.
    """
    start = rng.normal(size=3)
    start /= np.linalg.norm(start) or 1.0
    out = np.empty((n_steps, 3), dtype=float)
    point = start
    for i in range(n_steps):
        out[i] = point
        tangent = rng.normal(size=3)
        tangent -= point * float(np.dot(tangent, point))     # project into the plane
        norm = np.linalg.norm(tangent)
        if norm > 0:
            point = point + tangent / norm * (step / max(radius, 1e-9))
            point /= np.linalg.norm(point) or 1.0
    return out * radius


def simulate_tracking_shells(params: dict, *, precision_nm: float = 5.0,
                             seed: int | None = None
                             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Molecules diffusing on spherical shells: a **pure tracking** dataset.

    The shape of the reference file, whose molecules move on the surface of a
    shell. Returns ``(coords, tid, attrs)`` with ``attrs["tim"]`` set, so the
    result is a dataset the Tracking View can play.
    """
    rng = np.random.default_rng(seed)
    p = {**default_params("tracking_shells"), **(params or {})}
    n_tracks = int(p["n_tracks"])
    radius = float(p["radius_nm"])
    n_shells = max(int(p["n_shells"]), 1)
    field = float(p["field_nm"])
    dt_s = float(p["dt_ms"]) * 1.0e-3

    half = field / 2.0
    xy = place_nonoverlapping(n_shells, 2.2 * radius, rng, box_half=(half, half))
    if xy.shape[0] == 0:
        xy = np.zeros((1, 2))
    # Shells sit at different depths too, so the XZ/YZ projections are not a line.
    centres = np.column_stack([xy, rng.uniform(-half / 4.0, half / 4.0, xy.shape[0])])
    counts = _track_lengths(n_tracks, p["locs_per_track"], rng)
    shell_of = rng.integers(0, centres.shape[0], n_tracks)

    parts = [
        _walk_on_sphere(int(c), radius, float(p["step_nm"]), rng) + centres[s]
        for c, s in zip(counts, shell_of)
    ]
    coords = np.vstack(parts) + rng.normal(0.0, float(precision_nm),
                                           size=(int(counts.sum()), 3))
    tid = np.repeat(np.arange(1, n_tracks + 1, dtype=np.int64), counts)
    tim, _onsets = _track_times(counts, dt_s, float(p["acquisition_s"]), rng)
    attrs = simulate_attributes(coords.shape[0], rng)
    attrs["tim"] = tim
    return coords, tid, attrs


def _cargo_through_pore(n_steps: int, frame, centre, channel_radius: float,
                        travel: float, abort: bool, rng) -> np.ndarray:
    """One cargo trajectory along a pore axis, laterally confined in the channel.

    A translocation runs the length of the axis; an aborted one turns round part
    way and comes back, which is the event the two-colour experiment is looking
    for.
    """
    u, v, normal = frame
    progress = np.linspace(-0.5, 0.5, n_steps)
    if abort:
        turn = rng.uniform(0.15, 0.45)
        half = n_steps // 2
        progress = np.concatenate([
            np.linspace(-0.5, -0.5 + turn, half),
            np.linspace(-0.5 + turn, -0.5, n_steps - half),
        ])
    axial = progress * travel
    # A confined lateral wander: an Ornstein-Uhlenbeck-ish pull to the axis, so
    # the cargo stays inside the channel instead of diffusing out of the pore.
    lateral = np.zeros((n_steps, 2))
    pos = rng.normal(0.0, channel_radius * 0.3, 2)
    for i in range(n_steps):
        lateral[i] = pos
        pos = pos * 0.92 + rng.normal(0.0, channel_radius * 0.25, 2)
        radius = float(np.hypot(*pos))
        if radius > channel_radius:
            pos *= channel_radius / radius
    return (centre
            + axial[:, None] * normal
            + lateral[:, 0:1] * u
            + lateral[:, 1:2] * v)


def simulate_npc_tracking(params: dict, *, locs_per_trace: float = 4.0,
                          precision_nm: float = 5.0, seed: int | None = None
                          ) -> list[dict]:
    """A **structure channel and a tracking channel** on one shared NPC scaffold.

    The two-colour design of the NPC-trafficking work: one channel is the pore
    scaffold, which does not move, and the other is cargo translocating through
    the pores, which does. Returns a list of channel dicts; the tracking channel
    carries ``attrs["tim"]`` and the structure channel does not need it.
    """
    rng = np.random.default_rng(seed)
    p = {**default_params("npc_tracking_2ch"), **(params or {})}
    scaffold = _npc_scaffold(int(p["n_pores"]), float(p["size_nm"]),
                             float(p["field_curvature"]), float(p["local_tilt_deg"]),
                             rng, min_separation=float(p["scaffold_diameter_nm"]))
    centres, u_axes, v_axes, normals = scaffold

    subs = _npc_channel_subunits(scaffold, diameter=float(p["scaffold_diameter_nm"]),
                                 ring_sep=float(p["ring_sep_nm"]), rng=rng)
    struct_coords, struct_tid = _expand_to_localizations(
        subs, locs_per_trace, precision_nm, rng)

    n_tracks = int(p["n_tracks"])
    counts = _track_lengths(n_tracks, p["locs_per_track"], rng)
    pore_of = rng.integers(0, centres.shape[0], n_tracks)
    aborts = rng.random(n_tracks) < float(p["abort_fraction"])
    parts = [
        _cargo_through_pore(int(c), (u_axes[i], v_axes[i], normals[i]), centres[i],
                            float(p["channel_radius_nm"]), float(p["travel_nm"]),
                            bool(a), rng)
        for c, i, a in zip(counts, pore_of, aborts)
    ]
    cargo = np.vstack(parts) + rng.normal(0.0, precision_nm,
                                          size=(int(counts.sum()), 3))
    cargo_tid = np.repeat(np.arange(1, n_tracks + 1, dtype=np.int64), counts)
    tim, _ = _track_times(counts, float(p["dt_ms"]) * 1.0e-3,
                          float(p["acquisition_s"]), rng)

    struct_attrs = simulate_attributes(struct_coords.shape[0], rng)
    cargo_attrs = simulate_attributes(cargo.shape[0], rng)
    cargo_attrs["tim"] = tim
    return [
        {"name": "NPC scaffold", "lut": "Green", "coords": struct_coords,
         "tid": struct_tid, "attrs": struct_attrs, "role": "structure"},
        {"name": "Cargo tracks", "lut": "Magenta", "coords": cargo,
         "tid": cargo_tid, "attrs": cargo_attrs, "role": "tracking"},
    ]
