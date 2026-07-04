"""
minflux_viewer.analysis.npc_geomfit
====================================
**Canonical NPC geometry fit** — an unbiased, model-based particle averager.

Where the Wanlu two-ring method (:mod:`minflux_viewer.analysis.npc_wanlu`)
requires pre-detected **sub-units** (trace clustering), assigns them to rings by
a **Z-peak KDE**, and imposes the 8-fold by a **separate phase-alignment
heuristic**, this fits a **parametric canonical NPC directly to each particle's
raw 3-D localizations** — no sub-unit detection, no ring pre-assignment, every
parameter fit jointly. That removes those model biases and yields a per-particle
parameter table (diameter, inter-ring, rim, tilt, …) suitable for studying how a
parameter varies across conditions (e.g. ring diameter vs. wild type).

Model
-----
A canonical NPC = **two rings of ``symmetry`` (=8) Gaussian corners**. In the
canonical frame the corners sit at radius ``R = diameter/2``, evenly spaced in
angle (common in-plane phase ``φ``), on two planes ``z = ±inter/2``
(``inter = 0`` → a single ring). The whole scaffold is tilted (axis tilt +
azimuth) and centred. Each localization is modelled as a draw from the
equal-weight Gaussian mixture of the ``2·symmetry`` corners (width = ``rim``)
plus a uniform background, so the fit is a **maximum-likelihood** estimate over

    R (via diameter), inter, rim, tilt, azimuth, phase, centre(x, y, z).

Fitting is Nelder-Mead on the negative mean log-likelihood, from a data-driven
initialisation (PCA tilt + 8-fold phase of the flattened points), with bounded
parameters via logistic/tanh transforms. Pure NumPy/SciPy — Qt-free, unit-tested.
Distances are in nm.
"""

from __future__ import annotations

import numpy as np

from .geometry import rotation_a_to_b

DEFAULT_DIAMETER_BOUNDS = (60.0, 130.0)
DEFAULT_INTER_BOUNDS = (0.0, 80.0)
DEFAULT_RIM_BOUNDS = (1.0, 30.0)
DEFAULT_TILT_MAX_DEG = 45.0
DEFAULT_SYMMETRY = 8


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def npc_corner_centers(radius: float, inter: float, phase: float,
                       symmetry: int = DEFAULT_SYMMETRY) -> np.ndarray:
    """Canonical corner centres ``(2·symmetry, 3)`` (or ``(symmetry, 3)`` when
    ``inter≈0``): ``symmetry`` corners at *radius* and phase *phase*, on planes
    ``z = ±inter/2``."""
    ang = float(phase) + 2.0 * np.pi * np.arange(int(symmetry)) / int(symmetry)
    ring = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(int(symmetry))])
    if inter <= 1e-6:
        return ring
    lower = ring.copy(); lower[:, 2] = -0.5 * inter
    upper = ring.copy(); upper[:, 2] = +0.5 * inter
    return np.vstack([lower, upper])


def canonical_overlay(radius: float, inter: float, symmetry: int = DEFAULT_SYMMETRY,
                      n: int = 73):
    """Overlay geometry for the inspector → ``(corner_points, [ring_polyline, …])``
    in the canonical frame (nm): the ``2·symmetry`` corner markers + one/two ring
    circles."""
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    z_levels = [0.0] if inter <= 1e-6 else [-0.5 * inter, 0.5 * inter]
    curves = [np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.full(n, z)])
              for z in z_levels]
    corners = npc_corner_centers(radius, inter, 0.0, symmetry)
    return corners, curves


# --------------------------------------------------------------------------- #
# Bounded parameter transforms + decode
# --------------------------------------------------------------------------- #
def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _sig(x, lo, hi):
    return lo + (hi - lo) * _expit(x)


def _inv_sig(v, lo, hi):
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 1e-4, 1.0 - 1e-4)
    return float(np.log(t / (1.0 - t)))


def _decode(q, centroid, opt) -> dict:
    R = _sig(q[0], *opt["r_bounds"])
    inter = _sig(q[1], opt["d_bounds"][0], opt["d_bounds"][1])
    rim = _sig(q[2], *opt["s_bounds"])
    tilt = _sig(q[3], 0.0, opt["tilt_max_deg"])
    az = float(q[4])
    phase = float(q[5])
    shift = opt["max_shift"] * np.tanh(q[6:9])
    tr = np.deg2rad(tilt)
    normal = np.array([np.sin(tr) * np.cos(az), np.sin(tr) * np.sin(az), np.cos(tr)])
    return dict(radius=float(R), inter=float(inter), rim=float(rim), tilt_deg=float(tilt),
                azimuth=az, phase=phase, normal=normal, center=centroid + shift)


def _model_world(m: dict, symmetry: int) -> np.ndarray:
    C = npc_corner_centers(m["radius"], m["inter"], m["phase"], symmetry)
    Rot = rotation_a_to_b([0.0, 0.0, 1.0], m["normal"])
    return (Rot @ C.T).T + m["center"]


def _neg_log_likelihood(q, points, centroid, opt) -> float:
    from scipy.special import logsumexp

    m = _decode(q, centroid, opt)
    corners = _model_world(m, opt["symmetry"])
    sig2 = max(m["rim"], 1e-3) ** 2
    d2 = ((points[:, None, :] - corners[None, :, :]) ** 2).sum(axis=-1)   # (N, K)
    log_norm = -1.5 * np.log(2.0 * np.pi * sig2)
    logcomp = log_norm - d2 / (2.0 * sig2)
    log_mean = logsumexp(logcomp, axis=1) - np.log(corners.shape[0])
    b = opt["background"]
    log_signal = np.log(max(1.0 - b, 1e-12)) + log_mean
    log_bg = np.log(b * opt["bg_density"] + 1e-300)
    log_mix = np.logaddexp(log_signal, log_bg)
    val = float(-np.mean(log_mix))
    return val if np.isfinite(val) else 1e18


def _subunit_centers(points, tid):
    """LoG sub-unit centres for initialisation — reuses the Wanlu sub-unit
    clustering (density + Laplacian-of-Gaussian blob gate + single-linkage). The
    sub-units are the NPC corners/labelled sites, so their radius / n-fold phase /
    Z-separation give a robust fit seed. Returns ``(M, 3)`` or ``None``."""
    if tid is None:
        return None
    try:
        from .npc_wanlu import estimate_trace_size, locate_unit_clusters
    except Exception:
        return None
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] < 3:
        return None
    t = np.asarray(tid).ravel()
    if t.size != P.shape[0] or np.unique(t).size < 3:
        return None
    try:
        d_unit = 2.0 * float(estimate_trace_size(P[:, :3], t))
        centers, _ = locate_unit_clusters(P[:, :3], t, d_unit)
    except Exception:
        return None
    centers = np.asarray(centers, float)
    return centers if (centers.ndim == 2 and centers.shape[0] >= 3) else None


def _pca_normal(pc: np.ndarray):
    if pc.shape[0] < 3 or not np.any(np.abs(pc[:, 2]) > 1e-9):
        return np.array([0.0, 0.0, 1.0]), 0.0
    _w, V = np.linalg.eigh(pc.T @ pc)
    n = V[:, 0]
    if n[2] < 0:
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), 0.0, 1.0))))
    return n, tilt


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_npc_model(points, *, init_points=None, init_tid=None, subunit_init=True,
                  diameter_bounds=DEFAULT_DIAMETER_BOUNDS,
                  inter_bounds=DEFAULT_INTER_BOUNDS, rim_bounds=DEFAULT_RIM_BOUNDS,
                  tilt_max_deg=DEFAULT_TILT_MAX_DEG, max_center_shift=25.0,
                  symmetry=DEFAULT_SYMMETRY, background=0.1,
                  max_iter=1500, max_fev=3000) -> dict:
    """Maximum-likelihood fit of the canonical NPC to a particle's 3-D points.

    ``subunit_init`` seeds the radius / n-fold phase / inter-ring from **LoG
    sub-unit** centres (the corners), detected on ``init_points`` + ``init_tid``
    (the raw localizations + trace ids; default to *points* with no tid → the
    moment init). Init only — the fit still refines on *points*, so this never
    biases the estimate, only makes the optimizer start near the corners.

    Returns a dict with ``ok``, ``diameter``, ``inter``, ``rim``, ``tilt_deg``,
    ``azimuth``, ``phase``, ``normal``, ``center`` and quality metrics ``gof``
    (R² of point-to-nearest-corner residuals), ``rmse``, ``nrmse``, ``nll``.
    """
    from scipy.optimize import minimize

    P = np.asarray(points, float)
    P = P[np.all(np.isfinite(P[:, :3]), axis=1), :3] if P.ndim == 2 and P.shape[1] >= 3 else \
        np.empty((0, 3))
    if P.shape[0] < max(2 * int(symmetry), 12):
        return {"ok": False}

    centroid = P.mean(axis=0)
    pc = P - centroid
    r_bounds = (diameter_bounds[0] / 2.0, diameter_bounds[1] / 2.0)
    ext = np.ptp(P, axis=0)
    vol = float(max(np.prod(np.maximum(ext, 1.0)), 1.0))
    opt = dict(r_bounds=r_bounds, d_bounds=tuple(inter_bounds), s_bounds=tuple(rim_bounds),
               tilt_max_deg=float(tilt_max_deg), max_shift=float(max_center_shift),
               symmetry=int(symmetry), background=float(background), bg_density=1.0 / vol)

    # data-driven initialisation (moment-based over the fit points)
    normal0, tilt0 = _pca_normal(pc)
    Rrot = rotation_a_to_b(normal0, [0.0, 0.0, 1.0])
    flat = (Rrot @ pc.T).T
    rad_flat = np.hypot(flat[:, 0], flat[:, 1])
    R0 = float(np.clip(np.median(rad_flat), *r_bounds))
    theta = np.arctan2(flat[:, 1], flat[:, 0])
    phi0 = float(np.angle(np.mean(np.exp(1j * int(symmetry) * theta))) / int(symmetry))
    az0 = float(np.arctan2(normal0[1], normal0[0]))
    d0 = float(np.clip(2.0 * np.std(flat[:, 2]), inter_bounds[0], inter_bounds[1]))
    s0 = float(np.clip(np.std(rad_flat - R0) or 5.0, rim_bounds[0], rim_bounds[1]))

    # LoG sub-unit seed: the corners give a more robust radius / phase / inter-ring
    if subunit_init:
        units = _subunit_centers(init_points if init_points is not None else P, init_tid)
        if units is not None:
            uflat = (Rrot @ (units - centroid).T).T
            r_u = np.hypot(uflat[:, 0], uflat[:, 1])
            if r_u.size >= 3 and np.median(r_u) > 0:
                R0 = float(np.clip(np.median(r_u), *r_bounds))
                th_u = np.arctan2(uflat[:, 1], uflat[:, 0])
                phi0 = float(np.angle(np.mean(np.exp(1j * int(symmetry) * th_u))) / int(symmetry))
                zc = uflat[:, 2]
                med = float(np.median(zc))
                if np.any(zc >= med) and np.any(zc < med):
                    d0 = float(np.clip(float(np.mean(zc[zc >= med]) - np.mean(zc[zc < med])),
                                       inter_bounds[0], inter_bounds[1]))
    q0 = np.array([
        _inv_sig(R0, *r_bounds), _inv_sig(d0, inter_bounds[0], inter_bounds[1]),
        _inv_sig(s0, *rim_bounds), _inv_sig(min(tilt0, tilt_max_deg * 0.99), 0.0, tilt_max_deg),
        az0, phi0, 0.0, 0.0, 0.0], dtype=float)

    res = minimize(_neg_log_likelihood, q0, args=(pc, np.zeros(3), opt),
                   method="Nelder-Mead",
                   options=dict(maxiter=max_iter, maxfev=max_fev, xatol=1e-4, fatol=1e-6))
    m = _decode(res.x, np.zeros(3), opt)
    m["center"] = m["center"] + centroid                     # back to world frame

    corners = _model_world(m, int(symmetry))
    mind2 = ((P[:, None, :] - corners[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
    rmse = float(np.sqrt(np.mean(mind2)))
    ss_res = float(np.sum(mind2))
    ss_tot = float(np.sum((P - P.mean(axis=0)) ** 2))
    gof = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    m.update(ok=bool(np.all(np.isfinite(m["center"])) and np.isfinite(m["radius"])),
             diameter=2.0 * m["radius"], nll=float(res.fun), rmse=rmse,
             nrmse=float(rmse / m["radius"]) if m["radius"] > 0 else float("nan"),
             gof=gof, n=int(P.shape[0]))
    return m


def align_to_canonical(points, fit: dict, symmetry: int = DEFAULT_SYMMETRY) -> np.ndarray:
    """Map a particle into the canonical frame using its fit: un-tilt (normal→+Z
    about the centre), centre, then rotate by ``−phase`` to phase-lock the n-fold."""
    P = np.asarray(points, float)[:, :3]
    p0 = (rotation_a_to_b(fit["normal"], [0.0, 0.0, 1.0]) @ (P - fit["center"]).T).T
    ph = -float(fit["phase"])
    c, s = np.cos(ph), np.sin(ph)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (Rz @ p0.T).T


def _row(i, n, fit) -> dict:
    if not fit.get("ok"):
        nan = float("nan")
        return dict(particle=i, n_locs=n, diameter=nan, inter=nan, rim=nan, tilt=nan,
                    phase=nan, gof=nan, rmse=nan, nrmse=nan, accepted=False)
    sym = DEFAULT_SYMMETRY
    return dict(particle=i, n_locs=n, diameter=fit["diameter"], inter=fit["inter"],
                rim=fit["rim"], tilt=fit["tilt_deg"],
                phase=float(np.rad2deg(fit["phase"]) % (360.0 / sym)),
                gof=fit["gof"], rmse=fit["rmse"], nrmse=fit["nrmse"], accepted=False)


def average_npc_geomfit(particles, *, init_particles=None, init_tids=None,
                        subunit_init=True, diameter_bounds=DEFAULT_DIAMETER_BOUNDS,
                        inter_bounds=DEFAULT_INTER_BOUNDS, rim_bounds=DEFAULT_RIM_BOUNDS,
                        tilt_max_deg=DEFAULT_TILT_MAX_DEG, symmetry=DEFAULT_SYMMETRY,
                        min_gof=0.0, progress=None, **fit_kw) -> dict:
    """Fit the canonical NPC to every particle, align accepted ones to the canonical
    frame and pool them. Returns the pooled ``points`` (+ ``average_loc``), a
    per-particle ``table`` (diameter/inter/rim/tilt/phase/gof/…), the per-particle
    ``fits`` and ``aligned`` points (parallel; for sub-selection re-pooling).

    ``init_particles`` / ``init_tids`` (parallel to *particles*) are the raw
    localizations + trace ids used only to seed each fit from LoG sub-units
    (``subunit_init``); *particles* itself is what the likelihood fits (raw or
    already trace-collapsed)."""
    parts = [np.asarray(p, float) for p in particles]
    out = {"points": np.empty((0, 3)), "average_loc": np.empty((0, 3)), "table": [],
           "fits": [], "aligned": [], "n_particles": len(parts), "n_accepted": 0,
           "min_gof": float(min_gof), "mode": "geomfit", "symmetry": int(symmetry)}
    pooled, table, fits, aligned = [], [], [], []
    total = len(parts)
    for i, P in enumerate(parts, start=1):
        if progress is not None:
            progress(i, total)
        ip = init_particles[i - 1] if (init_particles is not None and i - 1 < len(init_particles)) else None
        it = init_tids[i - 1] if (init_tids is not None and i - 1 < len(init_tids)) else None
        fit = fit_npc_model(P, init_points=ip, init_tid=it, subunit_init=subunit_init,
                            diameter_bounds=diameter_bounds, inter_bounds=inter_bounds,
                            rim_bounds=rim_bounds, tilt_max_deg=tilt_max_deg,
                            symmetry=symmetry, **fit_kw)
        fits.append(fit)
        row = _row(i, int(np.asarray(P).shape[0]), fit)
        if fit.get("ok"):
            a = align_to_canonical(P, fit, symmetry)
            aligned.append(a)
            accepted = bool(np.isfinite(fit["gof"]) and fit["gof"] >= min_gof)
            row["accepted"] = accepted
            if accepted:
                pooled.append(a[:, :3])
        else:
            aligned.append(None)
        table.append(row)
    out["table"], out["fits"], out["aligned"] = table, fits, aligned
    out["n_accepted"] = len(pooled)
    if pooled:
        out["average_loc"] = np.vstack(pooled)
        out["points"] = out["average_loc"]
    return out
