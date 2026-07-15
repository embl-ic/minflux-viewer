"""
minflux_viewer.analysis.particle_fusion
========================================
**Template-free 3-D particle fusion** (Heydarian-style), the reference-free
alternative to the 2-D-projection averaging in :mod:`particle_average`.

Pipeline
--------
1. **GMM representation** — each particle is a sum of isotropic Gaussians (one per
   localization, width ``sigma_nm``). The registration cost between two particles is
   their Gaussian-mixture **cross-correlation** ``∫ f_a f_b`` — smooth, robust to
   the point count, and naturally weighted by proximity.
2. **All-to-all pairwise 3-D registration** — every particle pair is rigidly
   registered in **full 3-D** (rotation + translation), giving relative rotations
   ``Q_ij``. Coarse init aligns the two PCA plane normals (both flips) + an in-plane
   scan, then one local 6-DoF refine — so tilt is handled *by the alignment*, not a
   fragile one-shot PCA pre-pass.
3. **Lie-algebraic rotation averaging** (Govindu, CVPR 2004) — the noisy, possibly
   inconsistent ``Q_ij`` are fused into globally consistent absolute orientations
   ``G_i`` via iterated tangent-space (chordal) averaging on SO(3), seeded by a
   spanning tree. Avoids the single-seed-template bias of reference-free 2-D
   averaging.
4. **Data-driven template refinement** — apply ``G_i``, fuse, then iterate
   *register-each-to-the-fused-GMM → rebuild* a few times (3-D, full orientation) to
   polish the alignment against an unbiased, self-consistent reference.
5. **Symmetry detection** — the C-fold of the fused structure from the azimuthal
   angular power spectrum; optionally imposed.

References
----------
* Heydarian et al., *Nat. Methods* **15**, 781 (2018) — template-free 2-D fusion.
* Heydarian et al., *Nat. Commun.* **12**, 2847 (2021) — 3-D fusion + symmetry.
* Govindu, *CVPR* (2004) — Lie-algebraic averaging for globally consistent motion.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances in nm.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

DEFAULT_SIGMA_NM = 5.0          # GMM kernel width (localization blur)
DEFAULT_N_ITER = 3              # data-driven template refinement rounds
DEFAULT_MAX_POINTS = 60         # per-particle points used for registration (downsample)
DEFAULT_MAX_PAIRS = 4000        # all-to-all pair budget; above this, subsample edges
DEFAULT_N_INPLANE = 8           # coarse in-plane rotations tried per normal alignment


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _as_xyz(points) -> np.ndarray:
    """Return finite ``(M, 3)`` XYZ (pad Z=0 if 2-D)."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.empty((0, 3))
    xyz = P[:, :3] if P.shape[1] >= 3 else np.column_stack([P[:, :2], np.zeros(P.shape[0])])
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def _pca_normal(P: np.ndarray) -> np.ndarray:
    """Smallest-variance (plane-normal) axis, sign-fixed to the +Z hemisphere."""
    if P.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    C = P - P.mean(axis=0)
    _w, V = np.linalg.eigh(C.T @ C)
    n = V[:, 0]
    return -n if n[2] < 0 else n


def _rotation_a_to_b(a, b) -> np.ndarray:
    """Rotation matrix taking unit vector *a* to unit vector *b* (shortest arc)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])  # 180° about X
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _rot_about_axis(axis, angle) -> np.ndarray:
    ax = np.asarray(axis, float)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    return Rotation.from_rotvec(ax * float(angle)).as_matrix()


def _downsample(P: np.ndarray, max_pts: int, rng) -> np.ndarray:
    """Subsample to ``max_pts`` points (uniform) for the O(M²) registration cost."""
    if P.shape[0] <= max_pts:
        return P
    idx = rng.choice(P.shape[0], size=max_pts, replace=False)
    return P[idx]


# --------------------------------------------------------------------------- #
# GMM cross-correlation registration (3-D rigid)
# --------------------------------------------------------------------------- #
def gmm_cross_correlation(A: np.ndarray, B: np.ndarray, two_h2: float) -> float:
    """Gaussian-mixture cross-correlation ``Σ_ij exp(-|a_i-b_j|² / (2h²))`` (the
    transform-dependent term of the GMM L² overlap; ``two_h2 = 2·h²``)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float(np.exp(-d2 / two_h2).sum())


def _neg_cc(params, A, B, two_h2) -> float:
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    TB = B @ R.T + params[3:6]
    d2 = ((A[:, None, :] - TB[None, :, :]) ** 2).sum(-1)
    return -float(np.exp(-d2 / two_h2).sum())


def _fibonacci_directions(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors on the sphere (Fibonacci spiral)."""
    i = np.arange(int(n)) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / int(n))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def so3_grid(n_dir: int = 20, n_inplane: int = 6) -> np.ndarray:
    """A near-uniform SO(3) rotation grid: ``n_dir`` viewing directions (where +Z
    maps) × ``n_inplane`` in-plane spins → ``(K, 3, 3)`` rotation matrices. Covers
    the **full** rotation space, so pairwise registration works for arbitrary (not
    just planar) 3-D structures — the fix for the PCA-normal init that only spanned
    rotations of a flat particle."""
    dirs = _fibonacci_directions(n_dir)
    spins = np.linspace(0.0, 2.0 * np.pi, int(max(n_inplane, 1)), endpoint=False)
    out = []
    for d in dirs:
        R_d = _rotation_a_to_b([0.0, 0.0, 1.0], d)
        for a in spins:
            out.append(R_d @ _rot_about_axis([0.0, 0.0, 1.0], a))
    return np.array(out)


_DEFAULT_GRID = None


def _default_grid() -> np.ndarray:
    global _DEFAULT_GRID
    if _DEFAULT_GRID is None:
        _DEFAULT_GRID = so3_grid(20, 6)          # 120 rotations, ≈30° / 60° spacing
    return _DEFAULT_GRID


def register_pair(A: np.ndarray, B: np.ndarray, *, sigmas, grid=None,
                  init_R=None, top_k: int = 4, maxiter: int = 100):
    """Best 3-D rigid transform aligning *B* onto *A* (both already centred).

    Returns ``(R, t, cc)`` maximising the GMM cross-correlation. ``sigmas`` is a
    **coarse→fine** schedule of GMM widths (nm): the SO(3) *grid* (+ optional
    warm-start ``init_R``) is scored at the widest, then the best ``top_k``
    candidates are each refined (6-DoF Nelder-Mead) through the shrinking widths —
    the coarse pass finds the basin (wide = smooth landscape) and the fine passes
    lock the peak. Coarse-to-fine is essential: a single width either misses the
    basin (too fine) or the peak (too coarse)."""
    sigmas = [float(s) for s in (sigmas if np.ndim(sigmas) else [sigmas])]
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.eye(3), np.zeros(3), 0.0
    grid = _default_grid() if grid is None else grid
    cands = list(grid)
    if init_R is not None:
        cands = [np.asarray(init_R, float)] + cands
    two_h2_coarse = 2.0 * sigmas[0] ** 2
    ccs = np.array([gmm_cross_correlation(A, B @ R0.T, two_h2_coarse) for R0 in cands])
    order = np.argsort(ccs)[::-1][:max(int(top_k), 1)]
    best_x, best_cc = None, -np.inf
    for idx in order:
        x = np.concatenate([Rotation.from_matrix(cands[idx]).as_rotvec(), np.zeros(3)])
        for sig in sigmas:
            res = minimize(_neg_cc, x, args=(A, B, 2.0 * sig ** 2), method="Nelder-Mead",
                           options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-4))
            x = res.x
        if -res.fun > best_cc:
            best_cc, best_x = -res.fun, x
    return Rotation.from_rotvec(best_x[:3]).as_matrix(), best_x[3:6], float(best_cc)


# --------------------------------------------------------------------------- #
# All-to-all + Lie-algebraic rotation averaging
# --------------------------------------------------------------------------- #
def _pairs(n: int, max_pairs: int, rng) -> list[tuple[int, int]]:
    """All ``i<j`` pairs, or — above ``max_pairs`` — a connected random subset (each
    node linked to ``k`` others, guaranteeing the graph stays connected for the
    global averaging)."""
    full = n * (n - 1) // 2
    if full <= max_pairs or n <= 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    k = max(3, int(2 * max_pairs / n))
    edges = set()
    for i in range(n):                       # connect a ring (keeps it connected) + k-1 random
        edges.add((i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i))
        for j in rng.choice(n, size=min(k, n), replace=False):
            if j != i:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def all_to_all_register(particles, *, sigmas, grid=None,
                        max_pairs=DEFAULT_MAX_PAIRS, executor=None, on_done=None, rng=None):
    """Register every (subsampled) pair. Returns ``(edges, Q_list, weights)`` where
    ``Q_list[k]`` is the rotation bringing particle ``j`` into ``i``'s frame for edge
    ``(i, j)`` and ``weights[k]`` its cross-correlation score."""
    rng = np.random.default_rng(0) if rng is None else rng
    grid = _default_grid() if grid is None else grid
    n = len(particles)
    edges = _pairs(n, max_pairs, rng)

    def _one(edge):
        i, j = edge
        R, _t, cc = register_pair(particles[i], particles[j], sigmas=sigmas, grid=grid)
        return R, cc

    Qs: list = [None] * len(edges)
    ws: list = [0.0] * len(edges)
    if executor is None:
        for k, e in enumerate(edges):
            Qs[k], ws[k] = _one(e)
            if on_done:
                on_done()
    else:
        futs = {executor.submit(_one, e): k for k, e in enumerate(edges)}
        for fut in as_completed(futs):
            Qs[futs[fut]], ws[futs[fut]] = fut.result()
            if on_done:
                on_done()
    return edges, Qs, ws


def _geo_angle(Ra, Rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    return float(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec()))


def lie_average_rotations(edges, Qs, weights, n, *, n_iter=100, robust_deg=25.0) -> list[np.ndarray]:
    """Globally consistent absolute rotations ``G_i`` from pairwise ``Q_ij``
    (``G_j = G_i Q_ij``), **robust to outlier edges** (a large fraction of pairwise
    registrations can be wrong).

    Max-weight spanning-tree init (propagate only the most confident edges) +
    iteratively-reweighted chordal (SO(3)) averaging: each neighbour prediction is
    weighted by its edge cross-correlation *and* an outlier weight
    ``exp(-(residual/robust_deg)²)`` that shrinks as the prediction disagrees with
    the current estimate — so wrong edges are progressively ignored. A plain L2 mean
    (no reweighting) is dragged everywhere by outliers scattered over SO(3)."""
    if n == 0:
        return []
    adj: list[list[tuple[int, np.ndarray, float]]] = [[] for _ in range(n)]
    elist = []
    for (i, j), Q, w in zip(edges, Qs, weights):
        if Q is None:
            continue
        adj[i].append((j, Q, float(w)))          # from i: G_j = G_i Q
        adj[j].append((i, Q.T, float(w)))        # from j: G_i = G_j Qᵀ
        elist.append((i, j, Q, float(w)))
    # --- max-weight spanning tree init (Kruskal on edge cross-correlation) ---
    G: list = [None] * n
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tadj: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]
    for i, j, Q, _w in sorted(elist, key=lambda e: -e[3]):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
            tadj[i].append((j, Q)); tadj[j].append((i, Q.T))
    for s in range(n):                           # set G per connected component
        if G[s] is None:
            G[s] = np.eye(3)
            stack = [s]
            while stack:
                a = stack.pop()
                for b, Q in tadj[a]:
                    if G[b] is None:
                        G[b] = G[a] @ Q
                        stack.append(b)
    # --- iteratively-reweighted averaging ---
    scale = np.deg2rad(robust_deg)
    for _ in range(n_iter):
        maxstep = 0.0
        for j in range(n):
            if not adj[j]:
                continue
            preds, wts = [], []
            for i, Q, w in adj[j]:
                p = G[i] @ Q
                resid = _geo_angle(p, G[j])
                preds.append(p)
                wts.append(w * np.exp(-(resid / scale) ** 2))
            wts = np.asarray(wts, float)
            if wts.sum() <= 1e-9:
                continue
            mean = Rotation.from_matrix(np.array(preds)).mean(weights=wts).as_matrix()
            maxstep = max(maxstep, _geo_angle(G[j], mean))
            G[j] = mean
        if maxstep < 1e-4:
            break
    return G


# --------------------------------------------------------------------------- #
# Fusion + data-driven refinement
# --------------------------------------------------------------------------- #
def _apply(P, R, t=None) -> np.ndarray:
    out = P @ np.asarray(R, float).T
    return out if t is None else out + np.asarray(t, float)


def refine_to_template(particles, G, *, two_h2, n_iter=DEFAULT_N_ITER, grid=None,
                       sigmas=None, tmpl_max=800, executor=None, on_done=None):
    """Polish absolute rotations by iterating *register-each-to-the-fused-GMM →
    rebuild*, **warm-started** from the current orientation ``G[i]`` (so the search
    is local, escaping only if the grid finds a clearly better basin). ``sigmas``
    (one GMM width per round, coarse→fine) overrides ``two_h2``. Returns
    ``(G_refined, translations, last_template)``."""
    G = [np.asarray(g, float) for g in G]
    T = [np.zeros(3) for _ in G]
    n = len(particles)
    grid = _default_grid() if grid is None else grid
    rng = np.random.default_rng(0)
    rounds = list(sigmas) if sigmas is not None else [np.sqrt(two_h2 / 2.0)] * max(int(n_iter), 1)
    ds_template_pts = None
    for sig in rounds:
        h2 = 2.0 * float(sig) ** 2
        template = np.vstack([_apply(particles[i], G[i], T[i]) for i in range(n)]) \
            if n else np.empty((0, 3))
        template = template - template.mean(axis=0) if template.shape[0] else template
        if template.shape[0] > tmpl_max:         # cap for the O(M²) cost
            template = template[rng.choice(template.shape[0], tmpl_max, replace=False)]
        ds_template_pts = template

        def _one(i):
            return register_pair(template, particles[i], sigmas=[2.0 * sig, sig],
                                 grid=grid, init_R=G[i], top_k=2)[:2]

        results: list = [None] * n
        if executor is None:
            for i in range(n):
                results[i] = _one(i)
                if on_done:
                    on_done()
        else:
            futs = {executor.submit(_one, i): i for i in range(n)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                if on_done:
                    on_done()
        for i in range(n):
            G[i], T[i] = results[i]
    return G, T, ds_template_pts


def detect_symmetry(points, *, max_fold=12, n_bins=720):
    """Detect the C-fold symmetry of a fused structure from the azimuthal angular
    power spectrum (assumes the symmetry axis ≈ +Z, as the fusion produces). Returns
    ``(fold, confidence)`` — confidence = dominant harmonic power / total (0..1)."""
    P = _as_xyz(points)
    if P.shape[0] < max_fold * 3:
        return 1, 0.0
    P = P - P.mean(axis=0)
    r = np.hypot(P[:, 0], P[:, 1])
    keep = r > np.median(r) * 0.3                # drop the axial core
    th = np.arctan2(P[keep, 1], P[keep, 0])
    hist, _ = np.histogram(th, bins=n_bins, range=(-np.pi, np.pi))
    spec = np.abs(np.fft.rfft(hist - hist.mean())) ** 2
    folds = np.arange(2, int(max_fold) + 1)
    power = spec[folds]
    best = int(folds[np.argmax(power)])
    conf = float(power.max() / (spec[2:].sum() + 1e-9))
    return best, conf


def symmetrize(points, fold: int) -> np.ndarray:
    """C``fold`` symmetry expansion about +Z: replicate the cloud at every 2π/fold
    rotation (boosts SNR of the fused reconstruction)."""
    P = _as_xyz(points)
    if fold <= 1 or P.shape[0] == 0:
        return P
    out = []
    for k in range(int(fold)):
        out.append(_apply(P, _rot_about_axis([0, 0, 1.0], 2 * np.pi * k / fold)))
    return np.vstack(out)


def _matrix_4x4(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, float)
    M[:3, 3] = np.asarray(t, float)
    return M


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
#: Confidence below which a detected C-fold is treated as noise (asymmetric → C1).
_SYM_CONF_MIN = 0.35


def fuse_particles(particles, *, sigma_nm=DEFAULT_SIGMA_NM, n_iter=DEFAULT_N_ITER,
                   max_points=DEFAULT_MAX_POINTS, max_pairs=DEFAULT_MAX_PAIRS,
                   n_dir=20, n_inplane=6, detect_sym=True, impose_symmetry=False,
                   seed=0, progress=None, n_workers=None) -> dict:
    """Template-free 3-D fusion of *particles* (each an ``(M, ≥2)`` re-zeroed nm
    cloud). Returns a dict with ``points`` (fused ``(N, 3)`` cloud), per-particle
    ``particle_transforms`` (4×4, incl. the centring), ``symmetry`` / ``sym_conf``,
    ``n_particles``, ``mode="fusion"``.

    Pairwise + template registration search the full SO(3) grid (``n_dir`` × n_inplane``);
    the template refinement runs **coarse-to-fine** in the GMM width (5·``sigma_nm`` →
    ``sigma_nm``). ``impose_symmetry`` C-fold-expands the fused cloud only when a
    confident symmetry is detected — **off by default** (imposing symmetry is a
    modelling bias). ``progress(done, total)`` optional; ``n_workers`` (``None`` → all
    CPUs) parallelises both heavy phases."""
    rng = np.random.default_rng(seed)
    grid = so3_grid(n_dir, n_inplane)
    two_h2 = 2.0 * float(sigma_nm) ** 2
    # centre each particle; keep full points + a downsampled copy for registration
    full, cents = [], []
    for p in particles:
        P = _as_xyz(p)
        c = P.mean(axis=0) if P.shape[0] else np.zeros(3)
        full.append(P - c)
        cents.append(c)
    reg = [_downsample(P, max_points, rng) for P in full]
    n = len(full)
    if n == 0:
        return {"points": np.empty((0, 3)), "particle_transforms": [], "symmetry": 1,
                "sym_conf": 0.0, "n_particles": 0, "mode": "fusion", "iterations": 0}
    if n == 1:
        return {"points": full[0], "particle_transforms": [np.eye(4)], "symmetry": 1,
                "sym_conf": 0.0, "n_particles": 1, "mode": "fusion", "iterations": 0}

    edges = _pairs(n, max_pairs, rng)
    total = len(edges) + n * max(int(n_iter), 1)
    done = [0]

    def _tick():
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    workers = max(1, min(int(os.cpu_count() or 1) if n_workers is None else int(n_workers), n * n))
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    n_it = max(int(n_iter), 1)
    try:
        # 1) all-to-all pairwise registration, coarse→fine GMM width per pair (wide =
        #    smooth landscape to find the basin; fine = lock the peak).
        a2a_sigmas = [4.0 * float(sigma_nm), 2.0 * float(sigma_nm), float(sigma_nm)]
        edges, Qs, ws = all_to_all_register(
            reg, sigmas=a2a_sigmas, grid=grid, max_pairs=max_pairs,
            executor=executor, on_done=_tick, rng=rng)
        # 2) global rotation averaging (Lie-algebraic, unbiased consensus over pairs)
        G = lie_average_rotations(edges, Qs, ws, n)
        # 3) data-driven template refinement, coarse→fine width across rounds
        sigmas = np.linspace(4.0 * float(sigma_nm), float(sigma_nm), n_it)
        G, T, _tmpl = refine_to_template(
            reg, G, two_h2=two_h2, sigmas=sigmas, grid=grid,
            executor=executor, on_done=_tick)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # fuse the FULL (non-downsampled) clouds with the found transforms
    pooled = np.vstack([_apply(full[i], G[i], T[i]) for i in range(n)]) if n else np.empty((0, 3))
    pooled = pooled - pooled.mean(axis=0) if pooled.shape[0] else pooled
    fold, conf = detect_symmetry(pooled) if detect_sym else (1, 0.0)
    if conf < _SYM_CONF_MIN:                      # low confidence → treat as asymmetric
        fold = 1
    display = symmetrize(pooled, fold) if (impose_symmetry and fold > 1) else pooled
    # per-particle 4×4: centre → rotate → translate  (re-zeroed input → fused frame)
    mats = []
    for i in range(n):
        Mc = np.eye(4); Mc[:3, 3] = -cents[i]
        mats.append(_matrix_4x4(G[i], T[i]) @ Mc)
    return {"points": display, "raw_points": pooled, "particle_transforms": mats,
            "symmetry": int(fold), "sym_conf": float(conf), "n_particles": n,
            "mode": "fusion", "iterations": int(n_iter), "sigma_nm": float(sigma_nm)}
