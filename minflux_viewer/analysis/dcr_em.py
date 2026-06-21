"""
minflux_viewer.analysis.dcr_em
==============================
One-dimensional **two-component Gaussian mixture** fit by Expectation-Maximization,
used to separate two fluorescent labels by their DCR (detector channel ratio).

The DCR histogram of an early (m2205 / legacy) two-colour MINFLUX dataset is
bimodal: one label sits at a lower DCR, the other higher. EM recovers the two
Gaussians (mean / sigma / weight); component **0 is always the lower-mean peak**
(rendered red), component **1 the higher** (green).

Pure NumPy — no SciPy/sklearn — so it stays dependency-light and unit-testable.
Posterior *responsibilities* give a principled, width/weight-aware channel
assignment (the Bayes-optimal :func:`decision_boundary`), strictly better than a
hand-picked DCR cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass
class GaussianMixture1D:
    """Result of a 1-D two-component Gaussian-mixture EM fit.

    ``mu``/``sigma``/``weight`` are length-2 arrays sorted so ``mu[0] < mu[1]``
    (component 0 = lower DCR = channel 1 = red; component 1 = higher = green).
    """

    mu: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray
    n: int
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def mu1(self) -> float:
        return float(self.mu[0])

    @property
    def mu2(self) -> float:
        return float(self.mu[1])

    @property
    def sigma1(self) -> float:
        return float(self.sigma[0])

    @property
    def sigma2(self) -> float:
        return float(self.sigma[1])

    @property
    def weight1(self) -> float:
        return float(self.weight[0])

    @property
    def weight2(self) -> float:
        return float(self.weight[1])

    def with_means(self, mu1: float, mu2: float) -> "GaussianMixture1D":
        """Return a copy with overridden means (for the editable centre fields),
        keeping sigmas/weights and re-sorting so component 0 stays the lower one."""
        mu = np.array([float(mu1), float(mu2)], dtype=float)
        order = np.argsort(mu)
        return GaussianMixture1D(
            mu=mu[order], sigma=self.sigma[order], weight=self.weight[order],
            n=self.n, log_likelihood=self.log_likelihood,
            n_iter=self.n_iter, converged=self.converged,
        )

    def with_params(self, mu1: float, mu2: float, sigma1: float, sigma2: float) -> "GaussianMixture1D":
        """Return a copy with user-overridden centres **and** widths for the two
        components (component 0 = red, component 1 = green). Order is preserved
        (no re-sort), so the red/green spin boxes map directly to the two
        components; weights are kept from the fit."""
        return GaussianMixture1D(
            mu=np.array([float(mu1), float(mu2)], dtype=float),
            sigma=np.array([max(float(sigma1), 1e-9), max(float(sigma2), 1e-9)], dtype=float),
            weight=self.weight.copy(),
            n=self.n, log_likelihood=self.log_likelihood,
            n_iter=self.n_iter, converged=self.converged,
        )


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * _SQRT_2PI)


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    return x[np.isfinite(x)]


def fit_two_component_gaussian(
    values,
    *,
    max_iter: int = 300,
    tol: float = 1e-7,
    n_init: int = 5,
    seed: int = 0,
) -> GaussianMixture1D:
    """Fit a two-component 1-D Gaussian mixture by EM (best of ``n_init`` starts).

    Raises ``ValueError`` if fewer than two finite values are supplied.
    """
    x = _clean(values)
    if x.size < 2:
        raise ValueError("need at least two finite values to fit two Gaussians")

    sd = float(np.std(x)) or 1.0
    rng = np.random.default_rng(seed)
    starts = [
        (np.percentile(x, 30.0), np.percentile(x, 70.0)),
        (np.percentile(x, 20.0), np.percentile(x, 80.0)),
        (np.percentile(x, 40.0), np.percentile(x, 60.0)),
    ]
    while len(starts) < n_init:
        starts.append(tuple(rng.choice(x, size=2, replace=x.size < 2)))

    best: tuple | None = None
    for m1, m2 in starts:
        mu = np.array([float(m1), float(m2)], dtype=float)
        sigma = np.array([sd / 2.0, sd / 2.0], dtype=float)
        w = np.array([0.5, 0.5], dtype=float)
        ll_old = -np.inf
        converged = False
        used = 0
        for used in range(1, max_iter + 1):
            p1 = w[0] * _gauss(x, mu[0], sigma[0])
            p2 = w[1] * _gauss(x, mu[1], sigma[1])
            denom = np.maximum(p1 + p2, 1e-300)
            r1 = p1 / denom
            r2 = p2 / denom
            n1 = float(r1.sum())
            n2 = float(r2.sum())
            if n1 < 1e-6 or n2 < 1e-6:          # a component collapsed
                break
            mu = np.array([(r1 * x).sum() / n1, (r2 * x).sum() / n2])
            sigma = np.array([
                np.sqrt((r1 * (x - mu[0]) ** 2).sum() / n1),
                np.sqrt((r2 * (x - mu[1]) ** 2).sum() / n2),
            ])
            sigma = np.maximum(sigma, 1e-9)
            w = np.array([n1, n2]) / x.size
            ll = float(np.log(denom).sum())
            if abs(ll - ll_old) < tol * max(1.0, abs(ll_old)):
                converged = True
                break
            ll_old = ll
        p1 = w[0] * _gauss(x, mu[0], sigma[0])
        p2 = w[1] * _gauss(x, mu[1], sigma[1])
        ll = float(np.log(np.maximum(p1 + p2, 1e-300)).sum())
        if best is None or ll > best[0]:
            best = (ll, mu.copy(), sigma.copy(), w.copy(), used, converged)

    ll, mu, sigma, w, used, converged = best
    order = np.argsort(mu)                       # component 0 = lower mean (red)
    return GaussianMixture1D(
        mu=mu[order], sigma=sigma[order], weight=w[order],
        n=int(x.size), log_likelihood=ll, n_iter=int(used), converged=bool(converged),
    )


def responsibilities(values, gmm: GaussianMixture1D) -> np.ndarray:
    """Posterior P(component | value): (N, 2) array, rows sum to 1."""
    x = np.asarray(values, dtype=float).ravel()
    p1 = gmm.weight[0] * _gauss(x, gmm.mu[0], gmm.sigma[0])
    p2 = gmm.weight[1] * _gauss(x, gmm.mu[1], gmm.sigma[1])
    denom = np.maximum(p1 + p2, 1e-300)
    return np.column_stack([p1 / denom, p2 / denom])


def assign(values, gmm: GaussianMixture1D, *, min_confidence: float = 0.5) -> np.ndarray:
    """Hard channel labels for *values*: 0 (red), 1 (green), or -1 (unassigned).

    A value is assigned to its most-likely component only when that component's
    responsibility is at least ``min_confidence``; otherwise it is -1 (the
    overlap). ``min_confidence == 0.5`` ⇒ pure Bayes boundary, nothing dropped.
    NaN values are always -1.
    """
    x = np.asarray(values, dtype=float).ravel()
    resp = responsibilities(x, gmm)
    label = np.argmax(resp, axis=1).astype(int)
    conf = resp[np.arange(resp.shape[0]), label]
    label[conf < float(min_confidence)] = -1
    label[~np.isfinite(x)] = -1
    return label


def assign_per_trace(
    values,
    tid,
    gmm: GaussianMixture1D,
    *,
    mode: str = "mean",
    min_confidence: float = 0.5,
    transform=None,
) -> np.ndarray:
    """Per-localization channel labels where **a whole trace shares one label**.

    Each trace's DCR is aggregated (``mode`` = ``"mean"`` or ``"median"``,
    NaN-ignoring), the trace is assigned from that single value via
    :func:`assign`, and the label is broadcast back to every localization of the
    trace. Returns an int array aligned to *values* (0 red, 1 green, -1 = overlap
    / unassigned). Falls back to per-localization :func:`assign` if *tid* does not
    align with *values*.
    """
    vals = np.asarray(values, dtype=float).ravel()
    tid = np.asarray(tid).ravel()
    if tid.shape[0] != vals.shape[0] or vals.size == 0:
        return assign(vals, gmm, min_confidence=min_confidence)
    uniq, inv = np.unique(tid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    s_inv = inv[order]
    s_vals = vals[order]
    bnd = np.flatnonzero(np.diff(s_inv)) + 1
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [s_inv.size]])
    fn = np.nanmedian if str(mode).lower().startswith("median") else np.nanmean
    trace_val = np.full(uniq.size, np.nan)
    with np.errstate(all="ignore"):
        for k, (a, b) in enumerate(zip(starts, ends)):
            grp = s_vals[a:b]
            grp = grp[np.isfinite(grp)]
            if grp.size:
                trace_val[k] = fn(grp)
    if transform is not None:                    # keep Log(data) consistent
        trace_val = np.asarray(transform(trace_val), dtype=float)
    trace_label = assign(trace_val, gmm, min_confidence=min_confidence)
    return trace_label[inv]


def decision_boundary(gmm: GaussianMixture1D) -> float:
    """The DCR where the two weighted Gaussians cross (Bayes boundary) — the
    point between the means with equal posterior. Falls back to the midpoint if
    no crossing lies between the means."""
    m0, m1 = float(gmm.mu[0]), float(gmm.mu[1])
    s0, s1 = float(gmm.sigma[0]), float(gmm.sigma[1])
    w0, w1 = float(gmm.weight[0]), float(gmm.weight[1])
    mid = 0.5 * (m0 + m1)
    if abs(s0 - s1) < 1e-12:                     # equal width ⇒ linear solution
        if abs(m1 - m0) < 1e-12:
            return mid
        return mid + (s0 * s0) / (m1 - m0) * np.log(w1 / w0)
    a = 1.0 / (2 * s0 * s0) - 1.0 / (2 * s1 * s1)
    b = -m0 / (s0 * s0) + m1 / (s1 * s1)
    c = (m0 * m0) / (2 * s0 * s0) - (m1 * m1) / (2 * s1 * s1) - np.log(w0 / s0) + np.log(w1 / s1)
    disc = b * b - 4 * a * c
    if disc < 0:
        return mid
    root = np.sqrt(disc)
    candidates = [(-b + root) / (2 * a), (-b - root) / (2 * a)]
    lo, hi = sorted((m0, m1))
    between = [r for r in candidates if lo <= r <= hi]
    if between:
        return float(between[0])
    return float(min(candidates, key=lambda r: abs(r - mid)))


def component_pdfs(x, gmm: GaussianMixture1D) -> tuple[np.ndarray, np.ndarray]:
    """Weighted per-component densities at *x* (each integrates to its weight).

    Multiply by ``n_points * bin_width`` to overlay on a count histogram.
    """
    x = np.asarray(x, dtype=float)
    c0 = gmm.weight[0] * _gauss(x, gmm.mu[0], gmm.sigma[0])
    c1 = gmm.weight[1] * _gauss(x, gmm.mu[1], gmm.sigma[1])
    return c0, c1


def goodness_of_fit(values, gmm: GaussianMixture1D, *, bins=64) -> dict:
    """RMSE / normalized RMSE of the mixture density vs the empirical histogram,
    plus the model BIC (lower = better). Returned as a dict for the legend."""
    x = _clean(values)
    if x.size < 2:
        return {"rmse": float("nan"), "nrmse": float("nan"), "bic": float("nan")}
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    c0, c1 = component_pdfs(centers, gmm)
    model = c0 + c1
    rmse = float(np.sqrt(np.mean((hist - model) ** 2)))
    peak = float(hist.max()) if hist.size and hist.max() > 0 else 1.0
    n_params = 5                                 # 2 mu, 2 sigma, 1 free weight
    bic = -2.0 * gmm.log_likelihood + n_params * np.log(max(gmm.n, 1))
    return {"rmse": rmse, "nrmse": rmse / peak, "bic": float(bic)}
