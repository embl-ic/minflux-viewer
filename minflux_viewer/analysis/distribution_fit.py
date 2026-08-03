"""
minflux_viewer.analysis.distribution_fit
=========================================
Generic **1-D mixture-distribution fitting** for attribute-based channel
separation. This is the attribute-agnostic core that the "Convert Dataset to
Multi-Channel Overlay" tool (and, as its first instance, "Separate Channel by
DCR") builds on: fit a mixture of a chosen distribution to *any* MINFLUX
attribute's histogram, turn the fitted components into channel ranges, and let
BIC pick the best distribution + component count automatically.

Backends (dependency-honest hybrid):

* **Gaussian** / **Log-Normal** — ``sklearn.mixture.GaussianMixture`` (the robust,
  professional EM; Log-Normal = a Gaussian mixture on ``log x``). sklearn's
  mixture module only supports Gaussian, hence:
* **Gamma** / **Poisson** — a small weighted-EM here (``scipy.stats`` component
  densities, moment/MLE M-step) since sklearn has no such mixture.
* **Uniform** — a data-adaptive equal-population partition (no peaks to fit); the
  useful "no dominant peak, just split the range by the data" case.

Every fit reports its parameters through one :class:`MixtureResult`, whose
``boundaries()`` (Bayes crossings between adjacent components) become the channel
cut points and whose ``assign()`` gives per-value channel labels. Log-likelihood
/ BIC / AIC are computed **uniformly** from this module's own ``pdf`` (not each
backend's) so :func:`auto_fit` compares candidates apples-to-apples.

Pure / Qt-free and unit-tested. NaNs and (for Gamma/Log-Normal) non-positive
values are dropped from the fit basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DISTRIBUTIONS = ["gaussian", "lognormal", "uniform", "gamma", "poisson"]
DISTRIBUTION_LABELS = {
    "gaussian": "Gaussian",
    "lognormal": "Log-Normal",
    "uniform": "Uniform",
    "gamma": "Gamma",
    "poisson": "Poisson",
}
# Distributions supported on strictly-positive data only.
_POSITIVE_ONLY = {"lognormal", "gamma"}
# Free parameters per component (for BIC/AIC penalty).
_PARAMS_PER_COMPONENT = {
    "gaussian": 2, "lognormal": 2, "uniform": 2, "gamma": 2, "poisson": 1,
}

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_DEFAULT_MAX_POINTS = 200_000


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
@dataclass
class MixtureComponent:
    """One fitted mixture component (ordered by ``location`` in the result)."""
    weight: float
    params: dict                       # distribution-specific (see _density)
    location: float                    # representative centre for ordering / labels


@dataclass
class MixtureResult:
    """A fitted 1-D mixture. Components are sorted by ``location`` ascending."""
    distribution: str
    components: list[MixtureComponent]
    n: int                             # finite fit points
    log_likelihood: float
    bic: float
    aic: float
    converged: bool
    domain: tuple[float, float]        # (min, max) of the fit basis
    detail: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.components], dtype=float)

    @property
    def locations(self) -> np.ndarray:
        return np.array([c.location for c in self.components], dtype=float)

    def component_pdfs(self, x) -> np.ndarray:
        """(n_components, len(x)) **weighted** component densities.

        Each row integrates to that component's weight, so their sum is the
        mixture pdf and ``row * n_points * bin_width`` overlays on a count
        histogram.
        """
        x = np.asarray(x, dtype=float).ravel()
        out = np.empty((self.n_components, x.size), dtype=float)
        for k, c in enumerate(self.components):
            out[k] = c.weight * _density(self.distribution, x, c.params)
        return out

    def pdf(self, x) -> np.ndarray:
        return self.component_pdfs(x).sum(axis=0)

    def responsibilities(self, x) -> np.ndarray:
        """(len(x), n_components) posterior P(component | value), rows sum to 1."""
        wpdf = self.component_pdfs(x)                       # (k, N)
        denom = np.maximum(wpdf.sum(axis=0), 1e-300)
        return (wpdf / denom).T

    def assign(self, x) -> np.ndarray:
        """Hard channel label per value: 0..n_components-1, or -1 for NaN."""
        x = np.asarray(x, dtype=float).ravel()
        resp = self.responsibilities(x)
        label = np.argmax(resp, axis=1).astype(int)
        label[~np.isfinite(x)] = -1
        return label

    def boundaries(self) -> np.ndarray:
        """Decision boundaries between adjacent components (length n_components-1).

        The Bayes crossing between component *k* and *k+1* (where their weighted
        densities are equal), found numerically between the two locations so it
        works for every distribution. These are the channel cut points.
        """
        locs = self.locations
        if self.n_components < 2:
            return np.empty(0, dtype=float)
        bounds = []
        for k in range(self.n_components - 1):
            a, b = float(locs[k]), float(locs[k + 1])
            if not (b > a):
                bounds.append(0.5 * (a + b))
                continue
            xs = np.linspace(a, b, 256)
            wa = self.components[k].weight * _density(self.distribution, xs, self.components[k].params)
            wb = self.components[k + 1].weight * _density(self.distribution, xs, self.components[k + 1].params)
            diff = wa - wb
            sign = np.signbit(diff)
            cross = np.flatnonzero(sign[:-1] != sign[1:])
            if cross.size:
                i = int(cross[0])
                # linear interpolation of the zero crossing
                d0, d1 = diff[i], diff[i + 1]
                t = 0.0 if d1 == d0 else d0 / (d0 - d1)
                bounds.append(float(xs[i] + t * (xs[i + 1] - xs[i])))
            else:
                bounds.append(0.5 * (a + b))
        return np.array(bounds, dtype=float)


# ---------------------------------------------------------------------------
# Component densities (unweighted)
# ---------------------------------------------------------------------------
def _density(distribution: str, x, params: dict) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if distribution == "gaussian":
        mu = float(params["mu"]); sigma = max(float(params["sigma"]), 1e-12)
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * _SQRT_2PI)
    if distribution == "lognormal":
        from scipy.stats import lognorm
        s = max(float(params["sigma_log"]), 1e-12)
        scale = float(params["scale"])                     # exp(mu_log)
        return np.where(x > 0.0, lognorm.pdf(x, s=s, scale=scale), 0.0)
    if distribution == "gamma":
        from scipy.stats import gamma
        a = max(float(params["shape"]), 1e-6)
        scale = max(float(params["scale"]), 1e-12)
        return np.where(x > 0.0, gamma.pdf(x, a=a, scale=scale), 0.0)
    if distribution == "poisson":
        from scipy.stats import poisson
        lam = max(float(params["lam"]), 1e-9)
        return poisson.pmf(np.rint(np.clip(x, 0, None)).astype(np.int64), lam)
    if distribution == "uniform":
        lo = float(params["lo"]); hi = float(params["hi"])
        width = max(hi - lo, 1e-12)
        return np.where((x >= lo) & (x <= hi), 1.0 / width, 0.0)
    raise ValueError(f"unknown distribution '{distribution}'")


def _location(distribution: str, params: dict) -> float:
    if distribution == "gaussian":
        return float(params["mu"])
    if distribution == "lognormal":
        return float(params["scale"])                      # median = exp(mu_log)
    if distribution == "gamma":
        return float(params["shape"]) * float(params["scale"])   # mean
    if distribution == "poisson":
        return float(params["lam"])
    if distribution == "uniform":
        return 0.5 * (float(params["lo"]) + float(params["hi"]))
    raise ValueError(f"unknown distribution '{distribution}'")


# ---------------------------------------------------------------------------
# Fit basis helpers
# ---------------------------------------------------------------------------
def _clean(values, distribution: str) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if distribution in _POSITIVE_ONLY:
        x = x[x > 0.0]
    return x


def _subsample(x: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points and x.size > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=int(max_points), replace=False)
        idx.sort()
        return x[idx]
    return x


def _finalize(distribution: str, comp_params: list[dict], weights, x_fit: np.ndarray,
              n_report: int, converged: bool) -> MixtureResult:
    """Build a MixtureResult with components sorted by location and BIC/AIC
    computed uniformly from this module's own pdf on the fit basis."""
    weights = np.asarray(weights, dtype=float)
    comps = [
        MixtureComponent(weight=float(w), params=p, location=_location(distribution, p))
        for w, p in zip(weights, comp_params)
    ]
    comps.sort(key=lambda c: c.location)
    k = len(comps)
    domain = (float(x_fit.min()), float(x_fit.max())) if x_fit.size else (0.0, 1.0)
    res = MixtureResult(
        distribution=distribution, components=comps, n=int(n_report),
        log_likelihood=0.0, bic=float("inf"), aic=float("inf"),
        converged=bool(converged), domain=domain,
    )
    # Uniform LL/BIC from this module's pdf on the fit basis (consistent across
    # distributions so auto_fit can compare them).
    dens = np.maximum(res.pdf(x_fit), 1e-300)
    ll = float(np.log(dens).sum())
    n_fit = int(x_fit.size)
    n_params = k * _PARAMS_PER_COMPONENT[distribution] + (k - 1)   # + free weights
    res.log_likelihood = ll
    res.bic = float(-2.0 * ll + n_params * np.log(max(n_fit, 1)))
    res.aic = float(-2.0 * ll + 2.0 * n_params)
    return res


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _fit_gaussian_family(x: np.ndarray, distribution: str, n_components: int,
                         *, seed: int, max_iter: int) -> MixtureResult:
    """Gaussian (or Log-Normal via log-space) mixture with sklearn."""
    from sklearn.mixture import GaussianMixture

    work = np.log(x) if distribution == "lognormal" else x
    gm = GaussianMixture(
        n_components=int(n_components), covariance_type="full",
        n_init=2, max_iter=int(max_iter), random_state=int(seed), reg_covar=1e-9,
    )
    gm.fit(work.reshape(-1, 1))
    mus = gm.means_.ravel()
    sig = np.sqrt(np.clip(gm.covariances_.reshape(-1), 1e-18, None))
    weights = gm.weights_
    comp_params: list[dict] = []
    for mu, s in zip(mus, sig):
        if distribution == "lognormal":
            comp_params.append({"mu_log": float(mu), "sigma_log": float(s),
                                "scale": float(np.exp(mu))})
        else:
            comp_params.append({"mu": float(mu), "sigma": float(s)})
    return _finalize(distribution, comp_params, weights, x, x.size, bool(gm.converged_))


def _gamma_mstep(x: np.ndarray, w: np.ndarray) -> dict:
    """Weighted MLE (Minka approximation) for a Gamma component."""
    W = float(w.sum())
    if W <= 0:
        return {"shape": 1.0, "scale": 1.0}
    mean = float((w * x).sum() / W)
    mean_log = float((w * np.log(x)).sum() / W)
    s = np.log(max(mean, 1e-12)) - mean_log
    s = max(s, 1e-6)
    shape = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    shape = float(max(shape, 1e-3))
    return {"shape": shape, "scale": float(mean / shape)}


def _poisson_mstep(x: np.ndarray, w: np.ndarray) -> dict:
    W = float(w.sum())
    lam = float((w * x).sum() / W) if W > 0 else float(x.mean())
    return {"lam": max(lam, 1e-9)}


def _fit_em_generic(x: np.ndarray, distribution: str, n_components: int,
                    *, seed: int, max_iter: int, tol: float) -> MixtureResult:
    """Weighted-EM mixture for scipy-backed distributions (gamma, poisson)."""
    mstep = {"gamma": _gamma_mstep, "poisson": _poisson_mstep}[distribution]
    k = int(n_components)
    # Quantile-seeded initial components.
    qs = np.quantile(x, np.linspace(0.0, 1.0, k + 2)[1:-1]) if k > 1 else np.array([np.median(x)])
    comp_params = []
    for q in np.atleast_1d(qs):
        sub = x[np.abs(x - q) <= (np.std(x) or 1.0)]
        sub = sub if sub.size >= 2 else x
        comp_params.append(mstep(sub, np.ones(sub.size)))
    weights = np.full(k, 1.0 / k)

    converged = False
    ll_old = -np.inf
    for _ in range(int(max_iter)):
        dens = np.stack([weights[j] * _density(distribution, x, comp_params[j]) for j in range(k)])
        denom = np.maximum(dens.sum(axis=0), 1e-300)
        resp = dens / denom
        Nk = resp.sum(axis=1)
        if np.any(Nk < 1e-6):
            break
        weights = Nk / x.size
        comp_params = [mstep(x, resp[j]) for j in range(k)]
        ll = float(np.log(denom).sum())
        if abs(ll - ll_old) < tol * max(1.0, abs(ll_old)):
            converged = True
            break
        ll_old = ll
    return _finalize(distribution, comp_params, weights, x, x.size, converged)


def _fit_uniform(x: np.ndarray, n_components: int) -> MixtureResult:
    """Equal-population partition into uniform components (data-adaptive split).

    No peaks to fit: split the range at empirical quantiles so each component
    holds an equal share of the data, each a uniform over its [lo, hi]. Useful
    when the attribute has no dominant modes and you just want the range cut by
    where the data actually lies.
    """
    k = int(n_components)
    edges = np.quantile(x, np.linspace(0.0, 1.0, k + 1))
    edges = np.asarray(edges, dtype=float)
    # Guard against zero-width bins (repeated quantiles).
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    comp_params = [{"lo": float(edges[i]), "hi": float(edges[i + 1])} for i in range(k)]
    weights = np.full(k, 1.0 / k)
    return _finalize("uniform", comp_params, weights, x, x.size, True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fit_mixture(
    values,
    distribution: str = "gaussian",
    n_components: int = 2,
    *,
    seed: int = 0,
    max_iter: int = 200,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> MixtureResult:
    """Fit an ``n_components`` mixture of ``distribution`` to *values*.

    Large inputs are subsampled to ``max_points`` for the fit (deterministic,
    seeded) so the tool stays responsive on millions of localizations. Raises
    ``ValueError`` for an unknown distribution or too few finite (positive, for
    Log-Normal/Gamma) values.
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution '{distribution}'")
    k = int(n_components)
    if k < 1:
        raise ValueError("n_components must be >= 1")
    x = _clean(values, distribution)
    if x.size < max(2, k):
        raise ValueError(f"need at least {max(2, k)} usable values to fit {k} component(s)")
    x = _subsample(x, max_points, seed)

    if distribution in ("gaussian", "lognormal"):
        return _fit_gaussian_family(x, distribution, k, seed=seed, max_iter=max_iter)
    if distribution == "uniform":
        return _fit_uniform(x, k)
    return _fit_em_generic(x, distribution, k, seed=seed, max_iter=max_iter, tol=1e-6)


def auto_fit(
    values,
    *,
    distributions: list[str] | None = None,
    max_components: int = 3,
    hard_cap: int = 7,
    seed: int = 0,
    max_points: int = _DEFAULT_MAX_POINTS,
    improvement: float = 0.01,
) -> MixtureResult:
    """Best (distribution, n_components) by BIC — the "Auto" button.

    Searches every distribution in *distributions* over 1..``max_components``
    components and keeps the lowest-BIC fit. If the winner sits at
    ``max_components`` (the data may want more), it keeps adding components to
    that distribution up to ``hard_cap`` while BIC keeps improving by more than
    ``improvement`` (relative). Restricting the primary search to 3 and only
    escalating on a still-improving fit keeps it fast, per the design intent.

    All candidates are fit on the **same** seeded subsample so their BICs are
    comparable. Returns the winning :class:`MixtureResult`.
    """
    dists = list(distributions) if distributions else list(DISTRIBUTIONS)
    best: MixtureResult | None = None
    for dist in dists:
        base = _clean(values, dist)
        if base.size < 2:
            continue
        xf = _subsample(base, max_points, seed)
        for k in range(1, int(max_components) + 1):
            if xf.size < max(2, k):
                break
            try:
                res = _dispatch_on_fitbasis(xf, dist, k, seed=seed)
            except Exception:
                continue
            if best is None or res.bic < best.bic:
                best = res
    if best is None:
        raise ValueError("no distribution could be fit to the data")

    # Escalate the winner's component count only while BIC keeps improving.
    if best.n_components == int(max_components):
        base = _clean(values, best.distribution)
        xf = _subsample(base, max_points, seed)
        k = int(max_components)
        while k < int(hard_cap) and xf.size >= (k + 1):
            k += 1
            try:
                cand = _dispatch_on_fitbasis(xf, best.distribution, k, seed=seed)
            except Exception:
                break
            if cand.bic < best.bic - improvement * abs(best.bic):
                best = cand
            else:
                break
    return best


def _dispatch_on_fitbasis(x_fit: np.ndarray, distribution: str, k: int, *, seed: int) -> MixtureResult:
    """fit_mixture on an already-cleaned, already-subsampled basis (auto_fit)."""
    if distribution in ("gaussian", "lognormal"):
        return _fit_gaussian_family(x_fit, distribution, k, seed=seed, max_iter=200)
    if distribution == "uniform":
        return _fit_uniform(x_fit, k)
    return _fit_em_generic(x_fit, distribution, k, seed=seed, max_iter=200, tol=1e-6)
