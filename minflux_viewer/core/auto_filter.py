"""Data-driven filter bounds: the rules behind the Filter dialog's *Auto*.

A rule names an attribute, a method and the method's parameters. The method
looks only at the values it is handed and returns the bounds it suggests,
together with a one-line account of how they were obtained, so a suggestion is
always reported and reproducible rather than applied silently.

Three methods are provided. Each is attribute-agnostic, so a rule may apply any
of them to any numeric attribute:

``single_emitter`` -- *single-emitter frequency cut.*  The photon emission rate
    (``efo``) of one active emitter forms the main population; two emitters
    active at once emit at twice that rate. Brightness varies multiplicatively,
    so the rates are modelled in log10 space as two Gaussians fitted by EM:
    single emitters ``N(mu, s1)`` and double emitters ``N(mu + log10 2, s2)`` --
    the second held at exactly twice the rate of the first. The upper bound is
    the rate above which a value is more likely double than single. The model is
    fitted to the values themselves, not to a histogram, so no bin width enters;
    ``efo`` is quantized (``eco`` / dwell, steps of 1/dwell), so the values are
    first spread uniformly within one quantum (seeded, reproducible).
    ``f1 = 10**mu`` is fitted unless the rule states it. Measured on the sample
    data (2026-10-21): rate histograms rarely show a resolved second peak, and a
    valley search either finds nothing or a dip inside the single population,
    while this model put the m2410 bounds at 1.55·f1 and 1.90·f1 (5.3 % and
    7.7 % removed) -- the second where the instrument's own invalid rate rises
    to ~2.5x its baseline. The premise is that single emitters dominate: with no
    double population (share < 1 %), no crossing, or a fitted double share above
    35 % (legacy and tracking samples fitted 51-74 %, where the crossing would
    remove 40-60 %), the bound falls back to ``fallback * f1`` (default 2; 0 =
    no bound) and the note says why. A removal above 10 % is flagged.

``robust_spread`` -- *robust spread bound.*  ``median +/- k * sigma``, with
    ``sigma = 1.4826 * MAD`` (the median absolute deviation scaled to the
    standard deviation of a normal population), following Leys et al., J. Exp.
    Soc. Psychol. 49:764 (2013), who recommend ``k = 2.5``. Upper, lower or
    both sides.

``percentile`` -- *percentile range*, with an optional absolute floor (e.g. at
    least 10 localizations per trace).

The caller decides which values a rule sees (per localization, or one value
per trace for a trace-wise attribute such as ``siz``) and in which order the
rules are chained; this module is pure and Qt-free.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

METHOD_SINGLE_EMITTER = "single_emitter"
METHOD_ROBUST_SPREAD = "robust_spread"
METHOD_PERCENTILE = "percentile"

#: Method key -> short label shown in Preferences and in reports.
METHOD_LABELS: dict[str, str] = {
    METHOD_SINGLE_EMITTER: "single-emitter frequency cut",
    METHOD_ROBUST_SPREAD: "robust spread (median ± k·σ_MAD)",
    METHOD_PERCENTILE: "percentile range",
}

#: Parameters each method accepts, with their defaults.
METHOD_DEFAULTS: dict[str, dict] = {
    METHOD_SINGLE_EMITTER: {"expected": 0.0, "fallback": 2.0},   # 0 = fitted / no bound
    METHOD_ROBUST_SPREAD: {"k": 2.5, "side": "upper"},
    METHOD_PERCENTILE: {"lower_pct": 0.0, "upper_pct": 99.0, "min_value": None},
}

ROBUST_SIDES = ("upper", "lower", "both")

#: The scale that makes the MAD a consistent estimator of a normal sigma,
#: 1 / Phi^-1(3/4).
MAD_TO_SIGMA = 1.482602218505602

#: Fewest finite values a rule is evaluated on; below it nothing is suggested.
MIN_VALUES = 20

#: Method keys an earlier build stored, read as their current method.
_METHOD_ALIASES = {"emitter_valley": METHOD_SINGLE_EMITTER}

#: log10 of the rate ratio between two emitters and one.
LOG10_TWO = math.log10(2.0)

#: Below this fitted double-emitter share there is no double population to cut.
MIN_DOUBLE_SHARE = 0.01

#: Above this fitted double-emitter share single emitters no longer clearly
#: dominate (legacy samples fitted 51-54 %, a tracking sample 74 %), so the
#: model's crossing is not used and the fallback applies, with the reason.
DOUBLE_SHARE_LIMIT = 0.35

#: The EM fit uses at most this many values (a seeded subsample; the bound moved
#: by 1-4 % against 200 000 on the samples, at 30x the speed); the removed
#: fraction is always reported over all of them.
MIXTURE_MAX_VALUES = 50_000

#: A rate cut removing more than this fraction is flagged in its note (measured:
#: 5.3 % and 7.7 % on two m2410 samples, 12 % on a 3-D one, 26-37 % for the legacy
#: and tracking fallbacks).
LARGE_REMOVAL = 0.10

DEFAULT_AUTO_RULES: list[dict] = [
    {"enabled": True, "attribute": "efo", "method": METHOD_SINGLE_EMITTER,
     "params": {"expected": 0.0, "fallback": 2.0}},
    {"enabled": True, "attribute": "cfr", "method": METHOD_ROBUST_SPREAD,
     "params": {"k": 2.5, "side": "upper"}},
    {"enabled": True, "attribute": "siz", "method": METHOD_PERCENTILE,
     "params": {"lower_pct": 0.0, "upper_pct": 99.0, "min_value": 10.0}},
]


def default_auto_rules() -> list[dict]:
    """A fresh copy of :data:`DEFAULT_AUTO_RULES`."""
    return copy.deepcopy(DEFAULT_AUTO_RULES)


@dataclass(frozen=True)
class AutoBound:
    """What one rule suggests for one attribute.

    ``lo`` / ``hi`` of ``None`` mean "no bound on that side" (the caller keeps
    the data range there). ``ok`` is False when nothing is suggested at all;
    ``note`` then says why.
    """

    attribute: str
    method: str
    lo: float | None
    hi: float | None
    note: str
    n_values: int
    ok: bool = True


def normalize_rule(rule: Mapping) -> dict:
    """A validated, complete copy of *rule*; raises ``ValueError`` if unusable."""
    if not isinstance(rule, Mapping):
        raise ValueError("an auto-filter rule must be a mapping")
    attribute = str(rule.get("attribute") or "").strip()
    if not attribute:
        raise ValueError("an auto-filter rule needs an attribute")
    method = str(rule.get("method") or "").strip()
    method = _METHOD_ALIASES.get(method, method)
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"unknown auto-filter method {method!r}")
    params = dict(METHOD_DEFAULTS[method])
    given = rule.get("params") or {}
    if not isinstance(given, Mapping):
        raise ValueError(f"the parameters of the {attribute} rule must be a mapping")
    for key in params:
        if key in given:
            params[key] = given[key]
    if method == METHOD_SINGLE_EMITTER:
        params["expected"] = _non_negative(params["expected"], "expected rate")
        params["fallback"] = _non_negative(params["fallback"], "fallback factor")
    elif method == METHOD_ROBUST_SPREAD:
        params["k"] = _positive(params["k"], "k")
        side = str(params["side"]).strip().lower()
        if side not in ROBUST_SIDES:
            raise ValueError(f"robust spread side must be one of {ROBUST_SIDES}, not {side!r}")
        params["side"] = side
    else:
        lower = _finite(params["lower_pct"], "lower percentile")
        upper = _finite(params["upper_pct"], "upper percentile")
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
        params["lower_pct"], params["upper_pct"] = lower, upper
        floor = params["min_value"]
        params["min_value"] = None if floor is None else _finite(floor, "minimum value")
    return {"enabled": bool(rule.get("enabled", True)), "attribute": attribute,
            "method": method, "params": params}


def normalize_rules(rules: Iterable) -> tuple[list[dict], list[str]]:
    """``(usable rules, one reason per rule that was skipped)``."""
    usable: list[dict] = []
    skipped: list[str] = []
    for position, rule in enumerate(rules or (), start=1):
        try:
            usable.append(normalize_rule(rule))
        except ValueError as exc:
            skipped.append(f"rule {position} skipped: {exc}")
    return usable, skipped


def describe_rule(rule: Mapping) -> str:
    """One line naming the attribute, method and parameters of *rule*."""
    r = normalize_rule(rule)
    p = r["params"]
    if r["method"] == METHOD_SINGLE_EMITTER:
        f1 = "fitted" if p["expected"] <= 0 else format_value(p["expected"])
        detail = f"single + double-emitter model, f1 {f1}; without doubles " + (
            f"{format_value(p['fallback'])}·f1" if p["fallback"] > 0 else "no bound")
    elif r["method"] == METHOD_ROBUST_SPREAD:
        detail = f"k = {format_value(p['k'])}, {p['side']} side" + ("s" if p["side"] == "both" else "")
    else:
        detail = f"P{format_value(p['lower_pct'])}–P{format_value(p['upper_pct'])}"
        if p["min_value"] is not None:
            detail += f", at least {format_value(p['min_value'])}"
    return f"{r['attribute']}: {METHOD_LABELS[r['method']]} ({detail})"


def suggest_bounds(values, rule: Mapping) -> AutoBound:
    """Apply *rule* to *values* (non-finite values are ignored)."""
    r = normalize_rule(rule)
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    attribute, method, params = r["attribute"], r["method"], r["params"]
    if x.size < MIN_VALUES:
        return AutoBound(attribute, method, None, None,
                         f"only {x.size} finite value(s); at least {MIN_VALUES} are needed",
                         int(x.size), ok=False)
    if method == METHOD_SINGLE_EMITTER:
        return _single_emitter(attribute, x, params["expected"], params["fallback"])
    if method == METHOD_ROBUST_SPREAD:
        return _robust_spread(attribute, x, params["k"], params["side"])
    return _percentile(attribute, x, params["lower_pct"], params["upper_pct"],
                       params["min_value"])


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def _robust_spread(attribute: str, x: np.ndarray, k: float, side: str) -> AutoBound:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = MAD_TO_SIGMA * mad
    if not sigma > 0.0:
        return AutoBound(attribute, METHOD_ROBUST_SPREAD, None, None,
                         "the median absolute deviation is zero, so no spread can be defined",
                         int(x.size), ok=False)
    lo = med - k * sigma if side in ("lower", "both") else None
    hi = med + k * sigma if side in ("upper", "both") else None
    sign = {"upper": "+", "lower": "−", "both": "±"}[side]
    note = (f"median {format_value(med)} {sign} {format_value(k)} × σ_MAD {format_value(sigma)}; "
            f"σ_MAD = 1.4826 × MAD; {x.size:,} values")
    return AutoBound(attribute, METHOD_ROBUST_SPREAD, lo, hi, note, int(x.size))


def _percentile(attribute: str, x: np.ndarray, lower_pct: float, upper_pct: float,
                floor: float | None) -> AutoBound:
    lo = float(np.percentile(x, lower_pct)) if lower_pct > 0.0 else None
    hi = float(np.percentile(x, upper_pct)) if upper_pct < 100.0 else None
    parts = []
    if lo is not None:
        parts.append(f"P{format_value(lower_pct)} = {format_value(lo)}")
    if floor is not None and (lo is None or floor > lo):
        lo = float(floor)
        parts.append(f"at least {format_value(floor)}")
    if hi is not None:
        parts.append(f"P{format_value(upper_pct)} = {format_value(hi)}")
    if lo is not None and hi is not None and lo > hi:
        return AutoBound(attribute, METHOD_PERCENTILE, None, None,
                         f"the floor {format_value(lo)} lies above P{format_value(upper_pct)} = {format_value(hi)}; "
                         "nothing would pass", int(x.size), ok=False)
    if lo is None and hi is None:
        return AutoBound(attribute, METHOD_PERCENTILE, None, None,
                         "P0–P100 with no floor bounds nothing", int(x.size), ok=False)
    return AutoBound(attribute, METHOD_PERCENTILE, lo, hi,
                     ", ".join(parts) + f"; {x.size:,} values", int(x.size))


def _single_emitter(attribute: str, x: np.ndarray, expected: float,
                    fallback: float = 0.0) -> AutoBound:
    x = x[x > 0.0]
    if x.size < MIN_VALUES:
        return AutoBound(attribute, METHOD_SINGLE_EMITTER, None, None,
                         "too few positive rates to fit", int(x.size), ok=False)
    y = np.log10(dequantize(x))
    if y.size > MIXTURE_MAX_VALUES:
        y = y[np.random.default_rng(1).choice(y.size, MIXTURE_MAX_VALUES, replace=False)]
    if expected > 0.0:
        fit = fit_single_double(y, mu=math.log10(expected))
    else:
        # EM finds a local optimum: started between two large populations it
        # settles on one component covering both. Start low as well and keep the
        # most likely fit.
        starts = np.percentile(y, [20.0, 35.0, 50.0])
        fit = max((fit_single_double(y, start=float(m0)) for m0 in starts),
                  key=lambda f: f.log_likelihood)
    f1 = 10.0 ** fit.mu
    source = "stated" if expected > 0.0 else "fitted"
    model = (f"single + double-emitter model in log space: f1 = {format_value(f1)} ({source}), "
             f"double share {100.0 * fit.double_share:.1f} %")
    cut_log = None
    if fit.double_share < MIN_DOUBLE_SHARE:
        lacking = f"{model}; no double-emitter population to separate"
    elif fit.double_share > DOUBLE_SHARE_LIMIT:
        lacking = f"{model}; single emitters do not clearly dominate, so the model is not used"
    else:
        cut_log = fit.boundary()
        lacking = f"{model}; the components do not cross"
    if cut_log is None:
        if not fallback > 0.0:
            return AutoBound(attribute, METHOD_SINGLE_EMITTER, None, None,
                             f"{lacking}; no bound suggested", int(x.size), ok=False)
        cut = float(fallback) * f1
        return AutoBound(attribute, METHOD_SINGLE_EMITTER, None, cut,
                         f"{lacking}; fallback bound {format_value(fallback)}·f1 = "
                         f"{format_value(cut)}; {_removal_text(float(np.mean(x > cut)), x.size)}",
                         int(x.size))
    cut = float(10.0 ** cut_log)
    note = (f"{model}; more likely double above {format_value(cut)} = "
            f"{cut / f1:.2f}·f1; {_removal_text(float(np.mean(x > cut)), x.size)}")
    return AutoBound(attribute, METHOD_SINGLE_EMITTER, None, cut, note, int(x.size))


@dataclass(frozen=True)
class SingleDoubleFit:
    """log10-space mixture ``(1-w) N(mu, s1) + w N(mu + log10 2, s2)``."""

    mu: float
    s1: float
    s2: float
    double_share: float
    log_likelihood: float

    def boundary(self) -> float | None:
        """First log10 value above ``mu`` where a value is at least as likely
        double as single; ``None`` if that never happens within ``mu + 1``."""
        ys = np.linspace(self.mu, self.mu + 1.0, 4001)
        single = (1.0 - self.double_share) * _normal_pdf(ys, self.mu, self.s1)
        double = self.double_share * _normal_pdf(ys, self.mu + LOG10_TWO, self.s2)
        crossing = np.flatnonzero(double >= single)
        return float(ys[crossing[0]]) if crossing.size else None


def fit_single_double(y: np.ndarray, *, mu: float | None = None, start: float | None = None,
                      max_iter: int = 300, tol: float = 1e-7) -> SingleDoubleFit:
    """EM fit of the single/double-emitter model to log10 rates *y*.

    The location is shared -- the double component sits exactly ``log10 2``
    above the single one -- and is fitted (from *start*, default the median)
    unless *mu* fixes it. ``s2`` is kept at least ``0.3 * s1`` so the double
    component cannot collapse onto a spike.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    fixed = mu is not None
    if mu is not None:
        m = float(mu)
    else:
        m = float(np.median(y)) if start is None else float(start)
    # Start narrow, from the values near the start alone: a width taken from the
    # whole distribution lets one broad component swallow both populations.
    near = y[np.abs(y - m) < 0.5 * LOG10_TWO]
    q75, q25 = np.percentile(near if near.size >= 10 else y, [75.0, 25.0])
    s1 = max(float(q75 - q25) / 1.349, 1e-3)
    s2, w = 1.5 * s1, 0.1
    previous = -np.inf
    log_l = -np.inf
    for _ in range(max_iter):
        p1 = (1.0 - w) * _normal_pdf(y, m, s1)
        p2 = w * _normal_pdf(y, m + LOG10_TWO, s2)
        total = p1 + p2 + 1e-300
        log_l = float(np.sum(np.log(total)))
        r2 = p2 / total
        r1 = 1.0 - r2
        w = float(np.clip(r2.mean(), 1e-6, 1.0 - 1e-6))
        if not fixed:
            a1, a2 = r1 / s1 ** 2, r2 / s2 ** 2
            m = float((np.sum(a1 * y) + np.sum(a2 * (y - LOG10_TWO))) / (a1.sum() + a2.sum()))
        s1 = max(float(np.sqrt(np.sum(r1 * (y - m) ** 2) / max(r1.sum(), 1e-12))), 1e-4)
        s2 = max(float(np.sqrt(np.sum(r2 * (y - m - LOG10_TWO) ** 2) / max(r2.sum(), 1e-12))),
                 0.3 * s1)
        if abs(log_l - previous) <= tol * abs(log_l):
            break
        previous = log_l
    return SingleDoubleFit(mu=m, s1=s1, s2=s2, double_share=w, log_likelihood=log_l)


def dequantize(x: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Spread quantized values uniformly within one quantum (seeded).

    ``efo`` is ``eco`` divided by the dwell, so it moves in steps of 1/dwell; a
    fit or histogram on the raw values sees a comb. The quantum is the median
    spacing of the distinct central values; already-continuous values get a
    negligible spread. Values stay positive.
    """
    x = np.asarray(x, dtype=float)
    central = x[(x > np.percentile(x, 5.0)) & (x < np.percentile(x, 95.0))]
    spacing = np.diff(np.unique(central))
    spacing = spacing[spacing > 0.0]
    if spacing.size == 0:
        return x
    quantum = float(np.median(spacing))
    spread = x + np.random.default_rng(seed).uniform(-0.5 * quantum, 0.5 * quantum, x.size)
    return np.where(spread > 0.0, spread, x)


def _normal_pdf(y, mean: float, sd: float):
    z = (np.asarray(y, dtype=float) - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def _removal_text(fraction: float, n: int) -> str:
    text = f"removes {100.0 * fraction:.1f} % of {n:,} values"
    if fraction > LARGE_REMOVAL:
        text += " — a large fraction: check the histogram before keeping it"
    return text


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _finite(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite")
    return number


def _non_negative(value, what: str) -> float:
    number = _finite(value, what)
    if number < 0.0:
        raise ValueError(f"{what} must not be negative")
    return number


def _positive(value, what: str) -> float:
    number = _finite(value, what)
    if number <= 0.0:
        raise ValueError(f"{what} must be positive")
    return number


def format_value(value: float) -> str:
    """Compact number: integers without decimals, else 3 significant digits."""
    number = float(value)
    if number == int(number) and abs(number) < 1e9:
        return f"{int(number):,}"
    if abs(number) >= 100:
        return f"{number:,.0f}"
    return f"{number:.3g}"
