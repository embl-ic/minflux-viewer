"""The rules behind the Filter dialog's *Auto* button (``core/auto_filter.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core import auto_filter as af


def _two_emitter_rates(seed: int = 0, n: int = 20000, double_fraction: float = 0.12):
    rng = np.random.default_rng(seed)
    n2 = int(n * double_fraction)
    single = rng.normal(60_000.0, 7_000.0, n - n2)
    double = rng.normal(120_000.0, 10_000.0, n2)
    return np.concatenate([single, double])


RULE = {"attribute": "efo", "method": "single_emitter"}


def test_single_emitter_model_separates_one_and_two_emitter_populations():
    rates = _two_emitter_rates()
    bound = af.suggest_bounds(rates, RULE)
    assert bound.ok, bound.note
    assert bound.lo is None                       # only an upper bound
    # Where 0.88·N(60k, 7k) and 0.12·N(120k, 10k) become equally likely (log space).
    assert 85_000.0 < bound.hi < 100_000.0
    assert "(fitted), double share 12.0 %" in bound.note
    assert "more likely double above" in bound.note
    removed = float(np.mean(rates > bound.hi))
    assert 0.10 < removed < 0.14


def test_a_stated_single_emitter_rate_fixes_f1():
    bound = af.suggest_bounds(_two_emitter_rates(seed=1),
                              {**RULE, "params": {"expected": 60_000.0}})
    assert bound.ok
    assert 85_000.0 < bound.hi < 100_000.0
    assert "f1 = 60,000 (stated)" in bound.note


def test_quantized_rates_give_the_same_bound():
    """efo is eco/dwell: steps of 1/dwell. Dequantizing makes the fit see the
    distribution, not the comb -- and no histogram bin width is involved."""
    rates = _two_emitter_rates(seed=3)
    smooth = af.suggest_bounds(rates, RULE)
    comb = af.suggest_bounds(np.round(rates / 1083.0) * 1083.0, RULE)
    assert comb.hi == pytest.approx(smooth.hi, rel=0.01)


def test_dequantize_spreads_within_one_step_and_keeps_continuous_values():
    steps = np.repeat(np.arange(20, 120) * 1000.0, 50)
    spread = af.dequantize(steps)
    assert np.all(np.abs(spread - steps) <= 500.0 + 1e-9)
    assert np.unique(spread).size > 10 * np.unique(steps).size
    np.testing.assert_array_equal(af.dequantize(steps), spread)          # seeded
    continuous = np.random.default_rng(4).normal(50_000.0, 5_000.0, 5000)
    assert np.max(np.abs(af.dequantize(continuous) - continuous)) < 50.0


def test_several_starts_avoid_one_component_swallowing_both():
    """Started between two equal populations, a single EM run settles on one
    broad component; the multi-start fit finds the 50 % double share -- which
    then fails the premise and falls back."""
    rates = _two_emitter_rates(double_fraction=0.5)
    bound = af.suggest_bounds(rates, RULE)
    assert "double share 49.9 %" in bound.note or "double share 50.0 %" in bound.note
    assert "single emitters do not clearly dominate" in bound.note
    assert "fallback bound 2·f1" in bound.note


def _one_population(seed: int = 2):
    rng = np.random.default_rng(seed)
    return 10.0 ** rng.normal(np.log10(60_000.0), 0.05, 20000)


def test_without_doubles_the_stated_fallback_multiple_is_used_and_said():
    bound = af.suggest_bounds(_one_population(), RULE)
    assert bound.ok
    f1 = float(bound.note.split("f1 = ")[1].split(" ")[0].replace(",", ""))
    assert bound.hi == pytest.approx(2.0 * f1, rel=1e-4)       # f1 is printed rounded
    assert "no double-emitter population" in bound.note
    assert "fallback bound 2·f1" in bound.note


def test_fallback_zero_suggests_nothing_without_doubles():
    bound = af.suggest_bounds(_one_population(), {**RULE, "params": {"fallback": 0.0}})
    assert not bound.ok
    assert bound.lo is None and bound.hi is None
    assert "no bound suggested" in bound.note


def test_the_model_wins_over_the_fallback_when_doubles_are_present():
    bound = af.suggest_bounds(_two_emitter_rates(seed=9), {**RULE, "params": {"fallback": 1.2}})
    assert 85_000.0 < bound.hi < 100_000.0
    assert "fallback" not in bound.note


def test_a_cut_removing_a_large_fraction_is_flagged():
    rng = np.random.default_rng(11)
    broad = rng.lognormal(np.log(30_000.0), 0.9, 20000)        # P90 ≈ 3-4·mode
    bound = af.suggest_bounds(broad, RULE)
    assert bound.ok
    assert float(np.mean(broad > bound.hi)) > af.LARGE_REMOVAL
    assert "a large fraction" in bound.note
    assert "a large fraction" not in af.suggest_bounds(_one_population(), RULE).note


def test_the_old_method_key_still_reads():
    usable, skipped = af.normalize_rules([{"attribute": "efo", "method": "emitter_valley"}])
    assert not skipped
    assert usable[0]["method"] == af.METHOD_SINGLE_EMITTER


def test_robust_spread_is_median_plus_k_scaled_mad():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0] * 10)
    med = 3.0
    sigma = af.MAD_TO_SIGMA * 1.0                 # |x - 3| -> median 1
    bound = af.suggest_bounds(values, {"attribute": "cfr", "method": "robust_spread",
                                       "params": {"k": 2.0, "side": "upper"}})
    assert bound.ok
    assert bound.lo is None
    assert bound.hi == pytest.approx(med + 2.0 * sigma)
    both = af.suggest_bounds(values, {"attribute": "cfr", "method": "robust_spread",
                                      "params": {"k": 2.0, "side": "both"}})
    assert both.lo == pytest.approx(med - 2.0 * sigma)
    assert both.hi == pytest.approx(med + 2.0 * sigma)


def test_robust_spread_reports_a_zero_spread_instead_of_bounding_everything():
    bound = af.suggest_bounds(np.full(50, 0.3), {"attribute": "cfr", "method": "robust_spread"})
    assert not bound.ok
    assert "zero" in bound.note


def test_percentile_with_a_floor_on_trace_length():
    lengths = np.arange(1, 201, dtype=float)       # one value per trace
    bound = af.suggest_bounds(lengths, {"attribute": "siz", "method": "percentile",
                                        "params": {"lower_pct": 0.0, "upper_pct": 99.0,
                                                   "min_value": 10.0}})
    assert bound.ok
    assert bound.lo == 10.0
    assert bound.hi == pytest.approx(np.percentile(lengths, 99.0))
    assert "at least 10" in bound.note


def test_percentile_floor_above_the_upper_percentile_suggests_nothing():
    bound = af.suggest_bounds(np.arange(1, 41, dtype=float),
                              {"attribute": "siz", "method": "percentile",
                               "params": {"upper_pct": 50.0, "min_value": 30.0}})
    assert not bound.ok


def test_too_few_values_are_reported_not_bounded():
    bound = af.suggest_bounds([1.0, 2.0, np.nan], {"attribute": "cfr", "method": "robust_spread"})
    assert not bound.ok
    assert bound.n_values == 2


def test_rules_are_validated_and_skips_are_explained():
    rules = [
        {"attribute": "efo", "method": "single_emitter"},
        {"attribute": "cfr", "method": "nonsense"},
        {"attribute": "", "method": "percentile"},
        {"attribute": "dcr", "method": "percentile", "params": {"lower_pct": 50, "upper_pct": 10}},
    ]
    usable, skipped = af.normalize_rules(rules)
    assert [r["attribute"] for r in usable] == ["efo"]
    assert usable[0]["params"] == {"expected": 0.0, "fallback": 2.0}
    assert len(skipped) == 3
    assert all(text.startswith("rule ") for text in skipped)


def test_default_rules_are_valid_and_independent_copies():
    rules = af.default_auto_rules()
    usable, skipped = af.normalize_rules(rules)
    assert not skipped
    assert [r["attribute"] for r in usable] == ["efo", "cfr", "siz"]
    rules[0]["attribute"] = "changed"
    assert af.DEFAULT_AUTO_RULES[0]["attribute"] == "efo"


def test_describe_rule_names_attribute_method_and_parameters():
    text = af.describe_rule({"attribute": "cfr", "method": "robust_spread",
                             "params": {"k": 2.5, "side": "upper"}})
    assert text.startswith("cfr: robust spread")
    assert "k = 2.5" in text


def test_user_facing_text_does_not_name_another_software_package():
    texts = [af.describe_rule(rule) for rule in af.default_auto_rules()]
    texts += list(af.METHOD_LABELS.values()) + [af.__doc__ or ""]
    assert not any("pyminflux" in text.lower() for text in texts)
