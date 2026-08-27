"""Stacked iterations are drawn on ``jet`` sampled at ten points, blue to red.

⚠ **Ten stops, not eleven.** Consumers index this table by
``round(k/(n-1) * (len-1))``; with eleven stops over a ten-iteration MINFLUX
sequence that expression skips stop 5 and shifts iterations 5..9 up by one, so
the drawn series stopped matching the legend swatch. Ten makes it the identity
for the common case — that is the invariant this file defends.

The Attribute Plot and the Histogram must agree exactly: they draw the same
series side by side.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.colors import DEFAULT_COLOR_PREFS, component_colors
from minflux_viewer.core.app_state import default_prefs

pytest.importorskip("PyQt6")

BLUE = (0, 0, 127, 255)                # jet's low end
RED = (127, 0, 0, 255)                 # jet's high end


def _colors(prefs=None):
    return list(component_colors(prefs or default_prefs(),
                                 "functions", "Iteration series").values())


def test_palette_is_ten_stops_of_jet_running_blue_to_red():
    from minflux_viewer.colormaps import make_colormap

    colors = _colors()
    assert len(colors) == 10
    assert colors[0] == BLUE
    assert colors[-1] == RED
    # Exactly ``jet`` sampled at ten points — not a hand-tuned lookalike.
    sampled = make_colormap("jet").map(np.linspace(0.0, 1.0, 10), mode="byte")
    assert [tuple(int(c) for c in row) for row in sampled] == colors


def test_ten_stops_map_one_to_one_onto_a_ten_iteration_sequence():
    """The bug: eleven stops skipped one and mis-coloured the last iterations."""
    from minflux_viewer.ui.attribute_window import _iter_color as attribute_color
    from minflux_viewer.ui.histogram_window import _iter_color as histogram_color

    prefs = default_prefs()
    palette = _colors(prefs)
    for k in range(10):
        expected = palette[k]                      # stop k, no skipping
        assert attribute_color(prefs, k, 10) == expected
        # Same result, different argument order — the two must never diverge.
        assert histogram_color(k, prefs, 10) == expected
    assert attribute_color(prefs, 0, 10) == BLUE
    assert attribute_color(prefs, 9, 10) == RED


def test_the_legend_swatch_is_the_colour_the_series_is_drawn_in():
    """Legend and plot read the same table, so they cannot disagree."""
    from minflux_viewer.ui.attribute_window import _iter_color as attribute_color

    prefs = default_prefs()
    drawn = [attribute_color(prefs, k, 10) for k in range(10)]
    assert len(set(drawn)) == 10               # no two iterations share a colour
    assert drawn == _colors(prefs)


def test_a_short_sequence_still_spans_the_whole_ramp():
    """Three iterations must not sit in one corner of the ramp."""
    from minflux_viewer.ui.attribute_window import _iter_color as attribute_color

    prefs = default_prefs()
    assert attribute_color(prefs, 0, 4) == BLUE
    assert attribute_color(prefs, 3, 4) == RED
    assert attribute_color(prefs, 0, 1) == BLUE     # a single series is defined


def test_migration_replaces_the_shipped_ramps_but_keeps_a_customised_palette():
    from minflux_viewer.core.app_state import _migrate_prefs

    superseded = (
        {   # v044: the 11-stop saturated rainbow
            "1st": [0, 0, 255, 255], "2nd": [0, 102, 255, 255],
            "3rd": [0, 204, 255, 255], "4th": [0, 255, 204, 255],
            "5th": [0, 255, 102, 255], "6th": [0, 255, 0, 255],
            "7th": [102, 255, 0, 255], "8th": [204, 255, 0, 255],
            "9th": [255, 204, 0, 255], "10th": [255, 102, 0, 255],
            "11th": [255, 0, 0, 255],
        },
        {   # v042: viridis sampled at 10 points
            "1st": [68, 1, 84, 255], "2nd": [71, 39, 119, 255],
            "3rd": [62, 74, 136, 255], "4th": [49, 104, 142, 255],
            "5th": [37, 130, 142, 255], "6th": [32, 157, 136, 255],
            "7th": [53, 183, 121, 255], "8th": [108, 204, 89, 255],
            "9th": [180, 221, 44, 255], "10th": [253, 231, 37, 255],
        },
    )
    for old in superseded:
        saved = _migrate_prefs(
            {"colors": {"functions": {"Iteration series": dict(old)}}}
        )["colors"]["functions"]["Iteration series"]
        assert saved == DEFAULT_COLOR_PREFS["functions"]["Iteration series"]
        # The surplus 11th stop must be gone, not merged back in from a save.
        assert "11th" not in saved

    custom = {"1st": [1, 2, 3, 255], "2nd": [4, 5, 6, 255]}
    kept = _migrate_prefs(
        {"colors": {"functions": {"Iteration series": dict(custom)}}}
    )["colors"]["functions"]["Iteration series"]
    for key, value in custom.items():
        assert list(kept[key]) == value


def test_a_fresh_install_does_not_run_the_migration_over_the_defaults():
    """``DEFAULT_COLOR_PREFS`` is authoritative for a new installation."""
    prefs = default_prefs()
    assert prefs["_migrations"]["v045_jet_iteration_colours"] is True
    assert (prefs["colors"]["functions"]["Iteration series"]
            == DEFAULT_COLOR_PREFS["functions"]["Iteration series"])
    assert "11th" not in prefs["colors"]["functions"]["Iteration series"]


def test_a_retired_stop_is_dropped_even_from_a_customised_palette():
    """The 11th row survived my first attempt, and this is why.

    ``v045`` only rewrites a palette that still matches a set this application
    shipped, so a user's own colours are never discarded — but ``_merge`` keeps
    a saved key the defaults do not have, so a *customised* eleven-stop palette
    kept the surplus row forever. The stop count is part of the contract, not a
    preference, so it is pruned unconditionally.
    """
    from minflux_viewer.core.app_state import _merge, _migrate_prefs, DEFAULT_PREFS

    rainbow = {
        "1st": [0, 0, 255, 255], "2nd": [0, 102, 255, 255],
        "3rd": [0, 204, 255, 255], "4th": [0, 255, 204, 255],
        "5th": [0, 255, 102, 255], "6th": [0, 255, 0, 255],
        "7th": [102, 255, 0, 255], "8th": [204, 255, 0, 255],
        "9th": [255, 204, 0, 255], "10th": [255, 102, 0, 255],
        "11th": [255, 0, 0, 255],
    }
    customised = dict(rainbow, **{"3rd": [1, 2, 3, 255]})
    saved = {
        "colors": {"functions": {"Iteration series": customised}},
        # Already ran with both earlier migrations, so only the prune is left.
        "_migrations": {"v044_rainbow_iteration_colours": True,
                        "v045_jet_iteration_colours": True},
    }
    series = _migrate_prefs(_merge(saved, DEFAULT_PREFS))[
        "colors"]["functions"]["Iteration series"]
    assert "11th" not in series
    assert len(series) == 10
    assert list(series["3rd"]) == [1, 2, 3, 255]        # the user's colour stays


def test_the_prune_runs_after_the_replacement_not_before():
    """Order matters: pruning first would stop v045 recognising the rainbow."""
    from minflux_viewer.core.app_state import _merge, _migrate_prefs, DEFAULT_PREFS

    rainbow = {
        "1st": [0, 0, 255, 255], "2nd": [0, 102, 255, 255],
        "3rd": [0, 204, 255, 255], "4th": [0, 255, 204, 255],
        "5th": [0, 255, 102, 255], "6th": [0, 255, 0, 255],
        "7th": [102, 255, 0, 255], "8th": [204, 255, 0, 255],
        "9th": [255, 204, 0, 255], "10th": [255, 102, 0, 255],
        "11th": [255, 0, 0, 255],
    }
    series = _migrate_prefs(_merge(
        {"colors": {"functions": {"Iteration series": dict(rainbow)}}},
        DEFAULT_PREFS))["colors"]["functions"]["Iteration series"]
    assert series == DEFAULT_COLOR_PREFS["functions"]["Iteration series"]


def test_normalisation_alone_also_refuses_to_carry_a_retired_stop():
    """Belt and braces: the one-shot migration is not the only guard."""
    from minflux_viewer.colors import normalize_color_preferences

    saved = {"functions": {"Iteration series": {
        "1st": [9, 9, 9, 255], "11th": [255, 0, 0, 255], "12th": [0, 0, 0, 255]}}}
    series = normalize_color_preferences(saved)["functions"]["Iteration series"]
    assert set(series) == set(DEFAULT_COLOR_PREFS["functions"]["Iteration series"])
    assert list(series["1st"]) == [9, 9, 9, 255]        # a real stop is untouched


def test_the_colour_dialog_lists_exactly_the_stops_that_exist(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.global_color_dialog import GlobalColorDialog

    dialog = GlobalColorDialog(AppState())
    qtbot.addWidget(dialog)
    dialog._function_combo.setCurrentText("Iteration series")
    rows = list(dialog._draft["functions"]["Iteration series"])
    assert rows == list(DEFAULT_COLOR_PREFS["functions"]["Iteration series"])
    assert "11th" not in rows
