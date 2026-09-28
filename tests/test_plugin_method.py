"""The ``[method]`` block: declared once, filled from what a run recorded."""

from __future__ import annotations

import pytest

from minflux_viewer.plugins.method import (
    MISSING_TEMPLATE,
    MethodError,
    MethodField,
    MethodSpec,
    format_value,
    parse_method_spec,
    render_method_text,
)

_RAW = {
    "title": "Staged short-range pair analysis",
    "description": (
        "Pair distances between inferred label sites are compared with a "
        "conditional null over the band {short_range_lo_nm} to "
        "{short_range_hi_nm}."
    ),
    "workflow": [
        "Traces with at least {min_loc_per_trace} localization(s) contributed a centroid.",
        "Repeats were consolidated within {site_merge_nm}.",
    ],
    "limitations": "Distances are population descriptors, not assigned pairs.",
    "inputs": [
        {"name": "min_loc_per_trace", "label": "minimum localizations per trace",
         "description": "Below this a trace contributes no centroid."},
        {"name": "site_merge_nm", "label": "same-site diameter", "unit": "nm",
         "description": "Hard diameter for consolidating repeated visits."},
        {"name": "short_range_lo_nm", "unit": "nm"},
        {"name": "short_range_hi_nm", "unit": "nm"},
    ],
    "outputs": [
        {"name": "band_ratio", "label": "observed/null ratio",
         "description": "Observed band pairs over the null expectation."},
    ],
}


def test_a_declared_method_renders_with_the_values_a_run_recorded():
    spec = parse_method_spec(_RAW)
    text, missing = render_method_text(spec, {
        "min_loc_per_trace": 10,
        "site_merge_nm": 4.0,
        "short_range_lo_nm": 8.0,
        "short_range_hi_nm": 25.0,
        "band_ratio": 2.041,
    }, dataset="sample")

    assert missing == []
    assert "Staged short-range pair analysis was applied to dataset 'sample'." in text
    # Units come from the declaration, so the template carries only the slot.
    assert "band 8 nm to 25 nm" in text
    assert "at least 10 localization(s)" in text
    assert "consolidated within 4 nm" in text
    # Declared parameters and outputs are reported with their own descriptions.
    assert "same-site diameter: 4 nm — Hard diameter" in text
    assert "observed/null ratio: 2.041" in text
    assert "Limitations." in text
    assert text.index("Workflow.") < text.index("Parameters.") < text.index("Results.")


def test_a_value_the_run_did_not_record_is_named_not_dropped():
    """A paragraph that quietly loses a number it promised is the worse bug."""
    spec = parse_method_spec(_RAW)
    text, missing = render_method_text(spec, {"min_loc_per_trace": 10})

    # Every unfilled slot AND every declared field with no value, so the
    # rendered block and the reported list cannot disagree with each other.
    assert missing == [
        "band_ratio", "short_range_hi_nm", "short_range_lo_nm", "site_merge_nm"]
    assert MISSING_TEMPLATE.format(name="site_merge_nm") in text
    # A declared output with no recorded value is listed as missing, so the
    # block cannot silently shrink away from the method it describes.
    assert MISSING_TEMPLATE.format(name="band_ratio") in text


def test_templates_may_contain_braces_and_arbitrary_recorded_values():
    """Substitution is a slot scan, not str.format."""
    spec = parse_method_spec({
        "description": "A literal {} brace, a }stray{ one, and {value}.",
        "inputs": [{"name": "value"}],
    })
    text, missing = render_method_text(spec, {"value": "kept"})

    assert missing == []
    assert "A literal {} brace, a }stray{ one, and kept." in text

    # No attribute access is reachable from a template.
    spec2 = parse_method_spec({"description": "{value}", "inputs": []})
    text2, _ = render_method_text(spec2, {"value": object()})
    assert "<" in text2 or "object" in text2


def test_values_are_formatted_for_prose():
    nm = MethodField("x", unit="nm")
    assert format_value(4.0, nm) == "4 nm"
    assert format_value(4.25, nm) == "4.25 nm"
    assert format_value(20627153) == "20,627,153"
    assert format_value(True) == "yes"
    assert format_value(False) == "no"
    assert format_value(None) == "not set"
    assert format_value([8.0, 25.0], nm) == "8, 25 nm"
    assert format_value("roi2d") == "roi2d"


def test_a_malformed_block_is_refused_with_a_reason_for_its_author():
    with pytest.raises(MethodError, match="must be a table"):
        parse_method_spec([1, 2])
    with pytest.raises(MethodError, match="needs a 'name'"):
        parse_method_spec({"description": "d", "inputs": [{"label": "no name"}]})
    with pytest.raises(MethodError, match="plain identifier"):
        parse_method_spec({"description": "d", "inputs": [{"name": "not a name"}]})
    with pytest.raises(MethodError, match="twice"):
        parse_method_spec({"description": "d",
                           "inputs": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(MethodError, match="empty step"):
        parse_method_spec({"workflow": ["ok", "  "]})
    with pytest.raises(MethodError, match="description.*or a 'workflow'"):
        parse_method_spec({"title": "nothing to say"})
    # Absent is fine; an empty table is the same as absent.
    assert parse_method_spec(None) is None
    assert parse_method_spec({}) is None


def test_slots_are_discoverable_so_a_declaration_can_be_checked():
    spec = parse_method_spec(_RAW)
    assert spec.slots() == {
        "short_range_lo_nm", "short_range_hi_nm",
        "min_loc_per_trace", "site_merge_nm",
    }
    # Every slot the templates use is declared, which is what makes the units
    # and descriptions reachable.
    assert spec.slots() <= set(spec.fields)


def test_an_input_wins_over_an_output_of_the_same_name():
    spec = MethodSpec(
        description="{x}",
        inputs=(MethodField("x", unit="nm"),),
        outputs=(MethodField("x", unit="s"),),
    )
    text, _ = render_method_text(spec, {"x": 3})
    assert "3 nm" in text
