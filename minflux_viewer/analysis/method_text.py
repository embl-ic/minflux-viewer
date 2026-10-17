"""
minflux_viewer.analysis.method_text
====================================
Compile a publication-style *Methods* paragraph from selected **Log events**.

The user picks the log events relevant to a dataset/method (auto-tagged with the
active dataset at emit time); each event is mapped — via a small regex rule
registry — to a prose sentence, grouped by processing stage. Rules can read the
dataset's current metadata for richer detail than the log string carries. Events
that match no rule are kept verbatim so nothing is lost.

Citations are ``(reference_text, url|None)`` tuples: paper-backed methods carry a
verified DOI (rendered as a hyperlink in the HTML output, inline as plain text in
the text output); custom in-house methods (anisotropy, NPC) carry an inline
methodology note with ``url=None``. Two formatters share :func:`_collect` —
:func:`generate_method_text` (plain) and :func:`generate_method_html` (links).
Pure Python (no Qt) → unit-testable.
"""

from __future__ import annotations

import html as _html
import logging
from datetime import datetime

from minflux_viewer.analysis.method_citations import (
    _SIMUFLUX_DONUT_SIGMA_Q_NM as _SIMUFLUX_DONUT_SIGMA_Q_NM,
)
from minflux_viewer.analysis.method_citations import (
    ANISOTROPY_NOTE as ANISOTROPY_NOTE,
)
from minflux_viewer.analysis.method_citations import (
    CITE_ANISOTROPY as CITE_ANISOTROPY,
)
from minflux_viewer.analysis.method_citations import (
    CITE_CRLB as CITE_CRLB,
)
from minflux_viewer.analysis.method_citations import (
    CITE_CRLB_MARIN as CITE_CRLB_MARIN,
)
from minflux_viewer.analysis.method_citations import (
    CITE_DRIFT_TIME_WINDOWS as CITE_DRIFT_TIME_WINDOWS,
)
from minflux_viewer.analysis.method_citations import (
    CITE_FRC_BANTERLE as CITE_FRC_BANTERLE,
)
from minflux_viewer.analysis.method_citations import (
    CITE_FRC_NIEUWENHUIZEN as CITE_FRC_NIEUWENHUIZEN,
)
from minflux_viewer.analysis.method_citations import (
    CITE_MINFLUX_STABILISATION as CITE_MINFLUX_STABILISATION,
)
from minflux_viewer.analysis.method_citations import (
    CITE_NPC as CITE_NPC,
)
from minflux_viewer.analysis.method_citations import (
    CITE_SIMUFLUX as CITE_SIMUFLUX,
)
from minflux_viewer.analysis.method_citations import (
    CITE_STDDEV as CITE_STDDEV,
)
from minflux_viewer.analysis.method_citations import (
    NPC_NOTE as NPC_NOTE,
)
from minflux_viewer.analysis.method_rules import (
    _HLYB_DEFINITIONS as _HLYB_DEFINITIONS,
)
from minflux_viewer.analysis.method_rules import (
    _PAIR_FIT_HYPOTHESIS as _PAIR_FIT_HYPOTHESIS,
)
from minflux_viewer.analysis.method_rules import (
    _SEPARATION_BASIS as _SEPARATION_BASIS,
)
from minflux_viewer.analysis.method_rules import (
    RULES as RULES,
)
from minflux_viewer.analysis.method_rules import (
    _channel_desc as _channel_desc,
)
from minflux_viewer.analysis.method_rules import (
    _crlb_finite as _crlb_finite,
)
from minflux_viewer.analysis.method_rules import (
    _crlb_issues as _crlb_issues,
)
from minflux_viewer.analysis.method_rules import (
    _crlb_summary_value as _crlb_summary_value,
)
from minflux_viewer.analysis.method_rules import (
    _describe_alignment as _describe_alignment,
)
from minflux_viewer.analysis.method_rules import (
    _ds_for as _ds_for,
)
from minflux_viewer.analysis.method_rules import (
    _ds_name as _ds_name,
)
from minflux_viewer.analysis.method_rules import (
    _fmt_matrix as _fmt_matrix,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_audit_families as _hlyb_audit_families,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_axis_text as _hlyb_axis_text,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_limitations_block as _hlyb_limitations_block,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_parameter_block as _hlyb_parameter_block,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_segmentation_text as _hlyb_segmentation_text,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_setting as _hlyb_setting,
)
from minflux_viewer.analysis.method_rules import (
    _hlyb_structure_size_text as _hlyb_structure_size_text,
)
from minflux_viewer.analysis.method_rules import (
    _method_count as _method_count,
)
from minflux_viewer.analysis.method_rules import (
    _method_number as _method_number,
)
from minflux_viewer.analysis.method_rules import (
    _overlay_members as _overlay_members,
)
from minflux_viewer.analysis.method_rules import (
    _pair_fit_ranking as _pair_fit_ranking,
)
from minflux_viewer.analysis.method_rules import (
    _render_aggregation as _render_aggregation,
)
from minflux_viewer.analysis.method_rules import (
    _render_channel_separation as _render_channel_separation,
)
from minflux_viewer.analysis.method_rules import (
    _render_crlb as _render_crlb,
)
from minflux_viewer.analysis.method_rules import (
    _render_crlb_provenance as _render_crlb_provenance,
)
from minflux_viewer.analysis.method_rules import (
    _render_dcr as _render_dcr,
)
from minflux_viewer.analysis.method_rules import (
    _render_drift_beads as _render_drift_beads,
)
from minflux_viewer.analysis.method_rules import (
    _render_drift_old as _render_drift_old,
)
from minflux_viewer.analysis.method_rules import (
    _render_drift_windows as _render_drift_windows,
)
from minflux_viewer.analysis.method_rules import (
    _render_frc as _render_frc,
)
from minflux_viewer.analysis.method_rules import (
    _render_hlyb_pair_fit as _render_hlyb_pair_fit,
)
from minflux_viewer.analysis.method_rules import (
    _render_hlyb_staged_short_range as _render_hlyb_staged_short_range,
)
from minflux_viewer.analysis.method_rules import (
    _render_hlyb_template3d as _render_hlyb_template3d,
)
from minflux_viewer.analysis.method_rules import (
    _render_load as _render_load,
)
from minflux_viewer.analysis.method_rules import (
    _render_msr_overlay as _render_msr_overlay,
)
from minflux_viewer.analysis.method_rules import (
    _render_npc as _render_npc,
)
from minflux_viewer.analysis.method_rules import (
    _render_stddev as _render_stddev,
)
from minflux_viewer.analysis.method_rules import (
    _render_stddev_auto as _render_stddev_auto,
)
from minflux_viewer.analysis.method_rules import (
    _render_z_scaling_factor as _render_z_scaling_factor,
)
from minflux_viewer.analysis.method_rules import (
    _summary as _summary,
)
from minflux_viewer.analysis.plugin_method_rendering import (
    plugin_method_entry as plugin_method_entry,
)

STAGE_ORDER = ["load", "filter", "transform", "analysis", "segmentation", "export", "other"]
STAGE_TITLES = {
    "load": "Data loading",
    "filter": "Filtering",
    "transform": "Channel and dataset operations",
    "analysis": "Analysis",
    "segmentation": "Structure segmentation",
    "export": "Export",
    "other": "Other operations",
}

# --- citations: (reference text, url|None) ----------------------------------
# Paper-backed methods carry a verified DOI; custom methods carry an inline note.
# The CRLB here is the MINFLUX targeted-donut Cramér-Rao bound (Balzarotti 2017;
# background-aware closed form of Marin & Ries 2024) — NOT the camera/Gaussian
# bound (Mortensen 2010).
# Ostersehlt et al. 2022 is also the source of the time-window drift correction.
# The commercial instrument and its fiducial-based stabilisation.
#: The default σ_q of the CRLB: SimuFLUX's analytic 2-D donut, FWHM 310 nm.


# --- renderers: (match, event, state) -> (sentence, [citation, ...]) ---------


# Terms used in the result report, defined once so the generated Methods text is
# self-contained.  Wording is deliberately operational: each entry says what the
# quantity is computed from, not what it is believed to mean biologically.


#: (compiled pattern, stage, render(match, event, state) -> (sentence, [citations]))


def _guess_stage(message: str) -> str:
    low = message.lower()
    if "filter" in low:
        return "filter"
    if "saved" in low or "export" in low:
        return "export"
    if "z scaling factor" in low or "anisotropy" in low or "precision" in low:
        return "analysis"
    if "crop" in low or "duplicat" in low or "overlay" in low:
        return "transform"
    return "other"


def _collect(state, events):
    """Map *events* to ``(by_stage, citations)``; citations de-duped by text, order-preserving."""
    by_stage: dict[str, list[str]] = {s: [] for s in STAGE_ORDER}
    citations: list[tuple] = []
    seen: set[str] = set()

    for ev in events:
        msg = str(ev.get("message", "")).strip()
        if not msg:
            continue
        # A plugin that declares its own method text renders from that, ahead
        # of the built-in rules: it knows what it did and they do not.
        try:
            declared = plugin_method_entry(ev)
        except MemoryError:
            raise
        except Exception:                                   # plugin boundary
            logging.getLogger(__name__).exception("Plugin method declaration failed")
            by_stage["other"].append(f"Plugin method details unavailable: {msg}")
            continue
        if declared is not None:
            stage, sentence, cites = declared
            by_stage[stage].append(sentence)
            for cit in cites or ():
                if cit[0] and cit[0] not in seen:
                    seen.add(cit[0])
                    citations.append(cit)
            continue
        matched = False
        for pattern, stage, render in RULES:
            m = pattern.search(msg)
            if m is None:
                continue
            sentence, cites = render(m, ev, state)
            if sentence:
                by_stage[stage].append(sentence)
            for cit in cites or ():
                text = cit[0]
                if text and text not in seen:
                    seen.add(text)
                    citations.append(cit)
            matched = True
            break
        if not matched:
            by_stage[_guess_stage(msg)].append(msg.rstrip("."))
    return by_stage, citations


def _footer(version: str) -> str:
    ver = f" v{version}" if version else ""
    return (f"Generated by MINFLUX Data Viewer{ver} on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


def _format_text(by_stage, citations, version: str) -> str:
    lines = ["METHODS — DATA PROCESSING", "=" * 60, ""]
    if not any(by_stage.values()):
        lines.append("(No log events were selected.)")
        lines.append("")
    for stage in STAGE_ORDER:
        items = by_stage[stage]
        if not items:
            continue
        lines.append(STAGE_TITLES[stage] + ".")
        for item in items:
            lines.append(f"  {item}")
        lines.append("")
    if citations:
        lines.append("Method notes and references.")
        for text, url in citations:
            lines.append(f"  - {text}. {url}" if url else f"  - {text}")
        lines.append("")
    lines.append("=" * 60)
    lines.append(_footer(version))
    return "\n".join(lines)


def _format_html(by_stage, citations, version: str) -> str:
    esc = _html.escape
    parts = ['<div style="font-family: monospace; white-space: pre-wrap;">']
    parts.append("<b>METHODS — DATA PROCESSING</b>")
    parts.append("=" * 60)
    parts.append("")
    if not any(by_stage.values()):
        parts.append("(No log events were selected.)")
        parts.append("")
    for stage in STAGE_ORDER:
        items = by_stage[stage]
        if not items:
            continue
        parts.append("<b>" + esc(STAGE_TITLES[stage]) + ".</b>")
        for item in items:
            parts.append("  " + esc(item))
        parts.append("")
    if citations:
        parts.append("<b>Method notes and references.</b>")
        for text, url in citations:
            if url:
                parts.append(f'  - {esc(text)}. <a href="{esc(url)}">{esc(url)}</a>')
            else:
                parts.append(f"  - {esc(text)}")
        parts.append("")
    parts.append("=" * 60)
    parts.append(esc(_footer(version)))
    parts.append("</div>")
    return "<br>".join(parts)


def generate_method_text(state, events, *, version: str = "") -> str:
    """Build a plain-text Methods paragraph from the selected log *events*.

    *events* is a list of dicts with at least ``message`` and ``dataset_idx``.
    Citations appear as ``reference. DOI-URL`` (the URL is plain text so it
    survives copy/paste into any editor).
    """
    by_stage, citations = _collect(state, events)
    return _format_text(by_stage, citations, version)


def generate_method_html(state, events, *, version: str = "") -> str:
    """Like :func:`generate_method_text`, but citations are ``<a href>`` hyperlinks.

    The anchor text is the URL itself, so converting back to plain text (copy /
    save as ``.txt``) still preserves the link as readable text.
    """
    by_stage, citations = _collect(state, events)
    return _format_html(by_stage, citations, version)
