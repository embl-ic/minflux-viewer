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
import re
from datetime import datetime

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
CITE_STDDEV = (
    "Ostersehlt et al., Nat. Methods 19:1072 (2022)",
    "https://doi.org/10.1038/s41592-022-01577-1",
)
# The CRLB here is the MINFLUX targeted-donut Cramér-Rao bound (Balzarotti 2017;
# background-aware closed form of Marin & Ries 2024) — NOT the camera/Gaussian
# bound (Mortensen 2010).
CITE_CRLB = (
    "Balzarotti et al., Science 355:606 (2017)",
    "https://doi.org/10.1126/science.aak9913",
)
CITE_CRLB_MARIN = (
    "Marin & Ries, arXiv:2410.12427 (2024)",
    "https://arxiv.org/abs/2410.12427",
)
CITE_SIMUFLUX = (
    "Marin & Ries, Nat. Commun. 17:246 (2026)",
    "https://doi.org/10.1038/s41467-025-66952-w",
)
CITE_FRC_BANTERLE = (
    "Banterle et al., J. Struct. Biol. 183:363 (2013)",
    "https://doi.org/10.1016/j.jsb.2013.05.004",
)
CITE_FRC_NIEUWENHUIZEN = (
    "Nieuwenhuizen et al., Nat. Methods 10:557 (2013)",
    "https://doi.org/10.1038/nmeth.2448",
)

ANISOTROPY_NOTE = (
    "Anisotropy / RIMF estimation (custom, MINFLUX Data Viewer): the refractive-index-"
    "mismatch factor is estimated from single-molecule traces by Gaussian fits to "
    "log-distance histograms of each localization's offset from its trace centroid "
    "(lateral vs. axial extent); it is applied as a Z-scaling view, never baked into "
    "the raw coordinates."
)
NPC_NOTE = (
    "NPC ring-convolution segmentation (custom, MINFLUX Data Viewer): NPC centres are "
    "detected by convolving the 2-D localization histogram with a normalized donut "
    "kernel exp(-|x^2 + y^2 - (d/2)^2| / (4*rim^2)) that peaks at the ring radius, "
    "followed by local-maximum peak finding and a ring 'support score' (annulus angular "
    "coverage and radial fit) acceptance filter."
)
CITE_ANISOTROPY = (ANISOTROPY_NOTE, None)
CITE_NPC = (NPC_NOTE, None)


def _ds_for(state, ev):
    idx = ev.get("dataset_idx")
    datasets = getattr(state, "datasets", [])
    if isinstance(idx, int) and 0 <= idx < len(datasets):
        return datasets[idx]
    name = ev.get("dataset_name")
    for ds in datasets:
        if getattr(ds, "name", None) == name:
            return ds
    return None


def _ds_name(state, ev):
    ds = _ds_for(state, ev)
    if ds is not None and getattr(ds, "name", None):
        return ds.name
    return ev.get("dataset_name") or "the dataset"


# --- renderers: (match, event, state) -> (sentence, [citation, ...]) ---------

def _render_load(m, ev, state):
    name = m.group("name")
    ds = _ds_for(state, ev)
    if ds is None:
        return f"A MINFLUX dataset '{name}' was loaded into the MINFLUX Data Viewer.", []
    md = getattr(ds, "metadata", {})
    container = md.get("source_format")
    ver = md.get("source_version", "an unknown version")
    ver_str = f"{ver} ({container})" if container else str(ver)
    n_dim = int(ds.prop.num_dim)
    n_itr = md.get("raw_num_itr", 1)
    n_traces = int(ds.prop.num_traces)
    valid = md.get("valid_num_loc", ds.prop.num_loc)
    load_mode = md.get("iteration_load_mode", "last")
    validity = "all (valid and invalid)" if md.get("includes_invalid") else "only valid"
    try:
        valid_str = f"{int(valid):,}"
    except Exception:
        valid_str = str(valid)
    return (
        f"A MINFLUX dataset '{name}' was loaded into the MINFLUX Data Viewer. The data was "
        f"recognized as version {ver_str}, containing {valid_str} valid localizations across "
        f"{n_itr} iteration(s) and {n_traces} trace(s), in {n_dim} dimension(s). For analysis, "
        f"{validity} localizations from the '{load_mode}' iteration were used."
    ), []


def _overlay_members(state, ev):
    """``[(idx, ds), ...]`` for every channel in the tagged dataset's overlay."""
    idx = ev.get("dataset_idx")
    if isinstance(idx, int):
        try:
            from ..core.overlay import overlay_members
            members = overlay_members(state, idx)
            if members:
                return members
        except Exception:
            pass
    ds = _ds_for(state, ev)
    return [(idx if isinstance(idx, int) else 0, ds)] if ds is not None else []


def _channel_desc(ds):
    md = getattr(ds, "metadata", {}) or {}
    name = md.get("msr_dataset_name") or getattr(ds, "name", None) or "channel"
    prop = getattr(ds, "prop", None)
    valid = md.get("valid_num_loc", getattr(prop, "num_loc", 0))
    try:
        valid_s = f"{int(valid):,}"
    except Exception:
        valid_s = str(valid)
    ndim = int(getattr(prop, "num_dim", 2) or 2)
    ntr = int(getattr(prop, "num_traces", 0) or 0)
    ver = md.get("source_version")
    ver_s = f", {ver}" if ver else ""
    return f"'{name}' ({valid_s} valid localizations, {ndim}D, {ntr} trace(s){ver_s})"


def _fmt_matrix(mat):
    try:
        rows = ["[" + ", ".join(f"{float(v):.4g}" for v in r) + "]" for r in mat]
        return "[" + ", ".join(rows) + "]"
    except Exception:
        return str(mat)


def _describe_alignment(mover, ref, t):
    """Human-readable translation + XY rotation (+ matrix) from a transform dict."""
    import math
    tr = t.get("translation_nm") or [0.0, 0.0]
    tx = float(tr[0]) if len(tr) > 0 else 0.0
    ty = float(tr[1]) if len(tr) > 1 else 0.0
    tz = float(t.get("z_translation_nm", tr[2] if len(tr) > 2 else 0.0))
    rot = t.get("rotation_2x2")
    angle = 0.0
    if rot:
        try:
            angle = math.degrees(math.atan2(float(rot[1][0]), float(rot[0][0])))
        except Exception:
            angle = 0.0
    nbeads = int(t.get("matched_bead_count", 0) or 0)
    rmse = float(t.get("rmse_xy_nm", 0.0) or 0.0)
    parts = [f"Channel {mover} was aligned to {ref}"]
    if nbeads:
        parts.append(f" using {nbeads} matched bead(s) (XY RMSE {rmse:.2f} nm)")
    parts.append(f": translated by {tx:+.2f} nm in X, {ty:+.2f} nm in Y, and {tz:+.2f} nm in Z")
    if abs(angle) >= 1e-3:
        direction = "counterclockwise" if angle > 0 else "clockwise"
        parts.append(f", with a {abs(angle):.2f}° {direction} rotation in the XY plane")
    else:
        parts.append(", with no rotation in the XY plane")
    mat = t.get("matrix_4x4")
    if mat is not None:
        parts.append(f"; transform matrix {_fmt_matrix(mat)}")
    parts.append(".")
    return "".join(parts)


def _render_msr_overlay(m, ev, state):
    file = m.group("file")
    members = _overlay_members(state, ev)
    tagged = _ds_for(state, ev)
    md = getattr(tagged, "metadata", {}) or {}
    ref = md.get("overlay_reference")
    mode = md.get("overlay_alignment_mode", "none")
    excluded = md.get("overlay_bead_excluded") or []

    n = len(members) or m.group("n")
    chans = ", ".join(_channel_desc(ds) for _i, ds in members) if members else m.group("n") + " channel(s)"
    sentences = [
        f"A multi-channel MINFLUX overlay was loaded from the .msr file '{file}' via the MSR "
        f"reader, comprising {n} channel(s): {chans}."
    ]
    if ref:
        sentences.append(f"Channel '{ref}' served as the alignment reference.")
    if mode and mode != "none":
        if mode == "mbm info":
            if excluded:
                sentences.append(
                    f"Channels were aligned to the reference using MBM bead fiducials; all "
                    f"available beads were used except {len(excluded)} excluded by the user "
                    f"(bead IDs {list(excluded)}).")
            else:
                sentences.append(
                    "Channels were aligned to the reference using MBM bead fiducials (all "
                    "available beads were used).")
        else:
            sentences.append(f"Channels were aligned to the reference by the '{mode}' method.")
    else:
        sentences.append("No inter-channel alignment was applied.")
    for _i, ds in members:
        dmd = getattr(ds, "metadata", {}) or {}
        t = dmd.get("overlay_transform")
        if not t:
            continue
        rc, mc = t.get("reference_channel"), t.get("moving_channel")
        if rc is not None and rc == mc:
            continue   # reference identity — no transform to describe
        mover = dmd.get("msr_dataset_name") or getattr(ds, "name", "channel")
        sentences.append(_describe_alignment(f"'{mover}'", f"'{ref or rc}'", t))
    return " ".join(sentences), []


def _render_rimf(m, ev, state):
    name, val = m.group("name"), m.group("value")
    note = (m.group("note") or "").lower()
    if "2d" in note or "2-d" in note:
        return f"'{name}' is two-dimensional, so no Z (RIMF) correction was applied (RIMF = {val}).", []
    if "fixed" in note:
        return (f"A fixed refractive-index-mismatch factor (RIMF) of {val} was applied to '{name}' "
                f"as a Z-scaling correction."), []
    return (
        f"The anisotropy of '{name}' was estimated to be approximately {val} using a custom "
        f"log-distance Gaussian-fit method (see method note); the resulting refractive-index-"
        f"mismatch factor (RIMF) was applied as a Z-scaling correction."
    ), [CITE_ANISOTROPY]


def _render_npc(m, ev, state):
    return (
        f"Nuclear pore complex (NPC) structures in '{m.group('name')}' were segmented by 2-D "
        f"ring-kernel convolution: the XY localizations were histogrammed into {m.group('pixel')} nm "
        f"pixels and convolved with a donut kernel matched to an NPC diameter of {m.group('diam')} nm "
        f"and rim width of {m.group('rim')} nm; local maxima with a ring support score above "
        f"{m.group('support')} were accepted. {m.group('n')} NPC(s) were detected and marked with "
        f"rectangle regions of interest."
    ), [CITE_NPC]


def _render_dcr(m, ev, state):
    return (
        f"Dataset '{m.group('name')}' was separated into {m.group('n')} channel(s) by DCR (detector "
        f"channel ratio): a two-component Gaussian mixture was fitted to the DCR distribution by "
        f"expectation-maximization, and each trace was assigned to a channel by its mean DCR."
    ), []


def _render_stddev(m, ev, state):
    name = _ds_name(state, ev)
    return (
        f"The localization precision of '{name}' was estimated from the standard deviation of "
        f"localizations within each trace (traces with at least 5 localizations): the n-weighted "
        f"combined lateral precision (σ_r) was {m.group('sr')} nm and the axial precision (σ_z) was "
        f"{m.group('sz')} nm, over {m.group('used')} of {m.group('total')} traces."
    ), [CITE_STDDEV]


def _render_stddev_auto(m, ev, state):
    name = m.group("name") or _ds_name(state, ev)
    return (
        f"The localization precision of '{name}' was estimated as the per-trace standard deviation "
        f"of localizations (StdDev per trace); the median precision was {m.group('med')} nm."
    ), [CITE_STDDEV]


def _render_crlb(m, ev, state):
    name = _ds_name(state, ev)
    s = (
        f"The theoretical localization precision of '{name}' was computed as the MINFLUX "
        f"Cramér-Rao lower bound: the median background-limited lateral precision (σ_xy) was "
        f"{m.group('sxy')} nm ({m.group('ideal')} nm in the ideal, background-free limit)"
    )
    if m.group("sz"):
        s += f", with an axial precision (σ_z) of {m.group('sz')} nm"
    s += (f", for a targeting-pattern diameter L = {m.group('L')} nm and a median of "
          f"{m.group('N')} detected photons.")
    cites = [CITE_CRLB, CITE_CRLB_MARIN]
    if m.groupdict().get("fl"):
        s += (f" Relative to the measured per-trace spread (σ_r = {m.group('mr')} nm), an "
              f"excess error of σ_fl = {m.group('fl')} nm beyond the photon-limited bound "
              "was identified (STD² = σ_fl² + σ_CRB²), attributable to fluorophore "
              "flickering, drift, vibration or misalignment rather than photon statistics.")
        cites.append(CITE_SIMUFLUX)
    return s, cites


def _render_frc(m, ev, state):
    name = _ds_name(state, ev)
    return (
        f"The image resolution of '{name}' was estimated by Fourier ring correlation (FRC) at the "
        f"1/7 threshold: {m.group('res')} nm ({m.group('mode')}, {m.group('n')} points, "
        f"{m.group('px')} nm pixels)."
    ), [CITE_FRC_BANTERLE, CITE_FRC_NIEUWENHUIZEN]


#: (compiled pattern, stage, render(match, event, state) -> (sentence, [citations]))
RULES = [
    (re.compile(r"^Loaded dataset '(?P<name>.+?)':"), "load", _render_load),
    (re.compile(r"^MSR overlay loaded from '(?P<file>.+?)': (?P<n>\d+) channel"),
     "load", _render_msr_overlay),
    (re.compile(r"Computed localization precision for '(?P<name>.+?)' using StdDev per trace: "
                r"median sigma=(?P<med>\([^)]*\)) nm"), "analysis", _render_stddev_auto),
    (re.compile(r"Localization precision \(StdDev per trace\): combined \(n-weighted\) "
                r"sigma_r = (?P<sr>[\d.]+) nm, sigma_z = (?P<sz>[\d.]+) nm over "
                r"(?P<used>[\d,]+) of (?P<total>[\d,]+) traces"), "analysis", _render_stddev),
    (re.compile(r"Localization precision \(CRLB[^)]*\): median σ_xy = (?P<sxy>[\d.]+) nm "
                r"\(background-limited\), (?P<ideal>[\d.]+) nm \(ideal\)"
                r"(?:, σ_z = (?P<sz>[\d.]+) nm[^,]*)?, L = (?P<L>[\d.]+) nm.*?"
                r"median N = (?P<N>[\d.]+) photons"
                r"(?:; measured σ_r = (?P<mr>[\d.]+) nm \(StdDev/trace\) → "
                r"excess σ_fl = (?P<fl>[\d.]+) nm)?"), "analysis", _render_crlb),
    (re.compile(r"Localization precision \(FRC\): resolution = (?P<res>[\d.]+) nm "
                r"\(1/7 threshold, (?P<mode>[^,]+), (?P<n>[\d,]+) points, "
                r"pixel (?P<px>[\d.]+) nm\)"), "analysis", _render_frc),
    (re.compile(r"RIMF for '(?P<name>.+?)': (?P<value>[\d.]+)\s*(?:\((?P<note>[^)]*)\))?"),
     "analysis", _render_rimf),
    (re.compile(r"NPC segmentation \(2D\): detected (?P<n>\d+) NPC\(s\) on '(?P<name>.+?)'.*?"
                r"diameter=(?P<diam>[\d.]+) nm, rim=(?P<rim>[\d.]+) nm, pixel=(?P<pixel>[\d.]+) nm, "
                r"min support=(?P<support>[\d.]+)"), "segmentation", _render_npc),
    (re.compile(r"Separated '(?P<name>.+?)' into (?P<n>\d+) DCR channel"), "transform", _render_dcr),
    (re.compile(r"Duplicated dataset as '(?P<name>.+?)'"), "transform",
     lambda m, ev, st: (f"Dataset '{m.group('name')}' was duplicated.", [])),
    (re.compile(r"Created overlay \d+ with (?P<n>\d+) dataset"), "transform",
     lambda m, ev, st: (f"{m.group('n')} datasets were combined into a multi-channel overlay.", [])),
]


def _guess_stage(message: str) -> str:
    low = message.lower()
    if "filter" in low:
        return "filter"
    if "saved" in low or "export" in low:
        return "export"
    if "rimf" in low or "anisotropy" in low or "precision" in low:
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
