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
import math
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


def _render_aggregation(m, ev, state):
    time_mode = m.group("time_mode")
    if time_mode == "first":
        time_sentence = (
            "The aggregate timestamp was assigned from the first contributing "
            "localization, following the legacy nested-record convention."
        )
    elif time_mode == "photon_weighted":
        time_sentence = (
            "The aggregate timestamp was the photon-count-weighted mean of the "
            "contributing localization timestamps, following the modern flat-record "
            "convention."
        )
    else:
        time_sentence = f"The recorded aggregate timestamp mode was '{time_mode}'."
    return (
        f"Photon-threshold aggregation was applied to valid final localizations from "
        f"'{m.group('source')}', separately within each trace and in timestamp order. "
        f"For each localization i, the photon count P_i was calculated by summing the "
        f"background-corrected effective counts (eco) over final-scale iteration(s) "
        f"{m.group('iters')} (0-based raw iteration indices; iteration selection: "
        f"{m.group('iter_source')}). Consecutive localizations were accumulated until "
        f"ΣP_i reached or exceeded "
        f"{m.group('thr')} photons per aggregated localization. Complete localizations "
        f"were not split, so completed groups could exceed the threshold; the final "
        f"sub-threshold remainder of each trace was retained. The aggregated coordinate "
        f"was the photon-weighted centroid Σ(P_i r_i)/ΣP_i, and the reported photon "
        f"count was ΣP_i. {time_sentence} This reduced {m.group('nin')} contributing "
        f"localizations to {m.group('nout')} aggregated localizations in "
        f"'{m.group('result')}'."
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


def _method_number(value, digits: int = 3, *, strip: bool = True) -> str:
    """Compact, locale-independent number formatting for method provenance.

    ``strip=False`` keeps trailing zeros, so a list of quantities of the same
    kind stays column-consistent (8.94 / 10.14 / 11.00 rather than 11).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not recorded"
    if not math.isfinite(number):
        return "not available"
    text = f"{number:.{digits}f}"
    # Only trailing zeros AFTER a decimal point are redundant.  Stripping
    # unconditionally turned 60 into 6 and 50 into 5 whenever digits=0.
    if strip and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _method_count(value) -> str:
    # Group separators appear in the Log line this rule may be parsed from, so
    # they must be tolerated rather than reported as a missing value.
    if isinstance(value, str):
        value = value.replace(",", "").replace(" ", "").replace("\xa0", "").strip()
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "not recorded"


def _hlyb_setting(raw, effective, *, digits: int = 3, unit: str = "") -> str:
    raw_text = _method_number(raw, digits)
    effective_text = _method_number(effective, digits)
    suffix = f" {unit}" if unit else ""
    try:
        automatic = float(raw) <= 0
    except (TypeError, ValueError):
        automatic = False
    if automatic:
        return f"automatic (effective value {effective_text}{suffix})"
    return f"{raw_text}{suffix}"


def _hlyb_structure_size_text(counts) -> str:
    if not isinstance(counts, dict) or not counts:
        return "not recorded"
    try:
        ordered = sorted(((int(size), int(count)) for size, count in counts.items()))
    except (TypeError, ValueError):
        return "not recorded"
    return ", ".join(f"{count:,} with {size} observed site(s)" for size, count in ordered)


def _render_hlyb_template3d(m, ev, state):
    """Render a complete scientific account from one HlyB analysis event.

    New events carry a structured provenance snapshot. The regex groups retain
    compatibility with historical log lines, for which unrecorded settings are
    explicitly identified instead of being reconstructed from mutable defaults.
    """
    method = ev.get("method_data")
    if not isinstance(method, dict) or method.get("schema") != "hlyb_template_matching_3d/v1":
        name = m.group("name")
        return (
            f"Input data. Three-dimensional HlyB template matching was applied to dataset "
            f"'{name}'. The implementation reads the canonical localization coordinates "
            f"(loc_x, loc_y and loc_z) and trace identifier (tid) at the valid final "
            f"iteration; raw coordinates are converted from metres to nanometres and Z is "
            f"multiplied by the recorded scale factor {_method_number(m.group('zscale'))}. "
            f"Viewer filter masks and ROI selections are not applied by this operation.\n\n"
            f"Parameters and computation. The selected historical event records a minimum "
            f"of {_method_count(m.group('minloc'))} localization(s) per trace, an effective "
            f"basic-unit diameter of {_method_number(m.group('dunit'))} nm, an effective "
            f"template pair tolerance of {_method_number(m.group('tolerance'))} nm, and a "
            f"candidate-graph edge radius of {_method_number(m.group('edge'))} nm. Trace "
            f"centres are screened by localization support and local density, by a "
            f"Laplacian-of-Gaussian response in an XY localization histogram, and then merged "
            f"with DBSCAN to obtain candidate subunit centres. Connected candidate groups are "
            f"tested against partial assignments of the six-site, C3-symmetric HlyB model by "
            f"minimizing the root-mean-square residual between all observed and expected "
            f"within-candidate pair distances. Accepted candidates are ranked by site count, "
            f"residual and trace support, after which overlapping lower-ranked candidates are "
            f"discarded. Other user settings and intermediate screening counts were not "
            f"serialized in this legacy Log event and are therefore not inferred.\n\n"
            f"Results. {_method_count(m.group('traces'))} input trace(s) yielded "
            f"{_method_count(m.group('subunits'))} candidate subunit(s) and "
            f"{_method_count(m.group('structures'))} non-overlapping HlyB structure(s). Of "
            f"{_method_count(m.group('tested'))} template candidate(s) tested, "
            f"{_method_count(m.group('passed'))} passed the numerical thresholds and "
            f"{_method_count(m.group('overlap'))} were subsequently rejected because they "
            f"overlapped a better-ranked match. The gated result contained "
            f"{_method_count(m.group('pairs'))} unique within-structure pair distance(s), with "
            f"a median of {_method_number(m.group('median'))} nm. These distances are not the "
            f"all-to-all distribution of every detected subunit; the optional 'show all "
            f"(remove template gating)' overlay computes that ungated distribution separately."
        ), []

    inp = method.get("input", {})
    params = method.get("parameters", {})
    effective = method.get("effective_parameters", {})
    template = method.get("template", {})
    screening = method.get("screening", {})
    result = method.get("result", {})

    name = inp.get("dataset_name") or m.group("name")
    source_path = str(inp.get("source_path") or "").strip()
    source = f" (source file '{source_path}')" if source_path else ""
    # source_version is the structural data version (m2410 / m2205 / legacy);
    # source_format is the transport container.  They are orthogonal and must
    # not be merged into one "source format" phrase.
    source_format = str(inp.get("source_format") or "").strip()
    source_version = str(inp.get("source_version") or "").strip()
    format_bits = []
    if source_version:
        format_bits.append(f"source version {source_version}")
    if source_format:
        format_bits.append(f"container {source_format}")
    format_text = f"; {', '.join(format_bits)}" if format_bits else ""
    coordinate_fields = inp.get("coordinate_fields", ["loc_x", "loc_y", "loc_z"])
    coordinate_text = ", ".join(str(field) for field in coordinate_fields)
    z_note = (
        "No axial coordinate was available, so Z was set to zero before scaling."
        if inp.get("z_was_synthesized")
        else "The raw axial coordinate was retained until this run-specific scaling step."
    )

    proj = method.get("projection") or {}
    if proj.get("is_2d"):
        cell = proj.get("cell_mask_stats") or {}
        shrink = (
            f"{_method_number(cell.get('border_fraction'), 2)} of each cell's half-width"
            if str(cell.get("border_mode")) == "relative"
            else f"{_method_number(cell.get('border_size_nm'), 0)} nm")
        template_projection_text = (
            "\n\nProjection. Matching was performed in the image plane. Each E.coli was "
            f"first delineated from the localization density "
            f"({_method_count(cell.get('n_cells'))} cell(s)) and shrunk inward by "
            f"{shrink}, which removed "
            f"{_method_count(proj.get('n_border_traces'))} of "
            f"{_method_count(proj.get('n_total_traces'))} trace(s) at the rim, where "
            f"the membrane is seen edge-on and an in-plane distance is most "
            f"foreshortened. A projected separation is never longer than the true one "
            f"and can be shorter by up to 1 - cos(tilt) of it, so scoring a projected "
            f"observation against three-dimensional model distances with a symmetric "
            f"residual would reject genuinely matching complexes for being tilted, and "
            f"would do so more the larger the distance. Shortening was therefore "
            f"forgiven up to the tilt the shrink admits, "
            f"{_method_number(proj.get('tilt_deg'), 1)}°, i.e. up to "
            f"{_method_number(100 * float(proj.get('max_shortening', 0.0)), 0)} % of each "
            f"modelled distance; lengthening was never forgiven, because projection "
            f"cannot produce it. The reference geometry is itself planar, so a face-on "
            f"complex projects to the modelled distances exactly. The projection still "
            f"superimposes the upper and lower membrane, which is not corrected."
        )
    else:
        template_projection_text = ""

    pair_tolerance = _hlyb_setting(
        params.get("pair_tolerance_nm"), effective.get("pair_tolerance_nm"), unit="nm")
    rms_threshold = _hlyb_setting(
        params.get("rms_threshold_nm"), effective.get("rms_threshold_nm"), unit="nm")
    max_residual = _hlyb_setting(
        params.get("max_pair_residual_nm"), effective.get("max_pair_residual_nm"), unit="nm")
    basic_unit = _hlyb_setting(
        params.get("basic_unit_size_nm"), effective.get("basic_unit_size_nm"), unit="nm")

    class_distances = template.get("class_distances_nm", {})
    if isinstance(class_distances, dict) and class_distances:
        class_text = "; ".join(
            f"{name} {_method_number(distance, 2, strip=False)} nm"
            for name, distance in class_distances.items()
        )
        model_values = [float(v) for v in class_distances.values()]
        min_model = min(model_values)
        max_model = max(model_values)
    else:
        class_text = "not recorded"
        min_model = max_model = float("nan")

    # Quantities the reader needs in order to judge the result, derived from
    # the recorded settings rather than asserted.
    tol = effective.get("pair_tolerance_nm")
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        tol = float("nan")
    try:
        merge_radius = float(effective.get("basic_unit_size_nm")) / 2.0
    except (TypeError, ValueError):
        merge_radius = float("nan")
    n_sites = len(template.get("site_labels") or []) or 6
    min_n = params.get("min_observed_subunits")
    try:
        min_n = int(min_n)
    except (TypeError, ValueError):
        min_n = 3
    n_assign = 1
    for k in range(min_n):
        n_assign *= max(n_sites - k, 1)

    if math.isfinite(merge_radius) and math.isfinite(min_model):
        if merge_radius >= min_model - (tol if math.isfinite(tol) else 0.0):
            merge_note = (
                f"This merge radius, {_method_number(merge_radius, 2)} nm, is not smaller "
                f"than the shortest modelled distance ({_method_number(min_model, 2)} nm) "
                f"less the matching tolerance, so two genuinely distinct sites at that "
                f"separation are merged into a single centre and the corresponding "
                f"distance class cannot be recovered from this run. Distances shorter "
                f"than the merge radius are absent from the output by construction, and "
                f"the resulting truncation of an otherwise decreasing pair-distance "
                f"distribution produces a maximum immediately above it that must not be "
                f"interpreted as a structural distance."
            )
        else:
            merge_note = (
                f"The merge radius, {_method_number(merge_radius, 2)} nm, is smaller than "
                f"the shortest modelled distance ({_method_number(min_model, 2)} nm), so "
                f"all modelled classes remain resolvable in principle. Distances shorter "
                f"than the merge radius are nonetheless absent by construction, and the "
                f"truncation produces a maximum immediately above it that is a property "
                f"of the detection step, not of the sample."
            )
    else:
        merge_note = ""

    z_source = str(inp.get("z_scaling_source") or "").strip()
    z_source_text = f" The scale factor was taken from {z_source}." if z_source else ""

    text = (
        f"Input data. Three-dimensional HlyB subunit-pair template matching was performed on "
        f"dataset '{name}'{source}. The input comprised "
        f"{_method_count(inp.get('n_localizations'))} valid localization record(s) from "
        f"{_method_count(inp.get('n_traces'))} trace(s){format_text}. The canonical "
        f"{coordinate_text} coordinate field(s) and tid were read from rows selected as "
        f"itr='last' (the global "
        f"final iteration) with vld_only=True. Viewer filter masks and ROI selections were "
        f"not applied. Coordinates were converted from metres to nanometres, and raw Z was "
        f"multiplied by {_method_number(params.get('z_scaling_factor'), 4)} (RIMF, the "
        f"refractive-index mismatch factor).{z_source_text} {z_note}"
        f"{template_projection_text}\n\n"
        f"User-defined parameters. The minimum localization support was "
        f"{_method_count(params.get('min_loc_per_trace'))} localization(s) per trace; the XY "
        f"rendering pixel used for subunit detection was "
        f"{_method_number(params.get('unit_render_pixel_size_nm'))} nm; and the basic-unit "
        f"diameter was {basic_unit}. The same minimum-support number is applied twice, as a "
        f"minimum localization count per trace and as a minimum neighbour count around the "
        f"trace centre; they are not independently adjustable. A partial structure required at "
        f"least {_method_count(params.get('min_observed_subunits'))} observed subunit(s). The "
        f"six-site model used equilateral A- and B-rings with side lengths "
        f"{_method_number(params.get('core_a_ring_side_nm'))} and "
        f"{_method_number(params.get('core_b_ring_side_nm'))} nm, respectively; the B-ring "
        f"was rotated by {_method_number(params.get('core_twist_deg'), 1)}° relative to the "
        f"A-ring and displaced axially by "
        f"{_method_number(params.get('core_axial_offset_nm'))} nm. The pair-distance tolerance "
        f"setting was {pair_tolerance}, the RMS-residual threshold setting was {rms_threshold}, "
        f"the maximum single-pair residual setting was {max_residual}, and the required fraction "
        f"of pairs "
        f"within tolerance was {_method_number(params.get('min_pair_match_fraction'))}.\n\n"
        f"Subunit detection. A per-axis median centre and localization count were computed "
        f"for each trace. When the basic-unit diameter was automatic, it was calculated as "
        f"twice the median of three estimates of the positive within-trace radial offsets: "
        f"the geometric mean in log space, the centre of the modal log-distance bin, and the "
        f"centre from a Gaussian fit to the log-distance histogram (with a 10 nm fallback if "
        f"estimation was degenerate). First, traces were retained when both their localization "
        f"count and their local-density score—the number of localizations within half the "
        f"basic-unit diameter of the trace centre, minus one—were at least "
        f"{_method_count(params.get('min_loc_per_trace'))}; "
        f"{_method_count(screening.get('n_after_trace_density'))} trace centre(s) passed. "
        f"Second, all XY localizations were histogrammed at the selected pixel size and filtered "
        f"with a Laplacian-of-Gaussian kernel of σ = Dunit/(2√2·pixel size); centres whose "
        f"normalized response exceeded the standard deviation of the response map were retained "
        f"({_method_count(screening.get('n_after_log'))} centre(s)). Third, surviving trace "
        f"centres were merged by three-dimensional DBSCAN with ε = Dunit/2 and minPts = 1. "
        f"Each resulting subunit centre was the per-axis median of all localizations from "
        f"its member traces. {merge_note}\n\n"
        f"Structural model. The six N-terminal labelling sites 1a, 1b, 2a, 2b, 3a and 3b of "
        f"the three HlyB dimers were represented by one Euclidean coordinate model: two "
        f"coplanar, C3-symmetric equilateral rings, the A-ring carrying 1a/2a/3a and the "
        f"B-ring 1b/2b/3b. All expected pair distances were derived from those coordinates, so "
        f"they are mutually consistent by construction: {class_text}. These are the "
        f"inter-domain distances of the reference structure, and agree with the values "
        f"annotated on the source structural diagram to better than 0.25 nm in every class. "
        f"They are approximately 4 nm shorter than the distances tabulated alongside that "
        f"diagram, which by the source's own statement include an allowance of 2 nm per "
        f"single-domain antibody at each endpoint. The model is therefore a model of the "
        f"protein domains, whereas the measurement localizes fluorophores displaced from those "
        f"domains by the antibody. If that displacement is isotropically oriented it lengthens "
        f"an observed distance by only about 1 nm, and the domain distances used here are "
        f"appropriate; if it is directed radially outward from the complex axis it lengthens "
        f"observed distances by 3-4 nm, reproducing the tabulated values, in which case every "
        f"match reported here is biased short by that amount. The displacement geometry was not "
        f"determined independently in this work, and this remains the dominant systematic "
        f"uncertainty of the measurement. Because both rings are coplanar, the model is "
        f"invariant under reflection and carries no information about the handedness or the "
        f"axial order of the complex.\n\n"
        f"Matching. An undirected "
        f"candidate graph joined detected subunit centres separated by at most "
        f"{_method_number(effective.get('candidate_edge_radius_nm'))} nm. Within each connected "
        f"component, every subset from "
        f"{_method_count(effective.get('max_observed_subunits'))} down to "
        f"{_method_count(params.get('min_observed_subunits'))} observed sites was evaluated, "
        f"unless the number of same-size subsets exceeded "
        f"{_method_count(effective.get('max_candidate_subsets_per_component'))}. For each subset, "
        f"all ordered assignments to distinct template sites were scored from all n(n−1)/2 "
        f"pair distances, using residual d_observed − d_expected; the assignment with the lowest "
        f"RMS residual was retained. Candidates with at least three sites had to satisfy the RMS, "
        f"maximum-residual and match-fraction thresholds simultaneously (a two-site candidate, "
        f"if enabled, was judged only by the pair tolerance). Passing candidates were ordered by "
        f"decreasing site count, increasing RMS residual and decreasing mean trace support, then "
        f"greedily accepted subject to non-overlap.\n\n"
        f"Specificity of the matching criterion. The reported residuals are minima over "
        f"{_method_count(n_assign)} ordered site assignments for the smallest accepted subset "
        f"size, so they are selection-biased downward and are not goodness-of-fit statistics in "
        f"the usual sense. The model contains {_method_count(len(class_distances) if isinstance(class_distances, dict) else 0)} "
        f"distinct distances spanning "
        f"{_method_number(min_model, 2)} to {_method_number(max_model, 2)} nm; with a tolerance "
        f"of {_method_number(tol, 2)} nm the accepted intervals overlap into the single "
        f"contiguous range {_method_number(min_model - tol, 2)} to "
        f"{_method_number(max_model + tol, 2)} nm, within which any pair distance satisfies at "
        f"least one class. The criterion is correspondingly permissive for the smallest subsets, "
        f"and at three observed sites the match-fraction threshold is degenerate, since the only "
        f"attainable fractions are 0, 1/3, 2/3 and 1. No chance-level expectation was computed "
        f"for this run: the number of accepted structures reported below is not accompanied by "
        f"the number that the same procedure would accept on spatially randomized data, and "
        f"should not be read as a detection count until that comparison is made.\n\n"
        f"Results. The screening produced {_method_count(result.get('n_subunits'))} detected "
        f"subunit centre(s) and {_method_count(screening.get('n_components'))} candidate-graph "
        f"component(s). {_method_count(screening.get('n_candidates_tested'))} subset candidate(s) "
        f"were tested; {_method_count(screening.get('n_candidates_passed_thresholds'))} passed "
        f"the numerical thresholds, {_method_count(screening.get('n_overlap_rejected'))} of those "
        f"were rejected during overlap resolution, and "
        f"{_method_count(screening.get('n_skipped_large_subsets'))} subset(s) were skipped by the "
        f"combinatorial cap. The final set contained "
        f"{_method_count(result.get('n_structures'))} non-overlapping HlyB structure(s) "
        f"({_hlyb_structure_size_text(result.get('structure_size_counts'))}). Their accepted "
        f"within-structure pairs contributed {_method_count(result.get('n_pairs'))} distances "
        f"from {_method_number(result.get('pair_distance_min_nm'))} to "
        f"{_method_number(result.get('pair_distance_max_nm'))} nm (median "
        f"{_method_number(result.get('pair_distance_median_nm'))} nm). The median structure RMS "
        f"residual was {_method_number(result.get('structure_rms_median_nm'))} nm, the median "
        f"absolute pair residual was {_method_number(result.get('residual_median_abs_nm'))} nm "
        f"(maximum {_method_number(result.get('residual_max_abs_nm'))} nm), and the median matched-"
        f"pair fraction was {_method_number(result.get('match_fraction_median'))}; the last of "
        f"these is uninformative for three-site structures, as noted above. The reported "
        f"blue histogram represents template-gated pairs within accepted structures, "
        f"not all detected-subunit pairs. The optional 'show all (remove template gating)' control "
        f"computes the all-to-all detected-subunit distance distribution exactly and overlays it "
        f"in light gray without changing the matching result; that overlay is not an unbiased "
        f"reference distribution, because it is subject to the same short-distance exclusion "
        f"imposed by the merge radius."
    )
    return text, []


_PAIR_FIT_HYPOTHESIS = {
    "dimer_gaussian": "a single inter-subunit distance with a Gaussian spread",
    "dimer_uniform": "a single inter-subunit distance uniform within a band",
    "dimer_lognormal": "a single inter-subunit distance with a log-normal spread",
    "trimer_six_site": "the published six-site C3 trimer",
    "no_structure": "no structure beyond the same-site population",
    # legacy keys from earlier runs
    "six_site": "the six-site HlyB complex",
    "dimer_only": "a single dimer distance",
}


def _render_hlyb_staged_short_range(m, ev, state):
    """Scientific account of the model-independent staged 3-D workflow."""
    method = ev.get("method_data")
    name = m.group("name")
    if not isinstance(method, dict) or method.get("schema") != "hlyb_staged_short_range_3d/v1":
        return (
            f"Analysis. A staged three-dimensional HlyB short-range population "
            f"analysis was applied to dataset '{name}'. Trace centroids were "
            f"conservatively consolidated into label-site estimates and their "
            f"within-component pair-distance profile compared with a conditional "
            f"rod-surface randomization. The settings of this run were not serialized."
        ), []

    inp = method.get("input", {})
    params = method.get("parameters", {})
    sites = method.get("site_inference", {})
    components = method.get("components", {})
    result = method.get("result", {})
    bootstrap = method.get("bootstrap", {})
    sensitivity = method.get("sensitivity") or []
    source_path = str(inp.get("source_path") or "").strip()
    source = f" (source file '{source_path}')" if source_path else ""

    if bootstrap.get("available"):
        ci = bootstrap.get("centroid_ci95_nm") or [float("nan"), float("nan")]
        ratio_ci = bootstrap.get("band_ratio_ci95") or [float("nan"), float("nan")]
        narrow = (
            " Because only a handful of components were available, this interval "
            "understates the true between-cell variance and the parameter-sensitivity "
            "spread below is the interval we quote."
            if bootstrap.get("narrow_ci_warning") else "")
        bootstrap_text = (
            f"Resampling the {_method_count(bootstrap.get('n_components'))} spatial "
            f"components as independent units gave a 95 % interval of "
            f"{_method_number(ci[0], 2)}–{_method_number(ci[1], 2)} nm for the "
            f"positive-excess centroid and {_method_number(ratio_ci[0], 2)}–"
            f"{_method_number(ratio_ci[1], 2)} for the band ratio.{narrow}"
        )
    else:
        bootstrap_text = (
            "A component-level interval was not reported because fewer than three "
            "independent retained components were available."
        )

    profile = method.get("stratum_profile") or {}
    profile_rows = profile.get("rows") or []
    if profile_rows:
        p_lo, p_hi = (profile.get("band_ratio_range")
                      or [float("nan"), float("nan")])
        c_lo, c_hi = (profile.get("centroid_range_nm")
                      or [float("nan"), float("nan")])
        strata = ", ".join(_method_count(row.get("null_stratum_sites"))
                           for row in profile_rows)
        conditional = bool(profile.get("band_ratio_is_stratum_conditional"))
        profile_text = (
            f"Because the stratum width sets the axial window within which the "
            f"randomization destroys structure, it also sets how much structure the "
            f"null absorbs. Holding the inferred sites and components fixed and "
            f"varying the stratum alone over {strata} sites, the observed/null ratio "
            f"moved from {_method_number(p_lo, 2)} to {_method_number(p_hi, 2)} while "
            f"the positive-excess centroid moved only from {_method_number(c_lo, 2)} "
            f"to {_method_number(c_hi, 2)} nm. "
            + ("The ratio is therefore conditional on the stratification scale and is "
               "reported together with it; the location of the excess, not the "
               "magnitude of the ratio, is the stable descriptor."
               if conditional else
               "The ratio was therefore stable across the stratification scale.")
        )
    else:
        profile_text = ""

    finite_ratios = [float(row.get("band_ratio")) for row in sensitivity
                     if isinstance(row, dict) and row.get("band_ratio") is not None
                     and math.isfinite(float(row.get("band_ratio")))]
    finite_centroids = [float(row.get("positive_excess_centroid_nm")) for row in sensitivity
                        if isinstance(row, dict)
                        and row.get("positive_excess_centroid_nm") is not None
                        and math.isfinite(float(row.get("positive_excess_centroid_nm")))]
    if finite_ratios and finite_centroids:
        robust = method.get("robust_short_range_excess_calibrated")
        if robust is None:
            robust = method.get("robust_short_range_excess")
        criterion = _method_number(method.get("calibrated_ratio_z"), 1)
        robust_text = (
            f"Every valid variant showed a ratio above unity and at least "
            f"{criterion} standard deviations above its own null, so the "
            f"short-range claim was classified as robust."
            if robust is True else
            "At least one valid variant failed that criterion, so the short-range "
            "claim was classified as parameter-sensitive."
            if robust is False else "")
        sensitivity_text = (
            f"The mandatory sensitivity audit varied the same-site diameter over "
            f"3, 4 and 5 nm, the axial null stratum over 32, 64 and 128 sites, "
            f"and the spatial-component link over ±25 % of its primary value. "
            f"Across those runs the observed/null ratio ranged from "
            f"{_method_number(min(finite_ratios), 2)} to "
            f"{_method_number(max(finite_ratios), 2)}, and the positive-excess "
            f"centroid from {_method_number(min(finite_centroids), 2)} to "
            f"{_method_number(max(finite_centroids), 2)} nm; the latter span is the "
            f"uncertainty we quote for the location of the excess. {robust_text}"
        )
    else:
        sensitivity_text = "No parameter-sensitivity audit was requested for this run."

    text = (
        f"Input data. A staged three-dimensional short-range population analysis was "
        f"performed on dataset '{inp.get('dataset_name') or name}'{source}. "
        f"{_method_count(inp.get('n_localizations'))} valid localization record(s) at "
        f"the final iteration were grouped by trace identifier into "
        f"{_method_count(inp.get('n_traces_total'))} trace(s); "
        f"{_method_count(inp.get('n_traces_used'))} trace(s) containing at least "
        f"{_method_count(params.get('min_loc_per_trace'))} localizations entered the "
        f"analysis. Coordinates were converted to nanometres and raw z multiplied "
        f"once by the fixed RIMF {_method_number(params.get('z_scaling_factor'), 4)}. "
        f"Viewer filters and ROIs were not applied.\n\n"
        f"Site inference. Each trace was represented by its mean localization and "
        f"coordinate-wise standard error. Repeated traces were consolidated without "
        f"LoG image filtering or DBSCAN. Candidate merges were weighted by their "
        f"measured uncertainty and accepted only when every trace-centroid pair in "
        f"the combined group remained within a hard "
        f"{_method_number(params.get('site_merge_nm'), 1)} nm complete-link diameter. "
        f"This produced {_method_count(sites.get('n_sites'))} inferred label-site "
        f"estimate(s), including {_method_count(sites.get('n_repeated_sites'))} "
        f"multi-trace site(s); {_method_count(sites.get('n_traces_consolidated'))} "
        f"redundant trace representation(s) were removed. These are label-site "
        f"estimates, not assignments to HlyB protomers. Acquisition time imposed no "
        f"maximum revisit gap, so spatially compatible visits throughout the recording "
        f"could consolidate; however, this implementation did not fit a complete "
        f"DDC/BaGoL temporal emitter model.\n\n"
        f"Spatial null and observable. Sites were connected at "
        f"{_method_number(params.get('cell_link_nm'), 0)} nm solely to separate "
        f"coarse spatial/cell components; pairs were formed only within a component. "
        f"{_method_count(components.get('n_retained'))} component(s) with at least "
        f"{_method_count(params.get('min_sites_per_component'))} sites were retained, "
        f"containing {_method_count(sites.get('n_sites_used'))} sites; "
        f"{_method_count(components.get('n_excluded_sites'))} site(s) in smaller "
        f"components were explicitly excluded. Pair distances through "
        f"{_method_number(params.get('r_max_nm'), 0)} nm were histogrammed in "
        f"{_method_number(params.get('bin_nm'), 2)} nm bins. For each retained "
        f"component, principal-component coordinates supplied a local rod axis. "
        f"The observed axial coordinate of every site was held fixed while complete "
        f"observed transverse membrane coordinates were permuted within adjacent "
        f"axial-rank strata of {_method_count(params.get('null_stratum_sites'))} sites. "
        f"Thus every surrogate preserved exact site count, axial density, local "
        f"radial support, one-sided visibility and component boundaries without "
        f"filling a three-dimensional volume. "
        f"{_method_count(params.get('null_replicates'))} conditional randomizations "
        f"defined the reference distribution.\n\n"
        f"Result. In the pre-declared {_method_number(params.get('short_range_lo_nm'), 1)}–"
        f"{_method_number(params.get('short_range_hi_nm'), 1)} nm interval, "
        f"{_method_count(result.get('band_observed_pairs'))} observed pair(s) were "
        f"compared with {_method_number(result.get('band_null_mean_pairs'), 1)} ± "
        f"{_method_number(result.get('band_null_sd_pairs'), 1)} under the surface null, "
        f"an observed/null ratio of {_method_number(result.get('band_ratio'), 3)}. "
        f"Evidence is reported as the position of that ratio within the null "
        f"replicate distribution, which was centred on "
        f"{_method_number(result.get('null_band_ratio_mean'), 3)} with a standard "
        f"deviation of {_method_number(result.get('null_band_ratio_sd'), 3)}, placing "
        f"the observation {_method_number(result.get('band_ratio_z'), 1)} standard "
        f"deviations above the null. The rank-based one-sided randomization p-value "
        f"was {_method_number(result.get('band_p'), 4)}, but it is bounded below by "
        f"{_method_number(result.get('band_p_resolution'), 4)} at this replicate count "
        f"and was found to be anti-conservative when surrogates drawn from the null "
        f"itself were re-tested; it is reported for completeness rather than as the "
        f"evidential statistic. The positive portion of the "
        f"observed-minus-null profile had a peak at "
        f"{_method_number(result.get('peak_nm'), 2)} nm, centroid at "
        f"{_method_number(result.get('positive_excess_centroid_nm'), 2)} nm and median "
        f"at {_method_number(result.get('positive_excess_median_nm'), 2)} nm. "
        f"{bootstrap_text} {sensitivity_text}"
        + (f"\n\nDependence on the null stratification. {profile_text}"
           if profile_text else "")
        + f"\n\nThe centroid and peak describe an excess "
        f"population only: the analysis neither identifies pair membership nor fits or "
        f"claims a molecular HlyB dimer distance."
    )
    return text, []


def _pair_fit_ranking(fits: dict, best: str) -> str:
    """Rank competing hypotheses by AIC difference, worst evidence first."""
    if not isinstance(fits, dict) or not fits:
        return "not recorded"
    parts = []
    for name, fit in sorted(fits.items(), key=lambda kv: kv[1].get("delta_aic", 0.0)):
        label = _PAIR_FIT_HYPOTHESIS.get(name, name)
        gap = fit.get("delta_aic", 0.0)
        if name == best:
            parts.append(f"{label} (preferred)")
        else:
            parts.append(f"{label} worse by {_method_number(gap, 1)} AIC units")
    return "; ".join(parts)


def _render_hlyb_pair_fit(m, ev, state):
    """Account of one ensemble pair-distance model fit.

    The structural claim this method supports is deliberately narrower than a
    per-complex detection, and the text is written so that it cannot be read as
    one.
    """
    method = ev.get("method_data")
    name = m.group("name")
    if not isinstance(method, dict) or method.get("schema") != "hlyb_pair_distance_fit_3d/v1":
        return (
            f"Analysis. An ensemble pair-distance model fit was applied to dataset "
            f"'{name}'. The method measures the distribution of distances between "
            f"trace centroids without merging them, compares it with an "
            f"envelope-preserving randomized reference, and fits a model comprising a "
            f"same-site short-range population, the six-site HlyB geometry and "
            f"unrelated pairs. The settings and fitted values of this run were not "
            f"serialized in the Log event and are therefore not reproduced here."
        ), []

    inp = method.get("input", {})
    params = method.get("parameters", {})
    obs = method.get("observable", {})
    kernel = method.get("repeat_kernel", {})
    model = method.get("model", {})
    fits = method.get("fits", {})
    relaxed = method.get("fits_relaxed_kernel", {})
    best = str(method.get("best_hypothesis", ""))
    best_fit = fits.get(best, {})

    dataset = inp.get("dataset_name") or name
    source_path = str(inp.get("source_path") or "").strip()
    source = f" (source file '{source_path}')" if source_path else ""
    bits = []
    if inp.get("source_version"):
        bits.append(f"source version {inp['source_version']}")
    if inp.get("source_format"):
        bits.append(f"container {inp['source_format']}")
    format_text = f"; {', '.join(bits)}" if bits else ""

    sem = obs.get("centroid_sem_nm") or []
    sem_text = (" / ".join(_method_number(v, 2) for v in sem) + " nm in x / y / z"
                if len(sem) == 3 else "not recorded")

    projection = method.get("projection") or {}
    if projection.get("is_2d"):
        cell = projection.get("cell_mask_stats") or {}
        shrink = (
            f"{_method_number(cell.get('border_fraction'), 2)} of each cell's "
            f"half-width" if str(cell.get("border_mode")) == "relative"
            else f"{_method_number(cell.get('border_size_nm'), 0)} nm")
        projection_text = (
            f"\n\nProjection. The analysis was performed in the image plane, but not "
            f"by discarding the axial coordinate: doing so shortens every distance by "
            f"an amount that depends on the pair's orientation, and for a membrane "
            f"protein on a rod-shaped cell that orientation varies systematically "
            f"across the projected cell — face-on at the centre, edge-on at the rim. "
            f"Each cell was therefore delineated from the localization density "
            f"({_method_count(cell.get('n_cells'))} cell(s), containing "
            f"{_method_number(100 * float(cell.get('in_mask_fraction', 0)), 0)} % of the "
            f"trace centres, median projected half-width "
            f"{_method_number(cell.get('median_half_width_nm'), 0)} nm) and shrunk "
            f"inward by {shrink}, retaining "
            f"{_method_number(100 * float(cell.get('retained_fraction', 0)), 0)} % of "
            f"the centres. "
        )
        if "implied_max_tilt_deg" in cell:
            projection_text += (
                f"That shrink admits only membrane normals within "
                f"{_method_number(cell.get('implied_max_tilt_deg'), 1)}° of face-on. ")
        projection_text += (
            f"The depth of each retained localization within its own cell gives its "
            f"local membrane tilt (median "
            f"{_method_number(projection.get('median_tilt_deg'), 1)}°), and the "
            f"foreshortening that survives the shrink was modelled from that measured "
            f"tilt distribution rather than ignored: a pair in a plane tilted by θ at "
            f"in-plane azimuth φ projects to √(1 − sin²θ·sin²φ) of its true length, "
            f"giving a mean projected/true ratio of "
            f"{_method_number(projection.get('median_foreshortening'), 3)}. Distances "
            f"were blurred with the two-dimensional (Rice) density accordingly. The "
            f"projection nonetheless superimposes the upper and lower membrane, so "
            f"some apparently close pairs are far apart along the optical axis; that "
            f"is not corrected here."
        )
    else:
        projection_text = ""

    classes = model.get("class_distances_nm") or []
    class_text = (", ".join(_method_number(v, 2) for v in classes) + " nm"
                  if classes else "not recorded")

    if inp.get("time_column_available") and kernel.get("n_pairs"):
        kernel_text = (
            f"calibrated empirically from {_method_count(kernel.get('n_pairs'))} "
            f"consecutive trace pairs separated by less than "
            f"{_method_number(params.get('repeat_gap_s'), 2)} s, with a median "
            f"separation of {_method_number(kernel.get('median_nm'), 2)} nm. "
            f"Those pairs were selected on the acquisition time alone and never on "
            f"distance, so their distance distribution is an unbiased sample of "
            f"same-site separations rather than a restatement of a distance "
            f"threshold"
        )
    else:
        kernel_text = (
            f"not calibrated from the data — no usable acquisition-time column was "
            f"available — and an assumed width of "
            f"{_method_number(kernel.get('sigma_nm'), 2)} nm was used instead"
        )

    bounds = best_fit.get("parameters_at_bounds") or []
    summary = best_fit.get("distance_summary") or {}
    scan = method.get("distance_scan") or {}
    reference_dimer = model.get("reference_dimer_nm")

    if summary:
        distance_text = (
            f"The fitted distribution of true inter-subunit distances has a median of "
            f"{_method_number(summary.get('median_nm'), 2)} nm, with the central 68 % of "
            f"the population between {_method_number(summary.get('p16_nm'), 2)} and "
            f"{_method_number(summary.get('p84_nm'), 2)} nm (mode "
            f"{_method_number(summary.get('mode_nm'), 2)} nm, mean "
            f"{_method_number(summary.get('mean_nm'), 2)} nm). Percentiles of the fitted "
            f"distribution are quoted in preference to the raw shape parameters, because "
            f"they are comparable between shapes and because a broad population must not "
            f"be reported as though it were a precise distance."
        )
    else:
        distance_text = "No structural distance distribution was fitted."

    if reference_dimer and summary:
        distance_text += (
            f" For reference, the corresponding distance tabulated on the published "
            f"diagram is {_method_number(reference_dimer, 2)} nm; it was used only as a "
            f"starting value and as a point of comparison, never as a constraint."
        )

    if scan.get("available"):
        ci68 = scan.get("ci68_nm") or [float("nan"), float("nan")]
        ci95 = scan.get("ci95_nm") or [float("nan"), float("nan")]
        scan_text = (
            f"A profile-likelihood scan of the distribution centre, refitting all other "
            f"parameters at each step, gave {_method_number(scan.get('best_nm'), 2)} nm "
            f"with a 68 % interval of {_method_number(ci68[0], 2)} to "
            f"{_method_number(ci68[1], 2)} nm and a 95 % interval of "
            f"{_method_number(ci95[0], 2)} to {_method_number(ci95[1], 2)} nm."
        )
        if not scan.get("constrained", True):
            scan_text += (
                " The scan did not rise by 1.92 log-likelihood units anywhere in the "
                "permitted range, so the data do not localize the centre and only the "
                "width of the population is being measured."
            )
        elif scan.get("ci68_below_scan_step"):
            scan_text += (
                f" The 68 % interval is no wider than the "
                f"{_method_number(scan.get('step_nm'), 2)} nm scan step and is therefore "
                f"unresolved rather than tight."
            )
    else:
        scan_text = ""

    relaxed_gap = min((f.get("delta_aic", 0.0) for key, f in relaxed.items()
                       if key != method.get("best_hypothesis_relaxed", "")),
                      default=float("nan"))
    pinned_gap = min((f.get("delta_aic", 0.0) for key, f in fits.items()
                      if key != best), default=float("nan"))
    trimer = fits.get("trimer_six_site")
    if trimer is not None and str(best).startswith("dimer"):
        architecture_text = (
            f" The published six-site trimer geometry was among the candidates and was "
            f"worse than the preferred single-distance description by "
            f"{_method_number(trimer.get('delta_aic'), 0)} AIC units. This is consistent "
            f"with the trimeric assembly not surviving sample preparation, leaving "
            f"dimers whose separation is neither fixed at the tabulated value nor sharp."
        )
    elif str(best) == "trimer_six_site":
        architecture_text = (
            " The published six-site trimer geometry described the data better than any "
            "of the single-distance shapes tested."
        )
    else:
        architecture_text = ""

    if summary and best_fit.get("sigma_nm"):
        try:
            broad = float(summary.get("spread_nm", 0.0)) > float(best_fit["sigma_nm"])
        except (TypeError, ValueError):
            broad = False
        width_text = (
            f" The half-width of the central 68 % interval, "
            f"{_method_number(summary.get('spread_nm'), 2)} nm, "
            + ("exceeds" if broad else "does not exceed")
            + f" the {_method_number(best_fit.get('sigma_nm'), 2)} nm positional blur, so "
            f"the width of the distribution is "
            + ("a property of the sample — a flexible linkage or several coexisting "
               "conformations — rather than of the measurement."
               if broad else
               "not resolved above the measurement blur, and the distance is consistent "
               "with being sharp.")
        )
    else:
        width_text = ""

    text = (
        f"Input data. An ensemble pair-distance model fit was performed on dataset "
        f"'{dataset}'{source}. {_method_count(inp.get('n_localizations'))} valid "
        f"localization record(s) at the final iteration were grouped into "
        f"{_method_count(inp.get('n_traces_total'))} trace(s), of which "
        f"{_method_count(inp.get('n_traces_used'))} carried at least "
        f"{_method_count(params.get('min_loc_per_trace'))} localizations and entered the "
        f"analysis{format_text}. Coordinates were converted to nanometres and the axial "
        f"coordinate multiplied by "
        f"{_method_number(params.get('z_scaling_factor'), 4)} (RIMF, the "
        f"refractive-index mismatch factor), taken from "
        f"{inp.get('z_scaling_source') or 'the dataset'}. Viewer filter masks and ROI "
        f"selections were not applied.\n\n"
        f"Observable. Each trace was reduced to the mean of its localizations, whose "
        f"standard error was {sem_text}. Unlike a detection-based analysis, trace "
        f"centres were **not** merged into sub-unit centres: a merge threshold "
        f"comparable to the distances being measured removes them from the result and "
        f"creates a maximum immediately above itself, so the short-range population is "
        f"modelled here rather than deleted. All pair distances up to "
        f"{_method_number(params.get('r_max_nm'), 0)} nm were histogrammed in "
        f"{_method_number(params.get('bin_nm'), 2)} nm bins. A reference distribution "
        f"was obtained by redrawing the same number of centres from their own "
        f"{_method_number(params.get('null_cell_nm'), 0)} nm occupancy histogram "
        f"({_method_count(obs.get('null_replicates'))} replicates), which preserves the "
        f"cell-scale density while destroying all finer structure. The measured "
        f"distribution exceeded that reference by more than three standard deviations "
        f"out to {_method_number(obs.get('excess_outer_nm'), 1)} nm, and was "
        f"indistinguishable from it beyond.{projection_text}\n\n"
        f"Model. The distribution was described as the sum of three terms. The first is "
        f"a same-site short-range population — one molecule re-acquired as several "
        f"traces, the two fluorophores carried by one divalent antibody, and drift "
        f"between re-acquisitions — whose shape was {kernel_text}. The second is a "
        f"structural term: a distribution p(d) over true inter-subunit distances, "
        f"convolved with the exact three-dimensional blurred-distance density for a "
        f"pair of centres, so that the width of p(d) is separated from the positional "
        f"error of the measurement. The published trimer geometry (inter-domain "
        f"distances {class_text}) was **not** imposed. It describes a reference "
        f"architecture that need not survive sample preparation, and fixing it would "
        f"assume the result; it was instead entered as one candidate shape among "
        f"several, alongside a single distance with a Gaussian spread, a single "
        f"distance uniform within a fitted band — the fully elastic case — and a "
        f"single distance with a log-normal spread. The third term is the randomized "
        f"reference, scaled by one free amplitude, representing pairs from different "
        f"assemblies. Amplitudes were estimated by Poisson maximum likelihood over "
        f"{_method_number(params.get('fit_r_min_nm'), 1)} to "
        f"{_method_number(params.get('fit_r_max_nm'), 1)} nm, with the distance itself "
        f"free over {_method_number((params.get('dimer_distance_bounds_nm') or [0, 0])[0], 1)} "
        f"to {_method_number((params.get('dimer_distance_bounds_nm') or [0, 0])[1], 1)} nm. "
        f"The positional blur was **not** fitted. It was computed from the measured "
        f"centroid precision ({_method_number(obs.get('sigma_floor_nm'), 2)} nm for a "
        f"pair) combined in quadrature with the labelling allowance, and held common to "
        f"every candidate shape. A freely fitted blur absorbs the very width the "
        f"analysis is meant to measure, and because it is shared, the shape that "
        f"absorbs the most then imposes a blur that handicaps the others — tested on "
        f"simulated data, that alone cost the true architecture ~2100 AIC units on its "
        f"own ground truth.\n\n"
        f"Results. The preferred description was {_PAIR_FIT_HYPOTHESIS.get(best, best)}. "
        f"Ranked by Akaike information criterion: {_pair_fit_ranking(fits, best)}. The "
        f"fit assigned {_method_number(best_fit.get('n_structure_pairs'), 0)} pair(s) to "
        f"the structural term and {_method_number(best_fit.get('n_repeat_pairs'), 0)} to "
        f"the same-site term, with the randomized reference scaled by "
        f"{_method_number(best_fit.get('background_scale'), 3)} and a fitted pair blur "
        f"of {_method_number(best_fit.get('sigma_nm'), 2)} nm. {distance_text} "
        f"{scan_text}{architecture_text}{width_text}\n\n"
        f"Sensitivity and limits. The comparison above holds the same-site kernel at "
        f"its measured width. Repeating it with that width free to increase — which is "
        f"physically plausible, since drift and a divalent label both broaden the "
        f"population — the preferred model was unchanged but its margin over the next "
        f"hypothesis fell from {_method_number(pinned_gap, 1)} to "
        f"{_method_number(relaxed_gap, 1)} AIC units. Both figures are reported because "
        f"the strength of the structural evidence depends on that choice. What this "
        f"analysis measures is the distribution of inter-subunit distances, its width, "
        f"and which candidate shape reproduces it. It does not assign any observed pair "
        f"to a particular subunit pairing, does not identify individual assemblies, and "
        f"the reported numbers are not a count of detected complexes."
    )
    return text, []


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
    (re.compile(r"^HlyB staged short-range population \(3D\) on '(?P<name>.+?)':"),
     "analysis", _render_hlyb_staged_short_range),
    (re.compile(r"^HlyB pair-distance model fit(?: \((?P<dims>[23])D\))? "
                r"on '(?P<name>.+?)':"),
     "analysis", _render_hlyb_pair_fit),
    (re.compile(
        r"^HlyB subunit pair analysis \(template matching (?P<dims>[23])D\) on "
        r"'(?P<name>.+?)': "
        # the 2-D entry inserts its border-shrink counts before the arrow
        r"(?P<traces>[\d,]+) trace\(s\)"
        r"(?:, (?P<border>[\d,]+) border-excluded, (?P<interior>[\d,]+) interior)?"
        r" → (?P<subunits>[\d,]+) subunit\(s\) → "
        r"(?P<structures>[\d,]+) HlyB structure\(s\); (?P<pairs>[\d,]+) pair\(s\), "
        r"median distance (?P<median>[-+\w.]+) nm \(unit Ø (?P<dunit>[-+\d.]+) nm, "
        r"candidate edge (?P<edge>[-+\d.]+) nm, min loc/trace (?P<minloc>[\d,]+), "
        r"z-scale (?P<zscale>[-+\d.]+), template tol (?P<tolerance>[-+\d.]+) nm, "
        r"tested (?P<tested>[\d,]+) candidate\(s\), passed (?P<passed>[\d,]+), "
        r"overlap-rejected (?P<overlap>[\d,]+)\)"),
     "analysis", _render_hlyb_template3d),
    (re.compile(r"RIMF for '(?P<name>.+?)': (?P<value>[\d.]+)\s*(?:\((?P<note>[^)]*)\))?"),
     "analysis", _render_rimf),
    (re.compile(
        r"^Aggregated '(?P<source>.+?)' into '(?P<result>.+?)': "
        r"(?P<nin>[\d,]+) -> (?P<nout>[\d,]+) localizations; "
        r"photon threshold = (?P<thr>[\d.eE+-]+) photons per aggregated localization; "
        r"photon iterations = (?P<iters>\[[^\]]*\]) "
        r"\((?P<iter_source>[^)]*)\); position = photon-weighted centroid; "
        r"timestamp mode = (?P<time_mode>[a-z_]+); valid final localizations grouped "
        r"per trace in time order; trailing remainder retained\."),
     "transform", _render_aggregation),
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
