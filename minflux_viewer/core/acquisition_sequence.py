"""MINFLUX acquisition sequences: recognising, validating and describing one.

A *sequence* is the JSON document that defines a MINFLUX measurement: a list
of iterations (``Itr``) that progressively tighten the targeted coordinate
pattern, plus sequence-wide settings (``id``, ``headstart``, ``bgcSense``, ...).
Imspector reads it from its sequence folder, and records the one it ran with
the data: ``mfx/.zattrs -> measurement.threads[].sequences[]`` in an m2410
file, ``mfx/.zattrs -> seq`` in an m2205 one -- one sequence per measurement
thread (the imaging thread on the ``search_0`` grid, the beam-line-monitoring
thread on ``mbm``). A standalone sequence file has the same schema.

This module is the display side; ``core.mfx_sequence`` keeps the analysis side
(photon-bearing iterations, pattern geometry for the CRB) and is reused here.
Parameter meanings follow the Abberior wiki pages *MINFLUX Sequence*,
*Targeted coordinate patterns* and *Guidelines for MINFLUX Methods Sections*;
the wording is our own. A value or key the pages do not document is shown as
recorded, never given an invented meaning.

Pure Python, Qt-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .mfx_sequence import (
    PATTERN_SCALE_NM,
    REFERENCE_WAVELENGTH_M,
    detect_from_sequence,
    parse_sequence,
    wavelength_scale,
)

__all__ = [
    "REFERENCE_WAVELENGTH_M",
    "DEFAULT_MAX_OFF_TIME_S",
    "ITERATION_COLUMNS",
    "SequenceSource",
    "IterationRow",
    "SequenceOverview",
    "describe_key",
    "is_sequence_payload",
    "is_sequence_json_file",
    "validate_sequence",
    "load_sequence_file",
    "pattern_diameters_nm",
    "pattern_role",
    "sequence_overview",
    "sequences_from_mfx_attrs",
    "sequences_for_dataset",
]

#: Imspector's default when an iteration sets no ``maxOffTime``.
DEFAULT_MAX_OFF_TIME_S = 0.003

_SEQUENCE_DESCRIPTIONS = {
    "id": "Sequence name, as Imspector lists it (the file name is not shown there).",
    "Itr": "The iterations, run in order; each tightens the targeted pattern.",
    "bgcSense": ("Background sensing: when on, the background is estimated during the "
                 "measurement and bgcThreshold is an offset above it."),
    "ctrDwellFactor": ("Time spent at the pattern centre for a CFR check, as a multiple of "
                       "the iteration's dwell time."),
    "damping": "Galvo scanner damping for tracking: each step moves 1/2^damping of the distance.",
    "headstart": ("Where the loop resumes after a final localization: a negative value goes "
                  "back that many iterations from the last; a positive value names an iteration."),
    "locLimit": "Valid final localizations after which a trace stops; -1 means no limit.",
    "stickiness": ("Attempts to localize again after a failed threshold before the trace "
                   "stops (not applied to the first iteration)."),
    "liveview": "Renderings Imspector opens when the measurement starts (deprecated in version 2410).",
    "schedule": ("Minimum time between starting localizations; the sample is not illuminated "
                 "while waiting."),
    "maxOffTime": ("Longest time a fluorophore may stay below the background threshold before "
                   "the localization is ended, in seconds. Documented per iteration; when an "
                   "iteration sets none, Imspector uses 3 ms."),
}
_GRID_DESCRIPTIONS = {
    "field": "Search grid: algo is the grid shape, fldGeoFactor sets the grid spacing.",
}
_ITERATION_DESCRIPTIONS = {
    "Mode": "Iteration type: pattern, dimensionality and the excitation/detection set-up.",
    "patGeoFactor": ("Targeted pattern diameter L as a factor: "
                     "L = patGeoFactor x 360 nm x wavelength / 642 nm."),
    "phtLimit": "Minimum photon count to pass the iteration.",
    "ccrLimit": ("CFR check: -1 = not measured; up to 1 = measured and localizations above "
                 "it rejected; above 1 = measured, not filtered."),
    "bgcThreshold": ("Background threshold in Hz; an offset above the sensed background when "
                     "bgcSense is on."),
    "patDwellTime": "Dwell time of the pattern, in seconds.",
    "patRepeat": "Pattern repeats within one dwell time (averaging).",
    "pwrFactor": "Excitation power relative to the power set when the measurement starts.",
    "wavelength": ("Reference wavelength for the pattern size, in metres; normally the 642 nm "
                   "laser."),
    "maxOffTime": _SEQUENCE_DESCRIPTIONS["maxOffTime"],
    "estCoeff": "Position estimator coefficients (not edited by users).",
    "estCoeffL": "Lateral estimator coefficients.",
    "estCoeffA": "Axial estimator coefficients.",
    "_lemd": "Median residual error after estimator coefficient optimization.",
    "_lemn": "Mean residual error after estimator coefficient optimization.",
    "_excitation": ("Excitation used, recorded by Imspector at acquisition: laser, wavelength, "
                    "power, DAC setting."),
    "_detection": "Detection channels used, recorded by Imspector at acquisition.",
    "_activation": "Activation laser used, if any, recorded by Imspector at acquisition.",
}
_MODE_DESCRIPTIONS = {
    "id": ("Iteration type: probing with the galvo scanners (prbt) or MINFLUX with the "
           "electro-optic deflectors (mflx and its variants)."),
    "dim": "Dimensions localized: 2 or 3.",
    "pattern": "Targeted coordinate pattern.",
    "phDiaAU": "Pinhole diameter, in Airy units.",
    "strategy": "Strategy used when the estimator coefficients were optimized.",
}
_PATTERN_DESCRIPTIONS = {
    "hexagon": "six positions; 2-D imaging and precise tracking",
    "triangle": "three positions; fast tracking",
    "square": "four positions; lateral iterations of 3-D imaging",
    "zline": "four positions along z; first axial localization",
    "zline2": "two positions along z; axial localizations after the first",
    "octahedron": "lateral and axial positions in one pattern; 3-D tracking",
}
_LATERAL = {"hexagon", "triangle", "square"}
_AXIAL = {"zline", "zline2", "zline3"}
_BOTH = {"octahedron"}


def describe_key(path: tuple[str, ...]) -> str:
    """Description of the sequence key at *path* (e.g. ``("Itr", "[3]", "Mode",
    "pattern")``), or "" when the key is not documented."""
    names = [p for p in path if not (p.startswith("[") and p.endswith("]")) and p != "root"]
    if not names:
        return ""
    key = names[-1]
    parent = names[-2] if len(names) > 1 else ""
    if parent == "Mode":
        return _MODE_DESCRIPTIONS.get(key, "")
    if parent == "Itr":
        return _ITERATION_DESCRIPTIONS.get(key, "")
    if len(names) == 1:
        return _SEQUENCE_DESCRIPTIONS.get(key, "") or _GRID_DESCRIPTIONS.get(key, "")
    return ""


# ---------------------------------------------------------------- recognition
def is_sequence_payload(payload: Any) -> bool:
    """Recognition, not validation: an object with an ``Itr`` list that is not
    a processing sidecar or a ROI set (which win when markers conflict)."""
    from .save import METADATA_JSON_MARKER

    return (isinstance(payload, dict) and isinstance(payload.get("Itr"), list)
            and METADATA_JSON_MARKER not in payload and "rois" not in payload)


def is_sequence_json_file(source) -> bool:
    """Whether *source* (a path or a ``JsonProbe``) is a sequence file.

    Decided from the bounded probe: a sequence is a top-level object with an
    ``Itr`` key. A damaged sequence (an iteration without ``Mode``, say) is
    still recognised here, so its loader can say what is wrong with it rather
    than the localization loader saying what it is not.
    """
    from .json_probe import as_probe
    from .save import METADATA_JSON_MARKER

    try:
        probe = as_probe(source)
        if not probe.conclusive:
            return False                 # formats.resolve_open deepens it first
        return (bool(probe.has_key("Itr")) and probe.kind("Itr") == "array"
                and not probe.has_key(METADATA_JSON_MARKER)
                and not probe.has_key("rois")
                and not probe.wraps_localizations)
    except (OSError, TypeError, ValueError):
        return False


def validate_sequence(sequence: Any, *, name: str = "sequence") -> dict:
    """*sequence* itself if it is a usable sequence, else ``ValueError`` saying why."""
    if not isinstance(sequence, dict):
        raise ValueError(f"{name}: a sequence is a JSON object, not {type(sequence).__name__}.")
    iterations = sequence.get("Itr")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError(f"{name}: 'Itr' must be a non-empty list of iterations.")
    for number, iteration in enumerate(iterations, start=1):
        if not isinstance(iteration, dict):
            raise ValueError(f"{name}: iteration {number} is not an object.")
        mode = iteration.get("Mode")
        if not isinstance(mode, dict):
            raise ValueError(f"{name}: iteration {number} has no 'Mode'.")
        if "id" not in mode:
            raise ValueError(f"{name}: iteration {number} has a 'Mode' without an 'id'.")
    return sequence


def load_sequence_file(path) -> dict:
    """Read and validate a sequence ``.json``; ``ValueError`` names the problem."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: not valid JSON ({exc}).") from None
    return validate_sequence(payload, name=path.name)


# --------------------------------------------------------------- the sources
@dataclass(frozen=True)
class SequenceSource:
    """One sequence to show, with where it came from."""

    label: str                                   # selector text
    sequence: dict
    origin: str                                  # file path or dataset name
    details: tuple[tuple[str, str], ...] = ()    # thread facts (grid, tag, ...)


def _thread_details(thread: dict) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for key, label in (("id", "Thread"), ("grid", "Grid"), ("tag", "Tag"),
                       ("dimensionality", "Dimensionality")):
        value = thread.get(key)
        if value not in (None, ""):
            rows.append((label, str(value)))
    detectors = [d for d in thread.get("detectors") or [] if isinstance(d, dict)
                 and d.get("active_use", True)]
    if detectors:
        rows.append(("Detectors", ", ".join(
            f"{d.get('id', '?')}" + (f" ({d['expcontrol_name']})" if d.get("expcontrol_name") else "")
            for d in detectors)))
    lasers = [d for d in thread.get("lasers") or [] if isinstance(d, dict)
              and d.get("active_use", True)]
    if lasers:
        rows.append(("Lasers", ", ".join(
            f"{d.get('id', '?')}"
            + (f" ({float(d['wavelength']) * 1e9:.0f} nm)"
               if isinstance(d.get("wavelength"), (int, float)) else "")
            for d in lasers)))
    return tuple(rows)


def sequences_from_mfx_attrs(attrs: Any, *, origin: str = "") -> list[SequenceSource]:
    """Every sequence recorded in a native ``mfx/.zattrs`` record, per thread."""
    out: list[SequenceSource] = []
    if not isinstance(attrs, dict):
        return out
    threads = (attrs.get("measurement") or {}).get("threads") or []
    for t_index, thread in enumerate(threads):
        if not isinstance(thread, dict):
            continue
        for s_index, sequence in enumerate(thread.get("sequences") or []):
            if not (isinstance(sequence, dict) and isinstance(sequence.get("Itr"), list)):
                continue
            thread_id = thread.get("id", t_index)
            label = f"Thread {thread_id}: {sequence.get('id', 'sequence')}"
            if thread.get("grid"):
                label += f" (grid {thread['grid']})"
            if s_index:
                label += f" #{s_index + 1}"
            out.append(SequenceSource(label, sequence, origin, _thread_details(thread)))
    seq = attrs.get("seq")
    if not out and isinstance(seq, dict) and isinstance(seq.get("Itr"), list):
        out.append(SequenceSource(str(seq.get("id", "sequence")), seq, origin))
    return out


def sequences_for_dataset(ds) -> list[SequenceSource]:
    """The sequences a loaded dataset carries: every thread of its native record,
    else the one stamped at import. Empty when the file recorded none."""
    meta = getattr(ds, "metadata", {}) or {}
    origin = str(getattr(ds, "name", "") or "")
    found = sequences_from_mfx_attrs(meta.get("native_zarr_mfx_attrs"), origin=origin)
    if found:
        return found
    seq = meta.get("mfx_sequence")
    if isinstance(seq, dict) and isinstance(seq.get("Itr"), list):
        return [SequenceSource(str(seq.get("id", "sequence")), seq, origin)]
    return []


# ------------------------------------------------------------------ overview
#: Iteration table columns: (key, header).
ITERATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("itr", "Iteration"),
    ("mode", "Mode"),
    ("dim", "Dim"),
    ("pattern", "Pattern"),
    ("role", "Localizes"),
    ("geo", "patGeoFactor"),
    ("l_nm", "L (nm)"),
    ("pht", "Photon limit"),
    ("dwell", "Dwell (µs)"),
    ("repeat", "Repeats"),
    ("power", "Power factor"),
    ("ccr", "CFR check"),
    ("bgc", "Background threshold"),
    ("max_off", "Max off-time"),
    ("final", "Final for"),
    ("excitation", "Excitation"),
    ("detection", "Detection"),
)


@dataclass(frozen=True)
class IterationRow:
    index: int                      # 0-based, matches ``itr`` in the data
    cells: dict
    final: str = ""                 # "lateral" / "axial" / "lateral + axial" / ""
    in_loop: bool = False           # repeated after a final localization


@dataclass(frozen=True)
class SequenceOverview:
    name: str
    globals: tuple[tuple[str, str, str], ...]     # (label, value, key)
    iterations: tuple[IterationRow, ...]
    loop_start: int | None                        # 0-based, from a negative headstart


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) else None


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(", ", ": "))
    return str(value)


def pattern_role(pattern: str) -> str:
    """What a pattern localizes, from its name only: unknown names are unknown."""
    name = (pattern or "").lower()
    if name in _BOTH:
        return "lateral + axial"
    if name in _LATERAL:
        return "lateral"
    if name in _AXIAL:
        return "axial"
    return "unknown"


def pattern_diameters_nm(geo_factor, wavelength_m) -> list[float]:
    """L for a scalar or list ``patGeoFactor`` at the recorded wavelength.

    ``L = patGeoFactor x 360 nm x wavelength / 642 nm``
    (:func:`~minflux_viewer.core.mfx_sequence.wavelength_scale`); a missing
    wavelength is taken as the 642 nm reference.
    """
    scale = wavelength_scale(wavelength_m)
    factors = geo_factor if isinstance(geo_factor, (list, tuple)) else [geo_factor]
    out = []
    for factor in factors:
        value = _number(factor)
        if value is not None:
            out.append(value * PATTERN_SCALE_NM * scale)
    return out


def _ccr_text(value) -> str:
    number = _number(value)
    if number is None:
        return _text(value)
    if number == -1:
        return "not measured"
    if 0 < number <= 1:
        return f"measured, > {number:g} rejected"
    if number > 1:
        return f"measured, not filtered ({number:g})"
    return f"{number:g} (undocumented value)"


def _max_off_text(iteration: dict) -> str:
    value = iteration.get("maxOffTime")
    number = _number(value)
    if number is not None:
        return f"{number * 1e3:g} ms"
    if value not in (None, ""):
        return f"{_text(value)} (as recorded)"
    return f"default {DEFAULT_MAX_OFF_TIME_S * 1e3:g} ms (not set)"


def _final_iterations(sequence: dict) -> dict[int, str]:
    """Final lateral/axial iterations, from patterns whose role is documented.

    An iteration with an unknown pattern is left out rather than assumed
    lateral (the CRB parser's fallback), so nothing is shown as fact that the
    sequence does not say.
    """
    known = [it for it in parse_sequence(sequence) if pattern_role(it.pattern) != "unknown"]
    for it in known:
        role = pattern_role(it.pattern)
        it.is_lateral = role in ("lateral", "lateral + axial")
        it.is_axial = role in ("axial", "lateral + axial")
    info = detect_from_sequence(known)
    out: dict[int, str] = {}
    if info is None:
        return out
    if info.lateral_iter is not None:
        out[info.lateral_iter] = "lateral"
    if info.axial_iter is not None:
        out[info.axial_iter] = ("lateral + axial" if out.get(info.axial_iter) == "lateral"
                                else "axial")
    return out


def sequence_overview(sequence: dict) -> SequenceOverview:
    """Everything the viewer shows, as plain text, from a validated sequence."""
    iterations = [it for it in sequence.get("Itr") or [] if isinstance(it, dict)]
    n = len(iterations)
    headstart = _number(sequence.get("headstart"))
    loop_start = None
    if headstart is not None and headstart < 0 and n:
        loop_start = max(0, int(n - 1 + headstart))
    bgc_sense = sequence.get("bgcSense")

    rows: list[tuple[str, str, str]] = [
        ("Sequence id", _text(sequence.get("id")), "id"),
        ("Iterations", str(n), "Itr"),
    ]
    dims = sorted({str((it.get("Mode") or {}).get("dim")) for it in iterations
                   if (it.get("Mode") or {}).get("dim") is not None})
    if dims:
        rows.append(("Dimensions", " / ".join(f"{d}-D" for d in dims), ""))
    for key, label in (("bgcSense", "Background sensing"), ("ctrDwellFactor", "Centre dwell factor"),
                       ("damping", "Damping"), ("stickiness", "Stickiness")):
        if key in sequence:
            rows.append((label, _text(sequence[key]), key))
    if "headstart" in sequence:
        value = _text(sequence["headstart"])
        if loop_start is not None:
            # Counted back from the last iteration, which is how the wiki's
            # 2-D/3-D imaging examples read; its tracking remark ("-1 ... only
            # the final iteration") reads differently, so it is labelled as derived.
            value += (f"  (after a final localization the loop returns to iteration "
                      f"{loop_start + 1}, counted as in the wiki's imaging examples)")
        elif headstart is not None and headstart >= 0:
            value += "  (an absolute iteration; how Imspector counts it is not documented)"
        rows.append(("Headstart", value, "headstart"))
    if "locLimit" in sequence:
        limit = _number(sequence["locLimit"])
        rows.append(("Localization limit",
                     "no limit" if limit == -1 else _text(sequence["locLimit"]), "locLimit"))
    field = sequence.get("field")
    if isinstance(field, dict):
        rows.append(("Search grid", ", ".join(f"{k} {_text(v)}" for k, v in field.items()),
                     "field"))
    for key, label in (("maxOffTime", "Max off-time (sequence-wide)"), ("schedule", "Schedule"),
                       ("liveview", "Live view (deprecated in 2410)"),
                       ("defaultField", "Default field"), ("system", "System")):
        if key in sequence:
            rows.append((label, _text(sequence[key]), key))
    shown = {"id", "Itr", "bgcSense", "ctrDwellFactor", "damping", "stickiness", "headstart",
             "locLimit", "field", "maxOffTime", "schedule", "liveview", "defaultField", "system"}
    for key in sequence:
        if key not in shown:
            rows.append((str(key), _text(sequence[key]), str(key)))

    finals = _final_iterations(sequence)
    table: list[IterationRow] = []
    for index, it in enumerate(iterations):
        mode = it.get("Mode") or {}
        pattern = str(mode.get("pattern", "") or "")
        diameters = pattern_diameters_nm(it.get("patGeoFactor"), it.get("wavelength"))
        wavelength = _number(it.get("wavelength"))
        l_text = " / ".join(f"{d:.1f}" for d in diameters)
        if wavelength is not None and abs(wavelength - REFERENCE_WAVELENGTH_M) > 1e-12:
            l_text += f" (at {wavelength * 1e9:.0f} nm)"
        dwell = _number(it.get("patDwellTime"))
        bgc = _number(it.get("bgcThreshold"))
        bgc_text = _text(it.get("bgcThreshold")) if bgc is None else f"{bgc:g} Hz"
        if bgc is not None and bgc_sense is True:
            bgc_text += " above sensed background"
        raw_excitation, raw_detection = it.get("_excitation"), it.get("_detection")
        excitation: dict = raw_excitation if isinstance(raw_excitation, dict) else {}
        detection: dict = raw_detection if isinstance(raw_detection, dict) else {}
        cells = {
            "itr": str(index + 1),
            "mode": _text(mode.get("id")),
            "dim": _text(mode.get("dim")),
            "pattern": pattern,
            "role": pattern_role(pattern),
            "geo": _text(it.get("patGeoFactor")),
            "l_nm": l_text,
            "pht": _text(it.get("phtLimit")),
            "dwell": "" if dwell is None else f"{dwell * 1e6:g}",
            "repeat": _text(it.get("patRepeat")),
            "power": _text(it.get("pwrFactor")),
            "ccr": _ccr_text(it.get("ccrLimit")),
            "bgc": bgc_text,
            "max_off": _max_off_text(it),
            "final": finals.get(index, ""),
            "excitation": ", ".join(
                _text(excitation[k]) for k in ("laser", "power") if k in excitation),
            "detection": ", ".join(map(str, detection.get("channels") or [])),
        }
        table.append(IterationRow(index=index, cells=cells, final=finals.get(index, ""),
                                  in_loop=loop_start is not None and index >= loop_start))
    return SequenceOverview(name=_text(sequence.get("id")) or "sequence", globals=tuple(rows),
                            iterations=tuple(table), loop_start=loop_start)


def pattern_description(pattern: str) -> str:
    return _PATTERN_DESCRIPTIONS.get((pattern or "").lower(), "")
