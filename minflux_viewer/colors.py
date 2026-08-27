"""Application-owned color settings and conversion helpers.

All configurable, non-colormap colors live in ``prefs["colors"]`` as JSON-
compatible ``[red, green, blue, alpha]`` lists.  UI modules read through the
helpers here so old/missing preference files always receive the same defaults.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

RGBA = tuple[int, int, int, int]

SOLID_COLOR_NAMES: tuple[str, ...] = (
    "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Orange",
    "White", "Gray", "Black",
)

SOLID_COLOR_LABELS: dict[str, str] = {
    "Red": "R", "Green": "G", "Blue": "B", "Cyan": "Cyan",
    "Magenta": "M", "Yellow": "Y", "Orange": "O", "White": "W",
    "Gray": "Gray", "Black": "K",
}

DEFAULT_SOLID_COLORS: dict[str, list[int]] = {
    "Red": [255, 0, 0, 255],
    "Green": [0, 255, 0, 255],
    "Blue": [0, 0, 255, 255],
    "Cyan": [0, 255, 255, 255],
    "Magenta": [255, 0, 255, 255],
    "Yellow": [255, 255, 0, 255],
    "Orange": [255, 128, 0, 255],
    "White": [255, 255, 255, 255],
    "Gray": [120, 120, 120, 255],
    "Black": [0, 0, 0, 255],
}

DEFAULT_COLOR_PREFS: dict = {
    "solid": copy.deepcopy(DEFAULT_SOLID_COLORS),
    "viewer": {
        "attribute_data": [70, 130, 180, 255],
        "attribute_background": [255, 255, 255, 255],
        "histogram_data": [70, 130, 180, 255],
        "histogram_background": [255, 255, 255, 255],
        "filter_range": [0, 255, 0, 115],
        "filter_bounds": [0, 255, 0, 255],
        "filter_text": [0, 255, 0, 255],
        "overlay": [
            [255, 0, 0, 255], [0, 255, 0, 255],
            [0, 0, 255, 255], [0, 255, 255, 255],
            [255, 0, 255, 255], [255, 255, 0, 255],
        ],
        # The drawn/draft ROI, split into the parts that are drawn separately.
        # Stored ROIs are styled by the "ROI Manager" group below.
        "roi_face": [255, 255, 0, 128],
        "roi_edge": [255, 255, 0, 255],
        "roi_corner": [0, 229, 255, 255],
        "roi_highlight": [255, 255, 0, 255],
    },
    "functions": {
        # A group may map row-name -> {item: color}; the COLOR dialog renders
        # each such mapping as its own labelled row instead of one flat list.
        "ROI Manager": {
            "ROI entries": {
                "face": [255, 255, 0, 128],
                "edge": [255, 255, 0, 255],
                "corner": [0, 229, 255, 255],
                "label": [255, 255, 0, 255],
            },
            "ROI selected": {
                "face": [0, 229, 255, 128],
                "edge": [0, 229, 255, 255],
                "corner": [0, 229, 255, 255],
                "label": [0, 229, 255, 255],
            },
        },
        # ``jet`` sampled at TEN points, blue -> red. Iterations are ordered, so
        # the colour must read as ordered.
        #
        # ⚠ The count is load-bearing: a MINFLUX sequence normally has ten
        # iterations, and consumers index this table by
        # ``round(k/(n-1) * (len-1))``. With ELEVEN stops over ten iterations
        # that expression skips stop 5 and shifts iterations 5..9 up by one, so
        # the drawn colours no longer matched the legend's -- the reported bug.
        # Ten stops make it the identity for the common case. Do not add a
        # stop here without re-checking that mapping.
        "Iteration series": {
            "1st": [0, 0, 127, 255], "2nd": [0, 5, 238, 255],
            "3rd": [0, 98, 255, 255], "4th": [0, 212, 255, 255],
            "5th": [76, 255, 168, 255], "6th": [168, 255, 76, 255],
            "7th": [255, 229, 0, 255], "8th": [255, 124, 0, 255],
            "9th": [238, 26, 0, 255], "10th": [127, 0, 0, 255],
        },
        "Localization precision": {
            "Lateral sigma": [34, 211, 211, 255],
            "Axial sigma": [255, 90, 90, 255],
            "CRLB lateral": [167, 139, 250, 255],
            "CRLB axial": [255, 90, 90, 255],
            "Photon count": [245, 166, 35, 255],
            "FRC curve": [74, 144, 255, 255],
            "FRC resolution": [53, 208, 127, 255],
        },
    },
    "plugins": {
        "Spatial Line Pattern": {
            "Total": [242, 242, 242, 255],
            "Positive": [0, 184, 217, 255],
            "Negative": [231, 84, 183, 255],
            "Centroid": [255, 159, 67, 255],
            "Autocorrelation": [141, 211, 95, 255],
        },
        "Drift Correction": {
            "X drift": [255, 90, 90, 255],
            "Y drift": [76, 175, 80, 255],
            "Z drift": [90, 155, 255, 255],
        },
        "Trace Viewer": {
            "Head marker": [255, 230, 40, 255],
            "Tail line": [230, 40, 200, 255],
            "Time region": [230, 40, 200, 60],
        },
    },
    # QColorDialog exposes a platform-defined number of custom slots.  Sixteen
    # is the common minimum; the dialog pads/truncates this list to customCount().
    "custom_palette": [
        [255, 255, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255],
        [0, 0, 255, 255], [0, 255, 255, 255], [255, 0, 255, 255],
        [255, 255, 0, 255], [255, 128, 0, 255], [128, 128, 128, 255],
        [0, 0, 0, 255], [128, 0, 0, 255], [0, 128, 0, 255],
        [0, 0, 128, 255], [0, 128, 128, 255], [128, 0, 128, 255],
        [128, 128, 0, 255],
    ],
}


def _clamp_channel(value) -> int:
    try:
        return max(0, min(255, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_rgba(value, fallback: Sequence[int] = (0, 0, 0, 255)) -> RGBA:
    """Return a clamped RGBA tuple from a sequence or a Qt-style hex string."""
    if isinstance(value, str):
        text = value.strip()
        named = DEFAULT_SOLID_COLORS.get(text.title())
        if named is not None:
            return tuple(named)  # type: ignore[return-value]
        if text.startswith("#"):
            raw = text[1:]
            try:
                if len(raw) == 6:                         # #RRGGBB
                    return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
                if len(raw) == 8:                         # app format: #RRGGBBAA
                    return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]
            except ValueError:
                pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = list(value)
        if len(vals) in (3, 4):
            if len(vals) == 3:
                vals.append(255)
            return tuple(_clamp_channel(v) for v in vals)  # type: ignore[return-value]
    return normalize_rgba(fallback, (0, 0, 0, 255)) if value is not fallback else (0, 0, 0, 255)


def rgba_hex(value, *, alpha: bool = True) -> str:
    """Serialize in the app-owned ``#RRGGBBAA`` (or ``#RRGGBB``) format."""
    r, g, b, a = normalize_rgba(value)
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}" if alpha else f"#{r:02x}{g:02x}{b:02x}"


def rgba_qt_hex(value) -> str:
    """Serialize as Qt's ``#AARRGGBB`` format.

    ⚠ Only for values handed to Qt itself (``QColor``, stylesheets).  PyQtGraph
    reads ``'#' + 8 hex digits`` as ``#RRGGBBAA``, so a string from here renders
    with the wrong color and, worse, the blue byte as alpha — opaque yellow
    ``#ffffff00`` becomes fully transparent.  Use :func:`rgba_hex` for anything
    that reaches ``pg.mkPen``/``mkColor``/``TextItem``.
    """
    r, g, b, a = normalize_rgba(value)
    return f"#{a:02x}{r:02x}{g:02x}{b:02x}"


def pg_safe_hex(value) -> str:
    """A stroke/label color PyQtGraph and :func:`normalize_rgba` agree on.

    Repairs the ``#AARRGGBB`` strings written before the app settled on
    ``#RRGGBBAA``.  The two cannot be told apart by shape — both are ``'#'``
    plus eight hex digits — so the tell is the alpha: PyQtGraph takes the last
    byte, and a *fully transparent* ROI stroke was never something a user asked
    for.  Those are re-read the Qt way; every other value passes through.
    """
    if isinstance(value, str) and len(value.strip()) == 9 and value.strip().startswith("#"):
        text = value.strip()
        if normalize_rgba(text)[3] == 0:
            return rgba_hex(rgba_from_qt_hex(text))
    return rgba_hex(normalize_rgba(value))


def rgba_from_qt_hex(value, fallback: Sequence[int] = (0, 0, 0, 255)) -> RGBA:
    """Read legacy names/#RRGGBB and Qt ``#AARRGGBB`` record colors."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 9:
            try:
                raw = text[1:]
                a, r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4, 6))
                return r, g, b, a
            except ValueError:
                pass
    return normalize_rgba(value, fallback)


def parse_hex_rgba(text, *, default_alpha: int = 255):
    """A hand-typed or pasted hex colour -> ``(r, g, b, a)``, or ``None``.

    ``QColor(str)`` is stricter than a user is: it rejects ``FF8000`` outright
    because there is no leading ``#``, which is exactly the form a hex code
    copied from a web page or a paper figure arrives in -- so the COLOR
    dialog's HEX field silently reverted whatever was pasted into it.

    Accepted: an optional ``#`` or ``0x`` prefix, and 3 (``RGB``), 4 (``RGBA``),
    6 (``RRGGBB``) or 8 hex digits.

    ⚠ Eight digits are read as **RRGGBBAA**, the CSS convention an external
    source uses -- *not* Qt's ``#AARRGGBB``, which would silently turn a pasted
    ``FF8000FF`` into a different colour. ``rgba_from_qt_hex`` remains the
    reader for this application's own stored record colours.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip().lstrip("#").strip()
    if raw[:2].lower() == "0x":
        raw = raw[2:]
    if len(raw) not in (3, 4, 6, 8) or any(c not in "0123456789abcdefABCDEF" for c in raw):
        return None
    if len(raw) in (3, 4):                       # shorthand: each digit doubled
        raw = "".join(c * 2 for c in raw)
    values = [int(raw[i:i + 2], 16) for i in range(0, len(raw), 2)]
    if len(values) == 3:
        values.append(int(default_alpha))
    return tuple(values)


def _merge(saved, defaults):
    result = copy.deepcopy(defaults)
    if not isinstance(saved, Mapping):
        return result
    for key, value in saved.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _merge(value, result[key])
        else:
            result[key] = copy.deepcopy(value)
    return result


def normalize_color_preferences(value) -> dict:
    """Merge defaults and normalize every color leaf to an RGBA list."""
    saved_solid = value.get("solid") if isinstance(value, Mapping) else None
    result = _merge(value, DEFAULT_COLOR_PREFS)
    # Once a solid registry exists it is authoritative: restoring missing
    # defaults here would make user deletion impossible.  JSON/Python mappings
    # retain insertion order, which is also the order used by every solid menu.
    if isinstance(saved_solid, Mapping):
        solid: dict[str, list[int]] = {}
        seen: set[str] = set()
        for raw_name, color in saved_solid.items():
            name = " ".join(str(raw_name).strip().split())
            key = name.casefold()
            if not name or key in seen or key.startswith("custom:"):
                continue
            solid[name] = list(normalize_rgba(color))
            seen.add(key)
        result["solid"] = solid
    else:
        result["solid"] = copy.deepcopy(DEFAULT_SOLID_COLORS)
    # ⚠ The iteration palette's SIZE is part of its contract, not just its
    # colours: ``_iter_color`` indexes ``round(k/(n-1) * (len-1))``, so a stop
    # the defaults no longer declare makes that mapping skip one and mis-colour
    # the last iterations. ``_merge`` keeps a saved key the defaults lack, so a
    # customised palette would carry a retired stop forever; prune it here as
    # well as in the one-shot migration, so it can never come back.
    functions = result.setdefault("functions", {})
    series = functions.get("Iteration series")
    declared = DEFAULT_COLOR_PREFS["functions"]["Iteration series"]
    if isinstance(series, Mapping):
        functions["Iteration series"] = {
            name: color for name, color in series.items() if name in declared
        }

    viewer = result.setdefault("viewer", {})
    # The single pre-split ROI color seeds face/edge/highlight, so an existing
    # preference carries over instead of silently reverting to the default.
    # Tested against the *saved* dict: the merge above has already filled the
    # split keys with defaults, so they are never absent from ``result``.
    saved_viewer = value.get("viewer") if isinstance(value, Mapping) else None
    saved_viewer = saved_viewer if isinstance(saved_viewer, Mapping) else {}
    viewer.pop("roi", None)
    if "roi" in saved_viewer:
        r, g, b, _a = normalize_rgba(saved_viewer["roi"])
        for key, alpha in (("roi_face", 128), ("roi_edge", 255), ("roi_highlight", 255)):
            if key not in saved_viewer:
                viewer[key] = [r, g, b, alpha]
    for key, default in DEFAULT_COLOR_PREFS["viewer"].items():
        if key == "overlay":
            source = viewer.get(key, default)
            viewer[key] = [
                list(normalize_rgba(source[i] if isinstance(source, Sequence) and i < len(source) else item, item))
                for i, item in enumerate(default)
            ]
        else:
            viewer[key] = list(normalize_rgba(viewer.get(key), default))
    for section in ("functions", "plugins"):
        groups = result.setdefault(section, {})
        for group, components in list(groups.items()):
            if not isinstance(components, Mapping):
                groups[group] = {}
                continue
            defaults = DEFAULT_COLOR_PREFS.get(section, {}).get(group, {})

            def _norm(items, fallbacks):
                """Normalize a group, one level of row-nesting deep."""
                out: dict = {}
                for name, color in items.items():
                    fallback = fallbacks.get(name, {}) if isinstance(fallbacks, Mapping) else {}
                    if isinstance(color, Mapping):
                        out[str(name)] = _norm(color, fallback)
                    else:
                        default = fallback if not isinstance(fallback, Mapping) else (0, 0, 0, 255)
                        out[str(name)] = list(normalize_rgba(color, default))
                return out

            groups[group] = _norm(components, defaults)
    palette = result.get("custom_palette", [])
    if not isinstance(palette, Sequence) or isinstance(palette, (str, bytes)):
        palette = []
    result["custom_palette"] = [list(normalize_rgba(item)) for item in palette]
    return result


def color_preferences(prefs: Mapping | None) -> dict:
    return normalize_color_preferences((prefs or {}).get("colors", {}))


def viewer_color(prefs: Mapping | None, key: str) -> RGBA:
    colors = color_preferences(prefs)
    default = DEFAULT_COLOR_PREFS["viewer"].get(key, (0, 0, 0, 255))
    return normalize_rgba(colors["viewer"].get(key), default)


def overlay_colors(prefs: Mapping | None) -> list[RGBA]:
    return [normalize_rgba(item) for item in color_preferences(prefs)["viewer"]["overlay"]]


def component_colors(prefs: Mapping | None, section: str, group: str) -> dict:
    """A group's colors; a row-nested group keeps its ``{row: {item: rgba}}`` shape."""
    values = color_preferences(prefs).get(section, {}).get(group, {})

    def _walk(items) -> dict:
        return {
            str(name): _walk(color) if isinstance(color, Mapping) else normalize_rgba(color)
            for name, color in items.items()
        }

    return _walk(values)


_RUNTIME_SOLID_COLORS: dict[str, RGBA] = {
    name: normalize_rgba(color) for name, color in DEFAULT_SOLID_COLORS.items()
}
_RUNTIME_SOLID_NAMES_BY_KEY: dict[str, str] = {
    name.casefold(): name for name in _RUNTIME_SOLID_COLORS
}
_RUNTIME_COMPONENT_COLORS: dict[str, dict[str, dict[str, RGBA]]] = {}


def runtime_component_colors(section: str, group: str) -> dict[str, RGBA]:
    """Current colors for a group, without threading ``prefs`` to the caller.

    ``component_colors`` needs the preferences dict, which analysis modules that
    only draw a plot do not otherwise carry.  This reads the same values from
    the process-wide cache ``configure_colors`` refreshes, falling back to the
    declared defaults so a caller never gets an empty mapping.
    """
    cached = _RUNTIME_COMPONENT_COLORS.get(section, {}).get(group)
    if cached:
        return dict(cached)
    declared = DEFAULT_COLOR_PREFS.get(section, {}).get(group, {})
    return {str(name): normalize_rgba(color) for name, color in declared.items()}


def configure_colors(prefs: Mapping | None) -> None:
    """Refresh the derived process-wide palettes for the one app instance."""
    global _RUNTIME_SOLID_COLORS, _RUNTIME_SOLID_NAMES_BY_KEY
    global _RUNTIME_COMPONENT_COLORS
    resolved = color_preferences(prefs)
    _RUNTIME_COMPONENT_COLORS = {
        section: {
            str(group): {
                str(name): normalize_rgba(color) for name, color in items.items()
            }
            for group, items in resolved.get(section, {}).items()
        }
        for section in ("functions", "plugins")
    }
    values = resolved["solid"]
    _RUNTIME_SOLID_COLORS = {
        str(name): normalize_rgba(color)
        for name, color in values.items()
    }
    _RUNTIME_SOLID_NAMES_BY_KEY = {
        name.casefold(): name for name in _RUNTIME_SOLID_COLORS
    }


def solid_color_names() -> tuple[str, ...]:
    """Current user-ordered solid-color names."""
    return tuple(_RUNTIME_SOLID_COLORS)


def canonical_solid_color_name(name: str) -> str:
    """Return the current spelling of a solid name, case-insensitively."""
    canonical = _RUNTIME_SOLID_NAMES_BY_KEY.get(str(name).strip().casefold())
    if canonical is None:
        raise KeyError(f"Unknown solid color: {name}")
    return canonical


def is_solid_color(name: str) -> bool:
    # Exact spelling preserves the long-standing distinction between the
    # solid ``Gray`` and the intensity colormap ``gray``.
    return str(name).strip() in _RUNTIME_SOLID_COLORS


def solid_color_rgba(name: str) -> RGBA:
    try:
        canonical = canonical_solid_color_name(name)
    except KeyError:
        return 120, 120, 120, 255
    return _RUNTIME_SOLID_COLORS[canonical]


def solid_color_rgb(name: str) -> tuple[int, int, int]:
    return solid_color_rgba(name)[:3]


def changed_color_paths(previous, current) -> set[str]:
    """Dotted paths whose color leaves changed (list indices included)."""
    changed: set[str] = set()

    def walk(a, b, path: tuple[str, ...]) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in set(a) | set(b):
                walk(a.get(key), b.get(key), (*path, str(key)))
            return
        if isinstance(a, list) and isinstance(b, list):
            # RGBA lists are leaves; other lists (overlay/palette) are containers.
            if len(a) in (3, 4) and len(b) in (3, 4) and all(
                not isinstance(item, (list, dict)) for item in (*a, *b)
            ):
                if normalize_rgba(a) != normalize_rgba(b):
                    changed.add(".".join(path))
                return
            for index in range(max(len(a), len(b))):
                walk(a[index] if index < len(a) else None,
                     b[index] if index < len(b) else None,
                     (*path, str(index)))
            return
        if a != b:
            changed.add(".".join(path))

    walk(normalize_color_preferences(previous), normalize_color_preferences(current), ())
    return changed
