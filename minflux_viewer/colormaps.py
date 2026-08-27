"""Application-owned colormaps for every MINFLUX Viewer display.

The project deliberately does not delegate named maps to optional third-party
map collections.  This module owns the color stops and returns
:class:`pyqtgraph.ColorMap` instances to the UI.

Custom maps are application preferences.  Their JSON-compatible definitions
live in ``prefs["plot"]["custom_colormaps"]``; the registry below is only the
derived runtime cache.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg

from .colors import (
    canonical_solid_color_name,
    is_solid_color,
    solid_color_names,
    solid_color_rgba,
)

# These are the intentionally small, user-facing set.  Each has a distinct job:
# intensity/density, rainbow contrast, diverging residuals, categorical labels,
# perceptually uniform quantitative values, high-contrast response images, and
# neutral grayscale respectively.
BUILTIN_COLORMAP_NAMES: tuple[str, ...] = (
    "hot",
    "jet",
    "HiLo",
    "glasbey",
    "viridis",
    "inferno",
    "parula",
    "gray",
)

# Previously exposed maps remain resolvable so saved datasets/preferences do not
# change appearance or fail to open.  They are intentionally omitted from new
# menus because their roles overlap the focused set above.
LEGACY_COLORMAP_NAMES: tuple[str, ...] = (
    "turbo",
    "magma",
    "plasma",
    "cividis",
)

# Black-to-color LUT endpoints. These intentionally use saturated channel
# primaries; ``PURE_COLOR_RGB`` remains the softer flat overlay/scatter palette.
PURE_COLORMAP_RGB: dict[str, tuple[int, int, int]] = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 128, 0),
    "White": (255, 255, 255),
    "Gray": (166, 166, 166),
    "Black": (0, 0, 0),
}

_MAX_CUSTOM_MAPS = 100
_MAX_CUSTOM_STOPS = 64


def _rgb_stops(values: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(values, dtype=np.uint8)
    rgba = np.column_stack(
        [rgb, np.full(rgb.shape[0], 255, dtype=np.uint8)]
    )
    return np.linspace(0.0, 1.0, rgba.shape[0], dtype=np.float64), rgba


# Compact application-owned control points.  PyQtGraph interpolates these into
# lookup tables of any requested length.
_CONTROL_POINTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "hot": _rgb_stops([
        [10, 0, 0], [55, 0, 0], [99, 0, 0], [144, 0, 0],
        [189, 0, 0], [233, 0, 0], [255, 23, 0], [255, 68, 0],
        [255, 112, 0], [255, 157, 0], [255, 201, 0], [255, 246, 0],
        [255, 255, 54], [255, 255, 121], [255, 255, 188], [255, 255, 255],
    ]),
    "jet": _rgb_stops([
        [0, 0, 127], [0, 0, 204], [0, 8, 255], [0, 76, 255],
        [0, 144, 255], [0, 212, 255], [41, 255, 205], [95, 255, 150],
        [150, 255, 95], [205, 255, 41], [255, 229, 0], [255, 166, 0],
        [255, 103, 0], [255, 40, 0], [204, 0, 0], [127, 0, 0],
    ]),
    "hilo": _rgb_stops([
        [0, 0, 255], [35, 35, 35], [255, 255, 255], [255, 0, 0],
    ]),
    "glasbey": _rgb_stops([
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
    ]),
    "viridis": _rgb_stops([
        [68, 1, 84], [72, 26, 108], [71, 47, 125], [65, 68, 135],
        [57, 86, 140], [49, 104, 142], [42, 120, 142], [35, 136, 142],
        [31, 152, 139], [34, 168, 132], [53, 183, 121], [84, 197, 104],
        [122, 209, 81], [165, 219, 54], [210, 226, 27], [253, 231, 37],
    ]),
    "inferno": _rgb_stops([
        [0, 0, 4], [12, 8, 38], [36, 12, 79], [66, 10, 104],
        [93, 18, 110], [120, 28, 109], [147, 38, 103], [174, 48, 92],
        [199, 62, 76], [221, 81, 58], [237, 105, 37], [248, 133, 15],
        [252, 165, 10], [250, 198, 45], [242, 230, 97], [252, 255, 164],
    ]),
    "gray": _rgb_stops([[0, 0, 0], [255, 255, 255]]),
    # Sampled at 64 points from a 256-entry MATLAB parula table, which keeps
    # every channel within 2/255 of it -- imperceptible, and a tenth of the
    # source lines. (The ten-point approximation this replaced was off by up
    # to 29/255, visibly flattening the blue-to-cyan half of the ramp.)
    "parula": _rgb_stops([
        [62, 38, 168], [64, 42, 180], [66, 46, 191], [67, 50, 202],
        [69, 55, 213], [70, 59, 222], [71, 65, 229], [71, 70, 235],
        [72, 76, 240], [72, 82, 244], [71, 87, 247], [70, 94, 251],
        [69, 100, 253], [66, 106, 254], [62, 112, 255], [56, 118, 254],
        [50, 124, 252], [47, 130, 250], [46, 135, 247], [45, 140, 243],
        [43, 145, 239], [39, 151, 235], [37, 155, 232], [35, 160, 229],
        [32, 165, 227], [29, 169, 224], [25, 173, 220], [18, 177, 214],
        [8, 180, 209], [1, 183, 202], [1, 186, 196], [9, 189, 189],
        [26, 192, 181], [37, 194, 174], [44, 196, 166], [50, 198, 159],
        [56, 200, 150], [64, 202, 141], [75, 203, 132], [87, 204, 122],
        [100, 205, 111], [114, 205, 100], [129, 204, 89], [143, 203, 78],
        [157, 201, 67], [171, 199, 57], [184, 196, 49], [197, 194, 42],
        [208, 191, 39], [219, 189, 40], [230, 187, 45], [239, 186, 53],
        [247, 186, 61], [254, 190, 60], [254, 196, 56], [254, 202, 51],
        [252, 208, 48], [249, 214, 45], [247, 221, 42], [245, 227, 39],
        [245, 233, 36], [246, 239, 32], [247, 245, 27], [249, 251, 21],
    ]),
    # Hidden compatibility maps -------------------------------------------------
    "turbo": _rgb_stops([
        [48, 18, 59], [65, 67, 167], [71, 113, 233], [62, 155, 254],
        [34, 197, 226], [26, 228, 182], [70, 248, 132], [136, 255, 78],
        [185, 246, 53], [225, 221, 55], [250, 186, 57], [253, 141, 39],
        [240, 91, 18], [214, 53, 6], [175, 24, 1], [122, 4, 3],
    ]),
    "magma": _rgb_stops([
        [0, 0, 4], [11, 9, 36], [32, 17, 75], [59, 15, 112],
        [87, 21, 126], [114, 31, 129], [140, 41, 129], [168, 50, 125],
        [196, 60, 117], [222, 73, 104], [241, 96, 93], [250, 127, 94],
        [254, 159, 109], [254, 191, 132], [253, 222, 160], [252, 253, 191],
    ]),
    "plasma": _rgb_stops([
        [13, 8, 135], [51, 5, 151], [80, 2, 162], [106, 0, 168],
        [132, 5, 167], [156, 23, 158], [177, 42, 144], [195, 61, 128],
        [211, 81, 113], [225, 100, 98], [237, 121, 83], [246, 143, 68],
        [252, 166, 54], [254, 192, 41], [249, 220, 36], [240, 249, 33],
    ]),
    "cividis": _rgb_stops([
        [0, 32, 77], [0, 44, 106], [15, 56, 110], [49, 68, 107],
        [70, 80, 107], [87, 92, 109], [102, 105, 112], [117, 117, 117],
        [132, 130, 121], [149, 143, 120], [166, 157, 117], [184, 171, 112],
        [203, 186, 105], [221, 201, 95], [241, 217, 81], [255, 234, 70],
    ]),
}

_CANONICAL_BY_KEY = {
    name.casefold(): name for name in (*BUILTIN_COLORMAP_NAMES, *LEGACY_COLORMAP_NAMES)
}
_CUSTOM_COLORMAPS: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_CUSTOM_NAME_BY_KEY: dict[str, str] = {}


def _normalise_stops(
    stops: Sequence[Sequence[object]],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(stops, Sequence) or isinstance(stops, (str, bytes)):
        raise ValueError("Colormap stops must be a sequence.")
    if not 2 <= len(stops) <= _MAX_CUSTOM_STOPS:
        raise ValueError(
            f"A custom colormap needs 2–{_MAX_CUSTOM_STOPS} color stops."
        )
    parsed: list[tuple[float, tuple[int, int, int, int]]] = []
    for stop in stops:
        if not isinstance(stop, Sequence) or len(stop) != 2:
            raise ValueError("Each colormap stop must contain a position and RGBA color.")
        position = float(stop[0])
        rgba_raw = stop[1]
        if not np.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("Colormap stop positions must be between 0 and 1.")
        if not isinstance(rgba_raw, Sequence) or len(rgba_raw) not in (3, 4):
            raise ValueError("Colormap colors must have three or four channels.")
        rgba = tuple(int(channel) for channel in rgba_raw)
        if any(channel < 0 or channel > 255 for channel in rgba):
            raise ValueError("Colormap color channels must be between 0 and 255.")
        if len(rgba) == 3:
            rgba = (*rgba, 255)
        parsed.append((position, rgba))
    parsed.sort(key=lambda item: item[0])
    if any(b[0] <= a[0] for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Colormap stop positions must be unique.")
    if parsed[0][0] > 0.0:
        parsed.insert(0, (0.0, parsed[0][1]))
    if parsed[-1][0] < 1.0:
        parsed.append((1.0, parsed[-1][1]))
    positions = np.asarray([item[0] for item in parsed], dtype=np.float64)
    rgba = np.asarray([item[1] for item in parsed], dtype=np.uint8)
    return positions, rgba


def validate_custom_colormap_name(name: str, *, replacing: str | None = None) -> str:
    clean = " ".join(str(name).strip().split())
    if not clean:
        raise ValueError("Enter a name for the custom colormap.")
    if len(clean) > 64:
        raise ValueError("Custom colormap names may contain at most 64 characters.")
    key = clean.casefold()
    # Only the maps actually OFFERED are reserved. A hidden compatibility map
    # (``LEGACY_COLORMAP_NAMES``) is deliberately allowed: adopting its name is
    # exactly how a user puts a preset that is not in the list back into it, and
    # both ``canonical_colormap_name`` and ``_base_colormap`` already resolve a
    # custom map before the built-in table, so the custom one wins consistently.
    reserved = {
        *(item.casefold() for item in solid_color_names()),
        *(item.casefold() for item in BUILTIN_COLORMAP_NAMES),
    }
    if key in reserved or key.startswith("solid:"):
        raise ValueError(f"'{clean}' is reserved by a built-in colormap.")
    replacement_key = replacing.casefold() if replacing else None
    if key in _CUSTOM_NAME_BY_KEY and key != replacement_key:
        raise ValueError(f"A custom colormap named '{clean}' already exists.")
    return clean


def configure_custom_colormaps(raw: object) -> None:
    """Replace the runtime custom-map cache from a preference dictionary."""
    global _CUSTOM_COLORMAPS, _CUSTOM_NAME_BY_KEY
    # Validation below must not treat the registry being replaced as a set of
    # duplicate names (for example when save_prefs reconfigures an unchanged map).
    _CUSTOM_COLORMAPS = {}
    _CUSTOM_NAME_BY_KEY = {}
    parsed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    names: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for name, record in list(raw.items())[:_MAX_CUSTOM_MAPS]:
            try:
                clean = validate_custom_colormap_name(str(name))
                if clean.casefold() in names:
                    raise ValueError(
                        f"A custom colormap named '{clean}' already exists."
                    )
                stops = record.get("stops") if isinstance(record, Mapping) else record
                parsed[clean] = _normalise_stops(stops)
                names[clean.casefold()] = clean
            except (TypeError, ValueError) as exc:
                warnings.warn(f"Ignoring invalid custom colormap {name!r}: {exc}")
    _CUSTOM_COLORMAPS = parsed
    _CUSTOM_NAME_BY_KEY = names


def custom_colormap_names() -> tuple[str, ...]:
    return tuple(sorted(_CUSTOM_COLORMAPS, key=str.casefold))


def is_custom_colormap(name: str) -> bool:
    return str(name).casefold() in _CUSTOM_NAME_BY_KEY


def custom_colormap_stops(name: str) -> list[list[object]]:
    canonical = _CUSTOM_NAME_BY_KEY.get(str(name).casefold())
    if canonical is None:
        raise KeyError(f"Unknown custom colormap: {name}")
    positions, rgba = _CUSTOM_COLORMAPS[canonical]
    return [
        [float(position), [int(channel) for channel in color]]
        for position, color in zip(positions, rgba)
    ]


def store_custom_colormap(
    prefs: dict,
    name: str,
    stops: Sequence[Sequence[object]],
    *,
    replacing: str | None = None,
) -> str:
    """Validate and persist one map in an application preferences dictionary."""
    clean = validate_custom_colormap_name(name, replacing=replacing)
    positions, rgba = _normalise_stops(stops)
    plot = prefs.setdefault("plot", {})
    records = plot.setdefault("custom_colormaps", {})
    if replacing and replacing != clean:
        records.pop(replacing, None)
    records[clean] = {
        "stops": [
            [float(position), [int(channel) for channel in color]]
            for position, color in zip(positions, rgba)
        ]
    }
    configure_custom_colormaps(records)
    return clean


def delete_custom_colormap(prefs: dict, name: str) -> bool:
    plot = prefs.setdefault("plot", {})
    records = plot.setdefault("custom_colormaps", {})
    canonical = _CUSTOM_NAME_BY_KEY.get(str(name).casefold())
    if canonical is None or canonical not in records:
        return False
    del records[canonical]
    configure_custom_colormaps(records)
    return True


#: Placeholder standing for "every solid colour, as one block" in a saved
#: order. Folded, the solids are not individually orderable -- their sequence
#: is the COLOR dialog's Solid Color List -- but the block's *position* is.
SOLID_GROUP_TOKEN = "@solids"

#: Runtime ordering state, configured from preferences by
#: :func:`configure_colormap_order`. Empty means "the order these are declared
#: in", which is what a fresh installation gets.
_COLORMAP_ORDER: tuple[str, ...] = ()
_FOLD_SOLIDS: bool = True


def configure_colormap_order(order: object, fold_solids: object = True) -> None:
    """Replace the runtime colormap order from a preference value."""
    global _COLORMAP_ORDER, _FOLD_SOLIDS
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
        _COLORMAP_ORDER = tuple(
            str(name) for name in order if isinstance(name, str) and str(name).strip()
        )
    else:
        _COLORMAP_ORDER = ()
    _FOLD_SOLIDS = bool(fold_solids)


def colormap_order() -> tuple[str, ...]:
    return _COLORMAP_ORDER


def solids_are_folded() -> bool:
    return _FOLD_SOLIDS


def ordered_colormap_names(
    colormaps: Sequence[str],
    solids: Sequence[str],
    order: Sequence[str] = (),
    *,
    fold_solids: bool = True,
) -> list[str]:
    """Apply a saved order to the colormaps and solids currently available.

    Pure, so the rule is testable without preferences or a window.

    * a saved entry that no longer exists is dropped (a deleted custom map);
    * anything available but unsaved is appended (a map added since, or a solid
      the user added to the Solid Color List) -- **never silently hidden**;
    * folded, the individual solids are replaced by :data:`SOLID_GROUP_TOKEN`,
      which expands in place into the Solid Color List's own order, so the
      block moves as one and the COLOR dialog stays authoritative for what is
      inside it;
    * unfolded, each solid is an ordinary orderable entry and any not yet
      placed default to the end of the list.
    """
    colormaps = [str(name) for name in colormaps]
    solids = [str(name) for name in solids]
    # ⚠ Solids and colormaps are SEPARATE namespaces, matched case-sensitively,
    # because the grayscale colormap ``gray`` and the solid colour ``Gray``
    # differ only by case -- exactly the distinction ``canonical_colormap_name``
    # relies on. Folding them into one case-insensitive key set made ``gray``
    # read as the solid and disappear from the list entirely.
    solid_set = set(solids)
    exact_maps = {name: name for name in colormaps}
    #: Case-insensitive fallback for colormaps only, so an order saved with a
    #: different spelling still resolves. Never consulted for a solid.
    folded_maps = {name.casefold(): name for name in colormaps}

    result: list[str] = []
    seen_solids: set[str] = set()
    seen_maps: set[str] = set()
    group_placed = False
    for raw in order:
        name = str(raw)
        if name == SOLID_GROUP_TOKEN:
            if fold_solids and not group_placed:
                result.append(SOLID_GROUP_TOKEN)
                group_placed = True
            continue
        if name in solid_set:                 # exact: 'Gray' yes, 'gray' no
            if fold_solids:
                continue                      # folded: only the group is placed
            if name not in seen_solids:
                result.append(name)
                seen_solids.add(name)
            continue
        canonical = exact_maps.get(name) or folded_maps.get(name.casefold())
        if canonical is not None and canonical not in seen_maps:
            result.append(canonical)
            seen_maps.add(canonical)

    for name in colormaps:                    # added since the order was saved
        if name not in seen_maps:
            result.append(name)
            seen_maps.add(name)
    if fold_solids:
        if not group_placed:
            result.append(SOLID_GROUP_TOKEN)
        index = result.index(SOLID_GROUP_TOKEN)
        result[index:index + 1] = list(solids)
    else:
        for name in solids:                   # unplaced solids default to the end
            if name not in seen_solids:
                result.append(name)
                seen_solids.add(name)
    return result


def _available_colormaps(*, include_custom: bool, include_legacy: bool) -> list[str]:
    names = list(BUILTIN_COLORMAP_NAMES)
    if include_legacy:
        names.extend(LEGACY_COLORMAP_NAMES)
    if include_custom:
        names.extend(custom_colormap_names())
    return names


def named_colormap_names(
    *, include_custom: bool = True, include_legacy: bool = False
) -> list[str]:
    """The colormaps offered, in the user's order (see the reorder dialog)."""
    names = _available_colormaps(
        include_custom=include_custom, include_legacy=include_legacy)
    if not _COLORMAP_ORDER:
        return names
    # No solids here: this list is colormaps only, so folding cannot apply.
    return ordered_colormap_names(names, (), _COLORMAP_ORDER, fold_solids=False)


def colormap_tree_entries(*, include_custom: bool = True) -> list[str]:
    """The offered list with the solid block left **folded**, for a tree widget.

    :func:`channel_colormap_names` is the flat answer every ordinary combo and
    menu needs, and folded it inlines the solids as a contiguous block --
    the best a flat list can do. A widget that can show child rows wants the
    block *unexpanded* instead, so :data:`SOLID_GROUP_TOKEN` is returned in its
    place and the widget nests the solids under it.

    Unfolded, this is exactly the flat list: there is no group to show.
    """
    solids = list(solid_color_names())
    names = _available_colormaps(include_custom=include_custom, include_legacy=False)
    if not _FOLD_SOLIDS:
        return channel_colormap_names(include_custom=include_custom)
    if not _COLORMAP_ORDER:
        return [SOLID_GROUP_TOKEN, *names]
    flat = ordered_colormap_names(names, solids, _COLORMAP_ORDER, fold_solids=True)
    out: list[str] = []
    placed = False
    for name in flat:
        if name in solids:
            if not placed:
                out.append(SOLID_GROUP_TOKEN)
                placed = True
            continue
        out.append(name)
    if not placed:
        out.append(SOLID_GROUP_TOKEN)
    return out


def channel_colormap_names(*, include_custom: bool = True) -> list[str]:
    """Solid colours and colormaps together, in the user's order.

    A fresh installation has no saved order, and then the solids lead as they
    always did.
    """
    solids = list(solid_color_names())
    names = _available_colormaps(include_custom=include_custom, include_legacy=False)
    if not _COLORMAP_ORDER:
        return [*solids, *names]
    return ordered_colormap_names(
        names, solids, _COLORMAP_ORDER, fold_solids=_FOLD_SOLIDS)


def canonical_colormap_name(name: str) -> str:
    """Return the stored display name for a known map, case-insensitively."""
    text = str(name).strip()
    if text.startswith("solid:"):
        _solid_colormap(text)
        return text
    # Exact title case distinguishes the pure ``Gray`` ramp from the named
    # grayscale map ``gray``.
    if is_solid_color(text):
        return canonical_solid_color_name(text)
    custom_name = _CUSTOM_NAME_BY_KEY.get(text.casefold())
    if custom_name is not None:
        return custom_name
    canonical = _CANONICAL_BY_KEY.get(text.casefold())
    if canonical is not None:
        return canonical
    raise KeyError(f"Unknown colormap: {name}")


def _solid_colormap(name: str) -> pg.ColorMap:
    color_part = name[6:]
    if color_part.startswith("custom:"):
        hex_value = color_part[7:].strip()
        if len(hex_value) not in (7, 9) or not hex_value.startswith("#"):
            raise ValueError(f"Invalid solid custom color: {name}")
        try:
            raw = hex_value[1:]
            values = tuple(int(raw[index:index + 2], 16) for index in range(0, len(raw), 2))
            rgba = values + (255,) if len(values) == 3 else values
        except ValueError as exc:
            raise ValueError(f"Invalid solid custom color: {name}") from exc
    else:
        color_part = canonical_solid_color_name(color_part)
        rgba = solid_color_rgba(color_part)
    table = np.asarray([rgba, rgba], dtype=np.uint8)
    return pg.ColorMap(np.asarray([0.0, 1.0]), table)


def _base_colormap(name: str) -> pg.ColorMap:
    text = canonical_colormap_name(name)
    if text.startswith("solid:"):
        return _solid_colormap(text)
    if is_solid_color(text):
        r, g, b, a = solid_color_rgba(text)
        rgba = np.asarray([[0, 0, 0, a], [r, g, b, a]], dtype=np.uint8)
        return pg.ColorMap(np.asarray([0.0, 1.0]), rgba)
    if text in _CUSTOM_COLORMAPS:
        positions, rgba = _CUSTOM_COLORMAPS[text]
        return pg.ColorMap(positions.copy(), rgba.copy())
    positions, rgba = _CONTROL_POINTS[text.casefold()]
    return pg.ColorMap(positions.copy(), rgba.copy())


def invert_colormap(cmap: pg.ColorMap) -> pg.ColorMap:
    lut = np.asarray(cmap.getLookupTable(0.0, 1.0, 256, alpha=True), dtype=np.uint8)
    return pg.ColorMap(np.linspace(0.0, 1.0, len(lut)), lut[::-1].copy())


def apply_gamma(cmap: pg.ColorMap, gamma: float) -> pg.ColorMap:
    try:
        value = float(gamma)
    except (TypeError, ValueError):
        return cmap
    if not np.isfinite(value) or value <= 0.0 or abs(value - 1.0) < 1e-6:
        return cmap
    lut = np.asarray(
        cmap.getLookupTable(0.0, 1.0, 256, alpha=True), dtype=np.float64
    )
    points = np.linspace(0.0, 1.0, len(lut))
    warped = points**value
    output = np.empty_like(lut)
    for channel in range(lut.shape[1]):
        output[:, channel] = np.interp(warped, points, lut[:, channel])
    return pg.ColorMap(points, np.rint(output).astype(np.uint8))


def make_colormap(
    name: str, *, invert: bool = False, gamma: float = 1.0
) -> pg.ColorMap:
    """Return an application-owned PyQtGraph colormap by name.

    Unknown names raise ``KeyError`` instead of silently changing the display.
    """
    cmap = _base_colormap(name)
    if invert:
        cmap = invert_colormap(cmap)
    return apply_gamma(cmap, gamma)


def colormap_lut(
    name: str,
    *,
    n: int = 256,
    invert: bool = False,
    gamma: float = 1.0,
    alpha: bool = True,
) -> np.ndarray:
    cmap = make_colormap(name, invert=invert, gamma=gamma)
    return np.asarray(
        cmap.getLookupTable(0.0, 1.0, max(2, int(n)), alpha=alpha),
        dtype=np.uint8,
    )


def map_to_rgba(
    name: str,
    values: np.ndarray,
    *,
    invert: bool = False,
    gamma: float = 1.0,
) -> np.ndarray:
    normalised = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    normalised = np.nan_to_num(normalised, nan=0.0, posinf=1.0, neginf=0.0)
    indices = np.rint(normalised * 255.0).astype(np.uint8)
    return colormap_lut(name, invert=invert, gamma=gamma, alpha=True)[indices]


def representative_rgb(name: str, *, position: float = 0.85) -> tuple[float, float, float]:
    rgba = map_to_rgba(name, np.asarray([position], dtype=float))[0]
    return tuple(float(channel) / 255.0 for channel in rgba[:3])
