"""
minflux_viewer.core.sample_presets
==================================
Named **sample-data presets** — the editable entries under *File › Open Sample
Data*. Each preset stores a full :mod:`core.simulate` configuration; invoking a
menu entry silently regenerates the dataset(s) from it. Presets live in
``prefs["simulate"]["presets"]`` and can be saved to / loaded from a JSON file
(marker ``minflux_viewer_sample_presets``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .simulate import MULTI_SIMS, STRUCTURES, default_params

PRESETS_MARKER = "minflux_viewer_sample_presets"

#: keys of one preset (a superset of the simulate_localizations kwargs + name/channels)
_PRESET_KEYS = ("name", "structure", "n_points", "locs_per_trace", "precision_nm",
                "dim", "seed", "channels", "params")


def normalize_preset(preset: dict) -> dict:
    """Fill defaults / coerce types for one preset (tolerant of partial dicts)."""
    structure = str(preset.get("structure", "homogeneous"))
    if structure not in STRUCTURES and structure not in MULTI_SIMS:
        structure = "homogeneous"
    params = dict(default_params(structure))
    params.update({k: v for k, v in (preset.get("params") or {}).items() if k in params})
    seed = preset.get("seed", None)
    return {
        "name": str(preset.get("name") or f"{structure} (simulation)"),
        "structure": structure,
        "n_points": int(preset.get("n_points", 2000)),
        "locs_per_trace": float(preset.get("locs_per_trace", 4.0)),
        "precision_nm": float(preset.get("precision_nm", 5.0)),
        "dim": 3 if int(preset.get("dim", 3)) == 3 else 2,
        "seed": None if seed in (None, -1, "") else int(seed),
        "channels": max(int(preset.get("channels", 1)), 1),
        "params": params,
    }


def default_presets() -> list[dict]:
    """The presets shipped when the user has none yet."""
    npc = default_params("npc")
    mt = default_params("microtubule")
    sph = default_params("sphere")
    return [
        normalize_preset({"name": "NPC 3D (simulation)", "structure": "npc",
                          "dim": 3, "n_points": 3000, "params": npc}),
        normalize_preset({"name": "NPC 2D (simulation)", "structure": "npc",
                          "dim": 2, "n_points": 3000, "params": npc}),
        normalize_preset({"name": "Microtubule 3D 2Channel (simulation)",
                          "structure": "microtubule", "dim": 3, "channels": 2,
                          "n_points": 4000, "params": mt}),
        normalize_preset({"name": "Sphere Shell 3D (simulation)", "structure": "sphere",
                          "dim": 3, "n_points": 2000, "params": sph}),
        normalize_preset({"name": "NPC 3-channel overlay (simulation)",
                          "structure": "npc_overlay_3ch", "dim": 3}),
        normalize_preset({"name": "NPC 2-channel by DCR (simulation)",
                          "structure": "npc_dcr_2ch", "dim": 3}),
    ]


def load_presets(prefs: dict) -> list[dict]:
    """Presets from *prefs*, seeding the defaults the first time."""
    if prefs is None:
        return default_presets()
    store = prefs.setdefault("simulate", {})          # NB: {} is falsy — don't use `prefs or {}`
    raw = store.get("presets")
    if not raw:
        presets = default_presets()
        store["presets"] = presets
        return presets
    return [normalize_preset(p) for p in raw]


def save_presets(prefs: dict, presets: list[dict]) -> None:
    """Write *presets* into *prefs* (caller persists via ``AppState.save_prefs``)."""
    if prefs is None:
        return
    prefs.setdefault("simulate", {})["presets"] = [normalize_preset(p) for p in presets]


def write_presets_file(path: str | Path, presets: list[dict]) -> None:
    payload = {"format": PRESETS_MARKER, "version": 1,
               "presets": [normalize_preset(p) for p in presets]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_presets_file(path: str | Path) -> list[dict]:
    """Load presets from a JSON file (accepts a bare list or the wrapped payload)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("presets", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    return [normalize_preset(p) for p in items if isinstance(p, dict)]


def simulate_kwargs(preset: dict) -> dict:
    """The :func:`core.simulate.simulate_localizations` kwargs from a preset."""
    p = normalize_preset(preset)
    return {"structure": p["structure"], "n_points": p["n_points"],
            "locs_per_trace": p["locs_per_trace"], "precision_nm": p["precision_nm"],
            "params": p["params"], "dim": p["dim"], "seed": p["seed"]}
