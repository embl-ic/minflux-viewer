"""Sample-data presets (core/sample_presets.py) for File › Open Sample Data."""

from __future__ import annotations

from minflux_viewer.core.sample_presets import (
    default_presets,
    load_presets,
    normalize_preset,
    read_presets_file,
    save_presets,
    simulate_kwargs,
    write_presets_file,
)
from minflux_viewer.core.simulate import simulate_localizations


def test_default_presets_are_the_named_entries():
    names = [p["name"] for p in default_presets()]
    assert names == [
        "NPC 3D (simulation)",
        "NPC 2D (simulation)",
        "Microtubule 3D 2Channel (simulation)",
        "Sphere Shell 3D (simulation)",
        "NPC 3-channel overlay (simulation)",
        "NPC 2-channel by DCR (simulation)",
    ]
    overlay = next(p for p in default_presets() if p["structure"] == "npc_overlay_3ch")
    assert "ch1_diameter_nm" in overlay["params"]        # multi-sim params merged
    dcr = next(p for p in default_presets() if p["structure"] == "npc_dcr_2ch")
    assert "dcr_low" in dcr["params"] and "dcr_high" in dcr["params"]
    mt = next(p for p in default_presets() if "2Channel" in p["name"])
    assert mt["channels"] == 2 and mt["structure"] == "microtubule" and mt["dim"] == 3
    npc2d = next(p for p in default_presets() if p["name"].startswith("NPC 2D"))
    assert npc2d["dim"] == 2


def test_normalize_preset_fills_and_coerces():
    p = normalize_preset({"structure": "npc", "seed": -1, "params": {"n_pores": 12}})
    assert p["name"] and p["channels"] == 1 and p["dim"] == 3
    assert p["seed"] is None                          # -1 → random
    assert p["params"]["n_pores"] == 12               # kept
    assert "field_curvature" in p["params"]           # defaults merged in
    bad = normalize_preset({"structure": "nope"})
    assert bad["structure"] == "homogeneous"          # unknown → fallback


def test_load_seeds_and_save_roundtrips_prefs():
    prefs: dict = {}
    loaded = load_presets(prefs)                       # seeds defaults
    assert len(loaded) == 6 and prefs["simulate"]["presets"]
    custom = [normalize_preset({"name": "My NPC", "structure": "npc"})]
    save_presets(prefs, custom)
    assert [p["name"] for p in load_presets(prefs)] == ["My NPC"]


def test_presets_file_roundtrip(tmp_path):
    presets = default_presets()
    path = tmp_path / "presets.json"
    write_presets_file(path, presets)
    back = read_presets_file(path)
    assert [p["name"] for p in back] == [p["name"] for p in presets]


def test_simulate_kwargs_drive_the_generator():
    p = normalize_preset({"structure": "sphere", "n_points": 50, "dim": 2, "seed": 7})
    coords, tid, attrs = simulate_localizations(**simulate_kwargs(p))
    assert coords.shape[1] == 3 and tid.shape[0] == coords.shape[0]
    assert (coords[:, 2] == 0).all()                  # dim=2 flattens Z
