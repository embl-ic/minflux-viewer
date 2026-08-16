"""Application-owned and user-defined colormap registry."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.colormaps import (
    BUILTIN_COLORMAP_NAMES,
    LEGACY_COLORMAP_NAMES,
    canonical_colormap_name,
    configure_custom_colormaps,
    custom_colormap_names,
    delete_custom_colormap,
    make_colormap,
    named_colormap_names,
    store_custom_colormap,
)


@pytest.fixture(autouse=True)
def _empty_custom_registry():
    configure_custom_colormaps({})
    yield
    configure_custom_colormaps({})


def test_every_owned_map_returns_a_pyqtgraph_lut():
    signatures = set()
    for name in (*BUILTIN_COLORMAP_NAMES, *LEGACY_COLORMAP_NAMES):
        lut = make_colormap(name).getLookupTable(0.0, 1.0, 256, alpha=True)
        assert lut.shape == (256, 4)
        assert lut.dtype == np.uint8
        signatures.add(lut.tobytes())

    assert len(signatures) == len(BUILTIN_COLORMAP_NAMES) + len(
        LEGACY_COLORMAP_NAMES
    )


def test_focused_menu_hides_legacy_maps_but_keeps_them_resolvable():
    names = named_colormap_names()
    assert names[: len(BUILTIN_COLORMAP_NAMES)] == list(BUILTIN_COLORMAP_NAMES)
    assert not set(LEGACY_COLORMAP_NAMES).intersection(names)
    assert make_colormap("parula") is not None
    assert canonical_colormap_name("hilo") == "HiLo"


def test_unknown_map_does_not_silently_change_appearance():
    with pytest.raises(KeyError, match="Unknown colormap"):
        make_colormap("not-a-real-map")


def test_custom_map_round_trips_through_preferences_and_registry():
    prefs = {"plot": {"custom_colormaps": {}}}
    name = store_custom_colormap(
        prefs,
        "Membrane signal",
        [[0.0, [5, 10, 15, 255]], [0.4, [20, 80, 160, 255]], [1.0, [255, 240, 80, 255]]],
    )

    assert name == "Membrane signal"
    assert custom_colormap_names() == ("Membrane signal",)
    assert named_colormap_names()[-1] == "Membrane signal"
    lut = make_colormap("membrane SIGNAL").getLookupTable(
        0.0, 1.0, 256, alpha=True
    )
    assert np.array_equal(lut[0], [5, 10, 15, 255])
    assert np.array_equal(lut[-1], [255, 240, 80, 255])

    saved = prefs["plot"]["custom_colormaps"]
    configure_custom_colormaps(saved)
    assert custom_colormap_names() == ("Membrane signal",)
    assert delete_custom_colormap(prefs, "MEMBRANE signal")
    assert custom_colormap_names() == ()


@pytest.mark.parametrize("reserved", ["hot", "HiLo", "Red", "solid:Blue"])
def test_custom_map_cannot_shadow_an_owned_name(reserved):
    with pytest.raises(ValueError, match="reserved"):
        store_custom_colormap(
            {"plot": {"custom_colormaps": {}}},
            reserved,
            [[0.0, [0, 0, 0, 255]], [1.0, [255, 255, 255, 255]]],
        )
