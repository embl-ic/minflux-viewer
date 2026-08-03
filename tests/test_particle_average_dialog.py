"""
Particle Average dialog — ROI collection.

Regression: detection boxes are commonly made on a **processed copy** of the
data (aggregated + convolved) and then applied to the source dataset or a
DCR-separated channel of it, which has a *different* ``dataset_idx``. The dialog
must still offer those boxes (the earlier ``dataset_idx == active`` filter hid
them, leaving the particle list empty).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def _dataset_with_cluster():
    from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset

    rng = np.random.default_rng(0)
    inside = rng.uniform(400, 500, size=(40, 2))          # inside the [400..500] box
    outside = rng.uniform(0, 200, size=(40, 2))
    xy = np.vstack([inside, outside])
    n = xy.shape[0]
    z = rng.normal(300, 10, n)
    attrs = AttrStore({
        "loc_x": xy[:, 0] * 1e-9, "loc_y": xy[:, 1] * 1e-9, "loc_z": z * 1e-9,
        "tid": np.repeat(np.arange(n // 2), 2).astype(float),
        "efo": rng.normal(8e4, 1e3, n), "ftr": np.ones(n, bool),
    })
    prop = DataProp(num_loc=n, num_itr=1, num_dim=3, num_traces=n // 2,
                    trace_idx=np.zeros((n // 2, 2), int),
                    num_loc_per_trace=np.full(n // 2, 2),
                    attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr"])
    return MinfluxDataset(file=FileInfo(name="synth.mat", folder="/tmp"), prop=prop, attr=attrs)


def _detection_box(name="det-1", from_dataset_idx=999):
    from minflux_viewer.core.roi import RoiRecord
    rec = RoiRecord.create("rectangle", {"bounds": [400, 400, 100, 100], "angle": 0.0}, name=name)
    rec.context = {"dataset_idx": from_dataset_idx, "source": "conv_segmentation"}
    return rec


def _overlay_two_channels():
    """Two distinctly-named datasets sharing an overlay id, each with a cluster in
    the [400..500] box (like the two channels of a DCR separation)."""
    from minflux_viewer.core.dataset import FileInfo
    d0 = _dataset_with_cluster()
    d1 = _dataset_with_cluster()
    d0.file = FileInfo(name="chA", folder="/tmp")
    d1.file = FileInfo(name="chB", folder="/tmp")
    oid = "overlay:test:abc"
    for order, d in enumerate((d0, d1), start=1):
        lut = "Red" if order == 1 else "Green"
        d.state["overlay_id"] = oid
        d.state["overlay_order"] = order
        d.state["render_channel_lut"] = lut
        d.state["overlay_lut"] = lut
    return d0, d1


def test_geomfit_multichannel_option_available_for_overlay(qtbot):
    """The 'Combine overlay channels' option now also applies to the NPC two-ring
    (geomfit) method: it collects the other channel and is active for geomfit."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.particle_average_dialog import ParticleAverageWindow

    d0, d1 = _overlay_two_channels()
    state = AppState()
    state.add_dataset(d0)
    state.add_dataset(d1)
    state.set_active(0)                                        # channel 0 = reference
    state.rois.add(_detection_box(from_dataset_idx=0))

    dlg = ParticleAverageWindow(state)
    qtbot.addWidget(dlg)

    dlg._method.setCurrentIndex(dlg._method.findData("geomfit"))
    dlg._multichannel.setChecked(True)
    dlg._collect()

    assert len(dlg._particles) >= 1
    assert len(dlg._channel_luts) >= 2                        # both channels captured
    # geomfit now participates in multi-channel replay (was free/template only)
    assert dlg._multichannel_active() is True
    # and the per-particle transforms it exports are replayable
    from minflux_viewer.analysis import npc_geomfit as gf
    res = gf.average_npc_geomfit([np.asarray(p.points, float) for p in dlg._particles],
                                 min_gof=-10.0)
    assert "particle_transforms" in res
    assert len(res["particle_transforms"]) == len(dlg._particles)


def test_boxes_from_a_processed_copy_are_offered_and_extracted(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.particle_average_dialog import ParticleAverageWindow

    state = AppState()
    state.add_dataset(_dataset_with_cluster())
    state.set_active(0)
    # box was detected on a *different* dataset index (the aggregated copy).
    state.rois.add(_detection_box(from_dataset_idx=999))

    dlg = ParticleAverageWindow(state)
    qtbot.addWidget(dlg)

    # offered despite the mismatched dataset_idx (regression: was filtered out)
    assert len(dlg._box_records()) == 1
    dlg._refresh_rois()
    assert dlg._roi_list.count() == 1

    # and extraction from the active dataset succeeds across datasets
    dlg._collect()
    assert len(dlg._particles) == 1
    assert dlg._particles[0].n_locs == 40                 # the 40 in-box localizations
