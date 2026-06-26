"""Legacy MBM (bead) layout → model points translation (msr/legacy_mbm.py)."""

import numpy as np
import zarr

from minflux_viewer.msr.legacy_mbm import POINTS_DTYPE, build_legacy_mbm

_BEAD_DTYPE = np.dtype([("tim", "f8"), ("pos", "f8", (3,)), ("str", "f8"), ("tid", "i4")])


def _legacy_store() -> dict:
    """An in-memory zarr store mimicking the legacy grd/mbm/<R-ID> layout."""
    store: dict = {}
    root = zarr.open(store, mode="w")
    mbm = root.create_group("grd").create_group("mbm")
    mbm.attrs["used"] = ["R5", "R7"]

    a = np.zeros(3, _BEAD_DTYPE)
    a["tim"] = [1.0, 2.0, 3.0]
    a["pos"] = [[1e-6, 2e-6, 3e-7]] * 3
    a["str"] = [10.0, 11.0, 12.0]
    d = mbm.create_dataset("R5", data=a)
    d.attrs["gri"], d.attrs["name"] = 100, "R5"

    a2 = np.zeros(2, _BEAD_DTYPE)
    a2["tim"] = [5.0, 6.0]
    a2["pos"] = [[4e-6, 5e-6, 6e-7]] * 2
    d2 = mbm.create_dataset("R7", data=a2)
    d2.attrs["gri"], d2.attrs["name"] = 200, "R7"

    d3 = mbm.create_dataset("R9", data=np.zeros(0, _BEAD_DTYPE))   # empty, not used
    d3.attrs["gri"], d3.attrs["name"] = 300, "R9"
    return store


def test_build_legacy_mbm_translates_points_pbg_used():
    arch = zarr.open(_legacy_store(), mode="r")
    pts, pbg, used = build_legacy_mbm(arch, log=lambda *a, **k: None)
    assert pts.dtype.names == POINTS_DTYPE.names
    assert pts.size == 5                                  # 3 + 2 (empty R9 contributes none)
    assert set(np.unique(pts["gri"]).tolist()) == {100, 200}
    # gri↔R-ID map includes every bead with a gri (even the empty one)
    assert pbg == {"100": {"name": "R5", "gri": 100},
                   "200": {"name": "R7", "gri": 200},
                   "300": {"name": "R9", "gri": 300}}
    assert used == ["R5", "R7"]
    r5 = pts[pts["gri"] == 100]
    np.testing.assert_allclose(r5["xyz"][0], [1e-6, 2e-6, 3e-7])   # metres preserved
    np.testing.assert_allclose(np.sort(r5["tim"]), [1.0, 2.0, 3.0])


def test_build_legacy_mbm_none_without_grd_mbm():
    store: dict = {}
    zarr.open(store, mode="w").create_group("mfx")
    assert build_legacy_mbm(zarr.open(store, mode="r"), log=lambda *a, **k: None) is None


def test_build_legacy_mbm_consumed_by_drift():
    """The translated points + metadata feed extract_bead_drift to per-bead tracks."""
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift

    arch = zarr.open(_legacy_store(), mode="r")
    pts, pbg, used = build_legacy_mbm(arch, log=lambda *a, **k: None)
    beads = extract_bead_drift(pts, pbg, used)
    assert {b["rid"] for b in beads} == {"R5", "R7"}     # only used beads
    r5 = next(b for b in beads if b["rid"] == "R5")
    assert r5["gri"] == 100 and r5["n"] == 3 and r5["xyz_nm"].shape == (3, 3)
