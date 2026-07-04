"""Target-A ``.msr`` writer (minflux_viewer/msr/writer.py).

Round-trips MINFLUX localizations (+ optional MBM beads) through *our own*
reader: ``write_minflux_msr`` → ``GeneralMSRParser`` → ``mfx_map`` / ``mbm_map``.
It does not attempt Imspector/Abberior byte-compatibility (see the backlog).
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.save import dataset_to_mfx_array
from minflux_viewer.core.simulate import simulate_localizations
from minflux_viewer.msr import state as S
from minflux_viewer.msr.mfxdta import extract_zarr_store
from minflux_viewer.msr.msr_parser import GeneralMSRParser
from minflux_viewer.msr.writer import (
    BEAD_DTYPE,
    MsrChannel,
    ObfStack,
    build_channel_store,
    channel_from_dataset,
    obf_bytes,
    pack_mfxdta,
    write_datasets_msr,
    write_minflux_msr,
)


def _sim_dataset(name="sim ch1", n=150, seed=0):
    coords, tid, attrs = simulate_localizations("npc", n_points=n, seed=seed)
    return build_localization_dataset(
        name=name, x_nm=coords[:, 0], y_nm=coords[:, 1], z_nm=coords[:, 2],
        tid=tid, attrs=attrs, source_version="simulation")


def _make_beads(n_beads=4, n_time=20):
    recs = []
    for b in range(n_beads):
        home = np.array([1000.0 * b, -500.0 * b, 50.0 * b]) * 1e-9
        for k in range(n_time):
            drift = np.array([k * 0.5, k * 0.3, k * 0.1]) * 1e-9
            recs.append((b + 1, home + drift, float(k), 100.0))
    beads = np.array(recs, dtype=BEAD_DTYPE)
    pbg = {str(b + 1): {"name": f"R{b + 1}", "gri": b + 1} for b in range(n_beads)}
    used = [f"R{b + 1}" for b in range(n_beads)]
    return beads, pbg, used


# --- Layer 2: MFXDTA pack ↔ decode ------------------------------------------
def test_pack_mfxdta_roundtrips_zarr_store():
    mfx = dataset_to_mfx_array(_sim_dataset())
    store = build_channel_store(
        MsrChannel(mfx=mfx, name="c"), did="00000000-0000-0000-0000-000000000001")
    blob = pack_mfxdta(store)
    assert blob[:6] == b"MFXDTA"
    back = extract_zarr_store(blob)
    # extract_zarr_store strips the 'zarr/' prefix → keys equal the store keys.
    assert set(back.keys()) == set(store.keys())
    for k in store:
        assert back[k] == store[k]


# --- Layer 3: OBF container ↔ msr-reader ------------------------------------
def test_obf_bytes_roundtrips_through_obffile(tmp_path):
    from msr_reader import OBFFile

    payloads = [b"MFXDTA\x00hello-stack-one", b"MFXDTA\x00second-stack-payload!!"]
    data = obf_bytes([ObfStack(payload=p, name=f"s{i}", description=f"d{i}")
                      for i, p in enumerate(payloads)], file_description="[]")
    path = tmp_path / "raw.obf"
    path.write_bytes(data)
    with OBFFile(str(path)) as f:
        assert f.num_stacks == 2
        for i, p in enumerate(payloads):
            arr = np.ascontiguousarray(f.read_stack(i)).ravel()
            assert arr.dtype == np.uint8
            assert arr.tobytes() == p
            assert f.stack_headers[i].description == f"d{i}"


# --- Full write → GeneralMSRParser ------------------------------------------
def test_write_minflux_msr_roundtrips_mfx(tmp_path):
    ds = _sim_dataset()
    mfx = dataset_to_mfx_array(ds)
    out = write_minflux_msr(tmp_path / "sim.msr", [MsrChannel(mfx=mfx, name="sim ch1")])
    assert out.exists()

    res = GeneralMSRParser().parse(str(out), log=lambda *a: None)
    assert res["mode"] == "modern"
    assert res["source_format"] == "obf / mfxdta"
    assert len(res["datasets"]) == 1
    name = res["datasets"][0]["display_name"]
    assert name == "sim ch1"                                  # from did→label map

    mfx2 = S.mfx_map[name]
    assert mfx2.shape[0] == mfx.shape[0]
    np.testing.assert_allclose(mfx2["loc"], mfx["loc"])
    np.testing.assert_array_equal(mfx2["tid"], mfx["tid"])
    np.testing.assert_allclose(mfx2["efo"], mfx["efo"])
    # No beads supplied → no MBM entry.
    assert not getattr(S.mbm_map.get(name), "size", 0)

    # The recovered mfx is not just parseable — it loads into a full dataset,
    # detected as m2410, with the original coordinates.
    from minflux_viewer.core.loader import load_from_mfx_array

    ds2 = load_from_mfx_array(mfx2, name=name, folder=str(tmp_path))
    assert ds2.prop.num_loc == ds.prop.num_loc
    assert str(ds2.metadata.get("source_version", "")).lower() == "m2410"
    np.testing.assert_allclose(
        np.sort(np.asarray(ds2.loc_nm)[:, 0]),
        np.sort(np.asarray(ds.loc_nm)[:, 0]), atol=1e-3)


def test_write_includes_mbm_beads_and_drift(tmp_path):
    ds = _sim_dataset()
    mfx = dataset_to_mfx_array(ds)
    beads, pbg, used = _make_beads(n_beads=4, n_time=20)
    ch = MsrChannel(mfx=mfx, name="sim ch1", beads=beads, points_by_gri=pbg, used=used)
    out = write_minflux_msr(tmp_path / "sim.msr", [ch])

    res = GeneralMSRParser().parse(str(out), log=lambda *a: None)
    name = res["datasets"][0]["display_name"]
    beads2 = S.mbm_map[name]
    assert beads2.shape[0] == beads.shape[0] == 80
    assert {"gri", "xyz", "tim"} <= set(beads2.dtype.names)
    np.testing.assert_allclose(np.asarray(beads2["xyz"], float), beads["xyz"])

    # Show-Beads-Drift path recovers all four beads (via in-store zarr attrs).
    from minflux_viewer.plugins.msr_reader.beads_drift import gather_msr_bead_drift

    drift = gather_msr_bead_drift(res["datasets"], S.mbm_map)
    assert len(drift) == 1 and drift[0]["name"] == name
    assert len(drift[0]["beads"]) == 4
    for bead in drift[0]["beads"]:
        assert bead["xyz_nm"].shape == (20, 3)


def test_multichannel_overlay_roundtrips(tmp_path):
    dsa, dsb = _sim_dataset("red", seed=1), _sim_dataset("green", seed=2)
    mfa, mfb = dataset_to_mfx_array(dsa), dataset_to_mfx_array(dsb)
    out = write_minflux_msr(tmp_path / "ov.msr", [
        MsrChannel(mfx=mfa, name="red"), MsrChannel(mfx=mfb, name="green")])

    res = GeneralMSRParser().parse(str(out), log=lambda *a: None)
    names = [d["display_name"] for d in res["datasets"]]
    assert names == ["red", "green"]
    assert S.mfx_map["red"].shape[0] == mfa.shape[0]
    assert S.mfx_map["green"].shape[0] == mfb.shape[0]
    # Distinct channels — different localization counts survive independently.
    assert S.mfx_map["red"].shape[0] != 0 and S.mfx_map["green"].shape[0] != 0


def test_write_datasets_msr_from_dataset_with_attached_beads(tmp_path):
    ds = _sim_dataset()
    beads, pbg, used = _make_beads()
    # The convention the MSR loader / simulator use to carry beads on a dataset.
    ds.metadata["mbm_points"] = beads
    ds.metadata["mbm_points_by_gri"] = pbg
    ds.metadata["mbm_used"] = used

    ch = channel_from_dataset(ds)
    assert ch.beads is beads and ch.name == "sim ch1"

    out = write_datasets_msr(tmp_path / "ds.msr", [ds])
    res = GeneralMSRParser().parse(str(out), log=lambda *a: None)
    name = res["datasets"][0]["display_name"]
    assert S.mfx_map[name].shape[0] == ds.prop.num_loc
    assert S.mbm_map[name].shape[0] == beads.shape[0]
