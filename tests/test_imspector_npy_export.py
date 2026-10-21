"""Raw ``.npy`` exports in the layout Imspector itself writes.

A MATLAB source loads ``logical`` flags as ``uint8`` and a one-channel ``dcr``
as a column; written back that way, a reader built against Imspector's own
files selects rows with ``a[a["vld"]]`` -- integer indexing on ``uint8`` -- and
cannot take ``a["dcr"][:, 0]``. The export now writes ``bool`` flags and an N×k
``dcr``; our own loader reads both forms.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io as sio

from minflux_viewer.api import io
from minflux_viewer.core.save import imspector_npy_layout


@pytest.fixture
def imspector_mat(tmp_path):
    """An m2410 file as Imspector writes it: top-level variables, logical flags,
    a one-channel ``dcr`` (N×1)."""
    rng = np.random.default_rng(3)
    tid = np.repeat(np.arange(1, 9, dtype=np.int32), 6)          # 8 traces
    itr = np.tile(np.array([0, 1, 2, 3, 3, 3], dtype=np.int32), 8)
    n = tid.size
    fnl = itr == 3
    bot = np.zeros(n, bool)
    bot[::6] = True
    eot = np.zeros(n, bool)
    eot[5::6] = True
    vld = np.ones(n, bool)
    vld[17] = False                                              # one invalid final row
    path = tmp_path / "imspector_m2410.mat"
    sio.savemat(str(path), {
        "vld": vld, "fnl": fnl, "bot": bot, "eot": eot,
        "sta": np.zeros(n, np.int32), "tim": np.linspace(0.0, 4.0, n), "tid": tid,
        "gri": np.zeros(n, np.int32), "thi": np.zeros(n, np.int32), "sqi": np.zeros(n, np.int32),
        "itr": itr, "loc": rng.normal(size=(n, 3)) * 1e-6, "lnc": rng.normal(size=(n, 3)) * 1e-6,
        "eco": rng.integers(50, 200, n).astype(np.int32), "ecc": rng.integers(0, 20, n).astype(np.int32),
        "efo": rng.random(n) * 1e5, "efc": rng.random(n) * 1e4, "fbg": rng.random(n) * 1e3,
        "cfr": rng.random(n), "dcr": rng.random((n, 1)),
    })
    return path


def test_raw_npy_export_uses_bool_flags_and_a_dcr_matrix(imspector_mat, tmp_path):
    ds = io.load(imspector_mat)
    raw = np.asarray(ds.mfx_raw["vld"])
    assert raw.dtype != np.bool_                 # the premise: MATLAB logical loads as uint8
    out = io.save(ds, tmp_path / "export.npy", fmt="npy", content="raw")[0]
    arr = np.load(out, allow_pickle=False)
    for flag in ("vld", "fnl", "bot", "eot"):
        assert arr.dtype[flag] == np.bool_, flag
    assert arr["dcr"].shape == (arr.size, 1)
    # What an Imspector-file reader does with it:
    valid = arr[arr["vld"]]
    assert valid.size == int(np.asarray(ds.mfx_raw["vld"]).astype(bool).sum())
    assert np.all(valid["vld"])
    assert arr["dcr"][:, 0].shape == (arr.size,)


def test_the_export_reloads_identically_through_our_loader(imspector_mat, tmp_path):
    original = io.load(imspector_mat)
    out = io.save(original, tmp_path / "export.npy", fmt="npy", content="raw")[0]
    again = io.load(out)
    assert again.prop.num_loc == original.prop.num_loc
    assert again.prop.num_traces == original.prop.num_traces
    for name in ("loc_x", "loc_y", "loc_z", "efo", "cfr", "dcr", "tid", "tim", "eco"):
        np.testing.assert_array_equal(np.asarray(again.attr[name]),
                                      np.asarray(original.attr[name]), err_msg=name)


def test_layout_leaves_what_is_already_imspector_shaped():
    already = np.zeros(4, dtype=[("vld", np.bool_), ("dcr", np.float64, (2,)), ("itr", np.int32)])
    assert imspector_npy_layout(already) is already


def test_a_flag_column_that_is_not_0_or_1_is_left_alone():
    arr = np.zeros(3, dtype=[("vld", np.uint8), ("sta", np.int32)])
    arr["vld"] = [0, 1, 7]
    out = imspector_npy_layout(arr)
    assert out.dtype["vld"] == np.uint8
    np.testing.assert_array_equal(out["vld"], [0, 1, 7])


def test_values_are_unchanged():
    arr = np.zeros(5, dtype=[("vld", np.uint8), ("dcr", np.float64), ("efo", np.float64)])
    arr["vld"] = [1, 0, 1, 1, 0]
    arr["dcr"] = np.linspace(0.1, 0.5, 5)
    arr["efo"] = np.arange(5.0)
    out = imspector_npy_layout(arr)
    np.testing.assert_array_equal(out["vld"], arr["vld"].astype(bool))
    np.testing.assert_array_equal(out["dcr"][:, 0], arr["dcr"])
    np.testing.assert_array_equal(out["efo"], arr["efo"])
